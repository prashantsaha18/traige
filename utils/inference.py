"""
Inference Engine
-----------------
Wraps the trained model for fast, stateless inference in the Streamlit app.
Handles: tokenization, vitals normalization, image preprocessing,
         SHAP / attention / GradCAM, and result formatting.
"""

import os
import math
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoTokenizer

# ── Local imports ──
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.triage_model import MultiModalTriageModel
from utils.dataset import VitalsNormaliser, VITALS_COLS, LABEL_MAP_INV
from explainability.xai import (
    compute_shap_vitals, compute_attention_saliency,
    tokens_to_html, GradCAM, overlay_gradcam,
    _fallback_vitals_importance
)

# Synthetic background for SHAP
from data.synthetic_generator import generate_dataset


# ─────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────

class TriageInferenceEngine:
    """
    Singleton-friendly inference wrapper.
    Load once with st.cache_resource, then call .predict() per request.
    """

    MODEL_CLASSES  = ["Critical", "Urgent", "Non-urgent"]
    URGENCY_COLORS = {
        "Critical":   "#E24B4A",
        "Urgent":     "#BA7517",
        "Non-urgent": "#3B6D11",
    }
    URGENCY_BG = {
        "Critical":   "#FCEBEB",
        "Urgent":     "#FAEEDA",
        "Non-urgent": "#EAF3DE",
    }

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        normaliser_path: str = "checkpoints/normaliser.json",
        tokenizer_name:  str = "dmis-lab/biobert-base-cased-v1.2",
        with_image:      bool = True,
        device:          str = "auto",
    ):
        self.device = self._resolve_device(device)
        print(f"[TriageEngine] Loading on {self.device}...")

        # ── Tokeniser ──
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # ── Normaliser ──
        if os.path.exists(normaliser_path):
            self.normaliser = VitalsNormaliser.load(normaliser_path)
        else:
            self.normaliser = VitalsNormaliser()

        # ── Model ──
        self.model = MultiModalTriageModel(
            text_model=tokenizer_name,
            with_image=with_image,
        ).to(self.device)

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            state = ckpt.get("model_state", ckpt)
            self.model.load_state_dict(state, strict=False)
            print(f"[TriageEngine] Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[TriageEngine] No checkpoint found — using random weights (demo mode)")

        self.model.eval()

        # ── Background distribution for SHAP ──
        self._bg_vitals = self._build_background_vitals()

        # ── GradCAM ──
        self.gradcam = GradCAM(self.model) if with_image else None

        # ── Differential diagnoses bank ──
        self._ddx_bank = self._build_ddx_bank()

        print("[TriageEngine] Ready.")

    # ──────────────────────────────────────────
    # MAIN PREDICT METHOD
    # ──────────────────────────────────────────

    def predict(
        self,
        chief_complaint: str,
        vitals: Dict[str, Optional[float]],
        image: Optional[Any] = None,          # PIL Image or None
        explain: bool = True,
    ) -> Dict:
        """
        Full prediction pipeline.

        Parameters
        ----------
        chief_complaint : free-text symptom description
        vitals          : dict mapping vital name → float (None = missing)
        image           : PIL Image of chest X-ray, or None
        explain         : compute XAI outputs

        Returns
        -------
        dict with keys: urgency, probabilities, diagnoses,
                        shap_values, attention_html, gradcam_overlay,
                        inference_time_ms
        """
        t0 = time.time()

        # ── 1. Tokenise text ──
        enc = self.tokenizer(
            chief_complaint,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(self.device) for k, v in enc.items()}

        # ── 2. Normalise vitals ──
        vitals_arr = self.normaliser.transform_single(vitals)
        vitals_t   = torch.tensor(vitals_arr, dtype=torch.float32).unsqueeze(0).to(self.device)

        # ── 3. Process image (optional) ──
        image_t = None
        image_np = None
        if image is not None:
            image_t, image_np = self._preprocess_image(image)
            image_t = image_t.to(self.device)

        # ── 4. Forward pass ──
        with torch.no_grad():
            logits, probs, attentions, modalities = self.model(
                text_inputs, vitals_t, image_t
            )

        probs_np = probs[0].cpu().numpy()         # (3,)
        pred_idx = int(probs_np.argmax())
        urgency  = self.MODEL_CLASSES[pred_idx]

        # ── 5. Differential diagnoses ──
        ddx = self._generate_ddx(chief_complaint, vitals, urgency, probs_np)

        # ── 6. XAI ──
        shap_values  = None
        attention_html = None
        gradcam_overlay = None

        if explain:
            # SHAP / gradient importance for vitals
            try:
                shap_values = _fallback_vitals_importance(
                    self.model, vitals_t, text_inputs, image_t,
                    pred_idx, self.device
                )
            except Exception as e:
                print(f"[XAI] Vitals importance failed: {e}")
                shap_values = {col: 0.0 for col in VITALS_COLS}

            # Attention heatmap for text
            try:
                token_scores   = compute_attention_saliency(
                    attentions, enc["input_ids"], self.tokenizer
                )
                attention_html = tokens_to_html(token_scores)
            except Exception as e:
                print(f"[XAI] Attention failed: {e}")
                attention_html = f"<i style='color:gray'>Attention unavailable: {e}</i>"

            # GradCAM on X-ray
            if image_t is not None and self.gradcam is not None and image_np is not None:
                try:
                    cam = self.gradcam.generate(
                        text_inputs, vitals_t, image_t, pred_idx, self.device
                    )
                    if cam is not None:
                        gradcam_overlay = overlay_gradcam(image_np, cam)
                except Exception as e:
                    print(f"[XAI] GradCAM failed: {e}")

        elapsed_ms = (time.time() - t0) * 1000

        return {
            "urgency":          urgency,
            "urgency_color":    self.URGENCY_COLORS[urgency],
            "urgency_bg":       self.URGENCY_BG[urgency],
            "pred_idx":         pred_idx,
            "probabilities":    {cls: float(p) for cls, p in zip(self.MODEL_CLASSES, probs_np)},
            "diagnoses":        ddx,
            "shap_values":      shap_values,
            "attention_html":   attention_html,
            "gradcam_overlay":  gradcam_overlay,
            "inference_time_ms": elapsed_ms,
            "modalities_used":  ["text", "vitals"] + (["image"] if image_t is not None else []),
        }

    # ──────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────

    def _preprocess_image(self, pil_image):
        """Returns (tensor, numpy_array) for inference + GradCAM overlay."""
        try:
            from torchvision import transforms
            import numpy as np

            transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
            img_rgb = pil_image.convert("RGB")
            img_np  = np.array(img_rgb.resize((300, 300))).astype(np.uint8)
            img_t   = transform(img_rgb).unsqueeze(0)
            return img_t, img_np
        except Exception as e:
            print(f"[Engine] Image preprocessing failed: {e}")
            return None, None

    def _build_background_vitals(self) -> torch.Tensor:
        """Build a small background distribution from synthetic data for SHAP."""
        try:
            cases = generate_dataset(n=200, seed=0)
            rows  = []
            for c in cases:
                arr = self.normaliser.transform_single(c)
                rows.append(arr)
            return torch.tensor(np.stack(rows), dtype=torch.float32)
        except Exception:
            return torch.zeros(50, len(VITALS_COLS))

    def _build_ddx_bank(self) -> Dict:
        """Condition → likely diagnoses mapping for context-aware DDx."""
        return {
            "Critical": {
                "chest": ["STEMI / ACS", "Aortic dissection", "Massive PE", "Tension pneumothorax", "Cardiac tamponade"],
                "neuro": ["Haemorrhagic stroke", "Ischaemic stroke", "Subarachnoid haemorrhage", "Status epilepticus"],
                "shock": ["Septic shock", "Hypovolaemic shock", "Anaphylaxis", "Cardiogenic shock"],
                "default":["Septic shock", "STEMI", "Acute stroke", "DKA", "Pulmonary embolism"],
            },
            "Urgent": {
                "chest": ["Community-acquired pneumonia", "Pleuritis", "Pericarditis", "Unstable angina"],
                "abd":   ["Acute appendicitis", "Cholecystitis", "Bowel obstruction", "Diverticulitis"],
                "neuro": ["TIA", "Migraine with aura", "Meningitis"],
                "default":["Pneumonia", "Acute appendicitis", "Hypertensive urgency", "Pyelonephritis", "Cellulitis"],
            },
            "Non-urgent": {
                "default": ["Upper respiratory tract infection", "Musculoskeletal strain", "Gastritis", "Anxiety disorder", "Viral gastroenteritis"],
            },
        }

    def _generate_ddx(
        self,
        complaint: str,
        vitals: dict,
        urgency: str,
        probs: np.ndarray,
    ) -> list:
        """Generate top-3 differential diagnoses with confidence estimates."""
        bank = self._ddx_bank.get(urgency, self._ddx_bank["Non-urgent"])
        cl   = complaint.lower()

        # Pick symptom bucket
        if any(w in cl for w in ["chest", "heart", "cardiac", "palpitation"]):
            candidates = bank.get("chest", bank["default"])
        elif any(w in cl for w in ["abdom", "belly", "nausea", "vomit", "bowel"]):
            candidates = bank.get("abd", bank["default"])
        elif any(w in cl for w in ["head", "neuro", "confused", "speech", "vision", "weak"]):
            candidates = bank.get("neuro", bank["default"])
        elif any(w in cl for w in ["shock", "collapse", "altered", "unresponsive"]):
            candidates = bank.get("shock", bank["default"])
        else:
            candidates = bank["default"]

        # Assign pseudo-probabilities (descending)
        base_conf = probs[probs.argmax()] * 0.95
        ddx = []
        for i, dx in enumerate(candidates[:5]):
            conf = max(0.05, base_conf * (0.85 ** i) + np.random.uniform(-0.02, 0.02))
            ddx.append({"diagnosis": dx, "confidence": round(float(conf), 3)})

        return sorted(ddx, key=lambda x: -x["confidence"])[:5]

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():   return "cuda"
            if torch.backends.mps.is_available(): return "mps"
            return "cpu"
        return device


# ─────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    engine = TriageInferenceEngine(
        checkpoint_path="checkpoints/best_model.pt",
        normaliser_path="checkpoints/normaliser.json",
        with_image=False,
    )

    result = engine.predict(
        chief_complaint="Severe crushing chest pain radiating to left arm, diaphoresis, onset 30 minutes ago.",
        vitals={
            "heart_rate": 118, "systolic_bp": 88, "diastolic_bp": 55,
            "spo2": 91, "temperature": 36.9, "respiratory_rate": 24,
            "gcs": 14, "age": 62, "wbc": None, "creatinine": None,
            "glucose": 180, "hemoglobin": None,
        },
        explain=True,
    )

    print(f"\n{'='*50}")
    print(f"Urgency : {result['urgency']}")
    print(f"Probs   : {result['probabilities']}")
    print(f"Top DDx : {[d['diagnosis'] for d in result['diagnoses'][:3]]}")
    print(f"Time    : {result['inference_time_ms']:.1f} ms")
    print(f"Modalities: {result['modalities_used']}")
