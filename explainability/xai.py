"""
Explainability Engine
----------------------
Three complementary XAI methods:

1. SHAP (vitals)      — TreeExplainer / KernelExplainer waterfall charts
2. Attention weights   — BioBERT token-level saliency heatmap
3. GradCAM            — EfficientNet spatial saliency on X-ray
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple


# ─────────────────────────────────────────────
# 1. VITALS SHAP
# ─────────────────────────────────────────────

def compute_shap_vitals(
    model,
    vitals_tensor: torch.Tensor,      # (1, 12) normalised
    text_inputs: dict,
    image: Optional[torch.Tensor],
    background_vitals: torch.Tensor,  # (N, 12) background distribution
    class_idx: int = 0,               # which class to explain
    device: str = "cpu",
    n_samples: int = 50,
) -> Dict[str, float]:
    """
    Kernel SHAP for the vitals modality.
    Returns dict of {feature_name: shap_value}.
    """
    try:
        import shap
    except ImportError:
        return _fallback_vitals_importance(
            model, vitals_tensor, text_inputs, image, class_idx, device
        )

    model.eval()

    # Freeze text & image for efficiency — only perturb vitals
    def predict_fn(vitals_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            vt = torch.tensor(vitals_np, dtype=torch.float32).to(device)
            # Broadcast text / image inputs
            batch_size = vt.size(0)
            ti = {k: v.repeat(batch_size, 1).to(device)
                  for k, v in text_inputs.items()}
            img = (image.repeat(batch_size, 1, 1, 1).to(device)
                   if image is not None else None)
            _, probs, _, _ = model(ti, vt, img)
            return probs.cpu().numpy()

    bg = background_vitals[:n_samples].cpu().numpy()
    explainer   = shap.KernelExplainer(predict_fn, bg)
    input_np    = vitals_tensor.cpu().numpy()

    shap_values = explainer.shap_values(input_np, nsamples=100, silent=True)
    # shap_values is list of arrays (one per class) or array (binary)
    if isinstance(shap_values, list):
        sv = shap_values[class_idx][0]   # (12,)
    else:
        sv = shap_values[0, :, class_idx]

    from utils.dataset import VITALS_COLS
    return dict(zip(VITALS_COLS, sv.tolist()))


def _fallback_vitals_importance(
    model, vitals_tensor, text_inputs, image, class_idx, device
) -> Dict[str, float]:
    """
    Gradient-based feature importance when SHAP is not available.
    Computes ∂output/∂vitals via autograd.
    """
    from utils.dataset import VITALS_COLS
    model.eval()
    vt = vitals_tensor.detach().clone().requires_grad_(True).to(device)
    ti = {k: v.to(device) for k, v in text_inputs.items()}
    img = image.to(device) if image is not None else None

    _, probs, _, _ = model(ti, vt, img)
    probs[0, class_idx].backward()

    grads = vt.grad[0].detach().cpu().numpy()           # (12,)
    return dict(zip(VITALS_COLS, grads.tolist()))


# ─────────────────────────────────────────────
# 2. ATTENTION-BASED TEXT SALIENCY
# ─────────────────────────────────────────────

def compute_attention_saliency(
    attentions: list,
    input_ids: torch.Tensor,
    tokenizer,
    layer: int = -1,
    head_reduction: str = "mean",
) -> List[Dict]:
    """
    Compute per-token saliency from BioBERT attention weights.

    attentions   : list of (B, heads, seq, seq) from model forward pass
    input_ids    : (1, seq_len)
    layer        : which layer's attentions to use (-1 = last)
    head_reduction: 'mean' | 'max' across heads

    Returns list of {token, score} sorted by position.
    """
    if attentions is None:
        return []

    # Pick layer
    attn = attentions[layer]          # (B, heads, seq, seq)
    attn = attn[0]                    # (heads, seq, seq) — batch 0

    # Reduce across heads
    if head_reduction == "mean":
        attn = attn.mean(dim=0)       # (seq, seq)
    else:
        attn = attn.max(dim=0).values

    # Rollout: propagate attention through layers for more faithful attributions
    rollout = _attention_rollout(attentions)

    # CLS → all tokens
    cls_attn = rollout[0, 1:-1]       # drop [CLS] and [SEP] tokens
    cls_attn = cls_attn / (cls_attn.max() + 1e-8)

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    tokens = tokens[1:-1]             # strip [CLS] / [SEP]

    result = []
    for tok, score in zip(tokens, cls_attn.tolist()):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        result.append({
            "token": tok.replace("##", ""),
            "raw_token": tok,
            "score": float(score),
        })

    return result


def _attention_rollout(attentions: list) -> torch.Tensor:
    """
    Abnar & Zuidema (2020) attention rollout.
    Propagates attention through all layers to get more faithful attributions.
    """
    with torch.no_grad():
        rollout = None
        for attn in attentions:
            # attn: (B, heads, seq, seq) — use batch 0, mean over heads
            a = attn[0].mean(dim=0)              # (seq, seq)
            # Add residual connection (identity)
            a = a + torch.eye(a.size(0), device=a.device)
            a = a / a.sum(dim=-1, keepdim=True)
            if rollout is None:
                rollout = a
            else:
                rollout = torch.matmul(a, rollout)
    return rollout  # (seq, seq)


def tokens_to_html(token_scores: List[Dict], max_tokens: int = 80) -> str:
    """
    Render token saliency as inline HTML with colour-coded background.
    Returns HTML string for Streamlit's st.markdown(unsafe_allow_html=True).
    """
    if not token_scores:
        return "<p style='color:gray'>No attention data available.</p>"

    html_parts = ["<div style='line-height:2.2;font-size:15px;font-family:monospace'>"]
    for item in token_scores[:max_tokens]:
        score = item["score"]
        # Colour: low=white → high=red
        r  = int(255)
        g  = int(255 * (1 - score * 0.85))
        b  = int(255 * (1 - score * 0.85))
        bg = f"rgb({r},{g},{b})"
        tok = item["token"]
        html_parts.append(
            f"<span style='background:{bg};padding:2px 3px;"
            f"border-radius:3px;margin:1px'>{tok}</span> "
        )
    html_parts.append("</div>")
    return "".join(html_parts)


# ─────────────────────────────────────────────
# 3. GRADCAM FOR X-RAY
# ─────────────────────────────────────────────

class GradCAM:
    """
    GradCAM for EfficientNet-B3.
    Hooks into the last convolutional block to produce spatial heatmap.
    """

    def __init__(self, model, target_layer_name: str = "features.8"):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []

        # Find target layer
        target = None
        for name, module in model.named_modules():
            if name == target_layer_name:
                target = module
                break

        if target is None:
            # Fallback: use last conv-like layer in image_enc backbone
            for name, module in model.image_enc.named_modules():
                if hasattr(module, "weight") and len(list(module.children())) == 0:
                    target = module

        if target is not None:
            self._hooks.append(
                target.register_forward_hook(self._save_activation)
            )
            self._hooks.append(
                target.register_full_backward_hook(self._save_gradient)
            )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        text_inputs: dict,
        vitals: torch.Tensor,
        image: torch.Tensor,
        class_idx: int,
        device: str = "cpu",
    ) -> Optional[np.ndarray]:
        """
        Returns (H, W) numpy array, values in [0, 1].
        Returns None if image encoder not available.
        """
        if not hasattr(self.model, "image_enc"):
            return None

        self.model.eval()
        self.model.zero_grad()

        ti  = {k: v.to(device) for k, v in text_inputs.items()}
        vt  = vitals.to(device)
        img = image.to(device)

        logits, _, _, _ = self.model(ti, vt, img)
        score = logits[0, class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            return None

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam     = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def overlay_gradcam(
    original_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay GradCAM heatmap onto original X-ray image.
    original_image : (H, W, 3) uint8
    cam            : (h, w) float32 in [0, 1]
    Returns        : (H, W, 3) uint8
    """
    import cv2
    H, W = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8),
        getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET)
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (original_image * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return overlay


# ─────────────────────────────────────────────
# 4. CALIBRATION CURVE
# ─────────────────────────────────────────────

def compute_calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    class_idx: int = 0,
    n_bins: int = 10,
) -> Dict:
    """
    Compute reliability diagram data for a single class.
    Returns dict with bin_confidence, bin_accuracy, bin_counts, ece.
    """
    class_probs = probs[:, class_idx]
    binary_labels = (labels == class_idx).astype(int)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_conf, bin_acc, bin_count = [], [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (class_probs >= lo) & (class_probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf.append(float(class_probs[mask].mean()))
        bin_acc.append(float(binary_labels[mask].mean()))
        bin_count.append(int(mask.sum()))

    # Expected Calibration Error
    n = len(labels)
    ece = sum(
        (cnt / n) * abs(c - a)
        for c, a, cnt in zip(bin_conf, bin_acc, bin_count)
    )

    return {
        "bin_confidence": bin_conf,
        "bin_accuracy":   bin_acc,
        "bin_counts":     bin_count,
        "ece":            ece,
    }
