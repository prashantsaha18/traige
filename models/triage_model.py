"""
Multi-modal Triage Model
Fuses: BioBERT (text) + FT-Transformer (vitals) + EfficientNet-B3 (X-ray)
Late fusion → MLP head → 3-class urgency prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math


# ─────────────────────────────────────────────
# 1. TEXT ENCODER  (BioBERT fine-tune)
# ─────────────────────────────────────────────
class TextEncoder(nn.Module):
    """
    Wraps dmis-lab/biobert-base-cased-v1.2.
    Outputs a 768-dim CLS embedding, projected to `out_dim`.
    """
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.2",
                 out_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size          # 768

        # Freeze bottom 8 layers, fine-tune top 4
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < 8:
                for p in layer.parameters():
                    p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
        )
        cls = out.last_hidden_state[:, 0, :]          # (B, 768)
        attentions = out.attentions                    # list of (B, heads, seq, seq)
        return self.proj(cls), attentions


# ─────────────────────────────────────────────
# 2. VITALS ENCODER  (Feature Tokenizer + Transformer)
# ─────────────────────────────────────────────
class FeatureTokenizer(nn.Module):
    """Embeds each scalar vital as a learnable token."""
    def __init__(self, n_features: int, d_token: int = 64):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (B, n_features)
        return x.unsqueeze(-1) * self.weight + self.bias   # (B, n_features, d_token)


class FTTransformer(nn.Module):
    """
    Lightweight Feature Tokenizer + Transformer for tabular vitals.
    Gorishniy et al. 2021 — https://arxiv.org/abs/2106.11959
    """
    def __init__(self, n_features: int = 12, d_token: int = 64,
                 n_heads: int = 4, n_layers: int = 2, out_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)

        self.proj = nn.Sequential(
            nn.Linear(d_token, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, n_features) — NaN-masked to 0 before forward
        tokens = self.tokenizer(x)                     # (B, n_features, d_token)
        out    = self.transformer(tokens)              # (B, n_features, d_token)
        cls    = self.norm(out.mean(dim=1))            # mean-pool over features
        return self.proj(cls)                          # (B, out_dim)


# ─────────────────────────────────────────────
# 3. IMAGE ENCODER  (EfficientNet-B3, optional)
# ─────────────────────────────────────────────
class ImageEncoder(nn.Module):
    """
    EfficientNet-B3 backbone for chest X-ray.
    Drop-in — branch is skipped at inference if no image supplied.
    """
    def __init__(self, out_dim: int = 256, dropout: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b3(weights=weights)
        except ImportError:
            raise ImportError("torchvision required for image encoder")

        # Replace classifier head
        in_features = backbone.classifier[1].in_features   # 1536
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # fallback — backbone already pools
            nn.Flatten(),
            nn.Linear(in_features, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Freeze early stages
        for name, p in self.backbone.named_parameters():
            if "features.0" in name or "features.1" in name:
                p.requires_grad = False

    def forward(self, x):
        # x: (B, 3, 300, 300)
        feat = self.backbone.features(x)               # (B, 1536, h, w)
        feat = feat.mean(dim=[2, 3])                   # global avg pool → (B, 1536)
        return self.head[1:](feat)                     # skip the Identity pool


# ─────────────────────────────────────────────
# 4. FUSION + CLASSIFICATION HEAD
# ─────────────────────────────────────────────
class ModalityDropout(nn.Module):
    """Randomly zeroes an entire modality embedding during training."""
    def __init__(self, p: float = 0.15):
        super().__init__()
        self.p = p

    def forward(self, emb):
        if self.training and torch.rand(1).item() < self.p:
            return torch.zeros_like(emb)
        return emb


class TriageHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 5. FULL MULTI-MODAL MODEL
# ─────────────────────────────────────────────
class MultiModalTriageModel(nn.Module):
    """
    Late-fusion multi-modal triage classifier.

    Inputs
    ------
    text_inputs  : dict with input_ids, attention_mask (required)
    vitals       : (B, 12) float tensor — NaN → 0 before passing
    image        : (B, 3, 300, 300) float tensor OR None (optional)

    Outputs
    -------
    logits       : (B, 3)   — [critical, urgent, non-urgent]
    probs        : (B, 3)   — softmax probabilities
    attentions   : BERT attention weights (for text highlighting)
    modalities   : dict of per-modality embeddings (for SHAP / ablation)
    """

    VITALS_FEATURES = [
        "heart_rate", "systolic_bp", "diastolic_bp", "spo2",
        "temperature", "respiratory_rate", "gcs",
        "age", "wbc", "creatinine", "glucose", "hemoglobin",
    ]
    N_VITALS = len(VITALS_FEATURES)       # 12
    EMB_DIM  = 256
    CLASSES  = ["Critical", "Urgent", "Non-urgent"]

    def __init__(self, text_model: str = "dmis-lab/biobert-base-cased-v1.2",
                 with_image: bool = True, modality_drop_p: float = 0.15):
        super().__init__()
        self.with_image = with_image

        self.text_enc   = TextEncoder(text_model, out_dim=self.EMB_DIM)
        self.vitals_enc = FTTransformer(self.N_VITALS, out_dim=self.EMB_DIM)

        self.text_drop   = ModalityDropout(modality_drop_p)
        self.vitals_drop = ModalityDropout(modality_drop_p)

        fusion_dim = self.EMB_DIM * 2

        if with_image:
            self.image_enc  = ImageEncoder(out_dim=self.EMB_DIM)
            self.image_drop = ModalityDropout(modality_drop_p)
            fusion_dim += self.EMB_DIM

        self.head = TriageHead(fusion_dim, n_classes=3)

        # Temperature scaling (set post-training via calibration)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, text_inputs: dict, vitals: torch.Tensor,
                image: torch.Tensor = None):

        text_emb, attentions = self.text_enc(**text_inputs)
        vitals_emb           = self.vitals_enc(vitals)

        text_emb   = self.text_drop(text_emb)
        vitals_emb = self.vitals_drop(vitals_emb)

        parts = [text_emb, vitals_emb]

        if self.with_image and image is not None:
            img_emb = self.image_enc(image)
            img_emb = self.image_drop(img_emb)
            parts.append(img_emb)
        elif self.with_image and image is None:
            # zero-pad missing image branch
            img_emb = torch.zeros(text_emb.size(0), self.EMB_DIM,
                                  device=text_emb.device)
            parts.append(img_emb)

        fused  = torch.cat(parts, dim=-1)
        logits = self.head(fused)
        probs  = F.softmax(logits / self.temperature, dim=-1)

        modalities = {
            "text":   text_emb.detach(),
            "vitals": vitals_emb.detach(),
            **({"image": img_emb.detach()} if self.with_image else {}),
        }

        return logits, probs, attentions, modalities


# ─────────────────────────────────────────────
# 6. ASYMMETRIC FOCAL LOSS
# ─────────────────────────────────────────────
class AsymmetricTriageLoss(nn.Module):
    """
    Focal loss with class weights that heavily penalise
    missed Critical cases (false negatives are life-threatening).

    Weights: Critical=5.0, Urgent=2.0, Non-urgent=1.0
    gamma   : focal modulation (2.0 standard)
    """
    def __init__(self, class_weights=(5.0, 2.0, 1.0), gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weights", torch.tensor(class_weights))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # targets: (B,) long   logits: (B, 3)
        log_probs = F.log_softmax(logits, dim=-1)
        probs     = log_probs.exp()

        # Gather probability of the true class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_w = (1 - p_t) ** self.gamma

        # Class weight
        class_w = self.weights[targets]

        # NLL loss
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = focal_w * class_w * nll
        return loss.mean()


# ─────────────────────────────────────────────
# 7. TEMPERATURE CALIBRATION
# ─────────────────────────────────────────────
def calibrate_temperature(model: MultiModalTriageModel,
                           val_loader,
                           device: str = "cpu",
                           n_epochs: int = 50,
                           lr: float = 0.01):
    """
    Post-hoc temperature scaling.
    Optimises model.temperature on a held-out validation set.
    Call after main training is done.
    """
    model.temperature.requires_grad_(True)
    model.eval()
    optimizer = torch.optim.LBFGS([model.temperature], lr=lr, max_iter=50)
    ce = nn.CrossEntropyLoss()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits, _, _, _ = model(
                batch["text_inputs"],
                batch["vitals"].to(device),
                batch.get("image"),
            )
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    def eval_fn():
        optimizer.zero_grad()
        scaled_logits = all_logits / model.temperature.cpu()
        loss = ce(scaled_logits, all_labels)
        loss.backward()
        return loss

    for _ in range(n_epochs):
        optimizer.step(eval_fn)

    model.temperature.requires_grad_(False)
    print(f"Calibrated temperature: {model.temperature.item():.4f}")
    return model
