"""
pages/3_🏗️_Architecture.py
---------------------------
Model architecture documentation and parameter breakdown.
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Architecture · DeepTriage", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Outfit',sans-serif!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:1.5rem!important}
code{font-family:'DM Mono',monospace!important;font-size:12px!important}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🏗️ Model Architecture")

# ─────────────────────────────────────────────
# ARCHITECTURE DIAGRAM (text-based)
# ─────────────────────────────────────────────
st.markdown("""
<div style='background:#0d1117;border:1px solid #1f2937;border-radius:12px;
            padding:1.5rem;font-family:DM Mono,monospace;font-size:13px;
            color:#9ca3af;line-height:2.0'>

<span style='color:#6b7280'>┌──────────────────────────────────────────────────────────────────────────┐</span><br>
<span style='color:#6b7280'>│                     MultiModalTriageModel                               │</span><br>
<span style='color:#6b7280'>├──────────────────┬─────────────────────┬────────────────────────────────┤</span><br>
<span style='color:#6b7280'>│</span>  <span style='color:#a78bfa'>TEXT ENCODER</span>     <span style='color:#6b7280'>│</span>  <span style='color:#34d399'>VITALS ENCODER</span>      <span style='color:#6b7280'>│</span>  <span style='color:#60a5fa'>IMAGE ENCODER</span>  (optional)  <span style='color:#6b7280'>│</span><br>
<span style='color:#6b7280'>│</span>  Chief complaint  <span style='color:#6b7280'>│</span>  12 vital features   <span style='color:#6b7280'>│</span>  Chest X-ray (300×300)      <span style='color:#6b7280'>│</span><br>
<span style='color:#6b7280'>│</span>       ↓           <span style='color:#6b7280'>│</span>         ↓           <span style='color:#6b7280'>│</span>          ↓                  <span style='color:#6b7280'>│</span><br>
<span style='color:#6b7280'>│</span>  <span style='color:#a78bfa'>BioBERT-base</span>     <span style='color:#6b7280'>│</span>  <span style='color:#34d399'>FeatureTokenizer</span>    <span style='color:#6b7280'>│</span>  <span style='color:#60a5fa'>EfficientNet-B3</span>             <span style='color:#6b7280'>│</span><br>
<span style='color:#6b7280'>│</span>  12 layers/768d   <span style='color:#6b7280'>│</span>  d_token=64          <span style='color:#6b7280'>│</span>  pretrained ImageNet        <span style='color:#6b7280'>│</span><br>
<span style='color:#6b7280'>│</span>       ↓           <span style='color:#6b7280'>│</span>         ↓           <span style='color:#6b7280'>│</span>          ↓                  <span style='color:#6b7280'>│</span><br>
<span style='color:#6b7280'>│</span>  <span style='color:#a78bfa'>CLS token</span>        <span style='color:#6b7280'>│</span>  <span style='color:#34d399'>TransformerEncoder</span>  <span style='color:#6b7280'>│</span>  <span style='color:#60a5fa'>Global Avg Pool</span>             <span style='color:#6b7280'>│</span><br>
<span style='color:#6b7280'>│</span>  → proj 256d      <span style='color:#6b7280'>│</span>  2L, 4H, mean-pool  <span style='color:#6b7280'>│</span>  → proj 256d                <span style='color:#6b7280'>│</span><br>
<span style='color:#6b7280'>│</span>  (256,)           <span style='color:#6b7280'>│</span>  (256,)             <span style='color:#6b7280'>│</span>  (256,)                     <span style='color:#6b7280'>│</span><br>
<span style='color:#6b7280'>│</span>       ↓           <span style='color:#6b7280'>│</span>         ↓           <span style='color:#6b7280'>│</span>          ↓                  <span style='color:#6b7280'>│</span><br>
<span style='color:#6b7280'>│</span>  ModalityDrop     <span style='color:#6b7280'>│</span>  ModalityDrop       <span style='color:#6b7280'>│</span>  ModalityDrop               <span style='color:#6b7280'>│</span><br>
<span style='color:#6b7280'>│</span>  p=0.15           <span style='color:#6b7280'>│</span>  p=0.15             <span style='color:#6b7280'>│</span>  p=0.15                     <span style='color:#6b7280'>│</span><br>
<span style='color:#6b7280'>└──────────┬───────┴────────────┬────────┴──────────────┬─────────────────┘</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#f59e0b'>↓</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#f59e0b'>↓</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#f59e0b'>↓</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#f59e0b'>└──────────────────────────────────┘</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#f59e0b'>Concatenate → (768,)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#f59e0b'>↓</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#fbbf24'>┌──────────────────────┐</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#fbbf24'>│     Triage Head      │</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#fbbf24'>│  768 → 256 → 128 → 3 │</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#fbbf24'>│  LayerNorm + GELU     │</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#fbbf24'>│  + Temperature Scale  │</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#fbbf24'>└──────────────────────┘</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#e5e7eb'>↓</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#e5e7eb'>Critical | Urgent | Non-urgent</span>

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COMPONENT DEEP-DIVES
# ─────────────────────────────────────────────
col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown("#### Text Encoder")
    st.markdown("""
    **BioBERT-base-cased-v1.2** — Pre-trained on PubMed + PMC biomedical text.
    
    - Input: tokenised chief complaint (max 256 tokens)
    - Extract: `[CLS]` token → 768-dim embedding
    - Project: 768 → 256 (Linear + LayerNorm + GELU)
    - **Differential fine-tuning**: layers 0–7 frozen, layers 8–11 trainable (lr = 2e-5)
    - Output: 256-dim text embedding + attention weights (for XAI)
    
    *Why BioBERT over vanilla BERT?* Pre-training on clinical text means it already 
    understands terms like "diaphoresis", "Kussmaul breathing", "McBurney's point" 
    without needing to learn them from scratch.
    """)

    st.markdown("#### FT-Transformer (Vitals)")
    st.markdown("""
    **Feature Tokenizer + Transformer** (Gorishniy et al. 2021).
    
    - Each scalar vital → individual learnable token (d=64)
    - 2 transformer encoder layers, 4 attention heads
    - Mean-pool over feature tokens → 64-dim → projected to 256
    - Handles missing values: NaN → imputed to population mean (z=0)
    - **ModalityDropout**: randomly zeroes entire modality (p=0.15) during training
      for robustness to missing modalities at inference
    """)

with col_b:
    st.markdown("#### Image Encoder (optional)")
    st.markdown("""
    **EfficientNet-B3** — ImageNet pre-trained, fine-tuned on CheXpert.
    
    - Input: 300×300 RGB chest X-ray (normalised)
    - Early stages frozen (features.0–1), deep stages trainable
    - Global average pool → 1536-dim → projected to 256
    - **GradCAM** hooks attached to `features.8` for saliency maps
    - Gracefully skipped if no X-ray provided (zero-padded embedding)
    """)

    st.markdown("#### Training Design Choices")
    st.markdown("""
    **Asymmetric Focal Loss** — Critical class weight 5×, Urgent 2×, Non-urgent 1×.
    Focal modulation γ=2.0 downweights easy examples.
    *Rationale*: missing a Critical case is life-threatening; we accept more false positives.
    
    **Weighted Sampler** — corrects class imbalance (25/35/40 split) during training.
    
    **Temperature Scaling** — post-hoc calibration via L-BFGS on validation set.
    Target ECE < 0.05 after calibration.
    
    **Differential LR** — BioBERT at 2e-5, other modules at 3e-4.
    Cosine annealing with warm restarts (T₀=7 epochs).
    """)

# ─────────────────────────────────────────────
# PARAMETER COUNT TABLE
# ─────────────────────────────────────────────
st.markdown("#### Parameter Breakdown")

param_data = {
    "Component": [
        "BioBERT encoder (frozen layers 0-7)",
        "BioBERT encoder (trainable layers 8-11)",
        "BioBERT projection head",
        "FT-Transformer (vitals)",
        "EfficientNet-B3 (frozen early)",
        "EfficientNet-B3 (trainable deep)",
        "EfficientNet projection head",
        "Fusion + Triage head",
        "Total trainable",
        "Total parameters",
    ],
    "Parameters": [
        "~55M", "~27M", "~197K", "~280K",
        "~5M", "~7M", "~393K",
        "~420K",
        "~36M", "~91M",
    ],
    "Status": [
        "Frozen", "Trainable", "Trainable", "Trainable",
        "Frozen", "Trainable", "Trainable", "Trainable",
        "—", "—",
    ],
    "Notes": [
        "Standard pre-trained BioBERT weights",
        "Fine-tuned at 2e-5 lr",
        "Linear → LayerNorm → GELU",
        "FeatureTokenizer + 2L Transformer",
        "features.0 and features.1",
        "features.2–8 + classifier replacement",
        "Linear → LayerNorm → GELU",
        "MLP: 768→256→128→3",
        "Without frozen layers",
        "Full model",
    ]
}

import pandas as pd
df_params = pd.DataFrame(param_data)
st.dataframe(
    df_params.style.apply(
        lambda col: ["background-color: #1f2937" if i % 2 == 0 else "" for i in range(len(col))],
        axis=0
    ),
    use_container_width=True,
    hide_index=True,
)

# ─────────────────────────────────────────────
# QUICK-START CODE
# ─────────────────────────────────────────────
st.markdown("#### Quick-Start: Model Instantiation")
st.code("""
from models.triage_model import MultiModalTriageModel, AsymmetricTriageLoss

# Initialise (downloads BioBERT weights ~440 MB first time)
model = MultiModalTriageModel(
    text_model="dmis-lab/biobert-base-cased-v1.2",
    with_image=True,          # set False to skip X-ray branch
    modality_drop_p=0.15,     # modality dropout probability
)

# Loss function with asymmetric class weights
criterion = AsymmetricTriageLoss(
    class_weights=(5.0, 2.0, 1.0),  # Critical, Urgent, Non-urgent
    gamma=2.0,
)

# Forward pass
logits, probs, attentions, modalities = model(
    text_inputs={"input_ids": ..., "attention_mask": ...},
    vitals=vitals_tensor,      # (B, 12) normalised floats
    image=None,                # optional (B, 3, 300, 300)
)
# probs: (B, 3)  — [P(Critical), P(Urgent), P(Non-urgent)]
""", language="python")
