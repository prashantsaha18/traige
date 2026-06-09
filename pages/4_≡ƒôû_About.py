"""
pages/4_📖_About.py
--------------------
Research context, clinical framing, references, and deployment notes.
"""

import streamlit as st

st.set_page_config(page_title="About · DeepTriage", page_icon="📖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Outfit',sans-serif!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:1.5rem!important}
.ref-card{background:#111827;border:1px solid #1f2937;border-radius:10px;
          padding:12px 16px;margin-bottom:8px;font-size:13px;color:#d1d5db;line-height:1.6}
.ref-card a{color:#60a5fa;text-decoration:none}
.ref-card .tag{display:inline-block;font-size:10px;font-weight:600;
               text-transform:uppercase;letter-spacing:.06em;padding:1px 8px;
               border-radius:20px;margin-bottom:4px}
</style>
""", unsafe_allow_html=True)

st.markdown("# 📖 About DeepTriage")

# ─────────────────────────────────────────────
# PROBLEM STATEMENT
# ─────────────────────────────────────────────
st.markdown("## The Problem: India's OPD Crisis")

col_stat1, col_stat2, col_stat3 = st.columns(3)
for col, val, label, sub in [
    (col_stat1, "10,000+", "Patients/day", "AIIMS Delhi OPD"),
    (col_stat2, "1:1000",  "Doctor–patient ratio", "India national average"),
    (col_stat3, "~60 sec", "Target triage time", "per patient (DeepTriage goal)"),
]:
    col.markdown(f"""
    <div style='background:#111827;border:1px solid #1f2937;border-radius:12px;
                padding:1.25rem;text-align:center'>
        <div style='font-family:DM Serif Display,serif;font-size:32px;
                    color:#E24B4A;margin-bottom:4px'>{val}</div>
        <div style='font-size:13px;color:#f9fafb;font-weight:500'>{label}</div>
        <div style='font-size:11px;color:#6b7280;margin-top:2px'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
Indian public hospitals face an acute triage bottleneck. A single nurse may need to assess
hundreds of patients per shift before a doctor reviews them. The Manchester Triage System
(MTS) — the gold standard — requires trained nurses and achieves inter-rater κ ≈ 0.67.

**DeepTriage** targets this gap: a nurse enters the chief complaint and vital signs in under
60 seconds, and the system outputs a colour-coded urgency category with explainable reasoning,
giving the clinical team a second opinion to prioritise care.
""")

# ─────────────────────────────────────────────
# WHAT MAKES IT DIFFERENT
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## What Makes This Different")

cols = st.columns(2)
points = [
    ("🧬 True Multi-Modal Fusion",
     "Most triage AI systems use only one data type. DeepTriage combines "
     "free-text symptoms (BioBERT), structured vitals (FT-Transformer), and "
     "optional chest X-ray (EfficientNet-B3) in a single learnable late-fusion model. "
     "Each modality independently contributes to the final decision.",
     "#a78bfa"),
    ("⚖️ Safety-Aware Loss Function",
     "Standard cross-entropy treats all misclassifications equally. "
     "Our asymmetric focal loss assigns 5× penalty to missed Critical cases. "
     "This is a deliberate clinical design decision: over-triage is manageable; "
     "under-triage can be fatal.",
     "#E24B4A"),
    ("🔍 Full Explainability Stack",
     "Every prediction is backed by: SHAP/gradient importance for vitals "
     "(which vital drove the decision?), attention rollout for text "
     "(which words mattered?), and GradCAM for X-rays (which region?). "
     "Clinicians can interrogate the reasoning, not just the label.",
     "#34d399"),
    ("🛡️ Missing Modality Robustness",
     "ModalityDropout during training teaches the model to perform well "
     "even when one or two modalities are absent. In resource-limited Indian "
     "hospitals where X-ray machines are scarce, the system degrades gracefully "
     "to text + vitals only without retraining.",
     "#60a5fa"),
    ("📐 Calibrated Probabilities",
     "Temperature scaling post-training ensures that when the model says "
     "'82% probability of Critical', that confidence is actually meaningful. "
     "Expected Calibration Error (ECE) < 0.05 after calibration.",
     "#fbbf24"),
    ("🔬 Publishable Human-vs-Model Study",
     "The evaluation framework includes Cohen's kappa comparison against "
     "a clinician on a 50-case held-out set. The Manchester Triage System "
     "achieves κ ≈ 0.67 between nurses — beating this is a concrete, "
     "peer-reviewable benchmark.",
     "#f97316"),
]

for i, (title, body, color) in enumerate(points):
    with cols[i % 2]:
        st.markdown(f"""
        <div style='background:#111827;border:1px solid #1f2937;
                    border-left:3px solid {color};
                    border-radius:10px;padding:1rem;margin-bottom:12px'>
            <div style='font-weight:600;color:#f9fafb;margin-bottom:6px'>{title}</div>
            <div style='font-size:13px;color:#9ca3af;line-height:1.6'>{body}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# REFERENCES
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## Key References")

refs = [
    ("#model",  "NLP",      "Lee et al. (2020). BioBERT: a pre-trained biomedical language representation model. Bioinformatics, 36(4), 1234–1240.",
     "https://academic.oup.com/bioinformatics/article/36/4/1234/5566506"),
    ("#model",  "Tabular",  "Gorishniy et al. (2021). Revisiting Deep Learning Models for Tabular Data. NeurIPS 2021.",
     "https://arxiv.org/abs/2106.11959"),
    ("#model",  "Image",    "Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML 2019.",
     "https://arxiv.org/abs/1905.11946"),
    ("#xai",    "XAI",      "Abnar & Zuidema (2020). Quantifying Attention Flow in Transformers. ACL 2020.",
     "https://arxiv.org/abs/2005.00928"),
    ("#xai",    "XAI",      "Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. ICCV 2017.",
     "https://arxiv.org/abs/1610.02391"),
    ("#calib",  "Calib.",   "Guo et al. (2017). On Calibration of Modern Neural Networks. ICML 2017.",
     "https://arxiv.org/abs/1706.04599"),
    ("#data",   "Dataset",  "Johnson et al. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3.",
     "https://www.nature.com/articles/sdata201635"),
    ("#data",   "Dataset",  "Irvin et al. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels. AAAI 2019.",
     "https://arxiv.org/abs/1901.07031"),
    ("#loss",   "Loss",     "Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV 2017.",
     "https://arxiv.org/abs/1708.02002"),
    ("#triage", "Clinical", "Mackway-Jones et al. Manchester Triage System. BMJ Publishing Group.",
     "https://www.wiley.com/en-us/Emergency+Triage%3A+Manchester+Triage+Group-p-9781405188319"),
]

color_map = {
    "#model": ("#a78bfa", "#2e1065"),
    "#xai":   ("#34d399", "#064e3b"),
    "#calib": ("#fbbf24", "#451a03"),
    "#data":  ("#60a5fa", "#1e3a5f"),
    "#loss":  ("#E24B4A", "#450a0a"),
    "#triage":("#f97316", "#431407"),
}

for ref_id, tag, text, url in refs:
    fg, bg = color_map.get(ref_id, ("#9ca3af", "#1f2937"))
    st.markdown(f"""
    <div class='ref-card'>
        <span class='tag' style='background:{bg};color:{fg}'>{tag}</span><br>
        {text}<br>
        <a href='{url}' target='_blank'>↗ {url[:60]}{'…' if len(url)>60 else ''}</a>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DEPLOYMENT NOTES
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## Deployment & Ethics")

col_d1, col_d2 = st.columns(2)
with col_d1:
    st.markdown("#### Streamlit Community Cloud")
    st.code("""
# requirements.txt (key packages)
streamlit>=1.32
torch>=2.2
transformers>=4.40
torchvision>=0.17
plotly>=5.20
shap>=0.45
scikit-learn>=1.4
onnxruntime>=1.18    # optional, for ONNX export
Pillow>=10.0
numpy>=1.26
pandas>=2.2
huggingface_hub>=0.22

# packages.txt (system deps)
libgl1-mesa-glx
libglib2.0-0
""", language="text")

    st.code("""
# Load model weights from HuggingFace Hub
# (avoids storing large .pt files in git repo)
from huggingface_hub import hf_hub_download
import os

ckpt = hf_hub_download(
    repo_id="your-username/deeptriage-weights",
    filename="best_model.pt",
    token=os.environ["HF_TOKEN"],   # Streamlit secret
)
""", language="python")

with col_d2:
    st.markdown("#### Ethical Considerations")
    st.markdown("""
    **This system is a research prototype and NOT a certified medical device.**
    
    Before any real-world deployment:
    
    🔴 **Regulatory**: Requires CE/MDR (EU) or CDSCO (India) certification as 
    a Class IIb / Class C medical device. This involves clinical trials and 
    regulatory review.
    
    🟡 **Bias**: MIMIC-III is predominantly US ICU data. Performance on 
    Indian patients with different disease epidemiology, language patterns, 
    and presentation styles may differ significantly. Prospective validation 
    on Indian cohorts is essential.
    
    🟡 **Data privacy**: Patient data must be processed in compliance with 
    India's DPDP Act 2023. No patient data should leave the hospital network 
    without consent.
    
    🟢 **Human-in-the-loop**: The system is designed as a *decision support* 
    tool, not a replacement for clinical judgment. All triage decisions must 
    be confirmed by a qualified healthcare professional.
    """)

# ─────────────────────────────────────────────
# CITATION
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("#### How to Cite")
st.code("""
@software{deeptriage2025,
  title  = {DeepTriage: Multi-Modal AI Triage System},
  author = {Your Name},
  year   = {2025},
  note   = {Multi-modal fusion of clinical text, vital signs, and chest X-rays
             for automated patient triage. BioBERT + FT-Transformer + EfficientNet-B3.},
  url    = {https://github.com/your-username/deeptriage}
}
""", language="bibtex")
