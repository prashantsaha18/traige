# 🏥 DeepTriage — Intelligent Medical Triage System

> Multi-modal AI triage combining clinical text (BioBERT) + vital signs (FT-Transformer) + chest X-ray (EfficientNet-B3) → Critical / Urgent / Non-urgent classification with full explainability.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3-ee4c2c)
![License MIT](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Problem Statement

Indian public hospitals face an acute triage bottleneck — AIIMS Delhi OPD alone handles 10,000+ patients/day with a 1:1000 doctor-to-patient ratio. DeepTriage provides a 60-second AI-assisted triage decision, colour-coded and fully explainable, giving nurses and doctors a reliable second opinion.

## 🏗️ Architecture

```
Symptoms (text)  →  BioBERT-base-cased-v1.2  →  256-dim embedding  ↘
Vitals (tabular) →  FT-Transformer (12 feats) →  256-dim embedding  → Late Fusion (768d) → MLP Head → [Critical | Urgent | Non-urgent]
X-ray (image)    →  EfficientNet-B3 (opt.)   →  256-dim embedding  ↗
```

**Key design decisions:**
- **Asymmetric focal loss** — Critical class gets 5× penalty weight (missing a Critical case is life-threatening)
- **ModalityDropout** (p=0.15) — model learns to work without any one modality (robustness in resource-limited settings)
- **Temperature scaling** — post-hoc calibration for trustworthy probability outputs (ECE < 0.05)
- **Late fusion** — each encoder trains independently, avoids gradient interference between modalities

## 📊 Target Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Macro F1 | ≥ 0.84 | Equally weighted across 3 classes |
| AUC-ROC (Critical) | ≥ 0.96 | Life-safety metric |
| Cohen's κ | ≥ 0.79 | vs. Manchester Triage System (κ≈0.67) |
| ECE (calibration) | < 0.05 | After temperature scaling |
| Inference time | < 500 ms | On CPU |

## 🔬 Explainability Stack

| Modality | Method | Output |
|----------|--------|--------|
| Vitals | Gradient importance | SHAP waterfall chart |
| Text | Attention rollout (Abnar & Zuidema 2020) | Token saliency heatmap |
| X-ray | GradCAM (Selvaraju et al. 2017) | Spatial heatmap overlay |

## 📦 Datasets

| Dataset | Size | Modality | Access |
|---------|------|----------|--------|
| MIMIC-IV | ~40K admissions | Text + vitals | PhysioNet (credentialed) |
| MIMIC-CXR | 227K image-report pairs | Image + text | PhysioNet (credentialed) |
| CheXpert | 224K X-rays | Image | Open access |
| Synthetic (built-in) | 1,000–10,000 | Text + vitals | Generated — no credentials needed |

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/deeptriage
cd deeptriage
pip install -r requirements.txt
```

### 2. Generate synthetic data (no MIMIC access needed)

```bash
python data/synthetic_generator.py --n 2000 --output data/synthetic/cases.csv
```

### 3. Train on synthetic data

```bash
python train.py \
    --data data/synthetic/cases.csv \
    --epochs 20 \
    --batch_size 16 \
    --tokenizer dmis-lab/biobert-base-cased-v1.2
```

> **First run** downloads BioBERT weights (~440 MB) from HuggingFace Hub automatically.

### 4. Launch the app

```bash
streamlit run app.py
```

### 5. (Optional) Train with real MIMIC data

```bash
# Step 1: Preprocess MIMIC-IV files
python utils/mimic_preprocess.py \
    --discharge  /path/to/mimic-iv/hosp/discharge.csv \
    --chartevents /path/to/mimic-iv/icu/chartevents.csv \
    --icustays   /path/to/mimic-iv/icu/icustays.csv \
    --diagnoses  /path/to/mimic-iv/hosp/diagnoses_icd.csv \
    --output     data/mimic_merged.csv

# Step 2: Train with MIMIC + optional image branch
python train.py \
    --mimic_notes  data/mimic_merged.csv \
    --with_image \
    --epochs 30 \
    --batch_size 16
```

---

## 🗂️ Project Structure

```
deeptriage/
├── app.py                          # Main Streamlit app (entry point)
├── train.py                        # Training script
├── requirements.txt
├── packages.txt                    # System deps for Streamlit Cloud
├── .streamlit/
│   └── config.toml                 # Dark theme config
│
├── models/
│   └── triage_model.py             # MultiModalTriageModel, encoders, loss
│
├── utils/
│   ├── dataset.py                  # TriageDataset, VitalsNormaliser, DataLoaders
│   ├── inference.py                # TriageInferenceEngine (Streamlit-facing)
│   └── mimic_preprocess.py         # MIMIC-IV preprocessing pipeline
│
├── explainability/
│   └── xai.py                      # SHAP, attention rollout, GradCAM, calibration
│
├── data/
│   ├── synthetic_generator.py      # Synthetic patient case generator
│   └── synthetic/
│       └── cases.csv               # Generated after running synthetic_generator.py
│
├── pages/
│   ├── 1_📊_Evaluation.py          # Confusion matrix, metrics, ablation study
│   ├── 2_🗃️_Dataset.py             # Dataset explorer and MIMIC guide
│   ├── 3_🏗️_Architecture.py        # Model architecture documentation
│   └── 4_📖_About.py               # Research context, references, ethics
│
└── checkpoints/                    # Created by train.py
    ├── best_model.pt
    ├── final_model.pt
    ├── normaliser.json
    └── training_history.json
```

---

## ☁️ Streamlit Community Cloud Deployment

### Option A: Load weights from HuggingFace Hub (recommended)

```python
# In utils/inference.py, replace local checkpoint loading with:
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(
    repo_id="your-username/deeptriage-weights",
    filename="best_model.pt",
    token=st.secrets["HF_TOKEN"],
)
```

Add `HF_TOKEN` to Streamlit secrets (Settings → Secrets).

### Option B: Git LFS for model files

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes checkpoints/best_model.pt
git commit -m "Add model weights via LFS"
```

### Important: Never commit MIMIC data to a public repository.

---

## 📐 Evaluation Methodology

### Metrics
- **Macro F1** — primary metric, equal weight to all 3 classes
- **AUC-ROC** (one-vs-rest per class)
- **Cohen's κ** — human-model agreement on 50-case held-out set
- **ECE** — Expected Calibration Error (lower is better)

### Human-vs-Model Study
Ask a doctor/nurse friend to triage 50 held-out cases independently.
Compare their triage labels with DeepTriage predictions using Cohen's kappa.
Manchester Triage System inter-rater κ ≈ 0.67 — this is your baseline to beat.

```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(doctor_labels, model_labels)
print(f"Human-model agreement κ = {kappa:.3f}")
```

---

## 📚 Key References

- **BioBERT**: Lee et al. (2020). Bioinformatics 36(4).
- **FT-Transformer**: Gorishniy et al. (2021). NeurIPS 2021.
- **EfficientNet**: Tan & Le (2019). ICML 2019.
- **Attention Rollout**: Abnar & Zuidema (2020). ACL 2020.
- **GradCAM**: Selvaraju et al. (2017). ICCV 2017.
- **Focal Loss**: Lin et al. (2017). ICCV 2017.
- **Temperature Scaling**: Guo et al. (2017). ICML 2017.
- **MIMIC-IV**: Johnson et al. (2023). Scientific Data.
- **CheXpert**: Irvin et al. (2019). AAAI 2019.

---

## ⚠️ Disclaimer

**This is a research prototype and NOT a certified medical device.**
It must not be used for actual clinical triage decisions without regulatory approval,
prospective clinical validation, and qualified medical supervision.

Any deployment in a clinical setting requires compliance with India's DPDP Act 2023
and CDSCO medical device regulations.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

MIMIC data is subject to its own PhysioNet Credentialed Health Data License.
Never include MIMIC data in public repositories.
