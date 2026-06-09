"""
DeepTriage — Intelligent Medical Triage System
================================================
Multi-modal ML triage: symptoms (BioBERT) + vitals (FT-Transformer)
+ optional chest X-ray (EfficientNet-B3) → Critical / Urgent / Non-urgent

Run:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DeepTriage · Medical AI Triage",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# INITIALISE SESSION STATE KEYS
# ─────────────────────────────────────────────
default_states = {
    "complaint": "",
    "age": 45,
    "gender": "Male",
    "hr": 88,
    "sbp": 120,
    "dbp": 78,
    "spo2": 98,
    "rr": 16,
    "gcs": 15,
    "temp": 37.0,
    "wbc": 0.0,
    "creat": 0.0,
    "gluc": 0.0,
    "hgb": 0.0,
}
for k, v in default_states.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Imports ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090d14 0%, #0d1117 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
[data-testid="stSidebar"] * { color: #e5e7eb !important; }
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stNumberInput > label,
[data-testid="stSidebar"] .stSelectbox > label { color: #9ca3af !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.03em; }

/* ── Cards ── */
.triage-card {
    background: rgba(22, 27, 34, 0.4);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

/* ── Urgency badge ── */
.badge-critical  { 
    background: rgba(226, 75, 74, 0.1); 
    color: #ff6b6b; 
    border: 1px solid rgba(226, 75, 74, 0.4);
    box-shadow: 0 0 15px rgba(226, 75, 74, 0.15);
    text-shadow: 0 0 5px rgba(226, 75, 74, 0.3);
    padding: 6px 20px; 
    border-radius: 30px; 
    font-weight: 600; 
    font-size: 20px; 
    display: inline-block; 
}
.badge-urgent    { 
    background: rgba(245, 158, 11, 0.1); 
    color: #fbbf24; 
    border: 1px solid rgba(245, 158, 11, 0.4);
    box-shadow: 0 0 15px rgba(245, 158, 11, 0.15);
    text-shadow: 0 0 5px rgba(245, 158, 11, 0.3);
    padding: 6px 20px; 
    border-radius: 30px; 
    font-weight: 600; 
    font-size: 20px; 
    display: inline-block; 
}
.badge-nonurgent { 
    background: rgba(16, 185, 129, 0.1); 
    color: #34d399; 
    border: 1px solid rgba(16, 185, 129, 0.4);
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
    text-shadow: 0 0 5px rgba(16, 185, 129, 0.3);
    padding: 6px 20px; 
    border-radius: 30px; 
    font-weight: 600; 
    font-size: 20px; 
    display: inline-block; 
}

/* ── Section headers ── */
.section-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 600;
    color: #6b7280;
    margin-bottom: 8px;
    margin-top: 1.5rem;
}

/* ── DDx row ── */
.ddx-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    font-size: 14px;
}
.ddx-row:last-child { border-bottom: none; }

/* ── Metric pill ── */
.metric-pill {
    background: rgba(31, 41, 55, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 12px 18px;
    text-align: center;
}
.metric-pill .val { font-family:'DM Serif Display',serif; font-size:28px; color:#f9fafb; }
.metric-pill .lbl { font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:.05em; }

/* ── Alert box ── */
.disclaimer {
    background: rgba(245, 158, 11, 0.05);
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 11px;
    color: #fbbf24;
    margin-top: 1rem;
}

/* ── Streamlit overrides ── */
.stTextArea textarea {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background-color: rgba(17, 24, 39, 0.3) !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all 0.2s ease !important;
}
.stTextArea textarea:focus {
    border-color: rgba(226, 75, 74, 0.5) !important;
    box-shadow: 0 0 0 2px rgba(226, 75, 74, 0.2) !important;
}

/* Base button overrides (Preset Buttons) */
.stButton button {
    width: 100% !important;
    border-radius: 10px !important;
    background: rgba(31, 41, 55, 0.4) !important;
    color: #e5e7eb !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.2s ease !important;
}
.stButton button:hover {
    background: rgba(31, 41, 55, 0.8) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
    color: white !important;
    transform: translateY(-1px);
}

/* Primary Button Container Override */
.primary-button-container .stButton button {
    background: linear-gradient(135deg, #E24B4A 0%, #c53030 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 0.75rem !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(226, 75, 74, 0.3) !important;
    border-radius: 12px !important;
}
.primary-button-container .stButton button:hover {
    background: linear-gradient(135deg, #c53030 0%, #a62626 100%) !important;
    box-shadow: 0 6px 20px rgba(226, 75, 74, 0.5) !important;
    transform: translateY(-1px);
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 13px !important;
}

/* ── Progress bar colors ── */
.stProgress > div > div { border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading DeepTriage model…")
def load_engine():
    from utils.inference import TriageInferenceEngine
    return TriageInferenceEngine(
        checkpoint_path="checkpoints/best_model.pt",
        normaliser_path="checkpoints/normaliser.json",
        with_image=True,
        device="auto",
    )


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title, col_badge = st.columns([0.08, 0.70, 0.22])
with col_logo:
    st.markdown("<div style='font-size:48px;margin-top:4px'>🏥</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <h1 style='margin:0;font-size:32px;color:#f9fafb'>
        Deep<span style='color:#E24B4A'>Triage</span>
    </h1>
    <p style='margin:0;color:#6b7280;font-size:14px'>
        Multi-modal AI triage · BioBERT + FT-Transformer + EfficientNet-B3
    </p>
    """, unsafe_allow_html=True)
with col_badge:
    st.markdown("""
    <div style='text-align:right;margin-top:8px'>
        <span style='background:#1f2937;color:#9ca3af;font-size:11px;
                     padding:4px 10px;border-radius:20px;font-family:DM Mono,monospace'>
            v1.0 · Research Demo
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1px solid #1f2937;margin:1rem 0'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR — VITALS INPUT
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 Patient Vitals")
    st.markdown("<p style='font-size:12px;color:#8892b0'>Configure patient clinical profile below.</p>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:12px;font-weight:600;color:#a78bfa;margin-top:16px;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em'>📋 Demographics</div>", unsafe_allow_html=True)
    age    = st.number_input("Age (years)",  min_value=1,  max_value=120, key="age", step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")

    st.markdown("<div style='font-size:12px;font-weight:600;color:#ff6b6b;margin-top:16px;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em'>💓 Haemodynamics</div>", unsafe_allow_html=True)
    hr  = st.slider("Heart Rate (bpm)",      40,  200, key="hr")
    sbp = st.slider("Systolic BP (mmHg)",    60,  250, key="sbp")
    dbp = st.slider("Diastolic BP (mmHg)",   30,  150, key="dbp")

    st.markdown("<div style='font-size:12px;font-weight:600;color:#60a5fa;margin-top:16px;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em'>🫁 Respiration & Neuro</div>", unsafe_allow_html=True)
    spo2 = st.slider("SpO₂ (%)",             50,  100, key="spo2")
    rr   = st.slider("Respiratory Rate (/min)", 6, 60, key="rr")
    gcs  = st.slider("GCS (Neurological)",    3,  15, key="gcs")

    st.markdown("<div style='font-size:12px;font-weight:600;color:#fbbf24;margin-top:16px;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em'>🌡️ Temperature & Labs</div>", unsafe_allow_html=True)
    temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=43.0, key="temp", step=0.1)

    with st.expander("🔬 Optional Laboratory Values", expanded=False):
        wbc   = st.number_input("WBC (×10⁹/L)",    0.0, 50.0, key="wbc", step=0.1, help="0 = not measured")
        creat = st.number_input("Creatinine (mg/dL)", 0.0, 15.0, key="creat", step=0.1, help="0 = not measured")
        gluc  = st.number_input("Glucose (mg/dL)", 0.0, 800.0, key="gluc", step=1.0, help="0 = not measured")
        hgb   = st.number_input("Haemoglobin (g/dL)", 0.0, 25.0, key="hgb", step=0.1, help="0 = not measured")

    st.markdown("---")
    st.markdown("**Chest X-ray (optional)**")
    xray_file = st.file_uploader("Upload X-ray (JPG/PNG)", type=["jpg","jpeg","png"])

    st.markdown("""
    <div class='disclaimer'>
        ⚠️ Research prototype only. Not a certified medical device.
        Do not use for clinical decisions.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN PANEL — SYMPTOM INPUT + PREDICT
# ─────────────────────────────────────────────
col_input, col_results = st.columns([0.42, 0.58], gap="large")

with col_input:
    st.markdown("#### Chief Complaint & History")
    complaint = st.text_area(
        "Describe symptoms in free text",
        height=160,
        placeholder=(
            "e.g. 58-year-old male with sudden onset severe "
            "crushing chest pain radiating to the left arm, "
            "diaphoresis, and nausea for the past 40 minutes. "
            "History of hypertension and T2DM."
        ),
        key="complaint",
        label_visibility="collapsed",
    )

    # Example buttons
    st.markdown("<p style='font-size:12px;color:#8892b0;margin-top:4px;font-weight:500'>Quick Clinical Profiles (Prefills Text & Vitals):</p>", unsafe_allow_html=True)
    eg1, eg2, eg3 = st.columns(3)
    
    if eg1.button("🔴 Critical: STEMI"):
        st.session_state["complaint"] = "Sudden crushing chest pain radiating to left arm and jaw, profuse diaphoresis, nausea. Onset 30 min ago. History of hypertension."
        st.session_state["age"] = 62
        st.session_state["gender"] = "Male"
        st.session_state["hr"] = 118
        st.session_state["sbp"] = 88
        st.session_state["dbp"] = 55
        st.session_state["spo2"] = 91
        st.session_state["rr"] = 24
        st.session_state["gcs"] = 14
        st.session_state["temp"] = 36.9
        st.session_state["wbc"] = 11.2
        st.session_state["creat"] = 1.2
        st.session_state["gluc"] = 180.0
        st.session_state["hgb"] = 13.5
        st.rerun()

    if eg2.button("🟡 Urgent: Pneumonia"):
        st.session_state["complaint"] = "Productive cough with yellow sputum for 4 days, fever 38.5°C, right-sided pleuritic chest pain, shortness of breath on exertion."
        st.session_state["age"] = 48
        st.session_state["gender"] = "Female"
        st.session_state["hr"] = 102
        st.session_state["sbp"] = 120
        st.session_state["dbp"] = 78
        st.session_state["spo2"] = 93
        st.session_state["rr"] = 22
        st.session_state["gcs"] = 15
        st.session_state["temp"] = 38.8
        st.session_state["wbc"] = 16.5
        st.session_state["creat"] = 0.9
        st.session_state["gluc"] = 120.0
        st.session_state["hgb"] = 12.8
        st.rerun()

    if eg3.button("🟢 Stable: URTI"):
        st.session_state["complaint"] = "Runny nose, sore throat, mild dry cough for 2 days. No fever. Well-appearing, good oral intake."
        st.session_state["age"] = 24
        st.session_state["gender"] = "Female"
        st.session_state["hr"] = 72
        st.session_state["sbp"] = 120
        st.session_state["dbp"] = 80
        st.session_state["spo2"] = 99
        st.session_state["rr"] = 14
        st.session_state["gcs"] = 15
        st.session_state["temp"] = 36.7
        st.session_state["wbc"] = 7.4
        st.session_state["creat"] = 0.8
        st.session_state["gluc"] = 95.0
        st.session_state["hgb"] = 14.1
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='primary-button-container'>", unsafe_allow_html=True)
    run_btn = st.button("⚡ Run Triage Analysis", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Modality status
    st.markdown("<div class='section-label'>Active Modalities</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.markdown("<div style='text-align:center;font-size:22px'>📝</div><div style='text-align:center;font-size:11px;color:#9ca3af'>Text</div>", unsafe_allow_html=True)
    m2.markdown("<div style='text-align:center;font-size:22px'>📊</div><div style='text-align:center;font-size:11px;color:#9ca3af'>Vitals</div>", unsafe_allow_html=True)
    xray_color = "#22c55e" if xray_file else "#374151"
    m3.markdown(f"<div style='text-align:center;font-size:22px;color:{xray_color}'>🫁</div>"
                f"<div style='text-align:center;font-size:11px;color:#9ca3af'>X-ray {'✓' if xray_file else '—'}</div>",
                unsafe_allow_html=True)

    if xray_file:
        from PIL import Image
        xray_img = Image.open(xray_file)
        st.image(xray_img, caption="Uploaded X-ray", use_container_width=True)
    else:
        xray_img = None


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if run_btn:
    if not complaint or len(complaint.strip()) < 10:
        st.error("Please enter a chief complaint (at least 10 characters).")
    else:
        vitals_dict = {
            "age":             age,
            "heart_rate":      hr,
            "systolic_bp":     sbp,
            "diastolic_bp":    dbp,
            "spo2":            spo2,
            "temperature":     temp,
            "respiratory_rate": rr,
            "gcs":             gcs,
            "wbc":             wbc   if wbc   > 0 else None,
            "creatinine":      creat if creat > 0 else None,
            "glucose":         gluc  if gluc  > 0 else None,
            "hemoglobin":      hgb   if hgb   > 0 else None,
        }

        with st.spinner("Analysing…"):
            try:
                engine = load_engine()
                result = engine.predict(
                    chief_complaint=complaint,
                    vitals=vitals_dict,
                    image=xray_img,
                    explain=True,
                )
                st.session_state["result"]    = result
                st.session_state["complaint"] = complaint
                st.session_state["vitals"]    = vitals_dict
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()


# ─────────────────────────────────────────────
# RESULTS PANEL
# ─────────────────────────────────────────────
with col_results:
    if "result" not in st.session_state:
        st.markdown("""
        <div style='text-align:center;padding:3rem 0;color:#374151'>
            <div style='font-size:64px;margin-bottom:1rem'>🩺</div>
            <p style='font-family:DM Serif Display,serif;font-size:20px;color:#6b7280'>
                Enter symptoms and vitals,<br>then run the triage analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        result = st.session_state["result"]
        urgency = result["urgency"]
        probs   = result["probabilities"]

        # ── URGENCY BADGE ──
        badge_cls = {
            "Critical": "badge-critical",
            "Urgent": "badge-urgent",
            "Non-urgent": "badge-nonurgent"
        }[urgency]

        icons = {"Critical": "🔴", "Urgent": "🟡", "Non-urgent": "🟢"}

        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:16px;margin-bottom:1.5rem'>
            <div>
                <div style='font-size:11px;text-transform:uppercase;letter-spacing:.1em;
                             color:#6b7280;margin-bottom:4px'>Triage Result</div>
                <span class='{badge_cls}'>{icons[urgency]} {urgency}</span>
            </div>
            <div style='margin-left:auto;text-align:right'>
                <div style='font-size:11px;color:#6b7280'>Inference</div>
                <div style='font-family:DM Mono,monospace;font-size:16px;color:#9ca3af'>
                    {result['inference_time_ms']:.0f} ms
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── PROBABILITY GAUGE ──
        tabs = st.tabs(["📊 Probabilities", "🧬 Diagnoses", "🔬 Explainability"])

        with tabs[0]:
            fig = go.Figure()
            classes = ["Critical", "Urgent", "Non-urgent"]
            colors  = ["#E24B4A", "#BA7517", "#3B6D11"]
            bgs     = ["#FCEBEB", "#FAEEDA", "#EAF3DE"]

            for cls, col, bg in zip(classes, colors, bgs):
                p = probs[cls]
                fig.add_trace(go.Bar(
                    x=[p], y=[cls], orientation="h",
                    marker_color=col,
                    text=f"{p*100:.1f}%",
                    textposition="outside",
                    name=cls,
                    showlegend=False,
                ))

            fig.update_layout(
                xaxis=dict(range=[0, 1.15], showgrid=False, zeroline=False,
                           tickformat=".0%", color="#9ca3af"),
                yaxis=dict(showgrid=False, color="#e5e7eb"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=40, t=10, b=10),
                height=160,
                font=dict(family="Outfit", color="#e5e7eb"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Confidence interpretation
            top_prob = probs[urgency]
            if top_prob > 0.80:
                conf_label = "High confidence"
                conf_color = "#22c55e"
            elif top_prob > 0.55:
                conf_label = "Moderate confidence"
                conf_color = "#f59e0b"
            else:
                conf_label = "Low confidence — consider clinical judgment"
                conf_color = "#ef4444"

            st.markdown(f"<p style='font-size:13px;color:{conf_color}'>"
                        f"● {conf_label} ({top_prob*100:.1f}%)</p>",
                        unsafe_allow_html=True)

            st.markdown(f"<p style='font-size:12px;color:#6b7280'>"
                        f"Modalities used: {', '.join(result['modalities_used'])}</p>",
                        unsafe_allow_html=True)

        with tabs[1]:
            ddx = result.get("diagnoses", [])
            if ddx:
                st.markdown("<div class='section-label'>Top Differential Diagnoses</div>",
                            unsafe_allow_html=True)
                for i, d in enumerate(ddx[:5]):
                    conf  = d["confidence"]
                    width = int(conf * 100)
                    bar_col = colors[0] if urgency == "Critical" else \
                              colors[1] if urgency == "Urgent" else colors[2]
                    st.markdown(f"""
                    <div class='ddx-row'>
                        <div>
                            <span style='color:#e5e7eb;font-weight:500'>{i+1}. {d['diagnosis']}</span>
                        </div>
                        <div style='display:flex;align-items:center;gap:10px'>
                            <div style='width:80px;height:6px;background:#1f2937;border-radius:3px'>
                                <div style='width:{width}%;height:100%;background:{bar_col};border-radius:3px'></div>
                            </div>
                            <span style='font-family:DM Mono,monospace;font-size:13px;
                                         color:#9ca3af;min-width:38px'>{conf*100:.0f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div style='font-size:11px;color:#4b5563;margin-top:8px;font-style:italic'>
                    Note: differential diagnoses are AI-generated suggestions and require clinical validation.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No differential diagnoses available.")

        with tabs[2]:
            xai_tabs = st.tabs(["🔢 Vital Signs (SHAP)", "📝 Text Attention", "🫁 X-ray GradCAM"])

            with xai_tabs[0]:
                shap_vals = result.get("shap_values")
                if shap_vals:
                    # Sort by absolute value
                    items = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
                    names  = [k.replace("_", " ").title() for k, _ in items]
                    values = [v for _, v in items]
                    bar_colors = ["#E24B4A" if v > 0 else "#185FA5" for v in values]

                    fig_shap = go.Figure(go.Bar(
                        x=values, y=names, orientation="h",
                        marker_color=bar_colors,
                        text=[f"{v:+.3f}" for v in values],
                        textposition="outside",
                    ))
                    fig_shap.update_layout(
                        title=dict(text="Feature importance for prediction",
                                   font=dict(size=13, color="#9ca3af")),
                        xaxis=dict(showgrid=True, gridcolor="#1f2937",
                                   zeroline=True, zerolinecolor="#374151",
                                   color="#9ca3af"),
                        yaxis=dict(showgrid=False, color="#e5e7eb"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=60, t=30, b=10),
                        height=280,
                        font=dict(family="Outfit", color="#e5e7eb"),
                    )
                    st.plotly_chart(fig_shap, use_container_width=True,
                                   config={"displayModeBar": False})
                    st.markdown("<p style='font-size:11px;color:#6b7280'>"
                                "Red = pushes toward predicted class · Blue = pushes away</p>",
                                unsafe_allow_html=True)
                else:
                    st.info("SHAP values not available.")

            with xai_tabs[1]:
                attn_html = result.get("attention_html")
                if attn_html:
                    st.markdown("<p style='font-size:13px;color:#9ca3af'>"
                                "Token saliency from BioBERT attention rollout. "
                                "Darker red = higher importance.</p>",
                                unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='background:rgba(17, 24, 39, 0.45); border: 1px solid rgba(255, 255, 255, 0.08); padding: 1.25rem; border-radius: 12px; max-height: 250px; overflow-y: auto;'>"
                        f"{attn_html}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.info("Attention heatmap not available.")

            with xai_tabs[2]:
                gradcam = result.get("gradcam_overlay")
                if gradcam is not None:
                    col_xray_orig, col_xray_cam = st.columns(2)
                    with col_xray_orig:
                        st.image(xray_img, caption="Original Chest X-ray", use_container_width=True)
                    with col_xray_cam:
                        st.image(gradcam, caption="GradCAM Activation Overlay", use_container_width=True)
                    st.markdown("<p style='font-size:11px;color:#8892b0;margin-top:8px'>"
                                "Warm regions (red/yellow) indicate anatomical structures of high importance for the "
                                "classification decision.</p>",
                                unsafe_allow_html=True)
                elif xray_file is None:
                    st.info("Upload a chest X-ray to see GradCAM visualisation.")
                else:
                    st.warning("GradCAM could not be computed for this image.")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><hr style='border:none;border-top:1px solid #1f2937'>", unsafe_allow_html=True)
st.markdown("""
<div style='display:flex;justify-content:space-between;align-items:center;
            font-size:11px;color:#374151;padding:4px 0'>
    <span>DeepTriage · Multi-modal Medical Triage System</span>
    <span style='color:#E24B4A'>⚠ Not a medical device — for research and educational use only</span>
    <span>BioBERT + FT-Transformer + EfficientNet-B3</span>
</div>
""", unsafe_allow_html=True)
