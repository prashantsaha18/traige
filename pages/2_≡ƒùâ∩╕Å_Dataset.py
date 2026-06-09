"""
pages/2_🗃️_Dataset.py
-----------------------
Dataset explorer: distribution charts, sample cases,
and MIMIC preprocessing guide.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Dataset · DeepTriage", page_icon="🗃️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Outfit',sans-serif!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:1.5rem!important}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🗃️ Dataset Explorer")

# ─────────────────────────────────────────────
# Load or generate synthetic data
# ─────────────────────────────────────────────
@st.cache_data
def get_synthetic_data():
    try:
        import sys
        sys.path.insert(0, ".")
        from data.synthetic_generator import generate_dataset
        cases = generate_dataset(n=500, seed=42)
        return pd.DataFrame(cases)
    except Exception as e:
        st.warning(f"Could not generate data: {e}")
        return pd.DataFrame()

df = get_synthetic_data()

if df.empty:
    st.error("No data available. Run `python data/synthetic_generator.py` first.")
    st.stop()

# ─────────────────────────────────────────────
# OVERVIEW CARDS
# ─────────────────────────────────────────────
counts = df["label"].value_counts()
c1, c2, c3, c4 = st.columns(4)
for col, (lbl, count, color) in zip(
    [c1, c2, c3, c4],
    [
        ("Total Cases",  len(df),                    "#185FA5"),
        ("Critical",     counts.get("Critical", 0),   "#E24B4A"),
        ("Urgent",       counts.get("Urgent", 0),     "#BA7517"),
        ("Non-urgent",   counts.get("Non-urgent", 0), "#3B6D11"),
    ]
):
    col.markdown(f"""
    <div style='background:#111827;border:1px solid #1f2937;
                border-left:3px solid {color};
                border-radius:10px;padding:1rem;text-align:center'>
        <div style='font-size:11px;color:#6b7280;text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:4px'>{lbl}</div>
        <div style='font-family:DM Serif Display,serif;font-size:28px;color:#f9fafb'>{count}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DISTRIBUTION CHARTS
# ─────────────────────────────────────────────
col_pie, col_vitals = st.columns([0.38, 0.62], gap="large")

with col_pie:
    st.markdown("#### Class Distribution")
    fig_pie = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.55,
        marker_colors=["#E24B4A", "#BA7517", "#3B6D11"],
        textfont=dict(family="Outfit", color="white", size=13),
    ))
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit", color="#e5e7eb"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9ca3af")),
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

with col_vitals:
    st.markdown("#### Vital Sign Distribution by Triage Class")
    vital_sel = st.selectbox(
        "Select vital",
        ["heart_rate", "systolic_bp", "spo2", "temperature", "respiratory_rate", "gcs"],
        format_func=lambda x: x.replace("_", " ").title(),
        label_visibility="collapsed",
    )
    label_colors = {"Critical": "#E24B4A", "Urgent": "#BA7517", "Non-urgent": "#3B6D11"}

    fig_box = go.Figure()
    for label, color in label_colors.items():
        subset = df[df["label"] == label][vital_sel].dropna()
        fig_box.add_trace(go.Box(
            y=subset, name=label,
            marker_color=color, line_color=color,
            fillcolor=color.replace("#", "rgba(") + ",0.2)",
            boxmean=True,
        ))
    fig_box.update_layout(
        xaxis=dict(color="#9ca3af"),
        yaxis=dict(color="#9ca3af", gridcolor="#1f2937",
                   title=vital_sel.replace("_", " ").title()),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit", color="#e5e7eb"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9ca3af")),
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────
# VITALS CORRELATION HEATMAP
# ─────────────────────────────────────────────
st.markdown("#### Vitals Correlation Matrix")
vital_cols = ["heart_rate", "systolic_bp", "diastolic_bp", "spo2",
              "temperature", "respiratory_rate", "gcs", "age"]
corr = df[vital_cols].corr().round(2)

fig_corr = go.Figure(go.Heatmap(
    z=corr.values,
    x=[c.replace("_", " ").title() for c in corr.columns],
    y=[c.replace("_", " ").title() for c in corr.index],
    colorscale=[[0, "#185FA5"], [0.5, "#111827"], [1.0, "#E24B4A"]],
    zmin=-1, zmax=1,
    text=corr.values,
    texttemplate="%{text:.2f}",
    textfont=dict(size=10, color="white"),
))
fig_corr.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit", color="#e5e7eb"),
    xaxis=dict(color="#9ca3af"),
    yaxis=dict(color="#9ca3af"),
    height=320,
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────
# SAMPLE CASES BROWSER
# ─────────────────────────────────────────────
st.markdown("#### Browse Sample Cases")
filter_label = st.selectbox("Filter by urgency", ["All", "Critical", "Urgent", "Non-urgent"])
filtered = df if filter_label == "All" else df[df["label"] == filter_label]
sample = filtered.sample(min(5, len(filtered)), random_state=42)

for _, row in sample.iterrows():
    color_map = {"Critical": "#E24B4A", "Urgent": "#BA7517", "Non-urgent": "#3B6D11"}
    bg_map    = {"Critical": "#3d0b0b", "Urgent": "#3d2a0a", "Non-urgent": "#0b2d14"}
    lbl = row["label"]
    st.markdown(f"""
    <div style='background:#111827;border:1px solid #1f2937;
                border-left:3px solid {color_map[lbl]};
                border-radius:10px;padding:1rem;margin-bottom:8px'>
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:8px'>
            <span style='background:{bg_map[lbl]};color:{color_map[lbl]};
                         font-size:12px;padding:2px 10px;border-radius:20px;
                         font-weight:600'>{lbl}</span>
            <span style='font-size:12px;color:#6b7280'>{row.get("condition","")}</span>
            <span style='margin-left:auto;font-size:12px;color:#4b5563'>
                Age {int(row["age"])} · {row.get("gender","M")}
            </span>
        </div>
        <div style='font-size:13px;color:#d1d5db;line-height:1.6;margin-bottom:8px'>
            "{row['chief_complaint']}"
        </div>
        <div style='display:flex;gap:12px;font-size:12px;color:#6b7280;
                    font-family:DM Mono,monospace;flex-wrap:wrap'>
            <span>HR {row.get('heart_rate','?')}</span>
            <span>BP {row.get('systolic_bp','?')}/{row.get('diastolic_bp','?')}</span>
            <span>SpO₂ {row.get('spo2','?')}%</span>
            <span>Temp {row.get('temperature','?')}°C</span>
            <span>RR {row.get('respiratory_rate','?')}</span>
            <span>GCS {row.get('gcs','?')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MIMIC SETUP GUIDE
# ─────────────────────────────────────────────
st.markdown("#### MIMIC-III/IV Setup Guide")
with st.expander("How to connect real MIMIC data", expanded=False):
    st.markdown("""
    **Step 1: Apply for PhysioNet access**
    - Go to [physionet.org](https://physionet.org/register/)
    - Complete the CITI Program ethics training (~2 hours)
    - Submit credentialing request — typically approved in 1–3 weeks

    **Step 2: Download MIMIC-IV**
    ```bash
    wget -r -N -c -np --user YOUR_USERNAME --ask-password \\
        https://physionet.org/files/mimiciv/2.2/
    ```

    **Step 3: Preprocess discharge notes + vitals**
    ```python
    # Merge notes + chartevents + icustays
    python utils/mimic_preprocess.py \\
        --notes_csv mimic-iv/hosp/discharge.csv \\
        --chartevents_csv mimic-iv/icu/chartevents.csv \\
        --output data/mimic_merged.csv
    ```

    **Step 4: Run training with MIMIC data**
    ```bash
    python train.py \\
        --mimic_notes data/mimic_merged.csv \\
        --epochs 30 \\
        --batch_size 16 \\
        --with_image
    ```

    **Label derivation from MIMIC**

    Map SOFA score + ICD-10 codes to triage classes:
    - SOFA ≥ 6 or ICD: I21/I22 (MI), G45 (stroke), J96 (resp failure) → **Critical**
    - SOFA 3–5 or ICD: J18 (pneumonia), K35 (appendicitis) → **Urgent**
    - SOFA < 3 + non-acute ICD codes → **Non-urgent**
    """)

st.markdown("""
<div style='font-size:11px;color:#4b5563;margin-top:1rem'>
    MIMIC data contains real patient records. Never include it in public repositories.
    Store in a private location and load via environment variable or HuggingFace Hub (private repo).
</div>
""", unsafe_allow_html=True)
