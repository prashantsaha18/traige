"""
utils/mimic_preprocess.py
--------------------------
Preprocesses raw MIMIC-IV/III files into a single merged CSV
ready for TriageDataset.

Usage:
    python utils/mimic_preprocess.py \
        --discharge  data/mimic-iv/hosp/discharge.csv \
        --chartevents data/mimic-iv/icu/chartevents.csv \
        --icustays   data/mimic-iv/icu/icustays.csv \
        --diagnoses  data/mimic-iv/hosp/diagnoses_icd.csv \
        --output     data/mimic_merged.csv

MIMIC-IV itemids for chartevents vitals:
    220045 = Heart Rate
    220179 = Non Invasive Blood Pressure systolic
    220180 = Non Invasive Blood Pressure diastolic
    220277 = SpO2
    223762 = Temperature Celsius
    220210 = Respiratory Rate
    223900 = GCS - Verbal Response (proxy)
    226531 = GCS Total
    225309 = ART BP Systolic (arterial line)
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# MIMIC-IV Item IDs → our column names
# ─────────────────────────────────────────────
VITAL_ITEMIDS = {
    220045: "heart_rate",
    220179: "systolic_bp",
    225309: "systolic_bp",        # arterial fallback
    220180: "diastolic_bp",
    220277: "spo2",
    223762: "temperature",        # Celsius
    676:    "temperature",        # also Celsius in MIMIC-III
    220210: "respiratory_rate",
    226531: "gcs",
    198:    "gcs",                # MIMIC-III GCS motor
}

# ICD-10/9 prefix → triage severity
# (simplified rule-based mapping — extend with clinical input)
CRITICAL_ICD_PREFIXES = [
    "I21", "I22",   # STEMI
    "I63", "I64",   # Ischaemic / unspecified stroke
    "I60", "I61",   # Haemorrhagic stroke
    "J96",          # Respiratory failure
    "A41",          # Sepsis
    "K72",          # Hepatic failure
    "N17",          # Acute kidney injury
    "E10", "E11",   # DM — only if SOFA ≥ 4
    "I26",          # PE
    "I71",          # Aortic aneurysm / dissection
    "J81",          # Pulmonary oedema
    "R57",          # Shock
    "G93",          # Brain disorders
    "S06",          # Head injury
    "T71",          # Asphyxiation
]

URGENT_ICD_PREFIXES = [
    "J18", "J15",   # Pneumonia
    "K35", "K37",   # Appendicitis
    "K80", "K81",   # Cholelithiasis / cholecystitis
    "N10", "N12",   # Pyelonephritis
    "I10", "I11",   # Hypertension
    "M54",          # Back pain (urgent if severe)
    "G43",          # Migraine
    "R55",          # Syncope
    "J45",          # Asthma
    "K57",          # Diverticulitis
]


def load_chartevents_vitals(chartevents_path: str, icustays_path: str) -> pd.DataFrame:
    """
    Load chartevents and extract first-measurement vitals per ICU stay.
    Returns DataFrame indexed by (subject_id, hadm_id).
    """
    print("Loading icustays…")
    icustays = pd.read_csv(icustays_path, usecols=["subject_id", "hadm_id", "stay_id", "intime"])
    icustays["intime"] = pd.to_datetime(icustays["intime"])

    print("Loading chartevents (this may take a while for large files)…")
    # Read in chunks to handle large files
    chunks = []
    item_ids = list(VITAL_ITEMIDS.keys())

    for chunk in pd.read_csv(
        chartevents_path,
        usecols=["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"],
        dtype={"valuenum": float},
        chunksize=2_000_000,
    ):
        subset = chunk[chunk["itemid"].isin(item_ids)].copy()
        if len(subset) > 0:
            chunks.append(subset)

    if not chunks:
        print("Warning: no matching itemids found in chartevents")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"])
    df["vital"] = df["itemid"].map(VITAL_ITEMIDS)

    # Join with icustays to get hadm_id and sort by time
    df = df.merge(
        icustays[["stay_id", "subject_id", "hadm_id", "intime"]],
        on=["stay_id", "subject_id"],
        how="left",
        suffixes=("", "_icu"),
    )

    # Keep only measurements within first 6 hours of ICU admission
    df = df.dropna(subset=["intime"])
    df["hours_in"] = (df["charttime"] - df["intime"]).dt.total_seconds() / 3600
    df = df[(df["hours_in"] >= 0) & (df["hours_in"] <= 6)]

    # First measurement per vital per admission
    df = df.sort_values("charttime")
    vitals_wide = (
        df.groupby(["subject_id", "hadm_id", "vital"])["valuenum"]
        .first()
        .unstack("vital")
        .reset_index()
    )

    # Sanity clip (remove physiologically impossible values)
    clips = {
        "heart_rate":       (20, 300),
        "systolic_bp":      (40, 300),
        "diastolic_bp":     (20, 200),
        "spo2":             (50, 100),
        "temperature":      (28, 45),
        "respiratory_rate": (4, 80),
        "gcs":              (3, 15),
    }
    for col, (lo, hi) in clips.items():
        if col in vitals_wide.columns:
            vitals_wide[col] = vitals_wide[col].clip(lo, hi)

    print(f"Vitals extracted: {len(vitals_wide)} admissions")
    return vitals_wide


def load_discharge_notes(notes_path: str) -> pd.DataFrame:
    """Load MIMIC-IV discharge notes (or MIMIC-III NOTEEVENTS)."""
    print("Loading discharge notes…")

    if "discharge" in Path(notes_path).name.lower():
        # MIMIC-IV format
        notes = pd.read_csv(
            notes_path,
            usecols=["subject_id", "hadm_id", "text"],
            dtype=str,
        )
    else:
        # MIMIC-III NOTEEVENTS format
        notes = pd.read_csv(
            notes_path,
            usecols=["SUBJECT_ID", "HADM_ID", "TEXT", "CATEGORY"],
            dtype=str,
        )
        notes = notes[notes["CATEGORY"].str.lower().str.contains("discharge", na=False)]
        notes.columns = ["subject_id", "hadm_id", "text", "_cat"]
        notes = notes.drop(columns=["_cat"])

    # Extract chief complaint section
    notes["chief_complaint"] = notes["text"].apply(_extract_chief_complaint)

    notes = notes.dropna(subset=["chief_complaint"])
    notes = notes[notes["chief_complaint"].str.len() > 20]
    print(f"Discharge notes loaded: {len(notes)} admissions")
    return notes[["subject_id", "hadm_id", "chief_complaint"]]


def _extract_chief_complaint(text: str) -> str:
    """
    Pull chief complaint / HPI from MIMIC discharge text.
    Tries multiple section headers common in MIMIC notes.
    """
    if not isinstance(text, str):
        return ""

    patterns = [
        r"chief complaint[:\s]+(.+?)(?:\n\n|\nhistory of present illness|\nHPI)",
        r"reason for admission[:\s]+(.+?)(?:\n\n|\n[A-Z])",
        r"history of present illness[:\s]+(.+?)(?:\n\n|\npast medical)",
        r"HPI[:\s]+(.+?)(?:\n\n|\n[A-Z])",
        r"presenting complaint[:\s]+(.+?)(?:\n\n|\n[A-Z])",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            snippet = m.group(1).strip()
            # Clean up
            snippet = re.sub(r"\s+", " ", snippet)
            snippet = snippet[:512]  # truncate to 512 chars
            if len(snippet) > 20:
                return snippet

    # Fallback: first 300 chars of note body
    return text.strip()[:300]


def derive_triage_labels(
    diagnoses_path: str,
    hadm_ids: pd.Series,
) -> pd.DataFrame:
    """
    Derive Critical / Urgent / Non-urgent labels from ICD codes.
    Returns DataFrame with (subject_id, hadm_id, label, severity).
    """
    print("Loading diagnoses…")
    diag = pd.read_csv(
        diagnoses_path,
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype=str,
    )
    diag = diag[diag["hadm_id"].isin(hadm_ids.astype(str))]

    def classify_hadm(codes):
        codes = [str(c).upper() for c in codes]
        for code in codes:
            for prefix in CRITICAL_ICD_PREFIXES:
                if code.startswith(prefix):
                    return "Critical"
        for code in codes:
            for prefix in URGENT_ICD_PREFIXES:
                if code.startswith(prefix):
                    return "Urgent"
        return "Non-urgent"

    labels = (
        diag.groupby(["subject_id", "hadm_id"])["icd_code"]
        .apply(list)
        .reset_index()
    )
    labels["label"] = labels["icd_code"].apply(classify_hadm)
    labels = labels.drop(columns=["icd_code"])
    print(labels["label"].value_counts().to_string())
    return labels


def load_patient_demographics(hosp_path: str) -> pd.DataFrame:
    """Load age and gender from MIMIC patients table."""
    patients_path = os.path.join(os.path.dirname(hosp_path), "patients.csv")
    if not os.path.exists(patients_path):
        return pd.DataFrame()

    patients = pd.read_csv(
        patients_path,
        usecols=["subject_id", "gender", "anchor_age"],
        dtype={"anchor_age": float},
    )
    patients = patients.rename(columns={"anchor_age": "age"})
    return patients


def preprocess_mimic(
    discharge_path: str,
    chartevents_path: str,
    icustays_path: str,
    diagnoses_path: str,
    output_path: str,
    max_rows: int = None,
):
    # Load components
    notes   = load_discharge_notes(discharge_path)
    vitals  = load_chartevents_vitals(chartevents_path, icustays_path)
    labels  = derive_triage_labels(diagnoses_path, notes["hadm_id"])

    # Convert IDs to same type for merge
    for df in [notes, vitals, labels]:
        for col in ["subject_id", "hadm_id"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

    # Merge
    merged = notes.merge(vitals,  on=["subject_id", "hadm_id"], how="inner")
    merged = merged.merge(labels, on=["subject_id", "hadm_id"], how="inner")

    # Try to add demographics
    try:
        demo = load_patient_demographics(discharge_path)
        if not demo.empty:
            demo["subject_id"] = demo["subject_id"].astype(str)
            merged = merged.merge(demo[["subject_id", "age", "gender"]], 
                                  on="subject_id", how="left")
    except Exception as e:
        print(f"Demographics not loaded: {e}")
        merged["age"] = 55  # median fallback

    # Deduplicate (keep first admission per patient)
    merged = merged.drop_duplicates(subset=["subject_id"], keep="first")

    if max_rows:
        merged = merged.head(max_rows)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\nSaved {len(merged)} merged records → {output_path}")
    print("\nColumn summary:")
    print(merged.dtypes.to_string())
    print("\nLabel distribution:")
    print(merged["label"].value_counts().to_string())
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-IV for DeepTriage")
    parser.add_argument("--discharge",    required=True,  help="Path to discharge.csv")
    parser.add_argument("--chartevents",  required=True,  help="Path to chartevents.csv")
    parser.add_argument("--icustays",     required=True,  help="Path to icustays.csv")
    parser.add_argument("--diagnoses",    required=True,  help="Path to diagnoses_icd.csv")
    parser.add_argument("--output",       default="data/mimic_merged.csv")
    parser.add_argument("--max_rows",     type=int, default=None)
    args = parser.parse_args()

    preprocess_mimic(
        discharge_path=args.discharge,
        chartevents_path=args.chartevents,
        icustays_path=args.icustays,
        diagnoses_path=args.diagnoses,
        output_path=args.output,
        max_rows=args.max_rows,
    )
