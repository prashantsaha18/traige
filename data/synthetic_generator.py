"""
Synthetic Patient Case Generator
---------------------------------
Generates realistic patient cases for demo, testing, and training augmentation
for rare conditions when MIMIC data is not yet available.

Usage:
    python data/synthetic_generator.py --n 500 --output data/synthetic/cases.csv
"""

import random
import json
import csv
import argparse
import math
from dataclasses import dataclass, asdict, field
from typing import Optional


# ─────────────────────────────────────────────
# CLINICAL TEMPLATES
# ─────────────────────────────────────────────

CRITICAL_CASES = [
    {
        "condition": "STEMI",
        "symptoms": [
            "Severe crushing chest pain radiating to the left arm and jaw, onset 45 minutes ago. Profuse diaphoresis. Nausea and vomiting.",
            "Sudden onset of 9/10 substernal chest pressure with diaphoresis. History of hypertension and diabetes. Looks pale and anxious.",
            "Acute chest tightness radiating to left shoulder. Patient is diaphoretic and short of breath. ECG shows ST elevation in V1-V4.",
        ],
        "vitals_range": {
            "hr": (95, 130), "sbp": (80, 100), "dbp": (50, 70),
            "spo2": (88, 95), "temp": (36.5, 37.2), "rr": (20, 28),
            "gcs": (14, 15), "wbc": (10, 15), "creatinine": (0.9, 1.4),
            "glucose": (140, 220), "hgb": (11, 14),
        },
    },
    {
        "condition": "Septic Shock",
        "symptoms": [
            "Altered mental status, high fever, and hypotension. Came from nursing home. Urine appears dark and cloudy. Rigors.",
            "Confusion, chills, and vomiting for 18 hours. Fever of 39.8°C at home. Septic appearance, mottled extremities.",
            "Unresponsive to verbal commands. Tachycardia and hypotension. Wife reports 2 days of fever and cough.",
        ],
        "vitals_range": {
            "hr": (120, 145), "sbp": (70, 90), "dbp": (40, 60),
            "spo2": (90, 96), "temp": (38.8, 40.5), "rr": (24, 34),
            "gcs": (8, 13), "wbc": (18, 28), "creatinine": (1.8, 3.5),
            "glucose": (160, 280), "hgb": (9, 12),
        },
    },
    {
        "condition": "Acute Stroke",
        "symptoms": [
            "Sudden left-sided facial droop and arm weakness. Slurred speech. Onset 1 hour ago. No loss of consciousness.",
            "Found by family with right arm paralysis and aphasia. Last known well 2 hours ago. History of atrial fibrillation.",
            "Acute onset of inability to speak and right hemiplegia. Hypertensive at baseline. FAST positive.",
        ],
        "vitals_range": {
            "hr": (80, 110), "sbp": (170, 210), "dbp": (95, 120),
            "spo2": (93, 98), "temp": (36.8, 37.5), "rr": (16, 22),
            "gcs": (9, 14), "wbc": (8, 14), "creatinine": (0.8, 1.3),
            "glucose": (110, 190), "hgb": (11, 15),
        },
    },
    {
        "condition": "Pulmonary Embolism",
        "symptoms": [
            "Acute onset pleuritic chest pain and haemoptysis. Recent long-haul flight. Tachycardic and hypoxic.",
            "Sudden dyspnoea and right-sided chest pain. Left leg swollen for 3 days. SpO₂ 88% on room air.",
            "Syncope followed by chest tightness and shortness of breath. DVT risk factors including oral contraceptive use.",
        ],
        "vitals_range": {
            "hr": (105, 130), "sbp": (85, 105), "dbp": (55, 75),
            "spo2": (85, 93), "temp": (36.5, 37.8), "rr": (22, 32),
            "gcs": (13, 15), "wbc": (9, 14), "creatinine": (0.8, 1.2),
            "glucose": (100, 150), "hgb": (11, 14),
        },
    },
    {
        "condition": "Diabetic Ketoacidosis",
        "symptoms": [
            "Type 1 diabetic, 3 days of polyuria and polydipsia. Nausea, vomiting. Kussmaul breathing. Fruity odour on breath.",
            "Known T1DM presenting with vomiting, abdominal pain, and confusion. Glucose 480 mg/dL at home.",
            "Missed insulin doses for 2 days. Presents with deep rapid breathing, altered mentation, and severe dehydration.",
        ],
        "vitals_range": {
            "hr": (110, 135), "sbp": (85, 110), "dbp": (55, 70),
            "spo2": (94, 99), "temp": (36.2, 38.0), "rr": (22, 36),
            "gcs": (10, 14), "wbc": (12, 20), "creatinine": (1.5, 3.0),
            "glucose": (400, 600), "hgb": (12, 16),
        },
    },
]

URGENT_CASES = [
    {
        "condition": "Pneumonia",
        "symptoms": [
            "Productive cough with yellow sputum for 4 days, fever 38.5°C, right-sided pleuritic chest pain.",
            "Worsening dyspnoea and fever for 3 days. SpO₂ 94% on room air. Decreased breath sounds at right base.",
            "Elderly patient with confusion and productive cough. Chest X-ray shows right lower lobe consolidation.",
        ],
        "vitals_range": {
            "hr": (90, 110), "sbp": (110, 135), "dbp": (70, 85),
            "spo2": (92, 96), "temp": (38.2, 39.5), "rr": (18, 26),
            "gcs": (14, 15), "wbc": (14, 22), "creatinine": (0.8, 1.3),
            "glucose": (100, 180), "hgb": (10, 14),
        },
    },
    {
        "condition": "Acute Appendicitis",
        "symptoms": [
            "Periumbilical pain migrating to RLQ over 12 hours. Nausea, anorexia, fever. Rebound tenderness at McBurney's point.",
            "Young male with 10/10 right lower abdominal pain, vomiting, and fever for 8 hours. Unable to walk straight.",
            "Right lower quadrant pain for 16 hours, worsening with movement. Low-grade fever, elevated WBC.",
        ],
        "vitals_range": {
            "hr": (88, 108), "sbp": (115, 135), "dbp": (72, 88),
            "spo2": (97, 100), "temp": (37.8, 39.2), "rr": (16, 22),
            "gcs": (15, 15), "wbc": (14, 22), "creatinine": (0.7, 1.1),
            "glucose": (90, 130), "hgb": (11, 15),
        },
    },
    {
        "condition": "Hypertensive Urgency",
        "symptoms": [
            "Severe headache and blurred vision. Known hypertensive, stopped medications 2 weeks ago. BP 195/115.",
            "Pounding headache, nausea, and epistaxis. History of poorly controlled hypertension.",
            "Headache and dizziness with BP reading of 210/120 at pharmacy. No chest pain or neurological deficits.",
        ],
        "vitals_range": {
            "hr": (80, 100), "sbp": (185, 220), "dbp": (110, 130),
            "spo2": (96, 99), "temp": (36.5, 37.2), "rr": (16, 20),
            "gcs": (15, 15), "wbc": (7, 11), "creatinine": (1.1, 1.8),
            "glucose": (100, 160), "hgb": (11, 14),
        },
    },
]

NON_URGENT_CASES = [
    {
        "condition": "URTI / Common Cold",
        "symptoms": [
            "Runny nose, sore throat, mild dry cough for 2 days. No fever. Well appearing, good oral intake.",
            "Congestion, sneezing, and mild headache for 3 days. Temperature 37.1°C. No shortness of breath.",
            "Sore throat and mild fatigue for 24 hours. No dysphagia or drooling. Tolerating fluids.",
        ],
        "vitals_range": {
            "hr": (68, 85), "sbp": (110, 130), "dbp": (70, 82),
            "spo2": (97, 100), "temp": (36.8, 37.4), "rr": (14, 18),
            "gcs": (15, 15), "wbc": (7, 11), "creatinine": (0.7, 1.0),
            "glucose": (85, 110), "hgb": (12, 15),
        },
    },
    {
        "condition": "Ankle Sprain",
        "symptoms": [
            "Twisted ankle while playing cricket 2 hours ago. Mild swelling, no deformity. Able to weight-bear.",
            "Right ankle pain after stepping off a curb. No distal neurovascular deficit. Normal sensation.",
            "Lateral ankle sprain from fall. Minimal bruising, can hobble. Ottawa criteria negative for fracture.",
        ],
        "vitals_range": {
            "hr": (65, 80), "sbp": (112, 128), "dbp": (68, 80),
            "spo2": (98, 100), "temp": (36.5, 37.0), "rr": (14, 17),
            "gcs": (15, 15), "wbc": (6, 9), "creatinine": (0.7, 1.0),
            "glucose": (80, 105), "hgb": (12, 16),
        },
    },
    {
        "condition": "Constipation",
        "symptoms": [
            "No bowel movement for 4 days, mild lower abdominal cramps. Soft abdomen. No red flags.",
            "Bloating and discomfort. Last bowel movement 5 days ago. On opioids for back pain.",
            "Straining at stool, hard stools for 1 week. No blood. Well-appearing.",
        ],
        "vitals_range": {
            "hr": (65, 78), "sbp": (110, 125), "dbp": (68, 78),
            "spo2": (98, 100), "temp": (36.5, 37.0), "rr": (14, 16),
            "gcs": (15, 15), "wbc": (6, 9), "creatinine": (0.7, 1.0),
            "glucose": (80, 105), "hgb": (12, 15),
        },
    },
]

LABEL_MAP = {"Critical": 0, "Urgent": 1, "Non-urgent": 2}
AGE_RANGES = {
    "Critical": (35, 80),
    "Urgent":   (18, 75),
    "Non-urgent": (5, 65),
}


def rand_float(lo, hi, decimals=1):
    v = lo + random.random() * (hi - lo)
    return round(v, decimals)


def generate_case(category: str, template: dict, age: int) -> dict:
    r = template["vitals_range"]

    def rv(k):
        return rand_float(*r[k])

    # Add realistic noise / missingness
    creatinine = rv("creatinine") if random.random() > 0.05 else None   # 5% missing
    wbc        = rv("wbc")        if random.random() > 0.08 else None
    hgb        = rv("hgb")        if random.random() > 0.08 else None
    glucose    = rv("glucose")    if random.random() > 0.06 else None

    return {
        "age":             age,
        "gender":          random.choice(["M", "F"]),
        "chief_complaint": random.choice(template["symptoms"]),
        "condition":       template["condition"],
        "label":           category,
        "label_id":        LABEL_MAP[category],
        "heart_rate":      rv("hr"),
        "systolic_bp":     rv("sbp"),
        "diastolic_bp":    rv("dbp"),
        "spo2":            rv("spo2"),
        "temperature":     rv("temp"),
        "respiratory_rate": rv("rr"),
        "gcs":             random.randint(*[int(x) for x in r["gcs"]]),
        "wbc":             wbc,
        "creatinine":      creatinine,
        "glucose":         glucose,
        "hemoglobin":      hgb,
        "has_xray":        random.random() < 0.35,   # 35% of cases have X-ray
    }


def generate_dataset(n: int = 1000, seed: int = 42) -> list:
    random.seed(seed)
    cases = []

    # Class distribution: 25% critical, 35% urgent, 40% non-urgent
    n_critical   = int(n * 0.25)
    n_urgent     = int(n * 0.35)
    n_non_urgent = n - n_critical - n_urgent

    for _ in range(n_critical):
        tmpl = random.choice(CRITICAL_CASES)
        age  = random.randint(*AGE_RANGES["Critical"])
        cases.append(generate_case("Critical", tmpl, age))

    for _ in range(n_urgent):
        tmpl = random.choice(URGENT_CASES)
        age  = random.randint(*AGE_RANGES["Urgent"])
        cases.append(generate_case("Urgent", tmpl, age))

    for _ in range(n_non_urgent):
        tmpl = random.choice(NON_URGENT_CASES)
        age  = random.randint(*AGE_RANGES["Non-urgent"])
        cases.append(generate_case("Non-urgent", tmpl, age))

    random.shuffle(cases)
    return cases


def save_csv(cases: list, path: str):
    if not cases:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cases[0].keys())
        writer.writeheader()
        writer.writerows(cases)
    print(f"Saved {len(cases)} cases → {path}")


def save_json(cases: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(cases)} cases → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=1000)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output", type=str, default="data/synthetic/cases.csv")
    parser.add_argument("--json",   action="store_true")
    args = parser.parse_args()

    cases = generate_dataset(args.n, args.seed)

    if args.json:
        save_json(cases, args.output.replace(".csv", ".json"))
    else:
        save_csv(cases, args.output)

    # Print class distribution
    from collections import Counter
    counts = Counter(c["label"] for c in cases)
    for label, count in sorted(counts.items()):
        print(f"  {label:12s}: {count:5d}  ({100*count/len(cases):.1f}%)")
