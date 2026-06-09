"""
Dataset & Preprocessing Pipeline
----------------------------------
Handles:
  - MIMIC-III / MIMIC-IV structured data
  - Synthetic generated cases (CSV)
  - Tokenization for BioBERT
  - Vitals normalisation (z-score, missing value imputation)
  - Optional image loading (CheXpert / MIMIC-CXR)
  - PyTorch Dataset + DataLoader construction
"""

import os
import json
import math
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

VITALS_COLS = [
    "heart_rate", "systolic_bp", "diastolic_bp", "spo2",
    "temperature", "respiratory_rate", "gcs",
    "age", "wbc", "creatinine", "glucose", "hemoglobin",
]

# Population-level reference ranges for z-score normalisation
VITALS_STATS = {
    "heart_rate":        {"mean": 85.0,  "std": 18.0},
    "systolic_bp":       {"mean": 122.0, "std": 22.0},
    "diastolic_bp":      {"mean": 78.0,  "std": 14.0},
    "spo2":              {"mean": 96.5,  "std": 3.5},
    "temperature":       {"mean": 37.2,  "std": 0.8},
    "respiratory_rate":  {"mean": 18.0,  "std": 5.0},
    "gcs":               {"mean": 14.2,  "std": 1.8},
    "age":               {"mean": 48.0,  "std": 18.0},
    "wbc":               {"mean": 10.5,  "std": 5.0},
    "creatinine":        {"mean": 1.1,   "std": 0.6},
    "glucose":           {"mean": 130.0, "std": 60.0},
    "hemoglobin":        {"mean": 12.5,  "std": 2.2},
}

LABEL_MAP     = {"Critical": 0, "Urgent": 1, "Non-urgent": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


# ─────────────────────────────────────────────
# VITALS NORMALISER
# ─────────────────────────────────────────────

class VitalsNormaliser:
    """
    Z-score normalisation with per-column stats.
    NaN values (missing labs) are imputed with 0 (= population mean after scaling).
    """

    def __init__(self, stats: dict = None):
        self.stats = stats or VITALS_STATS

    def fit(self, df: pd.DataFrame) -> "VitalsNormaliser":
        """Recompute stats from a training DataFrame (override defaults)."""
        for col in VITALS_COLS:
            if col in df.columns:
                self.stats[col] = {
                    "mean": float(df[col].mean()),
                    "std":  max(float(df[col].std()), 1e-6),
                }
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        result = np.zeros((len(df), len(VITALS_COLS)), dtype=np.float32)
        for i, col in enumerate(VITALS_COLS):
            if col not in df.columns:
                continue
            vals = df[col].values.astype(float)
            mean = self.stats[col]["mean"]
            std  = self.stats[col]["std"]
            normed = (vals - mean) / std
            # NaN → 0 (population mean)
            normed = np.where(np.isnan(normed), 0.0, normed)
            result[:, i] = normed
        return result

    def transform_single(self, row: dict) -> np.ndarray:
        result = np.zeros(len(VITALS_COLS), dtype=np.float32)
        for i, col in enumerate(VITALS_COLS):
            val = row.get(col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                result[i] = 0.0
            else:
                mean = self.stats[col]["mean"]
                std  = self.stats[col]["std"]
                result[i] = (float(val) - mean) / std
        return result

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "VitalsNormaliser":
        with open(path) as f:
            stats = json.load(f)
        return cls(stats)


# ─────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────

class TriageDataset(Dataset):
    """
    Multi-modal triage dataset.

    Parameters
    ----------
    df           : DataFrame with chief_complaint, vitals cols, label, and
                   optionally xray_path
    tokenizer    : HuggingFace tokenizer (BioBERT)
    normaliser   : VitalsNormaliser instance
    max_len      : max token length for text
    image_dir    : root directory for X-ray images (optional)
    image_size   : resize target for EfficientNet (300)
    augment      : apply text/image augmentation (training only)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        normaliser: VitalsNormaliser,
        max_len: int = 256,
        image_dir: Optional[str] = None,
        image_size: int = 300,
        augment: bool = False,
    ):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.normaliser = normaliser
        self.max_len    = max_len
        self.image_dir  = image_dir
        self.image_size = image_size
        self.augment    = augment

        # Precompute normalised vitals matrix
        self.vitals_matrix = normaliser.transform(df)

        # Image transform
        self._image_transform = self._build_image_transform()

    def _build_image_transform(self):
        try:
            from torchvision import transforms
            if self.augment:
                return transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.RandomRotation(5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ])
            else:
                return transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ])
        except ImportError:
            return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row   = self.df.iloc[idx]
        vitals = torch.tensor(self.vitals_matrix[idx], dtype=torch.float32)

        # ── Text tokenisation ──
        text = str(row.get("chief_complaint", ""))
        if self.augment:
            text = self._augment_text(text)

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {k: v.squeeze(0) for k, v in enc.items()}

        # ── Label ──
        label_raw = row.get("label", row.get("triage_label", "Non-urgent"))
        if isinstance(label_raw, str):
            label = LABEL_MAP.get(label_raw, 2)
        else:
            label = int(label_raw)

        item = {
            "text_inputs": text_inputs,
            "vitals":      vitals,
            "label":       torch.tensor(label, dtype=torch.long),
            "condition":   str(row.get("condition", "")),
            "text_raw":    text,
        }

        # ── Image (optional) ──
        xray_path = row.get("xray_path", None)
        if xray_path and self.image_dir and self._image_transform:
            full_path = os.path.join(self.image_dir, str(xray_path))
            if os.path.exists(full_path):
                try:
                    from PIL import Image
                    img = Image.open(full_path).convert("RGB")
                    item["image"] = self._image_transform(img)
                except Exception:
                    item["image"] = None
            else:
                item["image"] = None
        else:
            item["image"] = None

        return item

    @staticmethod
    def _augment_text(text: str) -> str:
        """Lightweight text augmentation: random synonym swap for common terms."""
        swaps = {
            "chest pain": ["chest discomfort", "chest tightness", "chest pressure"],
            "shortness of breath": ["dyspnoea", "breathlessness", "difficulty breathing"],
            "fever": ["pyrexia", "high temperature", "febrile"],
            "nausea": ["feeling sick", "nausea and retching"],
            "vomiting": ["emesis", "throwing up"],
            "headache": ["cephalgia", "head pain"],
        }
        import random
        for term, alternatives in swaps.items():
            if term in text.lower() and random.random() < 0.3:
                replacement = random.choice(alternatives)
                text = text.replace(term, replacement, 1)
        return text


def collate_fn(batch: list) -> dict:
    """Custom collate: handles optional image (None → skip)."""
    text_input_ids      = torch.stack([b["text_inputs"]["input_ids"]      for b in batch])
    text_attention_mask = torch.stack([b["text_inputs"]["attention_mask"] for b in batch])
    vitals  = torch.stack([b["vitals"] for b in batch])
    labels  = torch.stack([b["label"]  for b in batch])

    text_token_type_ids = None
    if "token_type_ids" in batch[0]["text_inputs"]:
        text_token_type_ids = torch.stack(
            [b["text_inputs"]["token_type_ids"] for b in batch]
        )

    text_inputs = {
        "input_ids":      text_input_ids,
        "attention_mask": text_attention_mask,
    }
    if text_token_type_ids is not None:
        text_inputs["token_type_ids"] = text_token_type_ids

    images = None
    if any(b["image"] is not None for b in batch):
        imgs = []
        for b in batch:
            imgs.append(b["image"] if b["image"] is not None
                        else torch.zeros(3, 300, 300))
        images = torch.stack(imgs)

    return {
        "text_inputs": text_inputs,
        "vitals":      vitals,
        "image":       images,
        "label":       labels,
        "condition":   [b["condition"] for b in batch],
        "text_raw":    [b["text_raw"]  for b in batch],
    }


# ─────────────────────────────────────────────
# DATA LOADING HELPERS
# ─────────────────────────────────────────────

def load_synthetic_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} synthetic cases from {path}")
    print(df["label"].value_counts().to_string())
    return df


def load_mimic_data(notes_path: str, vitals_path: str) -> pd.DataFrame:
    """
    Load and join MIMIC discharge notes with vitals.
    Expects:
      notes_path  : CSV with columns [subject_id, hadm_id, text]
      vitals_path : CSV with columns [subject_id, hadm_id, heart_rate, ...]
    Returns merged DataFrame ready for TriageDataset.
    """
    notes  = pd.read_csv(notes_path, usecols=["subject_id", "hadm_id", "text"])
    vitals = pd.read_csv(vitals_path)
    df = notes.merge(vitals, on=["subject_id", "hadm_id"], how="inner")
    df = df.rename(columns={"text": "chief_complaint"})

    # Map ICD/SOFA-derived severity to triage labels
    # (user must supply a 'severity' column from their MIMIC preprocessing)
    if "severity" in df.columns:
        sev_map = {"high": "Critical", "moderate": "Urgent", "low": "Non-urgent"}
        df["label"] = df["severity"].map(sev_map)
    else:
        warnings.warn("No 'severity' column found — defaulting all to 'Non-urgent'")
        df["label"] = "Non-urgent"

    print(f"MIMIC data: {len(df)} admissions")
    return df


def build_dataloaders(
    df: pd.DataFrame,
    tokenizer_name: str = "dmis-lab/biobert-base-cased-v1.2",
    val_split: float = 0.15,
    test_split: float = 0.10,
    batch_size: int = 16,
    max_len: int = 256,
    image_dir: Optional[str] = None,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, VitalsNormaliser]:
    """
    Split → normalise → tokenise → DataLoader.
    Returns (train_loader, val_loader, test_loader, normaliser)
    """
    from sklearn.model_selection import train_test_split

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Stratified splits
    train_val, test_df = train_test_split(
        df, test_size=test_split, stratify=df["label"], random_state=seed
    )
    val_frac = val_split / (1 - test_split)
    train_df, val_df = train_test_split(
        train_val, test_size=val_frac, stratify=train_val["label"], random_state=seed
    )

    # Fit normaliser on training set only
    normaliser = VitalsNormaliser()
    normaliser.fit(train_df)

    # Datasets
    train_ds = TriageDataset(train_df, tokenizer, normaliser,
                             max_len, image_dir, augment=True)
    val_ds   = TriageDataset(val_df,   tokenizer, normaliser,
                             max_len, image_dir, augment=False)
    test_ds  = TriageDataset(test_df,  tokenizer, normaliser,
                             max_len, image_dir, augment=False)

    # Weighted sampler for class imbalance (training only)
    label_counts = train_df["label"].map(LABEL_MAP).value_counts().sort_index()
    weights_per_class = 1.0 / label_counts.values.astype(float)
    sample_weights = [
        weights_per_class[LABEL_MAP[lab]]
        for lab in train_df["label"].tolist()
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_ds), replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, normaliser
