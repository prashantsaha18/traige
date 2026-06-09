"""
Training Script — Multi-Modal Triage Model
------------------------------------------
Usage:
    python train.py --data data/synthetic/cases.csv --epochs 20 --batch_size 16
    python train.py --data data/mimic_merged.csv   --epochs 30 --with_image

Checkpoints saved to: checkpoints/
Best model saved to:  checkpoints/best_model.pt
"""

import os
import sys
import json
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ── local imports ──
sys.path.insert(0, str(Path(__file__).parent))
from models.triage_model import MultiModalTriageModel, AsymmetricTriageLoss, calibrate_temperature
from utils.dataset import (
    load_synthetic_csv, load_mimic_data, build_dataloaders,
    LABEL_MAP, LABEL_MAP_INV
)

try:
    from sklearn.metrics import (
        f1_score, roc_auc_score, classification_report,
        cohen_kappa_score, confusion_matrix
    )
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("Warning: scikit-learn not found. Metrics will be limited.")


# ─────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────

def move_to_device(batch: dict, device: str) -> dict:
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, dict):
            result[k] = {kk: vv.to(device) for kk, vv in v.items()
                         if isinstance(vv, torch.Tensor)}
        else:
            result[k] = v
    return result


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits, probs, _, _ = model(
                    batch["text_inputs"],
                    batch["vitals"],
                    batch.get("image"),
                )
                loss = criterion(logits, batch["label"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, probs, _, _ = model(
                batch["text_inputs"],
                batch["vitals"],
                batch.get("image"),
            )
            loss = criterion(logits, batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        preds = probs.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["label"].cpu().numpy().tolist())

        if (step + 1) % 20 == 0:
            print(f"  step {step+1}/{len(loader)} | loss {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    if SKLEARN_OK:
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    else:
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        macro_f1 = correct / len(all_preds)

    return avg_loss, macro_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        batch  = move_to_device(batch, device)
        logits, probs, _, _ = model(
            batch["text_inputs"],
            batch["vitals"],
            batch.get("image"),
        )
        loss = criterion(logits, batch["label"])
        total_loss += loss.item()

        all_preds.extend(probs.argmax(-1).cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(batch["label"].cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics


def compute_metrics(labels, preds, probs=None):
    if not SKLEARN_OK:
        correct = sum(p == l for p, l in zip(preds, labels))
        return {"accuracy": correct / len(preds), "macro_f1": correct / len(preds)}

    m = {}
    m["macro_f1"]   = f1_score(labels, preds, average="macro",    zero_division=0)
    m["weighted_f1"]= f1_score(labels, preds, average="weighted", zero_division=0)
    m["kappa"]      = cohen_kappa_score(labels, preds)

    # Per-class
    f1s = f1_score(labels, preds, average=None, zero_division=0)
    for i, cls in LABEL_MAP_INV.items():
        m[f"f1_{cls.lower().replace('-','_')}"] = float(f1s[i]) if i < len(f1s) else 0.0

    # AUC-ROC (one-vs-rest)
    if probs is not None:
        probs_arr = np.array(probs)
        try:
            m["auc_macro"] = roc_auc_score(
                labels, probs_arr, multi_class="ovr", average="macro"
            )
            for i, cls in LABEL_MAP_INV.items():
                binary = [1 if l == i else 0 for l in labels]
                m[f"auc_{cls.lower().replace('-','_')}"] = roc_auc_score(binary, probs_arr[:, i])
        except ValueError:
            pass

    return m


def print_metrics(metrics: dict, prefix: str = ""):
    lines = [f"  {prefix}macro_F1={metrics.get('macro_f1',0):.4f}",
             f"  AUC={metrics.get('auc_macro',0):.4f}",
             f"  kappa={metrics.get('kappa',0):.4f}",
             f"  F1[Critical]={metrics.get('f1_critical',0):.4f}",
             f"  F1[Urgent]={metrics.get('f1_urgent',0):.4f}",
             f"  F1[Non-urgent]={metrics.get('f1_non_urgent',0):.4f}"]
    print(" | ".join(lines))


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train(args):
    # ── Device ──
    device = (
        "cuda"  if torch.cuda.is_available() else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    # ── Data ──
    if args.mimic_notes and args.mimic_vitals:
        df = load_mimic_data(args.mimic_notes, args.mimic_vitals)
    else:
        df = load_synthetic_csv(args.data)

    train_loader, val_loader, test_loader, normaliser = build_dataloaders(
        df,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        max_len=args.max_len,
        val_split=0.15,
        test_split=0.10,
        num_workers=args.num_workers,
    )

    # Save normaliser for inference
    os.makedirs("checkpoints", exist_ok=True)
    normaliser.save("checkpoints/normaliser.json")

    # ── Model ──
    model = MultiModalTriageModel(
        text_model=args.tokenizer,
        with_image=args.with_image,
        modality_drop_p=0.15,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Loss ──
    criterion = AsymmetricTriageLoss(
        class_weights=(5.0, 2.0, 1.0),
        gamma=2.0,
    )

    # ── Optimiser — differential learning rates ──
    no_decay = ["bias", "LayerNorm", "layer_norm"]
    bert_params = [(n, p) for n, p in model.named_parameters()
                   if "text_enc.bert" in n and p.requires_grad]
    other_params= [(n, p) for n, p in model.named_parameters()
                   if "text_enc.bert" not in n and p.requires_grad]

    param_groups = [
        {"params": [p for n, p in bert_params  if not any(nd in n for nd in no_decay)], "lr": args.bert_lr,  "weight_decay": 0.01},
        {"params": [p for n, p in bert_params  if     any(nd in n for nd in no_decay)], "lr": args.bert_lr,  "weight_decay": 0.0},
        {"params": [p for n, p in other_params if not any(nd in n for nd in no_decay)], "lr": args.lr,       "weight_decay": 0.01},
        {"params": [p for n, p in other_params if     any(nd in n for nd in no_decay)], "lr": args.lr,       "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups)

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.epochs // 3), T_mult=1, eta_min=1e-6
    )

    # AMP scaler (CUDA only)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # ── Training loop ──
    best_val_f1 = 0.0
    history     = []

    print(f"\n{'─'*60}")
    print(f"Training for {args.epochs} epochs")
    print(f"{'─'*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  train loss={train_loss:.4f}  F1={train_f1:.4f}  |  "
              f"val loss={val_loss:.4f}  elapsed={elapsed:.0f}s")
        print_metrics(val_metrics, "val ")

        val_f1 = val_metrics.get("macro_f1", 0.0)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_f1": val_f1, **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "args":        vars(args),
            }, "checkpoints/best_model.pt")
            print(f"  ✓ New best! Saved to checkpoints/best_model.pt")

        # Periodic checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/epoch_{epoch:03d}.pt")

        print()

    # ── Test evaluation ──
    print("─" * 60)
    print("Test set evaluation")
    print("─" * 60)

    ckpt = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    # Temperature calibration
    calibrate_temperature(model, val_loader, device=device)

    _, test_metrics = evaluate(model, test_loader, criterion, device)
    print_metrics(test_metrics, "test ")

    if SKLEARN_OK:
        # Full classification report
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = move_to_device(batch, device)
                _, probs, _, _ = model(
                    batch["text_inputs"], batch["vitals"], batch.get("image")
                )
                all_preds.extend(probs.argmax(-1).cpu().numpy().tolist())
                all_labels.extend(batch["label"].cpu().numpy().tolist())

        print("\nClassification Report:")
        print(classification_report(
            all_labels, all_preds,
            target_names=["Critical", "Urgent", "Non-urgent"]
        ))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

    # Save history
    with open("checkpoints/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\nTraining complete. History saved to checkpoints/training_history.json")

    # Save final calibrated model
    torch.save({
        "epoch":       args.epochs,
        "model_state": model.state_dict(),
        "test_metrics": test_metrics,
        "args":        vars(args),
    }, "checkpoints/final_model.pt")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Modal Triage Model")

    # Data
    parser.add_argument("--data",         type=str, default="data/synthetic/cases.csv")
    parser.add_argument("--mimic_notes",  type=str, default=None)
    parser.add_argument("--mimic_vitals", type=str, default=None)
    parser.add_argument("--with_image",   action="store_true")

    # Model
    parser.add_argument("--tokenizer", type=str,
                        default="dmis-lab/biobert-base-cased-v1.2")
    parser.add_argument("--max_len",   type=int, default=256)

    # Training
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--bert_lr",     type=float, default=2e-5)
    parser.add_argument("--num_workers", type=int,   default=0)

    args = parser.parse_args()
    train(args)
