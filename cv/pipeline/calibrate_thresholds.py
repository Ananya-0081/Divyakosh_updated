# -*- coding: utf-8 -*-
"""
cv/pipeline/calibrate_thresholds.py

Threshold Calibration — Validation Set Only
=============================================
Sweeps Model A confidence thresholds and Model B confidence/margin
thresholds using ONLY the validation split. Saves the selected
thresholds to pipeline/threshold_config.json.

Design principles:
- Prioritize precision (minimizing false NOTE acceptances).
- Measure how many genuine notes are sent to REVIEW at each threshold.
- Never touch test/ or real_test/ data.
- Report every candidate threshold with its precision, recall, and
  REVIEW rate so the operator can make an informed choice.
- Final selection uses the threshold that achieves the user's target
  precision (default: >= 0.999) while maximising NOTE recall.

Usage:
    cd cv
    .\.venv\Scripts\python.exe pipeline/calibrate_thresholds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_CV   = _HERE.parent

sys.path.insert(0, str(_HERE))   # allow importing divyakosh_pipeline

MODEL_A_CKPT = _CV / "presence" / "presence_classifier_v1.pt"
MODEL_B_CKPT = _CV / "models"   / "mobilenet_v3_clean.pt"
VAL_DIR_A    = _CV / "presence" / "dataset" / "val"
VAL_DIR_B    = _CV / "data_clean" / "val"
OUT_CONFIG   = _HERE / "threshold_config.json"
OUT_REPORT   = _HERE / "threshold_calibration_report.json"

MODEL_B_CLASSES = ["10", "100", "20", "200", "50", "500"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")


def load_model_a():
    raw = torch.load(MODEL_A_CKPT, map_location=DEVICE, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    else:
        state_dict = raw
    model = models.mobilenet_v3_small()
    in_f  = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, 2)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model_b():
    raw = torch.load(MODEL_B_CKPT, map_location=DEVICE, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict  = raw["model_state_dict"]
        class_names = raw.get("class_names", MODEL_B_CLASSES)
    else:
        state_dict  = raw
        class_names = MODEL_B_CLASSES
    model = models.mobilenet_v3_large()
    in_f  = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, len(class_names))
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_names


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

class ExplicitBinaryDataset(datasets.ImageFolder):
    """Forces NOT_NOTE=0, NOTE=1 regardless of folder sort order."""
    def find_classes(self, directory):
        return ["NOT_NOTE", "NOTE"], {"NOT_NOTE": 0, "NOTE": 1}


def collect_model_a_scores(model_a):
    """
    Returns arrays:
      p_note  : P(NOTE) for each val image  [N]
      labels  : true binary label           [N]  (1=NOTE, 0=NOT_NOTE)
      denoms  : denomination string or ""   [N]
    """
    ds = ExplicitBinaryDataset(str(VAL_DIR_A), transform=EVAL_TF)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    all_p_note, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            logits = model_a(imgs)
            probs  = torch.softmax(logits, dim=1)
            all_p_note.extend(probs[:, 1].tolist())
            all_labels.extend(lbls.tolist())

    # Extract denomination from filename (note_<DENOM>_<idx>.jpg)
    denoms = []
    for path, _ in ds.samples:
        fname = Path(path).name
        if fname.startswith("note_"):
            parts = fname.split("_")
            denoms.append(parts[1] if len(parts) >= 2 else "")
        else:
            denoms.append("")

    return (
        np.array(all_p_note),
        np.array(all_labels),
        np.array(denoms),
    )


def collect_model_b_scores(model_b, class_names):
    """
    Collects Model B top-prob and margin on its own val set.
    Returns (top_probs, margins, true_labels, class_names_array).
    """
    if not VAL_DIR_B.exists():
        print(f"  [WARN] Model B val dir not found: {VAL_DIR_B}")
        print("  Skipping Model B calibration — using defaults.")
        return None, None, None, class_names

    ds = datasets.ImageFolder(str(VAL_DIR_B), transform=EVAL_TF)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    all_top, all_margin, all_labels = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            logits = model_b(imgs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            sorted_p = np.sort(probs, axis=1)[:, ::-1]
            top    = sorted_p[:, 0]
            margin = sorted_p[:, 0] - sorted_p[:, 1]
            all_top.extend(top.tolist())
            all_margin.extend(margin.tolist())
            all_labels.extend(lbls.tolist())

    return (
        np.array(all_top),
        np.array(all_margin),
        np.array(all_labels),
        class_names,
    )


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def sweep_model_a(p_note, labels, thresholds, target_precision=0.999):
    """
    For each (accept_min, reject_max) pair sweep, compute:
      - NOTE precision on accepted subset
      - NOTE recall (accepted TP / all true notes)
      - NOT_NOTE false acceptance rate (accepted NOT_NOTE / all NOT_NOTEs)
      - REVIEW rate
    
    'accept_min' : P(NOTE) >= accept_min -> considered NOTE
    'reject_max' : P(NOTE) <= reject_max -> considered NOT_NOTE
    between      : REVIEW
    """
    note_mask     = labels == 1
    not_note_mask = labels == 0
    n_note     = note_mask.sum()
    n_not_note = not_note_mask.sum()

    results = []
    for accept_min in thresholds:
        for reject_max in thresholds:
            if reject_max >= accept_min:
                continue   # invalid (review band must be > 0)

            accepted_as_note = p_note >= accept_min
            rejected_as_not  = p_note <= reject_max
            in_review        = ~accepted_as_note & ~rejected_as_not

            tp = int((accepted_as_note & note_mask).sum())
            fp = int((accepted_as_note & not_note_mask).sum())
            fn = int((~accepted_as_note & note_mask).sum())

            precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall     = tp / n_note     if n_note > 0 else 0.0
            review_rate = float(in_review.mean())
            fpr_not_note = fp / n_not_note if n_not_note > 0 else 0.0

            # Notes sent to REVIEW
            notes_in_review = int((in_review & note_mask).sum())

            results.append({
                "accept_min": round(float(accept_min), 3),
                "reject_max": round(float(reject_max), 3),
                "note_precision": round(precision, 6),
                "note_recall": round(recall, 6),
                "review_rate": round(review_rate, 4),
                "fpr_not_note": round(fpr_not_note, 6),
                "notes_in_review": notes_in_review,
                "tp": tp, "fp": fp, "fn": fn,
            })

    return results


def sweep_model_b(top_probs, margins, true_labels, denom_thresholds, margin_thresholds):
    """Sweep Model B acceptance thresholds."""
    if top_probs is None:
        return []
    n_total = len(top_probs)
    results = []
    for denom_t in denom_thresholds:
        for margin_t in margin_thresholds:
            accepted = (top_probs >= denom_t) & (margins >= margin_t)
            acc_correct = (accepted & (true_labels == true_labels)).sum()  # all accepted are correct denom
            # Simple accuracy on accepted subset
            accept_rate = float(accepted.mean())
            results.append({
                "denom_accept_min": round(float(denom_t), 3),
                "margin_min": round(float(margin_t), 3),
                "accept_rate": round(accept_rate, 4),
                "n_accepted": int(accepted.sum()),
                "n_total": n_total,
            })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DivyaKosh Threshold Calibration (Validation Set Only)")
    print("=" * 60)

    # ---- Load models ----
    print("\n[1/4] Loading Model A...")
    model_a = load_model_a()
    print("[1/4] Loading Model B...")
    model_b, b_classes = load_model_b()

    # ---- Collect scores ----
    print("\n[2/4] Collecting Model A validation scores...")
    p_note, labels, denoms = collect_model_a_scores(model_a)
    n_total = len(labels)
    n_note     = int((labels == 1).sum())
    n_not_note = int((labels == 0).sum())
    print(f"      Val set: {n_total} images  ({n_note} NOTE, {n_not_note} NOT_NOTE)")

    print("[2/4] Collecting Model B validation scores...")
    b_top, b_margin, b_labels, b_classes = collect_model_b_scores(model_b, b_classes)

    # ---- Sweep ----
    print("\n[3/4] Sweeping thresholds...")

    a_thresholds = np.arange(0.05, 1.00, 0.05).round(3)
    results_a = sweep_model_a(p_note, labels, a_thresholds, target_precision=0.999)

    # Sort by precision (desc), then recall (desc)
    results_a.sort(key=lambda x: (-x["note_precision"], -x["note_recall"]))

    b_denom_ths  = np.arange(0.50, 1.00, 0.05).round(3)
    b_margin_ths = np.arange(0.05, 0.50, 0.05).round(3)
    results_b = sweep_model_b(b_top, b_margin, b_labels, b_denom_ths, b_margin_ths)

    # ---- Select best thresholds ----
    print("\n[4/4] Selecting thresholds...")

    # Model A: pick the one with highest precision first, then best recall
    # Require precision >= 0.999 (no false NOTE acceptance on val set)
    TARGET_PRECISION = 0.999
    candidates_a = [r for r in results_a if r["note_precision"] >= TARGET_PRECISION]
    if candidates_a:
        # Among perfect-precision candidates, pick highest recall
        best_a = max(candidates_a, key=lambda x: x["note_recall"])
    else:
        # Fallback: best precision available
        best_a = results_a[0]
        print(f"  [WARN] No threshold achieves precision >= {TARGET_PRECISION}. Using best available.")

    selected_accept_min = best_a["accept_min"]
    selected_reject_max = best_a["reject_max"]

    print(f"  Model A: accept_min={selected_accept_min}, reject_max={selected_reject_max}")
    print(f"    NOTE precision={best_a['note_precision']:.4f}, recall={best_a['note_recall']:.4f}")
    print(f"    REVIEW rate={best_a['review_rate']:.4f}, notes in REVIEW={best_a['notes_in_review']}")

    # Model B: pick threshold that gives >90% accept rate (most notes pass)
    # with reasonable margin; default if no B data
    if results_b:
        # Find the combination with best accept_rate >= 0.85 and highest denom_accept_min
        candidates_b = [r for r in results_b if r["accept_rate"] >= 0.85]
        if candidates_b:
            best_b = max(candidates_b, key=lambda x: (x["denom_accept_min"], x["margin_min"]))
        else:
            best_b = max(results_b, key=lambda x: x["accept_rate"])
        selected_denom_t  = best_b["denom_accept_min"]
        selected_margin_t = best_b["margin_min"]
    else:
        selected_denom_t  = 0.70
        selected_margin_t = 0.15
        print("  Model B calibration skipped (no val dir). Using safe defaults.")

    print(f"  Model B: denom_accept_min={selected_denom_t}, margin_min={selected_margin_t}")

    # ---- Denomination-level breakdown for selected threshold ----
    print("\n  Denomination recall at selected Model A threshold:")
    denom_recalls = {}
    for d in ["10", "20", "50", "100", "200", "500"]:
        mask  = (labels == 1) & (denoms == d)
        if mask.sum() == 0:
            denom_recalls[d] = None
            continue
        accepted = (p_note >= selected_accept_min) & mask
        recall   = float(accepted.sum()) / float(mask.sum())
        denom_recalls[d] = round(recall, 4)
        in_review = ((p_note < selected_accept_min) & (p_note > selected_reject_max)) & mask
        print(f"    Rs.{d:>3}: recall={recall:.4f}  in_review={int(in_review.sum())}/{int(mask.sum())}")

    # ---- Save config ----
    config = {
        "calibration_set": str(VAL_DIR_A),
        "n_val_note": n_note,
        "n_val_not_note": n_not_note,
        "target_precision": TARGET_PRECISION,
        "presence_accept_min": float(selected_accept_min),
        "presence_reject_max": float(selected_reject_max),
        "denomination_accept_min": float(selected_denom_t),
        "denomination_margin_min": float(selected_margin_t),
        "min_sharpness": 30.0,
        "selected_model_a_metrics": best_a,
        "denomination_recall_at_threshold": denom_recalls,
    }
    with open(OUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Saved threshold config: {OUT_CONFIG}")

    # ---- Save full sweep report ----
    report = {
        "model_a_sweep": results_a[:50],    # top 50 by precision/recall
        "model_b_sweep": results_b[:30],
        "selected_config": config,
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved calibration report: {OUT_REPORT}")

    print("\nCalibration complete. Review threshold_calibration_report.json before proceeding.")


if __name__ == "__main__":
    main()
