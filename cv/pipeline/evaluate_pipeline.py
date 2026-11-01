# -*- coding: utf-8 -*-
"""
cv/pipeline/evaluate_pipeline.py

Pipeline Evaluation Script
===========================
Evaluates the DivyaKosh three-outcome pipeline on:
  1. Synthetic held-out test set  (cv/presence/dataset/test/)
  2. Real-camera challenge set    (cv/presence/dataset/real_test/)

Reports results separately for each set. Never uses training or val data.
Never modifies checkpoints or datasets.

Metrics reported per set:
  - Pipeline decision distribution (ACCEPT_NOTE / REJECT_NOT_NOTE / REVIEW)
  - NOTE precision, recall, F1 (accepted vs ground-truth)
  - False acceptance rate for NOT_NOTE
  - REVIEW rate (total, and for genuine notes)
  - Denomination accuracy on accepted notes
  - Per-denomination precision, recall
  - Per-distractor-category false acceptance rate
  - Confusion matrix
  - Latency (mean, p50, p95)

Usage:
    cd cv
    .\.venv\Scripts\python.exe pipeline/evaluate_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_CV   = _HERE.parent
sys.path.insert(0, str(_HERE))

from divyakosh_pipeline import DivyaKoshPipeline, ThresholdConfig, CALIB_CONFIG

TEST_DIR      = _CV / "presence" / "dataset" / "test"
REAL_TEST_DIR = _CV / "presence" / "dataset" / "real_test"
OUT_REPORT    = _HERE / "pipeline_evaluation_report.json"

# Denomination strings that map to ground truth label from filename
DENOM_LIST = ["10", "20", "50", "100", "200", "500"]

# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(split_dir: Path) -> list[dict]:
    """
    Walk a split directory (NOTE/ and NOT_NOTE/ subdirs) and
    return a list of {path, gt_label, gt_denom, gt_subcat} dicts.

    gt_label:  1=NOTE, 0=NOT_NOTE
    gt_denom:  denomination str or None
    gt_subcat: NOT_NOTE subcategory str or None
    """
    records = []
    for cls_name, gt_label in [("NOTE", 1), ("NOT_NOTE", 0)]:
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            fname = img_path.name

            gt_denom  = None
            gt_subcat = None

            if gt_label == 1:
                # note_<DENOM>_<idx>.jpg  or  rc_note_<DENOM>_<idx>.jpg
                for prefix in ("note_", "rc_note_"):
                    if fname.startswith(prefix):
                        rest = fname[len(prefix):]
                        parts = rest.split("_")
                        if parts:
                            gt_denom = parts[0]
                        break
            else:
                # <SUBCAT>_<idx>.jpg  or  rc_challenge_<SUBCAT>_<idx>.jpg
                for prefix in ("rc_challenge_", ""):
                    stripped = fname[len(prefix):]
                    # subcat is everything before the last _<digits>.jpg
                    # try removing prefix then grabbing subcat
                for prefix in ("rc_challenge_",):
                    if fname.startswith(prefix):
                        rest = fname[len(prefix):]
                        # rest = subcat_idx.jpg
                        parts = rest.rsplit("_", 1)
                        gt_subcat = parts[0] if len(parts) == 2 else rest
                        break
                else:
                    # no rc_challenge_ prefix: take everything up to last _<digits>
                    parts = fname.rsplit("_", 1)
                    gt_subcat = parts[0] if len(parts) == 2 else fname.replace(".jpg", "")

            records.append({
                "path":     img_path,
                "gt_label": gt_label,
                "gt_denom": gt_denom,
                "gt_subcat": gt_subcat,
            })
    return records


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_set(pipeline: DivyaKoshPipeline, records: list[dict], set_name: str) -> dict:
    print(f"\n  Evaluating {set_name} ({len(records)} images)...")
    t_start = time.perf_counter()

    latencies  = []
    decisions  = []
    gt_labels  = []
    gt_denoms  = []
    gt_subcats = []
    pred_denoms = []

    for rec in records:
        img = Image.open(rec["path"]).convert("RGB")
        t0  = time.perf_counter()
        res = pipeline.predict(img)
        lat = (time.perf_counter() - t0) * 1000

        latencies.append(lat)
        decisions.append(res.decision)
        gt_labels.append(rec["gt_label"])
        gt_denoms.append(rec["gt_denom"])
        gt_subcats.append(rec["gt_subcat"])
        pred_denoms.append(res.predicted_denomination)

    decisions   = np.array(decisions)
    gt_labels   = np.array(gt_labels)
    gt_denoms   = np.array(gt_denoms)
    gt_subcats  = np.array(gt_subcats)
    pred_denoms = np.array(pred_denoms)
    latencies   = np.array(latencies)

    # ---- Ground truth masks ----
    note_mask     = gt_labels == 1
    not_note_mask = gt_labels == 0
    n_note     = int(note_mask.sum())
    n_not_note = int(not_note_mask.sum())

    # ---- Decision masks ----
    d_accept = decisions == "ACCEPT_NOTE"
    d_reject = decisions == "REJECT_NOT_NOTE"
    d_review = decisions == "REVIEW"

    # ---- Primary metrics ----
    # True positives: ACCEPT_NOTE on a genuine NOTE
    tp = int((d_accept & note_mask).sum())
    # False positives: ACCEPT_NOTE on a NOT_NOTE
    fp = int((d_accept & not_note_mask).sum())
    # False negatives: NOT (ACCEPT_NOTE) on a genuine NOTE
    fn = int(((d_reject | d_review) & note_mask).sum())
    # True negatives: REJECT or REVIEW on NOT_NOTE
    tn = int(((d_reject | d_review) & not_note_mask).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / n_note    if n_note > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    far       = fp / n_not_note if n_not_note > 0 else 0.0   # false acceptance rate

    # Notes in REVIEW
    notes_in_review    = int((d_review & note_mask).sum())
    not_note_in_review = int((d_review & not_note_mask).sum())
    review_rate        = float(d_review.mean())

    # Decision distribution
    dist = {
        "ACCEPT_NOTE":      int(d_accept.sum()),
        "REJECT_NOT_NOTE":  int(d_reject.sum()),
        "REVIEW":           int(d_review.sum()),
    }

    # ---- Per-denomination recall ----
    denom_metrics = {}
    for d in DENOM_LIST:
        mask = note_mask & (gt_denoms == d)
        if mask.sum() == 0:
            continue
        tp_d    = int((d_accept & mask).sum())
        total_d = int(mask.sum())
        recall_d = tp_d / total_d if total_d > 0 else 0.0

        # Denomination accuracy among accepted notes of this denom
        # (did Model B pick the right class?)
        accepted_d = d_accept & mask
        correct_d  = int((accepted_d & (pred_denoms == d)).sum())
        denom_acc  = correct_d / int(accepted_d.sum()) if accepted_d.sum() > 0 else None

        denom_metrics[d] = {
            "total": total_d,
            "tp_accepted": tp_d,
            "recall_presence": round(recall_d, 4),
            "denom_accuracy_on_accepted": round(denom_acc, 4) if denom_acc is not None else None,
        }

    # ---- Per-distractor false acceptance rate ----
    subcats_seen = sorted(set(gt_subcats[not_note_mask]))
    subcat_far = {}
    for sc in subcats_seen:
        if sc is None or sc == "":
            continue
        mask = not_note_mask & (gt_subcats == sc)
        fp_sc = int((d_accept & mask).sum())
        total_sc = int(mask.sum())
        subcat_far[sc] = {
            "total": total_sc,
            "false_accepted": fp_sc,
            "fpr": round(fp_sc / total_sc, 4) if total_sc > 0 else 0.0,
        }

    # ---- Denomination accuracy on all accepted notes ----
    accepted_notes = d_accept & note_mask
    if accepted_notes.sum() > 0:
        correct_denom = int((accepted_notes & (pred_denoms == gt_denoms)).sum())
        denom_accuracy_overall = correct_denom / int(accepted_notes.sum())
    else:
        denom_accuracy_overall = None

    # ---- Latency ----
    lat_report = {
        "mean_ms": round(float(latencies.mean()), 2),
        "std_ms":  round(float(latencies.std()), 2),
        "p50_ms":  round(float(np.percentile(latencies, 50)), 2),
        "p95_ms":  round(float(np.percentile(latencies, 95)), 2),
        "max_ms":  round(float(latencies.max()), 2),
    }

    elapsed = time.perf_counter() - t_start
    print(f"    Done in {elapsed:.1f}s")
    print(f"    ACCEPT={dist['ACCEPT_NOTE']}  REJECT={dist['REJECT_NOT_NOTE']}  REVIEW={dist['REVIEW']}")
    print(f"    NOTE precision={precision:.4f}  recall={recall:.4f}  F1={f1:.4f}")
    print(f"    False acceptance rate={far:.4f}  Review rate={review_rate:.4f}")
    print(f"    Notes sent to REVIEW={notes_in_review}/{n_note}")

    return {
        "set_name": set_name,
        "n_total": len(records),
        "n_note": n_note,
        "n_not_note": n_not_note,
        "decision_distribution": dist,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "note_precision": round(precision, 6),
        "note_recall":    round(recall, 6),
        "note_f1":        round(f1, 6),
        "false_acceptance_rate": round(far, 6),
        "review_rate": round(review_rate, 4),
        "notes_in_review": notes_in_review,
        "not_note_in_review": not_note_in_review,
        "denomination_accuracy_on_accepted_notes": round(denom_accuracy_overall, 4) if denom_accuracy_overall is not None else None,
        "per_denomination_metrics": denom_metrics,
        "per_distractor_fpr": subcat_far,
        "latency_ms": lat_report,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DivyaKosh Pipeline Evaluation")
    print("=" * 60)
    print("[!] Using HELD-OUT test sets only. No training or val data touched.")

    # Load thresholds (calibrated on val set)
    config = ThresholdConfig.from_file(CALIB_CONFIG)
    print(f"\nThresholds loaded from: {CALIB_CONFIG}")
    print(f"  presence_accept_min={config.presence_accept_min}")
    print(f"  presence_reject_max={config.presence_reject_max}")
    print(f"  denomination_accept_min={config.denomination_accept_min}")
    print(f"  denomination_margin_min={config.denomination_margin_min}")
    print(f"  min_sharpness={config.min_sharpness}")

    print("\nLoading pipeline models...")
    pipe = DivyaKoshPipeline(config=config)
    print("  Models loaded.")

    # ---- Evaluate synthetic test set ----
    print("\n--- Synthetic Test Set ---")
    syn_records = load_dataset(TEST_DIR)
    syn_results = evaluate_set(pipe, syn_records, "synthetic_test")

    # ---- Evaluate real-camera challenge set ----
    print("\n--- Real-Camera Challenge Set ---")
    real_records = load_dataset(REAL_TEST_DIR)
    real_results = evaluate_set(pipe, real_records, "real_camera_challenge")

    # ---- Example predictions ----
    print("\nGenerating example predictions...")
    examples = generate_examples(pipe, syn_records, real_records)

    # ---- Compile report ----
    report = {
        "pipeline_version": "DivyaKosh v1.0 Three-Outcome Pipeline",
        "model_a_checkpoint": str(_CV / "presence" / "presence_classifier_v1.pt"),
        "model_b_checkpoint": str(_CV / "models" / "mobilenet_v3_clean.pt"),
        "threshold_config": {
            "presence_accept_min": config.presence_accept_min,
            "presence_reject_max": config.presence_reject_max,
            "denomination_accept_min": config.denomination_accept_min,
            "denomination_margin_min": config.denomination_margin_min,
            "min_sharpness": config.min_sharpness,
        },
        "synthetic_test_results": syn_results,
        "real_camera_challenge_results": real_results,
        "example_predictions": examples,
        "important_caveats": [
            "Synthetic test accuracy does NOT represent real-world performance.",
            "Real-camera results use only 102 images (51 NOTE, 51 NOT_NOTE).",
            "Real NOT_NOTE images in real_test are synthetic distractors, not real temple objects.",
            "False-positive risk from flowers, prasad, diyas, and envelopes is UNKNOWN.",
            "Model A missed 33% of real banknotes at threshold 0.50 (see false_negative analysis).",
            "Results may differ significantly in actual temple deployment.",
        ],
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved: {OUT_REPORT}")

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for res in [syn_results, real_results]:
        print(f"\n[{res['set_name']}]  n={res['n_total']}")
        print(f"  ACCEPT={res['decision_distribution']['ACCEPT_NOTE']}  "
              f"REJECT={res['decision_distribution']['REJECT_NOT_NOTE']}  "
              f"REVIEW={res['decision_distribution']['REVIEW']}")
        print(f"  NOTE precision={res['note_precision']:.4f}  "
              f"recall={res['note_recall']:.4f}  "
              f"F1={res['note_f1']:.4f}")
        print(f"  False acceptance rate (NOT_NOTE)={res['false_acceptance_rate']:.4f}")
        print(f"  Review rate={res['review_rate']:.4f}  "
              f"Notes in REVIEW={res['notes_in_review']}/{res['n_note']}")
        if res.get("denomination_accuracy_on_accepted_notes") is not None:
            print(f"  Denomination accuracy (on accepted notes)="
                  f"{res['denomination_accuracy_on_accepted_notes']:.4f}")
        print(f"  Latency: mean={res['latency_ms']['mean_ms']}ms  "
              f"p50={res['latency_ms']['p50_ms']}ms  "
              f"p95={res['latency_ms']['p95_ms']}ms")


def generate_examples(
    pipeline: DivyaKoshPipeline,
    syn_records: list[dict],
    real_records: list[dict],
) -> dict:
    """Generate representative example predictions for each decision type."""
    examples = {}

    # Find examples from synthetic set
    all_records = syn_records + real_records

    for target_decision in ["ACCEPT_NOTE", "REJECT_NOT_NOTE", "REVIEW"]:
        for rec in all_records:
            img = Image.open(rec["path"]).convert("RGB")
            res = pipeline.predict(img)
            if res.decision == target_decision:
                examples[target_decision] = {
                    "image_path": str(rec["path"]),
                    "ground_truth_label": "NOTE" if rec["gt_label"] == 1 else "NOT_NOTE",
                    "ground_truth_denomination": rec["gt_denom"],
                    "result": res.to_dict(),
                }
                break

    return examples


if __name__ == "__main__":
    main()

