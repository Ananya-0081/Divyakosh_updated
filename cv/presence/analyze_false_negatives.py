"""
cv/presence/analyze_false_negatives.py

Performs detailed false-negative analysis for Model A:
1. Identifies all 17 real-camera false negatives.
2. Exports them to cv/presence/analysis/false_negatives/.
3. Generates a contact sheet of all 17 missed notes.
4. Generates a detailed CSV with predictions and probabilities.
5. Computes visual and statistical comparison between the 17 FN and 34 TP images.
6. Analyzes Model A probability distributions strictly on the validation set
   and produces a threshold sweep table.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRESENCE_DIR = PROJECT_ROOT / "presence"
DATASET_DIR = PRESENCE_DIR / "dataset"
ANALYSIS_DIR = PRESENCE_DIR / "analysis" / "false_negatives"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
INPUT_SIZE = (224, 224)
MODEL_A_TRANSFORM = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model_a(ckpt_path: Path):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def predict_probs(model: nn.Module, img_bgr: np.ndarray):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = MODEL_A_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        prob_not_note = float(probs[0].item())
        prob_note = float(probs[1].item())
    return prob_not_note, prob_note


def extract_features(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    h, w = gray.shape
    # Glare / overexposed pixels (>245)
    glare_pct = float((gray > 245).sum() / gray.size * 100)
    # Background coverage (near black < 25)
    bg_pct = float((gray < 25).sum() / gray.size * 100)
    note_coverage = 100.0 - bg_pct
    return {
        "mean_brightness": round(mean_val, 2),
        "contrast_std": round(std_val, 2),
        "sharpness_lap_var": round(lap_var, 2),
        "glare_pct": round(glare_pct, 2),
        "note_coverage_pct": round(note_coverage, 2),
        "dims": f"{w}x{h}"
    }


def main():
    print("=" * 60)
    print("RUNNING FALSE NEGATIVE INVESTIGATION FOR MODEL A")
    print("=" * 60)

    model_path = PRESENCE_DIR / "presence_classifier_v1.pt"
    model = load_model_a(model_path)

    # 1. Inspect real-camera NOTE test set
    real_note_dir = DATASET_DIR / "real_test" / "NOTE"
    real_files = sorted(list(real_note_dir.glob("*.jpg")))
    print(f"Total real-camera test notes: {len(real_files)}")

    fn_records = []
    tp_records = []

    for f in real_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        p_not_note, p_note = predict_probs(model, img)
        pred_class = "NOTE" if p_note >= 0.5 else "NOT_NOTE"

        # Extract denomination from filename
        denom = "unknown"
        for d in ["10", "20", "50", "100", "200", "500"]:
            if f"_{d}_" in f.name:
                denom = d
                break

        feats = extract_features(img)
        rec = {
            "filename": f.name,
            "filepath": str(f),
            "ground_truth": f"Rs.{denom}",
            "denomination": denom,
            "predicted_class": pred_class,
            "p_note": round(p_note, 6),
            "p_not_note": round(p_not_note, 6),
            "threshold": 0.50,
            **feats
        }

        if pred_class == "NOT_NOTE":
            fn_records.append(rec)
            # Copy to false_negatives folder
            shutil.copy2(f, ANALYSIS_DIR / f.name)
        else:
            tp_records.append(rec)

    print(f"False Negatives identified: {len(fn_records)}")
    print(f"True Positives identified:  {len(tp_records)}")

    # 2. Write CSV
    csv_path = ANALYSIS_DIR / "false_negatives.csv"
    fieldnames = [
        "filename", "ground_truth", "denomination", "predicted_class",
        "p_note", "p_not_note", "threshold", "dims",
        "mean_brightness", "contrast_std", "sharpness_lap_var",
        "glare_pct", "note_coverage_pct"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in fn_records:
            writer.writerow(r)
    print(f"Saved CSV: {csv_path}")

    # 3. Generate Contact Sheet of all 17 Missed Notes
    print("Generating contact sheet of missed notes...")
    thumbs = []
    for r in fn_records:
        img = cv2.imread(r["filepath"])
        # Resize to 240x180
        t = cv2.resize(img, (240, 180))
        # Overlay label
        cv2.putText(t, f"{r['ground_truth']} (p={r['p_note']:.3f})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        thumbs.append(t)

    # 17 images -> 3 rows of 6 (last row has 5)
    cols = 6
    rows = []
    for i in range(0, len(thumbs), cols):
        chunk = thumbs[i:i+cols]
        while len(chunk) < cols:
            chunk.append(np.zeros_like(thumbs[0]))
        rows.append(np.hstack(chunk))
    cs = np.vstack(rows)
    cs_path = ANALYSIS_DIR / "contact_sheet_fn.jpg"
    cv2.imwrite(str(cs_path), cs)
    print(f"Saved FN Contact Sheet: {cs_path}")

    # 4. Statistical comparison between FN and TP
    def get_avg(records, key):
        vals = [r[key] for r in records]
        return round(float(np.mean(vals)), 2)

    stat_comp = {
        "count": {"FN": len(fn_records), "TP": len(tp_records)},
        "mean_brightness": {"FN": get_avg(fn_records, "mean_brightness"), "TP": get_avg(tp_records, "mean_brightness")},
        "contrast_std": {"FN": get_avg(fn_records, "contrast_std"), "TP": get_avg(tp_records, "contrast_std")},
        "sharpness_lap_var": {"FN": get_avg(fn_records, "sharpness_lap_var"), "TP": get_avg(tp_records, "sharpness_lap_var")},
        "glare_pct": {"FN": get_avg(fn_records, "glare_pct"), "TP": get_avg(tp_records, "glare_pct")},
        "note_coverage_pct": {"FN": get_avg(fn_records, "note_coverage_pct"), "TP": get_avg(tp_records, "note_coverage_pct")},
        "average_p_note": {
            "FN": round(float(np.mean([r["p_note"] for r in fn_records])), 4),
            "TP": round(float(np.mean([r["p_note"] for r in tp_records])), 4)
        }
    }
    print(f"Statistical comparison: {json.dumps(stat_comp, indent=2)}")

    # 5. Probability distribution and threshold sweep ON VALIDATION SET ONLY
    print("\nEvaluating Model A threshold sweep ON VALIDATION SET ONLY (1,200 images)...")
    val_note_dir = DATASET_DIR / "val" / "NOTE"
    val_not_note_dir = DATASET_DIR / "val" / "NOT_NOTE"

    val_notes = sorted(list(val_note_dir.glob("*.jpg")))
    val_not_notes = sorted(list(val_not_note_dir.glob("*.jpg")))

    val_p_notes = []
    val_y_true = []

    for f in val_notes:
        img = cv2.imread(str(f))
        if img is not None:
            _, p_note = predict_probs(model, img)
            val_p_notes.append(p_note)
            val_y_true.append(1)

    for f in val_not_notes:
        img = cv2.imread(str(f))
        if img is not None:
            _, p_note = predict_probs(model, img)
            val_p_notes.append(p_note)
            val_y_true.append(0)

    val_p_notes = np.array(val_p_notes)
    val_y_true = np.array(val_y_true)

    thresholds = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    sweep_results = []

    for t in thresholds:
        preds = (val_p_notes >= t).astype(int)
        tp = int(np.sum((val_y_true == 1) & (preds == 1)))
        tn = int(np.sum((val_y_true == 0) & (preds == 0)))
        fp = int(np.sum((val_y_true == 0) & (preds == 1)))
        fn = int(np.sum((val_y_true == 1) & (preds == 0)))

        prec = round(tp / max(1, (tp + fp)), 4)
        rec = round(tp / max(1, (tp + fn)), 4)
        f1 = round(2 * prec * rec / max(1e-8, (prec + rec)), 4)
        fpr = round(fp / max(1, (fp + tn)), 4)
        acc = round((tp + tn) / len(val_y_true), 4)

        sweep_results.append({
            "threshold": t,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fpr": fpr,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn
        })

    # Save summary json
    summary = {
        "fn_records": fn_records,
        "statistical_comparison": stat_comp,
        "validation_threshold_sweep": sweep_results
    }
    with open(ANALYSIS_DIR / "fn_investigation_data.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Investigation script completed successfully.")


if __name__ == "__main__":
    main()

