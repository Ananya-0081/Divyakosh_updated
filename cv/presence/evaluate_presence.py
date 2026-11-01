"""
cv/presence/evaluate_presence.py

Comprehensive evaluation and benchmark comparing:
  - Model A: MobileNetV3-Small Binary Presence Classifier
  - YOLO: Existing YOLOv8n Banknote Detector (banknote_detector.pt)

Evaluates:
  1. Synthetic Test Set (600 NOTE + 600 NOT_NOTE = 1,200 images):
     - NOTE precision, recall, F1
     - NOT_NOTE precision, recall, F1
     - Confusion matrix [[TN, FP], [FN, TP]]
     - Breakdown by denomination: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500
     - Breakdown of NOT_NOTE false positives by subcategory:
       empty_black_bg, white_paper, receipt, coins, wrappers, ambient_shadows
  2. Untouched Real-Camera Challenge Set (51 real notes + 51 challenge negatives = 102 images)
  3. Latency benchmark on CPU (100 single-image forward passes):
     - Average, p50, p95 latency
  4. Apples-to-apples comparison between Model A and YOLO

Saves output report to: cv/presence/evaluation_report.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRESENCE_DIR = PROJECT_ROOT / "presence"
DATASET_DIR = PRESENCE_DIR / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"

DEVICE = torch.device("cpu")  # Strictly benchmark on CPU as in production

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


def load_yolo_model(ckpt_path: Path):
    if YOLO is None:
        raise ImportError("Ultralytics YOLO not installed in environment.")
    return YOLO(str(ckpt_path))


def predict_model_a(model: nn.Module, img_bgr: np.ndarray, threshold: float = 0.5) -> Tuple[int, float]:
    """Returns (pred_label: 0=NOT_NOTE, 1=NOTE, prob_note: float)"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = MODEL_A_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        prob_note = float(probs[1].item())
        pred = 1 if prob_note >= threshold else 0
    return pred, prob_note


def predict_yolo(yolo_model, img_bgr: np.ndarray, conf: float = 0.25) -> Tuple[int, float]:
    """Returns (pred_label: 0=NOT_NOTE, 1=NOTE, max_conf: float)"""
    results = yolo_model.predict(img_bgr, conf=conf, verbose=False)
    boxes = results[0].boxes
    if len(boxes) > 0:
        max_c = float(boxes.conf.max().item())
        return 1, max_c
    return 0, 0.0


def compute_full_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    acc = (tp + tn) / max(1, total)

    # NOTE metrics (class 1)
    prec_note = tp / max(1, (tp + fp))
    rec_note = tp / max(1, (tp + fn))
    f1_note = 2 * prec_note * rec_note / max(1e-8, (prec_note + rec_note))

    # NOT_NOTE metrics (class 0)
    prec_not_note = tn / max(1, (tn + fn))
    rec_not_note = tn / max(1, (tn + fp))
    f1_not_note = 2 * prec_not_note * rec_not_note / max(1e-8, (prec_not_note + rec_not_note))

    fpr = fp / max(1, (fp + tn))

    return {
        "accuracy": round(acc, 4),
        "note": {
            "precision": round(prec_note, 4),
            "recall": round(rec_note, 4),
            "f1": round(f1_note, 4),
            "support": tp + fn
        },
        "not_note": {
            "precision": round(prec_not_note, 4),
            "recall": round(rec_not_note, 4),
            "f1": round(f1_not_note, 4),
            "support": tn + fp
        },
        "false_positive_rate": round(fpr, 4),
        "confusion_matrix": {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "matrix_2x2": [[tn, fp], [fn, tp]]
        },
        "total_support": total
    }


def benchmark_latency(model_a, yolo_model, sample_img_bgr: np.ndarray, num_runs: int = 100) -> Dict:
    print(f"\nBenchmarking inference latency on CPU ({num_runs} runs)...")

    # Warmup
    for _ in range(10):
        predict_model_a(model_a, sample_img_bgr)
        predict_yolo(yolo_model, sample_img_bgr)

    # Model A
    latencies_a = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        predict_model_a(model_a, sample_img_bgr)
        latencies_a.append((time.perf_counter() - t0) * 1000)

    # YOLO
    latencies_yolo = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        predict_yolo(yolo_model, sample_img_bgr)
        latencies_yolo.append((time.perf_counter() - t0) * 1000)

    return {
        "model_a_cpu": {
            "mean_ms": round(float(np.mean(latencies_a)), 2),
            "std_ms": round(float(np.std(latencies_a)), 2),
            "p50_ms": round(float(np.percentile(latencies_a, 50)), 2),
            "p95_ms": round(float(np.percentile(latencies_a, 95)), 2),
        },
        "yolo_cpu": {
            "mean_ms": round(float(np.mean(latencies_yolo)), 2),
            "std_ms": round(float(np.std(latencies_yolo)), 2),
            "p50_ms": round(float(np.percentile(latencies_yolo, 50)), 2),
            "p95_ms": round(float(np.percentile(latencies_yolo, 95)), 2),
        },
        "speedup_factor": round(float(np.mean(latencies_yolo) / max(0.01, np.mean(latencies_a))), 2)
    }


def evaluate_dataset_split(split_dir: Path, model_a, yolo_model, desc: str):
    print(f"\nEvaluating on {desc} at {split_dir}...")
    note_dir = split_dir / "NOTE"
    not_note_dir = split_dir / "NOT_NOTE"

    note_files = sorted(list(note_dir.glob("*.jpg")))
    not_note_files = sorted(list(not_note_dir.glob("*.jpg")))

    y_true = []
    preds_a = []
    preds_yolo = []

    # Per-denomination tracking
    denom_a = {d: {"correct": 0, "total": 0} for d in ["10", "20", "50", "100", "200", "500", "unknown"]}
    denom_yolo = {d: {"correct": 0, "total": 0} for d in ["10", "20", "50", "100", "200", "500", "unknown"]}

    for f in tqdm(note_files, desc=f"  Evaluating Notes ({len(note_files)})"):
        img = cv2.imread(str(f))
        if img is None: continue

        pa, _ = predict_model_a(model_a, img)
        py, _ = predict_yolo(yolo_model, img)

        y_true.append(1)
        preds_a.append(pa)
        preds_yolo.append(py)

        # Identify denomination from filename
        d = "unknown"
        for candidate in ["10", "20", "50", "100", "200", "500"]:
            if f"_{candidate}_" in f.name:
                d = candidate
                break
        denom_a[d]["total"] += 1
        denom_yolo[d]["total"] += 1
        if pa == 1: denom_a[d]["correct"] += 1
        if py == 1: denom_yolo[d]["correct"] += 1

    # Per-subcategory tracking for NOT_NOTE
    subcats = ["empty_black_bg", "white_paper", "receipt", "coins", "wrappers", "ambient_shadows"]
    subcat_a = {s: {"fp": 0, "total": 0} for s in subcats}
    subcat_yolo = {s: {"fp": 0, "total": 0} for s in subcats}

    for f in tqdm(not_note_files, desc=f"  Evaluating NOT_NOTE ({len(not_note_files)})"):
        img = cv2.imread(str(f))
        if img is None: continue

        pa, _ = predict_model_a(model_a, img)
        py, _ = predict_yolo(yolo_model, img)

        y_true.append(0)
        preds_a.append(pa)
        preds_yolo.append(py)

        # Determine subcategory
        matched_sub = "unknown"
        for s in subcats:
            if s in f.name:
                matched_sub = s
                break
        if matched_sub in subcat_a:
            subcat_a[matched_sub]["total"] += 1
            subcat_yolo[matched_sub]["total"] += 1
            if pa == 1: subcat_a[matched_sub]["fp"] += 1
            if py == 1: subcat_yolo[matched_sub]["fp"] += 1

    metrics_a = compute_full_metrics(y_true, preds_a)
    metrics_yolo = compute_full_metrics(y_true, preds_yolo)

    # Subcategory FPR tables
    subcat_table_a = {
        s: {
            "total": subcat_a[s]["total"],
            "fp": subcat_a[s]["fp"],
            "fpr": round(subcat_a[s]["fp"] / max(1, subcat_a[s]["total"]), 4)
        } for s in subcats if subcat_a[s]["total"] > 0
    }
    subcat_table_yolo = {
        s: {
            "total": subcat_yolo[s]["total"],
            "fp": subcat_yolo[s]["fp"],
            "fpr": round(subcat_yolo[s]["fp"] / max(1, subcat_yolo[s]["total"]), 4)
        } for s in subcats if subcat_yolo[s]["total"] > 0
    }

    # Denomination recall tables
    denom_table_a = {
        d: {
            "total": denom_a[d]["total"],
            "recall": round(denom_a[d]["correct"] / max(1, denom_a[d]["total"]), 4)
        } for d in denom_a if denom_a[d]["total"] > 0
    }
    denom_table_yolo = {
        d: {
            "total": denom_yolo[d]["total"],
            "recall": round(denom_yolo[d]["correct"] / max(1, denom_yolo[d]["total"]), 4)
        } for d in denom_yolo if denom_yolo[d]["total"] > 0
    }

    return {
        "model_a": {
            "overall": metrics_a,
            "distractor_fpr_breakdown": subcat_table_a,
            "denomination_recall_breakdown": denom_table_a
        },
        "yolo": {
            "overall": metrics_yolo,
            "distractor_fpr_breakdown": subcat_table_yolo,
            "denomination_recall_breakdown": denom_table_yolo
        }
    }


def run_evaluation():
    print("=" * 60)
    print("Evaluating Model A vs. YOLO Banknote Presence Detector")
    print("=" * 60)

    model_a_path = PRESENCE_DIR / "presence_classifier_v1.pt"
    yolo_path = MODELS_DIR / "banknote_detector.pt"

    if not model_a_path.exists():
        raise FileNotFoundError(f"Model A checkpoint missing: {model_a_path}")
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO checkpoint missing: {yolo_path}")

    model_a = load_model_a(model_a_path)
    yolo_model = load_yolo_model(yolo_path)

    # 1. Synthetic Test Set Evaluation (1,200 images)
    synthetic_eval = evaluate_dataset_split(
        DATASET_DIR / "test",
        model_a,
        yolo_model,
        "Held-Out Synthetic Test Set (1,200 images)"
    )

    # 2. Real-Camera Challenge Test Set Evaluation (102 images)
    real_eval = evaluate_dataset_split(
        DATASET_DIR / "real_test",
        model_a,
        yolo_model,
        "Untouched Real-Camera Challenge Set (102 images)"
    )

    # 3. Latency Benchmark
    sample_img = cv2.imread(str(list((DATASET_DIR / "test" / "NOTE").glob("*.jpg"))[0]))
    latency_report = benchmark_latency(model_a, yolo_model, sample_img, num_runs=100)

    full_report = {
        "evaluation_name": "DivyaKosh Model A (MobileNetV3-Small) vs. YOLOv8n Benchmark",
        "timestamp": "2026-10-31",
        "latency_benchmark": latency_report,
        "synthetic_test_results": synthetic_eval,
        "real_camera_challenge_results": real_eval,
    }

    report_path = PRESENCE_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nComplete evaluation report saved to: {report_path}")
    return full_report


if __name__ == "__main__":
    run_evaluation()
