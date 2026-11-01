"""
cv/detector/train_detector.py

Fine-tune YOLOv8n on the banknote detection dataset built by
prepare_detector_dataset.py.

The trained weights are saved to:
    cv/models/banknote_detector.pt   (best val mAP50 checkpoint)

This script does NOT modify any existing files.

Usage:
    cd cv
    .venv/Scripts/python detector/train_detector.py
    .venv/Scripts/python detector/train_detector.py --epochs 50 --imgsz 416
    .venv/Scripts/python detector/train_detector.py --resume   # resume interrupted run

CPU-only performance estimate (12-thread machine):
    YOLOv8n, 4500 images, imgsz=320, batch=8:
    ~4-6 min/epoch → 30 epochs ≈ 2-3 hours total
    Use --epochs 30 for a quick first run; --epochs 100 for full training.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_FILE    = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent          # cv/
_DETECTOR_DIR = _THIS_FILE.parent                 # cv/detector/
_DATASET_YAML = _DETECTOR_DIR / "yolo_dataset" / "data.yaml"
_MODELS_DIR   = _PROJECT_ROOT / "models"
_RUNS_DIR     = _DETECTOR_DIR / "runs"            # ultralytics output
_OUTPUT_PT    = _MODELS_DIR / "banknote_detector.pt"


def train(
    epochs: int = 30,
    imgsz:  int = 320,
    batch:  int = 8,
    resume: bool = False,
    verbose: bool = True,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed.\n"
            "Run: .venv/Scripts/pip install 'ultralytics>=8.3.0'"
        )

    if not _DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset not found: {_DATASET_YAML}\n"
            "Run: python detector/prepare_detector_dataset.py"
        )

    if verbose:
        print("=" * 60)
        print("DivyaKosh — YOLOv8n Banknote Detector Training")
        print("=" * 60)
        print(f"  Dataset   : {_DATASET_YAML}")
        print(f"  Model     : yolov8n (pretrained COCO)")
        print(f"  Epochs    : {epochs}")
        print(f"  Image size: {imgsz}px")
        print(f"  Batch     : {batch}")
        print(f"  Output    : {_OUTPUT_PT}")
        print()

    # Load pretrained YOLOv8n
    if resume:
        last_run = _find_last_run()
        if last_run:
            model = YOLO(last_run / "weights" / "last.pt")
            print(f"Resuming from: {last_run}")
        else:
            print("No previous run found, starting fresh.")
            model = YOLO("yolov8n.pt")
    else:
        model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data=str(_DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=4,
        project=str(_RUNS_DIR),
        name="banknote_detector",
        exist_ok=True,
        verbose=verbose,
        # Augmentation — some are already in our synthetic data, so reduce overlap
        degrees=10.0,       # slight rotation (our synthetic already rotates heavily)
        translate=0.1,
        scale=0.3,
        fliplr=0.0,         # ❌ no horizontal flip — currency notes are not symmetric
        flipud=0.0,         # ❌ no vertical flip
        mosaic=0.5,
        mixup=0.0,
    )

    # Copy best weights to cv/models/
    best_pt = _RUNS_DIR / "banknote_detector" / "weights" / "best.pt"
    if best_pt.exists():
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, _OUTPUT_PT)
        print(f"\nBest weights saved → {_OUTPUT_PT}")
    else:
        print(f"\nWarning: best.pt not found at {best_pt}")

    # Print final metrics
    if results and hasattr(results, "results_dict"):
        rd = results.results_dict
        print("\nFinal validation metrics:")
        for k in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
            if k in rd:
                print(f"  {k}: {rd[k]:.4f}")


def _find_last_run() -> Path | None:
    """Find the most recent training run directory."""
    run_dir = _RUNS_DIR / "banknote_detector"
    last_pt = run_dir / "weights" / "last.pt"
    return run_dir if last_pt.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n banknote detector for DivyaKosh."
    )
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--imgsz",  type=int, default=320,
                        help="Input image size (default: 320 for CPU speed)")
    parser.add_argument("--batch",  type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()
    train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, resume=args.resume)


if __name__ == "__main__":
    main()

