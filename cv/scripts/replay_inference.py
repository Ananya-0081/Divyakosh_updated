# -*- coding: utf-8 -*-
"""
cv/scripts/replay_inference.py

Read-Only Replay & Diagnostic Tool for DivyaKosh Banknote Images.
=================================================================
Takes a saved chamber crop / image and displays:
  - Image quality metrics (Laplacian sharpness variance, brightness)
  - Model A binary presence probabilities [P(NOT_NOTE), P(NOTE)]
  - Model B full six-class denomination distribution (10, 20, 50, 100, 200, 500)
  - Top-1 and top-2 predictions and margin
  - Decision outcome (ACCEPT_NOTE, REJECT_NOT_NOTE, REVIEW)
  - Operational action (COUNT, REVIEW) and count_authorized status

SECURITY / SAFETY GUARANTEE:
  - STRICTLY READ-ONLY.
  - Never writes to logs/inference.jsonl, logs/review.jsonl, SQLite, or Monad blockchain.

Usage:
    python scripts/replay_inference.py logs/review_images/review_20261101_044442_340997.jpg
    python scripts/replay_inference.py --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Setup project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from PIL import Image

from pipeline.divyakosh_pipeline import (
    DivyaKoshPipeline,
    compute_brightness,
    compute_laplacian_variance,
)


def replay_image(image_path: Path, device: torch.device | None = None) -> dict:
    """
    Run diagnostic inference on a single saved image without writing any records.
    Returns a dictionary of raw inspection metrics.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path.resolve()}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = DivyaKoshPipeline(device=device)

    pil_img = Image.open(image_path).convert("RGB")

    # 1. Image Quality
    sharpness = compute_laplacian_variance(pil_img)
    brightness = compute_brightness(pil_img)

    # 2. Model A Probabilities
    a_probs = pipeline._run_model_a(pil_img)
    p_not_note, p_note = a_probs[0], a_probs[1]

    # 3. Model B Full 6-Class Distribution
    b_probs_dict, top_denom, top_prob, margin = pipeline._run_model_b(pil_img)

    # Sort Model B classes by descending probability
    sorted_b = sorted(b_probs_dict.items(), key=lambda kv: kv[1], reverse=True)
    top1_denom, top1_prob = sorted_b[0]
    top2_denom, top2_prob = sorted_b[1] if len(sorted_b) > 1 else ("N/A", 0.0)

    # 4. Pipeline Final Decision
    result = pipeline.predict(pil_img)

    # 5. Display Report
    print("=" * 65)
    print("DIVYAKOSH READ-ONLY INFERENCE REPLAY")
    print("=" * 65)
    print(f"Image Path        : {image_path.resolve()}")
    print(f"Dimensions (WxH)  : {pil_img.size[0]} x {pil_img.size[1]} pixels")
    print(f"Device            : {device}")
    print("-" * 65)
    print("IMAGE QUALITY METRICS:")
    print(f"  Sharpness (Laplacian Var) : {sharpness:.2f} (min required: {pipeline.config.min_sharpness:.1f})")
    print(f"  Brightness (Mean L)       : {brightness:.2f} / 255")
    print("-" * 65)
    print("MODEL A: PRESENCE CLASSIFIER (MobileNetV3-Small):")
    print(f"  P(NOTE)                   : {p_note:.4f} (accept >= {pipeline.config.presence_accept_min:.2f})")
    print(f"  P(NOT_NOTE)               : {p_not_note:.4f} (reject <= {pipeline.config.presence_reject_max:.2f})")
    print("-" * 65)
    print("MODEL B: DENOMINATION CLASSIFIER (MobileNetV3-Large):")
    for denom, prob in sorted_b:
        bar = "#" * int(prob * 30)
        marker = " <== TOP 1" if denom == top1_denom else ""
        print(f"  Rs. {denom:>3} : {prob:.4f} | {bar:<30}{marker}")
    print(f"  Top-1 Class               : Rs. {top1_denom} ({top1_prob:.4f})")
    print(f"  Top-2 Class               : Rs. {top2_denom} ({top2_prob:.4f})")
    print(f"  Prediction Margin         : {margin:.4f} (min required: {pipeline.config.denomination_margin_min:.2f})")
    print("-" * 65)
    print("FINAL PIPELINE OUTCOME:")
    print(f"  Classification            : {result.classification}")
    print(f"  Operational Action        : {result.operational_action}")
    print(f"  Count Authorized          : {result.count_authorized}")
    print(f"  Predicted Denomination    : Rs. {result.predicted_denomination or 'None'}")
    print(f"  Review Reason             : {result.review_reason or 'None (Authorized for Counting)'}")
    print(f"  Inference Latency         : {result.latency_ms:.2f} ms")
    print("=" * 65)

    return {
        "image_path": str(image_path),
        "sharpness": sharpness,
        "brightness": brightness,
        "p_note": p_note,
        "p_not_note": p_not_note,
        "model_b_probs": b_probs_dict,
        "top1_denom": top1_denom,
        "top1_prob": top1_prob,
        "top2_denom": top2_denom,
        "top2_prob": top2_prob,
        "margin": margin,
        "classification": result.classification,
        "operational_action": result.operational_action,
        "count_authorized": result.count_authorized,
        "predicted_denomination": result.predicted_denomination,
        "review_reason": result.review_reason,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Read-only diagnostic replay of saved DivyaKosh banknote images."
    )
    parser.add_argument(
        "positional_image",
        nargs="?",
        default=None,
        help="Path to image file to replay",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to image file to replay (named flag)",
    )
    args = parser.parse_args()

    target = args.positional_image or args.image
    if not target:
        parser.error("Please provide an image path: python scripts/replay_inference.py <image_path>")

    replay_image(Path(target))


if __name__ == "__main__":
    main()

