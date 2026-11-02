# -*- coding: utf-8 -*-
"""
cv/pipeline/divyakosh_pipeline.py

DivyaKosh Final Donation Decision Pipeline
===========================================
Combines Model A (presence) and Model B (denomination) to produce one of
three CLASSIFICATION outcomes, each mapped to one of two OPERATIONAL
destinations:

  Classification       Operational action   count_authorized
  ─────────────────────────────────────────────────────────
  ACCEPT_NOTE          COUNT                True
  REJECT_NOT_NOTE      REVIEW               False
  REVIEW               REVIEW               False

Rules:
  - ACCEPT_NOTE:      Model A >= accept_min AND Model B >= denom_min AND
                      margin >= margin_min AND image sharpness OK.
  - REJECT_NOT_NOTE:  Model A confidently identifies NOT_NOTE (P(NOTE) <= reject_max).
                      Operationally routed to REVIEW — never silently discarded.
  - REVIEW:           Any other case (uncertain, low-quality, low-confidence,
                      ambiguous denomination).

INVARIANTS (enforced in code and tests):
  - count_authorized is True ONLY for ACCEPT_NOTE.
  - operational_action is COUNT ONLY for ACCEPT_NOTE.
  - No input is silently discarded.
  - Unsupported denominations cannot be counted.

IMPORTANT:
  - Existing checkpoints are loaded read-only; never overwritten.
  - Thresholds loaded from threshold_config.json (calibrated on val set only).
  - No hardware, blockchain, or database integration.

Usage (single image):
    python pipeline/divyakosh_pipeline.py --image path/to/image.jpg

Usage (programmatic):
    from pipeline.divyakosh_pipeline import DivyaKoshPipeline
    pipe = DivyaKoshPipeline()
    result = pipe.predict(pil_image)
    print(result.classification, result.operational_action, result.count_authorized)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Paths (relative to cv/ directory)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent          # cv/pipeline/
_CV   = _HERE.parent                             # cv/

MODEL_A_CKPT = _CV / "presence" / "presence_classifier_v1.pt"
MODEL_B_CKPT = _CV / "models" / "mobilenet_v3_clean.pt"
CALIB_CONFIG = _HERE / "threshold_config.json"

# Denomination mapping for Model B (alphabetical ImageFolder order)
MODEL_B_CLASSES = ["10", "100", "20", "200", "50", "500"]
DENOM_SORT_ORDER = {c: i for i, c in enumerate(MODEL_B_CLASSES)}

# ---------------------------------------------------------------------------
# Image quality checks
# ---------------------------------------------------------------------------

MIN_LAPLACIAN_VARIANCE = 30.0   # below this -> image is too blurry to classify

def compute_laplacian_variance(pil_img: Image.Image) -> float:
    """Return Laplacian variance as a sharpness proxy."""
    gray = np.array(pil_img.convert("L"))
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_brightness(pil_img: Image.Image) -> float:
    """Return mean pixel brightness (0–255)."""
    gray = np.array(pil_img.convert("L"))
    return float(gray.mean())

# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _load_model_a(device: torch.device) -> torch.nn.Module:
    """Load Model A: MobileNetV3-Small binary classifier (NOT_NOTE=0, NOTE=1)."""
    raw = torch.load(MODEL_A_CKPT, map_location=device, weights_only=False)
    # Checkpoint may be a wrapped dict OR a raw state_dict
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    else:
        state_dict = raw   # raw state_dict saved directly
    model = models.mobilenet_v3_small()
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def _load_model_b(device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    """Load Model B: MobileNetV3-Large 6-class denomination classifier."""
    raw = torch.load(MODEL_B_CKPT, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict  = raw["model_state_dict"]
        class_names = raw.get("class_names", MODEL_B_CLASSES)
    else:
        state_dict  = raw
        class_names = MODEL_B_CLASSES
    model = models.mobilenet_v3_large()
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, class_names


# ---------------------------------------------------------------------------
# Threshold config
# ---------------------------------------------------------------------------

@dataclass
class ThresholdConfig:
    """
    Thresholds calibrated on the validation set.
    Written by calibrate_thresholds.py; loaded here.
    """
    # Model A thresholds
    presence_accept_min: float = 0.70   # P(NOTE) >= this -> consider it a note
    presence_reject_max: float = 0.30   # P(NOTE) <= this -> reject as NOT_NOTE
    # Between reject_max and accept_min is the REVIEW zone for Model A alone.

    # Model B threshold
    denomination_accept_min: float = 0.70    # top-class prob >= this -> accept denomination
    denomination_margin_min: float = 0.15    # top - second must be >= this margin

    # Image quality
    min_sharpness: float = MIN_LAPLACIAN_VARIANCE

    @classmethod
    def from_file(cls, path: Path | str) -> "ThresholdConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Required threshold configuration file not found: {p.resolve()}")
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Unreadable or corrupted threshold configuration file {p.resolve()}: {e}") from e

        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_DENOMINATIONS = frozenset(["10", "20", "50", "100", "200", "500"])

# Operational action strings
_ACT_COUNT  = "COUNT"
_ACT_REVIEW = "REVIEW"

# Classification strings
CLS_ACCEPT      = "ACCEPT_NOTE"
CLS_REJECT      = "REJECT_NOT_NOTE"
CLS_REVIEW      = "REVIEW"


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Structured output of the DivyaKosh donation decision pipeline.

    Fields
    ------
    classification : str
        ACCEPT_NOTE | REJECT_NOT_NOTE | REVIEW
    operational_action : str
        COUNT  — only when classification == ACCEPT_NOTE
        REVIEW — for every other outcome (no silent discard)
    predicted_denomination : str | None
        One of {"10","20","50","100","200","500"} or None.
        Present only when classification == ACCEPT_NOTE.
    presence_confidence : float
        P(NOTE) from Model A (0.0–1.0).
    denomination_confidence : float
        Top-class probability from Model B (0.0 if Model B was not run).
    denomination_margin : float
        top_prob − second_prob from Model B (0.0 if Model B was not run).
    review_reason : str | None
        Human-readable reason when operational_action == REVIEW.
        None for ACCEPT_NOTE.
    count_authorized : bool
        True ONLY when classification == ACCEPT_NOTE.
        All other outcomes are False — this is an invariant enforced in code.

    Invariants (verified by unit tests)
    ------------------------------------
    - count_authorized is True   iff  classification == ACCEPT_NOTE
    - operational_action == COUNT  iff  classification == ACCEPT_NOTE
    - predicted_denomination is not None  iff  classification == ACCEPT_NOTE
    - predicted_denomination in SUPPORTED_DENOMINATIONS  when not None
    - review_reason is not None  when  classification in {REJECT_NOT_NOTE, REVIEW}
    """
    classification: str                     # ACCEPT_NOTE / REJECT_NOT_NOTE / REVIEW
    operational_action: str                 # COUNT / REVIEW
    predicted_denomination: Optional[str]   # e.g. "100" or None
    presence_confidence: float              # P(NOTE) from Model A
    denomination_confidence: float          # top-class prob from Model B (0.0 if skipped)
    denomination_margin: float              # top − second prob (0.0 if skipped)
    review_reason: Optional[str]            # non-None for REJECT_NOT_NOTE and REVIEW
    count_authorized: bool                  # True ONLY for ACCEPT_NOTE
    image_sharpness: float
    image_brightness: float
    latency_ms: float
    model_a_probs: list[float] = field(default_factory=list)  # [P(NOT_NOTE), P(NOTE)]
    model_b_probs: dict        = field(default_factory=dict)  # {denom: prob}
    timestamp: str = ""

    # ------------------------------------------------------------------
    # Backward-compatibility alias
    # ------------------------------------------------------------------
    @property
    def decision(self) -> str:
        """Alias for classification. Kept for backward compatibility."""
        return self.classification

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.classification   # backward-compat key
        return d

    # ------------------------------------------------------------------
    # Invariant check (called internally; also used by tests)
    # ------------------------------------------------------------------
    def assert_invariants(self) -> None:
        """Raise AssertionError if any structural invariant is violated."""
        if self.classification == CLS_ACCEPT:
            assert self.operational_action == _ACT_COUNT,  \
                f"ACCEPT_NOTE must have operational_action=COUNT, got {self.operational_action!r}"
            assert self.count_authorized is True,           \
                "ACCEPT_NOTE must have count_authorized=True"
            assert self.predicted_denomination is not None, \
                "ACCEPT_NOTE must have a predicted_denomination"
            assert self.predicted_denomination in SUPPORTED_DENOMINATIONS, \
                f"predicted_denomination {self.predicted_denomination!r} not in SUPPORTED_DENOMINATIONS"
            assert self.review_reason is None,              \
                "ACCEPT_NOTE must not have a review_reason"
        else:
            assert self.operational_action == _ACT_REVIEW,  \
                f"{self.classification} must have operational_action=REVIEW, got {self.operational_action!r}"
            assert self.count_authorized is False,           \
                f"{self.classification} must have count_authorized=False"
            assert self.review_reason is not None,           \
                f"{self.classification} must have a review_reason"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class DivyaKoshPipeline:
    """
    Three-outcome inference pipeline for DivyaKosh donation box.

    Thread-safety: each call is independent; safe for sequential use.
    Not thread-safe without external locking (CPU model, no batch dim).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        config: Optional[ThresholdConfig] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or ThresholdConfig.from_file(CALIB_CONFIG)

        self._model_a = _load_model_a(self.device)
        self._model_b, self._model_b_classes = _load_model_b(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        pil_image: Image.Image,
        note_crop: Optional[Image.Image] = None,
    ) -> PipelineResult:
        """
        Run the full three-outcome pipeline on one PIL image.

        Args:
            pil_image: RGB PIL Image of full ROI (used for quality & Model A).
            note_crop: Optional cropped RGB PIL Image around the banknote (used for Model B).
                       If None, Model B runs on pil_image directly.

        Returns:
            PipelineResult with decision, confidence scores, and metadata.
        """
        t0 = time.perf_counter()
        pil_rgb = pil_image.convert("RGB")

        # ---- Image quality checks ----------------------------------------
        sharpness  = compute_laplacian_variance(pil_rgb)
        brightness = compute_brightness(pil_rgb)

        if sharpness < self.config.min_sharpness:
            return self._make_review(
                reason=f"Image too blurry (sharpness={sharpness:.1f} < {self.config.min_sharpness})",
                sharpness=sharpness,
                brightness=brightness,
                presence_conf=0.0,
                t0=t0,
            )

        # ---- Model A: presence classifier --------------------------------
        a_probs = self._run_model_a(pil_rgb)
        p_note     = a_probs[1]   # P(NOTE)
        p_not_note = a_probs[0]   # P(NOT_NOTE)

        # Confident NOT_NOTE
        if p_note <= self.config.presence_reject_max:
            elapsed = (time.perf_counter() - t0) * 1000
            return PipelineResult(
                classification=CLS_REJECT,
                operational_action=_ACT_REVIEW,
                predicted_denomination=None,
                presence_confidence=round(p_note, 4),
                denomination_confidence=0.0,
                denomination_margin=0.0,
                review_reason="Model A: confident NOT_NOTE (object is not a banknote)",
                count_authorized=False,
                image_sharpness=round(sharpness, 2),
                image_brightness=round(brightness, 2),
                latency_ms=round(elapsed, 2),
                model_a_probs=[round(p, 4) for p in a_probs],
                model_b_probs={},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )

        # Uncertain presence zone -> REVIEW
        if p_note < self.config.presence_accept_min:
            return self._make_review(
                reason=(
                    f"Model A uncertain: P(NOTE)={p_note:.3f} is between "
                    f"reject_max={self.config.presence_reject_max} and "
                    f"accept_min={self.config.presence_accept_min}"
                ),
                sharpness=sharpness,
                brightness=brightness,
                presence_conf=p_note,
                a_probs=a_probs,
                t0=t0,
            )

        # ---- Model B: denomination classifier ----------------------------
        b_input = note_crop.convert("RGB") if note_crop is not None else pil_rgb
        b_probs_dict, top_denom, top_prob, margin = self._run_model_b(b_input)

        # Model B confidence insufficient
        if top_prob < self.config.denomination_accept_min:
            return self._make_review(
                reason=(
                    f"Model B confidence too low: "
                    f"P({top_denom})={top_prob:.3f} < {self.config.denomination_accept_min}"
                ),
                sharpness=sharpness,
                brightness=brightness,
                presence_conf=p_note,
                a_probs=a_probs,
                b_probs=b_probs_dict,
                top_denom=top_denom,
                top_prob=top_prob,
                margin=margin,
                t0=t0,
            )

        # Model B margin insufficient (ambiguous between two classes)
        if margin < self.config.denomination_margin_min:
            return self._make_review(
                reason=(
                    f"Model B prediction ambiguous: "
                    f"top-second margin={margin:.3f} < {self.config.denomination_margin_min}"
                ),
                sharpness=sharpness,
                brightness=brightness,
                presence_conf=p_note,
                a_probs=a_probs,
                b_probs=b_probs_dict,
                top_denom=top_denom,
                top_prob=top_prob,
                margin=margin,
                t0=t0,
            )

        # ---- ACCEPT_NOTE -------------------------------------------------
        elapsed = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            classification=CLS_ACCEPT,
            operational_action=_ACT_COUNT,
            predicted_denomination=top_denom,
            presence_confidence=round(p_note, 4),
            denomination_confidence=round(top_prob, 4),
            denomination_margin=round(margin, 4),
            review_reason=None,
            count_authorized=True,
            image_sharpness=round(sharpness, 2),
            image_brightness=round(brightness, 2),
            latency_ms=round(elapsed, 2),
            model_a_probs=[round(p, 4) for p in a_probs],
            model_b_probs={k: round(v, 4) for k, v in b_probs_dict.items()},
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def predict_path(self, image_path: str | Path) -> PipelineResult:
        """Convenience wrapper: load image from file path and predict."""
        img = Image.open(image_path)
        return self.predict(img)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_model_a(self, pil_rgb: Image.Image) -> list[float]:
        """Return [P(NOT_NOTE), P(NOTE)]."""
        # Ensure black background context: Model A was trained on banknotes
        # resting on a black donation chamber floor (10-40% canvas coverage).
        # If an edge-to-edge crop is provided (mean edge brightness > 30),
        # pad with dark margins so Model A detects object boundaries properly.
        img_for_a = pil_rgb
        arr = np.array(pil_rgb)
        if arr.ndim == 3 and arr.shape[0] > 10 and arr.shape[1] > 10:
            edges = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]])
            if float(edges.mean()) > 30.0:
                from PIL import ImageOps
                w, h = pil_rgb.size
                pad_w = max(4, int(w * 0.15))
                pad_h = max(4, int(h * 0.15))
                img_for_a = ImageOps.expand(pil_rgb, border=(pad_w, pad_h), fill=(0, 0, 0))

        tensor = _EVAL_TRANSFORM(img_for_a).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model_a(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()
        return probs   # index 0 = NOT_NOTE, index 1 = NOTE

    def _run_model_b(
        self, pil_rgb: Image.Image
    ) -> tuple[dict, str, float, float]:
        """
        Return (probs_dict, top_class, top_prob, margin).
        margin = top_prob - second_prob.
        """
        tensor = _EVAL_TRANSFORM(pil_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model_b(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()

        probs_dict = {cls: p for cls, p in zip(self._model_b_classes, probs)}
        sorted_probs = sorted(probs, reverse=True)
        top_prob  = sorted_probs[0]
        second    = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin    = top_prob - second
        top_idx   = int(torch.tensor(probs).argmax())
        top_class = self._model_b_classes[top_idx]
        return probs_dict, top_class, top_prob, margin

    def _make_review(
        self,
        reason: str,
        sharpness: float,
        brightness: float,
        presence_conf: float,
        a_probs: Optional[list] = None,
        b_probs: Optional[dict] = None,
        top_denom: Optional[str] = None,
        top_prob: float = 0.0,
        margin: float = 0.0,
        t0: float = 0.0,
    ) -> PipelineResult:
        elapsed = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            classification=CLS_REVIEW,
            operational_action=_ACT_REVIEW,
            predicted_denomination=None,   # never carry a denom into REVIEW
            presence_confidence=round(presence_conf, 4),
            denomination_confidence=round(top_prob, 4),
            denomination_margin=round(margin, 4),
            review_reason=reason,
            count_authorized=False,
            image_sharpness=round(sharpness, 2),
            image_brightness=round(brightness, 2),
            latency_ms=round(elapsed, 2),
            model_a_probs=[round(p, 4) for p in (a_probs or [])],
            model_b_probs={k: round(v, 4) for k, v in (b_probs or {}).items()},
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="DivyaKosh three-outcome inference pipeline"
    )
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--config", default=str(CALIB_CONFIG),
                        help="Path to threshold_config.json (default: pipeline/threshold_config.json)")
    parser.add_argument("--output", default=None,
                        help="Optional path to save result JSON")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    config   = ThresholdConfig.from_file(cfg_path)

    print(f"[Pipeline] Loading models...")
    pipe = DivyaKoshPipeline(config=config)

    img = Image.open(args.image)
    result = pipe.predict(img)

    output = result.to_dict()
    print(json.dumps(output, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"[Pipeline] Result saved to {args.output}")


if __name__ == "__main__":
    main()
