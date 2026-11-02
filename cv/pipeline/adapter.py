# -*- coding: utf-8 -*-
"""
cv/pipeline/adapter.py

Production Storage and Blockchain Adapter for DivyaKosh.
=========================================================
Bridges the three-outcome CV decision pipeline (`DivyaKoshPipeline`) with:
  1. The JSONL audit logs (`logs/inference.jsonl`, `logs/review.jsonl`)
  2. The SQLite persistent store (`donations` table)
  3. The Monad blockchain registration service (`/register-proof`)

Security Invariants Enforced:
  - Only records with `count_authorized is True`, `operational_action == "COUNT"`,
    and `classification == "ACCEPT_NOTE"` can ever be authorized for donation counting.
  - Denomination MUST belong to `SUPPORTED_DENOMINATIONS` (10, 20, 50, 100, 200, 500).
  - Every non-authorized result (REJECT_NOT_NOTE, REVIEW) is routed to the review destination.
  - Zero non-authorized records are submitted to SQLite or Monad blockchain.
  - The canonical proof format (`decision: "ACCEPT"`, `pocket_id`, `predicted_denomination`, etc.)
    is strictly preserved so smart contracts and blockchain SDK verifiers remain 100% compatible.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from pipeline.divyakosh_pipeline import (
    CLS_ACCEPT,
    CLS_REJECT,
    CLS_REVIEW,
    PipelineResult,
    SUPPORTED_DENOMINATIONS,
    _ACT_COUNT,
    _ACT_REVIEW,
)

# Supported denominations as both string and integer sets for fast lookup
SUPPORTED_DENOM_STRINGS = SUPPORTED_DENOMINATIONS
SUPPORTED_DENOM_INTS = frozenset([int(d) for d in SUPPORTED_DENOMINATIONS])

# Default log paths relative to cv/
_CV_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INFERENCE_LOG = _CV_ROOT / "logs" / "inference.jsonl"
DEFAULT_REVIEW_LOG = _CV_ROOT / "logs" / "review.jsonl"
DEFAULT_REVIEW_IMAGES_DIR = _CV_ROOT / "logs" / "review_images"
DEFAULT_ACCEPTED_IMAGES_DIR = _CV_ROOT / "logs" / "accepted_images"

MODEL_VERSION_TAG = "divyakosh_v1_three_outcome"


# ---------------------------------------------------------------------------
# Authorization Validation
# ---------------------------------------------------------------------------

def validate_for_donation(record: dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Strict validation gate for donation authorization.

    Returns (True, None) if and only if:
      - count_authorized is exactly True (boolean, not truthy)
      - operational_action == "COUNT"
      - classification == "ACCEPT_NOTE"
      - predicted_denomination is one of supported denominations
      - decision == "ACCEPT" (legacy field required by smart contracts/proofs)
    
    Returns (False, reason) for any violation.
    """
    if not isinstance(record, dict):
        return False, "record is not a dictionary"

    # 1. Exact boolean check
    if record.get("count_authorized") is not True:
        return False, "count_authorized is not True"

    # 2. Operational action must be COUNT
    if record.get("operational_action") != _ACT_COUNT:
        return False, f"operational_action is '{record.get('operational_action')}', expected '{_ACT_COUNT}'"

    # 3. Classification must be ACCEPT_NOTE
    if record.get("classification") != CLS_ACCEPT:
        return False, f"classification is '{record.get('classification')}', expected '{CLS_ACCEPT}'"

    # 4. Denomination must be supported
    denom_raw = record.get("predicted_denomination")
    if denom_raw is None:
        return False, "predicted_denomination is None"
    denom_str = str(denom_raw).strip()
    if denom_str not in SUPPORTED_DENOM_STRINGS:
        return False, f"predicted_denomination '{denom_str}' is not in supported list {sorted(SUPPORTED_DENOM_STRINGS)}"

    # 5. Legacy decision alignment
    if record.get("decision") != "ACCEPT":
        return False, f"legacy decision field is '{record.get('decision')}', expected 'ACCEPT'"

    return True, None


# ---------------------------------------------------------------------------
# Record Formatting
# ---------------------------------------------------------------------------

def format_accepted_donation_record(
    result: PipelineResult,
    pocket_id: str,
    source: str = "webcam",
    model_version: str = MODEL_VERSION_TAG,
    timestamp: Optional[str] = None,
    image_ref: Optional[str] = None,
) -> dict[str, Any]:
    """
    Convert an ACCEPT_NOTE PipelineResult into a fully authorized storage/blockchain record.
    Raises ValueError if the PipelineResult is not an authorized ACCEPT_NOTE.
    """
    result.assert_invariants()

    if (
        result.classification != CLS_ACCEPT
        or result.count_authorized is not True
        or result.operational_action != _ACT_COUNT
        or result.predicted_denomination not in SUPPORTED_DENOM_STRINGS
    ):
        raise ValueError(
            f"Cannot format non-accepted PipelineResult as donation record: "
            f"classification={result.classification}, count_authorized={result.count_authorized}"
        )

    ts = timestamp or datetime.now(timezone.utc).isoformat()
    denom_str = str(result.predicted_denomination)

    record = {
        # Legacy canonical fields (strictly preserved for blockchain proof / SQLite)
        "pocket_id": pocket_id,
        "timestamp": ts,
        "item_status": "NOTE_DETECTED",
        "predicted_denomination": denom_str,
        "classifier_confidence": round(float(result.denomination_confidence), 4),
        "confidence": round(float(result.denomination_confidence), 4),
        "source": source,
        "decision": "ACCEPT",
        "target_bin": f"BIN_{denom_str}",
        "model_version": model_version,
        # New security and operational policy fields
        "classification": CLS_ACCEPT,
        "operational_action": _ACT_COUNT,
        "count_authorized": True,
        "presence_confidence": round(float(result.presence_confidence), 4),
        "denomination_confidence": round(float(result.denomination_confidence), 4),
        "denomination_margin": round(float(result.denomination_margin), 4),
        "review_reason": None,
        "image_sharpness": round(float(result.image_sharpness), 2),
        "image_brightness": round(float(result.image_brightness), 2),
        "latency_ms": round(float(result.latency_ms), 2),
        "image_ref": image_ref,
    }

    # Verify authorization invariants on the formatted record
    valid, reason = validate_for_donation(record)
    if not valid:
        raise RuntimeError(f"Internal error formatting donation record: {reason}")

    return record


def format_review_record(
    result: PipelineResult,
    pocket_id: str,
    source: str = "webcam",
    model_version: str = MODEL_VERSION_TAG,
    timestamp: Optional[str] = None,
    image_ref: Optional[str] = None,
) -> dict[str, Any]:
    """
    Convert any non-accepted PipelineResult (REJECT_NOT_NOTE or REVIEW)
    into a review log record. count_authorized is guaranteed to be False.
    """
    result.assert_invariants()

    if result.count_authorized is True or result.operational_action == _ACT_COUNT:
        raise ValueError("Cannot format an authorized ACCEPT_NOTE result as a review record")

    ts = timestamp or datetime.now(timezone.utc).isoformat()
    status = "NOT_NOTE_REJECTED" if result.classification == CLS_REJECT else "NEEDS_OPERATOR_REVIEW"

    return {
        "pocket_id": pocket_id,
        "timestamp": ts,
        "item_status": status,
        "predicted_denomination": None,
        "classifier_confidence": round(float(result.denomination_confidence), 4),
        "confidence": round(float(result.denomination_confidence), 4),
        "source": source,
        "decision": "REVIEW",
        "target_bin": "BIN_REVIEW",
        "model_version": model_version,
        # Policy fields
        "classification": result.classification,
        "operational_action": _ACT_REVIEW,
        "count_authorized": False,
        "presence_confidence": round(float(result.presence_confidence), 4),
        "denomination_confidence": round(float(result.denomination_confidence), 4),
        "denomination_margin": round(float(result.denomination_margin), 4),
        "review_reason": result.review_reason or "Uncertain prediction routed to review",
        "image_sharpness": round(float(result.image_sharpness), 2),
        "image_brightness": round(float(result.image_brightness), 2),
        "latency_ms": round(float(result.latency_ms), 2),
        "image_ref": image_ref,
    }


# ---------------------------------------------------------------------------
# Logging & Storage Helpers
# ---------------------------------------------------------------------------

def append_accepted_log(
    record: dict[str, Any],
    log_path: Path = DEFAULT_INFERENCE_LOG,
    frame_bgr: Optional[np.ndarray] = None,
    images_dir: Path = DEFAULT_ACCEPTED_IMAGES_DIR,
) -> None:
    """
    Append an authorized donation record to the main inference.jsonl audit log.
    Optionally saves the candidate image to logs/accepted_images/ for auditing.
    """
    valid, reason = validate_for_donation(record)
    if not valid:
        raise PermissionError(f"Security check failed: cannot log unauthorized record as donation ({reason})")

    image_ref = record.get("image_ref")
    if frame_bgr is not None and image_ref is None:
        try:
            import cv2
            images_dir.mkdir(parents=True, exist_ok=True)
            ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            denom_str = str(record.get("predicted_denomination", "note"))
            fname = f"accept_{denom_str}_{ts_slug}.jpg"
            save_path = images_dir / fname
            cv2.imwrite(str(save_path), frame_bgr)
            record["image_ref"] = str(save_path)
        except Exception as err:
            print(f"Warning: could not save accepted image ({err})")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def append_review_log(
    review_record: dict[str, Any],
    log_path: Path = DEFAULT_REVIEW_LOG,
    frame_bgr: Optional[np.ndarray] = None,
    images_dir: Path = DEFAULT_REVIEW_IMAGES_DIR,
) -> None:
    """
    Append an unaccepted candidate to logs/review.jsonl.
    Optionally saves the candidate image for human operator audit.
    """
    if review_record.get("count_authorized") is True:
        raise PermissionError("Security check failed: cannot write count_authorized=True into review log")

    image_ref = review_record.get("image_ref")
    if frame_bgr is not None and image_ref is None:
        try:
            import cv2
            images_dir.mkdir(parents=True, exist_ok=True)
            ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            fname = f"{review_record['classification'].lower()}_{ts_slug}.jpg"
            save_path = images_dir / fname
            cv2.imwrite(str(save_path), frame_bgr)
            review_record["image_ref"] = str(save_path)
        except Exception as err:
            print(f"Warning: could not save review image ({err})")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(review_record, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Blockchain Gateway
# ---------------------------------------------------------------------------

def safe_register_blockchain_proof(
    record: dict[str, Any],
    register_fn: Optional[Any] = None,
) -> Optional[dict[str, Any]]:
    """
    Gateway for Monad blockchain proof registration.
    Strictly prevents any non-authorized or non-ACCEPT record from being submitted.
    """
    valid, reason = validate_for_donation(record)
    if not valid:
        print(f"Blockchain gateway: Registration REJECTED by security adapter ({reason})")
        return None

    if register_fn is None:
        from src.blockchain import register_blockchain_proof
        register_fn = register_blockchain_proof

    return register_fn(record)

