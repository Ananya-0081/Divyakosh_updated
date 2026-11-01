# -*- coding: utf-8 -*-
"""
cv/scripts/live_camera.py

DivyaKosh Production Live Camera Pipeline
=========================================
Integrated with the calibrated three-outcome CV decision pipeline:
  Model A (MobileNetV3-Small binary presence)
    +
  Model B (MobileNetV3-Large 6-class denomination)
    +
  Quality checks (Laplacian sharpness) & calibrated acceptance thresholds.

Operational Destinations:
  1. COUNT  (Only for ACCEPT_NOTE with count_authorized=True)
     -> Logged to inference.jsonl
     -> Inserted into SQLite donations table
     -> Registered on Monad blockchain
     -> Enters WAIT_FOR_REMOVAL (prevents duplicate counting)

  2. REVIEW (For REJECT_NOT_NOTE and REVIEW)
     -> Logged to review.jsonl (with image reference)
     -> NEVER inserted into SQLite donations
     -> NEVER registered on blockchain
     -> NEVER counted as currency

Anti-Duplicate Protection:
  - Temporal presence streak ensures note is stationary before evaluation.
  - After acceptance, the pipeline enters WAIT_FOR_REMOVAL.
  - Consecutive frames cannot re-count the same physical note.
  - Full removal (absence streak >= 5 frames) required before another note is accepted.

Usage:
    python scripts/live_camera.py
    python scripts/live_camera.py --camera 1
    python scripts/live_camera.py --continuous
    python scripts/live_camera.py --no_blockchain
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple

# ============================================================
# PROJECT ROOT & PATH SETUP
# ============================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image

# ============================================================
# PIPELINE & ADAPTER IMPORTS
# ============================================================

from pipeline.divyakosh_pipeline import (
    CLS_ACCEPT,
    CLS_REJECT,
    CLS_REVIEW,
    DivyaKoshPipeline,
    PipelineResult,
    SUPPORTED_DENOMINATIONS,
    ThresholdConfig,
    _ACT_COUNT,
    _ACT_REVIEW,
)
from pipeline.adapter import (
    DEFAULT_INFERENCE_LOG,
    DEFAULT_REVIEW_LOG,
    DEFAULT_REVIEW_IMAGES_DIR,
    MODEL_VERSION_TAG,
    append_accepted_log,
    append_review_log,
    format_accepted_donation_record,
    format_review_record,
    safe_register_blockchain_proof,
    validate_for_donation,
)

# ============================================================
# SQLITE INGESTION
# ============================================================

try:
    from sqlite.importer import import_single_record
    from sqlite.config import DATABASE_FILE as _SQLITE_DB_FILE

    _SQLITE_AVAILABLE = True
except Exception as _sqlite_err:
    _SQLITE_AVAILABLE = False
    _SQLITE_DB_FILE = None
    print(f"Warning: SQLite importer unavailable ({_sqlite_err}).")

# ============================================================
# CONSTANTS & CONFIGURATION
# ============================================================

WINDOW_NAME = "DivyaKosh - Smart Donation Box"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_POS_X = 50
WINDOW_POS_Y = 50

DENOM_LABELS = {
    "10": "Rs. 10",
    "20": "Rs. 20",
    "50": "Rs. 50",
    "100": "Rs. 100",
    "200": "Rs. 200",
    "500": "Rs. 500",
}

WARMUP_FRAMES = 20
PRESENCE_PIXEL_THRESHOLD = 30
PRESENCE_RATIO_THRESHOLD = 0.08
MIN_CONTOUR_RATIO = 0.04
MAX_CONTOUR_RATIO = 0.85
REQUIRED_PRESENCE_FRAMES = 4
REQUIRED_REMOVAL_FRAMES = 30   # Strict 30 consecutive valid ABSENT frames (1.0s at 30fps)
CLASSIFICATION_COOLDOWN = 0.30


class CameraState(Enum):
    CALIBRATING = auto()
    WAITING_FOR_NOTE = auto()
    CONFIRMING_PRESENCE = auto()
    INSPECTING = auto()
    ACCEPTED = auto()
    WAIT_FOR_REMOVAL = auto()


class ChamberStatus(Enum):
    ABSENT = auto()     # Clean empty black chamber floor
    PRESENT = auto()    # Candidate object inside chamber
    OCCLUDED = auto()   # Hand, arm, lid, or high-coverage occlusion (>85%)
    UNKNOWN = auto()    # Ambiguous transitional state


# ============================================================
# ROI & PREPROCESSING
# ============================================================

def compute_roi(frame_h: int, frame_w: int) -> Tuple[int, int, int, int]:
    """
    Compute the centered region of interest inside the donation box.
    Uses a symmetrical chamber floor to support banknotes in any orientation
    (horizontal, vertical, diagonal) without edge truncation.
    """
    chamber_size = int(round(min(frame_w, frame_h) * 0.70))
    x1 = (frame_w - chamber_size) // 2
    y1 = (frame_h - chamber_size) // 2
    x2 = x1 + chamber_size
    y2 = y1 + chamber_size
    return x1, y1, x2, y2


def preprocess_gray(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (21, 21), 0)


def build_background(background_frames: list[np.ndarray]) -> np.ndarray:
    background = np.median(np.stack(background_frames, axis=0), axis=0)
    return background.astype(np.uint8)


def detect_chamber_status(
    background_gray: np.ndarray,
    current_gray: np.ndarray,
) -> Tuple[ChamberStatus, float, float]:
    """
    Background subtraction gate to detect chamber status against the black background.
    Returns (ChamberStatus, foreground_ratio, contour_ratio).

    Statuses:
      - ABSENT: Empty black chamber (foreground < threshold, contour < threshold)
      - PRESENT: Candidate banknote object present
      - OCCLUDED: Hand, lid, or large occlusion covering >85% of chamber
      - UNKNOWN: Ambiguous transitional frame
    """
    diff = cv2.absdiff(background_gray, current_gray)
    _, mask = cv2.threshold(diff, PRESENCE_PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    roi_area = mask.shape[0] * mask.shape[1]
    foreground_pixels = cv2.countNonZero(mask)
    foreground_ratio = foreground_pixels / float(roi_area)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area

    contour_ratio = largest_area / float(roi_area)

    # 1. Occlusion check: hand, lid, or large coverage covering >85% of chamber
    if contour_ratio > MAX_CONTOUR_RATIO or foreground_ratio > MAX_CONTOUR_RATIO:
        return ChamberStatus.OCCLUDED, foreground_ratio, contour_ratio

    # 2. Present check: candidate object inside chamber
    if (
        foreground_ratio >= PRESENCE_RATIO_THRESHOLD
        and MIN_CONTOUR_RATIO <= contour_ratio <= MAX_CONTOUR_RATIO
    ):
        return ChamberStatus.PRESENT, foreground_ratio, contour_ratio

    # 3. Absent check: clean empty black chamber
    if foreground_ratio < PRESENCE_RATIO_THRESHOLD and contour_ratio < MIN_CONTOUR_RATIO:
        return ChamberStatus.ABSENT, foreground_ratio, contour_ratio

    # 4. Unknown / transitional
    return ChamberStatus.UNKNOWN, foreground_ratio, contour_ratio


def detect_note_presence(
    background_gray: np.ndarray,
    current_gray: np.ndarray,
) -> Tuple[bool, float, float]:
    """Backward-compatible wrapper around detect_chamber_status."""
    status, fg_r, c_r = detect_chamber_status(background_gray, current_gray)
    return (status == ChamberStatus.PRESENT), fg_r, c_r


# ============================================================
# UI OVERLAY
# ============================================================

def draw_overlay(
    frame: np.ndarray,
    roi_coords: Tuple[int, int, int, int],
    status: str,
    fps: float,
    presence_ratio: float,
    result: Optional[PipelineResult] = None,
    session_donations: int = 0,
) -> None:
    x1, y1, x2, y2 = roi_coords
    h, w = frame.shape[:2]

    # Color coding based on operational state
    if status.startswith("ACCEPTED"):
        color = (0, 220, 0)       # Green
    elif status.startswith("REJECTED") or status.startswith("REVIEW"):
        color = (0, 140, 255)     # Amber / Orange
    elif status.startswith("INSPECTING") or status.startswith("CHECKING") or status.startswith("DETECTING"):
        color = (0, 200, 255)     # Yellow
    else:
        color = (180, 180, 180)   # Grey

    # ROI rectangle & corner brackets
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    bracket = 20
    thickness = 3
    corners = [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]
    for cx, cy, dx, dy in corners:
        cv2.line(frame, (cx, cy), (cx + dx * bracket, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy * bracket), color, thickness)

    # Top instructions
    cv2.putText(frame, "DivyaKosh Smart Donation Box", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (max(10, w - 130), 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # Bottom status panel
    panel_y = h - 70
    cv2.rectangle(frame, (0, panel_y - 20), (w, h), (20, 20, 20), -1)

    cv2.putText(frame, status, (15, panel_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    info_line = f"Confirmed Donations: {session_donations}  |  Presence: {presence_ratio * 100:.1f}%  |  Q = Quit"
    cv2.putText(frame, info_line, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)

    # If result available, show detailed breakdown
    if result is not None and result.classification == CLS_ACCEPT:
        denom_text = DENOM_LABELS.get(result.predicted_denomination, str(result.predicted_denomination))
        detail = f"{denom_text}  (Confidence: {result.denomination_confidence * 100:.1f}%)"
        cv2.putText(frame, detail, (x1, max(25, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    elif result is not None and result.classification in (CLS_REJECT, CLS_REVIEW):
        reason_text = (result.review_reason or "Routed to review")[:50]
        cv2.putText(frame, reason_text, (x1, max(25, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 1, cv2.LINE_AA)


# ============================================================
# OPERATIONAL EXECUTION ENGINE
# ============================================================

def execute_donation_acceptance(
    result: PipelineResult,
    camera_index: int,
    log_path: Path,
    no_blockchain: bool = False,
) -> dict:
    """
    Execute the authorized donation counting workflow.
    Guaranteed to run ONLY when count_authorized is True.
    """
    pocket_id = f"webcam-{camera_index}"
    record = format_accepted_donation_record(result, pocket_id=pocket_id)

    # 1. Main JSONL inference audit log
    append_accepted_log(record, log_path=log_path)
    print(f"\n[COUNT AUTHORIZED] Rs.{record['predicted_denomination']} logged to {log_path}")

    # 2. SQLite Ingestion
    if _SQLITE_AVAILABLE and _SQLITE_DB_FILE:
        try:
            res = import_single_record(record=record, database_file=_SQLITE_DB_FILE)
            print(f"SQLite: {res.inserted} record(s) inserted into {_SQLITE_DB_FILE}")
        except Exception as e:
            print(f"Warning: SQLite insertion failed ({e}). Audit log remains intact.")

    # 3. Blockchain Registration
    if not no_blockchain:
        try:
            br = safe_register_blockchain_proof(record)
            if br:
                print(f"Blockchain: proof registered ({br.get('proof_hash', '')[:20]}...)")
            else:
                print("Blockchain: proof registration deferred/offline.")
        except Exception as e:
            print(f"Warning: blockchain registration failed ({e}).")

    return record


def execute_review_routing(
    result: PipelineResult,
    camera_index: int,
    review_log_path: Path,
    frame_bgr: Optional[np.ndarray] = None,
) -> dict:
    """
    Execute the safe review routing workflow.
    Never authorizes counting, never writes to SQLite donations, never calls blockchain.
    """
    pocket_id = f"webcam-{camera_index}"
    review_record = format_review_record(result, pocket_id=pocket_id)

    append_review_log(review_record, log_path=review_log_path, frame_bgr=frame_bgr)
    print(f"\n[REVIEW ROUTED] {result.classification}: {result.review_reason} -> logged to {review_log_path}")
    return review_record


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="DivyaKosh Production Live Camera Pipeline")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--log_path", type=str, default=str(DEFAULT_INFERENCE_LOG), help="Path to inference.jsonl")
    parser.add_argument("--review_log_path", type=str, default=str(DEFAULT_REVIEW_LOG), help="Path to review.jsonl")
    parser.add_argument("--continuous", action="store_true", help="Keep running across multiple donations (no auto-exit)")
    parser.add_argument("--no_blockchain", action="store_true", help="Skip blockchain registration")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("DivyaKosh Live Camera Pipeline")
    print("=" * 60)
    print(f"Device               : {device}")
    print(f"Camera index         : {args.camera}")
    print(f"Donation audit log   : {args.log_path}")
    print(f"Review audit log     : {args.review_log_path}")
    print(f"Continuous mode      : {args.continuous}")
    print("Loading DivyaKoshPipeline (Model A + Model B + calibrated thresholds)...")

    pipeline = DivyaKoshPipeline(device=device)
    print("Pipeline ready. Verified Model A + Model B checkpoints loaded.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}. Try --camera 1.")
        return

    log_path = Path(args.log_path)
    review_log_path = Path(args.review_log_path)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.moveWindow(WINDOW_NAME, WINDOW_POS_X, WINDOW_POS_Y)

    # State variables
    state = CameraState.CALIBRATING
    background_frames: list[np.ndarray] = []
    background_gray: Optional[np.ndarray] = None
    frame_counter = 0

    presence_streak = 0
    absence_streak = 0
    session_donations = 0

    last_result: Optional[PipelineResult] = None
    last_status = "CALIBRATING BACKGROUND"
    last_eval_time = 0.0

    prev_time = time.perf_counter()
    fps = 0.0

    print("\nStarting camera. KEEP ROI EMPTY during background calibration.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            now = time.perf_counter()
            dt = now - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = now

            h, w = frame.shape[:2]
            roi_coords = compute_roi(h, w)
            x1, y1, x2, y2 = roi_coords
            roi_crop = frame[y1:y2, x1:x2]
            roi_gray = preprocess_gray(roi_crop)

            # --------------------------------------------------------
            # 1. BACKGROUND CALIBRATION
            # --------------------------------------------------------
            if state == CameraState.CALIBRATING:
                background_frames.append(roi_gray.copy())
                frame_counter += 1
                last_status = f"CALIBRATING BACKGROUND {frame_counter}/{WARMUP_FRAMES}"

                if frame_counter >= WARMUP_FRAMES:
                    background_gray = build_background(background_frames)
                    state = CameraState.WAITING_FOR_NOTE
                    last_status = "WAITING FOR NOTE - PLACE ITEM INSIDE"
                    print("Background calibrated. Ready for donations.")

                draw_overlay(frame, roi_coords, last_status, fps, 0.0, None, session_donations)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # --------------------------------------------------------
            # 2. CHAMBER STATUS DETECTION (Gated by black background subtraction)
            # --------------------------------------------------------
            chamber_status, presence_ratio, contour_ratio = detect_chamber_status(
                background_gray, roi_gray
            )
            presence_detected = (chamber_status == ChamberStatus.PRESENT)

            # --------------------------------------------------------
            # 3. STATE MACHINE DISPATCH
            # --------------------------------------------------------

            # A. WAIT FOR REMOVAL (Anti-Duplicate Counting Protection with Hand-Occlusion Immunity)
            if state == CameraState.WAIT_FOR_REMOVAL:
                if chamber_status == ChamberStatus.OCCLUDED:
                    last_status = "CHAMBER OCCLUDED - REMOVE HAND FROM BOX"
                    absence_streak = 0   # OCCLUDED must NEVER count as absence!
                elif chamber_status == ChamberStatus.UNKNOWN:
                    last_status = "STABILIZING CHAMBER..."
                    absence_streak = 0   # UNKNOWN must NEVER count as absence!
                elif chamber_status == ChamberStatus.PRESENT:
                    last_status = "PLEASE REMOVE ITEM FROM BOX"
                    absence_streak = 0   # Item still present
                elif chamber_status == ChamberStatus.ABSENT:
                    absence_streak += 1
                    last_status = f"CLEARING CHAMBER ({absence_streak}/{REQUIRED_REMOVAL_FRAMES})..."
                    if absence_streak >= REQUIRED_REMOVAL_FRAMES:
                        # 30 consecutive valid ABSENT frames confirmed: box is truly empty
                        state = CameraState.WAITING_FOR_NOTE
                        last_status = "WAITING FOR NOTE - PLACE ITEM INSIDE"
                        last_result = None
                        presence_streak = 0
                        absence_streak = 0
                        print("Box cleared and confirmed empty. Ready for next item.")

            # B. WAITING FOR NOTE
            elif state == CameraState.WAITING_FOR_NOTE:
                if chamber_status == ChamberStatus.PRESENT:
                    presence_streak += 1
                    state = CameraState.CONFIRMING_PRESENCE
                    last_status = f"DETECTING OBJECT ({presence_streak}/{REQUIRED_PRESENCE_FRAMES})..."
                elif chamber_status == ChamberStatus.OCCLUDED:
                    last_status = "CHAMBER OCCLUDED - KEEP CLEAR"
                    presence_streak = 0
                else:
                    last_status = "WAITING FOR NOTE - PLACE ITEM INSIDE"
                    presence_streak = 0

            # C. CONFIRMING TEMPORAL STABILITY
            elif state == CameraState.CONFIRMING_PRESENCE:
                if chamber_status != ChamberStatus.PRESENT:
                    state = CameraState.WAITING_FOR_NOTE
                    last_status = "WAITING FOR NOTE - PLACE ITEM INSIDE"
                    presence_streak = 0
                else:
                    presence_streak += 1
                    if presence_streak >= REQUIRED_PRESENCE_FRAMES:
                        # Object is settled and stationary -> ready for pipeline evaluation
                        state = CameraState.INSPECTING
                        last_status = "INSPECTING CANDIDATE WITH AI PIPELINE..."

            # D. INSPECTING WITH THREE-OUTCOME PIPELINE
            elif state == CameraState.INSPECTING:
                # Cooldown check
                if (now - last_eval_time) >= CLASSIFICATION_COOLDOWN:
                    last_eval_time = now

                    # Convert ROI crop to PIL Image and evaluate with exception safety
                    try:
                        roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(roi_rgb)
                        result = pipeline.predict(pil_img)
                    except Exception as exc:
                        print(f"Warning: inference exception ({exc}). Routing safely to REVIEW.")
                        result = PipelineResult(
                            classification=CLS_REVIEW,
                            operational_action=_ACT_REVIEW,
                            predicted_denomination=None,
                            presence_confidence=0.0,
                            denomination_confidence=0.0,
                            denomination_margin=0.0,
                            review_reason=f"Inference exception: {exc}",
                            count_authorized=False,
                            image_sharpness=0.0,
                            image_brightness=0.0,
                            latency_ms=0.0,
                        )
                    last_result = result

                    # ---- OUTCOME 1: ACCEPT_NOTE -> COUNT ----
                    if (
                        result.classification == CLS_ACCEPT
                        and result.count_authorized is True
                        and result.operational_action == _ACT_COUNT
                        and str(result.predicted_denomination) in SUPPORTED_DENOMINATIONS
                    ):
                        denom_name = DENOM_LABELS.get(result.predicted_denomination, result.predicted_denomination)
                        last_status = f"ACCEPTED: {denom_name} ({result.denomination_confidence*100:.1f}%)"
                        session_donations += 1

                        # Execute authorized counting workflow
                        execute_donation_acceptance(
                            result=result,
                            camera_index=args.camera,
                            log_path=log_path,
                            no_blockchain=args.no_blockchain,
                        )

                        # Enforce anti-duplicate lock: MUST remove note before another count can occur
                        state = CameraState.WAIT_FOR_REMOVAL

                        if not args.continuous:
                            # Show confirmation for 2.5 seconds then exit (single-note demo)
                            draw_overlay(frame, roi_coords, last_status, fps, presence_ratio, result, session_donations)
                            cv2.imshow(WINDOW_NAME, frame)
                            cv2.waitKey(2500)
                            print(f"\nDonation complete: {denom_name}. Exiting single-donation session.")
                            break

                    # ---- OUTCOME 2: REJECT_NOT_NOTE -> REVIEW ----
                    elif result.classification == CLS_REJECT:
                        last_status = "REJECTED: NOT A NOTE -> ROUTED TO REVIEW"
                        execute_review_routing(
                            result=result,
                            camera_index=args.camera,
                            review_log_path=review_log_path,
                            frame_bgr=roi_crop,
                        )
                        # Lock until item is removed so we don't log duplicate reviews
                        state = CameraState.WAIT_FOR_REMOVAL

                    # ---- OUTCOME 3: REVIEW -> REVIEW ----
                    else:
                        last_status = f"REVIEW: {result.review_reason or 'Uncertain input'}"
                        execute_review_routing(
                            result=result,
                            camera_index=args.camera,
                            review_log_path=review_log_path,
                            frame_bgr=roi_crop,
                        )
                        # Lock until item is removed or adjusted
                        state = CameraState.WAIT_FOR_REMOVAL

            # --------------------------------------------------------
            # 4. DRAW OVERLAY & REFRESH WINDOW
            # --------------------------------------------------------
            draw_overlay(frame, roi_coords, last_status, fps, presence_ratio, last_result, session_donations)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("Quit requested by operator.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession terminated. Total donations confirmed in session: {session_donations}")


if __name__ == "__main__":
    main()
