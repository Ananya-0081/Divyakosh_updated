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
    compute_brightness,
    compute_laplacian_variance,
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

# Settled-object & multi-frame confirmation parameters (Item A, B, C)
SETTLED_FRAMES_REQUIRED = 10     # N=10 consecutive settled frames before inference starts
SETTLE_TIMEOUT_SECONDS = 5.0      # Maximum wait time to settle before routing to REVIEW
MAX_CENTER_DRIFT_PIXELS = 8.0     # Max center movement (pixels) between frames
MAX_SIZE_DRIFT_RATIO = 0.15       # Max width/height drift ratio (15%)
MAX_FG_DRIFT_RATIO = 0.05         # Max foreground ratio drift (5%)
BORDER_MARGIN_PIXELS = 4          # If bbox touches border (within 4px), note is clipped / entering
MULTI_FRAME_K = 5                 # K=5 settled frames required for unanimous decision agreement


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


def extract_square_note_crop(
    roi_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pad_ratio: float = 0.10,
) -> Image.Image:
    """
    Crop the settled bounding box with ~10% padding, made square without
    stretching the aspect ratio. Centered on a black background.
    """
    bx, by, bw, bh = bbox
    roi_h, roi_w = roi_bgr.shape[:2]

    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    x1 = max(0, bx - pad_w)
    y1 = max(0, by - pad_h)
    x2 = min(roi_w, bx + bw + pad_w)
    y2 = min(roi_h, by + bh + pad_h)

    crop = roi_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if ch <= 0 or cw <= 0:
        return Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))

    # Pad to square without stretching aspect ratio
    side = max(ch, cw)
    square_img = np.zeros((side, side, 3), dtype=np.uint8)
    offset_x = (side - cw) // 2
    offset_y = (side - ch) // 2
    square_img[offset_y:offset_y + ch, offset_x:offset_x + cw] = crop

    return Image.fromarray(cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB))


def is_bbox_touching_border(
    bbox: Tuple[int, int, int, int],
    roi_w: int,
    roi_h: int,
    margin: int = BORDER_MARGIN_PIXELS,
) -> bool:
    """Check if the bounding box touches or is near the ROI border (entering or clipped)."""
    bx, by, bw, bh = bbox
    return (
        bx <= margin
        or by <= margin
        or (bx + bw) >= (roi_w - margin)
        or (by + bh) >= (roi_h - margin)
    )


def detect_chamber_status(
    background_gray: np.ndarray,
    current_gray: np.ndarray,
) -> Tuple[ChamberStatus, float, float, Optional[Tuple[int, int, int, int]]]:
    """
    Background subtraction gate to detect chamber status against the black background.
    Returns (ChamberStatus, foreground_ratio, contour_ratio, largest_bbox).

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
    largest_bbox: Optional[Tuple[int, int, int, int]] = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_bbox = cv2.boundingRect(contour)

    contour_ratio = largest_area / float(roi_area)

    # 1. Occlusion check: hand, lid, or large coverage covering >85% of chamber
    if contour_ratio > MAX_CONTOUR_RATIO or foreground_ratio > MAX_CONTOUR_RATIO:
        return ChamberStatus.OCCLUDED, foreground_ratio, contour_ratio, largest_bbox

    # 2. Present check: candidate object inside chamber
    if (
        foreground_ratio >= PRESENCE_RATIO_THRESHOLD
        and MIN_CONTOUR_RATIO <= contour_ratio <= MAX_CONTOUR_RATIO
    ):
        return ChamberStatus.PRESENT, foreground_ratio, contour_ratio, largest_bbox

    # 3. Absent check: clean empty black chamber
    if foreground_ratio < PRESENCE_RATIO_THRESHOLD and contour_ratio < MIN_CONTOUR_RATIO:
        return ChamberStatus.ABSENT, foreground_ratio, contour_ratio, largest_bbox

    # 4. Unknown / transitional
    return ChamberStatus.UNKNOWN, foreground_ratio, contour_ratio, largest_bbox


def detect_note_presence(
    background_gray: np.ndarray,
    current_gray: np.ndarray,
) -> Tuple[bool, float, float]:
    """Backward-compatible wrapper around detect_chamber_status."""
    status, fg_r, c_r, _ = detect_chamber_status(background_gray, current_gray)
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
    frame_bgr: Optional[np.ndarray] = None,
) -> dict:
    """
    Execute the authorized donation counting workflow.
    Guaranteed to run ONLY when count_authorized is True.
    Saves the exact candidate crop to logs/accepted_images/ and stores image_ref in record.
    """
    pocket_id = f"webcam-{camera_index}"
    record = format_accepted_donation_record(result, pocket_id=pocket_id)

    # 1. Main JSONL inference audit log with exact crop image
    append_accepted_log(record, log_path=log_path, frame_bgr=frame_bgr)
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

    # Settled-object & multi-frame confirmation state (Item A, B, C)
    settled_streak = 0
    prev_settled_bbox: Optional[Tuple[int, int, int, int]] = None
    prev_fg_ratio = 0.0
    first_present_time: Optional[float] = None
    inspect_frame_results: list[PipelineResult] = []
    inspect_frame_crops: list[np.ndarray] = []

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
            chamber_status, presence_ratio, contour_ratio, current_bbox = detect_chamber_status(
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
                        settled_streak = 0
                        first_present_time = None
                        inspect_frame_results = []
                        inspect_frame_crops = []
                        print("Box cleared and confirmed empty. Ready for next item.")

            # B. WAITING FOR NOTE
            elif state == CameraState.WAITING_FOR_NOTE:
                if chamber_status == ChamberStatus.PRESENT and current_bbox is not None:
                    state = CameraState.CONFIRMING_PRESENCE
                    first_present_time = now
                    settled_streak = 0
                    prev_settled_bbox = current_bbox
                    prev_fg_ratio = presence_ratio
                    last_status = "DETECTING CANDIDATE NOTE..."
                elif chamber_status == ChamberStatus.OCCLUDED:
                    last_status = "CHAMBER OCCLUDED - KEEP CLEAR"
                    settled_streak = 0
                    first_present_time = None
                else:
                    last_status = "WAITING FOR NOTE - PLACE ITEM INSIDE"
                    settled_streak = 0
                    first_present_time = None

            # C. CONFIRMING TEMPORAL STABILITY (Settled-object gate: Item A)
            elif state == CameraState.CONFIRMING_PRESENCE:
                if chamber_status != ChamberStatus.PRESENT or current_bbox is None:
                    state = CameraState.WAITING_FOR_NOTE
                    last_status = "WAITING FOR NOTE - PLACE ITEM INSIDE"
                    settled_streak = 0
                    first_present_time = None
                else:
                    # 1. Border check (treat as entering or clipped if touching border)
                    roi_h, roi_w = roi_crop.shape[:2]
                    _dbg_frame = int(now * 30)
                    if is_bbox_touching_border(current_bbox, roi_w, roi_h):
                        settled_streak = 0
                        last_status = "NOTE ENTERING OR CLIPPED - WAITING TO SETTLE..."
                        if _dbg_frame % 15 == 0:
                            print(f"[SETTLE] touching ROI border bbox={current_bbox} roi={roi_w}x{roi_h}")
                    else:
                        # 2. Stability check against previous frame
                        if prev_settled_bbox is not None:
                            cx = current_bbox[0] + current_bbox[2] / 2.0
                            cy = current_bbox[1] + current_bbox[3] / 2.0
                            pcx = prev_settled_bbox[0] + prev_settled_bbox[2] / 2.0
                            pcy = prev_settled_bbox[1] + prev_settled_bbox[3] / 2.0
                            center_drift = float(np.hypot(cx - pcx, cy - pcy))
                            w_drift = abs(current_bbox[2] - prev_settled_bbox[2]) / max(prev_settled_bbox[2], 1)
                            h_drift = abs(current_bbox[3] - prev_settled_bbox[3]) / max(prev_settled_bbox[3], 1)
                            fg_drift = abs(presence_ratio - prev_fg_ratio)

                            if (
                                center_drift <= MAX_CENTER_DRIFT_PIXELS
                                and w_drift <= MAX_SIZE_DRIFT_RATIO
                                and h_drift <= MAX_SIZE_DRIFT_RATIO
                                and fg_drift <= MAX_FG_DRIFT_RATIO
                            ):
                                settled_streak += 1
                            else:
                                settled_streak = 1
                                if _dbg_frame % 15 == 0:
                                    print(f"[SETTLE] moving: center_drift={center_drift:.1f}px (max {MAX_CENTER_DRIFT_PIXELS}) "
                                          f"w_drift={w_drift:.2f} h_drift={h_drift:.2f} (max {MAX_SIZE_DRIFT_RATIO}) "
                                          f"fg_drift={fg_drift:.3f} (max {MAX_FG_DRIFT_RATIO}) bbox={current_bbox}")
                        else:
                            settled_streak = 1

                        prev_settled_bbox = current_bbox
                        prev_fg_ratio = presence_ratio
                        last_status = f"SETTLING OBJECT ({settled_streak}/{SETTLED_FRAMES_REQUIRED})..."

                        if settled_streak >= SETTLED_FRAMES_REQUIRED:
                            state = CameraState.INSPECTING
                            inspect_frame_results = []
                            inspect_frame_crops = []
                            last_status = "OBJECT SETTLED - RUNNING MULTI-FRAME AI INSPECTION..."

                    # 3. Timeout check (5 seconds max before routing to REVIEW)
                    if first_present_time is not None and (now - first_present_time) >= SETTLE_TIMEOUT_SECONDS:
                        print("\n[TIMEOUT] Object failed to settle within 5.0s -> routing to REVIEW")
                        timeout_reason = "Object failed to settle or remained clipped/in motion for >5.0s"
                        pil_roi = Image.fromarray(cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB))
                        res_timeout = PipelineResult(
                            classification=CLS_REVIEW,
                            operational_action=_ACT_REVIEW,
                            predicted_denomination=None,
                            presence_confidence=round(presence_ratio, 4),
                            denomination_confidence=0.0,
                            denomination_margin=0.0,
                            review_reason=timeout_reason,
                            count_authorized=False,
                            image_sharpness=round(compute_laplacian_variance(pil_roi), 2),
                            image_brightness=round(compute_brightness(pil_roi), 2),
                            latency_ms=0.0,
                        )
                        last_result = res_timeout
                        execute_review_routing(
                            result=res_timeout,
                            camera_index=args.camera,
                            review_log_path=review_log_path,
                            frame_bgr=roi_crop,
                        )
                        state = CameraState.WAIT_FOR_REMOVAL
                        settled_streak = 0
                        first_present_time = None

            # D. INSPECTING WITH MULTI-FRAME AGREEMENT (Items B & C)
            elif state == CameraState.INSPECTING:
                if chamber_status != ChamberStatus.PRESENT or current_bbox is None:
                    # Object removed or disturbed during multi-frame inspection
                    state = CameraState.WAITING_FOR_NOTE
                    inspect_frame_results = []
                    inspect_frame_crops = []
                    settled_streak = 0
                    first_present_time = None
                else:
                    # Model A receives full ROI; Model B receives square ~10% padded note crop
                    pil_roi = Image.fromarray(cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB))
                    pil_note_crop = extract_square_note_crop(roi_crop, current_bbox)

                    try:
                        frame_res = pipeline.predict(pil_roi, note_crop=pil_note_crop)
                    except Exception as exc:
                        print(f"Warning: inference exception ({exc}). Routing safely to REVIEW.")
                        frame_res = PipelineResult(
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

                    inspect_frame_results.append(frame_res)
                    inspect_frame_crops.append(roi_crop.copy())
                    last_status = f"MULTI-FRAME INSPECTION ({len(inspect_frame_results)}/{MULTI_FRAME_K})..."

                    if len(inspect_frame_results) >= MULTI_FRAME_K:
                        all_accept = all(
                            r.classification == CLS_ACCEPT and r.count_authorized for r in inspect_frame_results
                        )
                        denoms = [r.predicted_denomination for r in inspect_frame_results]
                        all_same_denom = (
                            len(set(denoms)) == 1
                            and denoms[0] is not None
                            and str(denoms[0]) in SUPPORTED_DENOMINATIONS
                        )

                        best_crop_bgr = inspect_frame_crops[-1]

                        # Outcome 1: Every frame is ACCEPT_NOTE with unanimous denomination
                        if all_accept and all_same_denom:
                            final_res = inspect_frame_results[-1]
                            mean_conf = float(np.mean([r.denomination_confidence for r in inspect_frame_results]))
                            denom_name = DENOM_LABELS.get(final_res.predicted_denomination, final_res.predicted_denomination)
                            last_status = f"ACCEPTED: {denom_name} ({mean_conf*100:.1f}%)"
                            session_donations += 1
                            last_result = final_res

                            print(f"\n[MULTI-FRAME AGREED] {MULTI_FRAME_K}/{MULTI_FRAME_K} frames confirmed {denom_name} (mean conf: {mean_conf:.4f})")

                            # Execute authorized counting and SAVE EXACT CROP
                            execute_donation_acceptance(
                                result=final_res,
                                camera_index=args.camera,
                                log_path=log_path,
                                no_blockchain=args.no_blockchain,
                                frame_bgr=best_crop_bgr,
                            )

                            state = CameraState.WAIT_FOR_REMOVAL

                            if not args.continuous:
                                draw_overlay(frame, roi_coords, last_status, fps, presence_ratio, final_res, session_donations)
                                cv2.imshow(WINDOW_NAME, frame)
                                cv2.waitKey(2500)
                                print(f"\nDonation complete: {denom_name}. Exiting single-donation session.")
                                break

                        # Outcome 2: Any disagreement or non-accept -> route to REVIEW
                        else:
                            frame_details = []
                            for idx, r in enumerate(inspect_frame_results):
                                d_str = r.predicted_denomination or "None"
                                frame_details.append(f"F{idx+1}:{r.classification}({d_str}, conf={r.denomination_confidence:.2f})")
                            disagreement_reason = f"Multi-frame disagreement across {MULTI_FRAME_K} frames: {', '.join(frame_details)}"
                            print(f"\n[MULTI-FRAME DISAGREEMENT] {disagreement_reason}")

                            first_non_accept = next(
                                (r for r in inspect_frame_results if r.classification != CLS_ACCEPT),
                                inspect_frame_results[0],
                            )

                            rev_res = PipelineResult(
                                classification=CLS_REVIEW,
                                operational_action=_ACT_REVIEW,
                                predicted_denomination=None,
                                presence_confidence=first_non_accept.presence_confidence,
                                denomination_confidence=first_non_accept.denomination_confidence,
                                denomination_margin=first_non_accept.denomination_margin,
                                review_reason=disagreement_reason,
                                count_authorized=False,
                                image_sharpness=first_non_accept.image_sharpness,
                                image_brightness=first_non_accept.image_brightness,
                                latency_ms=first_non_accept.latency_ms,
                                model_a_probs=first_non_accept.model_a_probs,
                                model_b_probs=first_non_accept.model_b_probs,
                            )
                            last_result = rev_res
                            last_status = "REVIEW: Multi-frame disagreement"

                            execute_review_routing(
                                result=rev_res,
                                camera_index=args.camera,
                                review_log_path=review_log_path,
                                frame_bgr=best_crop_bgr,
                            )
                            state = CameraState.WAIT_FOR_REMOVAL

                        inspect_frame_results = []
                        inspect_frame_crops = []
                        settled_streak = 0
                        first_present_time = None

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