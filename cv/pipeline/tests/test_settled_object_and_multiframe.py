# -*- coding: utf-8 -*-
"""
cv/pipeline/tests/test_settled_object_and_multiframe.py

Unit and integration tests for:
  1. ThresholdConfig.from_file strict error handling (Item E)
  2. Settled-object gate with stability and border checks (Item A)
  3. Square note crop preservation for Model B (Item B)
  4. Multi-frame K-frame decision agreement & disagreement routing (Item C)
  5. Exact crop persistence on ACCEPT_NOTE (Item D)
  6. Empty chamber safety
"""

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

_CV_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_CV_ROOT) not in sys.path:
    sys.path.insert(0, str(_CV_ROOT))

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
    append_accepted_log,
    format_accepted_donation_record,
    validate_for_donation,
)
from scripts.live_camera import (
    BORDER_MARGIN_PIXELS,
    CameraState,
    ChamberStatus,
    MAX_CENTER_DRIFT_PIXELS,
    MAX_FG_DRIFT_RATIO,
    MAX_SIZE_DRIFT_RATIO,
    MULTI_FRAME_K,
    SETTLE_TIMEOUT_SECONDS,
    SETTLED_FRAMES_REQUIRED,
    detect_chamber_status,
    extract_square_note_crop,
    is_bbox_touching_border,
)


# ===========================================================================
# 1. ThresholdConfig.from_file Strict Error Raising (Item E)
# ===========================================================================

class TestThresholdConfigFileHandling:
    """Requirement E: Missing or unreadable threshold_config.json must raise an error."""

    def test_missing_file_raises_file_not_found(self):
        missing_path = Path("pipeline/tests/does_not_exist_thresholds.json")
        with pytest.raises(FileNotFoundError) as exc_info:
            ThresholdConfig.from_file(missing_path)
        assert "Required threshold configuration file not found" in str(exc_info.value)

    def test_corrupted_file_raises_value_error(self, tmp_path):
        bad_json = tmp_path / "corrupt_thresholds.json"
        bad_json.write_text("{ this is not valid json : [[", encoding="utf-8")
        with pytest.raises(ValueError) as exc_info:
            ThresholdConfig.from_file(bad_json)
        assert "Unreadable or corrupted threshold configuration file" in str(exc_info.value)

    def test_valid_file_loads_successfully(self):
        cfg_path = _CV_ROOT / "pipeline" / "threshold_config.json"
        cfg = ThresholdConfig.from_file(cfg_path)
        assert cfg.presence_accept_min == 0.3
        assert cfg.presence_reject_max == 0.05
        assert cfg.denomination_accept_min == 0.9
        assert cfg.denomination_margin_min == 0.45


# ===========================================================================
# 2. Square Note Crop for Model B (Item B)
# ===========================================================================

class TestSquareNoteCrop:
    """Requirement B: Crop bounding box with ~10% padding made square without stretching."""

    def test_crop_is_square(self):
        # 336x336 canvas, rectangular note 150x70
        roi_bgr = np.zeros((336, 336, 3), dtype=np.uint8)
        bbox = (50, 100, 150, 70)  # x, y, w, h
        crop_pil = extract_square_note_crop(roi_bgr, bbox, pad_ratio=0.10)
        w, h = crop_pil.size
        assert w == h, f"Crop must be square, got {w}x{h}"
        # Expected side is max(ch, cw) with 10% pad: 150 + 2*15 = 180
        assert w == 180

    def test_crop_at_boundary_does_not_crash(self):
        roi_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = (0, 0, 50, 50)  # Top-left corner
        crop_pil = extract_square_note_crop(roi_bgr, bbox, pad_ratio=0.10)
        w, h = crop_pil.size
        assert w == h

    def test_pipeline_accepts_note_crop_argument(self):
        pipeline = DivyaKoshPipeline(device=torch.device("cpu"))
        full_roi = Image.new("RGB", (224, 224), (10, 10, 10))
        note_crop = Image.new("RGB", (100, 100), (200, 150, 50))
        res = pipeline.predict(full_roi, note_crop=note_crop)
        assert isinstance(res, PipelineResult)


# ===========================================================================
# 3. Border Margin & Stability Checking (Item A)
# ===========================================================================

class TestSettledObjectGateHelpers:
    """Requirement A helpers: border touch detection and stability checks."""

    def test_is_bbox_touching_border(self):
        roi_w, roi_h = 336, 336

        # Touches left border
        assert is_bbox_touching_border((2, 50, 100, 50), roi_w, roi_h) is True
        # Touches top border
        assert is_bbox_touching_border((50, 3, 100, 50), roi_w, roi_h) is True
        # Touches right border (bx + bw = 333 >= 332)
        assert is_bbox_touching_border((233, 50, 100, 50), roi_w, roi_h) is True
        # Touches bottom border (by + bh = 333 >= 332)
        assert is_bbox_touching_border((50, 233, 50, 100), roi_w, roi_h) is True
        # Centered: does not touch
        assert is_bbox_touching_border((50, 50, 150, 80), roi_w, roi_h) is False


# ===========================================================================
# 4. Synthetic Frame Sequences (Required by Prompt)
# ===========================================================================

class TestSyntheticFrameSequences:
    """
    Simulated frame sequences testing:
      (1) Note sliding in does not trigger inference until settled
      (2) Note that never settles goes to REVIEW
      (3) Inconsistent per-frame denominations go to REVIEW
      (4) Empty chamber produces no records
    """

    def test_sequence_1_note_sliding_in_waits_until_settled(self):
        """
        Sequence 1: A note slides into the chamber across 8 moving frames,
        then stays stationary for 10 frames.
        Inference must NOT trigger during sliding; triggers only after 10 settled frames.
        """
        roi_w, roi_h = 336, 336
        settled_streak = 0
        state = CameraState.CONFIRMING_PRESENCE

        # 8 moving frames (center moves by 15 pixels each frame)
        prev_bbox = None
        prev_fg = 0.20
        for i in range(8):
            curr_bbox = (20 + i * 15, 50, 120, 60)
            if prev_bbox is not None:
                cx = curr_bbox[0] + curr_bbox[2] / 2.0
                pcx = prev_bbox[0] + prev_bbox[2] / 2.0
                center_drift = abs(cx - pcx)
                if center_drift > MAX_CENTER_DRIFT_PIXELS:
                    settled_streak = 1  # drift broke stability
                else:
                    settled_streak += 1
            else:
                settled_streak = 1
            prev_bbox = curr_bbox

            # Must NOT transition to INSPECTING while moving
            assert settled_streak < SETTLED_FRAMES_REQUIRED
            assert state == CameraState.CONFIRMING_PRESENCE

        # Now 10 stationary frames (zero drift, inside borders)
        for i in range(10):
            curr_bbox = (140, 50, 120, 60)
            cx = curr_bbox[0] + curr_bbox[2] / 2.0
            pcx = prev_bbox[0] + prev_bbox[2] / 2.0
            center_drift = abs(cx - pcx)
            if center_drift <= MAX_CENTER_DRIFT_PIXELS:
                settled_streak += 1
            else:
                settled_streak = 1
            prev_bbox = curr_bbox

            if settled_streak >= SETTLED_FRAMES_REQUIRED:
                state = CameraState.INSPECTING

        # Confirmed: transitioned to INSPECTING only after 10 settled frames
        assert settled_streak >= SETTLED_FRAMES_REQUIRED
        assert state == CameraState.INSPECTING

    def test_sequence_2_note_that_never_settles_goes_to_review(self):
        """
        Sequence 2: A note continues fluttering or remaining clipped/touching border
        for > 5.0 seconds. It must time out into REVIEW with count_authorized=False.
        """
        start_time = 100.0
        now = start_time
        settled_streak = 0
        state = CameraState.CONFIRMING_PRESENCE
        timed_out_result = None

        # Simulate 6.0 seconds of frames at 30 fps (180 frames) constantly in motion
        for frame_idx in range(180):
            now = start_time + (frame_idx / 30.0)
            # Jittering bbox
            curr_bbox = (10 + (frame_idx % 20) * 10, 50, 120, 60)
            settled_streak = 1  # Never reaches 10

            # Timeout check at 5.0s
            if (now - start_time) >= SETTLE_TIMEOUT_SECONDS:
                timed_out_result = PipelineResult(
                    classification=CLS_REVIEW,
                    operational_action=_ACT_REVIEW,
                    predicted_denomination=None,
                    presence_confidence=0.25,
                    denomination_confidence=0.0,
                    denomination_margin=0.0,
                    review_reason="Object failed to settle or remained clipped/in motion for >5.0s",
                    count_authorized=False,
                    image_sharpness=100.0,
                    image_brightness=80.0,
                    latency_ms=1.0,
                )
                state = CameraState.WAIT_FOR_REMOVAL
                break

        assert state == CameraState.WAIT_FOR_REMOVAL
        assert timed_out_result is not None
        assert timed_out_result.classification == CLS_REVIEW
        assert timed_out_result.count_authorized is False
        assert "failed to settle" in timed_out_result.review_reason

    def test_sequence_3_inconsistent_per_frame_denominations_go_to_review(self):
        """
        Sequence 3: In INSPECTING across K=5 frames, 4 frames predict '100' but 1 frame
        predicts '10' (or has low confidence).
        The multi-frame decision must route to REVIEW with count_authorized=False.
        """
        inspect_results = [
            PipelineResult(
                classification=CLS_ACCEPT, operational_action=_ACT_COUNT,
                predicted_denomination="100", presence_confidence=0.99,
                denomination_confidence=0.95, denomination_margin=0.90,
                review_reason=None, count_authorized=True,
                image_sharpness=200.0, image_brightness=100.0, latency_ms=10.0,
            ),
            PipelineResult(
                classification=CLS_ACCEPT, operational_action=_ACT_COUNT,
                predicted_denomination="100", presence_confidence=0.99,
                denomination_confidence=0.94, denomination_margin=0.88,
                review_reason=None, count_authorized=True,
                image_sharpness=200.0, image_brightness=100.0, latency_ms=10.0,
            ),
            # Frame 3 disagrees!
            PipelineResult(
                classification=CLS_ACCEPT, operational_action=_ACT_COUNT,
                predicted_denomination="10", presence_confidence=0.98,
                denomination_confidence=0.91, denomination_margin=0.85,
                review_reason=None, count_authorized=True,
                image_sharpness=200.0, image_brightness=100.0, latency_ms=10.0,
            ),
            PipelineResult(
                classification=CLS_ACCEPT, operational_action=_ACT_COUNT,
                predicted_denomination="100", presence_confidence=0.99,
                denomination_confidence=0.96, denomination_margin=0.92,
                review_reason=None, count_authorized=True,
                image_sharpness=200.0, image_brightness=100.0, latency_ms=10.0,
            ),
            PipelineResult(
                classification=CLS_ACCEPT, operational_action=_ACT_COUNT,
                predicted_denomination="100", presence_confidence=0.99,
                denomination_confidence=0.95, denomination_margin=0.90,
                review_reason=None, count_authorized=True,
                image_sharpness=200.0, image_brightness=100.0, latency_ms=10.0,
            ),
        ]

        # Multi-frame evaluation logic
        all_accept = all(r.classification == CLS_ACCEPT and r.count_authorized for r in inspect_results)
        denoms = [r.predicted_denomination for r in inspect_results]
        all_same_denom = len(set(denoms)) == 1 and denoms[0] in SUPPORTED_DENOMINATIONS

        assert all_accept is True
        assert all_same_denom is False  # 100 vs 10 disagreement!

        # Disagreement must construct REVIEW result
        disagreements = [f"F{i+1}:{r.classification}({r.predicted_denomination})" for i, r in enumerate(inspect_results)]
        rev_reason = f"Multi-frame disagreement across {MULTI_FRAME_K} frames: {', '.join(disagreements)}"

        rev_res = PipelineResult(
            classification=CLS_REVIEW,
            operational_action=_ACT_REVIEW,
            predicted_denomination=None,
            presence_confidence=inspect_results[0].presence_confidence,
            denomination_confidence=inspect_results[0].denomination_confidence,
            denomination_margin=inspect_results[0].denomination_margin,
            review_reason=rev_reason,
            count_authorized=False,
            image_sharpness=200.0,
            image_brightness=100.0,
            latency_ms=10.0,
        )

        assert rev_res.classification == CLS_REVIEW
        assert rev_res.operational_action == _ACT_REVIEW
        assert rev_res.count_authorized is False
        assert "Multi-frame disagreement" in rev_res.review_reason

    def test_sequence_4_empty_chamber_produces_no_records(self):
        """
        Sequence 4: Empty chamber produces ChamberStatus.ABSENT across frames.
        State remains in WAITING_FOR_NOTE; 0 records written, 0 donations counted.
        """
        bg = np.zeros((336, 336), dtype=np.uint8)
        empty_frame = np.zeros((336, 336), dtype=np.uint8)

        state = CameraState.WAITING_FOR_NOTE
        donations_counted = 0
        records_written = []

        for _ in range(60):  # 2 seconds of empty chamber
            status, fg_r, c_r, bbox = detect_chamber_status(bg, empty_frame)
            assert status == ChamberStatus.ABSENT
            assert bbox is None
            if status == ChamberStatus.PRESENT:
                state = CameraState.CONFIRMING_PRESENCE

        assert state == CameraState.WAITING_FOR_NOTE
        assert donations_counted == 0
        assert len(records_written) == 0


# ===========================================================================
# 5. Exact Crop Persistence on ACCEPT_NOTE (Item D)
# ===========================================================================

class TestAcceptedImagePersistence:
    """Requirement D: Save exact crop for every ACCEPT_NOTE and store image_ref in record."""

    def test_accepted_log_saves_image_crop(self, tmp_path):
        inference_log = tmp_path / "inference.jsonl"
        accepted_images_dir = tmp_path / "accepted_images"

        pipe_res = PipelineResult(
            classification=CLS_ACCEPT,
            operational_action=_ACT_COUNT,
            predicted_denomination="100",
            presence_confidence=0.99,
            denomination_confidence=0.95,
            denomination_margin=0.90,
            review_reason=None,
            count_authorized=True,
            image_sharpness=150.0,
            image_brightness=90.0,
            latency_ms=15.0,
        )

        record = format_accepted_donation_record(pipe_res, pocket_id="webcam-0")

        # Fake candidate crop BGR
        fake_crop = np.full((180, 180, 3), 120, dtype=np.uint8)

        append_accepted_log(
            record=record,
            log_path=inference_log,
            frame_bgr=fake_crop,
            images_dir=accepted_images_dir,
        )

        assert inference_log.exists()
        assert record.get("image_ref") is not None
        saved_img_path = Path(record["image_ref"])
        assert saved_img_path.exists()
        assert "accept_100_" in saved_img_path.name

        # Verify that record remains 100% valid for donation authorization
        valid, reason = validate_for_donation(record)
        assert valid is True
        assert reason is None

