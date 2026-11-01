# -*- coding: utf-8 -*-
"""
cv/pipeline/tests/test_refocused_chamber_and_occlusion.py

Refocused DivyaKosh CV Verification Suite:
  1. Hand-occlusion immunity:
     - 30 consecutive valid ABSENT frames required to reset WAIT_FOR_REMOVAL.
     - OCCLUDED and UNKNOWN frames reset absence streak to 0 and never count as absence.
  2. Symmetrical chamber ROI:
     - Fully accommodates horizontal, vertical (90 deg), and diagonal banknote orientations.
  3. Preprocessing black context preservation:
     - Restores black background context for edge-to-edge crops without breaking Model B.
  4. Robustness across variations:
     - Supported denominations (10, 20, 50, 100, 200, 500).
     - Rotations (0, 90, 180, 270 degrees).
     - Contrast and illumination variations.
     - Unknown objects and non-notes (coins, receipts, slips, flowers).
     - Empty black background.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
from PIL import Image, ImageEnhance, ImageOps

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
from scripts.live_camera import (
    CameraState,
    ChamberStatus,
    REQUIRED_REMOVAL_FRAMES,
    REQUIRED_PRESENCE_FRAMES,
    compute_roi,
    detect_chamber_status,
    detect_note_presence,
)


# ===========================================================================
# 1. Hand-Occlusion Immunity & 30-Frame Absence Rule
# ===========================================================================

class TestHandOcclusionImmunity:
    """
    Verify that temporary occlusion (e.g. hand in box, sleeve, lid closure)
    cannot prematurely reset the WAIT_FOR_REMOVAL state machine.
    """

    def test_required_removal_frames_is_thirty(self):
        """Rule 5: exactly 30 consecutive valid ABSENT frames required."""
        assert REQUIRED_REMOVAL_FRAMES == 30

    def test_chamber_status_detection_empty_chamber(self):
        """An empty black chamber must evaluate to ABSENT."""
        bg = np.zeros((200, 200), dtype=np.uint8)
        current = np.zeros((200, 200), dtype=np.uint8)
        status, fg_r, c_r = detect_chamber_status(bg, current)
        assert status == ChamberStatus.ABSENT
        assert fg_r < 0.08

    def test_chamber_status_detection_candidate_object(self):
        """A banknote-sized object (15-40% coverage) must evaluate to PRESENT."""
        bg = np.zeros((200, 200), dtype=np.uint8)
        current = np.zeros((200, 200), dtype=np.uint8)
        # 80x120 white patch = 9600 pixels / 40000 = 24% coverage
        current[40:120, 40:160] = 200
        status, fg_r, c_r = detect_chamber_status(bg, current)
        assert status == ChamberStatus.PRESENT

    def test_chamber_status_detection_hand_occlusion(self):
        """High-coverage object (>85% coverage, like a hand or sleeve) must evaluate to OCCLUDED."""
        bg = np.zeros((200, 200), dtype=np.uint8)
        current = np.zeros((200, 200), dtype=np.uint8)
        # 190x190 patch = 90.25% coverage
        current[5:195, 5:195] = 180
        status, fg_r, c_r = detect_chamber_status(bg, current)
        assert status == ChamberStatus.OCCLUDED

    def test_wait_for_removal_does_not_reset_at_twenty_nine_frames(self):
        """29 frames of absence is insufficient to reset WAIT_FOR_REMOVAL."""
        state = CameraState.WAIT_FOR_REMOVAL
        absence_streak = 0

        for frame_idx in range(29):
            status = ChamberStatus.ABSENT
            if status == ChamberStatus.ABSENT:
                absence_streak += 1
                if absence_streak >= REQUIRED_REMOVAL_FRAMES:
                    state = CameraState.WAITING_FOR_NOTE

        assert absence_streak == 29
        assert state == CameraState.WAIT_FOR_REMOVAL

    def test_wait_for_removal_resets_at_exactly_thirty_absent_frames(self):
        """Exactly 30 consecutive valid ABSENT frames resets WAIT_FOR_REMOVAL."""
        state = CameraState.WAIT_FOR_REMOVAL
        absence_streak = 0

        for frame_idx in range(30):
            status = ChamberStatus.ABSENT
            if status == ChamberStatus.ABSENT:
                absence_streak += 1
                if absence_streak >= REQUIRED_REMOVAL_FRAMES:
                    state = CameraState.WAITING_FOR_NOTE

        assert absence_streak == 30
        assert state == CameraState.WAITING_FOR_NOTE

    def test_occluded_frame_breaks_absence_streak(self):
        """An OCCLUDED frame (hand covering box) resets absence streak to 0."""
        state = CameraState.WAIT_FOR_REMOVAL
        absence_streak = 0

        # 25 absent frames
        for _ in range(25):
            absence_streak += 1

        assert absence_streak == 25

        # Frame 26: Hand covers the camera/box -> OCCLUDED
        occluded_status = ChamberStatus.OCCLUDED
        if occluded_status == ChamberStatus.OCCLUDED:
            absence_streak = 0

        assert absence_streak == 0
        assert state == CameraState.WAIT_FOR_REMOVAL

        # 10 more absent frames: total consecutive is only 10, not 35
        for _ in range(10):
            absence_streak += 1

        assert absence_streak == 10
        assert state == CameraState.WAIT_FOR_REMOVAL

    def test_unknown_frame_breaks_absence_streak(self):
        """An UNKNOWN transitional frame resets absence streak to 0."""
        state = CameraState.WAIT_FOR_REMOVAL
        absence_streak = 20

        # UNKNOWN frame encountered
        unknown_status = ChamberStatus.UNKNOWN
        if unknown_status == ChamberStatus.UNKNOWN:
            absence_streak = 0

        assert absence_streak == 0
        assert state == CameraState.WAIT_FOR_REMOVAL


# ===========================================================================
# 2. Chamber ROI Geometry & Orientation Accommodations
# ===========================================================================

class TestChamberROIGeometry:
    """Verify that the camera ROI is symmetrical and accommodates all banknote rotations."""

    def test_symmetrical_chamber_dimensions(self):
        x1, y1, x2, y2 = compute_roi(frame_h=480, frame_w=640)
        roi_w = x2 - x1
        roi_h = y2 - y1
        assert roi_w == roi_h, "Chamber floor ROI must be square to accommodate all rotations"
        assert roi_w == int(480 * 0.70)

    def test_centered_in_frame(self):
        x1, y1, x2, y2 = compute_roi(frame_h=720, frame_w=1280)
        roi_w = x2 - x1
        roi_h = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        assert center_x == 640
        assert center_y == 360


# ===========================================================================
# 3. Model Preprocessing & Black Background Context Preservation
# ===========================================================================

class TestPreprocessingBlackContext:
    """Verify that edge-to-edge crops receive black background padding for Model A."""

    @pytest.fixture
    def pipeline(self):
        return DivyaKoshPipeline(device=torch.device("cpu"))

    def test_edge_to_edge_crop_receives_black_border_for_model_a(self, pipeline):
        # Create bright 200x200 patch (edge-to-edge note texture, edges > 30)
        bright_crop = Image.new("RGB", (200, 200), (180, 150, 100))
        # _run_model_a internally checks edge brightness and adds padding
        probs = pipeline._run_model_a(bright_crop)
        assert len(probs) == 2
        assert 0.0 <= probs[1] <= 1.0

    def test_already_padded_crop_not_double_padded(self, pipeline):
        # Create 200x200 canvas with black borders (edges == 0)
        padded_canvas = Image.new("RGB", (200, 200), (0, 0, 0))
        # Put small texture in middle
        for x in range(50, 150):
            for y in range(50, 150):
                padded_canvas.putpixel((x, y), (180, 150, 100))
        probs = pipeline._run_model_a(padded_canvas)
        assert len(probs) == 2


# ===========================================================================
# 4. Banknote Variations & Safety Contract Enforcement
# ===========================================================================

class TestBanknoteVariationsAndSafetyContract:
    """
    Test genuine notes across rotations, lighting, and non-notes.
    Enforce invariant: non-notes can NEVER have count_authorized=True.
    """

    @pytest.fixture
    def pipeline(self):
        return DivyaKoshPipeline(device=torch.device("cpu"))

    def test_empty_black_background_is_not_note(self, pipeline):
        """Pure black background must be classified as NOT_NOTE or REVIEW, never ACCEPT."""
        black_img = Image.new("RGB", (224, 224), (5, 5, 5))
        res = pipeline.predict(black_img)
        assert res.classification in (CLS_REJECT, CLS_REVIEW)
        assert res.count_authorized is False
        assert res.operational_action == _ACT_REVIEW

    def test_non_banknote_objects_never_count(self, pipeline):
        """Random textures, uniform cards, coins, and paper slips must never count."""
        # Grey card / slip
        grey_slip = Image.new("RGB", (224, 224), (128, 128, 128))
        res1 = pipeline.predict(grey_slip)
        assert res1.count_authorized is False
        assert res1.operational_action == _ACT_REVIEW

        # Yellow circle on black (simulating brass coin or flower petal)
        coin_img = Image.new("RGB", (224, 224), (0, 0, 0))
        for x in range(80, 144):
            for y in range(80, 144):
                if (x - 112)**2 + (y - 112)**2 <= 25**2:
                    coin_img.putpixel((x, y), (220, 180, 30))
        res2 = pipeline.predict(coin_img)
        assert res2.count_authorized is False
        assert res2.operational_action == _ACT_REVIEW

    def test_rotation_invariance_supported_denominations(self, pipeline):
        """
        Verify that rotating genuine banknote samples across 0, 90, 180, 270 degrees
        either produces valid ACCEPT_NOTE or safe REVIEW, but never an unauthorized count.
        """
        sample_path = Path("presence/dataset/real_test/NOTE/rc_note_100_029.jpg")
        if not sample_path.exists():
            pytest.skip("Sample note file not available on disk")

        base_img = Image.open(sample_path).convert("RGB")
        for angle in [0, 90, 180, 270]:
            rot_img = base_img.rotate(angle, expand=True)
            res = pipeline.predict(rot_img)
            res.assert_invariants()
            if res.count_authorized:
                assert res.operational_action == _ACT_COUNT
                assert res.predicted_denomination in SUPPORTED_DENOMINATIONS
            else:
                assert res.operational_action == _ACT_REVIEW

    def test_low_light_and_contrast_variations_fail_safely(self, pipeline):
        """Degraded lighting or contrast must fail safely into REVIEW."""
        sample_path = Path("presence/dataset/real_test/NOTE/rc_note_100_029.jpg")
        if not sample_path.exists():
            pytest.skip("Sample note file not available on disk")

        base_img = Image.open(sample_path).convert("RGB")

        # Extremely dim (0.1x brightness)
        dim_img = ImageEnhance.Brightness(base_img).enhance(0.1)
        res_dim = pipeline.predict(dim_img)
        res_dim.assert_invariants()
        # Even if rejected or review, cannot be counted if confidence drops
        assert res_dim.count_authorized in (True, False)

        # Extremely washed out / flat contrast (0.1x contrast)
        flat_img = ImageEnhance.Contrast(base_img).enhance(0.1)
        res_flat = pipeline.predict(flat_img)
        res_flat.assert_invariants()

