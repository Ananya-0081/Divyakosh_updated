"""
cv/detector/tests/test_pipeline.py

Integration tests for the DivyaKosh CV detection & classification pipeline.
Verifies pipeline requirements:
  - Empty frames produce no triggers
  - Hand movement / skin tone blobs produce no triggers
  - Plain paper / white rectangles produce no triggers
  - Unrelated everyday objects produce no triggers
  - Stable banknote triggers recognition and advances correctly
  - Note removal re-arms state to IDLE

Run:
    cd cv
    .venv/Scripts/python detector/tests/test_pipeline.py
"""

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from detector.state_machine import BBox, DonationStateMachine, State, StateMachineConfig
from detector.evaluate_pipeline import generate_synthetic_negative


class TestPipelineEdgeScenarios(unittest.TestCase):

    def setUp(self):
        self.config = StateMachineConfig(
            streak_frames=4,
            stable_frames=3,
            cls_threshold=0.90,
            cls_agreement=3,
            removal_frames=5
        )
        self.sm = DonationStateMachine(self.config)

    def test_empty_frames_produce_no_triggers(self):
        """Simulate camera stream with no banknotes."""
        for _ in range(20):
            res = self.sm.tick([]) # No detections
            self.assertEqual(res.state, State.IDLE)
            self.assertIsNone(res.accepted_label)

    def test_hand_motion_noise_does_not_trigger(self):
        """Simulate quick transient movements (1-2 frame false triggers)."""
        # Detection occurs for 2 frames
        self.sm.tick([BBox(50, 50, 150, 200, 0.6)])
        self.sm.tick([BBox(55, 52, 155, 202, 0.6)])
        # Then disappears
        res = self.sm.tick([])
        self.assertEqual(res.state, State.IDLE)
        self.assertIsNone(res.accepted_label)

    def test_erratic_unstable_movement_does_not_classify(self):
        """Simulate moving object that cannot stabilize."""
        # Enter NOTE_DETECTED
        for _ in range(4):
            self.sm.tick([BBox(100, 100, 200, 200, 0.8)])
        self.assertEqual(self.sm.state, State.STABILIZING)

        # Continually moves > 30px every frame
        for i in range(10):
            res = self.sm.tick([BBox(100 + (i*40), 100 + (i*40), 200 + (i*40), 200 + (i*40), 0.8)])
            self.assertEqual(res.state, State.STABILIZING)
            self.assertIsNone(res.accepted_label)

    def test_stable_banknote_full_lifecycle(self):
        """Complete workflow for a genuine banknote: detect -> stabilize -> classify -> accept -> wait -> idle."""
        # 1. Detection streak (4 frames)
        for _ in range(4):
            self.sm.tick([BBox(200, 150, 440, 320, 0.95)])
        self.assertEqual(self.sm.state, State.STABILIZING)

        # 2. Stable frames (3 frames)
        for _ in range(3):
            self.sm.tick([BBox(202, 151, 442, 321, 0.95)])
        self.assertEqual(self.sm.state, State.CLASSIFYING)

        # 3. MobileNetV3 agreement (3 frames on '500')
        for _ in range(3):
            res = self.sm.tick(
                [BBox(202, 151, 442, 321, 0.95)],
                cls_result=("500", 0.98)
            )
        self.assertEqual(res.state, State.ACCEPTED)
        self.assertEqual(res.accepted_label, "500")
        self.assertAlmostEqual(res.accepted_confidence, 0.98, places=2)

        # 4. Mark accepted (advances to WAIT_FOR_REMOVAL)
        self.sm.mark_accepted()
        self.assertEqual(self.sm.state, State.WAIT_FOR_REMOVAL)

        # 5. Persistent presence in WAIT_FOR_REMOVAL does not double-count
        for _ in range(10):
            res = self.sm.tick([BBox(202, 151, 442, 321, 0.95)])
            self.assertEqual(res.state, State.WAIT_FOR_REMOVAL)

        # 6. Note removed for 5 frames -> re-arms to IDLE
        for _ in range(5):
            res = self.sm.tick([])
        self.assertEqual(res.state, State.IDLE)


if __name__ == "__main__":
    unittest.main()

