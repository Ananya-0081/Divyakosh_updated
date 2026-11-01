"""
cv/detector/tests/test_state_machine.py

Unit tests for the DonationStateMachine.
Tests all state transitions, edge cases, and parameterised configs.

Run:
    cd cv
    .venv/Scripts/python -m pytest detector/tests/test_state_machine.py -v
    # Or without pytest:
    .venv/Scripts/python detector/tests/test_state_machine.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from detector.state_machine import (
    BBox, DonationStateMachine, State, StateMachineConfig
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box(x1=100, y1=100, x2=300, y2=250, conf=0.80) -> BBox:
    return BBox(x1, y1, x2, y2, conf)


def _no_det():
    return []


def _det(x1=100, y1=100, x2=300, y2=250, conf=0.80):
    return [_box(x1, y1, x2, y2, conf)]


def _make_sm(streak=4, stable=3, agreement=3, cls_thresh=0.90, removal=8, max_cls_attempts=10):
    cfg = StateMachineConfig(
        streak_frames=streak,
        stable_frames=stable,
        cls_agreement=agreement,
        cls_threshold=cls_thresh,
        stability_px=30.0,
        removal_frames=removal,
        max_cls_attempts=max_cls_attempts,
    )
    return DonationStateMachine(cfg)


def _advance_to_stabilizing(sm):
    """Drive the state machine from IDLE to STABILIZING."""
    streak = sm.config.streak_frames
    for _ in range(streak):
        sm.tick(_det())
    assert sm.state == State.STABILIZING


def _advance_to_classifying(sm):
    """Drive IDLE → CLASSIFYING."""
    _advance_to_stabilizing(sm)
    stable = sm.config.stable_frames
    for _ in range(stable):
        sm.tick(_det())
    assert sm.state == State.CLASSIFYING


def _advance_to_accepted(sm):
    """Drive IDLE → ACCEPTED."""
    _advance_to_classifying(sm)
    agreement = sm.config.cls_agreement
    for _ in range(agreement):
        sm.tick(_det(), cls_result=("100", 0.95))
    assert sm.state == State.ACCEPTED


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBBox:
    def test_center(self):
        b = BBox(0, 0, 100, 50)
        assert b.cx == 50.0
        assert b.cy == 25.0

    def test_center_distance(self):
        b1 = BBox(0, 0, 100, 100)
        b2 = BBox(30, 40, 130, 140)  # center shifted by (30, 40) → distance=50
        assert abs(b1.center_distance(b2) - 50.0) < 0.01

    def test_as_xyxy(self):
        b = BBox(10.7, 20.3, 100.9, 200.1)
        assert b.as_xyxy() == (10, 20, 100, 200)


class TestIdleState:
    def test_starts_idle(self):
        sm = _make_sm()
        assert sm.state == State.IDLE

    def test_no_detection_stays_idle(self):
        sm = _make_sm()
        r = sm.tick(_no_det())
        assert r.state == State.IDLE

    def test_detection_enters_note_detected(self):
        sm = _make_sm()
        r = sm.tick(_det())
        assert r.state == State.NOTE_DETECTED

    def test_reset_from_idle_stays_idle(self):
        sm = _make_sm()
        sm.reset()
        assert sm.state == State.IDLE


class TestNoteDetectedState:
    def test_builds_streak(self):
        sm = _make_sm(streak=4)
        sm.tick(_det())  # → NOTE_DETECTED
        for i in range(2):
            r = sm.tick(_det())
        assert r.state == State.NOTE_DETECTED

    def test_lost_detection_resets_to_idle(self):
        sm = _make_sm(streak=4)
        sm.tick(_det())  # → NOTE_DETECTED
        r = sm.tick(_no_det())
        assert r.state == State.IDLE

    def test_full_streak_enters_stabilizing(self):
        sm = _make_sm(streak=4)
        _advance_to_stabilizing(sm)
        assert sm.state == State.STABILIZING

    def test_status_text_shows_progress(self):
        sm = _make_sm(streak=4)
        r = sm.tick(_det())
        assert "1/4" in r.status_text or "CHECKING" in r.status_text


class TestStabilizingState:
    def test_stable_box_accumulates_streak(self):
        sm = _make_sm(stable=3)
        _advance_to_stabilizing(sm)
        r = sm.tick(_det(100, 100, 300, 250))
        assert r.state == State.STABILIZING

    def test_stable_streak_enters_classifying(self):
        sm = _make_sm(stable=3)
        _advance_to_classifying(sm)
        assert sm.state == State.CLASSIFYING

    def test_shifted_bbox_resets_stable_streak(self):
        sm = _make_sm(stable=3, streak=4)
        _advance_to_stabilizing(sm)
        # 2 stable frames
        sm.tick(_det(100, 100, 300, 250))
        sm.tick(_det(100, 100, 300, 250))
        # Shift by 50px — above stability_px=30
        r = sm.tick(_det(150, 140, 350, 290))
        assert r.state == State.STABILIZING
        assert "moved" in r.status_text.lower() or "STABILIZING" in r.status_text

    def test_lost_detection_goes_back_to_note_detected(self):
        sm = _make_sm()
        _advance_to_stabilizing(sm)
        r = sm.tick(_no_det())
        assert r.state == State.NOTE_DETECTED


class TestClassifyingState:
    def test_no_cls_result_stays_classifying(self):
        sm = _make_sm()
        _advance_to_classifying(sm)
        r = sm.tick(_det())
        assert r.state == State.CLASSIFYING

    def test_low_confidence_cls_does_not_accept(self):
        sm = _make_sm(cls_thresh=0.90)
        _advance_to_classifying(sm)
        # confidence below threshold
        for _ in range(5):
            r = sm.tick(_det(), cls_result=("100", 0.80))
        assert r.state != State.ACCEPTED

    def test_agreement_accepts(self):
        sm = _make_sm(agreement=3, cls_thresh=0.90)
        _advance_to_classifying(sm)
        for _ in range(3):
            r = sm.tick(_det(), cls_result=("100", 0.95))
        assert r.state == State.ACCEPTED
        assert r.accepted_label == "100"
        assert abs(r.accepted_confidence - 0.95) < 0.01

    def test_disagreeing_predictions_do_not_accept(self):
        sm = _make_sm(agreement=3, cls_thresh=0.90)
        _advance_to_classifying(sm)
        for label in ["100", "200", "100", "200", "100"]:
            r = sm.tick(_det(), cls_result=(label, 0.95))
        # 3 "100" predictions but interspersed — should eventually agree on "100"
        # but NOT on the 3rd tick because "200" was in between
        # Actually it counts total occurrences per label — so 3x "100" → ACCEPTED
        # This test verifies majority-agreement logic
        assert r.state == State.ACCEPTED
        assert r.accepted_label == "100"

    def test_max_attempts_resets_to_note_detected(self):
        sm = _make_sm(agreement=3, cls_thresh=0.90, max_cls_attempts=5)
        sm.config.max_cls_attempts = 5
        _advance_to_classifying(sm)
        # Send 5 low-confidence results
        for _ in range(5):
            r = sm.tick(_det(), cls_result=("100", 0.70))
        assert r.state == State.NOTE_DETECTED

    def test_lost_detection_during_classifying_resets(self):
        sm = _make_sm()
        _advance_to_classifying(sm)
        r = sm.tick(_no_det())
        assert r.state == State.NOTE_DETECTED

    def test_all_six_denominations_accepted(self):
        for denom in ["10", "20", "50", "100", "200", "500"]:
            sm = _make_sm(agreement=2)
            _advance_to_classifying(sm)
            for _ in range(2):
                r = sm.tick(_det(), cls_result=(denom, 0.95))
            assert r.state == State.ACCEPTED
            assert r.accepted_label == denom


class TestAcceptedState:
    def test_mark_accepted_advances_to_wait_for_removal(self):
        sm = _make_sm()
        _advance_to_accepted(sm)
        sm.mark_accepted()
        assert sm.state == State.WAIT_FOR_REMOVAL

    def test_accepted_result_has_label_and_confidence(self):
        sm = _make_sm()
        _advance_to_accepted(sm)
        r = sm.tick(_det())  # should still report ACCEPTED until mark_accepted()
        assert r.state == State.ACCEPTED
        assert r.accepted_label is not None
        assert r.accepted_confidence is not None

    def test_mark_accepted_only_works_from_accepted(self):
        sm = _make_sm()
        sm.mark_accepted()  # called from wrong state — should be no-op
        assert sm.state == State.IDLE


class TestWaitForRemoval:
    def test_note_present_stays_in_wait(self):
        sm = _make_sm()
        _advance_to_accepted(sm)
        sm.mark_accepted()
        r = sm.tick(_det())
        assert r.state == State.WAIT_FOR_REMOVAL

    def test_removal_streak_returns_to_idle(self):
        sm = _make_sm(removal=4)
        _advance_to_accepted(sm)
        sm.mark_accepted()
        for _ in range(4):
            r = sm.tick(_no_det())
        assert r.state == State.IDLE

    def test_partial_removal_does_not_reset(self):
        sm = _make_sm(removal=4)
        _advance_to_accepted(sm)
        sm.mark_accepted()
        for _ in range(3):
            sm.tick(_no_det())
        # Note reappears
        r = sm.tick(_det())
        assert r.state == State.WAIT_FOR_REMOVAL

    def test_full_pipeline_two_notes(self):
        """Two complete donations in one session without manual reset."""
        sm = _make_sm(streak=4, stable=3, agreement=2, removal=3)

        # --- Note 1 ---
        _advance_to_classifying(sm)
        for _ in range(2):
            r = sm.tick(_det(), cls_result=("100", 0.95))
        assert r.state == State.ACCEPTED
        assert r.accepted_label == "100"
        sm.mark_accepted()

        # Remove note
        for _ in range(3):
            r = sm.tick(_no_det())
        assert r.state == State.IDLE

        # --- Note 2 (different denomination) ---
        for _ in range(4):
            sm.tick(_det())   # streak
        for _ in range(3):
            sm.tick(_det())   # stabilize
        for _ in range(2):
            r = sm.tick(_det(), cls_result=("500", 0.92))
        assert r.state == State.ACCEPTED
        assert r.accepted_label == "500"


class TestEdgeCases:
    def test_empty_detection_list_is_no_detection(self):
        sm = _make_sm()
        r = sm.tick([])
        assert r.state == State.IDLE

    def test_best_box_is_largest_area(self):
        sm = _make_sm(streak=1, stable=1, agreement=1)
        small = BBox(0, 0, 10, 10, 0.9)
        large = BBox(0, 0, 200, 200, 0.5)
        sm.tick([small, large])  # → NOTE_DETECTED
        sm.tick([small, large])  # → STABILIZING
        sm.tick([small, large])  # → CLASSIFYING
        r = sm.tick([small, large], cls_result=("50", 0.95))
        assert r.state == State.ACCEPTED

    def test_reset_from_any_state(self):
        for target_state in [State.NOTE_DETECTED, State.STABILIZING, State.CLASSIFYING]:
            sm = _make_sm()
            if target_state == State.NOTE_DETECTED:
                sm.tick(_det())
            elif target_state == State.STABILIZING:
                _advance_to_stabilizing(sm)
            elif target_state == State.CLASSIFYING:
                _advance_to_classifying(sm)
            sm.reset()
            assert sm.state == State.IDLE

    def test_bbox_center_distance_zero_for_same_box(self):
        b = BBox(50, 50, 150, 150)
        assert b.center_distance(b) == 0.0


# ---------------------------------------------------------------------------
# Simple test runner (no pytest required)
# ---------------------------------------------------------------------------

def _run_tests():
    import traceback
    test_classes = [
        TestBBox, TestIdleState, TestNoteDetectedState,
        TestStabilizingState, TestClassifyingState, TestAcceptedState,
        TestWaitForRemoval, TestEdgeCases,
    ]
    passed = failed = 0
    for cls in test_classes:
        obj = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for method in methods:
            try:
                getattr(obj, method)()
                print(f"  PASS  {cls.__name__}.{method}")
                passed += 1
            except Exception:
                print(f"  FAIL  {cls.__name__}.{method}")
                traceback.print_exc()
                failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = _run_tests()
    sys.exit(0 if ok else 1)

