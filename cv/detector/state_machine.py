"""
cv/detector/state_machine.py

Temporal state machine for banknote presence detection.

States:
    IDLE              - No note in frame. Waiting.
    NOTE_DETECTED     - YOLO detected a note. Building detection streak.
    STABILIZING       - Streak threshold met. Waiting for stable bounding box.
    CLASSIFYING       - Bbox is stable. Running MobileNetV3, collecting agreement.
    ACCEPTED          - Classification agreed. Record written. Waiting for removal.
    WAIT_FOR_REMOVAL  - Note accepted. Waiting for it to leave frame before reset.

Transitions (all via tick() / result() calls — no CV inside this module):
    IDLE            -> NOTE_DETECTED     : YOLO box received
    NOTE_DETECTED   -> IDLE             : detection lost before streak
    NOTE_DETECTED   -> STABILIZING      : streak_frames consecutive detections
    STABILIZING     -> NOTE_DETECTED    : bbox shifted > stability_px
    STABILIZING     -> CLASSIFYING      : stable_frames consecutive stable frames
    CLASSIFYING     -> ACCEPTED         : cls_agreement same-label predictions >= threshold
    CLASSIFYING     -> NOTE_DETECTED    : too many attempts, no agreement (reset streak)
    ACCEPTED        -> WAIT_FOR_REMOVAL : (caller calls .mark_accepted())
    WAIT_FOR_REMOVAL-> IDLE             : removal_frames consecutive no-detections
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple


class State(Enum):
    IDLE = auto()
    NOTE_DETECTED = auto()
    STABILIZING = auto()
    CLASSIFYING = auto()
    ACCEPTED = auto()
    WAIT_FOR_REMOVAL = auto()


@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def center_distance(self, other: "BBox") -> float:
        """Euclidean distance between box centres."""
        return ((self.cx - other.cx) ** 2 + (self.cy - other.cy) ** 2) ** 0.5

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


@dataclass
class StateMachineConfig:
    """All configurable thresholds in one place."""
    # YOLO detection gate
    det_confidence: float = 0.40      # min YOLO confidence to count as detection
    # NOTE_DETECTED → STABILIZING
    streak_frames: int = 4            # consecutive detected frames required
    # STABILIZING
    stability_px: float = 30.0       # max bbox centre shift to stay stable
    stable_frames: int = 3           # stable frames before CLASSIFYING
    # CLASSIFYING
    cls_threshold: float = 0.90      # MobileNetV3 confidence for a valid prediction
    cls_agreement: int = 3           # consecutive agreeing predictions needed
    max_cls_attempts: int = 10       # give up and reset after this many attempts
    # WAIT_FOR_REMOVAL → IDLE
    removal_frames: int = 8          # no-detection frames before reset


@dataclass
class StateMachineResult:
    """What the caller needs to know after each tick."""
    state: State
    # Set when CLASSIFYING produces an accepted prediction
    accepted_label: Optional[str] = None
    accepted_confidence: Optional[float] = None
    # Human-readable status for UI overlay
    status_text: str = ""
    # Current best bbox (for drawing overlay)
    current_bbox: Optional[BBox] = None


class DonationStateMachine:
    """
    Temporal state machine for one donation session.

    Usage (per-frame loop):
        sm = DonationStateMachine()
        while True:
            detections = yolo(frame)           # list[BBox], may be empty
            cls_result = None
            if sm.state == State.CLASSIFYING and detections:
                crop = frame[best_box.as_xyxy()]
                label, conf = mobilenet(crop)
                cls_result = (label, conf)
            result = sm.tick(detections, cls_result)
            if result.state == State.ACCEPTED:
                write_record(result.accepted_label, result.accepted_confidence)
                sm.mark_accepted()             # → WAIT_FOR_REMOVAL
    """

    def __init__(self, config: Optional[StateMachineConfig] = None) -> None:
        self.config = config or StateMachineConfig()
        self._state: State = State.IDLE
        self._reset_counters()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    def tick(
        self,
        detections: list[BBox],
        cls_result: Optional[Tuple[str, float]] = None,
    ) -> StateMachineResult:
        """
        Advance the state machine by one frame.

        Args:
            detections:  List of BBox objects from YOLO (empty = no banknote).
            cls_result:  (label, confidence) from MobileNetV3, only required
                         when state == CLASSIFYING and detections is non-empty.
        Returns:
            StateMachineResult with the new state and relevant fields.
        """
        best_box = self._best_detection(detections)
        has_detection = best_box is not None

        if self._state == State.IDLE:
            return self._tick_idle(has_detection, best_box)
        elif self._state == State.NOTE_DETECTED:
            return self._tick_note_detected(has_detection, best_box)
        elif self._state == State.STABILIZING:
            return self._tick_stabilizing(has_detection, best_box)
        elif self._state == State.CLASSIFYING:
            return self._tick_classifying(has_detection, best_box, cls_result)
        elif self._state == State.ACCEPTED:
            # Caller must call mark_accepted(); until then stay in ACCEPTED.
            return StateMachineResult(
                state=self._state,
                accepted_label=self._accepted_label,
                accepted_confidence=self._accepted_confidence,
                status_text="ACCEPTED",
                current_bbox=self._last_bbox,
            )
        elif self._state == State.WAIT_FOR_REMOVAL:
            return self._tick_wait_for_removal(has_detection)
        else:
            raise RuntimeError(f"Unknown state: {self._state}")

    def mark_accepted(self) -> None:
        """Call after writing the JSONL/SQLite record to advance to WAIT_FOR_REMOVAL."""
        if self._state == State.ACCEPTED:
            self._transition(State.WAIT_FOR_REMOVAL)
            self._no_det_streak = 0

    def reset(self) -> None:
        """Force reset to IDLE (e.g., user presses Q)."""
        self._transition(State.IDLE)
        self._reset_counters()

    # ------------------------------------------------------------------
    # Per-state tick helpers
    # ------------------------------------------------------------------

    def _tick_idle(self, has_det: bool, best_box: Optional[BBox]) -> StateMachineResult:
        if has_det:
            self._transition(State.NOTE_DETECTED)
            self._det_streak = 1
            self._last_bbox = best_box
        return StateMachineResult(
            state=self._state,
            status_text="WAITING FOR NOTE" if not has_det else f"CHECKING... 1/{self.config.streak_frames}",
            current_bbox=best_box,
        )

    def _tick_note_detected(self, has_det: bool, best_box: Optional[BBox]) -> StateMachineResult:
        if not has_det:
            self._transition(State.IDLE)
            self._reset_counters()
            return StateMachineResult(state=self._state, status_text="WAITING FOR NOTE")

        self._det_streak += 1
        self._last_bbox = best_box

        if self._det_streak >= self.config.streak_frames:
            self._transition(State.STABILIZING)
            self._stable_streak = 0
            self._stable_ref_bbox = best_box
            return StateMachineResult(
                state=self._state,
                status_text="STABILIZING...",
                current_bbox=best_box,
            )

        return StateMachineResult(
            state=self._state,
            status_text=f"CHECKING... {self._det_streak}/{self.config.streak_frames}",
            current_bbox=best_box,
        )

    def _tick_stabilizing(self, has_det: bool, best_box: Optional[BBox]) -> StateMachineResult:
        if not has_det:
            self._transition(State.NOTE_DETECTED)
            self._det_streak = 0
            return StateMachineResult(state=self._state, status_text="CHECKING... 0/"+str(self.config.streak_frames))

        assert best_box is not None
        shift = self._stable_ref_bbox.center_distance(best_box) if self._stable_ref_bbox else float("inf")

        if shift > self.config.stability_px:
            # Bbox moved — restart stability check
            self._stable_streak = 0
            self._stable_ref_bbox = best_box
            self._last_bbox = best_box
            return StateMachineResult(
                state=self._state,
                status_text="STABILIZING... (moved)",
                current_bbox=best_box,
            )

        self._stable_streak += 1
        self._last_bbox = best_box

        if self._stable_streak >= self.config.stable_frames:
            self._transition(State.CLASSIFYING)
            self._cls_attempts = 0
            self._cls_label_counts: dict[str, int] = {}
            return StateMachineResult(
                state=self._state,
                status_text="CLASSIFYING...",
                current_bbox=best_box,
            )

        return StateMachineResult(
            state=self._state,
            status_text=f"STABILIZING... {self._stable_streak}/{self.config.stable_frames}",
            current_bbox=best_box,
        )

    def _tick_classifying(
        self,
        has_det: bool,
        best_box: Optional[BBox],
        cls_result: Optional[Tuple[str, float]],
    ) -> StateMachineResult:
        if not has_det:
            # Note disappeared during classification — reset
            self._transition(State.NOTE_DETECTED)
            self._det_streak = 0
            return StateMachineResult(state=self._state, status_text="CHECKING... 0/"+str(self.config.streak_frames))

        self._last_bbox = best_box

        if cls_result is None:
            return StateMachineResult(
                state=self._state,
                status_text="CLASSIFYING...",
                current_bbox=best_box,
            )

        label, confidence = cls_result
        self._cls_attempts += 1

        if confidence >= self.config.cls_threshold:
            self._cls_label_counts[label] = self._cls_label_counts.get(label, 0) + 1
            best_label = max(self._cls_label_counts, key=self._cls_label_counts.__getitem__)
            best_count = self._cls_label_counts[best_label]

            if best_count >= self.config.cls_agreement:
                # ACCEPT
                self._accepted_label = best_label
                self._accepted_confidence = confidence
                self._transition(State.ACCEPTED)
                return StateMachineResult(
                    state=self._state,
                    accepted_label=best_label,
                    accepted_confidence=confidence,
                    status_text="ACCEPTED",
                    current_bbox=best_box,
                )

        if self._cls_attempts >= self.config.max_cls_attempts:
            # Too many attempts — reset to NOTE_DETECTED for another try
            self._transition(State.NOTE_DETECTED)
            self._det_streak = self.config.streak_frames  # keep streak so we go back to STABILIZING quickly
            self._cls_attempts = 0
            self._cls_label_counts = {}
            return StateMachineResult(
                state=self._state,
                status_text="LOW CONFIDENCE - HOLD STEADY",
                current_bbox=best_box,
            )

        top_label = label if confidence >= self.config.cls_threshold else "?"
        return StateMachineResult(
            state=self._state,
            status_text=f"CLASSIFYING... {top_label} ({confidence*100:.0f}%)",
            current_bbox=best_box,
        )

    def _tick_wait_for_removal(self, has_det: bool) -> StateMachineResult:
        if has_det:
            self._no_det_streak = 0
            return StateMachineResult(
                state=self._state,
                status_text="REMOVE NOTE TO CONTINUE",
                current_bbox=self._last_bbox,
            )

        self._no_det_streak += 1
        if self._no_det_streak >= self.config.removal_frames:
            self._transition(State.IDLE)
            self._reset_counters()
            return StateMachineResult(state=self._state, status_text="WAITING FOR NOTE")

        return StateMachineResult(
            state=self._state,
            status_text=f"REMOVE NOTE... {self._no_det_streak}/{self.config.removal_frames}",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _transition(self, new_state: State) -> None:
        self._state = new_state

    def _reset_counters(self) -> None:
        self._det_streak: int = 0
        self._stable_streak: int = 0
        self._stable_ref_bbox: Optional[BBox] = None
        self._last_bbox: Optional[BBox] = None
        self._cls_attempts: int = 0
        self._cls_label_counts: dict[str, int] = {}
        self._no_det_streak: int = 0
        self._accepted_label: Optional[str] = None
        self._accepted_confidence: Optional[float] = None

    @staticmethod
    def _best_detection(detections: list[BBox]) -> Optional[BBox]:
        """Return the largest-area detection above confidence threshold."""
        if not detections:
            return None
        return max(detections, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))

