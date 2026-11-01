# -*- coding: utf-8 -*-
"""
cv/pipeline/tests/test_pipeline_policy.py

Unit tests for the DivyaKosh donation decision pipeline.

These tests validate:
  1. Output field contract (classification, operational_action, count_authorized)
  2. Invariants: count_authorized True ONLY for ACCEPT_NOTE
  3. Invariants: operational_action COUNT ONLY for ACCEPT_NOTE
  4. Genuine accepted notes are counted exactly once (no double-count)
  5. REJECT_NOT_NOTE is routed to REVIEW, never to COUNT
  6. Uncertain inputs are routed to REVIEW, never to COUNT
  7. Unsupported denominations cannot trigger count_authorized=True
  8. review_reason is always present when operational_action==REVIEW
  9. PipelineResult.assert_invariants() catches all violations
 10. Backward-compat .decision alias equals .classification
 11. to_dict() includes all required keys

Tests use MOCK models so they run without GPU and without any checkpoint.
Models can be forced to return any probability vector via the mock.

Run from cv/ directory:
    .\.venv\Scripts\python.exe -m pytest pipeline/tests/test_pipeline_policy.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
import numpy as np

# Make sure the pipeline package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.divyakosh_pipeline import (
    DivyaKoshPipeline,
    PipelineResult,
    ThresholdConfig,
    SUPPORTED_DENOMINATIONS,
    CLS_ACCEPT,
    CLS_REJECT,
    CLS_REVIEW,
    _ACT_COUNT,
    _ACT_REVIEW,
    _load_model_a,
    _load_model_b,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(
    presence_accept_min: float = 0.30,
    presence_reject_max: float = 0.05,
    denomination_accept_min: float = 0.90,
    denomination_margin_min: float = 0.45,
    min_sharpness: float = 30.0,
) -> ThresholdConfig:
    """Build a ThresholdConfig with explicit values (no file I/O)."""
    cfg = ThresholdConfig()
    cfg.presence_accept_min    = presence_accept_min
    cfg.presence_reject_max    = presence_reject_max
    cfg.denomination_accept_min = denomination_accept_min
    cfg.denomination_margin_min = denomination_margin_min
    cfg.min_sharpness          = min_sharpness
    return cfg


def _sharp_black_image(w: int = 64, h: int = 64) -> Image.Image:
    """Return a sharp (high-Laplacian) synthetic black image."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Add a checkerboard to ensure high Laplacian variance
    arr[::2, ::2] = 255
    return Image.fromarray(arr, "RGB")


def _blurry_black_image(w: int = 64, h: int = 64) -> Image.Image:
    """Return a nearly-uniform (low-Laplacian) image."""
    arr = np.full((h, w, 3), 10, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


class MockModelA:
    """
    Mock Model A that returns logits such that softmax gives the desired
    P(NOTE) value. Uses log-odds: logit(p) = log(p / (1-p)).
    """
    def __init__(self, p_note: float):
        import math
        # Clamp to avoid log(0)
        p_note = max(1e-6, min(1 - 1e-6, p_note))
        p_not  = 1.0 - p_note
        # log-prob logits: softmax(log(p)) = p  (when probabilities sum to 1)
        self._logits = torch.tensor([[math.log(p_not), math.log(p_note)]])

    def __call__(self, tensor):
        return self._logits.clone()

    def to(self, device):
        return self

    def eval(self):
        return self


class MockModelB:
    """
    Mock Model B that returns logits such that softmax gives the desired
    per-class probabilities (list of 6, summing to 1).
    """
    def __init__(self, probs: list[float]):
        import math
        assert len(probs) == 6
        # Use log-probs as logits; softmax(log(p)) = p when sum(p)=1
        log_probs = [math.log(max(p, 1e-9)) for p in probs]
        self._logits = torch.tensor([log_probs])

    def __call__(self, tensor):
        return self._logits.clone()

    def to(self, device):
        return self

    def eval(self):
        return self


def _make_pipeline(
    p_note: float,
    b_probs: list[float],
    cfg: Optional[ThresholdConfig] = None,
) -> DivyaKoshPipeline:
    """
    Return a DivyaKoshPipeline with mocked models.
    b_probs: probabilities for [10,100,20,200,50,500] (alphabetical order).
    """
    pipeline = object.__new__(DivyaKoshPipeline)
    pipeline.device       = torch.device("cpu")
    pipeline.config       = cfg or _cfg()
    pipeline._model_a     = MockModelA(p_note)
    pipeline._model_b     = MockModelB(b_probs)
    pipeline._model_b_classes = ["10", "100", "20", "200", "50", "500"]
    return pipeline


# Convenience: high-confidence ₹100 note
# b_probs: [10,100,20,200,50,500]; index 1 = 100 at 0.99 → margin = 0.99−0.005 = 0.985
_B_PROBS_100 = [0.001, 0.99, 0.001, 0.001, 0.005, 0.002]
_B_PROBS_AMBIGUOUS = [0.30, 0.32, 0.10, 0.10, 0.10, 0.08]  # top=0.32, margin=0.02 → REVIEW


# ---------------------------------------------------------------------------
# 1. ACCEPT_NOTE invariants
# ---------------------------------------------------------------------------

class TestAcceptNote:
    def _result(self, denom_probs=None) -> PipelineResult:
        pipe = _make_pipeline(0.99, denom_probs or _B_PROBS_100)
        return pipe.predict(_sharp_black_image())

    def test_classification_is_accept(self):
        assert self._result().classification == CLS_ACCEPT

    def test_operational_action_is_count(self):
        assert self._result().operational_action == _ACT_COUNT

    def test_count_authorized_true(self):
        assert self._result().count_authorized is True

    def test_predicted_denomination_is_set(self):
        r = self._result()
        assert r.predicted_denomination is not None
        assert r.predicted_denomination in SUPPORTED_DENOMINATIONS

    def test_review_reason_is_none(self):
        assert self._result().review_reason is None

    def test_decision_alias_equals_classification(self):
        r = self._result()
        assert r.decision == r.classification == CLS_ACCEPT

    def test_assert_invariants_passes(self):
        self._result().assert_invariants()   # must not raise

    def test_to_dict_has_all_required_keys(self):
        d = self._result().to_dict()
        required = {
            "classification", "operational_action", "predicted_denomination",
            "presence_confidence", "denomination_confidence", "review_reason",
            "count_authorized", "decision",
        }
        assert required.issubset(d.keys())

    def test_each_supported_denomination_can_be_accepted(self):
        """Every supported denomination class can reach ACCEPT_NOTE."""
        # Indices: 10=0, 100=1, 20=2, 200=3, 50=4, 500=5
        for idx, denom in enumerate(["10", "100", "20", "200", "50", "500"]):
            probs = [0.002] * 6
            probs[idx] = 0.99
            # Normalise
            total = sum(probs)
            probs = [p / total for p in probs]
            pipe = _make_pipeline(0.99, probs)
            r = pipe.predict(_sharp_black_image())
            assert r.classification == CLS_ACCEPT,   f"Denom {denom} should be ACCEPT"
            assert r.count_authorized is True,        f"Denom {denom} should be authorized"
            assert r.predicted_denomination == denom, f"Expected {denom}, got {r.predicted_denomination}"

    def test_genuine_note_counted_exactly_once(self):
        """Calling predict once produces exactly one COUNT authorisation."""
        pipe = _make_pipeline(0.99, _B_PROBS_100)
        results = [pipe.predict(_sharp_black_image()) for _ in range(5)]
        count_auth = sum(1 for r in results if r.count_authorized)
        assert count_auth == 5   # 5 independent calls, each authorises once
        # No call produces more than one authorisation
        for r in results:
            assert r.count_authorized in (True, False)


# ---------------------------------------------------------------------------
# 2. REJECT_NOT_NOTE invariants
# ---------------------------------------------------------------------------

class TestRejectNotNote:
    def _result(self) -> PipelineResult:
        # p_note=0.01 <= reject_max=0.05 → REJECT_NOT_NOTE
        pipe = _make_pipeline(0.01, _B_PROBS_100)
        return pipe.predict(_sharp_black_image())

    def test_classification_is_reject(self):
        assert self._result().classification == CLS_REJECT

    def test_operational_action_is_review(self):
        """REJECT_NOT_NOTE must be operationally routed to REVIEW, not COUNT."""
        assert self._result().operational_action == _ACT_REVIEW

    def test_count_authorized_false(self):
        assert self._result().count_authorized is False

    def test_predicted_denomination_none(self):
        assert self._result().predicted_denomination is None

    def test_review_reason_present(self):
        assert self._result().review_reason is not None
        assert len(self._result().review_reason) > 0

    def test_decision_alias(self):
        r = self._result()
        assert r.decision == r.classification == CLS_REJECT

    def test_assert_invariants_passes(self):
        self._result().assert_invariants()

    def test_model_b_not_run(self):
        """Model B should not run when Model A already classifies as NOT_NOTE."""
        r = self._result()
        assert r.denomination_confidence == 0.0
        assert r.model_b_probs == {}


# ---------------------------------------------------------------------------
# 3. REVIEW invariants
# ---------------------------------------------------------------------------

class TestReview:
    def _uncertain_model_a(self) -> PipelineResult:
        # p_note=0.15: between reject_max=0.05 and accept_min=0.30 → REVIEW
        pipe = _make_pipeline(0.15, _B_PROBS_100)
        return pipe.predict(_sharp_black_image())

    def _low_model_b_conf(self) -> PipelineResult:
        # Model A confident (0.99), but Model B top prob=0.32 < 0.90 → REVIEW
        pipe = _make_pipeline(0.99, _B_PROBS_AMBIGUOUS)
        return pipe.predict(_sharp_black_image())

    def _blurry(self) -> PipelineResult:
        pipe = _make_pipeline(0.99, _B_PROBS_100)
        return pipe.predict(_blurry_black_image())

    def _low_margin(self) -> PipelineResult:
        # top=0.92, second=0.86 → after softmax the margin will be < 0.45
        # Use raw-logit mock: log-probs for [10,100,20,200,50,500]
        # We want top=100 at ~0.92 and 500 at ~0.86 → margin < 0.45
        # Give 100 slightly more logit than 500, but close together
        # log(0.51) ≈ -0.67, log(0.49) ≈ -0.71  → after softmax: ~0.51 vs 0.49, margin≈0.02
        probs = [0.0, 0.51, 0.0, 0.0, 0.0, 0.49]  # 100 vs 500 almost tied
        pipe = _make_pipeline(0.99, probs)
        return pipe.predict(_sharp_black_image())

    @pytest.mark.parametrize("result_fn", [
        "_uncertain_model_a",
        "_low_model_b_conf",
        "_blurry",
        "_low_margin",
    ])
    def test_classification_is_review(self, result_fn):
        r = getattr(self, result_fn)()
        assert r.classification == CLS_REVIEW, \
            f"{result_fn}: expected REVIEW, got {r.classification}"

    @pytest.mark.parametrize("result_fn", [
        "_uncertain_model_a",
        "_low_model_b_conf",
        "_blurry",
        "_low_margin",
    ])
    def test_operational_action_is_review(self, result_fn):
        r = getattr(self, result_fn)()
        assert r.operational_action == _ACT_REVIEW

    @pytest.mark.parametrize("result_fn", [
        "_uncertain_model_a",
        "_low_model_b_conf",
        "_blurry",
        "_low_margin",
    ])
    def test_count_authorized_false(self, result_fn):
        r = getattr(self, result_fn)()
        assert r.count_authorized is False

    @pytest.mark.parametrize("result_fn", [
        "_uncertain_model_a",
        "_low_model_b_conf",
        "_blurry",
        "_low_margin",
    ])
    def test_review_reason_present(self, result_fn):
        r = getattr(self, result_fn)()
        assert r.review_reason is not None and len(r.review_reason) > 0

    @pytest.mark.parametrize("result_fn", [
        "_uncertain_model_a",
        "_low_model_b_conf",
        "_blurry",
        "_low_margin",
    ])
    def test_assert_invariants_passes(self, result_fn):
        getattr(self, result_fn)().assert_invariants()


# ---------------------------------------------------------------------------
# 4. Hard invariant: no non-ACCEPT outcome may authorize counting
# ---------------------------------------------------------------------------

class TestCountAuthorizedInvariant:
    @pytest.mark.parametrize("p_note,b_probs,expected_cls", [
        # confident NOT_NOTE
        (0.01, _B_PROBS_100,      CLS_REJECT),
        # uncertain Model A
        (0.15, _B_PROBS_100,      CLS_REVIEW),
        # low Model B confidence
        (0.99, _B_PROBS_AMBIGUOUS, CLS_REVIEW),
    ])
    def test_non_accept_never_authorizes_count(self, p_note, b_probs, expected_cls):
        pipe = _make_pipeline(p_note, b_probs)
        r = pipe.predict(_sharp_black_image())
        assert r.classification == expected_cls
        assert r.count_authorized is False
        assert r.operational_action == _ACT_REVIEW

    def test_only_accept_sets_count_authorized(self):
        """Exhaustive: for every classification value, count_authorized must match."""
        cases = [
            (0.01, _B_PROBS_100,       CLS_REJECT, False),
            (0.15, _B_PROBS_100,       CLS_REVIEW, False),
            (0.99, _B_PROBS_AMBIGUOUS, CLS_REVIEW, False),
            (0.99, _B_PROBS_100,       CLS_ACCEPT, True),
        ]
        for p_note, b_probs, expected_cls, expected_auth in cases:
            pipe = _make_pipeline(p_note, b_probs)
            r = pipe.predict(_sharp_black_image())
            assert r.classification == expected_cls
            assert r.count_authorized == expected_auth, \
                f"cls={expected_cls}: count_authorized should be {expected_auth}"


# ---------------------------------------------------------------------------
# 5. Unsupported denomination cannot be counted
# ---------------------------------------------------------------------------

class TestUnsupportedDenomination:
    def test_pipeline_b_classes_are_all_supported(self):
        """Every class that Model B can output is in SUPPORTED_DENOMINATIONS."""
        pipe = _make_pipeline(0.99, _B_PROBS_100)
        for cls in pipe._model_b_classes:
            assert cls in SUPPORTED_DENOMINATIONS, \
                f"Model B class {cls!r} is not in SUPPORTED_DENOMINATIONS"

    def test_injecting_unsupported_class_raises_on_accept(self):
        """
        If somehow an unsupported denomination reaches ACCEPT_NOTE,
        assert_invariants() must catch it.
        """
        bad_result = PipelineResult(
            classification=CLS_ACCEPT,
            operational_action=_ACT_COUNT,
            predicted_denomination="9999",  # unsupported
            presence_confidence=0.99,
            denomination_confidence=0.99,
            denomination_margin=0.90,
            review_reason=None,
            count_authorized=True,
            image_sharpness=200.0,
            image_brightness=50.0,
            latency_ms=10.0,
        )
        with pytest.raises(AssertionError, match="SUPPORTED_DENOMINATIONS"):
            bad_result.assert_invariants()


# ---------------------------------------------------------------------------
# 6. assert_invariants() catches all structural violations
# ---------------------------------------------------------------------------

class TestAssertInvariants:
    def _accept_result(self, **overrides) -> PipelineResult:
        base = dict(
            classification=CLS_ACCEPT,
            operational_action=_ACT_COUNT,
            predicted_denomination="100",
            presence_confidence=0.99,
            denomination_confidence=0.99,
            denomination_margin=0.90,
            review_reason=None,
            count_authorized=True,
            image_sharpness=200.0,
            image_brightness=50.0,
            latency_ms=10.0,
        )
        base.update(overrides)
        return PipelineResult(**base)

    def _review_result(self, **overrides) -> PipelineResult:
        base = dict(
            classification=CLS_REVIEW,
            operational_action=_ACT_REVIEW,
            predicted_denomination=None,
            presence_confidence=0.15,
            denomination_confidence=0.0,
            denomination_margin=0.0,
            review_reason="Model A uncertain",
            count_authorized=False,
            image_sharpness=200.0,
            image_brightness=50.0,
            latency_ms=10.0,
        )
        base.update(overrides)
        return PipelineResult(**base)

    def test_valid_accept_passes(self):
        self._accept_result().assert_invariants()

    def test_valid_review_passes(self):
        self._review_result().assert_invariants()

    def test_accept_with_wrong_action_fails(self):
        with pytest.raises(AssertionError):
            self._accept_result(operational_action=_ACT_REVIEW).assert_invariants()

    def test_accept_with_count_false_fails(self):
        with pytest.raises(AssertionError):
            self._accept_result(count_authorized=False).assert_invariants()

    def test_accept_with_no_denomination_fails(self):
        with pytest.raises(AssertionError):
            self._accept_result(predicted_denomination=None).assert_invariants()

    def test_accept_with_review_reason_fails(self):
        with pytest.raises(AssertionError):
            self._accept_result(review_reason="some reason").assert_invariants()

    def test_review_with_count_action_fails(self):
        with pytest.raises(AssertionError):
            self._review_result(operational_action=_ACT_COUNT).assert_invariants()

    def test_review_with_count_authorized_true_fails(self):
        with pytest.raises(AssertionError):
            self._review_result(count_authorized=True).assert_invariants()

    def test_review_with_no_reason_fails(self):
        with pytest.raises(AssertionError):
            self._review_result(review_reason=None).assert_invariants()

    def test_reject_routed_to_count_fails(self):
        with pytest.raises(AssertionError):
            PipelineResult(
                classification=CLS_REJECT,
                operational_action=_ACT_COUNT,   # WRONG
                predicted_denomination=None,
                presence_confidence=0.01,
                denomination_confidence=0.0,
                denomination_margin=0.0,
                review_reason="not a note",
                count_authorized=False,
                image_sharpness=200.0,
                image_brightness=50.0,
                latency_ms=5.0,
            ).assert_invariants()


# ---------------------------------------------------------------------------
# 7. to_dict() contract
# ---------------------------------------------------------------------------

class TestToDict:
    @pytest.mark.parametrize("p_note,b_probs", [
        (0.99, _B_PROBS_100),       # ACCEPT
        (0.01, _B_PROBS_100),       # REJECT
        (0.15, _B_PROBS_100),       # REVIEW (uncertain A)
        (0.99, _B_PROBS_AMBIGUOUS), # REVIEW (low B)
    ])
    def test_required_fields_present(self, p_note, b_probs):
        pipe = _make_pipeline(p_note, b_probs)
        d = pipe.predict(_sharp_black_image()).to_dict()
        for key in [
            "classification", "operational_action", "predicted_denomination",
            "presence_confidence", "denomination_confidence", "review_reason",
            "count_authorized", "decision",  # backward-compat alias
        ]:
            assert key in d, f"Missing key: {key!r}"

    def test_decision_alias_matches_classification_in_dict(self):
        pipe = _make_pipeline(0.99, _B_PROBS_100)
        d = pipe.predict(_sharp_black_image()).to_dict()
        assert d["decision"] == d["classification"]


# ---------------------------------------------------------------------------
# 8. No input is silently discarded
# ---------------------------------------------------------------------------

class TestNoSilentDiscard:
    @pytest.mark.parametrize("p_note,b_probs", [
        (0.99, _B_PROBS_100),
        (0.01, _B_PROBS_100),
        (0.15, _B_PROBS_100),
        (0.99, _B_PROBS_AMBIGUOUS),
    ])
    def test_every_input_produces_a_result(self, p_note, b_probs):
        pipe = _make_pipeline(p_note, b_probs)
        r = pipe.predict(_sharp_black_image())
        assert r is not None
        assert r.classification in {CLS_ACCEPT, CLS_REJECT, CLS_REVIEW}
        assert r.operational_action in {_ACT_COUNT, _ACT_REVIEW}

    @pytest.mark.parametrize("p_note,b_probs", [
        (0.01, _B_PROBS_100),
        (0.15, _B_PROBS_100),
        (0.99, _B_PROBS_AMBIGUOUS),
    ])
    def test_non_accept_produces_review_destination(self, p_note, b_probs):
        pipe = _make_pipeline(p_note, b_probs)
        r = pipe.predict(_sharp_black_image())
        assert r.operational_action == _ACT_REVIEW, \
            f"Expected REVIEW destination for cls={r.classification}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.call([
        sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
    ]))
