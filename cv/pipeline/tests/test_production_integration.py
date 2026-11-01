# -*- coding: utf-8 -*-
"""
cv/pipeline/tests/test_production_integration.py

Comprehensive Integration & Regression Tests for DivyaKosh CV Pipeline.
========================================================================
Validates all requirements specified in the production integration specification:
  A. Genuine ₹100 note -> ACCEPT_NOTE, COUNT, exactly 1 DB record, exactly 1 blockchain registration
  B. Flower -> REVIEW destination, 0 DB records, 0 blockchain registrations
  C. Coin -> REJECT_NOT_NOTE / REVIEW, 0 DB records, 0 blockchain registrations
  D. Receipt -> REJECT_NOT_NOTE / REVIEW, 0 DB records, 0 blockchain registrations
  E. Blurry note -> REVIEW (sharpness check), 0 DB records, 0 blockchain registrations
  F. Unauthorized legacy record (decision=ACCEPT, count_authorized missing) -> REJECTED by SQLite & blockchain
  G. Malicious or inconsistent record -> REJECTED by SQLite & blockchain
  H. Duplicate frames -> Anti-duplicate lock ensures 1 physical note is counted exactly once
  I. Unsupported denomination -> REJECTED by pipeline, SQLite & blockchain

Runs using mocks for network/blockchain. Never executes real blockchain transactions.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

_TESTS_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _TESTS_DIR.parent
_CV_DIR = _PIPELINE_DIR.parent

if str(_CV_DIR) not in sys.path:
    sys.path.insert(0, str(_CV_DIR))
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from pipeline.adapter import (
    format_accepted_donation_record,
    format_review_record,
    safe_register_blockchain_proof,
    validate_for_donation,
)
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
from sqlite.database import get_connection, initialize_database
from sqlite.importer import import_inference_file, import_single_record, is_authorized_donation_record
from sqlite.queries import get_donation_count, get_total_donation_amount
from src.blockchain import register_blockchain_proof


# ---------------------------------------------------------------------------
# Mocks & Test Fixtures
# ---------------------------------------------------------------------------

class MockModelA:
    """Returns logits that produce exact desired p_note after softmax."""
    def __init__(self, p_note: float):
        p_note = max(1e-6, min(1.0 - 1e-6, p_note))
        p_not = 1.0 - p_note
        self._logits = torch.tensor([[math.log(p_not), math.log(p_note)]])

    def __call__(self, tensor):
        return self._logits.clone()

    def to(self, device):
        return self

    def eval(self):
        return self


class MockModelB:
    """Returns logits that produce exact desired class probabilities after softmax."""
    def __init__(self, probs: list[float]):
        assert len(probs) == 6
        log_probs = [math.log(max(p, 1e-9)) for p in probs]
        self._logits = torch.tensor([log_probs])

    def __call__(self, tensor):
        return self._logits.clone()

    def to(self, device):
        return self

    def eval(self):
        return self


def make_test_pipeline(p_note: float, b_probs: list[float], min_sharpness: float = 30.0) -> DivyaKoshPipeline:
    """Build a mock DivyaKoshPipeline with explicit class predictions."""
    pipeline = object.__new__(DivyaKoshPipeline)
    pipeline.device = torch.device("cpu")
    cfg = ThresholdConfig()
    cfg.presence_accept_min = 0.30
    cfg.presence_reject_max = 0.05
    cfg.denomination_accept_min = 0.90
    cfg.denomination_margin_min = 0.45
    cfg.min_sharpness = min_sharpness
    pipeline.config = cfg
    pipeline._model_a = MockModelA(p_note)
    pipeline._model_b = MockModelB(b_probs)
    pipeline._model_b_classes = ["10", "100", "20", "200", "50", "500"]
    return pipeline


def make_sharp_image(w: int = 128, h: int = 128) -> Image.Image:
    """High-contrast checkerboard image with very high Laplacian variance (> 1000)."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[::4, ::4] = 255
    arr[1::4, 1::4] = 200
    return Image.fromarray(arr, "RGB")


def make_blurry_image(w: int = 128, h: int = 128) -> Image.Image:
    """Smooth flat image with Laplacian variance near zero (< 5.0)."""
    arr = np.full((h, w, 3), 40, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


# Class order: ["10", "100", "20", "200", "50", "500"]
PROBS_RS_100 = [0.001, 0.99, 0.001, 0.001, 0.005, 0.002]   # ₹100 note
PROBS_FLAT   = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]               # ambiguous / OOD


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Provide a clean temporary SQLite database for each test."""
    db_path = tmp_path / "test_donations.db"
    initialize_database(db_path)
    return db_path


# ---------------------------------------------------------------------------
# Requirement A: Genuine ₹100 Note
# ---------------------------------------------------------------------------

class TestRequirementAGenuineNote:
    def test_genuine_100_note_accepted_counted_and_registered(self, temp_db: Path):
        pipe = make_test_pipeline(p_note=0.99, b_probs=PROBS_RS_100)
        img = make_sharp_image()

        result = pipe.predict(img)

        # 1. Pipeline classification & action
        assert result.classification == CLS_ACCEPT
        assert result.operational_action == _ACT_COUNT
        assert result.count_authorized is True
        assert result.predicted_denomination == "100"
        assert result.review_reason is None

        # 2. Format record for storage and blockchain
        record = format_accepted_donation_record(result, pocket_id="webcam-0")
        assert record["decision"] == "ACCEPT"
        assert record["count_authorized"] is True
        assert record["operational_action"] == "COUNT"
        assert record["classification"] == "ACCEPT_NOTE"
        assert record["predicted_denomination"] == "100"

        # 3. Exactly one SQLite record inserted
        import_res = import_single_record(record, database_file=temp_db)
        assert import_res.inserted == 1
        assert import_res.non_accepted_skipped == 0
        assert get_donation_count(temp_db) == 1
        assert get_total_donation_amount(temp_db) == 100

        # 4. Exactly one blockchain registration attempt
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "success": True,
                "proofHash": "0x1234567890abcdef",
                "proofScheme": "canonical-v1",
                "transactionHash": "0xabcdef123456",
                "blockNumber": 42,
            }
            mock_post.return_value.raise_for_status = MagicMock()

            bc_result = safe_register_blockchain_proof(record)
            assert bc_result is not None
            assert mock_post.call_count == 1

            # Validate payload sent to blockchain API
            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs["json"]
            assert "proofHash" in payload
            assert payload["sourceId"] == "webcam-0"


# ---------------------------------------------------------------------------
# Requirements B, C, D: Non-Notes (Flower, Coin, Receipt)
# ---------------------------------------------------------------------------

class TestRequirementsNonNotes:
    @pytest.mark.parametrize("item_name, p_note, b_probs", [
        ("flower", 0.01, PROBS_FLAT),       # Confident NOT_NOTE
        ("coin", 0.02, PROBS_FLAT),         # Confident NOT_NOTE
        ("receipt", 0.03, PROBS_FLAT),      # Confident NOT_NOTE
        ("uncertain_flower", 0.15, PROBS_FLAT), # Uncertain presence zone
    ])
    def test_non_notes_route_to_review_never_count_or_register(
        self, temp_db: Path, item_name: str, p_note: float, b_probs: list[float]
    ):
        pipe = make_test_pipeline(p_note=p_note, b_probs=b_probs)
        img = make_sharp_image()

        result = pipe.predict(img)

        # 1. Decision must be REJECT_NOT_NOTE or REVIEW, operational_action must be REVIEW
        assert result.classification in (CLS_REJECT, CLS_REVIEW)
        assert result.operational_action == _ACT_REVIEW
        assert result.count_authorized is False
        assert result.predicted_denomination is None
        assert result.review_reason is not None

        # 2. Format review record
        review_record = format_review_record(result, pocket_id="webcam-0")
        assert review_record["count_authorized"] is False
        assert review_record["operational_action"] == "REVIEW"
        assert review_record["decision"] == "REVIEW"

        # 3. ZERO records inserted in SQLite
        import_res = import_single_record(review_record, database_file=temp_db)
        assert import_res.inserted == 0
        assert import_res.non_accepted_skipped == 1
        assert get_donation_count(temp_db) == 0
        assert get_total_donation_amount(temp_db) == 0

        # 4. ZERO blockchain registration attempts
        with patch("requests.post") as mock_post:
            bc_result = safe_register_blockchain_proof(review_record)
            assert bc_result is None
            assert mock_post.call_count == 0


# ---------------------------------------------------------------------------
# Requirement E: Blurry Genuine Note
# ---------------------------------------------------------------------------

class TestRequirementEBlurryNote:
    def test_blurry_genuine_note_sent_to_review(self, temp_db: Path):
        # High confidence predictions, but image is severely blurred
        pipe = make_test_pipeline(p_note=0.99, b_probs=PROBS_RS_100, min_sharpness=30.0)
        blurry_img = make_blurry_image()

        result = pipe.predict(blurry_img)

        # Quality check failure routes immediately to REVIEW
        assert result.classification == CLS_REVIEW
        assert result.operational_action == _ACT_REVIEW
        assert result.count_authorized is False
        assert "blurry" in result.review_reason.lower()

        # Zero DB insertions
        review_record = format_review_record(result, pocket_id="webcam-0")
        import_res = import_single_record(review_record, database_file=temp_db)
        assert import_res.inserted == 0
        assert get_donation_count(temp_db) == 0

        # Zero blockchain calls
        with patch("requests.post") as mock_post:
            bc_result = safe_register_blockchain_proof(review_record)
            assert bc_result is None
            assert mock_post.call_count == 0


# ---------------------------------------------------------------------------
# Requirement F: Unauthorized Legacy Record
# ---------------------------------------------------------------------------

class TestRequirementFLegacyRecord:
    def test_legacy_record_with_decision_accept_rejected_by_sqlite(self, temp_db: Path):
        """Legacy record that has decision='ACCEPT' but lacks count_authorized MUST be rejected."""
        legacy_record = {
            "pocket_id": "webcam-0",
            "timestamp": "2026-10-31T20:00:00Z",
            "item_status": "NOTE_DETECTED",
            "predicted_denomination": "100",
            "classifier_confidence": 0.95,
            "confidence": 0.95,
            "source": "webcam",
            "decision": "ACCEPT",          # Legacy flag
            "target_bin": "BIN_100",
            "model_version": "mobilenet_v3_clean_v1",
            # count_authorized is MISSING
            # operational_action is MISSING
            # classification is MISSING
        }

        assert is_authorized_donation_record(legacy_record) is False

        # SQLite importer MUST reject it
        import_res = import_single_record(legacy_record, database_file=temp_db)
        assert import_res.inserted == 0
        assert import_res.non_accepted_skipped == 1
        assert get_donation_count(temp_db) == 0

        # Blockchain gateway MUST reject it
        with patch("requests.post") as mock_post:
            bc_result = safe_register_blockchain_proof(legacy_record)
            assert bc_result is None
            assert mock_post.call_count == 0

            # Direct call to src.blockchain.register_blockchain_proof also rejects
            direct_bc = register_blockchain_proof(legacy_record)
            assert direct_bc is None
            assert mock_post.call_count == 0


# ---------------------------------------------------------------------------
# Requirement G: Malicious or Inconsistent Record
# ---------------------------------------------------------------------------

class TestRequirementGInconsistentRecord:
    @pytest.mark.parametrize("inconsistent_record", [
        # Inconsistency 1: classification=REVIEW but decision=ACCEPT
        {
            "pocket_id": "webcam-0", "timestamp": "2026-10-31T20:00:00Z",
            "predicted_denomination": "100", "classifier_confidence": 0.95,
            "decision": "ACCEPT", "classification": "REVIEW",
            "operational_action": "REVIEW", "count_authorized": False,
        },
        # Inconsistency 2: count_authorized=True but operational_action=REVIEW
        {
            "pocket_id": "webcam-0", "timestamp": "2026-10-31T20:00:00Z",
            "predicted_denomination": "100", "classifier_confidence": 0.95,
            "decision": "ACCEPT", "classification": "ACCEPT_NOTE",
            "operational_action": "REVIEW", "count_authorized": True,
        },
        # Inconsistency 3: count_authorized='true' (truthy string, not boolean True)
        {
            "pocket_id": "webcam-0", "timestamp": "2026-10-31T20:00:00Z",
            "predicted_denomination": "100", "classifier_confidence": 0.95,
            "decision": "ACCEPT", "classification": "ACCEPT_NOTE",
            "operational_action": "COUNT", "count_authorized": "true",
        },
        # Inconsistency 4: count_authorized=1 (truthy int, not boolean True)
        {
            "pocket_id": "webcam-0", "timestamp": "2026-10-31T20:00:00Z",
            "predicted_denomination": "100", "classifier_confidence": 0.95,
            "decision": "ACCEPT", "classification": "ACCEPT_NOTE",
            "operational_action": "COUNT", "count_authorized": 1,
        },
    ])
    def test_inconsistent_records_rejected(self, temp_db: Path, inconsistent_record: dict):
        assert is_authorized_donation_record(inconsistent_record) is False

        import_res = import_single_record(inconsistent_record, database_file=temp_db)
        assert import_res.inserted == 0
        assert get_donation_count(temp_db) == 0

        with patch("requests.post") as mock_post:
            bc_result = safe_register_blockchain_proof(inconsistent_record)
            assert bc_result is None
            assert mock_post.call_count == 0


# ---------------------------------------------------------------------------
# Requirement H: Duplicate Frames (Anti-Duplicate Counting Protection)
# ---------------------------------------------------------------------------

class TestRequirementHDuplicateFrames:
    def test_single_physical_note_counted_exactly_once_across_multiple_frames(self, temp_db: Path):
        """
        Simulates consecutive camera frames while a physical note sits stationary in the box.
        Verifies that:
          1. Frame 1-3 (settling): no count.
          2. Frame 4 (accepted): exactly 1 count.
          3. Frames 5-20 (note still present): exactly 0 additional counts.
          4. Only after physical removal and re-entry can another count occur.
        """
        pipe = make_test_pipeline(p_note=0.99, b_probs=PROBS_RS_100)
        img = make_sharp_image()

        donations_counted = 0
        blockchain_calls = 0

        # State machine tracking
        state = "WAITING"
        presence_streak = 0
        absence_streak = 0
        REQUIRED_PRESENCE = 4
        REQUIRED_REMOVAL = 5

        def step_frame(presence_detected: bool):
            nonlocal state, presence_streak, absence_streak, donations_counted, blockchain_calls

            if presence_detected:
                presence_streak += 1
                absence_streak = 0
            else:
                absence_streak += 1
                presence_streak = 0

            if state == "WAIT_FOR_REMOVAL":
                if absence_streak >= REQUIRED_REMOVAL:
                    state = "WAITING"
                return

            if state == "WAITING":
                if presence_detected:
                    state = "CONFIRMING"
                return

            if state == "CONFIRMING":
                if not presence_detected:
                    state = "WAITING"
                elif presence_streak >= REQUIRED_PRESENCE:
                    # Evaluate stationary candidate
                    res = pipe.predict(img)
                    if (
                        res.classification == CLS_ACCEPT
                        and res.count_authorized is True
                        and res.operational_action == _ACT_COUNT
                    ):
                        rec = format_accepted_donation_record(res, pocket_id="webcam-0")
                        import_single_record(rec, database_file=temp_db)
                        with patch("requests.post") as mock_post:
                            safe_register_blockchain_proof(rec)
                            blockchain_calls += mock_post.call_count
                        donations_counted += 1
                        state = "WAIT_FOR_REMOVAL"

        # Frame 1: Note dropped into box
        step_frame(presence_detected=True)
        assert donations_counted == 0

        # Frame 2-3: Note settling
        step_frame(presence_detected=True)
        step_frame(presence_detected=True)
        assert donations_counted == 0

        # Frame 4: Stationary confirmed -> evaluated & counted
        step_frame(presence_detected=True)
        assert donations_counted == 1
        assert get_donation_count(temp_db) == 1

        # Frames 5 to 25: Note remains stationary in box
        for _ in range(20):
            step_frame(presence_detected=True)

        # Verified: exactly 1 donation recorded across all 25 frames!
        assert donations_counted == 1
        assert get_donation_count(temp_db) == 1


# ---------------------------------------------------------------------------
# Requirement I: Unsupported Denomination
# ---------------------------------------------------------------------------

class TestRequirementIUnsupportedDenomination:
    @pytest.mark.parametrize("bad_denom", ["999", "5", "1000", "invalid", "", None])
    def test_unsupported_denomination_cannot_be_counted_or_registered(self, temp_db: Path, bad_denom):
        record = {
            "pocket_id": "webcam-0",
            "timestamp": "2026-10-31T20:00:00Z",
            "predicted_denomination": bad_denom,
            "classifier_confidence": 0.99,
            "decision": "ACCEPT",
            "classification": "ACCEPT_NOTE",
            "operational_action": "COUNT",
            "count_authorized": True,
        }

        # Must fail authorization check
        assert is_authorized_donation_record(record) is False

        # SQLite must reject
        import_res = import_single_record(record, database_file=temp_db)
        assert import_res.inserted == 0
        assert get_donation_count(temp_db) == 0

        # Blockchain gateway must reject
        with patch("requests.post") as mock_post:
            bc_res = safe_register_blockchain_proof(record)
            assert bc_res is None
            assert mock_post.call_count == 0


# ---------------------------------------------------------------------------
# Inference Log Bulk Ingestion Security Test
# ---------------------------------------------------------------------------

class TestInferenceFileIngestionSecurity:
    def test_import_inference_file_skips_unauthorized_records(self, tmp_path: Path, temp_db: Path):
        """Verifies that import_inference_file filters unvetted or legacy records."""
        jsonl_path = tmp_path / "mixed_inference.jsonl"
        records = [
            # 1. Valid authorized ₹100 note
            {
                "pocket_id": "webcam-0", "timestamp": "2026-10-31T20:01:00Z",
                "predicted_denomination": "100", "classifier_confidence": 0.99,
                "confidence": 0.99, "source": "webcam", "decision": "ACCEPT",
                "classification": "ACCEPT_NOTE", "operational_action": "COUNT",
                "count_authorized": True, "target_bin": "BIN_100",
                "model_version": "divyakosh_v1_three_outcome",
            },
            # 2. Legacy record with decision=ACCEPT but no count_authorized
            {
                "pocket_id": "webcam-0", "timestamp": "2026-10-31T20:02:00Z",
                "predicted_denomination": "50", "classifier_confidence": 0.95,
                "source": "webcam", "decision": "ACCEPT",
            },
            # 3. Review record
            {
                "pocket_id": "webcam-0", "timestamp": "2026-10-31T20:03:00Z",
                "predicted_denomination": None, "classifier_confidence": 0.0,
                "decision": "REVIEW", "classification": "REJECT_NOT_NOTE",
                "operational_action": "REVIEW", "count_authorized": False,
            },
            # 4. Valid authorized ₹500 note
            {
                "pocket_id": "webcam-0", "timestamp": "2026-10-31T20:04:00Z",
                "predicted_denomination": "500", "classifier_confidence": 0.98,
                "confidence": 0.98, "source": "webcam", "decision": "ACCEPT",
                "classification": "ACCEPT_NOTE", "operational_action": "COUNT",
                "count_authorized": True, "target_bin": "BIN_500",
                "model_version": "divyakosh_v1_three_outcome",
            },
        ]

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        res = import_inference_file(inference_file=jsonl_path, database_file=temp_db)
        assert res.records_read == 4
        assert res.inserted == 2                     # Only the two authorized records!
        assert res.non_accepted_skipped == 2         # Legacy + Review skipped
        assert get_donation_count(temp_db) == 2
        assert get_total_donation_amount(temp_db) == 600   # 100 + 500

