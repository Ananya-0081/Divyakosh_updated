# DivyaKosh Banknote Detector & Temporal Pipeline (v2)

This directory contains the upgraded two-stage detection and verification architecture for the DivyaKosh computer vision system.

## Key Upgrades
1. **Banknote Presence Detector (YOLOv8n)**:
   - Dedicated lightweight binary detector (`banknote`) running at 320x320 resolution (~19 ms on CPU).
   - Replaces naive frame motion / background difference gates.
2. **Preserved Denomination Classifier**:
   - Keeps `mobilenet_v3_clean.pt` intact.
   - Evaluated only on the tight bounding box cropped by the detector.
3. **Temporal State Machine (`state_machine.py`)**:
   - `IDLE` -> `NOTE_DETECTED` -> `STABILIZING` -> `CLASSIFYING` -> `ACCEPTED` -> `WAIT_FOR_REMOVAL` -> `IDLE`
   - Movement alone does not trigger a donation.
   - Multi-frame agreement required before finalizing any classification.
   - Prevents double-counting a single physical banknote.

## Directory Structure
- `prepare_detector_dataset.py`: Builds composite YOLO dataset using 5 `black_box` classes + verified Rs. 500 fallback.
- `train_detector.py`: Fine-tunes YOLOv8n detector and saves checkpoint to `models/banknote_detector.pt`.
- `state_machine.py`: Temporal state machine logic and bounding box stabilization.
- `live_camera_v2.py`: Production-ready webcam pipeline with UI overlay and blockchain/SQLite ingestion.
- `evaluate_pipeline.py`: Comprehensive test bench comparing false-positive resilience, latency, and denomination accuracy.
- `tests/test_state_machine.py`: 33 unit tests for temporal state transitions.
- `tests/test_pipeline.py`: Integration tests verifying rejection of empty frames, hands, and unstable motion.

