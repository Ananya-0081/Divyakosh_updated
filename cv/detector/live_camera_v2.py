"""
cv/detector/live_camera_v2.py

DivyaKosh live camera pipeline v2 — YOLO-gated two-stage inference.

Pipeline (per frame):
    Camera
      └─► YOLO banknote detector (banknote_detector.pt)
            ├─ No detection → state machine tick (IDLE / WAIT_FOR_REMOVAL)
            └─ Detection(s) → pick largest box
                  └─► State machine tick
                        └─ If state == CLASSIFYING:
                              crop = frame[bbox]
                              label, conf = MobileNetV3(crop)
                              State machine tick with cls_result
                                    └─ If ACCEPTED:
                                          write JSONL
                                          write SQLite
                                          register blockchain proof
                                          state.mark_accepted()
                                          └─ WAIT_FOR_REMOVAL → IDLE

Key improvements over live_camera.py:
  ✓ No non-banknote object can reach the MobileNetV3 classifier
  ✓ Temporal state machine requires detection streak + stable bbox + agreeing predictions
  ✓ Continuous session — processes multiple notes per run (no exit after first)
  ✓ WAIT_FOR_REMOVAL prevents double-counting the same physical note
  ✓ MobileNetV3 only runs on YOLO-detected crop, not fixed ROI

Usage:
    cd cv
    .venv/Scripts/python detector/live_camera_v2.py
    .venv/Scripts/python detector/live_camera_v2.py --camera 1
    .venv/Scripts/python detector/live_camera_v2.py --det_conf 0.35 --cls_thresh 0.90
    .venv/Scripts/python detector/live_camera_v2.py --no_blockchain  # skip proof registration

Press Q to quit at any time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from detector.state_machine import BBox, DonationStateMachine, State, StateMachineConfig

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DETECTOR_PT  = _PROJECT_ROOT / "models" / "banknote_detector.pt"
_CLASSIFIER_PT = _PROJECT_ROOT / "models" / "mobilenet_v3_clean.pt"
_LOG_PATH     = _PROJECT_ROOT / "logs" / "detector_v2_experimental.jsonl"
_MODEL_VERSION = "mobilenet_v3_clean_v1+yolov8n_detector_experimental"

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DENOM_LABELS = {"10":"Rs.10","20":"Rs.20","50":"Rs.50","100":"Rs.100","200":"Rs.200","500":"Rs.500"}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_detector(det_pt: Path) -> object:
    """Load YOLOv8n banknote detector."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed.\n"
            "Run: .venv/Scripts/pip install 'ultralytics>=8.3.0'"
        )
    if not det_pt.exists():
        raise FileNotFoundError(
            f"Detector not found: {det_pt}\n"
            "Run: python detector/train_detector.py"
        )
    model = YOLO(str(det_pt))
    print(f"Detector    : {det_pt.name}")
    return model


def load_classifier(cls_pt: Path, device: torch.device) -> tuple[nn.Module, list[str]]:
    """Load existing MobileNetV3 denomination classifier."""
    ckpt = torch.load(cls_pt, map_location=device, weights_only=True)
    class_names = ckpt["class_names"]
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Classifier  : {cls_pt.name}  classes={class_names}")
    return model, class_names


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_detector(yolo_model, frame_bgr: np.ndarray, conf: float, imgsz: int) -> list[BBox]:
    """Run YOLO and return list of BBox objects above confidence threshold."""
    results = yolo_model(frame_bgr, conf=conf, imgsz=imgsz, verbose=False)
    boxes: list[BBox] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            c = float(box.conf[0])
            boxes.append(BBox(x1, y1, x2, y2, c))
    return boxes


@torch.inference_mode()
def run_classifier(
    cls_model: nn.Module,
    class_names: list[str],
    crop_bgr: np.ndarray,
    device: torch.device,
) -> tuple[str, float]:
    """Run MobileNetV3 on a BGR crop and return (label, confidence)."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = EVAL_TRANSFORM(pil).unsqueeze(0).to(device)
    logits = cls_model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = torch.max(probs, 0)
    return class_names[idx.item()], float(conf.item())


# ---------------------------------------------------------------------------
# UI overlay
# ---------------------------------------------------------------------------

_STATE_COLORS = {
    State.IDLE:             (180, 180, 180),
    State.NOTE_DETECTED:    (0, 200, 255),
    State.STABILIZING:      (0, 165, 255),
    State.CLASSIFYING:      (50, 205, 50),
    State.ACCEPTED:         (0, 220, 0),
    State.WAIT_FOR_REMOVAL: (0, 100, 255),
}


def draw_overlay(
    frame: np.ndarray,
    sm_result,
    best_box: BBox | None,
    fps: float,
    session_count: int,
) -> None:
    h, w = frame.shape[:2]
    color = _STATE_COLORS.get(sm_result.state, (200, 200, 200))

    # Detection bbox
    if best_box is not None:
        x1, y1, x2, y2 = best_box.as_xyxy()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{best_box.confidence:.0%}", (x1, max(15, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Status bar at bottom
    bar_h = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    status = sm_result.status_text
    if sm_result.accepted_label:
        label_disp = DENOM_LABELS.get(sm_result.accepted_label, sm_result.accepted_label)
        conf_disp  = f"{sm_result.accepted_confidence*100:.1f}%"
        status = f"ACCEPTED: {label_disp}  {conf_disp}"

    cv2.putText(frame, status,         (15, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, "DaanDristi v2",(15, 30),   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS:{fps:4.1f}", (w-110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Session donations: {session_count}", (15, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q=quit", (w-90, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Record writing
# ---------------------------------------------------------------------------

def write_record(
    log_path: Path,
    camera_idx: int,
    label: str,
    confidence: float,
    bbox: BBox | None,
    det_confidence: float,
    no_blockchain: bool,
) -> dict:
    """
    EXPERIMENTAL LOGGING ONLY.
    This script is retained strictly for YOLO detector experimentation.
    It does NOT have production donation recording authority.
    Production donation counting is exclusively performed by scripts/live_camera.py
    via DivyaKoshPipeline.
    """
    record = {
        "pocket_id": f"webcam-{camera_idx}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "item_status": "EXPERIMENTAL_NOTE_DETECTED",
        "predicted_denomination": label,
        "classifier_confidence": confidence,
        "confidence": confidence,
        "source": "webcam_experimental_v2",
        "decision": "EXPERIMENTAL_ACCEPT",
        "classification": "EXPERIMENTAL_YOLO_CANDIDATE",
        "operational_action": "REVIEW",
        "count_authorized": False,
        "target_bin": f"BIN_{label}",
        "model_version": _MODEL_VERSION,
        "detector_confidence": round(det_confidence, 4) if bbox else None,
        "detector_bbox": list(bbox.as_xyxy()) if bbox else None,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"[EXPERIMENTAL LOG] {log_path} (count_authorized=False, no production database/blockchain entry)")
    return record


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DivyaKosh live camera v2 (YOLO+MobileNetV3)")
    parser.add_argument("--camera",      type=int,   default=0)
    parser.add_argument("--detector",    type=str,   default=str(_DETECTOR_PT))
    parser.add_argument("--classifier",  type=str,   default=str(_CLASSIFIER_PT))
    parser.add_argument("--log_path",    type=str,   default=str(_LOG_PATH))
    parser.add_argument("--det_conf",    type=float, default=0.40, help="YOLO detection confidence")
    parser.add_argument("--det_imgsz",   type=int,   default=320,  help="YOLO inference image size")
    parser.add_argument("--cls_thresh",  type=float, default=0.90, help="MobileNetV3 acceptance threshold")
    parser.add_argument("--streak",      type=int,   default=4,    help="Detection streak required")
    parser.add_argument("--agreement",   type=int,   default=3,    help="Classifier agreement required")
    parser.add_argument("--no_blockchain", action="store_true",    help="Skip blockchain proof registration")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")

    detector   = load_detector(Path(args.detector))
    classifier, class_names = load_classifier(Path(args.classifier), device)

    sm = DonationStateMachine(StateMachineConfig(
        det_confidence  = args.det_conf,
        streak_frames   = args.streak,
        cls_threshold   = args.cls_thresh,
        cls_agreement   = args.agreement,
    ))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    log_path      = Path(args.log_path)
    session_count = 0
    prev_time     = time.perf_counter()
    fps           = 0.0
    last_box: BBox | None = None

    print(f"\nCamera {args.camera} opened. Press Q to quit.")
    print(f"Waiting for banknotes...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed.")
                break

            # FPS
            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Detector
            detections = run_detector(detector, frame, args.det_conf, args.det_imgsz)
            best_box   = max(detections, key=lambda b: (b.x2-b.x1)*(b.y2-b.y1)) if detections else None
            if best_box:
                last_box = best_box

            # Classifier (only when state machine requests it)
            cls_result = None
            if sm.state == State.CLASSIFYING and best_box is not None:
                x1, y1, x2, y2 = best_box.as_xyxy()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        cls_result = run_classifier(classifier, class_names, crop, device)

            # State machine
            sm_result = sm.tick(detections, cls_result)

            # Accepted → write record, advance to WAIT_FOR_REMOVAL
            if sm_result.state == State.ACCEPTED:
                print(f"\n>>> ACCEPTED: Rs.{sm_result.accepted_label}  conf={sm_result.accepted_confidence:.2%}")
                write_record(
                    log_path, args.camera,
                    sm_result.accepted_label, sm_result.accepted_confidence,
                    last_box, last_box.confidence if last_box else 0.0,
                    args.no_blockchain,
                )
                session_count += 1
                sm.mark_accepted()

            # Draw
            draw_overlay(frame, sm_result, best_box, fps, session_count)
            cv2.imshow("DaanDristi v2 — YOLO+MobileNetV3", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession ended. {session_count} donation(s) recorded.")


if __name__ == "__main__":
    main()

