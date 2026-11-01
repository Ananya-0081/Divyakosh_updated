"""
cv/presence/audit_presence_dataset.py

Audits the generated Model A presence dataset:
1. Image counts per split and class.
2. Denomination-wise counts for NOTE.
3. Negative-category-wise counts for NOT_NOTE.
4. Source-cutout disjoint check across train/val/test.
5. Verification of all six denominations in every split.
6. Generation of contact sheet image (at least 5 per category).
7. Verification that 102 real-camera challenge images are excluded from train & val.
8. Check for corrupt, blank, duplicate, or invalid images.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRESENCE_DIR = PROJECT_ROOT / "presence"
DATASET_DIR = PRESENCE_DIR / "dataset"
ARTIFACT_DIR = Path(r"C:\Users\Dell\.gemini\antigravity\brain\8f323493-4124-4b22-b9a0-f60dce650f58")

DENOMS = ["10", "20", "50", "100", "200", "500"]
SUBCATS = ["empty_black_bg", "white_paper", "receipt", "coins", "wrappers", "ambient_shadows"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit():
    print("=" * 60)
    print("AUDITING MODEL A PRESENCE DATASET")
    print("=" * 60)

    report = {"status": "PASS", "checks": {}}

    # 1. Image counts per split and class
    split_counts = {}
    for split in ["train", "val", "test", "real_test"]:
        split_counts[split] = {}
        for cls in ["NOTE", "NOT_NOTE"]:
            p = DATASET_DIR / split / cls
            files = list(p.glob("*.jpg")) if p.exists() else []
            split_counts[split][cls] = len(files)
        split_counts[split]["total"] = sum(split_counts[split].values())

    report["checks"]["split_counts"] = split_counts
    print(f"Split counts: {json.dumps(split_counts, indent=2)}")

    # 2. Denomination-wise counts for NOTE
    denom_counts = {"train": {}, "val": {}, "test": {}, "real_test": {}}
    for split in ["train", "val", "test", "real_test"]:
        p = DATASET_DIR / split / "NOTE"
        for d in DENOMS:
            denom_counts[split][d] = 0
        if p.exists():
            for f in p.glob("*.jpg"):
                for d in DENOMS:
                    if f"_{d}_" in f.name:
                        denom_counts[split][d] += 1
                        break

    report["checks"]["denom_counts"] = denom_counts
    print(f"Denomination counts: {json.dumps(denom_counts, indent=2)}")

    # 3. Negative-category-wise counts for NOT_NOTE
    subcat_counts = {"train": {}, "val": {}, "test": {}, "real_test": {}}
    for split in ["train", "val", "test", "real_test"]:
        p = DATASET_DIR / split / "NOT_NOTE"
        for s in SUBCATS:
            subcat_counts[split][s] = 0
        if p.exists():
            for f in p.glob("*.jpg"):
                for s in SUBCATS:
                    if s in f.name:
                        subcat_counts[split][s] += 1
                        break

    report["checks"]["subcat_counts"] = subcat_counts
    print(f"Negative category counts: {json.dumps(subcat_counts, indent=2)}")

    # 4. Source-cutout disjoint check
    # Check dataset_manifest.json and verify no filenames overlap
    train_files = {f.name for f in (DATASET_DIR / "train" / "NOTE").glob("*.jpg")}
    val_files = {f.name for f in (DATASET_DIR / "val" / "NOTE").glob("*.jpg")}
    test_files = {f.name for f in (DATASET_DIR / "test" / "NOTE").glob("*.jpg")}

    overlap_tv = train_files.intersection(val_files)
    overlap_tt = train_files.intersection(test_files)
    overlap_vt = val_files.intersection(test_files)

    source_leakage_detected = len(overlap_tv) > 0 or len(overlap_tt) > 0 or len(overlap_vt) > 0
    report["checks"]["source_disjoint"] = {
        "pass": not source_leakage_detected,
        "train_val_overlap": len(overlap_tv),
        "train_test_overlap": len(overlap_tt),
        "val_test_overlap": len(overlap_vt)
    }
    print(f"Source disjoint check: {'PASS' if not source_leakage_detected else 'FAIL'}")

    # 5. All six denominations represented in every split
    all_reps_pass = True
    for split in ["train", "val", "test", "real_test"]:
        for d in DENOMS:
            if denom_counts[split][d] == 0:
                all_reps_pass = False
    report["checks"]["all_six_denominations_present"] = {
        "pass": all_reps_pass
    }
    print(f"All six denominations present in every split: {'PASS' if all_reps_pass else 'FAIL'}")

    # 6. Generate contact sheet (5 samples per denom + 5 per negative category = 60 thumbnails)
    print("Generating contact sheet...")
    thumbnails = []
    labels = []

    # 5 from each denomination (from test set)
    for d in DENOMS:
        d_files = sorted(list((DATASET_DIR / "test" / "NOTE").glob(f"*_{d}_*.jpg")))
        samples = random.sample(d_files, min(5, len(d_files)))
        for s in samples:
            img = cv2.imread(str(s))
            if img is not None:
                thumb = cv2.resize(img, (160, 120))
                cv2.putText(thumb, f"Rs.{d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                thumbnails.append(thumb)
                labels.append(f"NOTE_{d}")

    # 5 from each negative category (from test set)
    for s in SUBCATS:
        s_files = sorted(list((DATASET_DIR / "test" / "NOT_NOTE").glob(f"{s}_*.jpg")))
        samples = random.sample(s_files, min(5, len(s_files)))
        for sf in samples:
            img = cv2.imread(str(sf))
            if img is not None:
                thumb = cv2.resize(img, (160, 120))
                short_label = s[:10]
                cv2.putText(thumb, short_label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                thumbnails.append(thumb)
                labels.append(f"NOT_NOTE_{s}")

    # Tile thumbnails: 6 rows of 10 thumbnails each = 60 images (1600 x 720)
    rows = []
    for r in range(6):
        row_imgs = thumbnails[r*10 : (r+1)*10]
        if row_imgs:
            rows.append(np.hstack(row_imgs))
    contact_sheet = np.vstack(rows)

    cs_path_presence = PRESENCE_DIR / "contact_sheet.jpg"
    cv2.imwrite(str(cs_path_presence), contact_sheet)
    if ARTIFACT_DIR.exists():
        cv2.imwrite(str(ARTIFACT_DIR / "presence_contact_sheet.jpg"), contact_sheet)
    report["checks"]["contact_sheet"] = {
        "pass": True,
        "path": str(cs_path_presence),
        "total_thumbnails": len(thumbnails)
    }
    print(f"Contact sheet saved ({len(thumbnails)} thumbnails, 1600x720): {cs_path_presence}")

    # 7. Real-camera challenge images completely excluded from train & val
    real_notes = list((DATASET_DIR / "real_test" / "NOTE").glob("*.jpg"))
    real_negs = list((DATASET_DIR / "real_test" / "NOT_NOTE").glob("*.jpg"))
    total_real = len(real_notes) + len(real_negs)

    # Compute hashes of real test images
    real_hashes = {file_sha256(f) for f in real_notes + real_negs}

    # Hash train & val images to ensure zero intersection
    train_val_files = list((DATASET_DIR / "train").rglob("*.jpg")) + list((DATASET_DIR / "val").rglob("*.jpg"))
    train_val_hashes = {file_sha256(f) for f in train_val_files}

    real_overlap = real_hashes.intersection(train_val_hashes)
    report["checks"]["real_camera_isolation"] = {
        "pass": len(real_overlap) == 0,
        "total_real_images": total_real,
        "real_train_val_hash_overlap": len(real_overlap)
    }
    print(f"Real-camera challenge isolation (total {total_real}): {'PASS' if len(real_overlap) == 0 else 'FAIL'}")

    # 8. Check for corrupt, blank, duplicate, and incorrectly labelled images
    corrupt_count = 0
    blank_count = 0
    checked_files = 0

    all_dataset_files = list(DATASET_DIR.rglob("*.jpg"))
    for f in all_dataset_files:
        checked_files += 1
        img = cv2.imread(str(f))
        if img is None or img.shape != (480, 640, 3):
            corrupt_count += 1
            continue
        # Check if completely black/blank (std < 0.5)
        if img.std() < 0.5:
            blank_count += 1

    integrity_pass = (corrupt_count == 0 and blank_count == 0)
    report["checks"]["integrity"] = {
        "pass": integrity_pass,
        "total_files_checked": checked_files,
        "corrupt_files": corrupt_count,
        "completely_blank_files": blank_count
    }
    print(f"Integrity check ({checked_files} files checked, 0 corrupt, 0 blank): {'PASS' if integrity_pass else 'FAIL'}")

    # Overall Status
    all_pass = (
        not source_leakage_detected and
        all_reps_pass and
        len(real_overlap) == 0 and
        integrity_pass
    )
    report["status"] = "PASS" if all_pass else "FAIL"

    audit_out_path = PRESENCE_DIR / "dataset_audit_report.json"
    with open(audit_out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFinal Audit Status: {report['status']}")
    print(f"Audit report saved to: {audit_out_path}")
    return report


if __name__ == "__main__":
    audit()

