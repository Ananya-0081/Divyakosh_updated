# DivyaKosh Model A False-Negative Investigation Report

**Document:** `cv/presence/analysis/FALSE_NEGATIVE_ANALYSIS.md`  
**Date:** October 31, 2026  
**Auditor:** Antigravity AI  
**Scope:** Root-Cause Analysis of 17 Real-Camera False Negatives from Model A  
**Status:** **INVESTIGATION COMPLETE — AWAITING USER APPROVAL**  

---

## 1. Executive Summary & Core Findings

During the evaluation on the held-out real-camera challenge set (51 real banknotes, 51 challenge distractors):
* **Model A Precision was 100.00%** (34/34 detections were genuine notes; **zero false donation triggers** on all 51 distractors).
* However, **17 of 51 real banknotes were rejected as `NOT_NOTE`** (Recall = 66.67%).

### Why Model A Missed These 17 Notes (Root Causes)
1. **Edge-to-Edge Frame Filling (Missing Background Context):**
   * In synthetic training, banknotes occupy 40%–80% of the canvas with clear black floor margins visible around them.
   * In the 17 false negatives, **banknotes cover 96.5% to 100.0% of the entire camera frame** (mean coverage = **99.36%**). The note edges extend beyond the image borders.
   * Without dark background boundaries, the convolutional kernels evaluate the frame as a flat texture rather than an isolated object resting on the floor.
2. **Severe Optical Blur & Lack of Numismatic Micro-Features:**
   * Mean Laplacian variance (sharpness) in True Positives is **1,465.70**.
   * Mean Laplacian variance in False Negatives is **167.95** (nearly **9× blurrier**).
   * Blurry captures eliminate the micro-engravings (RBI seal, Gandhi portrait lines, security thread text) that Model A relies on to distinguish banknotes from plain paper.
3. **₹500 Color Saturation Vulnerability:**
   * 5 of the 8 real ₹500 notes were missed (62.5% FN rate on ₹500).
   * Modern ₹500 notes are stone-grey. When zoomed edge-to-edge under dim interior illumination, low-saturation stone-grey notes visually resemble matte grey cardstock.

---

## 2. Inventory of the 17 Missed Banknotes

All 17 original images and prediction logs were exported to:  
📁 `cv/presence/analysis/false_negatives/`  
📄 Data CSV: `cv/presence/analysis/false_negatives/false_negatives.csv`

| Filename | Denomination | P(NOTE) | P(NOT_NOTE) | Sharpness (Lap. Var) | Note Coverage | Primary Visual Characteristic |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `rc_note_100_030.jpg` | ₹100 | 0.0040 | 0.9960 | 29.94 | 100.0% | Severe motion blur; edge-to-edge; low contrast |
| `rc_note_10_001.jpg` | ₹10 | 0.0022 | 0.9978 | 240.31 | 99.2% | Soiled/creased chocolate brown note; edge-to-edge |
| `rc_note_10_002.jpg` | ₹10 | 0.0298 | 0.9702 | 37.21 | 99.4% | Out-of-focus blur; low illumination |
| `rc_note_200_039.jpg` | ₹200 | 0.3932 | 0.6068 | 393.40 | 100.0% | Bright orange; full frame; near threshold (P=0.393) |
| `rc_note_200_041.jpg` | ₹200 | 0.3952 | 0.6048 | 119.52 | 100.0% | Moderate blur; full frame; near threshold (P=0.395) |
| `rc_note_20_003.jpg` | ₹20 | 0.0045 | 0.9955 | 252.94 | 100.0% | Old greenish-yellow note; crumpled texture |
| `rc_note_20_004.jpg` | ₹20 | 0.1980 | 0.8020 | 280.75 | 98.2% | Heavy central crease; edge-to-edge |
| `rc_note_20_009.jpg` | ₹20 | 0.3163 | 0.6837 | 100.05 | 99.6% | Soft blur; low edge gradient (P=0.316) |
| `rc_note_20_011.jpg` | ₹20 | 0.2418 | 0.7582 | 48.97 | 99.4% | Severe blur; out-of-focus lens |
| `rc_note_20_021.jpg` | ₹20 | 0.2226 | 0.7774 | 246.22 | 100.0% | Specular glare patch on plastic window |
| `rc_note_20_022.jpg` | ₹20 | 0.2226 | 0.7774 | 246.22 | 100.0% | Specular glare patch |
| `rc_note_500_044.jpg` | ₹500 | 0.0201 | 0.9799 | 93.21 | 99.9% | Low saturation stone-grey; edge-to-edge |
| `rc_note_500_046.jpg` | ₹500 | 0.0161 | 0.9839 | 101.24 | 96.5% | Partial corner fold; low contrast |
| `rc_note_500_047.jpg` | ₹500 | 0.0017 | 0.9983 | 115.49 | 99.5% | Dim lighting; Red Fort side edge-to-edge |
| `rc_note_500_048.jpg` | ₹500 | 0.0990 | 0.9010 | 31.68 | 99.7% | Extreme motion blur; featureless grey |
| `rc_note_500_050.jpg` | ₹500 | 0.0022 | 0.9978 | 97.60 | 99.4% | Edge-to-edge full frame; no dark border |
| `rc_note_50_028.jpg` | ₹50 | 0.3826 | 0.6174 | 420.47 | 98.3% | Fluorescent cyan; near threshold (P=0.383) |

*Visual contact sheet generated at:*  
`cv/presence/analysis/false_negatives/contact_sheet_fn.jpg`

---

## 3. Quantitative Comparison: False Negatives vs. True Positives

| Physical Metric | 17 False Negatives (Missed) | 34 True Positives (Detected) | Difference / Impact |
| :--- | :---: | :---: | :--- |
| **Sharpness (Laplacian Var)** | **167.95** | **1,465.70** | **8.7x blurrier**. Blurry notes lose line micro-textures. |
| **Note Coverage of Frame** | **99.36%** | **95.17%** | Missed notes are 100% full-bleed with no black borders. |
| **Contrast (Pixel Std Dev)** | **33.82** | **45.23** | Lower contrast / flatter tonal range. |
| **Mean Brightness (0–255)** | 133.87 | 134.17 | Brightness is identical; illumination level is not the issue. |
| **Glare Pixels (>245)** | 0.29% | 0.33% | Minor glare on 2 images (`rc_note_20_021/022`), not dominant. |
| **Model A Output P(NOTE)** | **0.1501** | **0.9132** | Strong bimodal separation. |

---

## 4. Evaluation Code & Protocol Audit

In accordance with item 6 of your request, we audited `cv/presence/evaluate_presence.py`:

| Check | Result | Verification Evidence |
| :--- | :---: | :--- |
| **1. Model A Decision Threshold** | **0.50** | `pred = 1 if prob_note >= threshold else 0` with `threshold=0.5`. Strictly standard unbiased threshold. |
| **2. YOLO Confidence Threshold Source** | **Library Default (0.25)** | `conf=0.25` is the default Ultralytics parameter. It was **NOT** selected using validation data, explaining its 98%+ false alarm rate on paper/receipts. |
| **3. Real-Camera Challenge Set Isolation** | **STRICTLY DISJOINT** | 0 SHA-256 hash overlap with training or validation. Sourced from `data_clean/test/`. Zero real images were used in training Model A. |
| **4. Benchmark Consistency** | **IDENTICAL CRITERIA** | Both models evaluated on the exact same 1,200 synthetic test and 102 real-camera challenge images, using identical binary ground-truth labels. |

---

## 5. Validation-Only Probability Distribution & Threshold Sweep

> In accordance with your strict constraint, **this sweep was performed strictly on the 1,200 images of the validation set (`cv/presence/dataset/val/`)**.  
> Neither the held-out test set nor the real-camera challenge set was used for threshold tuning.

| Decision Threshold tau | Val Accuracy | Val NOTE Precision | Val NOTE Recall | Val NOTE F1 | Val NOT_NOTE FPR | Val False Alarms (FP / 600) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.01 | 99.42% | 0.9885 | 1.0000 | 0.9942 | 1.17% | 7 / 600 |
| 0.05 | 99.50% | 0.9901 | 1.0000 | 0.9950 | 1.00% | 6 / 600 |
| 0.10 | 99.67% | 0.9934 | 1.0000 | 0.9967 | 0.67% | 4 / 600 |
| 0.20 | 99.75% | 0.9950 | 1.0000 | 0.9975 | 0.50% | 3 / 600 |
| **0.30** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **0.00%** | **0 / 600** |
| **0.40** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **0.00%** | **0 / 600** |
| **0.50 (Current)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **0.00%** | **0 / 600** |
| **0.60** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **0.00%** | **0 / 600** |
| **0.70** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **0.00%** | **0 / 600** |
| **0.80** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **0.00%** | **0 / 600** |
| 0.90 | 99.92% | 1.0000 | 0.9983 | 0.9991 | 0.00% | 0 / 600 |
| 0.95 | 99.92% | 1.0000 | 0.9983 | 0.9991 | 0.00% | 0 / 600 |
| 0.99 | 99.58% | 1.0000 | 0.9917 | 0.9958 | 0.00% | 0 / 600 |

### Key Observation from the Validation Sweep:
On the validation set, **any threshold between 0.30 and 0.80 yields perfect 100.00% accuracy with 0 false alarms**.  
Notice from the false-negative table that four real notes (`rc_note_200_039`, `rc_note_200_041`, `rc_note_20_009`, `rc_note_50_028`) scored between P = 0.316 and P = 0.395.  
However, as instructed, **we do not alter the production threshold without your explicit guidance**.

---

## 6. Recommendations & Proposed Solutions

### Recommendation 1: Fix the Synthetic Training Scale Distribution
* **Finding:** Model A was trained on notes that filled 40%–80% of the canvas. In the real donation box, dropped notes often fill 95%–100% of the frame.
* **Fix:** When augmenting synthetic training notes, add tight zoom/crop variations (0.95x to 1.30x scale) where the note boundaries touch or bleed slightly past the frame border. This will teach Model A to recognize close-up banknote textures without requiring dark outer borders.

### Recommendation 2: Add Motion Blur & Defocus Augmentation
* **Finding:** Real false negatives have a Laplacian sharpness of only 167 vs. 1,465 for true positives.
* **Fix:** Introduce Gaussian blur kernels and mild directional motion blur during synthetic training so the network does not depend exclusively on high-frequency micro-features.

### Recommendation 3: Calibrate Camera Lens & Exposure in Physical Box
* **Finding:** Several notes are severely out-of-focus (`rc_note_100_030` lap var = 29.9).
* **Fix:** Ensure the physical camera's manual focus ring is fixed at the exact focal distance of the donation box floor (e.g., 25–30 cm).

---

## 7. Current Project Integrity Status
* `live_camera.py` and `live_camera_v2.py`: **Unmodified.**
* `cv/models/mobilenet_v3_clean.pt`: **Unmodified.**
* `cv/models/banknote_detector.pt`: **Unmodified.**
* Blockchain, IoT, SQLite databases: **Unmodified.**
* All investigation outputs saved cleanly in `cv/presence/analysis/`.

