# Temple-Scenario Dataset Audit
## DivyaKosh CV – Model A (Presence Classifier)

**Audit Date:** 2026-10-31  
**Auditor:** Automated (read-only, no dataset modifications)  
**Status:** READ-ONLY — No training, no dataset modification, no checkpoint changes.

---

## 1. Existing Dataset Inventory

### 1.1 Split Summary

| Split | NOTE | NOT_NOTE | Total |
|---|---|---|---|
| train | 3,000 | 3,000 | 6,000 |
| val | 600 | 600 | 1,200 |
| test | 600 | 600 | 1,200 |
| real_test | 51 | 51 | 102 |
| **Grand Total** | **4,251** | **4,251** | **8,502** |

### 1.2 NOTE – Denomination Distribution

All NOTE images are synthetically composited from black-box source renders.  
₹500 images are sourced from `cv/data_clean/train/500/SYN_*` (QC-filtered from 1,910 → 1,174 valid).

| Denomination | Train | Val | Test | real_test | Total |
|---|---|---|---|---|---|
| ₹10 | 500 | 100 | 100 | 3 | 703 |
| ₹20 | 500 | 100 | 100 | 24 | 724 |
| ₹50 | 500 | 100 | 100 | 2 | 702 |
| ₹100 | 500 | 100 | 100 | 10 | 710 |
| ₹200 | 500 | 100 | 100 | 4 | 704 |
| ₹500 | 500 | 100 | 100 | 8 | 708 |
| **Total** | **3,000** | **600** | **600** | **51** | **4,251** |

All six denominations represented in every split. ✅

### 1.3 NOT_NOTE – Subcategory Distribution

All NOT_NOTE images are **procedurally generated** — no real photos exist anywhere in the dataset.

| Subcategory | Train | Val | Test | real_test* | Total |
|---|---|---|---|---|---|
| empty_black_bg | 600 | 120 | 120 | 9 | 849 |
| white_paper | 600 | 120 | 120 | 9 | 849 |
| receipt | 600 | 120 | 120 | 9 | 849 |
| coins | 600 | 120 | 120 | 8 | 848 |
| wrappers | 300 | 60 | 60 | 8 | 428 |
| ambient_shadows | 300 | 60 | 60 | 8 | 428 |
| **Total** | **3,000** | **600** | **600** | **51** | **4,251** |

*real_test NOT_NOTE images are `rc_challenge_*` files — procedurally generated challenge variants, NOT real-camera photos of physical objects.

### 1.4 real_test Set – Important Clarification

> **CRITICAL NOTE:** The `real_test/` split (102 images total) is structured as follows:
> - `real_test/NOTE/`: 51 **real-camera** banknote photographs (`rc_note_DENOM_NNN.jpg`)
> - `real_test/NOT_NOTE/`: 51 **procedurally-generated** challenge distractors (`rc_challenge_SUBCAT_NNN.jpg`)
>
> The NOT_NOTE entries in real_test are **synthetic**, not real photos of physical distractors.  
> This means there are **zero real-world photos** of any non-banknote physical object in the entire dataset.

---

## 2. Temple Scenario Coverage Gap Analysis

### 2.1 Object Coverage Matrix

| Category | In Train/Val/Test | In real_test | Source Type | Notes |
|---|---|---|---|---|
| Empty black floor | ✅ YES | ✅ YES (synthetic) | Procedural | Solid black with noise |
| White/grey paper | ✅ YES | ✅ YES (synthetic) | Procedural | Flat rectangle with text lines |
| Receipt / paper slip | ✅ YES | ✅ YES (synthetic) | Procedural | Narrow rectangle with horizontal lines |
| Metallic coins (generic) | ✅ YES | ✅ YES (synthetic) | Procedural | Metallic circles on black bg |
| Wrappers | ✅ YES (partial) | ✅ YES (synthetic) | Procedural | Foil/plastic rectangle shapes |
| Ambient shadows | ✅ YES (partial) | ✅ YES (synthetic) | Procedural | Dark gradient patches |
| **Marigold flowers (whole)** | ❌ NO | ❌ NO | — | Very common temple offering |
| **Marigold / rose petals** | ❌ NO | ❌ NO | — | Scattered petals, orange/yellow/red |
| **Prasad (ladoo, peda, modak)** | ❌ NO | ❌ NO | — | Round beige/brown sweet objects |
| **Dried / wilted flower petals** | ❌ NO | ❌ NO | — | Brown/dark scattered shapes |
| **Incense sticks (agarbatti)** | ❌ NO | ❌ NO | — | Thin sticks, sometimes dropped in |
| **Coconut pieces / husk** | ❌ NO | ❌ NO | — | Occasional offering |
| **Currency envelope / cover** | ❌ NO | ❌ NO | — | Banknote inside paper envelope |
| **Prayer slips / nameplates** | ❌ NO | ❌ NO | — | Handwritten small paper slips |
| **Clay / brass diyas (lamps)** | ❌ NO | ❌ NO | — | Oil lamps, clay-orange with bright spot |
| **Small coins (₹1, ₹2, ₹5)** | ⚠️ PARTIAL | ⚠️ PARTIAL | Procedural only | No real coin photos; size not denomination-accurate |
| **Mixed / layered offerings** | ❌ NO | ❌ NO | — | Multiple objects overlapping |
| **Real banknote photos** | ⚠️ real_test only | ✅ YES | Real camera | 51 real notes in real_test/NOTE only |
| **Real distractor photos** | ❌ NO | ❌ NO | — | 0 real-world non-banknote photos anywhere |

**Summary: 10 of 19 relevant temple scenario categories are completely absent from the dataset.**

### 2.2 Challenging Note Condition Coverage

| Condition | Represented | Source | Notes |
|---|---|---|---|
| Standard centered note | ✅ YES | Synthetic (all splits) | Core training examples |
| Rotation variation (±30°) | ✅ YES | Synthetic augmentation | Applied during generation |
| Scale variation (40–80% canvas) | ✅ YES | Synthetic | Not full-bleed |
| Lighting variation | ✅ YES | Synthetic brightness jitter | |
| **Edge-to-edge full-frame fill** | ❌ NO | — | **Primary FN root cause** (17/51 FNs) |
| **Severe optical blur / defocus** | ❌ NO | — | **Primary FN root cause** (FN sharpness 8.7× lower) |
| Folded / creased notes | ⚠️ PARTIAL | Via augment only | No dedicated crease simulation |
| Partially cut-off / bleed-off frame | ❌ NO | — | FN contributing cause |
| ₹500 grey low-saturation notes at close range | ⚠️ PARTIAL | Synthetic ₹500 exists | 5/8 real ₹500 missed (62.5% FN rate) |
| Wrinkled / worn notes | ❌ NO | — | Not yet simulated |
| Multiple notes overlapping | N/A | — | Out of scope (single-note scenario) |

---

## 3. Real-Camera Dataset Inspection

### 3.1 real_test/NOTE – Denomination Breakdown

| Denomination | Count | FN Count (threshold=0.50) | FN Rate |
|---|---|---|---|
| ₹10 | 3 | 0 | 0.0% |
| ₹20 | 24 | 7 | 29.2% |
| ₹50 | 2 | 1 | 50.0% |
| ₹100 | 10 | 4 | 40.0% |
| ₹200 | 4 | 0 | 0.0% |
| ₹500 | 8 | 5 | 62.5% |
| **Total** | **51** | **17** | **33.3%** |

See `FALSE_NEGATIVE_ANALYSIS.md` for per-image detail.

### 3.2 real_test/NOT_NOTE – False Positive Rate

| Subcategory | Count | FP Count | FP Rate |
|---|---|---|---|
| empty_black_bg | 9 | 0 | 0.0% |
| white_paper | 9 | 0 | 0.0% |
| receipt | 9 | 0 | 0.0% |
| coins | 8 | 0 | 0.0% |
| wrappers | 8 | 0 | 0.0% |
| ambient_shadows | 8 | 0 | 0.0% |
| **Total** | **51** | **0** | **0.0%** |

**All distractors in real_test are synthetic** — no real temple object false-positive data exists.

---

## 4. Existing Negative Category Visual Inspection

Contact sheet of 5 samples from each existing NOT_NOTE subcategory (test split):  
`cv/presence/analysis/existing_negatives_contact_sheet.jpg`

### 4.1 Visual Characteristics of Each Procedural Subcategory

| Subcategory | Visual Description | Temple Realism | Risk Level |
|---|---|---|---|
| `empty_black_bg` | Uniform black with pixel noise | High (matches empty box) | LOW |
| `white_paper` | White/grey rectangle, faint gridlines | Moderate (resembles paper slip) | MEDIUM |
| `receipt` | Narrow white strip with horizontal text bands | Low (receipt shape, but not temple-specific) | MEDIUM |
| `coins` | Metallic circles with radial gradient, 1–3 objects | Low (generic metallic disk, not sized for ₹1/2/5) | MEDIUM-HIGH |
| `wrappers` | Shiny foil rectangle with colour stripes | Low (candy wrapper-like) | LOW-MEDIUM |
| `ambient_shadows` | Dark gradient patches on black background | Moderate (shadows from box interior) | LOW |

### 4.2 Key Observations

1. **`coins`:** Procedural circles look plausible but are not sized or textured like real Indian coins. Real ₹1/₹2/₹5 coins have distinct milled edges and surface relief not represented.
2. **`white_paper`:** Could be confused with a prayer slip or temple receipt; current generator produces clean paper, not handwritten slips.
3. **`receipt`:** Currently generates generic printed receipts. Temple-specific offerings (prayer chits, donation slips) are not represented.
4. **All subcategories:** Are perfectly rendered (no motion blur, no glare, no perspective distortion). Real objects in a donation box will have shadows, specular reflections, and variable orientation.

---

## 5. False-Positive and False-Negative Risk in Temple Deployment

### 5.1 Highest False-Positive Risk Scenarios (NOT yet tested)

| Temple Scenario | Risk | Reason |
|---|---|---|
| Yellow/orange marigold flower | **HIGH** | Bright colourful object not in training; model has not seen it |
| Prasad (pale yellow ladoo) | **HIGH** | Roundish textured object; no training examples |
| Currency envelope (white) | **HIGH** | Large white rectangle very similar to `white_paper`; but envelope has a flap that may look like note edges |
| Folded prayer slip | **MEDIUM** | Similar to `receipt` but smaller and handwritten |
| Small ₹1/₹2 coin | **MEDIUM** | Existing `coins` class may handle; but real coins have different texture |
| Diya lamp | **MEDIUM** | Clay-coloured circular object not present in training |
| Incense sticks | **LOW-MEDIUM** | Thin sticks; unusual shape for current model |
| Coconut piece | **LOW** | White/brown chunk; novel shape |

### 5.2 Highest False-Negative Risk Scenarios (Already identified)

| Note Condition | Risk | Evidence |
|---|---|---|
| ₹500 note edge-to-edge in frame | **CRITICAL** | 5/8 FNs, real-camera measured |
| Blurred/defocused note (any denom) | **HIGH** | Laplacian var FN=167 vs TP=1,466 |
| ₹100 / ₹20 close-fill frame | **HIGH** | 40% and 29% FN rates measured |
| Crumpled / folded note | **MEDIUM** | No specific training data |

---

## 6. Proposed Dataset Expansion Plan

> **This is a proposal only. No dataset files will be created until explicitly approved.**

### 6.1 New NOT_NOTE Subcategories

| Subcategory | Train | Val | Test | Source Strategy |
|---|---|---|---|---|
| `flower_marigold` | 500 | 100 | 100 | Procedural: orange/yellow multi-petal cluster on black; random scale/rotation |
| `flower_petals` | 500 | 100 | 100 | Procedural: scattered petal shapes (orange, red, yellow, white) |
| `prasad` | 400 | 80 | 80 | Procedural: globular sphere (ladoo/peda shape), beige/brown, 1–3 per frame |
| `coins_realistic` | 500 | 100 | 100 | Enhanced procedural: add ₹1/₹2/₹5 accurate diameter ratios, milled edge texture |
| `currency_envelope` | 300 | 60 | 60 | Procedural: note-sized white envelope shape with flap line and address bands |
| `folded_paper_slip` | 400 | 80 | 80 | Extended `receipt` generator: narrow, handwritten-style lines, random fold crease |
| `diya_lamp` | 300 | 60 | 60 | Procedural: clay-orange flattened disk with bright central flame spot |
| `mixed_offerings` | 300 | 60 | 60 | Multi-object composite: 2–4 items from above categories per frame |
| **New NOT_NOTE Total** | **3,200** | **640** | **640** | |

### 6.2 New Challenging NOTE Variations

| Subcategory | Train | Val | Test | Source Strategy |
|---|---|---|---|---|
| `note_fullbleed` | 600 | 120 | 120 | Scale synthetic note source to 95–130% canvas; crop to fill frame |
| `note_blurred` | 400 | 80 | 80 | Apply Gaussian blur (kernel 5–11, sigma 2–6) to existing synthetic notes |
| `note_creased` | 400 | 80 | 80 | Apply horizontal/diagonal fold simulation (brightness band) to synthetic notes |
| `note_500_closeup` | 400 | 80 | 80 | Zoom in specifically on ₹500 grey notes; ≥90% frame coverage |
| **New NOTE Total** | **1,800** | **360** | **360** | |

### 6.3 Expanded Dataset Summary

| Split | Existing NOTE | +New NOTE | Existing NOT_NOTE | +New NOT_NOTE | New Total |
|---|---|---|---|---|---|
| train | 3,000 | +1,800 | 3,000 | +3,200 | 11,000 |
| val | 600 | +360 | 600 | +640 | 2,200 |
| test | 600 | +360 | 600 | +640 | 2,200 |
| real_test | 51 | — | 51 | — | 102 (unchanged) |
| **Grand Total** | — | — | — | — | **~15,502** |

### 6.4 Data Split Leakage Rules for Expansion

1. **Procedural generators:** Use deterministic per-split random seeds. Example: `seed = base_seed + split_id * 10000 + subcat_id * 1000`. Verify seeds produce fully disjoint output ID sequences.
2. **No cross-split reuse:** Each generated image goes into exactly one split; generator state must not wrap between splits.
3. **₹500 source images:** Already disjoint by construction (sliced from 1,174-image pool with verified non-overlapping index ranges). No changes required.
4. **Real camera images (if ever added):** Group by physical capture session (date + lighting condition). Assign entire sessions to one split before any augmentation.
5. **New temple object real photos (future):** Follow the same session-based split assignment. Collect ≥50 images per category across ≥3 distinct sessions before inclusion.

---

## 7. Real-Camera Challenge Set Policy

### 7.1 Existing Challenge Set

`cv/presence/dataset/real_test/` — **102 images** — **DO NOT MODIFY**.

- All 51 NOTE images are real-camera banknote photographs.
- All 51 NOT_NOTE images are synthetic procedural distractors.
- This set must remain completely untouched through all future training iterations.
- It must be used only once per final evaluation, not for threshold tuning or hyperparameter selection.

### 7.2 Recommended Second Challenge Set (After Temple Photos Are Captured)

Capture a dedicated **Temple Object Real-Camera Challenge Set** separately:

| Target Category | Min Images | Capture Conditions |
|---|---|---|
| Marigold flowers | ≥20 | Inside physical box, various sizes/orientations |
| Flower petals (scattered) | ≥20 | Multiple petal counts per image |
| Prasad (ladoo/peda) | ≥20 | Various colors and surface textures |
| Real ₹1/₹2/₹5 coins | ≥20 | Single coin and clustered |
| Prayer slips / paper | ≥20 | Blank and handwritten |
| Currency envelopes | ≥15 | Sealed and partially open |
| Diyas | ≥15 | With and without wax/oil |
| Mixed offerings | ≥20 | Multiple items per frame |

**This second set must never be used for training. It is for evaluation only.**

---

## 8. Audit Conclusions

### 8.1 Pass/Fail Summary

| Check | Status | Detail |
|---|---|---|
| All six denominations in every split | ✅ PASS | 500 per denom in train, 100 in val/test |
| real_test excluded from training/val | ✅ PASS | 102 images in separate folder, not used in training |
| All NOT_NOTE types represented | ⚠️ PARTIAL | 6 synthetic subcategories; 0 real-world object photos |
| Temple objects represented | ❌ FAIL | 0 flower/prasad/diya/envelope images exist anywhere |
| Real-camera negative examples | ❌ FAIL | 0 real photos of non-banknote objects exist |
| Challenging note conditions covered | ❌ FAIL | No full-bleed, no blur, no crease training data |
| ₹500 low-saturation coverage | ⚠️ PARTIAL | Synthetic ₹500 exists; real ₹500 62.5% FN rate |
| Source-cutout disjoint across splits | ✅ PASS | Verified (pool slices are non-overlapping) |
| No corrupt/blank/duplicate images | ✅ PASS | Previously verified in DATASET_AUDIT_REPORT.md |
| Contact sheet generated | ✅ PASS | `cv/presence/analysis/existing_negatives_contact_sheet.jpg` |

### 8.2 Overall Assessment

**Model A performs excellently on its current training distribution (100% accuracy on synthetic test set) but has a significant domain gap for real temple deployment:**

1. **Zero real-world negative photos** — All distractors are procedural. Any novel object (flower, prasad, diya) that the model has never seen is an unknown risk.
2. **17/51 real banknotes missed** — Root causes (full-frame fill and blur) are not represented in training data.
3. **₹500 is the highest-risk denomination** — Grey saturation, close-up fill pattern causes 62.5% miss rate.
4. **The false-positive risk from temple objects is unknown** — No evaluation data exists for flowers, prasad, envelopes, or diyas.

### 8.3 Recommended Next Steps (Pending User Approval)

1. Implement and generate the 8 new NOT_NOTE subcategories (Section 6.1).
2. Implement and generate the 4 new NOTE variation subcategories (Section 6.2).
3. Retrain Model A on expanded dataset.
4. Re-evaluate on existing real_test (102 images) — keep it untouched.
5. Collect real temple-object photos for a second evaluation-only challenge set.
6. Only after (4) and (5): consider threshold adjustment using validation data.

---

*Audit is read-only. No files were modified, created, moved, or deleted during this audit.*  
*Report generated: 2026-10-31*

