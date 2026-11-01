# -*- coding: utf-8 -*-
"""Generate a contact sheet of 5 samples from each existing NOT_NOTE subcategory (test split)."""
import os, random, sys
from pathlib import Path
import cv2
import numpy as np

SEED = 42
random.seed(SEED)

BASE = Path(__file__).parent.parent / "dataset" / "test" / "NOT_NOTE"
OUT  = Path(__file__).parent / "existing_negatives_contact_sheet.jpg"

SUBCATS = ["empty_black_bg", "white_paper", "receipt", "coins", "wrappers", "ambient_shadows"]
N_PER_CAT = 5
THUMB_W, THUMB_H = 160, 120
LABEL_H = 22
BORDER = 4

cols = N_PER_CAT
rows = len(SUBCATS)
canvas_w = cols * (THUMB_W + BORDER) + BORDER
canvas_h = rows * (THUMB_H + LABEL_H + BORDER) + BORDER

canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)

for row, subcat in enumerate(SUBCATS):
    files = sorted(f for f in os.listdir(BASE) if f.startswith(subcat + "_"))
    chosen = random.sample(files, min(N_PER_CAT, len(files)))
    for col, fname in enumerate(chosen):
        img = cv2.imread(str(BASE / fname))
        if img is None:
            img = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
        img = cv2.resize(img, (THUMB_W, THUMB_H))
        x0 = BORDER + col * (THUMB_W + BORDER)
        y0 = BORDER + row * (THUMB_H + LABEL_H + BORDER)
        canvas[y0:y0+THUMB_H, x0:x0+THUMB_W] = img
        if col == 0:
            cv2.putText(canvas, subcat, (x0, y0 + THUMB_H + LABEL_H - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

cv2.imwrite(str(OUT), canvas)
print(f"Saved: {OUT}")

