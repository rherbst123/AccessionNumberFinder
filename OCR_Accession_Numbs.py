#!/usr/bin/env python3
"""
interactive_extract_numbers.py
──────────────────────────────
Extract 6–7-digit numbers from images using EasyOCR.

• Dataset type 1: numbers live in the **bottom-left** corner.
• Dataset type 2: numbers live on the **right‐hand third**.

The script walks every image in the folder (recursively), crops only the
relevant region, runs OCR, and writes “image_name<TAB>numbers” lines to a
text file.
"""
import os
import re
from pathlib import Path
import cv2
import numpy as np
import easyocr

# ---------------------------------------------------------------------------
# Configurable constants (tweak only if needed)
MIN_LEN = 6                 # shortest numeric string to accept
MAX_LEN = 7                 # longest
BL_FRAC = 0.35              # crop 35 % of width/height for bottom-left ROI
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
# ---------------------------------------------------------------------------


def crop_bottom_left(img: np.ndarray, frac: float = BL_FRAC) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(h * (1 - frac)) : h, 0 : int(w * frac)]


def crop_right_third(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[:, int(w * 2 / 3) : w]


def find_numbers(text: str, min_len: int = MIN_LEN, max_len: int = MAX_LEN):
    return re.findall(fr"\b\d{{{min_len},{max_len}}}\b", text)


def process_image(path: Path, reader, roi_fn):
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARN] Could not read {path}")
        return []

    roi = roi_fn(img)
    ocr_lines = reader.readtext(roi, detail=0, paragraph=False)
    hits = find_numbers(" ".join(ocr_lines))
    return sorted(set(hits))           


def main():
    print("Choose dataset layout:")
    print("  1) Bottom-left (accession # in bottom-left)   [default]")
    print("  2) Right-hand third (accession # on right side)")
    choice = input("Enter 1 or 2: ").strip()
    roi_fn = crop_bottom_left if choice != "2" else crop_right_third

    
    inp_dir = Path(input("Folder containing images: ").strip()).expanduser()
    while not inp_dir.is_dir():
        inp_dir = Path(input("❗  Not a folder. Try again: ").strip()).expanduser()

    
    default_out = Path(__file__).with_name("results.txt")
    out_path_str = input(f"Results file [{default_out}]: ").strip()
    out_path = Path(out_path_str) if out_path_str else default_out

    
    print("\nInitialising EasyOCR … (this can take a few seconds)")
    reader = easyocr.Reader(["en"], gpu=False)      


    results = []
    for root, _, files in os.walk(inp_dir):
        for fname in files:
            if Path(fname).suffix.lower() in IMG_EXTS:
                fpath = Path(root) / fname
                numbers = process_image(fpath, reader, roi_fn)
                line = f"{fname}\t{', '.join(numbers) if numbers else '<none>'}"
                results.append(line)
                print(line)            

   
    out_path.write_text("\n".join(results), encoding="utf-8")
    print(f"\nDone! {len(results)} images processed.")
    print(f"Results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
