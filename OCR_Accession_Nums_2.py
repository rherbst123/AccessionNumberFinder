#!/usr/bin/env python3
"""
accession_scraper.py
────────────────────
Extract 6-7-digit accession numbers from herbarium images listed in a URL file.

• Input  : a .txt file – one image URL per line
• Output : results.csv   (URL, image filename, accession number)

Dependencies
------------
pip install easyocr opencv-python-headless requests
"""
from pathlib import Path
from urllib.parse import urlparse, unquote
import csv
import re
import sys
import requests
import cv2
import numpy as np
import easyocr

# ─────────────── TUNABLE CONSTANTS ───────────────
MIN_LEN   = 6           # minimum digits in accession number
MAX_LEN   = 7           # maximum
BL_FRAC   = 0.35        # bottom-left ROI size (35 %)
IMG_DIR   = Path("downloaded_images")
CSV_PATH  = Path("results.csv")
IMG_DIR.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────


def crop_bottom_left(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(h * (1 - BL_FRAC)) :, : int(w * BL_FRAC)]


def crop_right_third(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[:, int(w * 2 / 3) :]


def find_numbers(text: str):
    """Return all unique 6-7 digit numbers in *text*."""
    pattern = rf"\b\d{{{MIN_LEN},{MAX_LEN}}}\b"
    return sorted(set(re.findall(pattern, text)))


def download_image(url: str, seq: int) -> Path | None:
    """Download *url* to IMG_DIR, return local path (or None on error)."""
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    except Exception as exc:
        print(f"[DL]  Failed {url}  ⇒  {exc}")
        return None

    # Derive filename: use last URL segment; fall back to seq#.jpg
    name = unquote(Path(urlparse(url).path).name) or f"img_{seq:04d}.jpg"
    local_path = IMG_DIR / name
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path


def extract_accession(img_path: Path, reader) -> str | None:
    """Return accession number or None."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[OCR] Could not read {img_path}")
        return None

    # 1️⃣ bottom-left first
    for roi_func in (crop_bottom_left, crop_right_third):
        roi = roi_func(img)
        text = " ".join(reader.readtext(roi, detail=0, paragraph=False))
        nums = find_numbers(text)
        if nums:
            return nums[0]            # take the first hit

    return None


def main(url_file: Path):
    if not url_file.is_file():
        sys.exit(f"No such file: {url_file}")

    urls = [line.strip() for line in url_file.read_text().splitlines() if line.strip()]
    if not urls:
        sys.exit("URL file is empty.")

    reader = easyocr.Reader(["en"], gpu=False)

    rows = []
    for idx, url in enumerate(urls, 1):
        print(f"[{idx}/{len(urls)}] {url}")
        img_path = download_image(url, idx)
        if img_path is None:
            rows.append([url, "<download-error>", ""])
            continue

        acc = extract_accession(img_path, reader)
        rows.append([url, img_path.name, acc or "<Error, Please Double Check>"])
        print(f"     ↳ {acc or 'no number found'}")

    # write CSV
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "image_name", "accession_number"])
        writer.writerows(rows)

    print(f"\nDone – results in {CSV_PATH.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python accession_scraper.py urls.txt")
        sys.exit(1)
    main(Path(sys.argv[1]))
