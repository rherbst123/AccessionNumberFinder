#!/usr/bin/env python3
"""
accession_scraper.py  (parallel edition)
────────────────────────────────────────
Extract 6-7-digit accession numbers from herbarium images listed in a URL file,
running up to 4 worker PROCESSES concurrently.

Usage
-----
python accession_scraper.py urls.txt [ /path/to/download_dir ]

All other behaviour (timer, incremental CSV, fallback to right-third ROI, etc.)
is unchanged.

Dependencies
------------
pip install easyocr opencv-python-headless requests
"""
from __future__ import annotations
from pathlib import Path
from urllib.parse import urlparse, unquote
import concurrent.futures as cf
import csv
import re
import sys
import time
import requests
import cv2
import numpy as np
import easyocr

# ─────────────── TUNABLE CONSTANTS ───────────────
MIN_LEN   = 6          # minimum digits in accession number
MAX_LEN   = 7          # maximum digits
BL_FRAC   = 0.35       # bottom-left ROI size (35 %)
CSV_PATH  = Path("results.csv")
MAX_WORKERS = 4        # number of parallel processes
# ─────────────────────────────────────────────────


def crop_bottom_left(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(h * (1 - BL_FRAC)) :, : int(w * BL_FRAC)]


def crop_right_third(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[:, int(w * 2 / 3) :]


def find_numbers(text: str):
    """Return all unique 6-7 digit numbers in *text*."""
    pat = rf"\b\d{{{MIN_LEN},{MAX_LEN}}}\b"
    return sorted(set(re.findall(pat, text)))


def download_image(url: str, seq: int, img_dir: Path) -> Path | None:
    """Download *url* to *img_dir*; return local path or *None* on error."""
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
    except Exception as exc:
        print(f"[DL]  Failed {url}  ⇒  {exc}")
        return None

    name = unquote(Path(urlparse(url).path).name) or f"img_{seq:04d}.jpg"
    local = img_dir / name
    with open(local, "wb") as f:
        f.write(r.content)
    return local


# ────── Worker side helpers ──────
_reader: easyocr.Reader | None = None   # lazily initialised once per process


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        # gpu=True auto-falls back to CPU if no GPU
        _reader = easyocr.Reader(["en"], gpu=True)
    return _reader


def extract_accession(img_path: Path) -> str | None:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[OCR] Could not read {img_path}")
        return None

    for roi_fn in (crop_bottom_left, crop_right_third):
        roi = roi_fn(img)
        text = " ".join(get_reader().readtext(roi, detail=0, paragraph=False))
        nums = find_numbers(text)
        if nums:
            return nums[0]
    return None


def worker(task: tuple[str, int, str]) -> list[str]:
    """
    Parameters
    ----------
    task : (url, seq_no, img_dir_str)

    Returns
    -------
    row : list[str]  – exactly one CSV line [url, image_name, accession]
    """
    url, seq, img_dir_s = task
    img_dir = Path(img_dir_s)

    img_path = download_image(url, seq, img_dir)
    if img_path is None:
        return [url, "<download-error>", ""]

    try:
        acc = extract_accession(img_path)
    except Exception as e:               # any OCR or OpenCV failure
        print(f"[ERR] {img_path.name}: {e}")
        acc = None

    return [url, img_path.name, acc or "<Error, Please Double Check>"]


# ────── Driver / main process ──────
def main(url_file: Path, img_dir: Path):
    if not url_file.is_file():
        sys.exit(f"No such file: {url_file}")

    urls = [ln.strip() for ln in url_file.read_text().splitlines() if ln.strip()]
    if not urls:
        sys.exit("URL file is empty.")

    img_dir.mkdir(parents=True, exist_ok=True)

    # open CSV in append mode (crash-safe)
    csv_exists = CSV_PATH.exists()
    f_csv = CSV_PATH.open("a", newline="", encoding="utf-8")
    writer = csv.writer(f_csv)
    if not csv_exists:
        writer.writerow(["url", "image_name", "accession_number"])
        f_csv.flush()

    start = time.perf_counter()

    # Build (url, idx, img_dir) tuples for the pool
    tasks = [(u, i, str(img_dir)) for i, u in enumerate(urls, 1)]

    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        # submit *all* tasks at once; handle rows as they come in
        future_to_idx = {pool.submit(worker, t): t[1] for t in tasks}

        for fut in cf.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                row = fut.result()
            except Exception as e:                      # safety catch
                url = urls[idx - 1]
                print(f"[ERR] worker crash on {url}: {e}")
                row = [url, "<worker-crash>", ""]

            writer.writerow(row)
            f_csv.flush()

            done = row[2] if row[2] else "no number found"
            elapsed = time.perf_counter() - start
            print(f"[{idx}/{len(urls)}] ↳ {done}  —  elapsed {elapsed:.1f}s")

    f_csv.close()
    print(f"\nDone – results in {CSV_PATH.resolve()} "
          f"(total {time.perf_counter() - start:.1f}s)")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python accession_scraper.py urls.txt [/path/to/download_dir]")
        sys.exit(1)

    url_file = Path(sys.argv[1])
    img_dir  = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("downloaded_images")
    main(url_file, img_dir)
