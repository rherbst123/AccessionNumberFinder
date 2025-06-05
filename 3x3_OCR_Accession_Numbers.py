#!/usr/bin/env python3
"""
accession_scraper.py
────────────────────
Extract 6-7-digit accession numbers from herbarium images listed in a URL file.

• Input  : a .txt file – one image URL per line
• Output : results_<inputfilename>_<date>.csv
• Images : downloaded to a default or user-specified directory

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
from datetime import datetime


MIN_LEN = 6           # minimum digits in accession number
MAX_LEN = 7           # maximum



def make_csv_path(input_file: Path) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    base = input_file.stem
    return Path(f"{base}_{date_str}.csv")


def find_numbers(text: str):
    pattern = rf"\b\d{{{MIN_LEN},{MAX_LEN}}}\b"
    return sorted(set(re.findall(pattern, text)))


def download_image(url: str, seq: int, download_dir: Path) -> Path | None:
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    except Exception as exc:
        print(f"[DL]  Failed {url}  ⇒  {exc}")
        return None

    name = unquote(Path(urlparse(url).path).name) or f"img_{seq:04d}.jpg"
    local_path = download_dir / name
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path


def split_into_chunks_3x3(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    chunks = []
    h_third, w_third = h // 3, w // 3
    for row in range(3):
        for col in range(3):
            y0, y1 = row * h_third, (row + 1) * h_third
            x0, x1 = col * w_third, (col + 1) * w_third
            chunks.append(img[y0:y1, x0:x1])
    return chunks


def extract_accession(img_path: Path, reader) -> str | None:
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[OCR] Could not read {img_path}")
        return None

    #Try each 3×3 chunk
    for chunk in split_into_chunks_3x3(img):
        text = " ".join(reader.readtext(chunk, detail=0, paragraph=False))
        nums = find_numbers(text)
        if nums:
            return nums[0]

    # Fallback: try the entire image
    print("     ↳ fallback: full image scan")
    full_text = " ".join(reader.readtext(img, detail=0, paragraph=False))
    nums = find_numbers(full_text)
    if nums:
        return nums[0]

    return None



def main(url_file: Path, download_dir: Path):
    if not url_file.is_file():
        sys.exit(f"No such file: {url_file}")

    urls = [line.strip() for line in url_file.read_text().splitlines() if line.strip()]
    if not urls:
        sys.exit("URL file is empty.")

    download_dir.mkdir(parents=True, exist_ok=True)
    csv_path = make_csv_path(url_file)
    is_new_file = not csv_path.exists()

    reader = easyocr.Reader(["en"], gpu=True)

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["url", "image_name", "accession_number"])

        for idx, url in enumerate(urls, 1):
            print(f"[{idx}/{len(urls)}] {url}")
            img_path = download_image(url, idx, download_dir)
            if img_path is None:
                row = [url, "<download-error>", ""]
            else:
                acc = extract_accession(img_path, reader)
                row = [url, img_path.name, acc or "<Error, Please Double Check>"]
                print(f"     ↳ {acc or 'no number found'}")

            writer.writerow(row)
            f.flush()

    print(f"\nDone – results in {csv_path.resolve()}")


if __name__ == "__main__":
    if not (2 <= len(sys.argv) <= 3):
        print("Usage: python accession_scraper.py urls.txt [download_dir]")
        sys.exit(1)

    url_file = Path(sys.argv[1])
    download_dir = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("downloaded_images")
    main(url_file, download_dir)
