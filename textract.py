#!/usr/bin/env python3
"""
accession_scraper.py
────────────────────
Extract 6-7-digit accession numbers from herbarium images using AWS Textract.

• Input  : a .txt file – one image URL per line
• Output : results_<inputfilename>_<date>.csv
• Images : downloaded to a default or user-specified directory

Dependencies
------------
pip install boto3 opencv-python-headless requests
"""
from pathlib import Path
from urllib.parse import urlparse, unquote
import csv
import re
import sys
import requests
import boto3
from datetime import datetime

# ─────────────── TUNABLE CONSTANTS ───────────────
MIN_LEN = 6
MAX_LEN = 7
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tiff"}
textract = boto3.client("textract")  # Default region; configure via ~/.aws
# ────────────────────────────────────────────────

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


def extract_accession(img_path: Path) -> str | None:
    try:
        with open(img_path, "rb") as image_file:
            img_bytes = image_file.read()

        response = textract.detect_document_text(Document={"Bytes": img_bytes})
        all_text = " ".join(
            block["Text"]
            for block in response.get("Blocks", [])
            if block["BlockType"] == "WORD"
        )
        nums = find_numbers(all_text)
        return nums[0] if nums else None
    except Exception as e:
        print(f"[OCR] Textract failed on {img_path.name}: {e}")
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
                acc = extract_accession(img_path)
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
