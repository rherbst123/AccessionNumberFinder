#!/usr/bin/env python3
"""
accession_scraper.py
────────────────────
Download herbarium images listed in a text file, then extract 6–7-digit
accession numbers (no leading zeroes).  Two-stage OCR:

   1. EasyOCR on a 3 × 3 grid of tiles (fast).
      • Only accept hits with confidence ≥ EASY_CONF_THRESH.
   2. AWS Textract full-image scan (fallback).

CSV rows are flushed after every image so progress is never lost.
"""
from pathlib import Path
from urllib.parse import urlparse, unquote
import csv, re, sys, requests, cv2, numpy as np, easyocr, boto3
from datetime import datetime

# ─────────── Tunables ───────────
MIN_LEN             = 6
MAX_LEN             = 7
EASY_CONF_THRESH    = 0.80   # raise/lower if needed
PREFER_LONGEST      = True   # choose 7-digit over 6-digit when both seen
TEXTRACT            = boto3.client("textract")
# ────────────────────────────────

# pre-compiled regex: 6- or 7-digit, first digit 1-9
DIGIT_RE = re.compile(rf"\b[1-9]\d{{{MIN_LEN-1},{MAX_LEN-1}}}\b")

def make_csv_path(input_file: Path) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    return Path(f"{input_file.stem}_{date_str}.csv")


def find_numbers(text: str) -> list[str]:
    hits = DIGIT_RE.findall(text)
    if PREFER_LONGEST:
        hits.sort(key=len, reverse=True)
    return hits


def download_image(url: str, seq: int, out_dir: Path) -> Path | None:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"[DL]  Failed {url} ➜ {e}")
        return None

    name = unquote(Path(urlparse(url).path).name) or f"img_{seq:04d}.jpg"
    path = out_dir / name
    path.write_bytes(r.content)
    return path


def split_3x3(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    h_th, w_th = h // 3, w // 3
    return [img[r*h_th:(r+1)*h_th, c*w_th:(c+1)*w_th]
            for r in range(3) for c in range(3)]


def ocr_easyocr_chunks(img: np.ndarray, reader) -> str | None:
    for tile in split_3x3(img):
        for *_bbox, text, conf in reader.readtext(tile, detail=1, paragraph=False):
            if conf < EASY_CONF_THRESH:
                continue
            nums = find_numbers(text)
            if nums:
                return nums[0]           # already longest-preferred
    return None


def ocr_textract(img_path: Path) -> str | None:
    try:
        bytes_ = img_path.read_bytes()
        resp = TEXTRACT.detect_document_text(Document={"Bytes": bytes_})
        text = " ".join(b["Text"] for b in resp.get("Blocks", [])
                        if b["BlockType"] == "WORD")
        nums = find_numbers(text)
        return nums[0] if nums else None
    except Exception as e:
        print(f"[OCR] Textract failed on {img_path.name}: {e}")
        return None


def extract_accession(img_path: Path, reader) -> str | None:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[OCR] Could not read {img_path}")
        return None

    acc = ocr_easyocr_chunks(img, reader)
    if acc:
        return acc

    print("      ↳ fallback to Textract")
    return ocr_textract(img_path)


def main(url_file: Path, download_dir: Path):
    if not url_file.is_file():
        sys.exit(f"No such file: {url_file}")
    urls = [u.strip() for u in url_file.read_text().splitlines() if u.strip()]
    if not urls:
        sys.exit("URL file is empty.")

    download_dir.mkdir(parents=True, exist_ok=True)
    csv_path = make_csv_path(url_file)
    header_needed = not csv_path.exists()

    reader = easyocr.Reader(["en"], gpu=True)

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["url", "image_name", "accession_number"])

        for idx, url in enumerate(urls, 1):
            print(f"[{idx}/{len(urls)}] {url}")
            img_path = download_image(url, idx, download_dir)
            if img_path is None:
                row = [url, "<download-error>", ""]
            else:
                acc = extract_accession(img_path, reader)
                row = [url, img_path.name, acc or "<not-found>"]
                print(f"      ↳ {acc or 'no number found'}")
            w.writerow(row)
            f.flush()

    print(f"\nDone – results saved to {csv_path.resolve()}")


# ───────────── CLI ─────────────
if __name__ == "__main__":
    if not (2 <= len(sys.argv) <= 3):
        print("Usage: python accession_scraper.py urls.txt [download_dir]")
        sys.exit(1)

    url_list = Path(sys.argv[1])
    img_dir  = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("downloaded_images")
    main(url_list, img_dir)
