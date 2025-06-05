
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


MIN_LEN   = 6           # minimum digits in accession number
MAX_LEN   = 7           # maximum
BL_FRAC   = 0.35        # bottom-left ROI size (35 %)
IMG_DIR   = Path("downloaded_images")
IMG_DIR.mkdir(exist_ok=True)



def make_csv_path(input_file: Path) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    base = input_file.stem
    return Path(f"{base}_{date_str}.csv")




def crop_bottom_left(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(h * (1 - BL_FRAC)) :, : int(w * BL_FRAC)]


def crop_right_third(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[:, int(w * 2 / 3) :]


def find_numbers(text: str):
    pattern = rf"\b\d{{{MIN_LEN},{MAX_LEN}}}\b"
    return sorted(set(re.findall(pattern, text)))


def download_image(url: str, seq: int) -> Path | None:
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    except Exception as exc:
        print(f"[DL]  Failed {url}  ⇒  {exc}")
        return None

   
    name = unquote(Path(urlparse(url).path).name) or f"img_{seq:04d}.jpg"
    local_path = IMG_DIR / name
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path


def extract_accession(img_path: Path, reader) -> str | None:
   
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[OCR] Could not read {img_path}")
        return None

   
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

    reader = easyocr.Reader(["en"], gpu=True)

    csv_path = make_csv_path(url_file)
    is_new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["url", "image_name", "accession_number"])

        for idx, url in enumerate(urls, 1):
            print(f"[{idx}/{len(urls)}] {url}")
            img_path = download_image(url, idx)
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
    if len(sys.argv) != 2:
        print("Usage: python accession_scraper.py urls.txt")
        sys.exit(1)
    main(Path(sys.argv[1]))
