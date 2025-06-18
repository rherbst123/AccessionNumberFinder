#!/usr/bin/env python3
from pathlib import Path
from urllib.parse import urlparse, unquote
import csv, re, sys, requests, cv2, numpy as np, easyocr, shutil, time
from datetime import datetime

#Config
MIN_LEN, MAX_LEN = 6, 7
TARGET_H = 2250
ROI_X_MAX, ROI_Y_MIN, ROI_Y_MAX = 380, 1900, 2250
MAX_TRIES = 5
THRESHOLDS = [0.80, 0.70, 0.65, 0.55, 0.45]
SAT_STEP = 0.10
DIGIT_RE = re.compile(rf"\b[1-9]\d{{{MIN_LEN-1},{MAX_LEN-1}}}\b")


def make_csv_path(p: Path) -> Path:
    d = Path.home() / "Desktop" / "AccessionNumbers"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{p.stem}_{datetime.now():%Y-%m-%d}.csv"

def resize_h(img: np.ndarray, h_out: int = TARGET_H) -> np.ndarray:
    h, w = img.shape[:2]
    s = h_out / h
    return cv2.resize(img, (int(w * s), h_out), interpolation=cv2.INTER_AREA)

def crop_roi(img):
    """Bottom-quarter strip, left 380 px."""
    h, w = img.shape[:2]
    y_start = int(h * 0.75)
    return img[y_start:h, 0:min(ROI_X_MAX + 1, w)]

def boost_sat(img: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def find_nums(s: str): 
    return DIGIT_RE.findall(s)

def seq_name(n: int, url: str) -> str:
    base = unquote(Path(urlparse(url).path).name) or "image"
    return f"{base}"

def dl(url: str, n: int, roi_dir: Path, full_dir: Path):
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"[DL]  Failed {url} → {e}")
        return None
    name = seq_name(n, url)
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[DL]  Decode error {url}")
        return None
    full = resize_h(img)
    cv2.imwrite(str(full_dir / name), full)
    roi = crop_roi(full)
    p = roi_dir / name
    cv2.imwrite(str(p), roi)
    return p

def ocr(img, rdr, thr):
    for *_b, txt, conf in rdr.readtext(img, detail=1, paragraph=False):
        if conf >= thr:
            f = find_nums(txt)
            if f:
                return f[0], conf
    return None, None

def ext(p: Path, rdr):
    roi0 = cv2.imread(str(p))
    if roi0 is None:
        print(f"[OCR] cannot read {p}")
        return None, None
    for i in range(MAX_TRIES):
        thr = THRESHOLDS[i] if i < len(THRESHOLDS) else THRESHOLDS[-1]
        roi = boost_sat(roi0, 1 + SAT_STEP * i)
        acc, conf = ocr(roi, rdr, thr)
        if acc:
            print(f"      ↳ hit at attempt {i+1} (≥{thr:.0%})")
            return acc, conf
        print(f"      ↳ miss at attempt {i+1} (≥{thr:.0%}), retrying…")
    return None, None

def main(u_file: Path, out_dir: Path):
    start_time = time.perf_counter()          # ⇦ TIMER START
    urls = [u.strip() for u in u_file.read_text().splitlines() if u.strip()]
    if not urls:
        sys.exit("empty URL file")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    roi_dir, full_dir = out_dir, out_dir / "full_images"
    roi_dir.mkdir(parents=True)
    full_dir.mkdir(parents=True)

    csv_p = make_csv_path(u_file)
    first = not csv_p.exists()
    rdr = easyocr.Reader(["en"], gpu=True)

    with csv_p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if first:
            w.writerow(["url", "image_name", "accession_number",
                        "confidence", "timestamp"])
        for i, url in enumerate(urls, 1):
            img_start = time.perf_counter()   # per-image timer
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{i}/{len(urls)}] {url}")
            p = dl(url, i, roi_dir, full_dir)
            if p:
                acc, conf = ext(p, rdr)
                cs = f"{conf:.2f}" if conf else ""
                row = [url, p.name, acc or "<not-found>", cs, ts]
            else:
                row = [url, f"{i:04d}_<download-error>", "", "", ts]
            w.writerow(row)
            f.flush()
            print(f"      ↳ {row[2] or 'no number found'}")
            # elapsed for this image
            print(f"      elapsed: {time.perf_counter() - img_start:.2f}s\n")

    total = time.perf_counter() - start_time  # ⇦ TIMER END
    print(f"Total runtime: {total:.2f} s "
          f"({total / 60:.1f} min)\nDone → {csv_p.resolve()}")

if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        print("Usage: accession_scraper.py urls.txt [output_dir]")
        sys.exit(1)
    main(Path(sys.argv[1]),Path(sys.argv[2]) if len(sys.argv) == 3 else Path("downloaded_images"))
