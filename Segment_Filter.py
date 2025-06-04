
from pathlib import Path
import re
import sys

import cv2
import easyocr
import numpy as np


ROOT_DIR = Path("/home/riley/Desktop/Kimberly_Images_Final_Test_1")   # top folder with masks

DRY_RUN = False   # True → just print; False → actually delete files

DIGIT_RE = re.compile(r"^\d{5,7}$")   # exactly 5-7 digits, nothing else
reader   = easyocr.Reader(["en"], gpu=True)  # gpu=False if CUDA memory is tight

def text_from_easyocr(img_path: Path) -> str:
    """Run EasyOCR → concatenated string with spaces stripped."""
    return "".join(reader.readtext(str(img_path), detail=0)).replace(" ", "").strip()

def main() -> None:
    kept = deleted = 0

    for img_path in ROOT_DIR.rglob("*.png"):
        # (Optional) light binarisation helps EasyOCR on beige tags
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        _, img_bin = cv2.threshold(img_gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # EasyOCR works fine from disk path; we can also feed the binarised image
        text = "".join(reader.readtext(img_bin, detail=0)).replace(" ", "").strip()

        if DIGIT_RE.fullmatch(text):
            kept += 1
            continue  # good mask → keep

        # otherwise delete (or report)
        deleted += 1
        rel = img_path.relative_to(ROOT_DIR)
        if DRY_RUN:
            print(f"[dry-run] would delete: {rel}")
        else:
            img_path.unlink(missing_ok=True)
            print(f"🗑 deleted  : {rel}")

    print(f"\nSummary   kept={kept}   deleted={deleted}"
          f"{' (no files actually removed)' if DRY_RUN else ''}")

if __name__ == "__main__":
    if not ROOT_DIR.is_dir():
        sys.exit(f"ROOT_DIR does not exist: {ROOT_DIR}")
    main()
