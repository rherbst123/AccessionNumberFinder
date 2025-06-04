import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np
import gc
from tqdm import tqdm
import psutil
import GPUtil
import threading
import time
import os
from torchvision.ops import boxes

"""
Segmentator with ROI support **and** mask dilation
-------------------------------------------------
* ROI prompt (bottom-left by default) – keep default to process full page.
* Mask dilation merges adjacent letters → one mask per ID string.
* Live GPU + CPU usage display in tqdm bar.
* Syntax-safe, complete `run_pipeline()` including cleanup.

Run:
    python Segmentation_Testing.py
"""

# -----------------------------------------------------------------------------
# torch 2.3 batched_nms CPU patch (SAM compatibility)
# -----------------------------------------------------------------------------
_original_batched_nms = boxes.batched_nms


def _patched_batched_nms(boxes_tensor, scores, idxs, iou_threshold):
    return _original_batched_nms(boxes_tensor.cpu(), scores.cpu(), idxs.cpu(), iou_threshold)


boxes.batched_nms = _patched_batched_nms

# -----------------------------------------------------------------------------
# Live resource usage helper
# -----------------------------------------------------------------------------

def _get_resource_usage():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    if gpus:
        g = gpus[0]
        gpu_load = g.load * 100
        gpu_mem = (g.memoryUsed / g.memoryTotal) * 100
    else:
        gpu_load = gpu_mem = 0
    return f"CPU:{cpu:.1f}% MEM:{mem:.1f}% GPU:{gpu_load:.1f}% GPU-Mem:{gpu_mem:.1f}%"


def _resource_monitor(pbar, stop_evt, lock):
    while not stop_evt.is_set():
        with lock:
            pbar.set_postfix_str(_get_resource_usage())
        time.sleep(1)

# -----------------------------------------------------------------------------
# SAM initialisation
# -----------------------------------------------------------------------------

def _init_sam():
    ckpt = "/home/riley/Documents/GitHub/Segmentator/Segmentator/models/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=ckpt).to("cpu")  # switch to "cuda" if VRAM allows
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=10,
        pred_iou_thresh=0.80,  # slightly lower → bigger masks
        stability_score_thresh=0.90,
        crop_n_layers=1,
        crop_n_points_downscale_factor=0.7,
        min_mask_region_area=8000,
    )

# -----------------------------------------------------------------------------
# ROI utilities
# -----------------------------------------------------------------------------

def _roi_coords(shape, location="bottom_left", w_ratio=0.35, h_ratio=0.35):
    h, w = shape[:2]
    w_ratio, h_ratio = np.clip(w_ratio, 0, 1), np.clip(h_ratio, 0, 1)
    if location == "bottom_left":
        return 0, int(h * (1 - h_ratio)), int(w * w_ratio), h
    # default → full image
    return 0, 0, w, h

# -----------------------------------------------------------------------------
# Mask dilation merges neighbouring glyphs
# -----------------------------------------------------------------------------

def _dilate_mask(mask, k_size=5, iterations=4):
    kernel = np.ones((k_size, k_size), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=iterations)
    return dilated > 0

# -----------------------------------------------------------------------------
# Segmentation (optional ROI)
# -----------------------------------------------------------------------------

def segment(image_path, gen, use_roi=False, w_ratio=0.35, h_ratio=0.35):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")

    # Downscale very large images
    max_dim = 2250
    scale = max_dim / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    if use_roi:
        x0, y0, x1, y1 = _roi_coords(img.shape, "bottom_left", w_ratio, h_ratio)
        roi = img[y0:y1, x0:x1]
        masks = gen.generate(roi)
        full_h, full_w = img.shape[:2]
        for m in masks:
            m["bbox"][0] += x0
            m["bbox"][1] += y0
            full_mask = np.zeros((full_h, full_w), dtype=bool)
            full_mask[y0:y1, x0:x1] = m["segmentation"]
            m["segmentation"] = full_mask
            m["area"] = int(full_mask.sum())
    else:
        masks = gen.generate(img)

    # Filter very large masks (full-page grabs)
    page_area = img.shape[0] * img.shape[1]
    masks = [m for m in masks if m["area"] < page_area * 0.9]
    return masks, img

# -----------------------------------------------------------------------------
# Duplicate suppression
# -----------------------------------------------------------------------------

def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-6)


def deduplicate(masks, thr=0.60):
    unique = []
    for m in masks:
        if all(_iou(m["segmentation"], u["segmentation"]) <= thr for u in unique):
            unique.append(m)
    return unique

# -----------------------------------------------------------------------------
# Save helpers
# -----------------------------------------------------------------------------

def _save_visual(img, masks, out_dir):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for m in masks:
        plt.contour(m["segmentation"], colors="red")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{os.path.basename(out_dir)}_viz.png"), bbox_inches="tight")
    plt.close()


def _crop_and_save(img, masks, out_dir, dilate_iter=4):
    for idx, m in enumerate(masks):
        dilated = _dilate_mask(m["segmentation"], iterations=dilate_iter)
        ys, xs = np.where(dilated)
        if xs.size == 0:
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        crop = img[y0:y1 + 1, x0:x1 + 1].copy()
        mask_crop = dilated[y0:y1 + 1, x0:x1 + 1]
        crop[~mask_crop] = 0
        cv2.imwrite(os.path.join(out_dir, f"mask_{idx + 1}.png"), crop)
        print(f"Segment {idx + 1}: (x:{x0}, y:{y0}, w:{x1 - x0}, h:{y1 - y0})")

# -----------------------------------------------------------------------------
# Main pipeline (complete)
# -----------------------------------------------------------------------------

def run_pipeline(inp_dir, out_dir, use_roi=False, w_ratio=0.35, h_ratio=0.35, dilate_iter=4):
    """Process all images in *inp_dir* and store results under *out_dir*."""
    imgs = [f for f in os.listdir(inp_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not imgs:
        print("No images found in", inp_dir)
        return

    os.makedirs(out_dir, exist_ok=True)

    lock = threading.Lock()
    with tqdm(total=len(imgs), desc="Processing", unit="img") as pbar:
        stop_evt = threading.Event()
        mon = threading.Thread(target=_resource_monitor, args=(pbar, stop_evt, lock), daemon=True)
        mon.start()
        try:
            with lock:
                pbar.set_description("Init SAM")
            gen = _init_sam()

            for name in imgs:
                path = os.path.join(inp_dir, name)
                try:
                    masks, img = segment(path, gen, use_roi, w_ratio, h_ratio)
                    masks = deduplicate(masks)
                    out_sub = os.path.join(out_dir, os.path.splitext(name)[0])
                    os.makedirs(out_sub, exist_ok=True)
                    _save_visual(img, masks, out_sub)
                    _crop_and_save(img, masks, out_sub, dilate_iter)
                except Exception as exc:
                    print(f"Error on {name}: {exc}")
                finally:
                    # per-image cleanup + progress tick
                    torch.cuda.empty_cache()
                    gc.collect()
                    with lock:
                        pbar.update(1)
        finally:
            # Ensure monitor thread stops and memory freed
            stop_evt.set()
            mon.join()
            del gen
            torch.cuda.empty_cache()
            gc.collect()

# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    INP = "/home/riley/Desktop/Kimberly_Images"          
    OUT = "/home/riley/Desktop/Kimberly_Images_Final"    

    print("GPU:" if torch.cuda.is_available() else "CPU-only run — slower")

    roi_answer = input("Restrict to bottom-left ROI? (Y/N): ").strip().lower() == "y"
    if roi_answer:
        try:
            w_r = float(input("Width ratio [0.35]: ") or 0.35)
            h_r = float(input("Height ratio [0.35]: ") or 0.35)
        except ValueError:
            w_r = h_r = 0.35
    else:
        w_r = h_r = 0.35

    try:
        dil_iter = int(input("Mask dilation iterations [4]: ") or 4)
    except ValueError:
        dil_iter = 4

    run_pipeline(INP, OUT, use_roi=roi_answer, w_ratio=w_r, h_ratio=h_r)