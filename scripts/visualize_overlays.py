from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def _parse_hex_color(hex_color: str) -> Tuple[int, int, int]:
    """Parse color string like '#RRGGBB' or 'RRGGBB' to BGR tuple for OpenCV."""
    s = hex_color.strip().lstrip('#')
    if len(s) != 6:
        raise ValueError(f"Invalid color '{hex_color}', expected #RRGGBB")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def _load_hha_as_u8(path: Path) -> np.ndarray:
    """Load HHA image preserving bit depth and convert to uint8 BGR for overlay.

    - If input is 3-channel uint8, return as-is.
    - If input is uint16 (e.g., saved by pipeline), downscale to uint8 via >> 8.
    - If input is single-channel, replicate to 3 channels.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if img.ndim == 2:
        # Single channel -> 3-channel gray
        if img.dtype == np.uint16:
            img8 = (img >> 8).astype(np.uint8)
        else:
            img8 = img.astype(np.uint8)
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    # 3-channel
    if img.dtype == np.uint16:
        return (img >> 8).astype(np.uint8)
    return img.astype(np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    """Load mask as uint8 single-channel array. Non-zero pixels will be overlaid."""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.dtype != np.uint8:
        m = np.clip(m, 0, 255).astype(np.uint8)
    return m


def _build_overlay(color_bgr: Tuple[int, int, int], mask: np.ndarray) -> np.ndarray:
    """Create a color image with given BGR color only where mask > 0."""
    h, w = mask.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    if np.any(mask):
        overlay[mask > 0] = color_bgr
    return overlay


def _parse_class_colors(spec: str | None) -> Dict[int, Tuple[int, int, int]]:
    """Parse mapping like '1=#00FF00,2=#FF00FF' into {1:(b,g,r), ...}."""
    mapping: Dict[int, Tuple[int, int, int]] = {}
    if not spec:
        return mapping
    parts = [p.strip() for p in spec.split(',') if p.strip()]
    for part in parts:
        if '=' not in part:
            raise ValueError(f"Invalid class-colors entry '{part}', expected id=#RRGGBB")
        k, v = part.split('=', 1)
        cls_id = int(k)
        mapping[cls_id] = _parse_hex_color(v)
    return mapping


def overlay_directory(images_dir: Path, masks_dir: Path, out_dir: Path, alpha: float, color_hex: str, class_colors: Dict[int, Tuple[int, int, int]], draw_contours: bool) -> int:
    """Overlay every mask in masks_dir onto corresponding image in images_dir.

    Returns number of visualizations written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    color_default = _parse_hex_color(color_hex)
    num_written = 0

    image_files = sorted([p for p in images_dir.glob('*.png')])
    for img_path in image_files:
        stem = img_path.name
        mask_path = masks_dir / stem
        if not mask_path.exists():
            continue

        base = _load_hha_as_u8(img_path)
        mask = _load_mask(mask_path)

        # Per-class colors if provided
        if class_colors:
            overlay = np.zeros_like(base, dtype=np.uint8)
            ids = np.unique(mask)
            for cls_id in ids:
                if cls_id == 0:
                    continue
                color = class_colors.get(int(cls_id), color_default)
                overlay[mask == cls_id] = color
        else:
            overlay = _build_overlay(color_default, mask)

        blended = cv2.addWeighted(base, 1.0, overlay, float(alpha), 0.0)

        if draw_contours and np.any(mask):
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, contours, -1, (0, 255, 255), thickness=1)

        out_path = out_dir / stem
        cv2.imwrite(str(out_path), blended)
        num_written += 1

    return num_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch overlay of segmentation masks onto HHA images.")
    parser.add_argument('--images-dir', required=True, help="Directory with HHA images (PNG)")
    parser.add_argument('--masks-dir', required=True, help="Directory with mask images (PNG, uint8)")
    parser.add_argument('--out-dir', required=True, help="Directory to write overlay visualizations")
    parser.add_argument('--alpha', type=float, default=0.45, help="Overlay opacity [0..1] (default: 0.45)")
    parser.add_argument('--color', type=str, default="#FF00FF", help="Default overlay color for non-zero mask, hex like #RRGGBB")
    parser.add_argument('--class-colors', type=str, default=None, help="Optional per-class mapping '1=#00FF00,2=#FF00FF'")
    parser.add_argument('--contours', action='store_true', help="Draw mask contours on top of overlay")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.out_dir)

    if not images_dir.exists() or not masks_dir.exists():
        raise SystemExit("images-dir or masks-dir does not exist")

    class_colors = _parse_class_colors(args.class_colors)
    written = overlay_directory(images_dir, masks_dir, out_dir, alpha=float(args.alpha), color_hex=str(args.color), class_colors=class_colors, draw_contours=bool(args.contours))
    print(f"Wrote {written} overlays to {out_dir}")


if __name__ == '__main__':
    main()


