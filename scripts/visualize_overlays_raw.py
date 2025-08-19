from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from pipeline.annotation_service import AnnotationService


def _parse_hex_color(hex_color: str) -> Tuple[int, int, int]:
    s = hex_color.strip().lstrip('#')
    if len(s) != 6:
        raise ValueError(f"Invalid color '{hex_color}', expected #RRGGBB")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)  # BGR for OpenCV


def _extract_frame_id_from_rgb(filename: str) -> str | None:
    # Same pattern as in FileService
    m = re.search(r"rgb_frame_(.+?)_png\.rf\.", filename)
    return m.group(1) if m else None


def _iter_frames(raw_dir: Path) -> Iterable[Tuple[Path, Path, str]]:
    rgb_dir = raw_dir / 'rgb'
    annot_dir = raw_dir / 'annotations'
    if not rgb_dir.exists() or not annot_dir.exists():
        return []
    for rgb_file in sorted(rgb_dir.glob('*.jpg')):
        frame_id = _extract_frame_id_from_rgb(rgb_file.name)
        if not frame_id:
            continue
        candidates = list(annot_dir.glob(f"rgb_frame_{frame_id}_png.rf.*.txt"))
        if not candidates:
            continue
        yield rgb_file, candidates[0], frame_id


def _read_polygons_txt(path: Path) -> List[Tuple[int, np.ndarray]]:
    polygons: List[Tuple[int, np.ndarray]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except Exception:
                continue
            coords = np.array(list(map(float, parts[1:])), dtype=np.float32)
            if coords.size % 2 != 0 or coords.size == 0:
                continue
            coords = coords.reshape(-1, 2)
            polygons.append((class_id, coords))
    return polygons


def _overlay_mask(base_bgr: np.ndarray, mask: np.ndarray, *, alpha: float, color_default: Tuple[int, int, int], class_colors: Dict[int, Tuple[int, int, int]], draw_contours: bool) -> np.ndarray:
    if mask.ndim != 2:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError('mask must be 2D')

    overlay = np.zeros_like(base_bgr, dtype=np.uint8)
    if class_colors:
        ids = np.unique(mask)
        for cls_id in ids:
            if cls_id == 0:
                continue
            color = class_colors.get(int(cls_id), color_default)
            overlay[mask == cls_id] = color
    else:
        overlay[mask > 0] = color_default

    blended = cv2.addWeighted(base_bgr, 1.0, overlay, float(alpha), 0.0)
    if draw_contours and np.any(mask):
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 255, 255), thickness=1)
    return blended


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Overlay RAW annotations (polygons) onto RAW RGB images.')
    parser.add_argument('--raw-dir', required=True, help='Root RAW directory containing rgb/ and annotations/')
    parser.add_argument('--out-dir', required=True, help='Directory to write visualizations')
    parser.add_argument('--alpha', type=float, default=0.45, help='Overlay opacity [0..1]')
    parser.add_argument('--color', type=str, default='#FF00FF', help='Default overlay color for non-zero mask, hex like #RRGGBB')
    parser.add_argument('--class-colors', type=str, default=None, help="Optional per-class mapping '1=#00FF00,2=#FF00FF'")
    parser.add_argument('--contours', action='store_true', help='Draw mask contours')
    return parser.parse_args()


def _parse_class_colors(spec: str | None) -> Dict[int, Tuple[int, int, int]]:
    mapping: Dict[int, Tuple[int, int, int]] = {}
    if not spec:
        return mapping
    for part in [p.strip() for p in spec.split(',') if p.strip()]:
        if '=' not in part:
            raise ValueError(f"Invalid class-colors entry '{part}', expected id=#RRGGBB")
        k, v = part.split('=', 1)
        mapping[int(k)] = _parse_hex_color(v)
    return mapping


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann = AnnotationService()
    color_default = _parse_hex_color(args.color)
    class_colors = _parse_class_colors(args.class_colors)

    written = 0
    for rgb_path, annot_path, frame_id in _iter_frames(raw_dir):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        polygons = _read_polygons_txt(annot_path)
        mask = ann.convert_polygons_to_mask(polygons, rgb.shape[:2])
        vis = _overlay_mask(rgb, mask, alpha=float(args.alpha), color_default=color_default, class_colors=class_colors, draw_contours=bool(args.contours))
        out_path = out_dir / f"{frame_id}.png"
        cv2.imwrite(str(out_path), vis)
        written += 1

    print(f"Wrote {written} overlays to {out_dir}")


if __name__ == '__main__':
    main()


