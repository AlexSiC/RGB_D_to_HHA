from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import sys

# Ensure project root is on sys.path for 'pipeline' imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from pipeline.annotation_service import AnnotationService


def _read_polygons_from_txt(path: str) -> List[Tuple[int, np.ndarray]]:
    polygons: list[tuple[int, np.ndarray]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            coords = np.array(list(map(float, parts[1:])), dtype=np.float32)
            if coords.size % 2 != 0:
                continue
            coords = coords.reshape(-1, 2)
            polygons.append((class_id, coords))
    return polygons


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLO-like polygons to mask")
    parser.add_argument("--annotation", required=True, help="Path to annotation .txt file")
    parser.add_argument("--rgb", required=True, help="Path to corresponding RGB image to infer size")
    parser.add_argument("--output", required=True, help="Path to output mask .png")
    args = parser.parse_args()

    rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Cannot read RGB image: {args.rgb}")
    h, w = rgb.shape[:2]

    polygons = _read_polygons_from_txt(args.annotation)
    mask = AnnotationService().convert_polygons_to_mask(polygons, (h, w))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask.astype(np.uint8))


if __name__ == "__main__":
    main()



