from __future__ import annotations

import argparse
from pathlib import Path

import sys

# Ensure project root is on sys.path for 'pipeline' imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from pipeline.inpainting_service import InpaintingService


def read_depth_txt_any(path: str) -> np.ndarray:
    """Read depth txt either as plain grid or header+sparse triples.

    Supported formats:
      - Plain whitespace-separated grid of millimeters (H x W)
      - Header with lines like 'Width: <w>', 'Height: <h>' and sparse lines
        'row,column,depth_value' following the header.
    """
    try:
        # Fast path: plain grid
        return np.loadtxt(path, dtype=np.float32)
    except Exception:
        # Fallback: parse header + triples
        height = width = None
        triples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("Width:"):
                    width = int(s.split(":", 1)[1].strip())
                    continue
                if s.startswith("Height:"):
                    height = int(s.split(":", 1)[1].strip())
                    continue
                if "," in s:
                    parts = s.split(",")
                    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
                        try:
                            r = int(parts[0])
                            c = int(parts[1])
                            v = float(parts[2])
                        except ValueError:
                            continue
                        triples.append((r, c, v))
        if height is None or width is None:
            raise ValueError("Cannot determine width/height from header and plain grid parsing failed")
        depth = np.zeros((height, width), dtype=np.float32)
        for r, c, v in triples:
            if 0 <= r < height and 0 <= c < width:
                depth[r, c] = v
        return depth


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inpainting on a depth txt file (mm)")
    parser.add_argument("--input", required=True, help="Path to depth .txt (mm)")
    parser.add_argument("--output", required=True, help="Path to output depth_filled.png (uint16 mm)")
    parser.add_argument("--method", default="linear_nearest", help="Inpainting method")
    args = parser.parse_args()

    depth_mm = read_depth_txt_any(args.input)
    filled_m = InpaintingService().apply(depth_mm, args.method)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    depth_mm_uint16 = np.clip(np.round(filled_m * 1000.0), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(out_path), depth_mm_uint16)


if __name__ == "__main__":
    main()



