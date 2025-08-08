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

from pipeline.config_service import ConfigService
from pipeline.hha_service import HHAService


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute HHA from depth (m) using camera intrinsics from config")
    parser.add_argument("--input", required=True, help="Path to depth_filled.png (uint16 mm) or .npy (m)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", required=True, help="Path to output hha.png (uint8)")
    args = parser.parse_args()

    if args.input.endswith(".npy"):
        depth_m = np.load(args.input).astype(np.float32)
    else:
        depth_mm_u16 = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
        if depth_mm_u16 is None:
            raise FileNotFoundError(f"Cannot read depth image: {args.input}")
        depth_m = depth_mm_u16.astype(np.float32) / 1000.0

    cfg = ConfigService().load_config(args.config)
    K = cfg.cameras.depth_camera_matrix.to_numpy_array().astype(np.float32)

    hha = HHAService().convert(depth_m, K)

    # Normalize/convert HHA to 8-bit without saturating channels
    def to_uint8(img: np.ndarray) -> np.ndarray:
        x = img
        if x.dtype == np.uint8:
            return x
        x = x.astype(np.float32)
        maxv = float(np.nanmax(x)) if np.isfinite(x).any() else 0.0
        minv = float(np.nanmin(x)) if np.isfinite(x).any() else 0.0
        if maxv <= 1.0:  # likely 0..1
            x = x * 255.0
        elif maxv <= 255.0 and minv >= 0.0:  # already 0..255 range
            pass
        else:
            # min-max normalize each channel independently to 0..255
            x_out = np.empty_like(x, dtype=np.float32)
            for c in range(x.shape[2]):
                ch = x[..., c]
                ch_min = float(np.nanmin(ch))
                ch_max = float(np.nanmax(ch))
                if ch_max - ch_min < 1e-6:
                    x_out[..., c] = 0.0
                else:
                    x_out[..., c] = (ch - ch_min) * (255.0 / (ch_max - ch_min))
            x = x_out
        x = np.clip(np.round(x), 0, 255).astype(np.uint8)
        return x

    hha_u8 = to_uint8(hha)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), hha_u8)


if __name__ == "__main__":
    main()



