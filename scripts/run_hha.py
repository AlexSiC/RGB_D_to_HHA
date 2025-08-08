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
    parser.add_argument("--output", required=True, help="Path to output hha.png (uint16 scaled)")
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
    hha_u16 = np.clip(np.round(hha * 1000.0), 0, 65535).astype(np.uint16)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), hha_u16)


if __name__ == "__main__":
    main()



