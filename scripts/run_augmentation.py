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
from pipeline.augmentation_service import AugmentationService


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply synchronous augmentations to RGB/Depth/Mask")
    parser.add_argument("--rgb", required=True, help="Path to RGB image")
    parser.add_argument("--depth", required=True, help="Path to depth (uint16 mm) or .npy (m)")
    parser.add_argument("--mask", required=True, help="Path to mask (uint8)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", required=True, help="Dir to save augmented outputs")
    args = parser.parse_args()

    rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Cannot read RGB image: {args.rgb}")

    if args.depth.endswith(".npy"):
        depth_m = np.load(args.depth).astype(np.float32)
    else:
        depth_mm = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise FileNotFoundError(f"Cannot read depth image: {args.depth}")
        depth_m = depth_mm.astype(np.float32) / 1000.0

    mask = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask image: {args.mask}")

    cfg = ConfigService().load_config(args.config)
    result = AugmentationService().apply(rgb, depth_m, mask, cfg.augmentation)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "rgb_aug.png"), result["rgb"]) 
    depth_u16 = np.clip(np.round(result["depth"] * 1000.0), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(out_dir / "depth_aug.png"), depth_u16)
    cv2.imwrite(str(out_dir / "mask_aug.png"), result["mask"].astype(np.uint8))


if __name__ == "__main__":
    main()



