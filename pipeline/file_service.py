from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .data_models import FrameIdentifier, RawFrameData, ProcessedFrameData


class FileService:
    """File IO utilities: discovery, loading raw data, saving processed artifacts."""

    RGB_DIR = "rgb"
    DEPTH_DIR = "depth"
    ANNOT_DIR = "annotations"

    def _extract_frame_id_from_rgb(self, filename: str) -> str | None:
        """Extract frame_id from RGB filename like 'rgb_frame_<id>_png.rf.<hash>.jpg'."""
        # regex capturing content between 'rgb_frame_' and '_png.rf'
        m = re.search(r"rgb_frame_(.+?)_png\.rf\.", filename)
        return m.group(1) if m else None

    def discover_frames(self, raw_base_dir: str) -> List[FrameIdentifier]:
        raw_path = Path(raw_base_dir)
        rgb_dir = raw_path / self.RGB_DIR
        depth_dir = raw_path / self.DEPTH_DIR
        annot_dir = raw_path / self.ANNOT_DIR

        frames: List[FrameIdentifier] = []
        if not rgb_dir.exists():
            return frames

        for rgb_file in rgb_dir.glob("*.jpg"):
            frame_id = self._extract_frame_id_from_rgb(rgb_file.name)
            if not frame_id:
                continue

            depth_file = depth_dir / f"depth_data_{frame_id}.txt"
            # annotation file could have varying hash suffix; pick the first match
            candidates = list(annot_dir.glob(f"rgb_frame_{frame_id}_png.rf.*.txt"))
            annot_file = candidates[0] if candidates else None

            if not depth_file.exists() or annot_file is None:
                continue

            frames.append(
                FrameIdentifier(
                    base_name=frame_id,
                    raw_rgb_path=str(rgb_file),
                    raw_depth_path=str(depth_file),
                    raw_mask_path=str(annot_file),
                )
            )

        return frames

    def load_raw_data(self, frame_id: FrameIdentifier) -> RawFrameData:
        rgb = cv2.imread(frame_id.raw_rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"Cannot read RGB image: {frame_id.raw_rgb_path}")

        # Depth txt: rows of millimeter values
        try:
            depth_mm = np.loadtxt(frame_id.raw_depth_path, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(f"Failed to read depth txt: {frame_id.raw_depth_path}") from exc

        polygons: list[tuple[int, np.ndarray]] = []
        with open(frame_id.raw_mask_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                coords = np.array(list(map(float, parts[1:])), dtype=np.float32)
                if coords.size % 2 != 0:
                    # skip malformed
                    continue
                coords = coords.reshape(-1, 2)
                polygons.append((class_id, coords))

        return RawFrameData(
            identifier=frame_id,
            rgb_image=rgb,
            depth_map_mm=depth_mm,
            polygons=polygons,
        )

    def _ensure_dir(self, path: Path) -> None:
        os.makedirs(path, exist_ok=True)

    def save_raw_depth_png(self, frame_id: FrameIdentifier, depth_mm: np.ndarray, run_dir: Path) -> Path:
        self._ensure_dir(run_dir)
        out_dir = run_dir / "depth_raw_png"
        self._ensure_dir(out_dir)
        out_path = out_dir / f"{frame_id.base_name}_depth_raw.png"
        depth_uint16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(out_path), depth_uint16)
        return out_path

    def save_processed_data(self, data: ProcessedFrameData, run_dir: Path) -> None:
        # Save filled depth (m -> uint16 mm)
        depth_dir = run_dir / "depth_filled_png"
        self._ensure_dir(depth_dir)
        depth_mm_uint16 = np.clip(np.round(data.depth_map_filled_m * 1000.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_dir / f"{data.identifier.base_name}_depth_filled.png"), depth_mm_uint16)

        # Save HHA (assumed float32 in [0..some_scale]); scale to uint16 via 1000 as per spec
        hha_dir = run_dir / "hha_png"
        self._ensure_dir(hha_dir)
        hha_uint16 = np.clip(np.round(data.hha_image * 1000.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(hha_dir / f"{data.identifier.base_name}_hha.png"), hha_uint16)

        # Save mask (uint8)
        masks_dir = run_dir / "masks"
        self._ensure_dir(masks_dir)
        mask_u8 = data.segmentation_mask.astype(np.uint8)
        cv2.imwrite(str(masks_dir / f"{data.identifier.base_name}_mask.png"), mask_u8)

        # Save (possibly augmented) RGB image
        rgb_dir = run_dir / "rgb"
        self._ensure_dir(rgb_dir)
        rgb_bgr = data.rgb_image
        # Ensure 8-bit
        if rgb_bgr.dtype != np.uint8:
            rgb_bgr = np.clip(np.round(rgb_bgr), 0, 255).astype(np.uint8)
        cv2.imwrite(str(rgb_dir / f"{data.identifier.base_name}_rgb.png"), rgb_bgr)



