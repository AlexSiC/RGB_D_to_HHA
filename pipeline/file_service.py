from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Any

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

            # Depth file may be named either 'depth_data_<id>.txt' or 'depth_data_frame_<id>.txt'
            depth_candidates = [
                depth_dir / f"depth_data_{frame_id}.txt",
                depth_dir / f"depth_data_frame_{frame_id}.txt",
            ]
            depth_file = next((p for p in depth_candidates if p.exists()), None)
            # annotation file could have varying hash suffix; pick the first match
            candidates = list(annot_dir.glob(f"rgb_frame_{frame_id}_png.rf.*.txt"))
            annot_file = candidates[0] if candidates else None

            if depth_file is None or annot_file is None:
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

        # Depth txt: header with Width/Height, then triples: row,column,depth_value (in mm)
        try:
            height = None
            width = None
            # Pre-scan first ~10 lines for metadata
            with open(frame_id.raw_depth_path, "r", encoding="utf-8") as f:
                header_lines = [next(f) for _ in range(10)]
            for line in header_lines:
                line_stripped = line.strip()
                if line_stripped.lower().startswith("width:"):
                    try:
                        width = int(line_stripped.split(":", 1)[1])
                    except Exception:
                        pass
                elif line_stripped.lower().startswith("height:"):
                    try:
                        height = int(line_stripped.split(":", 1)[1])
                    except Exception:
                        pass
        except StopIteration:
            # File shorter than expected; will fall back to robust path below
            pass
        except Exception as exc:
            raise RuntimeError(f"Failed to pre-read depth header: {frame_id.raw_depth_path}") from exc

        try:
            if height is None or width is None:
                # Robust parse: scan full file to detect max row/col
                max_r = 0
                max_c = 0
                with open(frame_id.raw_depth_path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s or s[0].isalpha():
                            continue
                        parts = s.split(",")
                        if len(parts) != 3:
                            continue
                        r = int(parts[0])
                        c = int(parts[1])
                        max_r = max(max_r, r)
                        max_c = max(max_c, c)
                height = max_r + 1
                width = max_c + 1

            depth_mm = np.zeros((height, width), dtype=np.float32)
            with open(frame_id.raw_depth_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s[0].isalpha():
                        # Skip headers like 'Width:', 'Height:', 'Format:' etc.
                        continue
                    parts = s.split(",")
                    if len(parts) != 3:
                        continue
                    try:
                        r = int(parts[0])
                        c = int(parts[1])
                        v = float(parts[2])
                    except Exception:
                        continue
                    if 0 <= r < height and 0 <= c < width:
                        depth_mm[r, c] = v
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

    def save_processed_data(
        self,
        data: ProcessedFrameData,
        run_dir: Path,
        save_hha_channels_jet: bool = False,
        outputs: Optional[Any] = None,
    ) -> None:
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

        # Also save visualization-friendly version normalized to uint8 per channel
        hha = data.hha_image.astype(np.float32)
        vis = np.zeros_like(hha, dtype=np.uint8)
        for ch in range(hha.shape[2]):
            chan = hha[:, :, ch]
            finite = np.isfinite(chan)
            if np.any(finite):
                vmin = float(chan[finite].min())
                vmax = float(chan[finite].max())
            else:
                vmin, vmax = 0.0, 1.0
            if vmax - vmin < 1e-6:
                scaled = np.zeros_like(chan, dtype=np.uint8)
            else:
                scaled = np.clip((chan - vmin) * 255.0 / (vmax - vmin), 0, 255).astype(np.uint8)
            vis[:, :, ch] = scaled
        cv2.imwrite(str(hha_dir / f"{data.identifier.base_name}_hha_vis_u8.png"), vis)

        # Optional: save per-channel JET visualizations if requested by caller
        if save_hha_channels_jet:
            # Angle (A), Height (H), Disparity (D) channels assumed order [A,H,D]
            names = ['angle', 'height', 'disparity']
            for ch, name in enumerate(names):
                chan = hha[:, :, ch]
                finite = np.isfinite(chan)
                if np.any(finite):
                    vmin = float(chan[finite].min())
                    vmax = float(chan[finite].max())
                else:
                    vmin, vmax = 0.0, 1.0
                if vmax - vmin < 1e-6:
                    norm = np.zeros_like(chan, dtype=np.uint8)
                else:
                    norm = np.clip((chan - vmin) * 255.0 / (vmax - vmin), 0, 255).astype(np.uint8)
                jet = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                cv2.imwrite(str(hha_dir / f"{data.identifier.base_name}_hha_{name}_jet.png"), jet)

        # Optional: export to train/images & train/masks if enabled in outputs config provided by caller
        try:
            if outputs and getattr(outputs, 'enable_train_export', False):
                train_root = Path(getattr(outputs, 'train_dir', 'data/train'))
                images_dir = train_root / 'images'
                masks_dir = train_root / 'masks'
                self._ensure_dir(images_dir)
                self._ensure_dir(masks_dir)

                # Build deterministic stem: frameId + variant suffix if present in run_dir (seed_xx)
                variant_suffix = ''
                parts = list(run_dir.parts)
                if parts and parts[-1].startswith('seed_'):
                    variant_suffix = f"__{parts[-1]}"

                # Hash from vis content to identify transform parameters implicitly
                import hashlib
                hasher = hashlib.sha1()
                hasher.update(vis.tobytes())
                ops_hash = hasher.hexdigest()[:8]

                stem = f"{data.identifier.base_name}{variant_suffix}__h{ops_hash}"

                # Choose HHA format for training
                if getattr(outputs, 'save_hha_u8_in_train', False):
                    hha_img = vis
                else:
                    hha_img = hha_uint16

                cv2.imwrite(str(images_dir / f"{stem}.png"), hha_img)
                cv2.imwrite(str(masks_dir / f"{stem}.png"), mask_u8)

                # Append manifest row
                import csv
                manifest = train_root / 'index.csv'
                header = ['image_path', 'mask_path', 'frame_id', 'variant', 'ops_hash']
                write_header = not manifest.exists()
                with manifest.open('a', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(header)
                    variant = parts[-1] if variant_suffix else 'baseline'
                    w.writerow([
                        str(images_dir / f"{stem}.png"),
                        str(masks_dir / f"{stem}.png"),
                        data.identifier.base_name,
                        variant,
                        ops_hash,
                    ])
        except Exception:
            # Do not fail pipeline due to export errors
            pass

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



