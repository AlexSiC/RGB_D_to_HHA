from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from .data_models import FrameIdentifier, RawFrameData, ProcessedFrameData, PipelineConfig
from .file_service import FileService
from .inpainting_service import InpaintingService
from .annotation_service import AnnotationService
from .augmentation_service import AugmentationService
from .hha_service import HHAService


class PipelineOrchestrator:
    """Coordinates end-to-end processing of frames according to PipelineConfig."""

    def __init__(
        self,
        config: PipelineConfig,
        file_service: FileService,
        inpainting_service: InpaintingService,
        annotation_service: AnnotationService,
        augmentation_service: AugmentationService,
        hha_service: HHAService,
    ) -> None:
        self.config = config
        self.file_service = file_service
        self.inpainting_service = inpainting_service
        self.annotation_service = annotation_service
        self.augmentation_service = augmentation_service
        self.hha_service = hha_service

        self._setup_logging()
        self.run_dir = self._create_run_dir()

    def _setup_logging(self) -> None:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(logs_dir / "pipeline.log", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

    def _create_run_dir(self) -> Path:
        processed_base = Path(self.config.paths.processed_dir)
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = processed_base / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def run_full_pipeline(self) -> None:
        frames = self.file_service.discover_frames(self.config.paths.raw_dir)
        logging.info("Discovered %d frames", len(frames))
        failed_list: list[str] = []

        for frame_id in tqdm(frames, desc="Processing frames"):
            try:
                self.process_single_frame(frame_id)
            except Exception as exc:  # noqa: BLE001
                logging.exception("Failed processing %s: %s", frame_id.base_name, exc)
                failed_list.append(frame_id.base_name)

        if failed_list:
            failed_file = Path("logs") / "failed_files.txt"
            with failed_file.open("w", encoding="utf-8") as f:
                for name in failed_list:
                    f.write(name + "\n")
            logging.warning("Completed with %d failures. See %s", len(failed_list), failed_file)
        else:
            logging.info("Completed successfully. All frames processed.")

    def _validate_dimensions(self, raw: RawFrameData) -> None:
        rgb_h, rgb_w = raw.rgb_image.shape[:2]
        depth_h, depth_w = raw.depth_map_mm.shape[:2]
        if (rgb_h, rgb_w) != (depth_h, depth_w):
            raise RuntimeError(
                f"Dimension mismatch RGB({rgb_w}x{rgb_h}) vs Depth({depth_w}x{depth_h}) for {raw.identifier.base_name}"
            )

    def process_single_frame(self, frame_id: FrameIdentifier) -> None:
        raw: RawFrameData = self.file_service.load_raw_data(frame_id)
        self._validate_dimensions(raw)

        # Save raw depth before inpainting
        self.file_service.save_raw_depth_png(frame_id, raw.depth_map_mm, self.run_dir)

        # Inpainting (mm -> m inside service)
        depth_filled_m = self.inpainting_service.apply(raw.depth_map_mm, self.config.inpainting.method)

        # Annotation conversion (normalized polygons -> mask)
        mask = self.annotation_service.convert_polygons_to_mask(raw.polygons, raw.rgb_image.shape[:2])

        # Augment synchronously (if enabled)
        aug = self.augmentation_service.apply(raw.rgb_image, depth_filled_m, mask, self.config.augmentation)
        rgb_aug = aug["rgb"]
        depth_aug = aug["depth"]
        mask_aug = aug["mask"]

        # HHA conversion using depth camera intrinsics
        K = self.config.cameras.depth_camera_matrix.to_numpy_array()
        hha = self.hha_service.convert(depth_aug.astype(np.float32), K.astype(np.float32))

        processed = ProcessedFrameData(
            identifier=frame_id,
            rgb_image=rgb_aug,
            depth_map_filled_m=depth_aug,
            hha_image=hha,
            segmentation_mask=mask_aug,
        )
        self.file_service.save_processed_data(processed, self.run_dir)



