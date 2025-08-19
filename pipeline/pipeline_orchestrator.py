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
from .diagnostics_service import DiagnosticsService


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
        self.diag_service = DiagnosticsService()

        self._setup_logging()
        self.run_dir = self._create_run_dir()

    def _setup_logging(self) -> None:
        """Configure logging to write both to file and console.

        Creates the `logs/` directory if it does not exist and attaches two
        handlers:
        - File handler: `logs/pipeline.log` (UTF-8)
        - Stream handler: standard output

        Logging level is set to INFO.
        """
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
        """Create and return a unique directory for current pipeline run.

        The directory is placed under `processed_dir` with a timestamped name
        `run_YYYYMMDD_HHMMSS` to keep runs separated and reproducible.
        """
        processed_base = Path(self.config.paths.processed_dir)
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = processed_base / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def run_full_pipeline(self) -> None:
        """Discover frames and process them with error handling and progress bar.

        - Discovers frames using `FileService.discover_frames` based on
          `config.paths.raw_dir`.
        - Iterates with `tqdm` progress bar.
        - For each frame, calls `process_single_frame` inside try/except so that
          errors are logged and processing continues.
        - Collects failed `frame_id.base_name` values and writes them into
          `logs/failed_files.txt` if any failures occurred.
        """
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
        """Ensure RGB and depth maps have identical spatial dimensions.

        Raises:
            RuntimeError: if dimensions do not match.
        """
        rgb_h, rgb_w = raw.rgb_image.shape[:2]
        depth_h, depth_w = raw.depth_map_mm.shape[:2]
        if (rgb_h, rgb_w) != (depth_h, depth_w):
            raise RuntimeError(
                f"Dimension mismatch RGB({rgb_w}x{rgb_h}) vs Depth({depth_w}x{depth_h}) for {raw.identifier.base_name}"
            )

    def process_single_frame(self, frame_id: FrameIdentifier) -> None:
        """Process a single frame end-to-end and save artifacts.

        Steps:
            1. Load raw data (RGB, depth, polygons).
            2. Validate dimensions.
            3. Save raw depth (PNG, uint16 mm) for traceability.
            4. Inpaint depth (mm -> m) using configured method.
            5. Convert polygons to a rasterized mask.
            6. Compute baseline HHA and diagnostics (no augmentation) and save.
            7. For each requested augmentation seed, apply augmentation, compute
               HHA and diagnostics, and save into per-seed subdirectories.

        Args:
            frame_id: Identifier with paths to RGB/depth/annotations.
        """
        raw: RawFrameData = self.file_service.load_raw_data(frame_id)
        self._validate_dimensions(raw)

        # Save raw depth before inpainting
        self.file_service.save_raw_depth_png(frame_id, raw.depth_map_mm, self.run_dir)

        # Inpainting (mm -> m inside service)
        depth_filled_m = self.inpainting_service.apply(raw.depth_map_mm, self.config.inpainting.method)

        # Annotation conversion (normalized polygons -> mask)
        mask = self.annotation_service.convert_polygons_to_mask(raw.polygons, raw.rgb_image.shape[:2])

        # Prepare list of seeds for multi-variant run (baseline + variants)
        seeds = list(self.config.augmentation.seeds or [])
        # If augmentation is disabled, do not produce any variants beyond baseline
        if getattr(self.config.augmentation, "enabled", True):
            variant_seeds = seeds if seeds else [self.config.augmentation.seed]
        else:
            variant_seeds = []

        # Always produce baseline (no augmentation) in root run_dir
        K = self.config.cameras.depth_camera_matrix.to_numpy_array()
        roi_cfg = {
            'bottom_band_frac': float(self.config.hha.bottom_band_frac),
            'side_band_frac': float(self.config.hha.side_band_frac),
            'center_exclude_width_frac': float(self.config.hha.center_exclude_width_frac),
            'ransac_seed': int(self.config.hha.ransac_seed),
            'gravity_init': str(self.config.hha.gravity_init),
        }
        hha_baseline = self.hha_service.convert(
            depth_filled_m.astype(np.float32),
            K.astype(np.float32),
            roi=roi_cfg,
            exclude_mask=mask.astype(bool),
            ransac_thresh_cm=float(self.config.hha.ransac_thresh_cm),
            min_inlier_ratio=float(self.config.hha.min_inlier_ratio),
        )
        # Diagnostics for baseline
        diagnostics = self.diag_service.compute(
            depth_filled_m.astype(np.float32),
            K.astype(np.float32),
            roi=roi_cfg,
            exclude_mask=mask.astype(bool),
            ransac_thresh_cm=float(self.config.hha.ransac_thresh_cm),
            min_inlier_ratio=float(self.config.hha.min_inlier_ratio),
        )
        processed_baseline = ProcessedFrameData(
            identifier=frame_id,
            rgb_image=raw.rgb_image,
            depth_map_filled_m=depth_filled_m,
            hha_image=hha_baseline,
            segmentation_mask=mask,
        )
        self.file_service.save_processed_data(
            processed_baseline,
            self.run_dir,
            save_hha_channels_jet=getattr(self.config.outputs, "save_hha_channels_jet", False),
            outputs=getattr(self.config, 'outputs', None),
            diagnostics=diagnostics,
        )

        for seed in variant_seeds:
            # Clone aug config with current seed
            aug_cfg = self.config.augmentation.model_copy(deep=True)
            aug_cfg.seed = seed

            aug = self.augmentation_service.apply(raw.rgb_image, depth_filled_m, mask, aug_cfg)
            rgb_aug = aug["rgb"]
            depth_aug = aug["depth"]
            mask_aug = aug["mask"]

            # HHA conversion using depth camera intrinsics
            K = self.config.cameras.depth_camera_matrix.to_numpy_array()
            hha = self.hha_service.convert(
                depth_aug.astype(np.float32),
                K.astype(np.float32),
                roi=roi_cfg,
                exclude_mask=mask_aug.astype(bool),
                ransac_thresh_cm=float(self.config.hha.ransac_thresh_cm),
                min_inlier_ratio=float(self.config.hha.min_inlier_ratio),
            )
            diagnostics_aug = self.diag_service.compute(
                depth_aug.astype(np.float32),
                K.astype(np.float32),
                roi=roi_cfg,
                exclude_mask=mask_aug.astype(bool),
                ransac_thresh_cm=float(self.config.hha.ransac_thresh_cm),
                min_inlier_ratio=float(self.config.hha.min_inlier_ratio),
            )

            processed = ProcessedFrameData(
                identifier=frame_id,
                rgb_image=rgb_aug,
                depth_map_filled_m=depth_aug,
                hha_image=hha,
                segmentation_mask=mask_aug,
            )

            # Create per-variant subdirectory if multiple variants
            run_dir = self.run_dir
            if len(variant_seeds) > 1:
                run_dir = run_dir / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

            self.file_service.save_processed_data(
                processed,
                run_dir,
                save_hha_channels_jet=getattr(self.config.outputs, "save_hha_channels_jet", False),
                outputs=getattr(self.config, 'outputs', None),
                diagnostics=diagnostics_aug,
            )



