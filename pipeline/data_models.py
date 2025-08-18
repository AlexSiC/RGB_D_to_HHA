from __future__ import annotations

from typing import List, Tuple

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class FrameIdentifier(BaseModel):
    """A unique identifier for a single data frame, based on filename.

    Serves as a universal key to bind raw RGB, depth and annotation files
    that belong to the same frame.
    """

    base_name: str
    raw_rgb_path: str
    raw_depth_path: str
    raw_mask_path: str


class CameraIntrinsics(BaseModel):
    """Camera calibration matrix parameters (pinhole intrinsics)."""

    fx: float
    fy: float
    cx: float
    cy: float

    def to_numpy_array(self) -> np.ndarray:
        """Return a 3x3 intrinsic matrix as a NumPy array."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


class PipelineConfig(BaseModel):
    """Strongly-typed representation of the config.yaml file."""

    class InpaintingConfig(BaseModel):
        method: str = Field(..., description="e.g., 'linear_nearest', 'rbf', 'none'")

    class AugmentationConfig(BaseModel):
        enabled: bool = True
        seed: int = 42
        horizontal_flip_prob: float = 0.5
        random_scale_limit: float = 0.1
        crop_size: Tuple[int, int]
        rotate_limit: int = 15
        pad_if_needed: bool = True
        # Optional list of seeds for multi-variant augmentation in a single run
        seeds: List[int] = []

    class CamerasConfig(BaseModel):
        """Configuration for camera intrinsic parameters."""

        rgb_camera_matrix: CameraIntrinsics
        depth_camera_matrix: CameraIntrinsics

    class HHAConfig(BaseModel):
        """Parameters controlling HHA base-plane estimation.

        All fractional values are relative to image dimensions.
        """

        # Use only the bottom band of the image when sampling points for plane fitting
        bottom_band_frac: float = Field(0.35, ge=0.0, le=1.0)
        # Additionally keep side stripes in the bottom band to catch visible floor at edges
        side_band_frac: float = Field(0.0, ge=0.0, le=0.5)
        # Exclude a centered horizontal stripe within the bottom band (e.g., where bags lie)
        center_exclude_width_frac: float = Field(0.4, ge=0.0, le=1.0)

        # RANSAC plane fit parameters (depth units are centimetres inside the vendor code)
        ransac_thresh_cm: float = Field(2.0, ge=0.1)
        min_inlier_ratio: float = Field(0.1, ge=0.0, le=1.0)
        # Deterministic RANSAC: set a fixed seed for reproducibility
        ransac_seed: int = 42
        # Initial guess for gravity direction in camera frame. Options: 'y', '-y', 'z', '-z'
        gravity_init: str = "-z"

    class PathsConfig(BaseModel):
        raw_dir: str
        processed_dir: str

    class OutputsConfig(BaseModel):
        """Output/visualization toggles."""
        save_hha_channels_jet: bool = False
        enable_train_export: bool = False
        train_dir: str = "data/train"
        save_hha_u8_in_train: bool = False

    inpainting: InpaintingConfig
    augmentation: AugmentationConfig
    cameras: CamerasConfig
    hha: HHAConfig = HHAConfig()
    paths: PathsConfig
    outputs: OutputsConfig = OutputsConfig()


class RawFrameData(BaseModel):
    """Data structure for a single, unprocessed frame."""

    # Allow numpy arrays to be used as field types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    identifier: FrameIdentifier
    rgb_image: np.ndarray
    depth_map_mm: np.ndarray  # Raw depth in millimeters
    polygons: List[Tuple[int, np.ndarray]]  # List of (class_id, polygon_coords)


class ProcessedFrameData(BaseModel):
    """Data structure for a frame after processing, ready for export."""

    # Allow numpy arrays to be used as field types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    identifier: FrameIdentifier
    rgb_image: np.ndarray  # Potentially augmented
    depth_map_filled_m: np.ndarray  # Inpainted and converted to meters
    hha_image: np.ndarray
    segmentation_mask: np.ndarray  # 8-bit single-channel mask



