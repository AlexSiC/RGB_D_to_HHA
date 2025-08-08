from __future__ import annotations

import random
from typing import Dict

import albumentations as A
import cv2
import numpy as np

from .data_models import PipelineConfig


class AugmentationService:
    """Synchronous geometric augmentations for RGB, depth and mask using Albumentations."""

    def apply(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        config: PipelineConfig.AugmentationConfig,
    ) -> Dict[str, np.ndarray]:
        """Apply the configured augmentations synchronously.

        Returns a dict with keys: 'rgb', 'depth', 'mask'.
        """
        if not config.enabled:
            return {"rgb": rgb, "depth": depth, "mask": mask}

        # Determinism
        random.seed(config.seed)
        np.random.seed(config.seed)

        height = int(config.crop_size[1])
        width = int(config.crop_size[0])

        transforms = [
            A.HorizontalFlip(p=config.horizontal_flip_prob),
            A.RandomScale(scale_limit=config.random_scale_limit, p=1.0),
            A.Rotate(limit=config.rotate_limit, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
            A.RandomCrop(height=height, width=width, p=1.0),
        ]

        if config.pad_if_needed:
            transforms.insert(
                0,
                A.PadIfNeeded(
                    min_height=height,
                    min_width=width,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=1.0,
                ),
            )

        pipeline = A.Compose(
            transforms,
            additional_targets={
                "depth": "image",  # treat as image for geometric transforms
                "mask": "mask",    # ensure nearest-neighbor for masks
            },
        )

        result = pipeline(image=rgb, depth=depth, mask=mask)
        return {"rgb": result["image"], "depth": result["depth"], "mask": result["mask"]}



