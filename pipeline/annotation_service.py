from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


class AnnotationService:
    """Converts polygon annotations into a single-channel mask image.

    Expects polygons with normalized coordinates in range [0, 1].
    Each polygon is provided as a tuple (class_id, coords), where coords is
    an array-like of shape (N, 2) with (x, y) vertex coordinates.
    """

    def convert_polygons_to_mask(self, polygons: List[Tuple[int, np.ndarray]], shape: Tuple[int, int]) -> np.ndarray:
        """Rasterize normalized polygons into an integer mask.

        Args:
            polygons: List of (class_id, vertices) where vertices are normalized.
            shape: Target mask shape as (height, width).

        Returns:
            np.ndarray: Single-channel uint8 mask with class indices.
        """
        height, width = int(shape[0]), int(shape[1])
        mask = np.zeros((height, width), dtype=np.uint8)

        for class_id, coords in polygons:
            if coords is None:
                continue
            vertices = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
            xs = np.clip(np.round(vertices[:, 0] * (width - 1)), 0, width - 1).astype(np.int32)
            ys = np.clip(np.round(vertices[:, 1] * (height - 1)), 0, height - 1).astype(np.int32)
            pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], int(class_id))

        return mask



