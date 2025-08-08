from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata


class InpaintingService:
    """Depth inpainting for filling gaps in depth maps.

    Methods:
        - 'linear_nearest': cascade of linear interpolation, then nearest for remaining gaps.
        - 'none': return input converted to meters without filling.
    """

    def apply(self, depth_map: np.ndarray, method: str) -> np.ndarray:
        """Apply inpainting to a depth map.

        Args:
            depth_map: Depth map in millimeters (2D array).
            method: Inpainting method. Supported: 'linear_nearest', 'none'.

        Returns:
            np.ndarray: Depth map in meters with gaps filled according to method.
        """
        if depth_map.ndim != 2:
            raise ValueError("depth_map must be a 2D array")

        # Normalize to meters (float32) from millimeters
        depth_m = depth_map.astype(np.float32) / 1000.0

        if method == "none":
            return depth_m

        if method != "linear_nearest":
            raise ValueError(f"Unsupported inpainting method: {method}")

        height, width = depth_m.shape

        # Invalid where zeros or NaNs
        invalid = (depth_map == 0) | np.isnan(depth_m)
        valid = ~invalid

        if not np.any(valid):
            # No valid points at all; return zeros
            return np.zeros_like(depth_m, dtype=np.float32)

        yy, xx = np.indices((height, width))
        points = np.stack([yy[valid], xx[valid]], axis=1)
        values = depth_m[valid]

        # First pass: linear interpolation
        filled_linear = griddata(points, values, (yy, xx), method="linear")

        # Second pass: nearest for remaining NaNs
        missing = np.isnan(filled_linear)
        if np.any(missing):
            filled_nearest = griddata(points, values, (yy, xx), method="nearest")
            filled_linear[missing] = filled_nearest[missing]

        # Where still NaN (edge cases), set to zero (no data)
        filled = np.nan_to_num(filled_linear, nan=0.0)
        return filled.astype(np.float32)



