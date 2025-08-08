from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class HHAService:
    """Wrapper around an external 'depth2hha' provider to compute HHA images.

    Expects input depth in meters and an intrinsic camera matrix (3x3).
    """

    def __init__(self) -> None:
        self._converter: Optional[Callable[..., np.ndarray]] = self._resolve_converter()

    def _resolve_converter(self) -> Optional[Callable[..., np.ndarray]]:
        try:
            import depth2hha  # type: ignore
        except Exception:
            return None

        # Try common entry points
        for name in ("convert", "depth_to_hha", "compute_hha", "computeHHA"):
            func = getattr(depth2hha, name, None)
            if callable(func):
                return func  # type: ignore[return-value]
        return None

    def convert(self, depth_map_m: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        """Convert a metric depth map to an HHA image using the external library.

        Raises a clear error if the converter is unavailable.
        """
        if depth_map_m.ndim != 2:
            raise ValueError("depth_map_m must be a 2D array of meters")
        if camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix must be 3x3")

        if self._converter is None:
            raise RuntimeError(
                "depth2hha backend is not available. Please ensure a local module 'depth2hha' "
                "is installed or available on PYTHONPATH with a callable one of: convert, depth_to_hha, "
                "compute_hha, computeHHA."
            )

        hha = self._converter(depth_map_m, camera_matrix)
        if not isinstance(hha, np.ndarray) or (hha.ndim != 3 or hha.shape[2] != 3):
            raise RuntimeError("depth2hha returned unexpected result; expected HxWx3 ndarray")
        return hha



