from __future__ import annotations

import numpy as np

from .types import CleanupArtifacts


def cleanup_pixels(rgba: np.ndarray) -> CleanupArtifacts:
    result = _snap_alpha(rgba)
    heatmap = np.zeros(result.shape[:2], dtype=np.float32)
    return CleanupArtifacts(cleaned_rgba=result, isolated_heatmap=heatmap)


def _snap_alpha(rgba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    result = rgba.copy()
    alpha = (result[..., 3] >= threshold).astype(np.float32)
    result[..., 3] = alpha
    result[..., :3] *= alpha[..., None]
    return result
