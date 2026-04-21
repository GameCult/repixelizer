from __future__ import annotations

import numpy as np

from .metrics import isolated_pixel_rate, luminance
from .types import CleanupArtifacts


def _local_energy(patch: np.ndarray) -> float:
    center = patch[1, 1]
    neighbors = np.stack(
        [
            patch[1, 0],
            patch[1, 2],
            patch[0, 1],
            patch[2, 1],
        ],
        axis=0,
    )
    dist = np.linalg.norm(neighbors - center[None, :], axis=-1)
    isolation = float(np.mean(dist))
    alpha = patch[..., 3]
    alpha_kink = float(np.mean(alpha * (1.0 - alpha)))
    return isolation + alpha_kink * 0.25


def cleanup_pixels(rgba: np.ndarray, source_guidance: np.ndarray, iterations: int = 2) -> CleanupArtifacts:
    result = rgba.copy()
    height, width = result.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    for _ in range(iterations):
        updated = result.copy()
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if source_guidance[y, x] > 0.55:
                    continue
                patch = result[y - 1 : y + 2, x - 1 : x + 2]
                current_energy = _local_energy(patch)
                neighbors = [
                    result[y, x],
                    result[y - 1, x],
                    result[y + 1, x],
                    result[y, x - 1],
                    result[y, x + 1],
                    np.mean(patch.reshape(-1, 4), axis=0),
                    np.median(patch.reshape(-1, 4), axis=0),
                ]
                best = result[y, x]
                best_energy = current_energy
                for candidate in neighbors[1:]:
                    patch_candidate = patch.copy()
                    patch_candidate[1, 1] = candidate
                    energy = _local_energy(patch_candidate)
                    if energy + 1e-6 < best_energy:
                        best_energy = energy
                        best = candidate
                updated[y, x] = best
                heatmap[y, x] = max(heatmap[y, x], current_energy - best_energy)
        result = np.clip(updated, 0.0, 1.0)
    return CleanupArtifacts(cleaned_rgba=result, isolated_heatmap=heatmap)
