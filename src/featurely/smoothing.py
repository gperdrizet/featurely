"""Spatial kernel smoothing of features.

Smoothing replaces each row's feature value with a weighted average over its
spatial neighborhood, suppressing block-level noise while preserving regional
structure. This is Nadaraya-Watson kernel regression truncated to the nearest
neighbors for tractability. Only feature columns are smoothed, never the
target, so the candidates are leakage-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def compute_spatial_smoothed(
    df: pd.DataFrame,
    features: list[str],
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    n_neighbors: int = 50,
    bandwidth: float | None = None,
    prefix: str = "smooth",
) -> pd.DataFrame:
    """Return Gaussian-kernel smoothed feature candidates.

    For each row, the smoothed value is a Gaussian-weighted average of the
    feature over its ``n_neighbors`` nearest points in latitude-longitude
    space (each row is its own nearest neighbor, so the original value gets
    the largest single weight). When ``bandwidth`` is None it defaults to
    the median distance to the farthest retained neighbor, which adapts the
    kernel width to local point density.
    """
    coords = df[[lat_col, lon_col]].values
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    dists, idx = nn.kneighbors(coords)

    if bandwidth is None:
        bandwidth = float(np.median(dists[:, -1]))
        print(f"Using adaptive bandwidth: {bandwidth:.4f} degrees")

    # Gaussian kernel weights, normalized per row so each smoothed value is
    # a proper weighted average of its neighborhood.
    weights = np.exp(-0.5 * (dists / bandwidth) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)

    out = {}
    for col in features:
        vals = df[col].values
        out[f"{prefix}_{col}"] = (weights * vals[idx]).sum(axis=1)

    return pd.DataFrame(out, index=df.index)
