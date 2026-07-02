"""K-means cluster membership features.

Clustering partitions rows into groups; one-hot membership lets a linear
model fit a separate intercept per group, and centroid distance adds a
within-group gradient. Clustering uses only feature columns, so the derived
features are target-free.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def plot_kmeans_selection(
    df: pd.DataFrame,
    features: list[str],
    k_range=None,
    random_state: int = 315,
    sample_size: int = 5000,
    title: str | None = None,
) -> dict[int, float]:
    """Plot inertia (elbow) and silhouette score across candidate k values.

    Inertia always decreases with k, so we look for the elbow where the
    marginal gain flattens. Silhouette measures how well separated the
    clusters are; it is computed on a random subsample because the full
    pairwise calculation is quadratic in row count. Returns a dict of
    silhouette score by k.
    """
    if k_range is None:
        k_range = range(2, 13)
    x = StandardScaler().fit_transform(df[list(features)])

    ks, inertias, silhouettes = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(x)
        ks.append(k)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(x, km.labels_, sample_size=sample_size, random_state=random_state))
        print(f"k = {k:>2}: inertia = {km.inertia_:>12.1f},  silhouette = {silhouettes[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ks, inertias, marker="o")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow curve")

    axes[1].plot(ks, silhouettes, marker="o", color="#ed7d31")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette by k")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()

    return dict(zip(ks, silhouettes, strict=False))


def compute_kmeans_features(
    df: pd.DataFrame,
    features: list[str],
    k: int,
    prefix: str,
    one_hot: bool = True,
    add_distance: bool = True,
    random_state: int = 315,
) -> pd.DataFrame:
    """Return cluster membership and centroid distance candidates.

    Features are standard-scaled before clustering so no single column
    dominates the distance metric. One-hot columns are named
    ``{prefix}_{label}``; the distance column is ``{prefix}_centroid_dist``
    and is measured in scaled feature space.
    """
    x = StandardScaler().fit_transform(df[list(features)])
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(x)

    out = pd.DataFrame(index=df.index)

    if one_hot:
        labels = pd.Series(km.labels_, index=df.index)
        dummies = pd.get_dummies(labels, prefix=prefix, dtype=float)
        out = pd.concat([out, dummies], axis=1)

    if add_distance:
        # Distance from each row to its own cluster center: a within-cluster
        # gradient that one-hot membership alone cannot express.
        out[f"{prefix}_centroid_dist"] = np.linalg.norm(x - km.cluster_centers_[km.labels_], axis=1)

    return out
