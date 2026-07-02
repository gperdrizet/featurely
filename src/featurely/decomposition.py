"""Polynomial expansion and PCA component selection.

Degree-2 polynomial expansion generates every squared term and pairwise
product, which quickly produces hundreds of correlated columns. PCA rotates
that expanded set into orthogonal components ordered by variance, and
cross-validation over component counts finds how many are worth keeping.
Note that PCA here is fit on the full dataset before cross-validation; it is
unsupervised (never sees the target), so any optimism this introduces is
mild, but a strict production pipeline would fit PCA inside each fold.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def make_polynomial_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    degree: int = 2,
    include_bias: bool = False,
) -> pd.DataFrame:
    """Return the polynomial expansion of the selected columns as a frame.

    Column names come from scikit-learn but are sanitized for CSV round
    trips: spaces (products) become ``_x_`` and carets (powers) become
    ``_pow``.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    expanded = poly.fit_transform(df[list(feature_cols)])
    names = [name.replace(" ", "_x_").replace("^", "_pow") for name in poly.get_feature_names_out(list(feature_cols))]
    return pd.DataFrame(expanded, columns=names, index=df.index)


def plot_pca_variance(
    x_df: pd.DataFrame,
    title: str = "PCA cumulative explained variance",
) -> PCA:
    """Plot cumulative explained variance and return the fitted PCA.

    Features are standard-scaled first; PCA directions are meaningless when
    columns live on wildly different scales. Reference lines mark the 90,
    95, and 99 percent variance thresholds.
    """
    x = StandardScaler().fit_transform(x_df)
    pca = PCA().fit(x)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, linewidth=1.5)

    for threshold in (0.90, 0.95, 0.99):
        n_at = int(np.searchsorted(cumulative, threshold) + 1)
        ax.axhline(threshold, color="gray", linewidth=0.6, linestyle="--")
        ax.annotate(
            f"{threshold:.0%} at n = {n_at}",
            xy=(n_at, threshold),
            xytext=(n_at + len(cumulative) * 0.03, threshold - 0.05),
            fontsize=8,
            arrowprops={"arrowstyle": "->", "lw": 0.6},
        )

    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    return pca


def scan_pca_components(
    x_df: pd.DataFrame,
    y: pd.Series,
    component_grid: list[int],
    cv: int = 10,
) -> pd.DataFrame:
    """Cross-validate a linear model on the first n principal components.

    PCA is fit once at the largest grid value; truncating to the first n
    columns of the projection is equivalent to fitting PCA with
    n_components=n, so the scan avoids refitting for every grid point.
    Returns a DataFrame with one row per component count.
    """
    x = StandardScaler().fit_transform(x_df)
    max_n = min(max(component_grid), x.shape[1])
    projected = PCA(n_components=max_n).fit_transform(x)

    rows = []
    for n in component_grid:
        if n > max_n:
            continue
        scores = cross_val_score(LinearRegression(), projected[:, :n], y, cv=cv, scoring="r2")
        rows.append(
            {
                "n_components": n,
                "mean_r2": scores.mean(),
                "std_r2": scores.std(),
                "scores": scores,
            }
        )
        print(f"n = {n:>4}: mean R2 = {scores.mean():.4f} ± {scores.std():.4f}")

    return pd.DataFrame(rows)


def plot_pca_component_scan(
    results_df: pd.DataFrame,
    title: str = "CV R2 by number of PCA components",
) -> int:
    """Plot the component scan curve and return the best component count.

    The best count maximizes mean CV R2; the shaded band shows one standard
    deviation across folds, a visual check on whether nearby counts are
    practically equivalent.
    """
    n_vals = results_df["n_components"].values
    means = results_df["mean_r2"].values
    stds = results_df["std_r2"].values

    best_idx = int(np.argmax(means))
    best_n = int(n_vals[best_idx])

    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_vals, means, marker="o", linewidth=1.5)
    ax.fill_between(n_vals, means - stds, means + stds, alpha=0.2)
    ax.axvline(best_n, color="#c00000", linewidth=1, linestyle="--")
    ax.annotate(
        f"best n = {best_n}\nR2 = {means[best_idx]:.4f}",
        xy=(best_n, means[best_idx]),
        xytext=(best_n + max(n_vals) * 0.05, means[best_idx] - stds[best_idx]),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.6},
    )
    ax.set_xlabel("Number of components")
    ax.set_ylabel("R2 score (10-fold CV)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    return best_n
