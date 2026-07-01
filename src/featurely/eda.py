from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Plot histogram distributions for all non-target feature columns."""
    features = [c for c in df.columns if c not in ("MedHouseVal", "p_censored")]

    n_cols = 4
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), squeeze=False)
    fig.suptitle("Feature distributions", fontsize=11)

    for i, col in enumerate(features):
        ax = axes[i // n_cols, i % n_cols]
        ax.hist(df[col], bins=50, edgecolor="black", color="grey")
        ax.set_title(col)
        ax.set_ylabel("Frequency")

    for j in range(len(features), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    plt.tight_layout()
    plt.show()


def get_feature_correlations(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations for each unique feature pair."""
    feature_pairs = [(f1, f2) for i, f1 in enumerate(features) for j, f2 in enumerate(features) if i < j]

    rows: list[dict[str, float | str]] = []
    for feature_a, feature_b in feature_pairs:
        pearson_r, pearson_p = pearsonr(df[feature_a], df[feature_b])
        spearman_r, spearman_p = spearmanr(df[feature_a], df[feature_b])
        rows.append(
            {
                "Feature A": feature_a,
                "Feature B": feature_b,
                "Pearson r": pearson_r,
                "Pearson p": pearson_p,
                "Spearman r": spearman_r,
                "Spearman p": spearman_p,
            }
        )

    corr_df = pd.DataFrame(rows).sort_values(by="Pearson r", ascending=False).reset_index(drop=True)
    corr_df.set_index(["Feature A", "Feature B"], inplace=True)
    return corr_df


def plot_feature_correlations(df: pd.DataFrame, features: list[str]) -> None:
    """Plot pairwise feature scatters with linear fits and correlation annotations."""
    feature_pairs = [(f1, f2) for i, f1 in enumerate(features) for j, f2 in enumerate(features) if i < j]
    feature_correlations_df = get_feature_correlations(df, features)

    n_pairs = len(feature_pairs)
    n_cols = min(4, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 2.5 * n_rows),
        layout="constrained",
        squeeze=False,
    )
    fig.suptitle("Feature correlations")

    for i, (feature_a, feature_b) in enumerate(feature_pairs):
        ax = axes[i // n_cols, i % n_cols]
        ax.scatter(df[feature_a], df[feature_b], color="black", s=4, alpha=0.2)
        ax.set_xlabel(feature_a)
        ax.set_ylabel(feature_b)

        x = df[feature_a].values.reshape(-1, 1)
        y = df[feature_b].values
        model = LinearRegression().fit(x, y)
        x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        ax.plot(x_range, y_pred, color="red", linewidth=1)

        pearson_r = feature_correlations_df.loc[(feature_a, feature_b), "Pearson r"]
        spearman_r = feature_correlations_df.loc[(feature_a, feature_b), "Spearman r"]
        ax.text(
            0.25,
            0.95,
            f"Pearson r: {pearson_r:.2f}\nSpearman r: {spearman_r:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.75),
        )

    for j in range(n_pairs, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    plt.show()


def plot_features_vs_label(
    df: pd.DataFrame, features: list[str], label: str = "MedHouseVal"
) -> None:
    """Plot each feature against target with fitted line and correlation values."""
    n_cols = min(4, len(features))
    n_rows = (len(features) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(f"Feature correlations with {label}")

    for i, feature in enumerate(features):
        ax = axes[i // n_cols, i % n_cols]
        ax.scatter(df[feature], df[label], color="black", s=2, alpha=0.15)
        ax.set_xlabel(feature)
        ax.set_ylabel(label)

        x = df[feature].values.reshape(-1, 1)
        y = df[label].values
        model = LinearRegression().fit(x, y)
        x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        ax.plot(x_range, model.predict(x_range), color="red", linewidth=1)

        pearson_r, _ = pearsonr(df[feature], df[label])
        spearman_r, _ = spearmanr(df[feature], df[label])
        ax.text(
            0.25,
            0.95,
            f"Pearson r: {pearson_r:.2f}\nSpearman r: {spearman_r:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.75),
        )

    for j in range(len(features), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    plt.tight_layout()
    plt.show()
