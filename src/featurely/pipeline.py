from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from ._display import show_figure

DEFAULT_PIPELINE_COLORS = {
    "raw": "#aaaaaa",
    "+ p_censored": "#ffc000",
    "+ cleaned": "#5b9bd5",
    "+ transforms": "#70ad47",
    "+ interactions": "#7030a0",
}

_PIPELINE_RESULTS_COLUMNS = ["stage", "mean_r2", "std_r2", "pct_vs_raw", "color", "scores"]


def _empty_pipeline_results_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_PIPELINE_RESULTS_COLUMNS)


def _load_pipeline_results(
    results_df: pd.DataFrame | None,
    results_path: str | Path | None,
) -> pd.DataFrame:
    """Load persisted pipeline results when a path is provided, otherwise use in-memory data."""
    if results_path is not None:
        path = Path(results_path)
        if path.exists():
            loaded = pd.read_pickle(path)
            # Keep only expected columns so downstream plotting remains stable.
            return loaded.reindex(columns=_PIPELINE_RESULTS_COLUMNS)
        return _empty_pipeline_results_df()

    if results_df is None:
        return _empty_pipeline_results_df()

    return results_df.reindex(columns=_PIPELINE_RESULTS_COLUMNS).copy()


def _save_pipeline_results(results_df: pd.DataFrame, results_path: str | Path | None) -> None:
    if results_path is None:
        return

    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_pickle(path)


def _recompute_pct_vs_raw(results_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute percent improvement for all stages based on the current raw baseline row."""
    if results_df.empty or "raw" not in results_df["stage"].values:
        return results_df

    raw_mean = float(results_df.loc[results_df["stage"] == "raw", "mean_r2"].iloc[0])
    if raw_mean == 0:
        results_df["pct_vs_raw"] = 0.0
        return results_df

    results_df["pct_vs_raw"] = (results_df["mean_r2"] - raw_mean) / abs(raw_mean) * 100
    results_df.loc[results_df["stage"] == "raw", "pct_vs_raw"] = 0.0
    return results_df


def add_pipeline_step(
    results_df: pd.DataFrame | None,
    label: str,
    x: pd.DataFrame,
    y: pd.Series,
    color: str | None = None,
    color_map: dict[str, str] | None = None,
    cv: int = 10,
    results_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run cross-validation for one pipeline step and upsert by stage name.

    When ``results_path`` is provided, prior results are loaded from disk before
    the update and saved back after the update. This supports sequential notebook
    runs without duplicated stage rows.

    Args:
        results_df: Prior results frame, or None to start fresh or load from disk.
        label: Stage name; an existing row with this name is replaced.
        x: Feature matrix for this stage.
        y: Target series.
        color: Explicit bar color; overrides ``color_map`` when given.
        color_map: Stage-name-to-color mapping; defaults to
            ``DEFAULT_PIPELINE_COLORS``.
        cv: Number of cross-validation folds.
        results_path: Optional pickle path for persisted, rerun-safe results.

    Returns:
        The updated results frame with recomputed percent-vs-raw values.
    """
    updated = _load_pipeline_results(results_df, results_path)
    scores = cross_val_score(LinearRegression(), x, y, cv=cv, scoring="r2")

    if color is None:
        palette = color_map or DEFAULT_PIPELINE_COLORS
        color = palette.get(label, "#aaaaaa")

    raw_rows = updated.loc[updated["stage"] == "raw", "mean_r2"] if not updated.empty else pd.Series(dtype=float)
    raw_mean = float(raw_rows.iloc[0]) if len(raw_rows) > 0 else 0.0
    pct_vs_raw = 0.0 if label == "raw" or raw_mean == 0.0 else (scores.mean() - raw_mean) / abs(raw_mean) * 100

    row = {
        "stage": label,
        "mean_r2": scores.mean(),
        "std_r2": scores.std(),
        "pct_vs_raw": pct_vs_raw,
        "color": color,
        "scores": scores,
    }

    if not updated.empty:
        # Keep latest entry when prior notebook runs created duplicate stage names.
        updated = updated.drop_duplicates(subset=["stage"], keep="last").reset_index(drop=True)

    stage_matches = updated.index[updated["stage"] == label].tolist()
    if stage_matches:
        idx = stage_matches[0]
        for col, value in row.items():
            updated.at[idx, col] = value
    else:
        updated.loc[len(updated)] = row

    updated = _recompute_pct_vs_raw(updated)
    _save_pipeline_results(updated, results_path)
    return updated


def plot_pipeline_steps(
    results_df: pd.DataFrame | None,
    title: str = "CV R2 pipeline steps",
    results_path: str | Path | None = None,
) -> None:
    """Draw stage-wise cross-validation boxplots and print a text summary.

    When ``results_path`` is provided, results are loaded from disk before plotting.

    Args:
        results_df: Results frame from ``add_pipeline_step``, or None when
            loading from ``results_path``.
        title: Plot title.
        results_path: Optional pickle path to load persisted results from.

    Raises:
        ValueError: If no results are available to plot.
    """
    results_df = _load_pipeline_results(results_df, results_path)
    if results_df.empty:
        raise ValueError("No pipeline results available to plot.")

    # Keep summaries consistent even when loading results created before pct_vs_raw existed.
    results_df = _recompute_pct_vs_raw(results_df.copy())

    labels = results_df["stage"].tolist()
    all_scores = results_df["scores"].tolist()
    colors = results_df["color"].tolist()

    _, ax = plt.subplots(figsize=(max(5, 2 * len(labels)), 4))
    bp = ax.boxplot(all_scores, tick_labels=labels, patch_artist=True)

    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for median in bp["medians"]:
        median.set(color="black", linewidth=1.5)

    ax.set_title(title)
    ax.set_ylabel("R2 score (10-fold CV)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    show_figure()

    for _, row in results_df.iterrows():
        pct_vs_raw = float(row["pct_vs_raw"]) if pd.notna(row["pct_vs_raw"]) else 0.0
        print(f"{row['stage']:>25}: mean R2 = {row['mean_r2']:.4f} ± {row['std_r2']:.4f}  ({pct_vs_raw:+.2f}% vs raw)")
