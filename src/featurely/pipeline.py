from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

DEFAULT_PIPELINE_COLORS = {
    "raw": "#aaaaaa",
    "+ p_censored": "#ffc000",
    "+ cleaned": "#5b9bd5",
    "+ transforms": "#70ad47",
    "+ interactions": "#7030a0",
}


def add_pipeline_step(
    results_df: pd.DataFrame,
    label: str,
    x: pd.DataFrame,
    y: pd.Series,
    color: str | None = None,
    color_map: dict[str, str] | None = None,
    cv: int = 10,
) -> pd.DataFrame:
    """Run cross-validation for one pipeline step and append row-level summary."""
    scores = cross_val_score(LinearRegression(), x, y, cv=cv, scoring="r2")

    if color is None:
        palette = color_map or DEFAULT_PIPELINE_COLORS
        color = palette.get(label, "#aaaaaa")

    pct_vs_raw = (
        0.0
        if len(results_df) == 0
        else (scores.mean() - results_df["mean_r2"].iloc[0])
        / abs(results_df["mean_r2"].iloc[0])
        * 100
    )

    return pd.concat(
        [
            results_df,
            pd.DataFrame(
                [
                    {
                        "stage": label,
                        "mean_r2": scores.mean(),
                        "std_r2": scores.std(),
                        "pct_vs_raw": pct_vs_raw,
                        "color": color,
                        "scores": scores,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )


def plot_pipeline_steps(results_df: pd.DataFrame, title: str = "CV R2 pipeline steps") -> None:
    """Draw stage-wise cross-validation boxplots and print a text summary."""
    labels = results_df["stage"].tolist()
    all_scores = results_df["scores"].tolist()
    colors = results_df["color"].tolist()

    _, ax = plt.subplots(figsize=(max(5, 2 * len(labels)), 4))
    bp = ax.boxplot(all_scores, tick_labels=labels, patch_artist=True)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for median in bp["medians"]:
        median.set(color="black", linewidth=1.5)

    ax.set_title(title)
    ax.set_ylabel("R2 score (10-fold CV)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()

    for _, row in results_df.iterrows():
        pct_str = (
            f"  ({row['pct_vs_raw']:+.2f}% vs raw)"
            if row["pct_vs_raw"] != 0.0
            else ""
        )
        print(
            f"{row['stage']:>25}: mean R2 = {row['mean_r2']:.4f} "
            f"+- {row['std_r2']:.4f}{pct_str}"
        )
