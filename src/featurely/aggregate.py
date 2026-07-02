"""Bin-level aggregate features.

Binning a driver feature and attaching per-bin summary statistics gives each
row context about the group it belongs to, for example the mean of one
feature across all rows in the same quantile bin of another. Only feature
columns are aggregated, so the candidates carry no target information and
cannot leak.
"""

from __future__ import annotations

import pandas as pd


def compute_bin_aggregates(
    df: pd.DataFrame,
    bin_feature: str,
    agg_features: list[str],
    n_bins: int = 10,
    stats: tuple[str, ...] = ("mean",),
) -> pd.DataFrame:
    """Return per-bin summary statistic candidates for one binned feature.

    Rows are assigned to quantile bins of ``bin_feature`` (equal-count bins,
    so sparse regions do not produce empty groups), then each row receives
    the bin-level statistic of every column in ``agg_features``.

    Args:
        df: Input frame; not modified.
        bin_feature: Column whose quantile bins define the groups.
        agg_features: Columns to summarize within each bin.
        n_bins: Number of quantile bins.
        stats: Summary statistics to compute, by pandas name, e.g.
            ``("mean", "median")``.

    Returns:
        A frame of candidate columns named
        ``{bin_feature}bin{n_bins}_{stat}_{col}``.
    """

    bins = pd.qcut(df[bin_feature], q=n_bins, duplicates="drop")
    grouped = df.groupby(bins, observed=True)

    out = {}

    for stat in stats:
        for col in agg_features:
            name = f"{bin_feature}bin{n_bins}_{stat}_{col}"

            # transform broadcasts the group statistic back onto each row.
            out[name] = grouped[col].transform(stat)

    return pd.DataFrame(out, index=df.index)
