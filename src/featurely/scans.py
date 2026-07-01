from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests


def _is_effectively_constant(values: np.ndarray, atol: float = 1e-12) -> bool:
    """Return True when a numeric vector has no meaningful variance."""
    if values.size == 0:
        return True
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    return bool(np.ptp(finite) <= atol)


def run_per_feature_scan(
    df: pd.DataFrame,
    features: list[str],
    transform_fn,
    label_prefix: str,
) -> dict[str, tuple[float, float]]:
    """Measure partial correlation of transformed features vs baseline residuals."""
    x = df.drop("MedHouseVal", axis=1).values
    y_arr = df["MedHouseVal"].values
    baseline_model = LinearRegression().fit(x, y_arr)
    residuals = y_arr - baseline_model.predict(x)
    baseline_r2 = baseline_model.score(x, y_arr)

    print(f"Baseline R2 (in-sample): {baseline_r2:.4f}")
    print()

    results: dict[str, tuple[float, float]] = {}
    width = len(label_prefix) + 1 + max(len(c) for c in features)

    for col in features:
        try:
            transformed = np.asarray(transform_fn(df[col]))
            if not np.isfinite(transformed).all():
                print(f"{label_prefix}_{col}: skipped (non-finite values)")
                continue

            if _is_effectively_constant(transformed):
                print(f"{label_prefix}_{col}: skipped (constant transformed values)")
                continue

            r, p = pearsonr(transformed, residuals)
            label = f"{label_prefix}_{col}"
            results[label] = (r, p)
            print(f"{label:>{width}}: r = {r:+.4f},  p = {p:.4f}")
        except Exception as exc:
            print(f"{label_prefix}_{col}: skipped ({exc})")

    return results


def plot_combined_per_feature_scan(scan_configs, title: str):
    """Grouped horizontal bar chart for per-feature scan results."""
    all_entries = []
    all_p_raws = []

    for prefix, name, results, color, _ in scan_configs:
        for label, (r, p) in results.items():
            col = label[len(prefix) + 1 :]
            all_entries.append({"transform": name, "feature": col, "r": r, "color": color})
            all_p_raws.append(p)

    _, p_corr, _, _ = multipletests(all_p_raws, alpha=0.05, method="fdr_bh")
    for entry, pc in zip(all_entries, p_corr):
        entry["sig"] = pc < 0.05

    first_prefix = scan_configs[0][0]
    first_results = scan_configs[0][2]
    features_ordered = [lbl[len(first_prefix) + 1 :] for lbl in first_results]

    n_f = len(features_ordered)
    n_t = len(scan_configs)
    group_height = 0.8
    bar_h = group_height / n_t
    offsets = np.linspace(-(n_t - 1) * bar_h / 2, (n_t - 1) * bar_h / 2, n_t)

    fig, ax = plt.subplots(figsize=(8, max(4, n_f * 0.8)))
    handles = []

    for t_idx, (prefix, name, results, color, _) in enumerate(scan_configs):
        y_base = np.arange(n_f, dtype=float)
        y_pos = y_base + offsets[t_idx]
        r_vals = []
        sigs = []

        for feat in features_ordered:
            label = f"{prefix}_{feat}"
            r_val = results.get(label, (0.0, 1.0))[0]
            r_vals.append(r_val)
            e_sig = next(
                (
                    e["sig"]
                    for e in all_entries
                    if e["transform"] == name and e["feature"] == feat
                ),
                False,
            )
            sigs.append(e_sig)

        ax.barh(y_pos, r_vals, height=bar_h * 0.85, color=color, alpha=0.7)
        handles.append(Patch(facecolor=color, alpha=0.7, label=name))

        for yp, r_val, sig in zip(y_pos, r_vals, sigs):
            if not sig:
                continue
            if r_val >= 0:
                ax.text(r_val + 0.003, yp, "*", va="center", ha="left", fontsize=9, color="black")
            else:
                ax.text(r_val - 0.003, yp, "*", va="center", ha="right", fontsize=9, color="black")

    ax.set_yticks(np.arange(n_f))
    ax.set_yticklabels(features_ordered)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r (vs baseline residuals)")
    ax.set_title(title)
    ax.legend(handles=handles, loc="lower right")
    plt.tight_layout()
    plt.show()

    return {(e["transform"], e["feature"]): e["sig"] for e in all_entries}


def plot_significant_transform_scatters(scan_configs, sig_dict, df: pd.DataFrame, title: str) -> None:
    """Plot transformed feature vs residuals for significant scan results."""
    x = df.drop("MedHouseVal", axis=1).values
    y_arr = df["MedHouseVal"].values
    residuals = y_arr - LinearRegression().fit(x, y_arr).predict(x)

    sig_pairs = [
        (prefix, name, col, transform_fn, color)
        for prefix, name, results, color, transform_fn in scan_configs
        for label in results
        for col in [label[len(prefix) + 1 :]]
        if sig_dict.get((name, col), False)
    ]

    if not sig_pairs:
        print("No statistically significant (transform, feature) pairs found.")
        return

    n_plots = len(sig_pairs)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=10)

    for i, (_, name, col, transform_fn, color) in enumerate(sig_pairs):
        ax = axes[i // n_cols, i % n_cols]
        try:
            transformed = np.asarray(transform_fn(df[col]))
        except Exception:
            ax.set_visible(False)
            continue

        ax.scatter(transformed, residuals, color=color, s=4, alpha=0.2)
        r, p = pearsonr(transformed, residuals)
        ax.set_xlabel(f"{name}({col})", fontsize=8)
        ax.set_ylabel("residuals", fontsize=8)
        ax.set_title(f"r = {r:+.3f}  p = {p:.3f}", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n_plots, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    plt.tight_layout()
    plt.show()


def run_pairwise_scan(
    df: pd.DataFrame,
    features: list[str],
    operation_fn,
    label_prefix: str,
    ordered: bool = False,
    include_self: bool = False,
) -> dict[str, tuple[float, float]]:
    """Evaluate pairwise interaction candidates via partial correlation."""
    n = len(features)

    if ordered:
        pairs = [(features[i], features[j]) for i in range(n) for j in range(n) if i != j]
    else:
        pairs = [
            (features[i], features[j])
            for i in range(n)
            for j in range(i if include_self else i + 1, n)
        ]

    x = df.drop("MedHouseVal", axis=1).values
    y = df["MedHouseVal"].values
    baseline_model = LinearRegression().fit(x, y)
    residuals = y - baseline_model.predict(x)
    baseline_r2 = baseline_model.score(x, y)

    print(f"Baseline R2 (in-sample): {baseline_r2:.4f}")
    print()

    results: dict[str, tuple[float, float]] = {}
    width = len(label_prefix) + 2 + max(len(a) + len(b) for a, b in pairs)

    for col_a, col_b in pairs:
        label = f"{label_prefix}_{col_a}_{col_b}"
        try:
            new_vals = operation_fn(df[col_a], df[col_b])
        except Exception:
            print(f"{label:>{width}}: skipped (operation error)")
            continue

        if not np.isfinite(new_vals).all():
            print(f"{label:>{width}}: skipped (non-finite values)")
            continue

        if _is_effectively_constant(np.asarray(new_vals)):
            print(f"{label:>{width}}: skipped (constant transformed values)")
            continue

        r, p = pearsonr(new_vals, residuals)
        results[label] = (r, p)
        print(f"{label:>{width}}: r = {r:+.4f},  p = {p:.4f}")

    return results


def plot_combined_pairwise_scan(scan_configs, title: str):
    """Grouped horizontal bar chart for pairwise interaction scan results."""
    all_entries = []
    all_p_raws = []

    for prefix, name, results, color, _ in scan_configs:
        for label, (r, p) in results.items():
            pair_suffix = label[len(prefix) + 1 :]
            all_entries.append({"op": name, "pair": pair_suffix, "r": r, "color": color})
            all_p_raws.append(p)

    _, p_corr, _, _ = multipletests(all_p_raws, alpha=0.05, method="fdr_bh")
    for entry, pc in zip(all_entries, p_corr):
        entry["sig"] = pc < 0.05

    pairs_ordered = sorted({e["pair"] for e in all_entries})
    n_p = len(pairs_ordered)
    n_op = len(scan_configs)
    group_height = 0.8
    bar_h = group_height / n_op
    offsets = np.linspace(-(n_op - 1) * bar_h / 2, (n_op - 1) * bar_h / 2, n_op)

    fig, ax = plt.subplots(figsize=(8, max(5, n_p * 0.45)))
    handles = []

    for op_idx, (prefix, name, results, color, _) in enumerate(scan_configs):
        y_base = np.arange(n_p, dtype=float)
        y_pos = y_base + offsets[op_idx]
        r_vals = []
        sigs = []

        for pair_suffix in pairs_ordered:
            label = f"{prefix}_{pair_suffix}"
            r_val = results.get(label, (0.0, 1.0))[0] if label in results else 0.0
            r_vals.append(r_val)
            e_sig = next(
                (e["sig"] for e in all_entries if e["op"] == name and e["pair"] == pair_suffix),
                False,
            )
            sigs.append(e_sig)

        ax.barh(y_pos, r_vals, height=bar_h * 0.85, color=color, alpha=0.7)
        handles.append(Patch(facecolor=color, alpha=0.7, label=name))

        for yp, r_val, sig in zip(y_pos, r_vals, sigs):
            if not sig:
                continue
            if r_val >= 0:
                ax.text(r_val + 0.003, yp, "*", va="center", ha="left", fontsize=9, color="black")
            else:
                ax.text(r_val - 0.003, yp, "*", va="center", ha="right", fontsize=9, color="black")

    ax.set_yticks(np.arange(n_p))
    ax.set_yticklabels(pairs_ordered, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r (vs baseline residuals)")
    ax.set_title(title)
    ax.legend(handles=handles, loc="lower right")
    plt.tight_layout()
    plt.show()

    return {(e["op"], e["pair"]): e["sig"] for e in all_entries}


def plot_significant_pairwise_scatters(scan_configs, sig_dict, df: pd.DataFrame, title: str) -> None:
    """Plot pairwise operation values vs residuals for significant results."""
    x = df.drop("MedHouseVal", axis=1).values
    y_arr = df["MedHouseVal"].values
    residuals = y_arr - LinearRegression().fit(x, y_arr).predict(x)

    sig_items = [
        (name, pair_suffix, color, op_fn)
        for _, name, _, color, op_fn in scan_configs
        for (op_name, pair_suffix), is_sig in sig_dict.items()
        if is_sig and op_name == name
    ]

    if not sig_items:
        print("No statistically significant pairwise features found.")
        return

    n_plots = len(sig_items)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=10)

    for i, (name, pair_suffix, color, op_fn) in enumerate(sig_items):
        ax = axes[i // n_cols, i % n_cols]

        col_a, col_b = next(
            (a, b)
            for a in df.columns
            for b in df.columns
            if pair_suffix == f"{a}_{b}" and a in df.columns and b in df.columns
        )

        try:
            new_vals = np.asarray(op_fn(df[col_a], df[col_b]))
        except Exception:
            ax.set_visible(False)
            continue

        ax.scatter(new_vals, residuals, color=color, s=4, alpha=0.2)
        r, p = pearsonr(new_vals, residuals)
        ax.set_xlabel(f"{name}({col_a}, {col_b})", fontsize=7)
        ax.set_ylabel("residuals", fontsize=8)
        ax.set_title(f"r = {r:+.3f}  p = {p:.3f}", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n_plots, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    plt.tight_layout()
    plt.show()
