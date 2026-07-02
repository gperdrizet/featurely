# featurely

Reusable feature engineering utilities for tabular machine learning with pandas and scikit-learn.

featurely provides function-based helpers for the screen-then-commit feature engineering loop: build candidate features, test whether they explain variance your current model misses, and keep only the winners.

- **Pipeline evaluation**: cross-validated stage-over-stage comparison with persisted, rerun-safe results and progressive box plots.
- **Candidate screening**: residual correlation scans with Benjamini-Hochberg false discovery rate correction, for individual features and grouped feature sets.
- **Feature builders**: outlier handling, monotonic transforms, geographic encodings (haversine distances, geohash cells, rotated coordinates), quantile bin aggregates, k-means cluster memberships, Gaussian kernel spatial smoothing, and polynomial expansion with PCA component selection.
- **Diagnostics and EDA**: distribution plots, pairwise correlation analysis, and variance inflation factors.

All helpers accept a pandas DataFrame, take explicit column names, and return transformed copies without mutating input.

## Install

```bash
pip install featurely
```

Requires Python 3.10 or newer.

## Quick start

```python
import pandas as pd
import featurely as fl

df = pd.read_csv("my_data.csv")
target = "price"
features = [c for c in df.columns if c != target]

# Establish a baseline
results = fl.add_pipeline_step(None, "raw", df[features], df[target])

# Clean outliers and measure the effect
df_clean = fl.clip_outliers(df, features, threshold=2.25)

results = fl.add_pipeline_step(
    results, "+ cleaned", df_clean[features], df_clean[target]
)

fl.plot_pipeline_steps(results, title="Effect of outlier clipping")

# Build candidates and screen them against baseline residuals
candidates = fl.compute_bin_aggregates(df_clean, "latitude", ["income"], n_bins=10)
scan = fl.run_candidate_scan(df_clean, candidates, target=target)
significant = fl.plot_candidate_scan(scan, title="Candidate scan")

keep = [name for name, is_sig in significant.items() if is_sig]
df_clean = pd.concat([df_clean, candidates[keep]], axis=1)
```

## Documentation

Full API reference, getting-started guide, and a complete worked example on the California housing dataset:

- Documentation: [gperdrizet.github.io/featurely](https://gperdrizet.github.io/featurely/)
- Source and example notebooks: [github.com/gperdrizet/featurely](https://github.com/gperdrizet/featurely)

## License

MIT
