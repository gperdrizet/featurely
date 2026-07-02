# featurely

Reusable feature engineering utilities for tabular machine learning with pandas and scikit-learn.

featurely grew out of a feature engineering exercise on the California housing dataset: improve linear regression performance using only feature engineering. The utilities that emerged are general: they accept any pandas DataFrame, take explicit column names, and return transformed copies without mutating input.

## What it provides

- **Pipeline evaluation**: cross-validated stage-over-stage comparison with persisted, rerun-safe results and progressive box plots.
- **Candidate screening**: residual correlation scans with Benjamini-Hochberg false discovery rate correction, for individual features and grouped feature sets.
- **Feature builders**: outlier handling, monotonic transforms, geographic encodings (haversine distances, geohash cells, rotated coordinates), quantile bin aggregates, k-means cluster memberships, Gaussian kernel spatial smoothing, and polynomial expansion with PCA component selection.
- **Diagnostics and EDA**: distribution plots, pairwise correlation analysis, and variance inflation factors.

## Install

```bash
pip install featurely
```

## Quick example

```python
import pandas as pd
import featurely as fl

df = pd.read_csv("my_data.csv")
features = [c for c in df.columns if c != "target"]

# Clean outliers and evaluate the change
df_clean = fl.clip_outliers(df, features, threshold=2.25)

results = fl.add_pipeline_step(None, "raw", df[features], df["target"])
results = fl.add_pipeline_step(results, "+ cleaned", df_clean[features], df_clean["target"])
fl.plot_pipeline_steps(results, title="Effect of outlier clipping")
```

## Worked example

The [example notebooks](https://github.com/gperdrizet/featurely/tree/main/example_notebooks/fsa-feature-engineering-challenge) walk a complete ten-stage feature engineering pipeline on the California housing dataset, improving linear regression cross-validation R² by 39 percent over the raw features, with a statistical gate justifying every added feature.
