# featurely

Solution repository for the lesson 16 feature engineering challenge, now refactored into a simple Python package layout.

## Project lineage

- Current repository: https://github.com/gperdrizet/featurely
- Origin: Fullstack Academy lesson 16 feature engineering assignment from https://github.com/gperdrizet/fullstack-2605

## Package layout

```text
featurely/
├── data/
├── notebooks/
│   ├── original-assignment.ipynb
│   ├── 01-EDA.ipynb
│   ├── 02-outlier-cleaning.ipynb
│   ├── 03-feature-transformations.ipynb
│   ├── 04-interaction-features.ipynb
│   ├── 05-p_censoring.ipynb
│   ├── complete-solution.ipynb
│   ├── config.py
│   └── data/
├── src/
│   └── featurely/
│       ├── __init__.py
│       ├── pipeline.py
│       ├── eda.py
│       ├── outliers.py
│       ├── transforms.py
│       ├── scans.py
│       └── diagnostics.py
├── tests/
│   └── test_imports.py
├── AGENTS.md
├── README.md
└── pyproject.toml
```

## What was refactored from the notebook

- Pipeline evaluation helpers:
	- `add_pipeline_step`
	- `plot_pipeline_steps`
- EDA helpers:
	- `plot_feature_distributions`
	- `get_feature_correlations`
	- `plot_feature_correlations`
	- `plot_features_vs_label`
- Outlier handling:
	- `impute_outliers_with_knn`
	- `clip_outliers`
	- `transform_outliers`
- Feature transforms:
	- `apply_standard_scale`
	- `apply_log1p`
	- `apply_sqrt`
	- `apply_yeo_johnson`
	- `apply_quantile_normal`
- Scan/selection utilities:
	- `run_per_feature_scan`
	- `plot_combined_per_feature_scan`
	- `plot_significant_transform_scatters`
	- `run_pairwise_scan`
	- `plot_combined_pairwise_scan`
	- `plot_significant_pairwise_scatters`
- Diagnostics:
	- `compute_vif`

## Install locally

```bash
pip install -e .
```

## Import in notebook

```python
from featurely import (
		add_pipeline_step,
		plot_pipeline_steps,
		plot_feature_distributions,
		get_feature_correlations,
		plot_feature_correlations,
		plot_features_vs_label,
		impute_outliers_with_knn,
		clip_outliers,
		transform_outliers,
		apply_standard_scale,
		apply_log1p,
		apply_sqrt,
		apply_yeo_johnson,
		apply_quantile_normal,
		run_per_feature_scan,
		plot_combined_per_feature_scan,
		plot_significant_transform_scatters,
		run_pairwise_scan,
		plot_combined_pairwise_scan,
		plot_significant_pairwise_scatters,
		compute_vif,
)
```
