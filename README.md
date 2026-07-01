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


## Install locally

```bash
pip install -e .
```


## Import in notebook

```python
import featurely as fl

fl.add_pipeline_step(...)
fl.plot_pipeline_steps(...)
fl.plot_feature_distributions(...)
fl.get_feature_correlations(...)
fl.plot_feature_correlations(...)
fl.plot_features_vs_label(...)
fl.impute_outliers_with_knn(...)
fl.clip_outliers(...)
fl.transform_outliers(...)
fl.apply_standard_scale(...)
fl.apply_log1p(...)
fl.apply_sqrt(...)
fl.apply_yeo_johnson(...)
fl.apply_quantile_normal(...)
fl.run_per_feature_scan(...)
fl.plot_combined_per_feature_scan(...)
fl.plot_significant_transform_scatters(...)
fl.run_pairwise_scan(...)
fl.plot_combined_pairwise_scan(...)
fl.plot_significant_pairwise_scatters(...)
fl.compute_vif(...)
```
