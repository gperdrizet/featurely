"""Unit tests for outliers, transforms, scans, pipeline, decomposition, diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import featurely as fl


class TestOutliers:
    def test_clip_outliers_bounds(self, housing_df):
        result = fl.clip_outliers(housing_df, ["MedInc"], threshold=1.5)
        q1, q3 = housing_df["MedInc"].quantile([0.25, 0.75])
        iqr = q3 - q1
        assert result["MedInc"].max() <= q3 + 1.5 * iqr + 1e-9
        assert result["MedInc"].min() >= q1 - 1.5 * iqr - 1e-9

    def test_clip_does_not_mutate_input(self, housing_df):
        before = housing_df.copy()
        fl.clip_outliers(housing_df, ["MedInc"])
        assert housing_df.equals(before)

    def test_impute_outliers_leaves_no_nans(self, housing_df):
        result = fl.impute_outliers_with_knn(housing_df, ["MedInc", "AveOccup"], n_neighbors=3)
        assert result.isna().sum().sum() == 0


class TestTransforms:
    def test_standard_scale(self, housing_df):
        result = fl.apply_standard_scale(housing_df, ["MedInc", "HouseAge"])
        np.testing.assert_allclose(result["MedInc"].mean(), 0.0, atol=1e-9)
        np.testing.assert_allclose(result["MedInc"].std(ddof=0), 1.0, atol=1e-9)

    def test_log1p_shifts_negatives(self, housing_df):
        df = housing_df.copy()
        df["neg"] = df["MedInc"] - df["MedInc"].max()  # min is negative
        result = fl.apply_log1p(df, ["neg"])
        assert np.isfinite(result["neg"]).all()


class TestScans:
    def test_candidate_scan_finds_planted_signal(self, housing_df, rng):
        # A candidate correlated with the residual noise component should rank
        # far above pure-noise candidates.
        x = housing_df.drop("MedHouseVal", axis=1)
        from sklearn.linear_model import LinearRegression

        residuals = housing_df["MedHouseVal"] - LinearRegression().fit(x, housing_df["MedHouseVal"]).predict(x)

        candidates = pd.DataFrame(
            {
                "signal": residuals + rng.normal(0, residuals.std() * 0.5, len(residuals)),
                "noise": rng.normal(0, 1, len(residuals)),
            },
            index=housing_df.index,
        )

        results = fl.run_candidate_scan(housing_df, candidates, target="MedHouseVal")
        sig = fl.plot_candidate_scan(results, title="test")

        assert sig["signal"] is True
        assert abs(results["signal"][0]) > abs(results["noise"][0])

    def test_candidate_scan_skips_constant(self, housing_df):
        candidates = pd.DataFrame({"const": np.ones(len(housing_df))}, index=housing_df.index)
        results = fl.run_candidate_scan(housing_df, candidates, target="MedHouseVal")
        assert results == {}


class TestPipeline:
    def test_add_pipeline_step_upserts_by_stage(self, housing_df, tmp_path):
        path = tmp_path / "results.pkl"
        x = housing_df.drop("MedHouseVal", axis=1)
        y = housing_df["MedHouseVal"]

        first = fl.add_pipeline_step(None, "raw", x, y, cv=3, results_path=path)
        assert len(first) == 1

        # Re-running the same stage must replace, not append.
        second = fl.add_pipeline_step(None, "raw", x, y, cv=3, results_path=path)
        assert len(second) == 1

        third = fl.add_pipeline_step(None, "+ stage2", x, y, cv=3, results_path=path)
        assert list(third["stage"]) == ["raw", "+ stage2"]

    def test_pct_vs_raw_computed(self, housing_df, tmp_path):
        path = tmp_path / "results.pkl"
        x = housing_df.drop("MedHouseVal", axis=1)
        y = housing_df["MedHouseVal"]
        fl.add_pipeline_step(None, "raw", x, y, cv=3, results_path=path)
        result = fl.add_pipeline_step(None, "+ same", x, y, cv=3, results_path=path)
        raw_pct = result.loc[result["stage"] == "raw", "pct_vs_raw"].iloc[0]
        assert raw_pct == 0.0


class TestDecomposition:
    def test_polynomial_feature_names(self, housing_df):
        poly = fl.make_polynomial_features(housing_df, ["MedInc", "HouseAge"], degree=2)
        assert "MedInc_x_HouseAge" in poly.columns
        assert "MedInc_pow2" in poly.columns
        assert poly.shape == (len(housing_df), 5)

    def test_polynomial_values(self, housing_df):
        poly = fl.make_polynomial_features(housing_df, ["MedInc"], degree=2)
        np.testing.assert_allclose(poly["MedInc_pow2"], housing_df["MedInc"] ** 2)

    def test_scan_pca_components_rows(self, housing_df):
        poly = fl.make_polynomial_features(housing_df, ["MedInc", "HouseAge", "AveRooms"], degree=2)
        scan = fl.scan_pca_components(poly, housing_df["MedHouseVal"], [1, 2, 3], cv=3)
        assert list(scan["n_components"]) == [1, 2, 3]
        assert scan["mean_r2"].notna().all()


class TestDiagnostics:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_vif_flags_collinear_column(self, housing_df):
        # Perfect collinearity makes statsmodels divide by zero internally;
        # that RuntimeWarning is the expected mechanism behind the inf VIF.
        df = housing_df.copy()
        df["MedInc_copy"] = df["MedInc"] * 2.0  # perfectly collinear
        result = fl.compute_vif(df, ["MedInc", "MedInc_copy", "HouseAge"])
        top = result.iloc[0]
        assert top["feature"] in ("MedInc", "MedInc_copy")
        assert top["VIF"] > 100 or np.isinf(top["VIF"])
