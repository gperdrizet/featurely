"""Unit tests for geo, aggregate, cluster, and smoothing feature builders."""

from __future__ import annotations

import numpy as np

import featurely as fl


class TestGeo:
    def test_haversine_known_distance(self):
        # LA to SF is roughly 559 km great-circle.
        d = fl.haversine_distance(34.05, -118.24, 37.77, -122.42)
        assert 540 < float(d) < 580

    def test_haversine_zero_for_identical_points(self):
        assert float(fl.haversine_distance(37.0, -120.0, 37.0, -120.0)) == 0.0

    def test_encode_geohash_known_vector(self):
        # Published geohash for downtown San Francisco.
        assert fl.encode_geohash(37.77, -122.42, precision=5) == "9q8yy"

    def test_city_distances_columns_and_nearest(self, housing_df):
        result = fl.compute_city_distances(housing_df)
        assert "dist_nearest_city" in result.columns
        dist_cols = [c for c in result.columns if c != "dist_nearest_city"]
        assert len(dist_cols) == len(fl.CA_CITY_COORDS)
        # Nearest-city distance can never exceed any individual city distance.
        assert (result["dist_nearest_city"] <= result[dist_cols].min(axis=1) + 1e-9).all()

    def test_geohash_cells_one_hot(self, housing_df):
        cells = fl.compute_geohash_cells(housing_df, precision=3, min_cell_count=5)
        assert cells.shape[0] == len(housing_df)
        # Every row belongs to exactly one cell.
        np.testing.assert_allclose(cells.sum(axis=1).values, 1.0)

    def test_rotated_coordinates_values(self, housing_df):
        rot = fl.compute_rotated_coordinates(housing_df)
        np.testing.assert_allclose(rot["coord_sum"], housing_df["Latitude"] + housing_df["Longitude"])
        np.testing.assert_allclose(rot["coord_diff"], housing_df["Latitude"] - housing_df["Longitude"])


class TestAggregate:
    def test_bin_aggregates_shape_and_names(self, housing_df):
        result = fl.compute_bin_aggregates(housing_df, "Latitude", ["MedInc", "HouseAge"], n_bins=5, stats=("mean",))
        assert result.shape == (len(housing_df), 2)
        assert "Latitudebin5_mean_MedInc" in result.columns

    def test_bin_aggregates_are_group_constant(self, housing_df):
        # Rows sharing a bin must share the bin statistic.
        result = fl.compute_bin_aggregates(housing_df, "Latitude", ["MedInc"], n_bins=4)
        col = result["Latitudebin4_mean_MedInc"]
        assert col.nunique() <= 4

    def test_input_not_mutated(self, housing_df):
        before = housing_df.copy()
        fl.compute_bin_aggregates(housing_df, "Latitude", ["MedInc"], n_bins=4)
        assert housing_df.equals(before)


class TestCluster:
    def test_kmeans_features_shape(self, housing_df):
        result = fl.compute_kmeans_features(housing_df, ["Latitude", "Longitude"], k=4, prefix="geo")
        one_hot = [c for c in result.columns if c != "geo_centroid_dist"]
        assert len(one_hot) == 4
        np.testing.assert_allclose(result[one_hot].sum(axis=1).values, 1.0)
        assert (result["geo_centroid_dist"] >= 0).all()

    def test_kmeans_deterministic(self, housing_df):
        a = fl.compute_kmeans_features(housing_df, ["Latitude", "Longitude"], k=3, prefix="g")
        b = fl.compute_kmeans_features(housing_df, ["Latitude", "Longitude"], k=3, prefix="g")
        assert a.equals(b)


class TestSmoothing:
    def test_smoothed_columns_and_range(self, housing_df):
        result = fl.compute_spatial_smoothed(housing_df, ["MedInc"], n_neighbors=10)
        assert list(result.columns) == ["smooth_MedInc"]
        # A weighted average must stay inside the original value range.
        assert result["smooth_MedInc"].min() >= housing_df["MedInc"].min() - 1e-9
        assert result["smooth_MedInc"].max() <= housing_df["MedInc"].max() + 1e-9

    def test_smoothing_reduces_variance(self, housing_df):
        result = fl.compute_spatial_smoothed(housing_df, ["MedInc"], n_neighbors=25)
        assert result["smooth_MedInc"].var() < housing_df["MedInc"].var()
