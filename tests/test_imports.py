import featurely as fl


def test_exports_are_importable():
    assert callable(fl.add_pipeline_step)
    assert callable(fl.clip_outliers)
    assert callable(fl.compute_vif)
    assert callable(fl.run_pairwise_scan)
    assert callable(fl.run_candidate_scan)
    assert callable(fl.plot_candidate_scan)
    assert callable(fl.haversine_distance)
    assert callable(fl.compute_city_distances)
    assert callable(fl.compute_geohash_cells)
    assert callable(fl.compute_rotated_coordinates)
    assert callable(fl.compute_bin_aggregates)
    assert callable(fl.plot_kmeans_selection)
    assert callable(fl.compute_kmeans_features)
    assert callable(fl.compute_spatial_smoothed)
    assert callable(fl.make_polynomial_features)
    assert callable(fl.plot_pca_variance)
    assert callable(fl.scan_pca_components)
    assert callable(fl.plot_pca_component_scan)
