import featurely as fl


def test_exports_are_importable():
    assert callable(fl.add_pipeline_step)
    assert callable(fl.clip_outliers)
    assert callable(fl.compute_vif)
    assert callable(fl.run_pairwise_scan)
