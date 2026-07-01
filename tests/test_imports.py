from featurely import (
    add_pipeline_step,
    clip_outliers,
    compute_vif,
    run_pairwise_scan,
)


def test_exports_are_importable():
    assert callable(add_pipeline_step)
    assert callable(clip_outliers)
    assert callable(compute_vif)
    assert callable(run_pairwise_scan)
