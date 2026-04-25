import numpy as np

from layertrak.visualization.plot_trak_results import (
    _normalize_scores,
    _top_indices_asc,
    _top_indices_desc,
)


def test_normalize_scores_accepts_both_orientations():
    summary = {"num_targets": 2, "train_set_size": 3}

    target_train = np.arange(6).reshape(2, 3)
    train_target = target_train.T

    np.testing.assert_array_equal(_normalize_scores(target_train, summary), target_train)
    np.testing.assert_array_equal(_normalize_scores(train_target, summary), target_train)


def test_top_indices_handle_zero_k():
    values = np.array([3.0, 1.0, 2.0])

    assert _top_indices_desc(values, 0).tolist() == []
    assert _top_indices_asc(values, 0).tolist() == []
