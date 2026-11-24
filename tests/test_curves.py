import numpy as np
import pytest

from nbs_runtime.curves import evaluate_nodes

X, S = [600, 700, 1000, 1200], [0.25, 0.5, 0.75, 0.9]


def test_nodes_are_exact():
    assert evaluate_nodes(X, S, X) == pytest.approx(S)


def test_linear_between_nodes():
    assert evaluate_nodes(X, S, [650, 850])[0] == pytest.approx(0.375)
    assert evaluate_nodes(X, S, [650, 850])[1] == pytest.approx(0.625)


def test_outside_is_unknown_by_default():
    out = evaluate_nodes(X, S, [599.9, 1200.1, 4000])
    assert np.isnan(out).all()


def test_clamp_repeats_end_nodes():
    assert evaluate_nodes(X, S, [100, 4000], outside="clamp") == pytest.approx([0.25, 0.9])


def test_missing_input_is_unknown_not_zero():
    out = evaluate_nodes(X, S, [np.nan, 800], outside="clamp")
    assert np.isnan(out[0]) and out[1] > 0


def test_works_on_2d_rasters():
    out = evaluate_nodes(X, S, np.array([[600, 700], [1000, 5000]]))
    assert out.shape == (2, 2) and np.isnan(out[1, 1])
