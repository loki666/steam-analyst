"""Small statistics helpers behind the trend and sentiment tools."""

import math

import pytest

from steam_game_analyst.tools import stats


def test_two_proportion_test_known_value():
    r = stats.two_proportions(40, 100, 60, 100)
    assert r["diff"] == pytest.approx(0.2)
    assert r["z"] == pytest.approx(2.828, abs=1e-3)  # pooled
    assert r["p_value"] == pytest.approx(0.00468, abs=1e-4)


def test_two_proportion_test_handles_degenerate_input():
    assert stats.two_proportions(0, 0, 5, 10)["p_value"] is None
    assert stats.two_proportions(10, 10, 10, 10)["p_value"] == 1.0


def test_kish_effective_n():
    assert stats.effective_n([1, 1, 1, 1]) == pytest.approx(4)
    assert stats.effective_n([1, 0, 0, 0]) == pytest.approx(1)
    assert stats.effective_n([]) == 0


def test_weighted_slope_recovers_a_linear_trend():
    xs = list(range(12))
    ys = [0.5 + 0.01 * x for x in xs]
    r = stats.weighted_slope(xs, ys, [100] * 12)
    assert r["slope"] == pytest.approx(0.01)
    assert r["p_value"] < 1e-6


def test_weighted_slope_flat_noise_is_not_significant():
    ys = [0.6, 0.62, 0.59, 0.61, 0.6, 0.6, 0.61, 0.59, 0.6, 0.62]
    r = stats.weighted_slope(list(range(10)), ys, [50] * 10)
    assert r["p_value"] > 0.05


def test_change_point_finds_the_step():
    n = [100] * 12
    pos = [50] * 6 + [80] * 6
    cp = stats.binomial_change_point(pos, n)
    assert cp["index"] == 6
    assert cp["before"] == pytest.approx(0.5) and cp["after"] == pytest.approx(0.8)
    assert cp["p_value"] < 1e-6


def test_change_point_needs_enough_points():
    assert stats.binomial_change_point([1, 2], [3, 4]) is None


def test_normal_sf_matches_known_value():
    assert stats.normal_two_sided_p(1.96) == pytest.approx(0.05, abs=1e-3)
    assert math.isclose(stats.normal_two_sided_p(0), 1.0)
