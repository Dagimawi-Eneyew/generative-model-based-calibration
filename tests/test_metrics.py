"""Tests for the evaluation metrics.

Each metric is checked against a value computed by hand or against a property it
must satisfy by construction, so an accidental change to a formula fails loudly
rather than quietly shifting a published number.
"""

from __future__ import annotations

import numpy as np
import pytest

from genphysical.evaluation import metrics


# ---------------------------------------------------------------------------
# Point estimate accuracy (Eq. 9)
# ---------------------------------------------------------------------------
def test_rmse_matches_hand_computation():
    actual = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    predicted = np.array([[1.0, 11.0], [4.0, 18.0], [3.0, 30.0]])
    # variable 0: residuals 0, -2, 0 -> sqrt(4/3)
    # variable 1: residuals -1, 2, 0 -> sqrt(5/3)
    expected = np.array([np.sqrt(4 / 3), np.sqrt(5 / 3)])
    np.testing.assert_allclose(metrics.rmse(actual, predicted), expected)


def test_rmse_is_zero_for_perfect_predictions():
    actual = np.random.default_rng(0).normal(size=(50, 9))
    np.testing.assert_allclose(metrics.rmse(actual, actual), np.zeros(9), atol=1e-12)


def test_point_estimate_statistics():
    # A posterior whose samples are 0..99 for every point and variable.
    posterior = np.tile(np.arange(100.0), (4, 3, 1))
    np.testing.assert_allclose(metrics.point_estimate(posterior, "mean"), 49.5)
    np.testing.assert_allclose(metrics.point_estimate(posterior, "median"), 49.5)


def test_point_estimate_rejects_unknown_statistic():
    with pytest.raises(ValueError, match="statistic must be"):
        metrics.point_estimate(np.zeros((2, 3, 4)), "maximum")


# ---------------------------------------------------------------------------
# Probability calibration (Eqs. 10-11)
# ---------------------------------------------------------------------------
def _perfectly_calibrated(n_points: int = 4000, n_samples: int = 400, seed: int = 0):
    """Ground truth drawn from the very distribution the posterior represents.

    Under this construction the predictive CDF at the truth is uniform, so the
    calibration curve must sit on the diagonal.
    """
    generator = np.random.default_rng(seed)
    mean = generator.normal(size=(n_points, 1))
    actual = mean + generator.normal(size=(n_points, 1))
    posterior = mean[:, :, None] + generator.normal(size=(n_points, 1, n_samples))
    return actual, posterior


def test_calibration_curve_is_diagonal_when_calibrated():
    actual, posterior = _perfectly_calibrated()
    levels = np.linspace(0.05, 0.95, 19)

    for definition in ("cdf", "interval"):
        curve = metrics.calibration_curve(actual, posterior, levels, definition)
        # Monte-Carlo error over 4000 points is a couple of percent.
        np.testing.assert_allclose(curve[0], levels, atol=0.04)
        error = metrics.calibration_error(curve, levels, reduction="sum")
        assert error[0] < 0.02


def test_calibration_error_penalises_overconfidence():
    """A posterior that is too narrow must score worse than a calibrated one."""
    actual, posterior = _perfectly_calibrated()
    levels = np.linspace(0.05, 0.95, 19)

    calibrated = metrics.calibration_error(
        metrics.calibration_curve(actual, posterior, levels), levels
    )
    # Shrink the posterior towards its mean: same centre, far too confident.
    centre = posterior.mean(axis=2, keepdims=True)
    overconfident_posterior = centre + 0.2 * (posterior - centre)
    overconfident = metrics.calibration_error(
        metrics.calibration_curve(actual, overconfident_posterior, levels), levels
    )
    assert overconfident[0] > 10 * calibrated[0]


def test_calibration_error_sum_and_mean_agree():
    curve = np.array([[0.1, 0.3, 0.6]])
    levels = [0.2, 0.4, 0.5]
    total = metrics.calibration_error(curve, levels, reduction="sum")
    average = metrics.calibration_error(curve, levels, reduction="mean")
    np.testing.assert_allclose(total, average * len(levels))


# ---------------------------------------------------------------------------
# Sharpness (Eq. 12)
# ---------------------------------------------------------------------------
def test_sharpness_recovers_known_interval_width():
    """For a standard normal posterior the 90% interval is 2 * 1.645 wide."""
    generator = np.random.default_rng(1)
    posterior = generator.normal(size=(600, 1, 8000))
    widths = metrics.sharpness(posterior, [0.9])
    np.testing.assert_allclose(widths[0, 0], 2 * 1.6449, rtol=0.02)


def test_sharpness_increases_with_coverage():
    generator = np.random.default_rng(2)
    posterior = generator.normal(size=(100, 3, 500))
    widths = metrics.sharpness(posterior, [0.5, 0.8, 0.95])
    assert np.all(np.diff(widths, axis=1) > 0)


def test_sharpness_rejects_invalid_coverage():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        metrics.sharpness(np.zeros((2, 1, 10)), [1.0])


# ---------------------------------------------------------------------------
# CRPS (Eq. 13)
# ---------------------------------------------------------------------------
def test_crps_ensemble_reduces_to_absolute_error_for_a_point_forecast():
    """With no spread, CRPS collapses to the mean absolute error."""
    actual = np.array([[2.0], [5.0]])
    posterior = np.array([[[3.0] * 50], [[5.0] * 50]])
    np.testing.assert_allclose(
        metrics.crps(actual, posterior, estimator="ensemble"), [0.5], atol=1e-10
    )


def test_crps_ensemble_matches_the_gaussian_closed_form():
    """A large Gaussian sample must reproduce the analytic Gaussian CRPS."""
    generator = np.random.default_rng(3)
    actual = np.array([[0.4]])
    posterior = generator.normal(loc=0.0, scale=1.0, size=(1, 1, 200_000))
    ensemble = metrics.crps(actual, posterior, estimator="ensemble")
    analytic = metrics.crps(actual, posterior, estimator="gaussian")
    np.testing.assert_allclose(ensemble, analytic, rtol=0.02)


def test_crps_rewards_the_better_forecast():
    generator = np.random.default_rng(4)
    actual = np.zeros((500, 1))
    accurate = generator.normal(0.0, 1.0, size=(500, 1, 400))
    biased = accurate + 3.0
    assert metrics.crps(actual, accurate)[0] < metrics.crps(actual, biased)[0]


# ---------------------------------------------------------------------------
# Model calibration accuracy (Eq. 14)
# ---------------------------------------------------------------------------
def test_cvrmse_matches_hand_computation():
    measured = np.array([100.0, 200.0, 300.0, 400.0])
    simulated = np.array([110.0, 190.0, 300.0, 400.0])
    # Sum of squared residuals = 100 + 100 = 200; n - p = 3; mean = 250.
    expected = np.sqrt(200.0 / 3.0) / 250.0 * 100.0
    assert metrics.cvrmse(measured, simulated, p=1) == pytest.approx(expected)


def test_cvrmse_is_zero_for_a_perfect_match():
    series = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.cvrmse(series, series) == pytest.approx(0.0)


def test_cvrmse_is_scale_invariant():
    """Reporting in joules or kilowatt-hours must not change the percentage."""
    generator = np.random.default_rng(5)
    measured = generator.uniform(10, 20, size=100)
    simulated = measured + generator.normal(0, 1, size=100)
    assert metrics.cvrmse(measured, simulated) == pytest.approx(
        metrics.cvrmse(measured * 3.6e6, simulated * 3.6e6)
    )


def test_cvrmse_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="shape"):
        metrics.cvrmse(np.zeros(5), np.zeros(6))


def test_nmbe_sign_follows_the_bias_direction():
    measured = np.full(100, 10.0)
    # Simulated below measured -> positive bias term.
    assert metrics.nmbe(measured, measured - 1.0) > 0
    assert metrics.nmbe(measured, measured + 1.0) < 0
