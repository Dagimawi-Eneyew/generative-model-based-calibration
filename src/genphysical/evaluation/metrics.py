"""Every metric reported in the paper.

Section 4.5 defines four evaluation aspects; this module covers the first three
(the fourth, inference time, lives in :mod:`.predict`):

    point estimate accuracy      :func:`rmse`                     Eq. 9
    probability calibration      :func:`calibration_curve`,       Eqs. 10-11
                                 :func:`calibration_error`
    sharpness                    :func:`sharpness`                Eq. 12
    CRPS                         :func:`crps`                     Eq. 13
    model calibration accuracy   :func:`cvrmse`                   Eq. 14

Array conventions
-----------------
``actual``
    ``(n_points, n_variables)`` ground-truth values.
``posterior``
    ``(n_points, n_variables, n_samples)`` - the full predicted distribution for
    every test point and every unobserved model input.  This is what Algorithm 1
    produces and what makes the probabilistic metrics possible.

All functions are vectorised over variables and return one value per variable
unless stated otherwise.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Point estimate accuracy (Section 5.5.1)
# ---------------------------------------------------------------------------
def point_estimate(posterior: np.ndarray, statistic: str = "mean") -> np.ndarray:
    """Collapse a posterior to one value per test point and variable.

    Parameters
    ----------
    posterior:
        ``(n_points, n_variables, n_samples)``.
    statistic:
        ``"mean"`` - the paper's choice for the reported RMSE and CVRMSE
        (Section 5.5.1: "The mean of the predicted distributions was used as
        point estimates").
        ``"median"`` - the 50th percentile.
        ``"mode"`` - the maximum-density point, estimated with a Gaussian kernel
        density estimate.  This is what Algorithm 1 line 9 describes.

    Returns
    -------
    numpy.ndarray
        ``(n_points, n_variables)``.
    """
    posterior = np.asarray(posterior, dtype=float)
    if statistic == "mean":
        return posterior.mean(axis=2)
    if statistic == "median":
        return np.median(posterior, axis=2)
    if statistic == "mode":
        return _kde_mode(posterior)
    raise ValueError(
        f"statistic must be 'mean', 'median' or 'mode'; got {statistic!r}."
    )


def _kde_mode(posterior: np.ndarray, grid_size: int = 128) -> np.ndarray:
    """Maximum-density point of each predicted distribution.

    Algorithm 1 selects "the maximum density point from the estimated
    distribution of x".  A Gaussian KDE is evaluated on a grid spanning each
    sample's range and the arg-max is returned.  Degenerate distributions (all
    samples identical) fall back to the mean.
    """
    n_points, n_variables, _ = posterior.shape
    modes = np.empty((n_points, n_variables))
    for point in range(n_points):
        for variable in range(n_variables):
            samples = posterior[point, variable, :]
            spread = samples.max() - samples.min()
            if not np.isfinite(spread) or spread <= 0:
                modes[point, variable] = samples.mean()
                continue
            try:
                density = stats.gaussian_kde(samples)
            except np.linalg.LinAlgError:  # singular covariance
                modes[point, variable] = samples.mean()
                continue
            grid = np.linspace(samples.min(), samples.max(), grid_size)
            modes[point, variable] = grid[int(np.argmax(density(grid)))]
    return modes


def rmse(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Root mean squared error per variable (Eq. 9).

        RMSE = sqrt( (1/n) Σ (x_i - x̂_i)² )

    Parameters
    ----------
    actual, predicted:
        ``(n_points, n_variables)``.

    Returns
    -------
    numpy.ndarray
        ``(n_variables,)`` - the values tabulated in Table 4.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return np.sqrt(np.mean((actual - predicted) ** 2, axis=0))


def mae(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Mean absolute error per variable. Not in the paper; useful context."""
    return np.mean(np.abs(np.asarray(actual) - np.asarray(predicted)), axis=0)


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Coefficient of determination per variable. Not in the paper."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = np.sum((actual - predicted) ** 2, axis=0)
    total = np.sum((actual - actual.mean(axis=0)) ** 2, axis=0)
    return 1.0 - np.divide(
        residual, total, out=np.full_like(residual, np.nan), where=total > 0
    )


# ---------------------------------------------------------------------------
# Probability calibration (Section 5.6.1, Eqs. 10-11)
# ---------------------------------------------------------------------------
def calibration_curve(
    actual: np.ndarray,
    posterior: np.ndarray,
    confidence_levels: Sequence[float],
    definition: str = "cdf",
) -> np.ndarray:
    """

    ``"cdf"`` - Eq. 10 as written:

        p̂_j = |{ i : F_i(y_i) <= p_j }| / N

    where ``F_i`` is the predictive CDF for test point ``i`` evaluated at its
    ground truth.  For a perfectly calibrated model ``F_i(y_i)`` is uniform on
    [0, 1], so ``p̂_j = p_j`` and the curve is the diagonal.

    ``"interval"`` - the coverage of the *centred* prediction interval:

        p̂_j = |{ i : q_i(0.5 - p_j/2) <= y_i <= q_i(0.5 + p_j/2) }| / N

    Both are diagonal under perfect calibration, but they respond differently to
    a biased posterior: the interval version is insensitive to which side of the
    median the truth falls on.

    Parameters
    ----------
    actual:
        ``(n_points, n_variables)``.
    posterior:
        ``(n_points, n_variables, n_samples)``.
    confidence_levels:
        The nominal levels ``p_j``.
    definition:
        ``"cdf"`` or ``"interval"``.

    Returns
    -------
    numpy.ndarray
        ``(n_variables, n_levels)`` empirical frequencies - the y-values of the
        calibration plots in Figs. 10-12.
    """
    actual = np.asarray(actual, dtype=float)
    posterior = np.asarray(posterior, dtype=float)
    levels = np.asarray(confidence_levels, dtype=float)
    _, n_variables, n_samples = posterior.shape

    if definition == "cdf":
        # Predictive CDF at the ground truth: the fraction of posterior samples
        # at or below the true value.  Vectorised over points and variables.
        cdf_at_truth = np.mean(posterior <= actual[:, :, None], axis=2)
        # p_hat_j = fraction of test points whose CDF value is <= p_j.
        return np.array(
            [
                [np.mean(cdf_at_truth[:, variable] <= level) for level in levels]
                for variable in range(n_variables)
            ]
        )

    if definition == "interval":
        empirical = np.empty((n_variables, len(levels)))
        for level_index, level in enumerate(levels):
            lower = np.quantile(posterior, 0.5 - level / 2.0, axis=2)
            upper = np.quantile(posterior, 0.5 + level / 2.0, axis=2)
            covered = (actual >= lower) & (actual <= upper)
            empirical[:, level_index] = covered.mean(axis=0)
        return empirical

    raise ValueError(f"definition must be 'cdf' or 'interval'; got {definition!r}.")


def calibration_error(
    empirical_frequencies: np.ndarray,
    confidence_levels: Sequence[float],
    reduction: str = "sum",
) -> np.ndarray:
    """Calibration error from a calibration curve (Eq. 11).

        CE = Σ_j (p_j - p̂_j)²

    Parameters
    ----------
    empirical_frequencies:
        ``(n_variables, n_levels)`` from :func:`calibration_curve`.
    confidence_levels:
        The same ``p_j`` used to build the curve.
    reduction:
        ``"sum"`` - Eq. 11 as written, and what is reported.
        ``"mean"`` - the average squared deviation.  It equals the sum divided
        by the number of levels, so its value depends on how many levels were
        chosen.

    Returns
    -------
    numpy.ndarray
        ``(n_variables,)``.
    """
    empirical_frequencies = np.asarray(empirical_frequencies, dtype=float)
    levels = np.asarray(confidence_levels, dtype=float)
    squared = (levels[None, :] - empirical_frequencies) ** 2
    if reduction == "sum":
        return squared.sum(axis=1)
    if reduction == "mean":
        return squared.mean(axis=1)
    raise ValueError(f"reduction must be 'sum' or 'mean'; got {reduction!r}.")


# ---------------------------------------------------------------------------
# Sharpness (Section 5.6.2, Eq. 12)
# ---------------------------------------------------------------------------
def sharpness(
    posterior: np.ndarray, coverages: Sequence[float]
) -> np.ndarray:
    """Mean width of the central prediction interval at each coverage (Eq. 12).

        S_α = (1/N) Σ_i ( q̂_{1-α/2} - q̂_{α/2} )

    A calibrated model with narrower intervals is more informative, so sharpness
    is only meaningful alongside :func:`calibration_error` - a model can be
    arbitrarily sharp by being overconfident.  These are the curves of
    Figs. 13-15.

    Parameters
    ----------
    posterior:
        ``(n_points, n_variables, n_samples)``.
    coverages:
        Nominal coverage rates ``1 - α``, e.g. 0.9 for the 90 % interval.

    Returns
    -------
    numpy.ndarray
        ``(n_variables, n_coverages)`` mean interval widths, in the physical
        units of each variable (people or kW).
    """
    posterior = np.asarray(posterior, dtype=float)
    coverages = np.asarray(coverages, dtype=float)
    if np.any((coverages <= 0) | (coverages >= 1)):
        raise ValueError("Coverage rates must lie strictly between 0 and 1.")

    widths = np.empty((posterior.shape[1], len(coverages)))
    for index, coverage in enumerate(coverages):
        alpha = 1.0 - coverage
        lower = np.quantile(posterior, alpha / 2.0, axis=2)
        upper = np.quantile(posterior, 1.0 - alpha / 2.0, axis=2)
        widths[:, index] = (upper - lower).mean(axis=0)
    return widths


# ---------------------------------------------------------------------------
# CRPS (Section 5.6.3, Eq. 13)
# ---------------------------------------------------------------------------
def crps(
    actual: np.ndarray,
    posterior: np.ndarray,
    estimator: str = "ensemble",
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """Continuous Ranked Probability Score per variable (Eq. 13).

        CRPS = (1/N) Σ_i ∫ ( F̂_pred(x) - F_xact(x) )² dx

    CRPS is a strictly proper scoring rule that penalises miscalibration and
    lack of sharpness together, so it summarises both in one number (Table 5).
    Lower is better, and it reduces to the mean absolute error for a
    deterministic forecast.

    Parameters
    ----------
    actual:
        ``(n_points, n_variables)``.
    posterior:
        ``(n_points, n_variables, n_samples)``.
    estimator:
        ``"ensemble"`` - the empirical CRPS of the sample, using the identity

            CRPS = E|X - y| - ½ E|X - X'|

        evaluated exactly with the order-statistic formula.  This is Eq. 13 as
        written and makes no distributional assumption.

       
    max_samples:
        Subsample the posterior to this many draws before computing the
        ensemble CRPS.  The exact estimator sorts the samples, so cost grows as
        ``n_samples log n_samples``; capping it bounds the runtime on large test
        sets.  ``None`` uses every sample.
    seed:
        Seed for that subsampling.

    Returns
    -------
    numpy.ndarray
        ``(n_variables,)`` mean CRPS.
    """
    actual = np.asarray(actual, dtype=float)
    posterior = np.asarray(posterior, dtype=float)

    if estimator == "gaussian":
        mean = posterior.mean(axis=2)
        std = posterior.std(axis=2)
        return _crps_gaussian(actual, mean, std).mean(axis=0)

    if estimator != "ensemble":
        raise ValueError(
            f"estimator must be 'ensemble' or 'gaussian'; got {estimator!r}."
        )

    if max_samples is not None and posterior.shape[2] > max_samples:
        generator = np.random.default_rng(seed)
        chosen = generator.choice(posterior.shape[2], size=max_samples, replace=False)
        posterior = posterior[:, :, chosen]

    return _crps_ensemble(actual, posterior).mean(axis=0)


def _crps_gaussian(
    actual: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Closed-form CRPS of ``N(mean, std²)`` evaluated at ``actual``.

        CRPS = σ [ ω (2 Φ(ω) - 1) + 2 φ(ω) - 1/√π ],   ω = (y - μ)/σ
    """
    std = np.where(std > 0, std, np.finfo(float).eps)
    standardised = (actual - mean) / std
    return std * (
        standardised * (2.0 * stats.norm.cdf(standardised) - 1.0)
        + 2.0 * stats.norm.pdf(standardised)
        - 1.0 / np.sqrt(np.pi)
    )


def _crps_ensemble(actual: np.ndarray, posterior: np.ndarray) -> np.ndarray:
    """Exact empirical CRPS of a finite sample.

    Uses the order-statistic identity

        CRPS = (1/m) Σ|x_k - y| - (1/m²) Σ_k (2k - m - 1) x_(k)

    where ``x_(k)`` are the sorted samples (1-indexed).  The second term is the
    exact value of ``½ E|X - X'|`` for the empirical distribution, so no
    ``O(m²)`` pairwise difference matrix is ever formed.

    Returns
    -------
    numpy.ndarray
        ``(n_points, n_variables)`` per-point CRPS.
    """
    n_samples = posterior.shape[2]
    absolute_error = np.mean(np.abs(posterior - actual[:, :, None]), axis=2)

    ordered = np.sort(posterior, axis=2)
    ranks = np.arange(1, n_samples + 1, dtype=float)
    weights = 2.0 * ranks - n_samples - 1.0
    spread = np.sum(ordered * weights, axis=2) / (n_samples**2)

    return absolute_error - spread


# ---------------------------------------------------------------------------
# Model calibration accuracy (Section 5.7.1, Eq. 14)
# ---------------------------------------------------------------------------
def cvrmse(measured: np.ndarray, simulated: np.ndarray, p: int = 1) -> float:
    """Coefficient of Variation of the RMSE, in percent (Eq. 14).

        CVRMSE = (1/m̄) sqrt( Σ (m_i - s_i)² / (n - p) ) × 100

    The standard measure of building energy model calibration accuracy.  Here
    ``m`` is the physical measurement and ``s`` the output of the model after it
    has been re-evaluated with the estimated calibration solution.  ASHRAE
    Guideline 14 and FEMP set the hourly acceptance threshold at 30 %.

    Parameters
    ----------
    measured, simulated:
        One-dimensional, equal-length series of hourly values.
    p:
        Degrees of freedom subtracted from ``n``; Robertson, Polly & Collis
        (2013) suggest 1, which is what the paper uses.

    Returns
    -------
    float
        CVRMSE as a percentage.
    """
    measured = np.asarray(measured, dtype=float).ravel()
    simulated = np.asarray(simulated, dtype=float).ravel()
    if measured.shape != simulated.shape:
        raise ValueError(
            f"measured has shape {measured.shape} but simulated has {simulated.shape}."
        )
    n_points = measured.size
    if n_points <= p:
        raise ValueError(f"Need more than p={p} data points; got {n_points}.")

    mean_measured = measured.mean()
    if mean_measured == 0:
        raise ValueError("Mean of the measured series is zero; CVRMSE is undefined.")

    root_mean_square = np.sqrt(np.sum((measured - simulated) ** 2) / (n_points - p))
    return float(root_mean_square / mean_measured * 100.0)


def nmbe(measured: np.ndarray, simulated: np.ndarray, p: int = 1) -> float:
    """Normalised Mean Bias Error, in percent.

        NMBE = Σ (m_i - s_i) / ((n - p) m̄) × 100

    Not reported in the paper, but the natural companion to CVRMSE: it separates
    systematic bias from random scatter, and ASHRAE Guideline 14 pairs the two
    (hourly threshold ±10 %).
    """
    measured = np.asarray(measured, dtype=float).ravel()
    simulated = np.asarray(simulated, dtype=float).ravel()
    n_points = measured.size
    mean_measured = measured.mean()
    if mean_measured == 0:
        raise ValueError("Mean of the measured series is zero; NMBE is undefined.")
    return float(
        np.sum(measured - simulated) / ((n_points - p) * mean_measured) * 100.0
    )
