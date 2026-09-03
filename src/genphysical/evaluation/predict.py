"""Solving the calibration problem with a trained calibrator (Algorithm 1).

Given observations ``y_o`` from the building, the calibrator produces a full
posterior over the unobserved model inputs ``x`` by drawing ``n`` samples from
the standard normal base distribution and pushing each through the inverse flow,
conditioned on ``y_o``:

    z_i ~ N(0, I)                        Algorithm 1, line 2
    x_i = f^-1(z_i ; C(y_o))             Algorithm 1, line 6

Because there is no search and no evaluation of the physics model, this is fast
enough for continuous calibration - Section 5.11 reports 0.043 s per calibration
problem.  :func:`measure_inference_time` reproduces that measurement.

:func:`predict_posterior` batches many observations into each inverse pass,
which keeps the 8760-row test set fast to evaluate.
:func:`measure_inference_time` deliberately keeps the one-observation-at-a-time
path, because that is what the reported latency means.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf

from ..constants import N_UNOBSERVED_INPUTS
from ..models.flow import ConditionalRealNVP
from ..utils.logging_utils import ProgressLogger, get_logger
from .metrics import point_estimate

logger = get_logger(__name__)


@dataclass
class PosteriorPredictions:
    """Posterior samples and point estimates for one experiment.

    Attributes
    ----------
    posterior:
        ``(n_points, 9, n_samples)`` predicted distributions, in physical units
        (people and kW).
    point:
        ``(n_points, 9)`` point estimates, collapsed with the configured
        statistic.
    actual:
        ``(n_points, 9)`` ground-truth values, in the same physical units.
    """

    posterior: np.ndarray
    point: np.ndarray
    actual: np.ndarray

    def __len__(self) -> int:
        return len(self.actual)


def _condition(model: ConditionalRealNVP, observations: tf.Tensor) -> tf.Tensor:
    """Map observations to the model's conditioning vector.

    For the baseline cINN this is the identity; for DECI-Net it runs the
    autoencoder's encoder, the only half of it needed at inference time.
    """
    return model.condition_from_observations(observations)


def predict_posterior(
    model: ConditionalRealNVP,
    observations: np.ndarray,
    actual_inputs: Optional[np.ndarray],
    input_scaler,
    n_samples: int = 1000,
    point_statistic: str = "mean",
    batch_size: int = 64,
    seed: int = 0,
) -> PosteriorPredictions:
    """Estimate the posterior over the unobserved model inputs.

    Parameters
    ----------
    model:
        A trained calibrator.
    observations:
        ``(n_points, 14)`` standardised observations - the ``y_o`` the building
        reports, already corrupted according to the experiment being run.
    actual_inputs:
        ``(n_points, 9)`` standardised ground truth, returned alongside the
        predictions for convenience.  ``None`` when calibrating a real building,
        where the truth is by definition unavailable.
    input_scaler:
        Fitted scaler for the 9 inputs, used to return results in physical units.
    n_samples:
        Posterior samples per observation (Algorithm 1's ``n``; 1000 in the paper).
    point_statistic:
        ``"mean"``, ``"median"`` or ``"mode"`` - see
        :func:`genphysical.evaluation.metrics.point_estimate`.
    batch_size:
        Observations processed per inverse pass.  Peak memory is roughly
        ``batch_size * n_samples * 9`` floats, so 64 x 1000 is a few MB.
    seed:
        Seed for the latent draw.

    Returns
    -------
    PosteriorPredictions
        Posterior samples, point estimates and ground truth, in physical units.
    """
    observations = np.asarray(observations, dtype=np.float32)
    n_points = len(observations)

    posterior = np.empty((n_points, N_UNOBSERVED_INPUTS, n_samples), dtype=np.float32)
    generator = np.random.default_rng(seed)
    progress = ProgressLogger(logger, n_points, "Sampling posteriors")

    for start in range(0, n_points, batch_size):
        stop = min(start + batch_size, n_points)
        chunk = observations[start:stop]
        chunk_size = stop - start

        # One conditioning vector per observation, repeated for its n samples.
        condition = _condition(model, tf.convert_to_tensor(chunk))
        condition = tf.repeat(condition, repeats=n_samples, axis=0)

        # Independent latent draws for every (observation, sample) pair.
        latent = tf.convert_to_tensor(
            generator.standard_normal(
                (chunk_size * n_samples, N_UNOBSERVED_INPUTS)
            ).astype(np.float32)
        )

        estimated = model.inverse(latent, condition).numpy()
        # (chunk * samples, 9) -> (chunk, 9, samples)
        posterior[start:stop] = estimated.reshape(
            chunk_size, n_samples, N_UNOBSERVED_INPUTS
        ).transpose(0, 2, 1)

        progress.update(stop - 1)

    # Back to people and kW.  A separate scaler for the 9 inputs makes this a
    # direct inverse transform.
    flat = posterior.transpose(0, 2, 1).reshape(-1, N_UNOBSERVED_INPUTS)
    posterior = (
        input_scaler.inverse_transform(flat)
        .reshape(n_points, n_samples, N_UNOBSERVED_INPUTS)
        .transpose(0, 2, 1)
    )

    point = point_estimate(posterior, statistic=point_statistic)
    actual = (
        input_scaler.inverse_transform(np.asarray(actual_inputs, dtype=float))
        if actual_inputs is not None
        else np.full_like(point, np.nan)
    )

    logger.info(
        "Estimated %d posteriors of %d samples each (point statistic: %s)",
        n_points,
        n_samples,
        point_statistic,
    )
    return PosteriorPredictions(posterior=posterior, point=point, actual=actual)


def measure_inference_time(
    model: ConditionalRealNVP,
    observations: np.ndarray,
    n_samples: int = 1000,
    n_timed: int = 500,
    n_warmup: int = 50,
) -> dict:
    """Time one calibration problem, as reported in Section 5.11.

    "The average inference time for computing the calibration solution per
    observation by the calibrator models (cINN and DECI-Net) is 0.038 second for
    the standard cINN model and 0.043 second for the DECI-Net model."

    Deliberately measured one observation at a time - the quantity of interest is
    the latency of solving a *single* calibration problem during continuous
    operation, not the throughput of a batched offline run.  The first calls
    trigger TensorFlow graph tracing and are discarded as warm-up.

    Parameters
    ----------
    model:
        A trained calibrator.
    observations:
        ``(n_points, 14)`` standardised observations to draw the timed calls from.
    n_samples:
        Posterior samples per calibration problem.
    n_timed, n_warmup:
        Timed calls and discarded warm-up calls.

    Returns
    -------
    dict
        Mean, standard deviation, median and percentiles of the per-observation
        time in seconds, plus the settings used.
    """
    observations = np.asarray(observations, dtype=np.float32)
    n_available = len(observations)
    n_warmup = min(n_warmup, n_available)
    n_timed = min(n_timed, n_available)

    def solve_one(index: int) -> None:
        """One complete calibration problem: encode, draw, invert."""
        single = tf.convert_to_tensor(observations[index : index + 1])
        condition = tf.repeat(_condition(model, single), repeats=n_samples, axis=0)
        latent = model.sample_latent(n_samples)
        model.inverse(latent, condition)

    for index in range(n_warmup):
        solve_one(index)

    durations = np.empty(n_timed)
    for position in range(n_timed):
        started = time.perf_counter()
        solve_one(position % n_available)
        durations[position] = time.perf_counter() - started

    result = {
        "n_timed": int(n_timed),
        "n_warmup": int(n_warmup),
        "n_posterior_samples": int(n_samples),
        "mean_seconds": float(durations.mean()),
        "std_seconds": float(durations.std()),
        "median_seconds": float(np.median(durations)),
        "p05_seconds": float(np.percentile(durations, 5)),
        "p95_seconds": float(np.percentile(durations, 95)),
    }
    logger.info(
        "Inference time per calibration problem: %.4f s (median %.4f s) over %d calls",
        result["mean_seconds"],
        result["median_seconds"],
        n_timed,
    )
    return result
