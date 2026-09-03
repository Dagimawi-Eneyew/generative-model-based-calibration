"""Virtual to Physical Observations Approximation (VPOA).

Implements Section 4.3.3 / 5.3.3 of the paper.  Simulated observations are
perfect; measurements from a real building are not.  VPOA closes part of that
gap by producing two corrupted copies of the simulated observations alongside
the original:

    version 0  ``clean``          the raw simulated observations
    version 1  ``noisy``          + independent Gaussian noise
    version 2  ``noisy_missing``  randomly masked sensors, with noise on the
                                  readings that survive

"The total data size tripled after the augmentation steps, resulting in data
points three times the initial training and testing data."

Only the 14 observed model outputs ``y_o`` are corrupted.  The 9 unobserved
model inputs ``x`` are the estimation target and are never touched.

Two ordering details matter and are easy to get wrong:

* **Noise is added after standardisation.**  Section 5.3.3: "the noise was
  applied after standardizing the masked data combined with the original,
  ensuring that the noise scale remained proportional to the original dataset."
  A single standard-normal draw scaled by ``noise_factor`` is therefore a 10 %
  perturbation *relative to each variable's own spread*, which is what makes one
  noise factor meaningful across temperatures, humidities and facility meters at
  wildly different magnitudes.
* **Masking is applied before standardisation.**  A failed sensor reports
  nothing; multiplying the raw value by zero is what represents that.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..constants import N_OBSERVED_OUTPUTS
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

#: The three dataset versions, in the order they are concatenated.  Experiment
#: k of Table 3 evaluates on version k-1.
VERSIONS: List[str] = ["clean", "noisy", "noisy_missing"]


def random_missing_mask(
    n_rows: int,
    n_observations: int = N_OBSERVED_OUTPUTS,
    min_masked: int = 1,
    max_masked: int = 5,
    protected: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw a random sensor-failure mask.

    Each row independently loses between ``min_masked`` and ``max_masked``
    observations, chosen uniformly at random among the unprotected ones.  With
    the defaults this drops up to 5 of 14 variables, the "up to 35 %" of Section
    5.3.3.

    Parameters
    ----------
    n_rows:
        Number of observation rows to generate a mask for.
    n_observations:
        Width of the observation vector (14).
    min_masked, max_masked:
        Inclusive bounds on the number of failed sensors per row.
    protected:
        Indices that must never be masked.  Empty by default; the original
        study protected ``Electricity:Facility`` (index 12)
    seed:
        Seed for the draw.

    Returns
    -------
    numpy.ndarray
        ``(n_rows, n_observations)`` array of 0.0 (failed) and 1.0 (working),
        ready to multiply elementwise into the observation block.
    """
    protected_set = set(protected or ())
    maskable = np.array(
        [index for index in range(n_observations) if index not in protected_set]
    )
    if max_masked > len(maskable):
        raise ValueError(
            f"max_masked={max_masked} exceeds the {len(maskable)} maskable "
            f"observations left after protecting {sorted(protected_set)}."
        )

    generator = np.random.default_rng(seed)
    mask = np.ones((n_rows, n_observations), dtype=np.float64)

    # How many sensors fail in each row (inclusive upper bound).
    counts = generator.integers(min_masked, max_masked + 1, size=n_rows)

    # Vectorised "choose `counts[i]` distinct maskable indices per row":
    # rank a random matrix per row and zero the lowest-ranked `counts[i]`.
    ranks = generator.random((n_rows, len(maskable))).argsort(axis=1)
    drop = ranks < counts[:, None]
    mask[:, maskable] = np.where(drop, 0.0, 1.0)

    logger.debug(
        "Drew missing mask for %d rows: %.1f%% of observations dropped on average",
        n_rows,
        100.0 * (1.0 - mask.mean()),
    )
    return mask


def gaussian_noise(
    n_rows: int,
    n_observations: int = N_OBSERVED_OUTPUTS,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw standard-normal noise for the observation block.

    The draw is standard normal; scaling by ``noise_factor`` happens in
    :func:`apply_noise`, after standardisation, so a single factor means the
    same relative perturbation for every variable.
    """
    generator = np.random.default_rng(seed)
    return generator.standard_normal((n_rows, n_observations))


def apply_noise(
    observations: np.ndarray,
    noise: np.ndarray,
    noise_factor: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Add scaled noise to a *standardised* observation block.

    Parameters
    ----------
    observations:
        ``(n_rows, 14)`` standardised observations.
    noise:
        Standard-normal draw of the same shape.
    noise_factor:
        Scale applied to the noise (0.1 = the paper's "10% noising factor").
    mask:
        Optional sensor-failure mask.  When given, noise is applied only to the
        sensors that are still working: a failed sensor reports nothing at all,
        not a noisy zero.

    Returns
    -------
    numpy.ndarray
        A new array; ``observations`` is not modified.
    """
    scaled = noise_factor * noise
    if mask is not None:
        scaled = scaled * mask
    return observations + scaled


@dataclass
class AugmentedDataset:
    """The three VPOA versions of one dataset, stacked.

    Attributes
    ----------
    inputs:
        ``(3 * n, 9)`` standardised unobserved model inputs ``x``.  Identical in
        all three blocks - corruption affects only the observations.
    observations:
        ``(3 * n, 14)`` standardised, corrupted observations ``y_o``.
    clean_observations:
        ``(3 * n, 14)`` standardised, *uncorrupted* observations.  This is the
        denoising autoencoder's reconstruction target in Eq. 8; the baseline
        cINN ignores it.
    version_index:
        ``(3 * n,)`` integer array giving each row's version, so
        :meth:`version_slice` can recover a single experiment's rows.
    n_per_version:
        Rows in each version, i.e. the size of the underlying simulated set.
    """

    inputs: np.ndarray
    observations: np.ndarray
    clean_observations: np.ndarray
    version_index: np.ndarray
    n_per_version: int

    def version_slice(self, version: str) -> slice:
        """Row slice of one version, e.g. ``"noisy_missing"`` for Experiment 3."""
        if version not in VERSIONS:
            raise KeyError(f"Unknown version {version!r}; expected one of {VERSIONS}.")
        position = VERSIONS.index(version)
        return slice(position * self.n_per_version, (position + 1) * self.n_per_version)

    def __len__(self) -> int:
        return len(self.inputs)


def build_augmented_dataset(
    inputs: np.ndarray,
    observations: np.ndarray,
    noise_factor: float,
    min_masked: int,
    max_masked: int,
    protected_observations: Sequence[int],
    noise_seed: int,
    mask_seed: int,
    input_scaler,
    observation_scaler,
) -> AugmentedDataset:
    """Produce the three VPOA versions from one simulated dataset.


    Parameters
    ----------
    inputs:
        ``(n, 9)`` unobserved model inputs, unstandardised.
    observations:
        ``(n, 14)`` clean simulated observations, unstandardised.
    noise_factor, min_masked, max_masked, protected_observations:
        VPOA settings from ``configs/data_generation.yaml``.
    noise_seed, mask_seed:
        Seeds for the two random draws.
    input_scaler, observation_scaler:
        Fitted :class:`sklearn.preprocessing.StandardScaler` instances for the
        9 input and 14 observation columns.  Keeping them separate means a
        predicted ``x`` is inverted with ``input_scaler.inverse_transform``
        directly.

    Returns
    -------
    AugmentedDataset
        Three stacked versions, 3 x ``n`` rows in total.
    """
    n_rows = len(inputs)
    if len(observations) != n_rows:
        raise ValueError(
            f"inputs has {n_rows} rows but observations has {len(observations)}."
        )

    # -- step 1: sensor failures, applied to the raw values -------------------
    mask = random_missing_mask(
        n_rows=n_rows,
        n_observations=observations.shape[1],
        min_masked=min_masked,
        max_masked=max_masked,
        protected=protected_observations,
        seed=mask_seed,
    )
    masked_raw = observations * mask

    # -- step 2: standardise with the clean-data statistics -------------------
    inputs_scaled = input_scaler.transform(inputs)
    clean_scaled = observation_scaler.transform(observations)
    masked_scaled = observation_scaler.transform(masked_raw)

    # -- step 3: noise, on the standardised values ----------------------------
    noise = gaussian_noise(n_rows, observations.shape[1], seed=noise_seed)
    noisy_scaled = apply_noise(clean_scaled, noise, noise_factor)
    noisy_missing_scaled = apply_noise(masked_scaled, noise, noise_factor, mask=mask)

    corrupted = np.concatenate(
        [clean_scaled, noisy_scaled, noisy_missing_scaled], axis=0
    )
    # The reconstruction target is the clean observations, repeated once per
    # version so it lines up row-for-row with the corrupted block.
    clean_repeated = np.concatenate([clean_scaled] * len(VERSIONS), axis=0)
    inputs_repeated = np.concatenate([inputs_scaled] * len(VERSIONS), axis=0)
    version_index = np.repeat(np.arange(len(VERSIONS)), n_rows)

    logger.info(
        "VPOA: %d simulated rows -> %d augmented rows (%s); "
        "noise factor %.2f, %d-%d of %d sensors masked per row",
        n_rows,
        len(corrupted),
        ", ".join(VERSIONS),
        noise_factor,
        min_masked,
        max_masked,
        observations.shape[1],
    )

    return AugmentedDataset(
        inputs=inputs_repeated,
        observations=corrupted,
        clean_observations=clean_repeated,
        version_index=version_index,
        n_per_version=n_rows,
    )
