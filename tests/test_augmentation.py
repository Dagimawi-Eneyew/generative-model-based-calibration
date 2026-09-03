"""Tests for the VPOA augmentation stage (Section 5.3.3).

These pin the behaviour the paper specifies: exactly three versions, a bounded
number of masked sensors, noise applied after standardisation, and a
reconstruction target that stays clean.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from genphysical.config import AugmentationConfig, ConfigError
from genphysical.constants import N_OBSERVED_OUTPUTS, N_UNOBSERVED_INPUTS
from genphysical.data.augmentation import (
    VERSIONS,
    build_augmented_dataset,
    random_missing_mask,
)


@pytest.fixture
def simulated():
    """A small stand-in for merged simulation output."""
    generator = np.random.default_rng(0)
    n_rows = 200
    inputs = generator.uniform(0, 10, size=(n_rows, N_UNOBSERVED_INPUTS))
    # Deliberately mismatched magnitudes, as in the real data: temperatures in
    # the tens, humidities in the tens of percent, facility meters ~1e6.
    observations = np.column_stack(
        [generator.normal(20 + 5 * i, 2, n_rows) for i in range(12)]
        + [generator.normal(3.6e6, 1e5, n_rows), generator.normal(2.0e6, 8e4, n_rows)]
    )
    return inputs, observations


def _scalers(inputs, observations):
    return StandardScaler().fit(inputs), StandardScaler().fit(observations)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
def test_mask_drops_between_min_and_max_sensors_per_row():
    mask = random_missing_mask(
        n_rows=2000, min_masked=1, max_masked=5, seed=0
    )
    dropped = (mask == 0).sum(axis=1)
    assert dropped.min() == 1
    assert dropped.max() == 5
    assert set(np.unique(mask)) <= {0.0, 1.0}


def test_mask_covers_the_full_paper_range():
    """Section 5.3.3 specifies up to 5 of 14 variables (~35%)."""
    mask = random_missing_mask(n_rows=5000, min_masked=1, max_masked=5, seed=1)
    assert mask.shape[1] == N_OBSERVED_OUTPUTS
    # Uniform over 1..5 dropped sensors -> mean fraction dropped = 3/14.
    assert (1.0 - mask.mean()) == pytest.approx(3.0 / 14.0, abs=0.02)


def test_protected_observations_are_never_masked():
    mask = random_missing_mask(
        n_rows=500, min_masked=1, max_masked=5, protected=[12], seed=2
    )
    assert np.all(mask[:, 12] == 1.0)


def test_mask_rejects_impossible_request():
    with pytest.raises(ValueError, match="exceeds"):
        random_missing_mask(n_rows=10, n_observations=4, min_masked=1, max_masked=4,
                            protected=[0, 1])


# ---------------------------------------------------------------------------
# Full augmentation
# ---------------------------------------------------------------------------
def test_produces_exactly_three_equal_versions(simulated):
    """
    """
    inputs, observations = simulated
    input_scaler, observation_scaler = _scalers(inputs, observations)

    dataset = build_augmented_dataset(
        inputs=inputs,
        observations=observations,
        noise_factor=0.1,
        min_masked=1,
        max_masked=5,
        protected_observations=[],
        noise_seed=0,
        mask_seed=1,
        input_scaler=input_scaler,
        observation_scaler=observation_scaler,
    )

    assert len(VERSIONS) == 3
    assert len(dataset) == 3 * len(inputs)
    assert dataset.n_per_version == len(inputs)
    for version in VERSIONS:
        assert len(range(*dataset.version_slice(version).indices(len(dataset)))) == len(
            inputs
        )


def test_clean_version_is_untouched(simulated):
    inputs, observations = simulated
    input_scaler, observation_scaler = _scalers(inputs, observations)

    dataset = build_augmented_dataset(
        inputs=inputs,
        observations=observations,
        noise_factor=0.1,
        min_masked=1,
        max_masked=5,
        protected_observations=[],
        noise_seed=0,
        mask_seed=1,
        input_scaler=input_scaler,
        observation_scaler=observation_scaler,
    )

    clean = dataset.observations[dataset.version_slice("clean")]
    np.testing.assert_allclose(clean, observation_scaler.transform(observations))


def test_reconstruction_target_is_clean_in_every_version(simulated):
    """The denoising autoencoder must always be shown the uncorrupted signal.

    Its input is corrupted; its target is not. If the two ever coincided, the
    MSE term of Eq. 8 would be reconstructing noise.
    """
    inputs, observations = simulated
    input_scaler, observation_scaler = _scalers(inputs, observations)

    dataset = build_augmented_dataset(
        inputs=inputs,
        observations=observations,
        noise_factor=0.1,
        min_masked=1,
        max_masked=5,
        protected_observations=[],
        noise_seed=0,
        mask_seed=1,
        input_scaler=input_scaler,
        observation_scaler=observation_scaler,
    )

    expected = observation_scaler.transform(observations)
    for version in VERSIONS:
        rows = dataset.version_slice(version)
        np.testing.assert_allclose(dataset.clean_observations[rows], expected)

    # And the corrupted versions really are different from their target.
    for version in ("noisy", "noisy_missing"):
        rows = dataset.version_slice(version)
        assert not np.allclose(dataset.observations[rows], expected)


def test_inputs_are_identical_across_versions(simulated):
    """Corruption applies to observations only; the estimation target is fixed."""
    inputs, observations = simulated
    input_scaler, observation_scaler = _scalers(inputs, observations)

    dataset = build_augmented_dataset(
        inputs=inputs,
        observations=observations,
        noise_factor=0.1,
        min_masked=1,
        max_masked=5,
        protected_observations=[],
        noise_seed=0,
        mask_seed=1,
        input_scaler=input_scaler,
        observation_scaler=observation_scaler,
    )

    reference = dataset.inputs[dataset.version_slice("clean")]
    for version in VERSIONS:
        np.testing.assert_allclose(dataset.inputs[dataset.version_slice(version)], reference)


def test_noise_scale_is_relative_after_standardisation(simulated):
    """A 10 % noise factor must mean 10 % of each variable's own spread.

    This is why Section 5.3.3 applies noise *after* standardisation. Applying it
    to raw values would leave the facility meters (~1e6 J) essentially untouched
    while swamping the zone temperatures.
    """
    inputs, observations = simulated
    input_scaler, observation_scaler = _scalers(inputs, observations)

    dataset = build_augmented_dataset(
        inputs=inputs,
        observations=observations,
        noise_factor=0.1,
        min_masked=1,
        max_masked=5,
        protected_observations=[],
        noise_seed=0,
        mask_seed=1,
        input_scaler=input_scaler,
        observation_scaler=observation_scaler,
    )

    clean = dataset.observations[dataset.version_slice("clean")]
    noisy = dataset.observations[dataset.version_slice("noisy")]
    perturbation = noisy - clean

    # Per-column perturbation magnitude ~ 0.1 in standardised units, for every
    # column, despite raw scales spanning six orders of magnitude.
    np.testing.assert_allclose(perturbation.std(axis=0), 0.1, atol=0.02)


def test_missing_version_has_zeroed_sensors_before_noise(simulated):
    """A failed sensor contributes nothing - not a noisy zero."""
    inputs, observations = simulated
    input_scaler, observation_scaler = _scalers(inputs, observations)

    dataset = build_augmented_dataset(
        inputs=inputs,
        observations=observations,
        noise_factor=0.1,
        min_masked=1,
        max_masked=5,
        protected_observations=[],
        noise_seed=0,
        mask_seed=1,
        input_scaler=input_scaler,
        observation_scaler=observation_scaler,
    )

    noisy_missing = dataset.observations[dataset.version_slice("noisy_missing")]
    # A masked entry equals the standardised value of a raw zero, exactly, with
    # no noise added on top.
    zero_in_scaled_space = observation_scaler.transform(
        np.zeros((1, N_OBSERVED_OUTPUTS))
    )[0]
    matches_zero = np.isclose(noisy_missing, zero_in_scaled_space[None, :], atol=1e-9)
    dropped_per_row = matches_zero.sum(axis=1)
    assert dropped_per_row.min() >= 1
    assert dropped_per_row.max() <= 5


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------
def test_augmentation_config_rejects_impossible_mask_bounds():
    with pytest.raises(ConfigError, match="min_masked"):
        AugmentationConfig(
            versions=VERSIONS,
            noise_factor=0.1,
            min_masked=6,
            max_masked=3,
            protected_observations=[],
            train_noise_seed=0,
            train_mask_seed=0,
            test_noise_seed=0,
            test_mask_seed=0,
        )


def test_augmentation_config_rejects_out_of_range_protection():
    with pytest.raises(ConfigError, match="outside"):
        AugmentationConfig(
            versions=VERSIONS,
            noise_factor=0.1,
            min_masked=1,
            max_masked=5,
            protected_observations=[99],
            train_noise_seed=0,
            train_mask_seed=0,
            test_noise_seed=0,
            test_mask_seed=0,
        )
