"""Assembling the model-ready arrays and persisting them with their scalers.

Bridges the merged simulation CSVs and the calibrator models:

    merged CSV  ->  standardise  ->  VPOA augment  ->  .npz + scalers

Every array produced here uses the canonical column layout of
:mod:`genphysical.constants`::

    columns  0 ..  8   unobserved model inputs   x    (standardised)
    columns  9 .. 22   observed model outputs    y_o  (standardised, corrupted)
    columns 23 .. 36   clean copy of y_o              (DECI-Net target only)

The baseline cINN consumes the first 23 columns; DECI-Net consumes all 37.

The scalers are fitted on the **clean training data only** and then reused for
the corrupted training blocks and for the whole test set, so that test
statistics never influence the transform and the denoising autoencoder's input
and target share one standardisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..config import DataGenerationConfig
from ..constants import (
    ALL_MODEL_COLUMNS,
    N_OBSERVED_OUTPUTS,
    N_UNOBSERVED_INPUTS,
    OBSERVED_OUTPUT_COLUMNS,
    UNOBSERVED_INPUT_COLUMNS,
)
from ..utils.logging_utils import get_logger
from .augmentation import VERSIONS, AugmentedDataset, build_augmented_dataset

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loading the merged simulation output
# ---------------------------------------------------------------------------
def load_merged_csv(
    path: str | Path, n_rows: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a merged simulation CSV into ``(inputs, observations)``.

    Parameters
    ----------
    path:
        A CSV written by
        :func:`genphysical.energyplus.postprocess.merge_simulation_outputs`.
    n_rows:
        Optional cap, useful for a quick smoke run over a slice of the data.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(n, 9)`` unobserved model inputs and ``(n, 14)`` observed outputs, both
        unstandardised.
    """
    frame = pd.read_csv(path, nrows=n_rows)
    missing = [column for column in ALL_MODEL_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(
            f"{path} is missing {len(missing)} canonical column(s): {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )
    inputs = frame.loc[:, UNOBSERVED_INPUT_COLUMNS].to_numpy(dtype=np.float64)
    observations = frame.loc[:, OBSERVED_OUTPUT_COLUMNS].to_numpy(dtype=np.float64)
    logger.info("Loaded %s: %d rows", path, len(inputs))
    return inputs, observations


# ---------------------------------------------------------------------------
# Prepared datasets
# ---------------------------------------------------------------------------
@dataclass
class PreparedData:
    """Everything the training and evaluation stages need.

    Attributes
    ----------
    train, test:
        The augmented training and test sets.
    input_scaler, observation_scaler:
        The fitted transforms, kept so predictions can be returned in physical
        units and so a saved model can be applied to fresh data.
    """

    train: AugmentedDataset
    test: AugmentedDataset
    input_scaler: StandardScaler
    observation_scaler: StandardScaler

    def inverse_transform_inputs(self, scaled: np.ndarray) -> np.ndarray:
        """Map standardised model inputs back to people / kW.

        Accepts ``(n, 9)`` or, for a whole posterior, ``(n, n_samples, 9)``.
        """
        scaled = np.asarray(scaled)
        if scaled.shape[-1] != N_UNOBSERVED_INPUTS:
            raise ValueError(
                f"Expected a trailing dimension of {N_UNOBSERVED_INPUTS}, got "
                f"{scaled.shape}."
            )
        flat = scaled.reshape(-1, N_UNOBSERVED_INPUTS)
        return self.input_scaler.inverse_transform(flat).reshape(scaled.shape)


def build_model_matrix(
    dataset: AugmentedDataset, model: str, version: Optional[str] = None
) -> np.ndarray:
    """Lay an :class:`AugmentedDataset` out as the matrix a model consumes.

    Parameters
    ----------
    dataset:
        Augmented training or test data.
    model:
        ``"cinn"`` for the 23-column layout, ``"decinet"`` for the 37-column
        layout that appends the clean reconstruction target.
    version:
        Restrict to one VPOA version, e.g. ``"noisy_missing"`` to evaluate
        Experiment 3.  ``None`` returns all three, concatenated.

    Returns
    -------
    numpy.ndarray
        ``(n, 23)`` or ``(n, 37)``, float32 - the dtype TensorFlow wants.
    """
    if model not in ("cinn", "decinet"):
        raise ValueError(f"model must be 'cinn' or 'decinet'; got {model!r}.")

    rows = dataset.version_slice(version) if version is not None else slice(None)
    blocks = [dataset.inputs[rows], dataset.observations[rows]]
    if model == "decinet":
        blocks.append(dataset.clean_observations[rows])
    return np.concatenate(blocks, axis=1).astype(np.float32)


def prepare_datasets(
    train_csv: str | Path,
    test_csv: str | Path,
    config: DataGenerationConfig,
    max_train_rows: Optional[int] = None,
    max_test_rows: Optional[int] = None,
) -> PreparedData:
    """Run the full preparation pipeline for the training and test sets.

    Parameters
    ----------
    train_csv, test_csv:
        Merged simulation output for the 400-run training batch and for the
        8760-hour test simulation.
    config:
        Parsed ``configs/data_generation.yaml``.
    max_train_rows, max_test_rows:
        Optional caps for smoke runs.

    Returns
    -------
    PreparedData
        Augmented train and test sets plus the fitted scalers.
    """
    train_inputs, train_observations = load_merged_csv(train_csv, n_rows=max_train_rows)
    test_inputs, test_observations = load_merged_csv(test_csv, n_rows=max_test_rows)

    # Fitted on the clean *training* data only, then applied everywhere else.
    input_scaler = StandardScaler().fit(train_inputs)
    observation_scaler = StandardScaler().fit(train_observations)
    logger.info(
        "Fitted scalers on %d clean training rows (%d inputs, %d observations)",
        len(train_inputs),
        N_UNOBSERVED_INPUTS,
        N_OBSERVED_OUTPUTS,
    )

    augmentation = config.augmentation
    train = build_augmented_dataset(
        inputs=train_inputs,
        observations=train_observations,
        noise_factor=augmentation.noise_factor,
        min_masked=augmentation.min_masked,
        max_masked=augmentation.max_masked,
        protected_observations=augmentation.protected_observations,
        noise_seed=augmentation.train_noise_seed,
        mask_seed=augmentation.train_mask_seed,
        input_scaler=input_scaler,
        observation_scaler=observation_scaler,
    )
    test = build_augmented_dataset(
        inputs=test_inputs,
        observations=test_observations,
        noise_factor=augmentation.noise_factor,
        min_masked=augmentation.min_masked,
        max_masked=augmentation.max_masked,
        protected_observations=augmentation.protected_observations,
        noise_seed=augmentation.test_noise_seed,
        mask_seed=augmentation.test_mask_seed,
        input_scaler=input_scaler,
        observation_scaler=observation_scaler,
    )

    return PreparedData(
        train=train,
        test=test,
        input_scaler=input_scaler,
        observation_scaler=observation_scaler,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
_ARRAY_KEYS = (
    "train_inputs",
    "train_observations",
    "train_clean_observations",
    "train_version_index",
    "test_inputs",
    "test_observations",
    "test_clean_observations",
    "test_version_index",
)


def save_prepared(data: PreparedData, directory: str | Path) -> Path:
    """Write the prepared arrays and scalers to ``directory``.

    Produces ``datasets.npz`` (float32 arrays) plus ``input_scaler.joblib`` and
    ``observation_scaler.joblib``.  Splitting the scalers out means the training
    stage never has to re-derive them, and the model-calibration stage can map
    predictions back to physical units without reloading the whole dataset.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    array_path = directory / "datasets.npz"
    np.savez_compressed(
        array_path,
        train_inputs=data.train.inputs.astype(np.float32),
        train_observations=data.train.observations.astype(np.float32),
        train_clean_observations=data.train.clean_observations.astype(np.float32),
        train_version_index=data.train.version_index.astype(np.int8),
        test_inputs=data.test.inputs.astype(np.float32),
        test_observations=data.test.observations.astype(np.float32),
        test_clean_observations=data.test.clean_observations.astype(np.float32),
        test_version_index=data.test.version_index.astype(np.int8),
        train_n_per_version=np.int64(data.train.n_per_version),
        test_n_per_version=np.int64(data.test.n_per_version),
        versions=np.array(VERSIONS, dtype=object),
    )
    joblib.dump(data.input_scaler, directory / "input_scaler.joblib")
    joblib.dump(data.observation_scaler, directory / "observation_scaler.joblib")

    size_mb = array_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Saved prepared datasets to %s (%.1f MB): %d train rows, %d test rows",
        directory,
        size_mb,
        len(data.train),
        len(data.test),
    )
    return array_path


def load_prepared(directory: str | Path) -> PreparedData:
    """Read back what :func:`save_prepared` wrote."""
    directory = Path(directory)
    array_path = directory / "datasets.npz"
    if not array_path.is_file():
        raise FileNotFoundError(
            f"No prepared dataset at {array_path}. Run scripts/03_build_datasets.py first."
        )

    with np.load(array_path, allow_pickle=True) as archive:
        arrays: Dict[str, np.ndarray] = {key: archive[key] for key in _ARRAY_KEYS}
        train_n = int(archive["train_n_per_version"])
        test_n = int(archive["test_n_per_version"])

    train = AugmentedDataset(
        inputs=arrays["train_inputs"],
        observations=arrays["train_observations"],
        clean_observations=arrays["train_clean_observations"],
        version_index=arrays["train_version_index"],
        n_per_version=train_n,
    )
    test = AugmentedDataset(
        inputs=arrays["test_inputs"],
        observations=arrays["test_observations"],
        clean_observations=arrays["test_clean_observations"],
        version_index=arrays["test_version_index"],
        n_per_version=test_n,
    )

    logger.info(
        "Loaded prepared datasets from %s: %d train rows, %d test rows",
        directory,
        len(train),
        len(test),
    )
    return PreparedData(
        train=train,
        test=test,
        input_scaler=joblib.load(directory / "input_scaler.joblib"),
        observation_scaler=joblib.load(directory / "observation_scaler.joblib"),
    )


# ---------------------------------------------------------------------------
# Synthetic data for smoke tests
# ---------------------------------------------------------------------------
def synthetic_merged_frame(n_rows: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Generate a stand-in for a merged simulation CSV.

    Lets the training and evaluation stages be exercised end to end without
    EnergyPlus (``--smoke`` on the stage scripts, and the test suite).  The
    generator is deliberately crude: schedule-shaped occupancy driving loads,
    with observations produced by a smooth non-linear function of the inputs
    plus noise.  It is representative in *shape*, not in physics.
    """
    generator = np.random.default_rng(seed)
    hours = np.arange(n_rows)
    time_of_day = hours % 24

    # Office-like occupancy profile: busy 08:00-18:00, near-empty overnight.
    occupancy_fraction = np.where(
        (time_of_day >= 8) & (time_of_day < 18),
        0.6 + 0.4 * generator.random(n_rows),
        0.05 * generator.random(n_rows),
    )
    density_scale = generator.uniform(0.5, 1.0, size=n_rows)

    areas = np.array([149.66, 113.45, 67.30])
    occupant_counts = (areas / 18.58)[None, :] * (
        occupancy_fraction * density_scale
    )[:, None]
    lighting_kw = (areas * 19.48 / 1000.0)[None, :] * (
        occupancy_fraction * density_scale
    )[:, None]
    plug_kw = (areas * 10.76 / 1000.0)[None, :] * (
        occupancy_fraction * density_scale
    )[:, None]
    inputs = np.concatenate([occupant_counts, lighting_kw, plug_kw], axis=1)

    # Observations: a smooth function of the total internal gain plus weather-like
    # seasonality, so the inverse problem is solvable but not trivial.
    total_gain = lighting_kw.sum(axis=1) + plug_kw.sum(axis=1)
    seasonal = 10.0 * np.sin(2 * np.pi * hours / 8760.0)
    daily = 5.0 * np.sin(2 * np.pi * time_of_day / 24.0)

    observations = np.empty((n_rows, N_OBSERVED_OUTPUTS))
    for zone in range(6):  # attic + 5 zones, temperature then humidity
        observations[:, 2 * zone] = (
            21.0 + 0.5 * zone + daily + seasonal + 0.8 * total_gain
        ) + generator.normal(0, 0.2, n_rows)
        observations[:, 2 * zone + 1] = np.clip(
            45.0 + 2.0 * zone - 0.4 * daily - 1.5 * total_gain
            + generator.normal(0, 0.5, n_rows),
            5.0,
            95.0,
        )
    observations[:, 12] = 3.6e6 * (total_gain + 2.0) + generator.normal(0, 1e5, n_rows)
    observations[:, 13] = 3.6e6 * np.clip(
        4.0 - 0.3 * (seasonal + daily) - 0.2 * total_gain, 0.05, None
    ) + generator.normal(0, 1e5, n_rows)

    frame = pd.DataFrame(
        np.concatenate([inputs, observations], axis=1), columns=ALL_MODEL_COLUMNS
    )
    return frame
