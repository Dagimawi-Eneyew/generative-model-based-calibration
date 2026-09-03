"""Building and reloading calibrator models.

A single entry point for both, so a checkpoint can only ever be loaded into the
architecture that produced it.  :func:`save_model` writes the configuration next
to the weights and :func:`load_trained_model` reads it back, so the architecture
never has to be restated at evaluation time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..config import ModelConfig, dump_config, load_model_config
from ..utils.logging_utils import get_logger
from .cinn import ConditionalINN
from .decinet import DECINet
from .flow import ConditionalRealNVP

logger = get_logger(__name__)

#: Filename of the configuration frozen alongside every checkpoint.
CONFIG_FILENAME = "model_config.yaml"
#: Stem of the TensorFlow-format weight files.
WEIGHTS_STEM = "weights"


def build_model(config: ModelConfig) -> ConditionalRealNVP:
    """Instantiate the calibrator described by ``config``.

    Returns an unbuilt Keras model; call :func:`compile_model` and then either
    ``fit`` or :func:`_materialise_variables` before loading weights.
    """
    if config.model == "cinn":
        model = ConditionalINN(config.architecture)
    elif config.model == "decinet":
        model = DECINet(
            config.architecture,
            lambda_reconstruction=config.training.lambda_reconstruction,
        )
    else:  # pragma: no cover - ModelConfig already validates this
        raise ValueError(f"Unknown model {config.model!r}.")

    logger.info(
        "Built %s: %d coupling blocks, %d x %d sub-network, condition dim %d",
        config.model,
        config.architecture.num_coupling_blocks,
        config.architecture.num_layers,
        config.architecture.num_neurons,
        config.architecture.condition_dim,
    )
    return model


def compile_model(model: ConditionalRealNVP, config: ModelConfig) -> ConditionalRealNVP:
    """Attach the optimiser from ``config`` (Table 1 specifies Adam)."""
    optimizer_name = config.training.optimizer.lower()
    if optimizer_name != "adam":
        raise ValueError(
            f"Only the Adam optimizer of Table 1 is supported; got {optimizer_name!r}."
        )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.training.learning_rate)
    )
    return model


def _materialise_variables(model: ConditionalRealNVP, config: ModelConfig) -> None:
    """Force variable creation by running one dummy batch through the model.

    Keras creates a subclassed model's weights lazily, so ``load_weights`` on a
    freshly constructed model would have nothing to restore into.
    """
    dummy = tf.zeros((2, config.n_input_columns), dtype=tf.float32)
    model(dummy, training=False)


def make_callbacks(config: ModelConfig, log_dir: Optional[Path] = None) -> list:
    """Build the training callbacks of Table 1.

    Early stopping on the validation loss with patience 5, restoring the best
    weights, plus an optional TensorBoard writer.
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.training.early_stopping_patience,
            restore_best_weights=config.training.restore_best_weights,
            verbose=1,
        )
    ]
    if log_dir is not None:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=str(log_dir)))
    return callbacks


def save_model(
    model: ConditionalRealNVP, config: ModelConfig, directory: Union[str, Path]
) -> Path:
    """Write weights and the frozen configuration to ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    weights_path = directory / WEIGHTS_STEM
    model.save_weights(str(weights_path), save_format="tf")
    dump_config(config, directory / CONFIG_FILENAME)

    logger.info("Saved %s weights and config to %s", config.model, directory)
    return directory


def load_trained_model(
    directory: Union[str, Path], config: Optional[ModelConfig] = None
) -> tuple:
    """Reload a trained calibrator together with its configuration.

    Parameters
    ----------
    directory:
        A directory written by :func:`save_model`.
    config:
        Optional override.  Normally omitted: the configuration frozen at save
        time is authoritative, which is what prevents an architecture mismatch
        between training and evaluation.

    Returns
    -------
    (ConditionalRealNVP, ModelConfig)
        The restored model and the configuration it was built from.
    """
    directory = Path(directory)
    config_path = directory / CONFIG_FILENAME
    if config is None:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"No {CONFIG_FILENAME} in {directory}. A checkpoint must ship "
                "with the configuration that produced it; re-run training with "
                "scripts/04_train.py, which writes both."
            )
        config = load_model_config(config_path)

    model = build_model(config)
    compile_model(model, config)
    _materialise_variables(model, config)

    weights_path = directory / WEIGHTS_STEM
    status = model.load_weights(str(weights_path))
    # Surfaces any variable that the checkpoint did not restore, rather than
    # letting a silent partial restore through.
    status.assert_existing_objects_matched()

    logger.info("Restored %s from %s", config.model, directory)
    return model, config


def count_parameters(model: keras.Model) -> int:
    """Total number of trainable scalars, for the model summary in the logs."""
    return int(sum(np.prod(variable.shape) for variable in model.trainable_variables))
