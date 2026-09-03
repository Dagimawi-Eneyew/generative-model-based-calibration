#!/usr/bin/env python
"""Stage 04 - train a calibrator model.

Trains either the baseline conditional invertible neural network (Section 2.3,
Eq. 4) or DECI-Net (Section 4.4, Eq. 8) on the augmented training data.

The architecture, optimiser and schedule all come from a YAML config, and the
config is written next to the checkpoint when training finishes.  Stage 05 then
rebuilds the model from *that* file rather than from anything typed a second
time, which is what makes it impossible to evaluate a different architecture
than the one that was trained.

Examples
--------
    python scripts/04_train.py --model cinn
    python scripts/04_train.py --model decinet
    python scripts/04_train.py --config configs/decinet_as_trained.yaml
    python scripts/04_train.py --model decinet --epochs 2 --max-rows 20000   # quick trial
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from genphysical.config import ModelConfig, load_model_config  # noqa: E402
from genphysical.data.datasets import build_model_matrix, load_prepared  # noqa: E402
from genphysical.models.builder import (  # noqa: E402
    build_model,
    compile_model,
    count_parameters,
    make_callbacks,
    save_model,
)
from genphysical.paths import add_path_arguments, paths_from_args  # noqa: E402
from genphysical.utils.logging_utils import get_logger, setup_logging  # noqa: E402
from genphysical.utils.seeding import seed_everything  # noqa: E402

logger = get_logger("stage04")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the cINN baseline or DECI-Net.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["cinn", "decinet"],
        default=None,
        help="Which calibrator to train. Selects configs/<model>.yaml.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Explicit model configuration file, overriding --model.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output directory name under <data-root>/06_models (default: the model name).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override the configured epoch count."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override the configured batch size."
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Train on a random subset of this many rows. For quick trials.",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help=(
            "Distribute training over all visible GPUs with MirroredStrategy."
        ),
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write TensorBoard logs alongside the checkpoint.",
    )
    add_path_arguments(parser)
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> ModelConfig:
    """Pick the model config from --config or --model."""
    if args.config:
        return load_model_config(args.config)
    if args.model:
        return load_model_config(args.model)
    raise SystemExit("Specify either --model {cinn,decinet} or --config <file>.")


def main() -> int:
    setup_logging()
    args = parse_args()

    paths = paths_from_args(args)
    config = resolve_config(args)

    # CLI overrides are folded into the config itself, so the file saved next to
    # the checkpoint always describes what actually ran.
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if overrides:
        config = dataclasses.replace(
            config, training=dataclasses.replace(config.training, **overrides)
        )
        logger.info("Applied CLI overrides: %s", overrides)

    seed_everything(config.training.seed)

    # --- data ---------------------------------------------------------------
    data = load_prepared(paths.dataset_dir)
    train_matrix = build_model_matrix(data.train, config.model)

    if args.max_rows is not None and args.max_rows < len(train_matrix):
        generator = np.random.default_rng(config.training.seed)
        chosen = generator.choice(len(train_matrix), size=args.max_rows, replace=False)
        train_matrix = train_matrix[chosen]
        logger.warning(
            "--max-rows: training on a random subset of %d rows", len(train_matrix)
        )

    logger.info(
        "Training %s on %d rows x %d columns",
        config.model,
        train_matrix.shape[0],
        train_matrix.shape[1],
    )
    if train_matrix.shape[1] != config.n_input_columns:
        raise ValueError(
            f"{config.model} expects {config.n_input_columns} columns but the "
            f"prepared dataset has {train_matrix.shape[1]}."
        )

    # --- model --------------------------------------------------------------
    import tensorflow as tf  # imported here so --help stays instant

    if args.multi_gpu:
        gpus = tf.config.list_logical_devices("GPU")
        strategy = tf.distribute.MirroredStrategy(gpus)
        logger.info("Distributing training across %d GPU(s)", len(gpus))
        scope = strategy.scope()
    else:
        scope = _NullScope()

    run_name = args.run_name or config.model
    output_dir = paths.model_dir / run_name
    log_dir = (
        output_dir / "tensorboard" / datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.tensorboard
        else None
    )

    with scope:
        model = build_model(config)
        compile_model(model, config)
        callbacks = make_callbacks(config, log_dir=log_dir)

        history = model.fit(
            train_matrix,
            batch_size=config.training.batch_size,
            epochs=config.training.epochs,
            validation_split=config.training.validation_split,
            shuffle=config.training.shuffle,
            callbacks=callbacks,
            verbose=1,
        )

    logger.info("Trainable parameters: %s", f"{count_parameters(model):,}")

    # --- persist ------------------------------------------------------------
    save_model(model, config, output_dir)
    history_frame = pd.DataFrame(history.history)
    history_frame.index.name = "epoch"
    history_frame.to_csv(output_dir / "training_history.csv")

    best_epoch = int(np.argmin(history_frame["val_loss"])) if "val_loss" in history_frame else -1
    logger.info(
        "Finished after %d epoch(s); best validation loss %.4f at epoch %d",
        len(history_frame),
        float(history_frame["val_loss"].min()) if "val_loss" in history_frame else float("nan"),
        best_epoch,
    )
    logger.info("Model written to %s", output_dir)
    logger.info("Next: python scripts/05_evaluate.py")
    return 0


class _NullScope:
    """Stand-in for a distribution strategy scope when running on one device."""

    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
