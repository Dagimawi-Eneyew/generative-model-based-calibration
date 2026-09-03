"""Reproducible seeding.

Every stochastic component takes an explicit
:class:`numpy.random.Generator` built from a named seed, so results depend only
on the configuration and not on the order in which functions were called.
:func:`seed_everything` remains for the framework-level state that cannot be
threaded through explicitly (TensorFlow weight initialisation and shuffling).
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int, deterministic_ops: bool = False) -> None:
    """Seed Python, NumPy and TensorFlow.

    Parameters
    ----------
    seed:
        Base seed applied to every framework RNG.
    deterministic_ops:
        Ask TensorFlow for deterministic GPU kernels.  This makes runs bitwise
        reproducible on identical hardware at a noticeable cost in throughput,
        so it is off by default.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic_ops:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    # Imported lazily so the non-modelling stages (sampling, simulation,
    # post-processing) do not pay TensorFlow's import cost.
    try:
        import tensorflow as tf
    except ImportError:  # pragma: no cover - TF is optional for those stages
        return

    tf.random.set_seed(seed)
    if deterministic_ops and hasattr(tf.config.experimental, "enable_op_determinism"):
        tf.config.experimental.enable_op_determinism()


def rng(seed: Optional[int]) -> np.random.Generator:
    """Return a fresh :class:`numpy.random.Generator` for ``seed``.

    Using an explicit generator rather than the legacy global ``np.random``
    functions keeps each stochastic step independent of every other one.
    """
    return np.random.default_rng(seed)
