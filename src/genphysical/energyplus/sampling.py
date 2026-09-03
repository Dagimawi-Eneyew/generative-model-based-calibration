"""Latin Hypercube sampling of the influential model-input space.

Implements Section 5.3.1 of the paper:

    "the three unobserved physics-model inputs were sampled using Latin
     Hypercube Sampling (LHS). The LHS samples [...] were generated using
     area-dependent inputs in the building energy model, specifically occupant
     density (m2/person), equipment power density (W/m2), and lighting power
     density (W/m2); these model inputs were varied from 50% to 100% of their
     default values. The input LHS sample sizes were set at 400 for training and
     8760 for testing and generated separately."


"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc

from ..config import SamplingConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def latin_hypercube_samples(
    bounds: Sequence[tuple],
    n_samples: int,
    names: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Draw a Latin Hypercube design over the given bounds.

    Parameters
    ----------
    bounds:
        One ``(low, high)`` pair per dimension.
    n_samples:
        Number of design points; each dimension is stratified into this many
        equiprobable intervals.
    names:
        Column names for the returned frame.  Defaults to ``x0, x1, ...``.
    seed:
        Seed for the sampler, making the design reproducible.

    Returns
    -------
    pandas.DataFrame
        ``n_samples`` rows, one column per dimension, values scaled into the
        requested bounds.
    """
    bounds = [tuple(float(v) for v in pair) for pair in bounds]
    for low, high in bounds:
        if not high > low:
            raise ValueError(f"Invalid sampling bound ({low}, {high}): need high > low.")

    n_dimensions = len(bounds)
    sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)

    # Unit-hypercube design, then affine-scaled into the real ranges.
    unit_design = sampler.random(n=n_samples)
    lower = np.array([low for low, _ in bounds])
    upper = np.array([high for _, high in bounds])
    design = qmc.scale(unit_design, lower, upper)

    columns = list(names) if names is not None else [f"x{i}" for i in range(n_dimensions)]
    if len(columns) != n_dimensions:
        raise ValueError(
            f"Got {len(columns)} names for {n_dimensions} sampled dimensions."
        )

    logger.info(
        "Drew %d LHS samples over %d dimensions (seed=%s)", n_samples, n_dimensions, seed
    )
    return pd.DataFrame(design, columns=columns)


def training_samples(config: SamplingConfig) -> pd.DataFrame:
    """The 400-point training design of Section 5.3.1.

    Each row becomes one whole-year EnergyPlus simulation in which the three
    densities are held constant, so the 400 runs yield 400 x 8760 = 3,504,000
    hourly training rows.
    """
    return latin_hypercube_samples(
        bounds=config.bounds,
        n_samples=config.n_train_samples,
        names=config.parameter_names,
        seed=config.train_seed,
    )


def test_samples(config: SamplingConfig) -> pd.DataFrame:
    """The 8760-point test design of Section 5.3.1.

    Unlike the training design, these samples are *not* one simulation each.
    They are consumed hour by hour inside a single annual run via a
    ``Schedule:File`` (see :mod:`genphysical.energyplus.idf_tools`), so that the
    unobserved model inputs change every hour - imitating "the dynamic and
    uncertain conditions of the physical building".
    """
    return latin_hypercube_samples(
        bounds=config.bounds,
        n_samples=config.n_test_samples,
        names=config.parameter_names,
        seed=config.test_seed,
    )


def save_samples(samples: pd.DataFrame, path: str | Path) -> Path:
    """Write a design matrix to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(path, index=False)
    logger.info("Wrote %d samples to %s", len(samples), path)
    return path
