#!/usr/bin/env python
"""Stage 01 - generate the simulated training data.

Implements the training half of Sections 5.3.1-5.3.2:

    1. Draw 400 Latin Hypercube samples over occupant density, lighting power
       density and equipment power density.
    2. Write one IDF per sample, with those densities applied to all five
       conditioned zones.
    3. Run all 400 whole-year hourly simulations in parallel.
    4. Merge the output, keep the 9 unobserved inputs and the 14 observed
       outputs, and convert the load columns from W to kW.

The result is the 400 x 8760 = 3,504,000-row training table.

Requires EnergyPlus. Configure its location once in ``configs/paths.yaml``
(copy ``configs/paths.example.yaml``), or pass ``--energyplus-dir``.

Examples
--------
Full run as in the paper::

    python scripts/01_generate_training_data.py --num-cores 32

Quick check that the toolchain works (3 simulations)::

    python scripts/01_generate_training_data.py --n-samples 3 --num-cores 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genphysical.config import (  # noqa: E402
    DEFAULT_DATA_GENERATION_CONFIG,
    DataGenerationConfig,
)
from genphysical.energyplus import idf_tools, postprocess, runner, sampling  # noqa: E402
from genphysical.paths import add_path_arguments, paths_from_args  # noqa: E402
from genphysical.utils.logging_utils import get_logger, setup_logging  # noqa: E402

logger = get_logger("stage01")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the simulated training dataset with EnergyPlus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_DATA_GENERATION_CONFIG),
        help="Data-generation configuration file.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override the number of LHS samples (default: from the config, 400).",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        help="Concurrent EnergyPlus processes (default: from the config).",
    )
    parser.add_argument(
        "--weather",
        default=None,
        help="EPW file to simulate against (default: the training file in paths.yaml).",
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Only write the LHS samples and the modified IDF files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse run directories that already contain output instead of re-running.",
    )
    add_path_arguments(parser)
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    paths = paths_from_args(args)
    config = DataGenerationConfig.from_yaml(args.config)

    n_samples = args.n_samples or config.sampling.n_train_samples
    num_cores = args.num_cores or config.simulation.num_cores
    weather = Path(args.weather) if args.weather else paths.train_weather

    paths.require_file(paths.baseline_idf, "Baseline IDF")
    paths.require_file(weather, "Weather file")
    idd_file = paths.require_energyplus()
    idf_tools.set_idd(idd_file)

    paths.ensure_dirs(
        paths.samples_dir, paths.modified_idf_dir, paths.simulation_dir, paths.merged_dir
    )

    # --- 1. Latin Hypercube design ------------------------------------------
    logger.info("Step 1/4: drawing %d LHS samples", n_samples)
    samples = sampling.latin_hypercube_samples(
        bounds=config.sampling.bounds,
        n_samples=n_samples,
        names=config.sampling.parameter_names,
        seed=config.sampling.train_seed,
    )
    sampling.save_samples(samples, paths.samples_dir / "train_samples.csv")

    # --- 2. one IDF per sample ----------------------------------------------
    logger.info("Step 2/4: writing %d modified IDF files", n_samples)
    train_idf_dir = paths.modified_idf_dir / "train"
    idf_paths = idf_tools.generate_modified_idfs(
        samples=samples,
        baseline_idf=paths.baseline_idf,
        epw_file=weather,
        output_dir=train_idf_dir,
        prefix="train",
    )

    if args.skip_simulation:
        logger.info("--skip-simulation given; stopping after IDF generation.")
        return 0

    # --- 3. run the batch ----------------------------------------------------
    simulation_root = paths.simulation_dir / "train"
    pending = idf_paths
    if args.skip_existing:
        pending = [
            path
            for path in idf_paths
            if runner.find_output_csv(simulation_root / path.stem) is None
        ]
        logger.info(
            "%d of %d simulations already have output and will be reused",
            len(idf_paths) - len(pending),
            len(idf_paths),
        )

    if pending:
        logger.info(
            "Step 3/4: running %d whole-year simulations on %d cores "
            "(this is the long one)",
            len(pending),
            num_cores,
        )
        runner.run_batch(
            idf_paths=pending,
            epw_file=weather,
            output_root=simulation_root,
            num_cores=num_cores,
        )
    else:
        logger.info("Step 3/4: nothing to run.")

    # --- 4. merge and reduce to the canonical columns -----------------------
    logger.info("Step 4/4: merging simulation output")
    run_directories = [simulation_root / path.stem for path in idf_paths]
    postprocess.merge_simulation_outputs(
        run_directories=run_directories,
        output_csv=paths.train_merged_csv,
        n_rows_per_run=config.simulation.hours_per_year,
        to_kilowatts=config.simulation.convert_loads_to_kw,
    )

    logger.info("Training data ready: %s", paths.train_merged_csv)
    logger.info("Next: python scripts/02_generate_test_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
