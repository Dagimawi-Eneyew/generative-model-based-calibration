#!/usr/bin/env python
"""Stage 02 - generate the simulated test data.

Implements the test half of Sections 5.3.1-5.3.2.  The test set is built very
differently from the training set: instead of 400 runs each holding its inputs
constant, it is a *single* whole-year run in which the unobserved model inputs
change every hour.

    "The 8760 LHS samples were used with a single IDF file, actual weather data
     for 2022, and dynamic scheduled inputs to alter the model inputs during
     every hour of the simulation."

That imitates a real building whose occupancy and loads drift continuously, and
it is what makes hour-by-hour continuous calibration a meaningful thing to
evaluate.

Steps:

    1. Run the untouched baseline once, to read the prototype's hourly
       fractional occupancy / lighting / equipment schedules.
    2. Draw 8760 LHS samples - one per hour of the year.
    3. Multiply densities by fractions by zone area to get absolute hourly
       occupant counts and loads for all five conditioned zones.
    4. Convert the model so those values drive it through ``Schedule:File``.
    5. Run the year, verify the simulation reproduced its input schedule, and
       reduce the output to the canonical 23 columns.

The reference run from step 1 is kept: stage 06 compares against it when
measuring model calibration accuracy.

Requires EnergyPlus.
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

logger = get_logger("stage02")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the hourly-varying simulated test dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_DATA_GENERATION_CONFIG),
        help="Data-generation configuration file.",
    )
    parser.add_argument(
        "--weather",
        default=None,
        help="EPW file to simulate against (default: the test file in paths.yaml).",
    )
    parser.add_argument(
        "--force-reference-run",
        action="store_true",
        help="Re-run the baseline reference simulation even if its output exists.",
    )
    add_path_arguments(parser)
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    paths = paths_from_args(args)
    config = DataGenerationConfig.from_yaml(args.config)
    weather = Path(args.weather) if args.weather else paths.test_weather

    paths.require_file(paths.baseline_idf, "Baseline IDF")
    paths.require_file(weather, "Weather file")
    idd_file = paths.require_energyplus()
    idf_tools.set_idd(idd_file)

    test_root = paths.simulation_dir / "test"
    reference_dir = test_root / "reference"
    paths.ensure_dirs(paths.samples_dir, paths.merged_dir, test_root)

    # --- 1. reference run, for the prototype's fractional schedules ---------
    reference_csv = runner.find_output_csv(reference_dir)
    if reference_csv is None or args.force_reference_run:
        logger.info(
            "Step 1/5: running the untouched baseline once, to read its hourly "
            "fractional schedules"
        )
        baseline = idf_tools.load_idf(paths.baseline_idf, weather)
        runner.run_single(baseline, reference_dir)
        reference_csv = runner.find_output_csv(reference_dir)
        if reference_csv is None:
            raise RuntimeError(
                f"The reference run in {reference_dir} produced no output CSV. "
                "Check its EnergyPlus .err file."
            )
    else:
        logger.info("Step 1/5: reusing the existing reference run at %s", reference_csv)

    schedules = idf_tools.extract_fractional_schedules(reference_csv)
    logger.info("Read %d hourly schedule values", len(schedules))

    # --- 2. one LHS sample per hour of the year -----------------------------
    n_samples = config.sampling.n_test_samples
    logger.info("Step 2/5: drawing %d LHS samples, one per simulated hour", n_samples)
    samples = sampling.latin_hypercube_samples(
        bounds=config.sampling.bounds,
        n_samples=n_samples,
        names=config.sampling.parameter_names,
        seed=config.sampling.test_seed,
    )
    sampling.save_samples(samples, paths.samples_dir / "test_samples.csv")

    # --- 3. densities -> absolute per-zone hourly values --------------------
    logger.info("Step 3/5: converting sampled densities into hourly zone loads")
    loads = idf_tools.densities_to_hourly_loads(samples, schedules)
    schedule_csv = idf_tools.write_schedule_csv(
        loads, test_root / "test_schedule.csv"
    )

    # --- 4. build the Schedule:File-driven model ----------------------------
    logger.info("Step 4/5: building the Schedule:File-driven model")
    test_idf_path = paths.modified_idf_dir / "test" / "test_schedule_driven.idf"
    idf = idf_tools.build_schedule_file_idf(
        baseline_idf=paths.baseline_idf,
        epw_file=weather,
        schedule_csv=schedule_csv,
        output_idf=test_idf_path,
        n_hours=config.simulation.hours_per_year,
    )

    # --- 5. simulate, verify, reduce ----------------------------------------
    logger.info("Step 5/5: running the annual test simulation")
    run_dir = test_root / "run"
    runner.run_single(idf, run_dir)
    output_csv = runner.find_output_csv(run_dir)
    if output_csv is None:
        raise RuntimeError(
            f"The test simulation in {run_dir} produced no output CSV. "
            "Check its EnergyPlus .err file."
        )

    # Confirm the model reproduced exactly the inputs it was handed, so the
    # recorded unobserved inputs match what the simulation actually used.
    idf_tools.verify_schedule_roundtrip(schedule_csv, output_csv)

    frame = postprocess.load_simulation_output(
        output_csv,
        n_rows=config.simulation.hours_per_year,
        to_kilowatts=config.simulation.convert_loads_to_kw,
    )
    paths.test_merged_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.test_merged_csv, index=False)

    logger.info("Test data ready: %s (%d rows)", paths.test_merged_csv, len(frame))
    logger.info(
        "Test run kept at %s - stage 06 treats its facility meters as the "
        "physical measurements when computing CVRMSE",
        run_dir,
    )
    logger.info("Next: python scripts/03_build_datasets.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
