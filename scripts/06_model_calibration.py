#!/usr/bin/env python
"""Stage 06 - model calibration accuracy (Section 5.10).

The closing step of Algorithm 1.  Every metric in stage 05 measures how well the
calibrator recovers the unobserved model inputs; this one measures the thing the
framework exists for: after the estimated inputs are fed back into the building
energy model, how closely do its outputs track the measurements?

For each model and each experiment:

    1. Load the point estimates saved by stage 05.
    2. Write them as the hourly Schedule:File that drives the model, clipping
       negatives to zero.
    3. Re-simulate the full year.
    4. Compare facility electricity and gas against the measured run with
       CVRMSE (Eq. 14) and NMBE.

The "measured" run is stage 02's annual test simulation - the one driven by the
*true* sampled hourly inputs, whose outputs are exactly the observations ``y_o``
the calibrator was given.  It stands in for the physical measurements, which is
what makes this a closed-loop check on a synthetic case study.  (Not to be
confused with stage 02's *reference* run of the untouched prototype, which
exists only to read the building's fractional schedules.)

ASHRAE Guideline 14 and FEMP set the hourly CVRMSE acceptance threshold at 30 %.

Requires EnergyPlus. Each experiment costs one whole-year simulation, so a full
run is 2 models x 3 experiments = 6 simulations.

Examples
--------
    python scripts/06_model_calibration.py
    python scripts/06_model_calibration.py --models decinet --experiments experiment_1

Validate the closed loop itself (should report ~0% CVRMSE)::

    python scripts/06_model_calibration.py --truth-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from genphysical.config import (  # noqa: E402
    DEFAULT_EVALUATION_CONFIG,
    EvaluationConfig,
)
from genphysical.constants import UNOBSERVED_INPUT_COLUMNS  # noqa: E402
from genphysical.energyplus import idf_tools, runner  # noqa: E402
from genphysical.evaluation.model_calibration import (  # noqa: E402
    CalibrationAccuracy,
    compare_meters,
    resimulate_with_predictions,
    results_table,
)
from genphysical.paths import add_path_arguments, paths_from_args  # noqa: E402
from genphysical.utils.logging_utils import get_logger, setup_logging  # noqa: E402

logger = get_logger("stage06")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure model calibration accuracy by re-simulating with "
        "the estimated inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_EVALUATION_CONFIG),
        help="Evaluation configuration file.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cinn", "decinet"],
        help="Model run names whose stage-05 predictions should be re-simulated.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Subset of experiment names (default: all in the config).",
    )
    parser.add_argument(
        "--weather",
        default=None,
        help=(
            "EPW file for the re-simulation. Must match the one stage 02 used, "
            "or the comparison measures the weather rather than the calibration."
        ),
    )
    parser.add_argument(
        "--truth-check",
        action="store_true",
        help=(
            "Validate the closed loop instead of evaluating a model: re-simulate "
            "with the TRUE unobserved inputs and confirm CVRMSE is ~0. Any "
            "non-zero result points at the re-simulation plumbing rather than at "
            "the calibrator."
        ),
    )
    parser.add_argument(
        "--measured-run",
        default=None,
        help=(
            "Run directory standing in for the physical measurements "
            "(default: <data-root>/03_simulations/test/run, i.e. stage 02's test "
            "simulation driven by the true sampled inputs)."
        ),
    )
    add_path_arguments(parser)
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    paths = paths_from_args(args)
    config = EvaluationConfig.from_yaml(args.config)
    weather = Path(args.weather) if args.weather else paths.test_weather

    paths.require_file(paths.baseline_idf, "Baseline IDF")
    paths.require_file(weather, "Weather file")
    idd_file = paths.require_energyplus()
    idf_tools.set_idd(idd_file)

    # The stand-in for the physical measurements is stage 02's test simulation,
    # driven by the TRUE sampled inputs - not the reference run of the untouched
    # prototype, which only exists to supply the fractional schedules.
    measured_dir = (
        Path(args.measured_run)
        if args.measured_run
        else paths.simulation_dir / "test" / "run"
    )
    measured_csv = runner.find_output_csv(measured_dir)
    if measured_csv is None:
        logger.error(
            "No test simulation output under %s.\n"
            "  Run scripts/02_generate_test_data.py first - its annual test run "
            "provides the measurements this stage calibrates against.",
            measured_dir,
        )
        return 1
    logger.info("Measured run (true inputs): %s", measured_csv)

    experiments = config.experiments
    if args.experiments:
        wanted = set(args.experiments)
        experiments = [item for item in experiments if item.name in wanted]
        if not experiments:
            logger.error("None of %s match the configured experiments.", args.experiments)
            return 1

    results: List[CalibrationAccuracy] = []

    if args.truth_check:
        # Feed the ground-truth inputs straight back through the re-simulation
        # path. A correct pipeline reproduces the measured run exactly, so this
        # separates "the calibrator is inaccurate" from "the loop is miswired".
        truth_csv = paths.test_merged_csv
        if not truth_csv.is_file():
            logger.error("Missing %s; run scripts/02_generate_test_data.py first.", truth_csv)
            return 1
        truth = pd.read_csv(truth_csv)[UNOBSERVED_INPUT_COLUMNS].to_numpy()
        logger.info("Truth check: re-simulating with %d hours of TRUE inputs", len(truth))

        calibrated_csv = resimulate_with_predictions(
            predictions=truth,
            baseline_idf=paths.baseline_idf,
            epw_file=weather,
            work_dir=paths.results_dir / "resimulation" / "truth_check",
            clip_at_zero=config.model_calibration.clip_predictions_at_zero,
        )
        accuracy = compare_meters(
            measured_csv=measured_csv,
            calibrated_csv=calibrated_csv,
            config=config.model_calibration,
            model="ground_truth",
            experiment="truth_check",
            n_rows=len(truth),
        )
        worst = max(accuracy.cvrmse_percent.values())
        if worst > 0.5:
            logger.error(
                "Truth check FAILED: CVRMSE reached %.4f%% when re-simulating with "
                "the true inputs. Expect ~0%%; the re-simulation path is miswired.",
                worst,
            )
            return 1
        logger.info("Truth check passed: worst CVRMSE %.6f%% with the true inputs.", worst)
        return 0

    for model_name in args.models:
        for experiment in experiments:
            prediction_path = (
                paths.results_dir / f"{model_name}_{experiment.name}_point_estimates.npz"
            )
            if not prediction_path.is_file():
                logger.error(
                    "Missing predictions: %s\n"
                    "  Run scripts/05_evaluate.py first.",
                    prediction_path,
                )
                return 1

            with np.load(prediction_path) as archive:
                point_estimates = archive["point"]

            logger.info(
                "=== %s / %s: re-simulating with %d hours of estimated inputs ===",
                model_name,
                experiment.name,
                len(point_estimates),
            )

            work_dir = paths.results_dir / "resimulation" / f"{model_name}_{experiment.name}"
            calibrated_csv = resimulate_with_predictions(
                predictions=point_estimates,
                baseline_idf=paths.baseline_idf,
                epw_file=weather,
                work_dir=work_dir,
                clip_at_zero=config.model_calibration.clip_predictions_at_zero,
            )

            results.append(
                compare_meters(
                    measured_csv=measured_csv,
                    calibrated_csv=calibrated_csv,
                    config=config.model_calibration,
                    model=model_name,
                    experiment=experiment.name,
                    n_rows=len(point_estimates),
                )
            )

    table = results_table(results, config.model_calibration.hourly_threshold_percent)
    output_path = paths.results_dir / "model_calibration_accuracy.csv"
    table.to_csv(output_path, index=False)

    logger.info("Model calibration accuracy:\n%s", table.to_string(index=False))
    logger.info("Wrote %s", output_path)

    failures = table[~table["meets_hourly_threshold"]]
    if len(failures):
        logger.warning(
            "%d model/experiment/meter combination(s) exceed the %.0f%% hourly "
            "CVRMSE threshold:\n%s",
            len(failures),
            config.model_calibration.hourly_threshold_percent,
            failures[["model", "experiment", "meter", "cvrmse_percent"]].to_string(
                index=False
            ),
        )

    logger.info("Next: python scripts/07_make_figures.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
