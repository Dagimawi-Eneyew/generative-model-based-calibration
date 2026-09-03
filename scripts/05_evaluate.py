#!/usr/bin/env python
"""Stage 05 - evaluate the calibrator models.

Runs the three experiments of Table 3 for every trained model and computes
three of the four evaluation aspects of Section 4.5:

    point estimate accuracy     RMSE per zone and model input   (Table 4)
    probabilistic accuracy      calibration error, sharpness,
                                CRPS                            (Figs. 10-15, Table 5)
    inference time              per calibration problem         (Section 5.11)

The fourth, model calibration accuracy, needs EnergyPlus and lives in stage 06.

Each experiment differs only in which VPOA version of the test data it is given:

    Experiment 1   clean            no noise, no missing readings
    Experiment 2   noisy            all sensors working, all noisy
    Experiment 3   noisy_missing    random sensor failures plus noise

Outputs go to ``<data-root>/07_results``: raw posterior samples per experiment,
a tidy metrics CSV, and a JSON of the timing measurements.

No EnergyPlus needed.

Examples
--------
    python scripts/05_evaluate.py
    python scripts/05_evaluate.py --models cinn decinet --crps gaussian
    python scripts/05_evaluate.py --max-test-rows 500      # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from genphysical.config import (  # noqa: E402
    DEFAULT_EVALUATION_CONFIG,
    EvaluationConfig,
)
from genphysical.constants import UNOBSERVED_INPUT_LABELS  # noqa: E402
from genphysical.data.datasets import load_prepared  # noqa: E402
from genphysical.evaluation import metrics  # noqa: E402
from genphysical.evaluation.predict import (  # noqa: E402
    measure_inference_time,
    predict_posterior,
)
from genphysical.models.builder import load_trained_model  # noqa: E402
from genphysical.paths import add_path_arguments, paths_from_args  # noqa: E402
from genphysical.utils.logging_utils import get_logger, setup_logging  # noqa: E402
from genphysical.utils.seeding import seed_everything  # noqa: E402

logger = get_logger("stage05")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiments 1-3 and compute the evaluation metrics.",
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
        help="Run names under <data-root>/06_models to evaluate.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Evaluate only the first N rows of each experiment. For quick checks.",
    )
    parser.add_argument(
        "--crps",
        choices=["ensemble", "gaussian"],
        default=None,
        help=(
            "CRPS estimator, overriding the config. 'gaussian' reproduces the "
            "published Table 5 numbers; 'ensemble' is Eq. 13 as written."
        ),
    )
    parser.add_argument(
        "--calibration-definition",
        choices=["cdf", "interval"],
        default=None,
        help="Empirical-frequency definition for the calibration curve.",
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Skip the inference-time benchmark.",
    )
    parser.add_argument(
        "--save-posteriors",
        action="store_true",
        help=(
            "Save the full posterior sample arrays. Large "
            "(n_points x 9 x n_samples floats) but needed to regenerate figures "
            "without re-running inference."
        ),
    )
    add_path_arguments(parser)
    return parser.parse_args()


def evaluate_experiment(
    model,
    model_name: str,
    experiment,
    data,
    config: EvaluationConfig,
    crps_estimator: str,
    calibration_definition: str,
    max_rows: int | None,
    n_posterior_samples: int,
    point_statistic: str,
    seed: int,
) -> tuple:
    """Run one experiment and return ``(metric rows, predictions)``."""
    rows = data.test.version_slice(experiment.data_version)
    observations = data.test.observations[rows]
    actual_inputs = data.test.inputs[rows]
    if max_rows is not None:
        observations = observations[:max_rows]
        actual_inputs = actual_inputs[:max_rows]

    logger.info(
        "=== %s / %s (%s): %d test points ===",
        model_name,
        experiment.name,
        experiment.label,
        len(observations),
    )

    predictions = predict_posterior(
        model=model,
        observations=observations,
        actual_inputs=actual_inputs,
        input_scaler=data.input_scaler,
        n_samples=n_posterior_samples,
        point_statistic=point_statistic,
        seed=seed,
    )

    # --- Section 5.5.1: point estimate accuracy -----------------------------
    rmse_values = metrics.rmse(predictions.actual, predictions.point)
    mae_values = metrics.mae(predictions.actual, predictions.point)
    r2_values = metrics.r2_score(predictions.actual, predictions.point)

    # --- Section 5.6.1: probability calibration -----------------------------
    curve = metrics.calibration_curve(
        predictions.actual,
        predictions.posterior,
        config.metrics.confidence_levels,
        definition=calibration_definition,
    )
    ce_sum = metrics.calibration_error(
        curve, config.metrics.confidence_levels, reduction="sum"
    )
    ce_mean = metrics.calibration_error(
        curve, config.metrics.confidence_levels, reduction="mean"
    )

    # --- Section 5.6.2: sharpness -------------------------------------------
    widths = metrics.sharpness(
        predictions.posterior, config.metrics.sharpness_coverages
    )
    # A single representative width: the 90 % interval, if it was requested.
    coverages = list(config.metrics.sharpness_coverages)
    sharpness_index = coverages.index(0.9) if 0.9 in coverages else len(coverages) // 2

    # --- Section 5.6.3: CRPS -------------------------------------------------
    crps_values = metrics.crps(
        predictions.actual,
        predictions.posterior,
        estimator=crps_estimator,
        max_samples=config.metrics.crps_max_samples,
    )

    metric_rows: List[dict] = []
    for variable, label in enumerate(UNOBSERVED_INPUT_LABELS):
        metric_rows.append(
            {
                "model": model_name,
                "experiment": experiment.name,
                "experiment_label": experiment.label,
                "variable": label,
                "rmse": float(rmse_values[variable]),
                "mae": float(mae_values[variable]),
                "r2": float(r2_values[variable]),
                "crps": float(crps_values[variable]),
                "calibration_error_sum": float(ce_sum[variable]),
                "calibration_error_mean": float(ce_mean[variable]),
                f"sharpness_{coverages[sharpness_index]:g}": float(
                    widths[variable, sharpness_index]
                ),
            }
        )

    curves = {
        "calibration_curve": curve,
        "sharpness_widths": widths,
    }
    return metric_rows, predictions, curves


def main() -> int:
    setup_logging()
    args = parse_args()

    paths = paths_from_args(args)
    config = EvaluationConfig.from_yaml(args.config)
    crps_estimator = args.crps or config.metrics.crps_estimator
    calibration_definition = (
        args.calibration_definition or config.metrics.calibration_definition
    )

    data = load_prepared(paths.dataset_dir)
    paths.ensure_dirs(paths.results_dir)

    all_metrics: List[dict] = []
    timings: Dict[str, dict] = {}
    curve_archive: Dict[str, np.ndarray] = {}

    for model_name in args.models:
        model_dir = paths.model_dir / model_name
        if not model_dir.is_dir():
            logger.error(
                "No trained model at %s. Run: python scripts/04_train.py --model %s",
                model_dir,
                model_name,
            )
            return 1

        model, model_config = load_trained_model(model_dir)
        seed_everything(model_config.inference.seed)

        for experiment in config.experiments:
            metric_rows, predictions, curves = evaluate_experiment(
                model=model,
                model_name=model_name,
                experiment=experiment,
                data=data,
                config=config,
                crps_estimator=crps_estimator,
                calibration_definition=calibration_definition,
                max_rows=args.max_test_rows,
                n_posterior_samples=model_config.inference.n_posterior_samples,
                point_statistic=model_config.inference.point_estimate,
                seed=model_config.inference.seed,
            )
            all_metrics.extend(metric_rows)

            prefix = f"{model_name}_{experiment.name}"
            # Point estimates and ground truth are small and always saved: they
            # feed the density figures and the model-calibration stage.
            np.savez_compressed(
                paths.results_dir / f"{prefix}_point_estimates.npz",
                actual=predictions.actual,
                point=predictions.point,
            )
            if args.save_posteriors:
                np.savez_compressed(
                    paths.results_dir / f"{prefix}_posterior.npz",
                    posterior=predictions.posterior,
                )
            for name, array in curves.items():
                curve_archive[f"{prefix}_{name}"] = array

        if not args.skip_timing:
            timings[model_name] = measure_inference_time(
                model=model,
                observations=data.test.observations[data.test.version_slice("clean")],
                n_samples=model_config.inference.n_posterior_samples,
                n_timed=config.timing.n_timed_observations,
                n_warmup=config.timing.n_warmup,
            )

    # --- write results -------------------------------------------------------
    metrics_frame = pd.DataFrame(all_metrics)
    metrics_path = paths.results_dir / "metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    logger.info("Wrote %s", metrics_path)

    np.savez_compressed(paths.results_dir / "curves.npz", **curve_archive)
    np.savez_compressed(
        paths.results_dir / "metric_axes.npz",
        confidence_levels=np.array(config.metrics.confidence_levels),
        sharpness_coverages=np.array(config.metrics.sharpness_coverages),
    )

    if timings:
        timing_path = paths.results_dir / "inference_time.json"
        timing_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
        logger.info("Wrote %s", timing_path)

    # --- console summary -----------------------------------------------------
    summary = (
        metrics_frame.groupby(["model", "experiment"])[
            ["rmse", "crps", "calibration_error_sum"]
        ]
        .mean()
        .round(4)
    )
    logger.info("Mean across the nine model inputs:\n%s", summary.to_string())
    logger.info(
        "CRPS estimator: %s | calibration definition: %s",
        crps_estimator,
        calibration_definition,
    )
    logger.info(
        "Next: python scripts/06_model_calibration.py (needs EnergyPlus), "
        "then python scripts/07_make_figures.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
