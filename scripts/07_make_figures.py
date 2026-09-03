#!/usr/bin/env python
"""Stage 07 - render the result figures.

Reads what stages 05 and 06 wrote and produces the figures of Section 5:

    Figs. 7-9    density of predicted vs actual model inputs, per experiment
    Figs. 10-12  probability calibration, per experiment
    Figs. 13-15  sharpness of the predicted distributions, per experiment
    Figs. 16-17  CVRMSE for facility electricity and gas
    Fig. 5       the Latin Hypercube design, if the sample files are present

Figures are written to ``<data-root>/08_figures``.

No EnergyPlus and no GPU needed.

Examples
--------
    python scripts/07_make_figures.py
    python scripts/07_make_figures.py --format png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from genphysical.config import (  # noqa: E402
    DEFAULT_EVALUATION_CONFIG,
    EvaluationConfig,
)
from genphysical.evaluation import plots  # noqa: E402
from genphysical.paths import add_path_arguments, paths_from_args  # noqa: E402
from genphysical.utils.logging_utils import get_logger, setup_logging  # noqa: E402

logger = get_logger("stage07")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the result figures from the saved evaluation output.",
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
        help="Model run names to include.",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png"],
        default="pdf",
        help="Output file format.",
    )
    add_path_arguments(parser)
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    paths = paths_from_args(args)
    config = EvaluationConfig.from_yaml(args.config)
    figures_dir = paths.figures_dir
    paths.ensure_dirs(figures_dir)
    suffix = args.format

    curves_path = paths.results_dir / "curves.npz"
    axes_path = paths.results_dir / "metric_axes.npz"
    if not curves_path.is_file():
        logger.error(
            "No evaluation output at %s. Run scripts/05_evaluate.py first.",
            curves_path,
        )
        return 1

    curves = dict(np.load(curves_path))
    with np.load(axes_path) as archive:
        confidence_levels = archive["confidence_levels"]
        sharpness_coverages = archive["sharpness_coverages"]

    written = 0

    # --- Figs. 7-15: one density, calibration and sharpness figure per experiment
    for experiment in config.experiments:
        point_estimates: Dict[str, np.ndarray] = {}
        calibration: Dict[str, np.ndarray] = {}
        sharpness: Dict[str, np.ndarray] = {}
        actual = None

        for model in args.models:
            prefix = f"{model}_{experiment.name}"
            estimates_path = paths.results_dir / f"{prefix}_point_estimates.npz"
            if not estimates_path.is_file():
                logger.warning("Skipping %s: %s not found", prefix, estimates_path)
                continue
            with np.load(estimates_path) as archive:
                point_estimates[model] = archive["point"]
                actual = archive["actual"]
            if f"{prefix}_calibration_curve" in curves:
                calibration[model] = curves[f"{prefix}_calibration_curve"]
            if f"{prefix}_sharpness_widths" in curves:
                sharpness[model] = curves[f"{prefix}_sharpness_widths"]

        if actual is None:
            continue

        label = f"{experiment.name.replace('_', ' ').title()} - {experiment.label}"

        if point_estimates:
            plots.plot_density_comparison(
                actual=actual,
                predictions=point_estimates,
                output_path=figures_dir / f"density_{experiment.name}.{suffix}",
                title=label,
            )
            written += 1
        if calibration:
            plots.plot_calibration(
                confidence_levels=confidence_levels,
                curves=calibration,
                output_path=figures_dir / f"calibration_{experiment.name}.{suffix}",
                title=label,
            )
            written += 1
        if sharpness:
            plots.plot_sharpness(
                coverages=sharpness_coverages,
                widths=sharpness,
                output_path=figures_dir / f"sharpness_{experiment.name}.{suffix}",
                title=label,
            )
            written += 1

    # --- Figs. 16-17: model calibration accuracy ----------------------------
    accuracy_path = paths.results_dir / "model_calibration_accuracy.csv"
    if accuracy_path.is_file():
        accuracy = pd.read_csv(accuracy_path)
        for meter in config.model_calibration.target_meters:
            if meter not in set(accuracy["meter"]):
                continue
            stem = "electricity" if "Electricity" in meter else "gas"
            plots.plot_cvrmse(
                results=accuracy,
                meter=meter,
                output_path=figures_dir / f"cvrmse_{stem}.{suffix}",
                threshold_percent=config.model_calibration.hourly_threshold_percent,
                title=f"Model calibration accuracy - {stem}",
            )
            written += 1
    else:
        logger.info(
            "No %s; skipping the CVRMSE figures. Run scripts/06_model_calibration.py "
            "to produce them.",
            accuracy_path.name,
        )

    # --- Fig. 5: the sampled model-input space ------------------------------
    samples_path = paths.samples_dir / "train_samples.csv"
    if samples_path.is_file():
        plots.plot_lhs_samples(
            samples=pd.read_csv(samples_path),
            output_path=figures_dir / f"lhs_samples.{suffix}",
            title="Latin Hypercube design",
        )
        written += 1

    logger.info("Wrote %d figure(s) to %s", written, figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
