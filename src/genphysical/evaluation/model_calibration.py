"""Model calibration accuracy: closing the loop through EnergyPlus.

Implements Section 5.10.  Every other metric asks how well the calibrator
recovers the unobserved model inputs.  This one asks the question the framework
actually exists to answer: *after* the estimated inputs are pushed back into the
building energy model, do its outputs match the measurements?

    "The mean predictions of the two calibrator models in all three experiments
     were input into the building energy model. The resulting electricity and gas
     consumption outputs of these simulations were then compared with the initial
     simulated outputs."

The procedure, which is Algorithm 1 lines 10-11:

    1. Take the point estimates x̂ for all 8760 test hours.
    2. Expand the 9 zone-group values into per-zone hourly schedules and clip
       any negatives to zero (an occupant count cannot be negative).
    3. Write them as the Schedule:File CSV that drives the model.
    4. Re-simulate the whole year: y = M(x̂).
    5. Compare y against the measured facility electricity and gas with
       CVRMSE (Eq. 14) and NMBE.

The "measured" series is stage 02's annual test simulation - the run driven by
the *true* sampled hourly inputs, whose outputs are the very observations the
calibrator was conditioned on.  It plays the role of the physical measurements
``y_o``, which is what makes this a closed-loop check on a synthetic case study.

This stage is the only part of evaluation that needs EnergyPlus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import ModelCalibrationConfig
from ..energyplus.idf_tools import (
    build_schedule_file_idf,
    group_predictions_to_hourly_loads,
    write_schedule_csv,
)
from ..energyplus.runner import find_output_csv, run_single
from ..utils.logging_utils import get_logger
from .metrics import cvrmse, nmbe

logger = get_logger(__name__)


@dataclass
class CalibrationAccuracy:
    """CVRMSE and NMBE for one experiment, per target meter."""

    experiment: str
    model: str
    cvrmse_percent: Dict[str, float]
    nmbe_percent: Dict[str, float]

    def to_rows(self, threshold_percent: float) -> List[dict]:
        """Flatten to tidy rows for a results table."""
        return [
            {
                "model": self.model,
                "experiment": self.experiment,
                "meter": meter,
                "cvrmse_percent": value,
                "nmbe_percent": self.nmbe_percent[meter],
                "meets_hourly_threshold": value < threshold_percent,
            }
            for meter, value in self.cvrmse_percent.items()
        ]


def resimulate_with_predictions(
    predictions: np.ndarray,
    baseline_idf: str | Path,
    epw_file: str | Path,
    work_dir: str | Path,
    clip_at_zero: bool = True,
) -> Path:
    """Re-run the building energy model with the estimated calibration solution.

    Parameters
    ----------
    predictions:
        ``(8760, 9)`` point estimates in physical units - occupant counts
        (people) then lighting and plug loads (kW), for the three zone groups.
    baseline_idf, epw_file:
        The prototype model and the weather file used for the test set.  The
        weather must match the reference run, or the comparison would be
        measuring the weather rather than the calibration.
    work_dir:
        Directory for the generated schedule CSV, the converted IDF and the run
        output.
    clip_at_zero:
        Clip negative predictions before writing the schedule.  The posterior
        mean of an occupant count can be slightly negative; EnergyPlus rejects
        negative design levels.

    Returns
    -------
    pathlib.Path
        The re-simulated run's output CSV.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    loads = group_predictions_to_hourly_loads(predictions, clip_at_zero=clip_at_zero)
    schedule_csv = write_schedule_csv(loads, work_dir / "predicted_schedule.csv")

    idf = build_schedule_file_idf(
        baseline_idf=baseline_idf,
        epw_file=epw_file,
        schedule_csv=schedule_csv,
        output_idf=work_dir / "calibrated_model.idf",
    )
    run_directory = run_single(idf, work_dir / "run")

    output_csv = find_output_csv(run_directory)
    if output_csv is None:
        raise RuntimeError(
            f"The re-simulation in {run_directory} produced no output CSV. "
            "Check the EnergyPlus .err file in that directory."
        )
    return output_csv


def compare_meters(
    measured_csv: str | Path,
    calibrated_csv: str | Path,
    config: ModelCalibrationConfig,
    model: str,
    experiment: str,
    n_rows: Optional[int] = None,
) -> CalibrationAccuracy:
    """Compute CVRMSE and NMBE between the measured run and a calibrated run.

    Parameters
    ----------
    measured_csv:
        Output of the run driven by the true sampled inputs; it stands in for the
        physical measurements ``y_o``.
    calibrated_csv:
        Output of :func:`resimulate_with_predictions`.
    config:
        Model-calibration section of ``configs/evaluation.yaml``.
    model, experiment:
        Labels carried through to the results table.
    n_rows:
        Compare only the leading ``n_rows`` hours; ``None`` compares the shorter
        of the two series.
    """
    measured_frame = pd.read_csv(measured_csv)
    calibrated = pd.read_csv(calibrated_csv)

    length = min(len(measured_frame), len(calibrated))
    if n_rows is not None:
        length = min(length, n_rows)

    cvrmse_values: Dict[str, float] = {}
    nmbe_values: Dict[str, float] = {}
    for meter in config.target_meters:
        for frame, label in ((measured_frame, "measured"), (calibrated, "calibrated")):
            if meter not in frame.columns:
                raise KeyError(f"Meter {meter!r} is absent from the {label} run output.")
        measured = measured_frame[meter].to_numpy()[:length]
        simulated = calibrated[meter].to_numpy()[:length]
        cvrmse_values[meter] = cvrmse(measured, simulated, p=config.cvrmse_p)
        nmbe_values[meter] = nmbe(measured, simulated, p=config.cvrmse_p)

        logger.info(
            "%s / %s / %s: CVRMSE = %.2f%% (threshold %.0f%%), NMBE = %.2f%%",
            model,
            experiment,
            meter,
            cvrmse_values[meter],
            config.hourly_threshold_percent,
            nmbe_values[meter],
        )

    return CalibrationAccuracy(
        experiment=experiment,
        model=model,
        cvrmse_percent=cvrmse_values,
        nmbe_percent=nmbe_values,
    )


def results_table(
    results: Sequence[CalibrationAccuracy], threshold_percent: float
) -> pd.DataFrame:
    """Assemble the per-experiment results behind Figs. 16 and 17."""
    rows: List[dict] = []
    for result in results:
        rows.extend(result.to_rows(threshold_percent))
    return pd.DataFrame(rows)
