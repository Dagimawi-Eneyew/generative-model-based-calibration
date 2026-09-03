"""Turning raw EnergyPlus output into the model-ready table.

Implements the tail of Section 5.3.2: merge the per-run CSVs, keep only the 9
unobserved model inputs and the 14 physically observable outputs, and convert
the lighting and plug-load columns from W to kW so the numbers match the units
used in the paper's tables and figures.

The result is a single CSV with exactly the 23 canonical columns of
:mod:`genphysical.constants`, in canonical order, which is all the downstream
stages ever read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from ..constants import (
    ALL_MODEL_COLUMNS,
    LOAD_COLUMN_INDICES,
    OBSERVED_OUTPUT_COLUMNS,
    UNOBSERVED_INPUT_COLUMNS,
    W_PER_KW,
)
from ..utils.logging_utils import ProgressLogger, get_logger
from .runner import find_output_csv

logger = get_logger(__name__)


def select_model_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the 23 canonical columns, in canonical order.

    Raises
    ------
    KeyError
        If the simulation output is missing any required variable, naming the
        missing columns so the IDF's ``Output:Variable`` list can be fixed.
    """
    missing = [column for column in ALL_MODEL_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(
            "Simulation output is missing "
            f"{len(missing)} required column(s):\n  " + "\n  ".join(missing)
        )
    return frame.loc[:, ALL_MODEL_COLUMNS].copy()


def convert_loads_to_kw(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the six lighting/plug-load columns from W to kW.

    Occupant counts and the facility meters are left untouched: the meters stay
    in joules, which is how CVRMSE is computed in Section 5.10 (the metric is
    scale-invariant, so the unit does not affect the reported percentage).
    """
    frame = frame.copy()
    load_columns = [UNOBSERVED_INPUT_COLUMNS[index] for index in LOAD_COLUMN_INDICES]
    frame[load_columns] = frame[load_columns] / W_PER_KW
    return frame


def load_simulation_output(
    csv_path: str | Path,
    n_rows: Optional[int] = None,
    to_kilowatts: bool = True,
) -> pd.DataFrame:
    """Read one run's CSV and reduce it to the canonical 23 columns.

    Parameters
    ----------
    csv_path:
        An ``eplusout.csv`` produced with ``readvars=True``.
    n_rows:
        Truncate to this many leading rows.  EnergyPlus prepends design-day
        results when sizing runs are enabled, and passing ``8760`` keeps only
        the annual run-period rows.
    to_kilowatts:
        Apply :func:`convert_loads_to_kw`.
    """
    frame = pd.read_csv(csv_path)
    frame = select_model_columns(frame)
    if n_rows is not None:
        if len(frame) < n_rows:
            raise ValueError(
                f"{csv_path} holds {len(frame)} rows, fewer than the {n_rows} "
                "requested. Check that the run period covers a full year."
            )
        frame = frame.iloc[-n_rows:].reset_index(drop=True)
    if to_kilowatts:
        frame = convert_loads_to_kw(frame)
    return frame


def merge_simulation_outputs(
    run_directories: Iterable[str | Path],
    output_csv: str | Path,
    n_rows_per_run: Optional[int] = None,
    to_kilowatts: bool = True,
) -> Path:
    """Concatenate every run's output into one training CSV.

    The 400 training runs contribute 8760 rows each, giving the 3,504,000-row
    training table of Section 5.3.2.  Runs are appended in the order given, and
    a run whose output is missing is logged and skipped rather than aborting a
    multi-hour batch.

    Parameters
    ----------
    run_directories:
        Per-run output directories from
        :func:`~genphysical.energyplus.runner.run_batch`.
    output_csv:
        Destination file.
    n_rows_per_run:
        Rows to keep from each run, typically 8760.
    to_kilowatts:
        Convert load columns from W to kW.

    Returns
    -------
    pathlib.Path
        The written CSV.
    """
    run_directories = [Path(directory) for directory in run_directories]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    progress = ProgressLogger(logger, len(run_directories), "Merging simulation output")
    total_rows = 0
    skipped: List[str] = []
    wrote_header = False

    # Appended run by run: the full training table does not fit comfortably in
    # memory as a list of DataFrames (3.5 M rows x 23 float columns).
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        for position, run_directory in enumerate(run_directories):
            csv_path = find_output_csv(run_directory)
            if csv_path is None:
                skipped.append(run_directory.name)
                progress.update(position)
                continue

            frame = load_simulation_output(
                csv_path, n_rows=n_rows_per_run, to_kilowatts=to_kilowatts
            )
            frame.to_csv(handle, index=False, header=not wrote_header)
            wrote_header = True
            total_rows += len(frame)
            progress.update(position)

    if skipped:
        logger.warning(
            "No output CSV found for %d run(s); they were skipped: %s",
            len(skipped),
            ", ".join(skipped[:10]) + (" ..." if len(skipped) > 10 else ""),
        )
    if not wrote_header:
        raise RuntimeError(
            f"None of the {len(run_directories)} runs produced an output CSV. "
            "Check the EnergyPlus error files (*.err) in the run directories."
        )

    logger.info("Wrote %d merged rows to %s", total_rows, output_csv)
    return output_csv


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column summary statistics, for a quick sanity check after merging."""
    stats = frame.describe().T[["count", "mean", "std", "min", "max"]]
    stats.insert(
        0,
        "role",
        ["unobserved input"] * len(UNOBSERVED_INPUT_COLUMNS)
        + ["observed output"] * len(OBSERVED_OUTPUT_COLUMNS),
    )
    return stats
