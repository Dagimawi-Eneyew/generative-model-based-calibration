"""Running EnergyPlus, singly and in parallel batches.

The training set of Section 5.3.2 needs 400 whole-year simulations.  Running
them one after another is impractical, so they are dispatched through eppy's
``runIDFs`` helper, which fans the runs out across processes.  The test set and
the Section 5.10 re-simulation are single runs and use :func:`run_single`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _format_duration(seconds: float) -> str:
    """Render an elapsed time as ``HH:MM:SS.ss``."""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"


def launch_options(idf, output_directory: str | Path) -> Dict[str, object]:
    """Build the keyword arguments for one EnergyPlus run.

    ``ep_version`` is taken from the IDF's own ``Version`` object so the run
    matches the model rather than whatever happens to be installed, and
    ``readvars`` is enabled so EnergyPlus emits the ``eplusout.csv`` that the
    post-processing stage reads.
    """
    version_fields = str(idf.idfobjects["version"][0].Version_Identifier).split(".")
    version_fields.extend(["0"] * (3 - len(version_fields)))
    ep_version = "-".join(str(field) for field in version_fields)

    return {
        "ep_version": ep_version,
        "output_prefix": Path(idf.idfname).stem,
        "output_suffix": "C",          # eplusout.csv rather than eplusout.eso only
        "output_directory": str(output_directory),
        "readvars": True,              # produce the CSV of Output:Variable results
        "expandobjects": True,         # expand HVACTemplate:* objects
        "verbose": "q",
    }


def run_single(idf, output_directory: str | Path) -> Path:
    """Run one simulation and return the directory holding its output.

    Used for the annual test-set simulation and for the re-simulation that
    measures model calibration accuracy.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    logger.info("Running EnergyPlus -> %s", output_directory)
    started = time.time()
    idf.run(
        output_directory=str(output_directory),
        readvars=True,
        expandobjects=True,
        verbose="q",
    )
    logger.info(
        "Simulation finished in %s", _format_duration(time.time() - started)
    )
    return output_directory


def run_batch(
    idf_paths: Iterable[str | Path],
    epw_file: str | Path,
    output_root: str | Path,
    num_cores: int = 4,
) -> List[Path]:
    """Run many IDFs in parallel, one output directory per model.

    Each run writes to ``<output_root>/<idf stem>/``, which is what
    :func:`genphysical.energyplus.postprocess.merge_simulation_outputs` walks
    when it assembles the merged training CSV.

    Parameters
    ----------
    idf_paths:
        The models to run, e.g. the output of
        :func:`~genphysical.energyplus.idf_tools.generate_modified_idfs`.
    epw_file:
        Weather file used for every run.
    output_root:
        Parent directory for the per-run output directories.
    num_cores:
        Number of concurrent EnergyPlus processes.  The study used 64; pick a
        value your machine and licence can sustain.  Each run needs roughly one
        core and a few hundred MB.

    Returns
    -------
    list of pathlib.Path
        The per-run output directories, in submission order.
    """
    from eppy.runner.run_functions import runIDFs  # lazy: needs eppy

    from .idf_tools import load_idf

    idf_paths = [Path(path) for path in idf_paths]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    num_cores = max(1, min(int(num_cores), os.cpu_count() or 1))
    logger.info(
        "Running %d EnergyPlus simulations across %d processes",
        len(idf_paths),
        num_cores,
    )

    run_directories: List[Path] = []
    jobs = []
    for idf_path in idf_paths:
        idf = load_idf(idf_path, epw_file)
        run_directory = output_root / idf_path.stem
        jobs.append((idf, launch_options(idf, run_directory)))
        run_directories.append(run_directory)

    started = time.time()
    runIDFs(jobs, num_cores, debug=False)
    logger.info(
        "Batch of %d simulations completed in %s",
        len(idf_paths),
        _format_duration(time.time() - started),
    )
    return run_directories


def find_output_csv(run_directory: str | Path) -> Optional[Path]:
    """Locate the ``*.csv`` of variable results inside a run directory.

    ``readvars`` names the file after the run's ``output_prefix``
    (``<prefix>out.csv``) for batch runs and ``eplusout.csv`` for single runs,
    so both spellings are probed before falling back to a glob.  Metered and
    tabular outputs (``*meter*``, ``*Table*``, ``*sz*``) are skipped.
    """
    run_directory = Path(run_directory)
    if not run_directory.is_dir():
        return None

    for candidate in (run_directory / "eplusout.csv", run_directory / f"{run_directory.name}out.csv"):
        if candidate.is_file():
            return candidate

    excluded = ("meter", "table", "sz", "ssz", "zsz")
    matches = [
        path
        for path in sorted(run_directory.glob("*.csv"))
        if not any(token in path.name.lower() for token in excluded)
    ]
    return matches[0] if matches else None
