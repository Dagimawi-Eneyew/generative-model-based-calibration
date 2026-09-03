"""Resolution of every filesystem path used by the pipeline.

No absolute path is hard-coded anywhere in this package.  A value is resolved
from the first source that supplies it:

    1. an explicit keyword argument (populated from a CLI flag)
    2. an environment variable
    3. ``configs/paths.yaml``   (git-ignored; copy from ``paths.example.yaml``)
    4. a built-in default

The resulting :class:`ProjectPaths` object also owns the layout of the data
root, so the stage scripts never build paths by string concatenation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
#: Repository root: .../src/genphysical/paths.py -> .../
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
ASSETS_DIR = REPO_ROOT / "assets"

#: The user's machine-local overrides, and the template it is copied from.
PATHS_FILE = CONFIG_DIR / "paths.yaml"
PATHS_TEMPLATE = CONFIG_DIR / "paths.example.yaml"

ENV_DATA_ROOT = "GENPHYSICAL_DATA_ROOT"
ENV_ENERGYPLUS_DIR = "ENERGYPLUS_DIR"


class PathConfigurationError(RuntimeError):
    """Raised when a required path is missing or does not exist.

    Carries an actionable message naming the setting, the environment variable
    and the CLI flag that can supply it, rather than a bare ``FileNotFoundError``.
    """


def _load_paths_file() -> dict:
    """Read ``configs/paths.yaml`` if the user has created one."""
    if PATHS_FILE.exists():
        with PATHS_FILE.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def _resolve(path_like: os.PathLike | str) -> Path:
    """Expand ``~`` and make a path absolute relative to the repository root.

    Relative paths in ``paths.yaml`` (``./data``) are interpreted against the
    repository root, so the defaults behave the same wherever the scripts are
    invoked from.
    """
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """All directories and files the pipeline reads from or writes to.

    Attributes are plain :class:`pathlib.Path` objects; the ``ensure_*`` helpers
    create output directories on demand and the ``require_*`` helpers validate
    inputs with a readable error.
    """

    data_root: Path
    idf_dir: Path
    weather_dir: Path
    baseline_idf: Path
    train_weather: Path
    test_weather: Path
    energyplus_dir: Optional[Path]
    idd_file: Optional[Path]

    # -- generated-data layout ---------------------------------------------
    # A single place that defines where each stage writes, so the scripts stay
    # free of path arithmetic.

    @property
    def samples_dir(self) -> Path:
        """LHS design matrices (stage 01/02)."""
        return self.data_root / "01_samples"

    @property
    def modified_idf_dir(self) -> Path:
        """The 400 per-sample IDF variants used for the training batch."""
        return self.data_root / "02_modified_idf"

    @property
    def simulation_dir(self) -> Path:
        """Raw EnergyPlus output, one sub-directory per run."""
        return self.data_root / "03_simulations"

    @property
    def merged_dir(self) -> Path:
        """Merged and column-selected simulation output (train/test CSVs)."""
        return self.data_root / "04_merged"

    @property
    def dataset_dir(self) -> Path:
        """Augmented, standardised arrays plus the fitted scaler (stage 03)."""
        return self.data_root / "05_datasets"

    @property
    def model_dir(self) -> Path:
        """Trained calibrator checkpoints and their frozen configs (stage 04)."""
        return self.data_root / "06_models"

    @property
    def results_dir(self) -> Path:
        """Predictions, metric tables and timings (stages 05/06)."""
        return self.data_root / "07_results"

    @property
    def figures_dir(self) -> Path:
        """Publication figures (stage 07)."""
        return self.data_root / "08_figures"

    @property
    def train_merged_csv(self) -> Path:
        return self.merged_dir / "train_simulated.csv"

    @property
    def test_merged_csv(self) -> Path:
        return self.merged_dir / "test_simulated.csv"

    # -- helpers ------------------------------------------------------------

    def ensure_dirs(self, *paths: Path) -> None:
        """Create the given directories (and their parents) if absent."""
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    def require_energyplus(self) -> Path:
        """Return the validated EnergyPlus IDD file.

        Raises
        ------
        PathConfigurationError
            If the EnergyPlus directory or the IDD file cannot be found.  Only
            the EnergyPlus-facing stages call this, so the rest of the pipeline
            runs without EnergyPlus installed.
        """
        if self.energyplus_dir is None:
            raise PathConfigurationError(
                "EnergyPlus location is not configured.\n"
                "  Set it with --energyplus-dir, the "
                f"{ENV_ENERGYPLUS_DIR} environment variable, or the "
                "'energyplus_dir' key in configs/paths.yaml.\n"
                f"  Start from the template: cp {PATHS_TEMPLATE} {PATHS_FILE}"
            )
        if not self.energyplus_dir.is_dir():
            raise PathConfigurationError(
                f"EnergyPlus directory does not exist: {self.energyplus_dir}\n"
                "  Check 'energyplus_dir' in configs/paths.yaml."
            )
        idd = self.idd_file or (self.energyplus_dir / "Energy+.idd")
        if not idd.is_file():
            raise PathConfigurationError(
                f"Energy+.idd not found at: {idd}\n"
                "  Set 'idd_file' in configs/paths.yaml if your EnergyPlus "
                "installation keeps the data dictionary elsewhere."
            )
        return idd

    def require_file(self, path: Path, what: str) -> Path:
        """Validate that an input file exists, naming it in the error."""
        if not path.is_file():
            raise PathConfigurationError(f"{what} not found: {path}")
        return path


def get_paths(
    data_root: Optional[str] = None,
    energyplus_dir: Optional[str] = None,
    idd_file: Optional[str] = None,
) -> ProjectPaths:
    """Build the :class:`ProjectPaths` for this invocation.

    Parameters
    ----------
    data_root, energyplus_dir, idd_file:
        Explicit overrides, normally taken straight from CLI flags.  ``None``
        means "fall back to the environment, then ``configs/paths.yaml``, then
        the built-in default".
    """
    file_cfg = _load_paths_file()

    resolved_data_root = _resolve(
        data_root
        or os.environ.get(ENV_DATA_ROOT)
        or file_cfg.get("data_root")
        or "./data"
    )

    ep_dir_value = (
        energyplus_dir
        or os.environ.get(ENV_ENERGYPLUS_DIR)
        or file_cfg.get("energyplus_dir")
    )
    resolved_ep_dir = _resolve(ep_dir_value) if ep_dir_value else None

    idd_value = idd_file or file_cfg.get("idd_file")
    resolved_idd = _resolve(idd_value) if idd_value else None

    idf_dir = _resolve(file_cfg.get("idf_dir") or ASSETS_DIR / "idf")
    weather_dir = _resolve(file_cfg.get("weather_dir") or ASSETS_DIR / "weather")

    return ProjectPaths(
        data_root=resolved_data_root,
        idf_dir=idf_dir,
        weather_dir=weather_dir,
        baseline_idf=idf_dir
        / file_cfg.get("baseline_idf", "baseline_model_atlanta.idf"),
        train_weather=weather_dir
        / file_cfg.get(
            "train_weather",
            "USA_GA_Atlanta-Hartsfield-Jackson-Intl-AP.722190_AMY_2021.epw",
        ),
        test_weather=weather_dir
        / file_cfg.get(
            "test_weather",
            "USA_GA_Atlanta-Hartsfield-Jackson-Intl-AP.722190_AMY_2022.epw",
        ),
        energyplus_dir=resolved_ep_dir,
        idd_file=resolved_idd,
    )


def add_path_arguments(parser) -> None:
    """Attach the shared path flags to an :mod:`argparse` parser.

    Every stage script calls this so the flags are spelled identically
    everywhere.
    """
    group = parser.add_argument_group("paths")
    group.add_argument(
        "--data-root",
        default=None,
        help=(
            "Root directory for all generated data. Overrides "
            f"${ENV_DATA_ROOT} and configs/paths.yaml (default: ./data)."
        ),
    )
    group.add_argument(
        "--energyplus-dir",
        default=None,
        help=(
            "EnergyPlus installation directory. Overrides "
            f"${ENV_ENERGYPLUS_DIR} and configs/paths.yaml."
        ),
    )
    group.add_argument(
        "--idd-file",
        default=None,
        help="Path to Energy+.idd (default: <energyplus-dir>/Energy+.idd).",
    )


def paths_from_args(args) -> ProjectPaths:
    """Build :class:`ProjectPaths` from a parsed argparse namespace."""
    return get_paths(
        data_root=getattr(args, "data_root", None),
        energyplus_dir=getattr(args, "energyplus_dir", None),
        idd_file=getattr(args, "idd_file", None),
    )
