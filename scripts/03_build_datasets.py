#!/usr/bin/env python
"""Stage 03 - VPOA augmentation and dataset assembly.

Implements Section 4.3.3 / 5.3.3, "Virtual to Physical Observations
Approximation".  Simulated observations are noiseless and complete; real
building measurements are neither.  This stage produces three versions of both
the training and the test data - clean, noisy, and noisy-with-missing-sensors -
so the calibrator learns representations that survive the sensing conditions it
will actually meet, and so Experiments 1-3 of Table 3 have data to run on.

It also fits the standardisation, on the clean training data only, and saves it
alongside the arrays so every later stage shares exactly one transform.

No EnergyPlus needed; this reads the merged CSVs from stages 01 and 02.

Examples
--------
    python scripts/03_build_datasets.py

    # exercise the pipeline without EnergyPlus, on synthetic data
    python scripts/03_build_datasets.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from genphysical.config import (  # noqa: E402
    DEFAULT_DATA_GENERATION_CONFIG,
    DataGenerationConfig,
)
from genphysical.data.datasets import (  # noqa: E402
    prepare_datasets,
    save_prepared,
    synthetic_merged_frame,
)
from genphysical.paths import add_path_arguments, paths_from_args  # noqa: E402
from genphysical.utils.logging_utils import get_logger, setup_logging  # noqa: E402

logger = get_logger("stage03")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment and standardise the simulated data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_DATA_GENERATION_CONFIG),
        help="Data-generation configuration file.",
    )
    parser.add_argument(
        "--train-csv",
        default=None,
        help="Merged training CSV (default: <data-root>/04_merged/train_simulated.csv).",
    )
    parser.add_argument(
        "--test-csv",
        default=None,
        help="Merged test CSV (default: <data-root>/04_merged/test_simulated.csv).",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Use only the first N training rows. Useful for a fast trial.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Use only the first N test rows.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Generate synthetic stand-in data instead of reading the simulated "
            "CSVs, so the pipeline can be exercised without EnergyPlus."
        ),
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=5000,
        help="Rows of synthetic training data to generate with --smoke.",
    )
    add_path_arguments(parser)
    return parser.parse_args()


def _write_smoke_inputs(paths, n_rows: int) -> tuple:
    """Write synthetic merged CSVs so the rest of the pipeline is exercisable."""
    logger.warning(
        "--smoke: generating %d rows of SYNTHETIC data. These results are a "
        "pipeline check only and say nothing about the paper's findings.",
        n_rows,
    )
    paths.ensure_dirs(paths.merged_dir)
    train_csv = paths.merged_dir / "train_simulated_smoke.csv"
    test_csv = paths.merged_dir / "test_simulated_smoke.csv"
    synthetic_merged_frame(n_rows=n_rows, seed=0).to_csv(train_csv, index=False)
    synthetic_merged_frame(n_rows=max(n_rows // 5, 200), seed=1).to_csv(
        test_csv, index=False
    )
    return train_csv, test_csv


def main() -> int:
    setup_logging()
    args = parse_args()

    paths = paths_from_args(args)
    config = DataGenerationConfig.from_yaml(args.config)

    if args.smoke:
        train_csv, test_csv = _write_smoke_inputs(paths, args.smoke_rows)
    else:
        train_csv = Path(args.train_csv) if args.train_csv else paths.train_merged_csv
        test_csv = Path(args.test_csv) if args.test_csv else paths.test_merged_csv
        for path, label in ((train_csv, "Training"), (test_csv, "Test")):
            if not path.is_file():
                logger.error(
                    "%s data not found: %s\n"
                    "  Run stages 01 and 02 first, or pass --smoke to try the "
                    "pipeline on synthetic data.",
                    label,
                    path,
                )
                return 1

    data = prepare_datasets(
        train_csv=train_csv,
        test_csv=test_csv,
        config=config,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )

    output_dir = paths.dataset_dir
    save_prepared(data, output_dir)

    # A short summary makes an accidental unit or scaling error obvious.
    summary = pd.DataFrame(
        {
            "split": ["train", "test"],
            "rows_per_version": [data.train.n_per_version, data.test.n_per_version],
            "total_rows": [len(data.train), len(data.test)],
        }
    )
    logger.info("Dataset summary:\n%s", summary.to_string(index=False))
    logger.info(
        "Observation mean/std after standardisation: %.4f / %.4f "
        "(clean training block should be ~0 / ~1)",
        float(data.train.clean_observations[: data.train.n_per_version].mean()),
        float(data.train.clean_observations[: data.train.n_per_version].std()),
    )

    logger.info("Prepared datasets written to %s", output_dir)
    logger.info(
        "Next: python scripts/04_train.py --model cinn "
        "&& python scripts/04_train.py --model decinet"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
