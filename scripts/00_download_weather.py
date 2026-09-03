#!/usr/bin/env python
"""Stage 00 - fetch Actual Meteorological Year (AMY) weather files.

The case study is driven by measured weather for Atlanta Hartsfield-Jackson
International Airport (WMO 722190) rather than a typical year, so that the
simulated operational data reflects a real year's conditions.

The five files the paper used are already bundled in ``assets/weather/``, so
this script is only needed to extend the study to other years or other stations.
It wraps :mod:`diyepw`, which downloads NOAA ISD observations and assembles them
into EPW files, interpolating or imputing short gaps.

Examples
--------
Rebuild the bundled 2019-2022 Atlanta files::

    python scripts/00_download_weather.py --years 2019 2020 2021 2022 --wmo 722190

Fetch a different station::

    python scripts/00_download_weather.py --years 2023 --wmo 725300 --output-dir ./my_weather
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when the repository has not been pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genphysical.paths import add_path_arguments, paths_from_args  # noqa: E402
from genphysical.utils.logging_utils import get_logger, setup_logging  # noqa: E402

logger = get_logger("stage00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download AMY weather files with diyepw.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2019, 2020, 2021, 2022],
        help="Calendar years to assemble.",
    )
    parser.add_argument(
        "--wmo",
        type=int,
        nargs="+",
        default=[722190],
        help="WMO station identifiers (722190 = Atlanta Hartsfield-Jackson Intl AP).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the EPW files (default: assets/weather).",
    )
    parser.add_argument(
        "--max-records-to-interpolate",
        type=int,
        default=10,
        help="Longest run of missing records filled by interpolation.",
    )
    parser.add_argument(
        "--max-records-to-impute",
        type=int,
        default=25,
        help="Longest run of missing records filled by imputation.",
    )
    parser.add_argument(
        "--max-missing-amy-rows",
        type=int,
        default=5,
        help="Reject a year with more than this many irreparable rows.",
    )
    add_path_arguments(parser)
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    paths = paths_from_args(args)

    try:
        import diyepw
    except ImportError:
        logger.error(
            "diyepw is not installed.\n"
            "  Install it with:  pip install 'genphysical[energyplus]'"
        )
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else paths.weather_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Assembling AMY files for station(s) %s, year(s) %s -> %s",
        args.wmo,
        args.years,
        output_dir,
    )
    # Downloads raw ISD observations, then writes one EPW per (station, year).
    diyepw.create_amy_epw_files_for_years_and_wmos(
        args.years,
        args.wmo,
        max_records_to_interpolate=args.max_records_to_interpolate,
        max_records_to_impute=args.max_records_to_impute,
        max_missing_amy_rows=args.max_missing_amy_rows,
        allow_downloads=True,
        amy_epw_dir=str(output_dir),
    )

    written = sorted(output_dir.glob("*.epw"))
    logger.info("%d EPW file(s) now present in %s", len(written), output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
