"""Console logging shared by every stage script.

A logger gives at-a-glance progress with timestamps, a consistent format, and
the option of a log file for the long unattended stages (the 400-run EnergyPlus
batch and the full training run).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO, log_file: Optional[str | Path] = None
) -> None:
    """Configure the root logger once, at process start.

    Parameters
    ----------
    level:
        Threshold for the console handler.
    log_file:
        Optional path; when given, the same records are also appended there.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Re-running in the same interpreter (e.g. a notebook) must not stack
    # duplicate handlers.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # TensorFlow's C++ layer is extremely chatty about unused GPU kernels.
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)


class ProgressLogger:
    """Log progress through a long loop at a fixed fraction of completion.

    Emits one line per reporting boundary, so the output stays readable when
    redirected to a file.

    Examples
    --------
    >>> progress = ProgressLogger(logger, total=8760, label="Predicting")
    >>> for i in range(8760):
    ...     progress.update(i)          # doctest: +SKIP
    """

    def __init__(self, logger: logging.Logger, total: int, label: str, every: int = 10):
        self._logger = logger
        self._total = max(int(total), 1)
        self._label = label
        # Report roughly every `every` percent, at least once per item.
        self._stride = max(self._total * every // 100, 1)

    def update(self, index: int) -> None:
        """Log if ``index`` lands on a reporting boundary or is the final item."""
        if index % self._stride == 0 or index == self._total - 1:
            percent = 100.0 * (index + 1) / self._total
            self._logger.info(
                "%s: %d/%d (%.0f%%)", self._label, index + 1, self._total, percent
            )
