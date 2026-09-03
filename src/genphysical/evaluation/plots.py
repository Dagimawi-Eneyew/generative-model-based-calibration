"""Publication figures.

Reproduces the result figures of Section 5:

    Figs. 7-9    :func:`plot_density_comparison`    predicted vs actual densities
    Figs. 10-12  :func:`plot_calibration`           probability calibration
    Figs. 13-15  :func:`plot_sharpness`             prediction-interval widths
    Figs. 16-17  :func:`plot_cvrmse`                model calibration accuracy

One figure is produced per experiment, comparing the baseline cINN with
DECI-Net.  The style is plain matplotlib, so no LaTeX installation is required.
Colours and markers are kept consistent across all figures so the two models
are always recognisable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: these are written to file, never shown

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants import UNOBSERVED_INPUT_LABELS, UNOBSERVED_INPUT_UNITS
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

#: One colour per model, used in every figure.
MODEL_COLOURS = {"cinn": "#B4553C", "decinet": "#2E5E8C"}
MODEL_LABELS = {"cinn": "cINN (baseline)", "decinet": "DECI-Net"}
ACTUAL_COLOUR = "#3B3B3B"

_FIGURE_DPI = 300


def _apply_style() -> None:
    """A compact, print-friendly style close to the paper's figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": _FIGURE_DPI,
            "savefig.bbox": "tight",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.2,
        }
    )


def _label_with_units(index: int) -> str:
    return f"{UNOBSERVED_INPUT_LABELS[index]} [{UNOBSERVED_INPUT_UNITS[index]}]"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info("Wrote figure %s", path)
    return path


def _kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian kernel density estimate, with a flat fallback for degenerate input."""
    from scipy import stats

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.ptp(values) <= 0:
        return np.zeros_like(grid)
    try:
        return stats.gaussian_kde(values)(grid)
    except np.linalg.LinAlgError:  # pragma: no cover - singular covariance
        return np.zeros_like(grid)


# ---------------------------------------------------------------------------
# Figures 7-9: density of the predicted point estimates
# ---------------------------------------------------------------------------
def plot_density_comparison(
    actual: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    output_path: str | Path,
    title: Optional[str] = None,
) -> Path:
    """Compare the distribution of predicted point estimates with the truth.

    Reproduces Figs. 7-9: a 3x3 grid of kernel density estimates, one panel per
    unobserved model input, overlaying the actual distribution with each model's
    predicted means.  Section 5.9.1 reads these panels for over-estimated peaks
    and for distributions that drift outside the true support.

    Parameters
    ----------
    actual:
        ``(n_points, 9)`` ground truth, in physical units.
    predictions:
        Mapping of model name to ``(n_points, 9)`` point estimates.
    output_path:
        Destination file (``.pdf`` or ``.png``).
    title:
        Optional figure title, typically the experiment label.
    """
    _apply_style()
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 6.0))

    for variable, axis in enumerate(axes.ravel()):
        series = [np.asarray(actual)[:, variable]] + [
            np.asarray(values)[:, variable] for values in predictions.values()
        ]
        finite = np.concatenate([column[np.isfinite(column)] for column in series])
        grid = np.linspace(np.min(finite), np.max(finite), 256)

        axis.plot(
            grid,
            _kde(np.asarray(actual)[:, variable], grid),
            color=ACTUAL_COLOUR,
            linestyle="--",
            label="Actual",
        )
        for model, values in predictions.items():
            axis.plot(
                grid,
                _kde(np.asarray(values)[:, variable], grid),
                color=MODEL_COLOURS.get(model, None),
                label=MODEL_LABELS.get(model, model),
            )
        axis.set_xlabel(_label_with_units(variable))
        axis.set_ylabel("Density" if variable % 3 == 0 else "")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97 if title else 1))
    return _save(fig, Path(output_path))


# ---------------------------------------------------------------------------
# Figures 10-12: probability calibration
# ---------------------------------------------------------------------------
def plot_calibration(
    confidence_levels: Sequence[float],
    curves: Mapping[str, np.ndarray],
    output_path: str | Path,
    title: Optional[str] = None,
) -> Path:
    """Plot empirical frequency against nominal confidence (Figs. 10-12).

    The diagonal is perfect calibration.  A curve below it means the model is
    over-confident (its intervals are too narrow for the accuracy they claim);
    above it means under-confident.

    Parameters
    ----------
    confidence_levels:
        Nominal levels ``p_j``.
    curves:
        Mapping of model name to the ``(9, n_levels)`` array returned by
        :func:`genphysical.evaluation.metrics.calibration_curve`.
    output_path:
        Destination file.
    title:
        Optional figure title.
    """
    _apply_style()
    levels = np.asarray(confidence_levels, dtype=float)
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 6.0), sharex=True, sharey=True)

    for variable, axis in enumerate(axes.ravel()):
        axis.plot(
            [0, 1], [0, 1], color=ACTUAL_COLOUR, linestyle="--", label="Ideal"
        )
        for model, curve in curves.items():
            axis.plot(
                levels,
                np.asarray(curve)[variable],
                marker="o",
                markersize=2.5,
                color=MODEL_COLOURS.get(model, None),
                label=MODEL_LABELS.get(model, model),
            )
        axis.set_title(UNOBSERVED_INPUT_LABELS[variable], fontsize=8)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        if variable >= 6:
            axis.set_xlabel("Predicted probability")
        if variable % 3 == 0:
            axis.set_ylabel("Empirical probability")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97 if title else 1))
    return _save(fig, Path(output_path))


# ---------------------------------------------------------------------------
# Figures 13-15: sharpness
# ---------------------------------------------------------------------------
def plot_sharpness(
    coverages: Sequence[float],
    widths: Mapping[str, np.ndarray],
    output_path: str | Path,
    title: Optional[str] = None,
) -> Path:
    """Plot mean prediction-interval width against nominal coverage (Figs. 13-15).

    Lower is sharper.  Read together with :func:`plot_calibration`: a sharper
    model is only better if it is still calibrated.

    Parameters
    ----------
    coverages:
        Nominal coverage rates ``1 - α``.
    widths:
        Mapping of model name to the ``(9, n_coverages)`` array returned by
        :func:`genphysical.evaluation.metrics.sharpness`.
    """
    _apply_style()
    coverage_values = np.asarray(coverages, dtype=float)
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 6.0), sharex=True)

    for variable, axis in enumerate(axes.ravel()):
        for model, width in widths.items():
            axis.plot(
                coverage_values,
                np.asarray(width)[variable],
                marker="o",
                markersize=2.5,
                color=MODEL_COLOURS.get(model, None),
                label=MODEL_LABELS.get(model, model),
            )
        axis.set_title(UNOBSERVED_INPUT_LABELS[variable], fontsize=8)
        if variable >= 6:
            axis.set_xlabel("Nominal coverage")
        axis.set_ylabel(f"Interval width [{UNOBSERVED_INPUT_UNITS[variable]}]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97 if title else 1))
    return _save(fig, Path(output_path))


# ---------------------------------------------------------------------------
# Figures 16-17: model calibration accuracy
# ---------------------------------------------------------------------------
def plot_cvrmse(
    results: pd.DataFrame,
    meter: str,
    output_path: str | Path,
    threshold_percent: float = 30.0,
    title: Optional[str] = None,
) -> Path:
    """Grouped bar chart of CVRMSE by experiment and model (Figs. 16-17).

    Parameters
    ----------
    results:
        Tidy frame from
        :func:`genphysical.evaluation.model_calibration.results_table`, with
        columns ``model``, ``experiment``, ``meter`` and ``cvrmse_percent``.
    meter:
        Which meter to plot, e.g. ``"Electricity:Facility [J](Hourly)"``.
    threshold_percent:
        ASHRAE Guideline 14 / FEMP hourly acceptance threshold, drawn as a line.
    """
    _apply_style()
    subset = results[results["meter"] == meter]
    if subset.empty:
        raise ValueError(f"No results for meter {meter!r}.")

    experiments = list(dict.fromkeys(subset["experiment"]))
    models = list(dict.fromkeys(subset["model"]))
    positions = np.arange(len(experiments))
    bar_width = 0.8 / max(len(models), 1)

    fig, axis = plt.subplots(figsize=(5.0, 3.2))
    for offset, model in enumerate(models):
        values = [
            float(
                subset[
                    (subset["model"] == model) & (subset["experiment"] == experiment)
                ]["cvrmse_percent"].iloc[0]
            )
            for experiment in experiments
        ]
        bars = axis.bar(
            positions + offset * bar_width,
            values,
            width=bar_width,
            color=MODEL_COLOURS.get(model, None),
            label=MODEL_LABELS.get(model, model),
        )
        axis.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)

    axis.axhline(
        threshold_percent,
        color=ACTUAL_COLOUR,
        linestyle="--",
        label=f"ASHRAE/FEMP hourly threshold ({threshold_percent:.0f}%)",
    )
    axis.set_xticks(positions + bar_width * (len(models) - 1) / 2)
    axis.set_xticklabels([label.replace("_", " ").title() for label in experiments])
    axis.set_ylabel("CVRMSE [%]")
    axis.legend(frameon=False, fontsize=7)
    if title:
        axis.set_title(title)
    fig.tight_layout()
    return _save(fig, Path(output_path))


# ---------------------------------------------------------------------------
# Supporting figure: the LHS design (Fig. 5)
# ---------------------------------------------------------------------------
def plot_lhs_samples(
    samples: pd.DataFrame, output_path: str | Path, title: Optional[str] = None
) -> Path:
    """3-D scatter of the sampled model-input space (Fig. 5).

    A visual check that the Latin Hypercube design fills the three-dimensional
    input space uniformly.
    """
    _apply_style()
    fig = plt.figure(figsize=(4.5, 4.0))
    axis = fig.add_subplot(111, projection="3d")

    columns = list(samples.columns[:3])
    axis.scatter(
        samples[columns[0]],
        samples[columns[1]],
        samples[columns[2]],
        s=3,
        alpha=0.5,
        color=MODEL_COLOURS["decinet"],
    )
    axis.set_xlabel(columns[0], fontsize=7)
    axis.set_ylabel(columns[1], fontsize=7)
    axis.set_zlabel(columns[2], fontsize=7)
    if title:
        axis.set_title(title)
    fig.tight_layout()
    return _save(fig, Path(output_path))
