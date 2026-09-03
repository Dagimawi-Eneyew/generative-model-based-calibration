"""Evaluation of the calibrator models (Sections 4.5-4.6 and 5.5-5.11).

Section 4.5 asks for four things, and each has a module here:

    predict.py            posterior sampling (Algorithm 1) and inference time
    metrics.py            point-estimate accuracy, probability calibration,
                          sharpness, CRPS and CVRMSE
    model_calibration.py  re-simulating the building energy model with the
                          estimated inputs, to measure calibration accuracy
    plots.py              Figures 7-17
"""

from .metrics import (
    calibration_curve,
    calibration_error,
    crps,
    cvrmse,
    nmbe,
    rmse,
    sharpness,
)
from .predict import PosteriorPredictions, measure_inference_time, predict_posterior

__all__ = [
    "PosteriorPredictions",
    "calibration_curve",
    "calibration_error",
    "crps",
    "cvrmse",
    "measure_inference_time",
    "nmbe",
    "predict_posterior",
    "rmse",
    "sharpness",
]
