"""Dataset preparation: VPOA augmentation and model-ready array assembly.

Implements Section 4.3.3 / 5.3.3 of the paper.
"""

from .augmentation import (
    AugmentedDataset,
    apply_noise,
    build_augmented_dataset,
    random_missing_mask,
)
from .datasets import (
    PreparedData,
    build_model_matrix,
    load_prepared,
    prepare_datasets,
    save_prepared,
)

__all__ = [
    "AugmentedDataset",
    "apply_noise",
    "build_augmented_dataset",
    "random_missing_mask",
    "PreparedData",
    "build_model_matrix",
    "load_prepared",
    "prepare_datasets",
    "save_prepared",
]
