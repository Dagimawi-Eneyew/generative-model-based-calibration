"""GenPhysiCal - Generative Model-based Calibration Framework for Physics-based
Models in Smart-Building Digital Twins.

Reference implementation of

    D. D. Eneyew, M. A. M. Capretz and G. T. Bitsuamlak,
    "Continuous model calibration framework for smart-building digital twin:
     A generative model-based approach",
    Applied Energy 375 (2024) 124080.
    https://doi.org/10.1016/j.apenergy.2024.124080

The package is organised along the stages of the framework in Fig. 2 of the
paper:

    genphysical.energyplus   Sections 4.2-4.3.2  sampling, simulation, post-processing
    genphysical.data         Section 4.3.3       VPOA augmentation, dataset assembly
    genphysical.models       Sections 2, 4.4     cINN baseline and DECI-Net
    genphysical.evaluation   Sections 4.5-4.6, 5 metrics, Algorithm 1, CVRMSE

See docs/paper_to_code.md for a per-equation map.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
