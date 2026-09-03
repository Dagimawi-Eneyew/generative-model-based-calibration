"""EnergyPlus-facing stages of the GenPhysiCal framework.

Covers Sections 4.2-4.3.2 / 5.3.1-5.3.2 of the paper:

    sampling      Latin Hypercube design over the influential model inputs
    idf_tools     editing the baseline IDF; building the Schedule:File variant
    runner        batch execution of EnergyPlus through eppy
    postprocess   merging run output and selecting the 9 + 14 model columns

Importing this subpackage does not import ``eppy``; the modules that need it do
so lazily, so the rest of the pipeline runs without EnergyPlus installed.
"""

from .sampling import latin_hypercube_samples, save_samples

__all__ = ["latin_hypercube_samples", "save_samples"]
