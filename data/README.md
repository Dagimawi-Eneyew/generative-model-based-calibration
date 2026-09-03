# `data/` — generated artefacts

Everything in this directory is produced by the pipeline scripts; nothing here
is tracked by git (see `.gitignore`). Point `data_root` somewhere with room to
spare — a full reproduction is roughly 30–60 GB, dominated by the raw EnergyPlus
output of the 400 training simulations.

The layout is owned by `src/genphysical/paths.py`, so no script builds these
paths by hand:

| Directory | Written by | Contents |
|---|---|---|
| `01_samples/` | 01, 02 | Latin Hypercube design matrices (`train_samples.csv`, `test_samples.csv`) |
| `02_modified_idf/` | 01, 02 | One IDF per training sample; the `Schedule:File`-driven test model |
| `03_simulations/` | 01, 02 | Raw EnergyPlus output, one sub-directory per run |
| `04_merged/` | 01, 02 | Merged, column-selected simulation output (`train_simulated.csv`, `test_simulated.csv`) |
| `05_datasets/` | 03 | Augmented arrays (`datasets.npz`) and the fitted scalers |
| `06_models/` | 04 | Trained checkpoints, each with the `model_config.yaml` that produced it |
| `07_results/` | 05, 06 | Predictions, `metrics.csv`, `inference_time.json`, `model_calibration_accuracy.csv`, re-simulations |
| `08_figures/` | 07 | Rendered figures |

To keep generated data elsewhere, set `data_root` in `configs/paths.yaml`, export
`GENPHYSICAL_DATA_ROOT`, or pass `--data-root` to any script.
