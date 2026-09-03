# GenPhysiCal / DECI-Net

**Reference implementation of the paper:**

> Dagimawi D. Eneyew, Miriam A. M. Capretz and Girma T. Bitsuamlak,
> **"Continuous model calibration framework for smart-building digital twin:
> A generative model-based approach"**,
> *Applied Energy* **375** (2024) 124080.
> [doi:10.1016/j.apenergy.2024.124080](https://doi.org/10.1016/j.apenergy.2024.124080)
> · Open access (CC BY 4.0) 

This repository contains a reference implementation of the method proposed in the above paper: the
EnergyPlus data generation, the two generative calibrator models, the three
evaluation experiments, and the metrics and figures.

---

## What problem this solves

A smart-building digital twin only stays useful if its physics-based model keeps
matching the real building. Keeping them aligned means continuously estimating
model inputs that nobody measures — how many people are in each zone, what the
lighting and plug loads are — from the sensors that *are* available. Existing
approaches search for those inputs iteratively (genetic algorithms, MCMC), which
is far too slow to run every hour and gives no uncertainty estimate.

**GenPhysiCal** turns the calibration problem around. Instead of searching, it
trains a *generative inverse model* once, offline, on simulations of the building
energy model. At runtime that calibrator maps sensor readings directly to a full
posterior distribution over the unobserved inputs — in 0.043 s, with calibrated
uncertainty, and without evaluating the physics model at all.

**DECI-Net** (Denoised-Encodings-Conditioned Invertible Neural Network) is the
calibrator architecture the paper proposes. It is a conditional invertible neural
network whose coupling blocks are conditioned not on raw sensor readings but on
the bottleneck of a denoising autoencoder, trained jointly end to end. That is
what lets it keep working when sensors are noisy or have failed outright — the
condition real buildings are actually in.

![Continuous calibration of a smart-building digital twin: sensor observations from the physical building are mapped by the generative calibrator model to distributions over the unobserved inputs of the physics-based model, whose simulated outputs are compared back against the measurements](assets/figures/graphicalAbstractCalibration.png)

*The calibration loop. Sensor observations `y_o` from the physical building are
mapped by the generative calibrator to a distribution over the unobserved inputs
`x*` of the physics-based model, whose simulated outputs `y` are compared back
against the measurements.*

![The GenPhysiCal framework and the DECI-Net architecture: contribution C1 covers problem formulation, input selection and data preparation; C2 the calibrator model architecture and training, where a denoising autoencoder encodes the observations into a latent vector that conditions a chain of conditional coupling blocks; C3 the continuous physics-model calibration procedure](assets/figures/latestframework_merged.png)

*The GenPhysiCal framework (C1–C3) and the DECI-Net architecture. The denoising
autoencoder compresses the observations `y_o` into the latent vector `C_e`, which
conditions the chain of conditional coupling blocks mapping between the
unobserved model inputs `x*` and the standard normal latent `z`. Only the shaded
encoder and coupling path are active at inference.*

### The case study

A US DOE small-office reference building in Atlanta, GA — five conditioned zones
plus an attic, 511 m², gas heating and electric cooling — simulated in
EnergyPlus 23.1 with measured (AMY) weather.

- **9 unobserved model inputs** `x`: occupant count, lighting load and plug load
  for the Core zone and for two pairs of geometrically identical perimeter zones.
- **14 observed model outputs** `y_o`: air temperature and relative humidity for
  six zones, plus facility-level electricity and gas.

Three experiments, matching the sensing conditions a real building presents:

| Experiment | Test data | Question |
|---|---|---|
| 1 | Clean | Can the calibrator learn the inverse mapping at all? |
| 2 | Noisy | Does it survive measurement noise? |
| 3 | Noisy + missing | Does it survive sensors failing outright? |

---

## Quick start

### 1. Install

```bash
git clone <this-repository>
cd generative-model-based-model-calibration

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .
```

Python 3.9–3.11. TensorFlow is pinned below 2.16 because Keras 3 does not support
the custom `train_step` these models use.

For the data-generation and model-calibration stages you also need
[EnergyPlus 23.1](https://energyplus.net/downloads) and the extra Python
dependencies:

```bash
pip install -e ".[energyplus]"
```

### 2. Point the code at your machine

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

Edit that copy: set `data_root` (where generated data goes) and `energyplus_dir`.
Both can also come from `--data-root` / `--energyplus-dir` or from
`$GENPHYSICAL_DATA_ROOT` / `$ENERGYPLUS_DIR`. `configs/paths.yaml` is
git-ignored, so your local paths stay local.

### 3. Try it without EnergyPlus first

The modelling stages run on synthetic stand-in data, so you can check the whole
chain in a couple of minutes before committing to hours of simulation:

```bash
python scripts/03_build_datasets.py --smoke
python scripts/04_train.py --model cinn    --epochs 3
python scripts/04_train.py --model decinet --epochs 3
python scripts/05_evaluate.py --max-test-rows 200
python scripts/07_make_figures.py
```

### 4. Full reproduction

```bash
python scripts/01_generate_training_data.py --num-cores 32   # 400 annual simulations
python scripts/02_generate_test_data.py                      # 1 annual simulation
python scripts/03_build_datasets.py
python scripts/04_train.py --model cinn
python scripts/04_train.py --model decinet
python scripts/05_evaluate.py
python scripts/06_model_calibration.py                       # 6 annual simulations
python scripts/07_make_figures.py
```

| Stage | Needs EnergyPlus | Rough cost |
|---|---|---|
| `01_generate_training_data.py` | yes | 400 annual runs — hours; the dominant cost |
| `02_generate_test_data.py` | yes | 2 annual runs — about a minute |
| `03_build_datasets.py` | no | minutes; memory-hungry at full size |
| `04_train.py` | no | GPU strongly recommended |
| `05_evaluate.py` | no | ~10 min per model for 3 × 8760 posteriors |
| `06_model_calibration.py` | yes | 6 annual runs — minutes |
| `07_make_figures.py` | no | seconds |

Budget roughly 30–60 GB under `data_root`, dominated by raw EnergyPlus output.
Multi-GPU training: add `--multi-gpu` to stage 04.

---

## Repository layout

```
configs/            YAML configuration — the only place hyperparameters live
  paths.example.yaml    machine-local paths; copy to paths.yaml
  data_generation.yaml  sampling, simulation and VPOA augmentation
  cinn.yaml             baseline cINN, paper Section 5.9 hyperparameters
  decinet.yaml          DECI-Net, paper Section 5.9 hyperparameters
  decinet_as_trained.yaml  an alternative DECI-Net configuration
  evaluation.yaml       experiments, metric definitions, CVRMSE settings

assets/
  idf/                DOE small-office prototype for Atlanta
  weather/            TMY + AMY 2019-2022 weather files

src/genphysical/
  constants.py        the 9 inputs, the 14 observations, building geometry
  paths.py            path resolution; no absolute path is hard-coded anywhere
  config.py           typed configuration objects
  energyplus/         sampling, IDF editing, batch runs, post-processing
  data/               VPOA augmentation and dataset assembly
  models/             coupling blocks, the flow, cINN, DECI-Net
  evaluation/         Algorithm 1, metrics, CVRMSE re-simulation, figures

scripts/            00-07, one per pipeline stage
tests/              invertibility, log-determinant, augmentation, metrics
docs/               paper-to-code map
data/               everything generated (git-ignored)
```

---

## How the method is implemented

**The flow** (`models/flow.py`). A stack of conditional affine coupling blocks
defines a bijection between the 9 unobserved inputs and a standard normal latent.
`forward` is the normalizing direction used for the exact likelihood (Eq. 1);
`inverse` is the generative direction used to solve the calibration problem
(Algorithm 1). Because the map is bijective with a triangular Jacobian, training
maximises an exact log-likelihood — no variational bound, and no evaluation of
the physics model in the loop.

**The baseline cINN** (`models/cinn.py`) conditions the coupling sub-networks on
the 14 raw observations and minimises Eq. 4.

**DECI-Net** (`models/decinet.py`) conditions them on an 8-dimensional
autoencoder bottleneck instead, and minimises Eq. 8 — the flow's negative
log-likelihood plus λ times the autoencoder's reconstruction error against the
*clean* observations — in a single end-to-end pass. The autoencoder sees
corrupted input and is asked for clean output, which is what makes the condition
robust. At inference the decoder is discarded entirely.

**VPOA augmentation** (`data/augmentation.py`) turns each simulated dataset into
three: clean, noisy, and noisy-with-failed-sensors.

**Closing the loop** (`evaluation/model_calibration.py`). The estimated inputs
are written back into the building energy model through an hourly
`Schedule:File`, the year is re-simulated, and the resulting facility meters are
compared against the measurements with CVRMSE (Eq. 14).

---



## License and acknowledgements

Code is released under the [MIT License](LICENSE). The bundled IDF derives from
the public-domain DOE/ORNL prototype building models; the paper itself is CC BY
4.0.

This work was supported by the Natural Sciences and Engineering Research Council
of Canada (NSERC) Discovery through Western University under grants
RGPIN-2021-04161 and RGPIN-2018-05454, and by the Carbon Solutions grant at
Western University, Canada.
