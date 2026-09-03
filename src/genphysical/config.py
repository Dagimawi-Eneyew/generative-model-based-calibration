"""Typed configuration objects loaded from the YAML files in ``configs/``.

Each stage of the pipeline takes one of these dataclasses rather than a loose
dictionary, so a typo in a YAML key fails immediately with a clear message
instead of falling back to a default.  Configs are also serialisable, which
lets :func:`dump_config` freeze the exact settings next to every trained
checkpoint, so the architecture never has to be restated when the checkpoint
is reloaded.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from .constants import N_OBSERVED_OUTPUTS, N_UNOBSERVED_INPUTS
from .paths import CONFIG_DIR


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or internally inconsistent."""


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------
def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a plain dictionary."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file is not a YAML mapping: {path}")
    return data


def _require(mapping: Dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key '{key}' in {context}.")
    return mapping[key]


def _unknown_keys(mapping: Dict[str, Any], known: Sequence[str], context: str) -> None:
    """Reject keys the loader does not understand, so typos surface early."""
    extra = sorted(set(mapping) - set(known))
    if extra:
        raise ConfigError(
            f"Unrecognised key(s) {extra} in {context}. "
            f"Known keys: {sorted(known)}."
        )


def dump_config(config: Any, path: str | Path) -> None:
    """Serialise a config dataclass to YAML.

    Called after training so that a checkpoint always ships with the exact
    architecture that produced it; :func:`load_model_config` reads it back.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataclasses.asdict(config), handle, sort_keys=False)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingConfig:
    """Latin Hypercube design over the influential model inputs (Section 5.3.1)."""

    occupant_density_range: tuple
    lighting_power_density_range: tuple
    equipment_power_density_range: tuple
    n_train_samples: int
    n_test_samples: int
    train_seed: int
    test_seed: int

    @property
    def parameter_names(self) -> List[str]:
        """Column names of the design matrix, in sampling order."""
        return [
            "Occupant Density [m2/person]",
            "Lighting Power Density [W/m2]",
            "Equipment Power Density [W/m2]",
        ]

    @property
    def bounds(self) -> List[tuple]:
        """(low, high) bounds in the same order as :attr:`parameter_names`."""
        return [
            self.occupant_density_range,
            self.lighting_power_density_range,
            self.equipment_power_density_range,
        ]


@dataclass(frozen=True)
class SimulationConfig:
    """Execution settings for the EnergyPlus batch (Section 5.3.2)."""

    hours_per_year: int
    num_cores: int
    convert_loads_to_kw: bool


@dataclass(frozen=True)
class AugmentationConfig:
    """Virtual to Physical Observations Approximation, VPOA (Section 5.3.3)."""

    versions: List[str]
    noise_factor: float
    min_masked: int
    max_masked: int
    protected_observations: List[int]
    train_noise_seed: int
    train_mask_seed: int
    test_noise_seed: int
    test_mask_seed: int

    def __post_init__(self) -> None:
        if not 1 <= self.min_masked <= self.max_masked <= N_OBSERVED_OUTPUTS:
            raise ConfigError(
                "augmentation: require 1 <= min_masked <= max_masked <= "
                f"{N_OBSERVED_OUTPUTS}, got "
                f"min_masked={self.min_masked}, max_masked={self.max_masked}."
            )
        n_protected = len(set(self.protected_observations))
        if self.max_masked > N_OBSERVED_OUTPUTS - n_protected:
            raise ConfigError(
                f"augmentation: max_masked={self.max_masked} exceeds the "
                f"{N_OBSERVED_OUTPUTS - n_protected} maskable observations left "
                f"after protecting {n_protected} of them."
            )
        for index in self.protected_observations:
            if not 0 <= index < N_OBSERVED_OUTPUTS:
                raise ConfigError(
                    f"augmentation: protected observation index {index} is "
                    f"outside [0, {N_OBSERVED_OUTPUTS})."
                )


@dataclass(frozen=True)
class DatasetConfig:
    """Assembly of the model-ready arrays."""

    validation_split: float
    include_clean_targets_for_decinet: bool


@dataclass(frozen=True)
class DataGenerationConfig:
    """Everything in ``configs/data_generation.yaml``."""

    sampling: SamplingConfig
    simulation: SimulationConfig
    augmentation: AugmentationConfig
    dataset: DatasetConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataGenerationConfig":
        raw = load_yaml(path)
        _unknown_keys(
            raw, ["sampling", "simulation", "augmentation", "dataset"], str(path)
        )

        sampling_raw = _require(raw, "sampling", str(path))
        ranges = _require(sampling_raw, "ranges", "sampling")
        sampling = SamplingConfig(
            occupant_density_range=tuple(ranges["occupant_density_m2_per_person"]),
            lighting_power_density_range=tuple(ranges["lighting_power_density_w_per_m2"]),
            equipment_power_density_range=tuple(
                ranges["equipment_power_density_w_per_m2"]
            ),
            n_train_samples=int(sampling_raw["n_train_samples"]),
            n_test_samples=int(sampling_raw["n_test_samples"]),
            train_seed=int(sampling_raw["train_seed"]),
            test_seed=int(sampling_raw["test_seed"]),
        )

        simulation_raw = _require(raw, "simulation", str(path))
        simulation = SimulationConfig(
            hours_per_year=int(simulation_raw["hours_per_year"]),
            num_cores=int(simulation_raw["num_cores"]),
            convert_loads_to_kw=bool(simulation_raw["convert_loads_to_kw"]),
        )

        augmentation_raw = _require(raw, "augmentation", str(path))
        augmentation = AugmentationConfig(
            versions=list(augmentation_raw["versions"]),
            noise_factor=float(augmentation_raw["noise_factor"]),
            min_masked=int(augmentation_raw["min_masked"]),
            max_masked=int(augmentation_raw["max_masked"]),
            protected_observations=list(
                augmentation_raw.get("protected_observations") or []
            ),
            train_noise_seed=int(augmentation_raw["train_noise_seed"]),
            train_mask_seed=int(augmentation_raw["train_mask_seed"]),
            test_noise_seed=int(augmentation_raw["test_noise_seed"]),
            test_mask_seed=int(augmentation_raw["test_mask_seed"]),
        )

        dataset_raw = _require(raw, "dataset", str(path))
        dataset = DatasetConfig(
            validation_split=float(dataset_raw["validation_split"]),
            include_clean_targets_for_decinet=bool(
                dataset_raw["include_clean_targets_for_decinet"]
            ),
        )

        return cls(sampling, simulation, augmentation, dataset)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AutoencoderConfig:
    """Denoising autoencoder conditioning network of DECI-Net (Section 4.4)."""

    encoder_units: List[int]
    bottleneck_size: int
    hidden_activation: str = "relu"
    output_activation: str = "linear"
    bottleneck_activation: str = "relu"
    kernel_initializer: str = "glorot_normal"
    l2_regularization: float = 1.0e-4

    def __post_init__(self) -> None:
        if not self.encoder_units:
            raise ConfigError("autoencoder: encoder_units must not be empty.")
        if self.bottleneck_size < 1:
            raise ConfigError("autoencoder: bottleneck_size must be >= 1.")


@dataclass(frozen=True)
class ArchitectureConfig:
    """Conditional invertible neural network shared by both calibrator models."""

    num_coupling_blocks: int
    num_layers: int
    num_neurons: int
    conditioning: str
    hidden_activation: str = "relu"
    translation_activation: str = "linear"
    scale_activation: str = "tanh"
    kernel_initializer: str = "glorot_normal"
    l2_regularization: float = 1.0e-4
    autoencoder: Optional[AutoencoderConfig] = None

    def __post_init__(self) -> None:
        # The alternating binary masks are built in pairs, so an odd block count
        # would silently drop the last block.
        if self.num_coupling_blocks % 2 != 0 or self.num_coupling_blocks < 2:
            raise ConfigError(
                "architecture: num_coupling_blocks must be a positive even "
                f"number (the coupling masks alternate in pairs); got "
                f"{self.num_coupling_blocks}."
            )
        if self.num_layers < 1:
            raise ConfigError("architecture: num_layers must be >= 1.")
        if self.conditioning not in ("raw_observations", "denoised_encoding"):
            raise ConfigError(
                "architecture: conditioning must be 'raw_observations' (cINN) "
                f"or 'denoised_encoding' (DECI-Net); got {self.conditioning!r}."
            )
        if self.conditioning == "denoised_encoding" and self.autoencoder is None:
            raise ConfigError(
                "architecture: conditioning='denoised_encoding' requires an "
                "'autoencoder' section."
            )

    @property
    def condition_dim(self) -> int:
        """Width of the conditioning vector fed to each coupling sub-network.

        The baseline conditions on the 14 raw observations; DECI-Net conditions
        on the autoencoder bottleneck C_e instead (Fig. 2).
        """
        if self.conditioning == "raw_observations":
            return N_OBSERVED_OUTPUTS
        return self.autoencoder.bottleneck_size  # type: ignore[union-attr]

    @property
    def subnet_input_dim(self) -> int:
        """Input width of each coupling sub-network: masked latent + condition."""
        return N_UNOBSERVED_INPUTS + self.condition_dim


@dataclass(frozen=True)
class TrainingConfig:
    """Optimiser and fitting schedule (Table 1)."""

    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str = "adam"
    early_stopping_patience: int = 5
    validation_split: float = 0.1
    restore_best_weights: bool = True
    shuffle: bool = True
    seed: int = 0
    lambda_reconstruction: float = 1.0


@dataclass(frozen=True)
class InferenceConfig:
    """Posterior sampling settings for Algorithm 1."""

    n_posterior_samples: int = 1000
    point_estimate: str = "mean"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.point_estimate not in ("mean", "median", "mode"):
            raise ConfigError(
                "inference: point_estimate must be 'mean', 'median' or 'mode'; "
                f"got {self.point_estimate!r}."
            )


@dataclass(frozen=True)
class ModelConfig:
    """A complete calibrator specification: ``configs/cinn.yaml`` or ``decinet.yaml``."""

    model: str
    architecture: ArchitectureConfig
    training: TrainingConfig
    inference: InferenceConfig

    def __post_init__(self) -> None:
        if self.model not in ("cinn", "decinet"):
            raise ConfigError(
                f"model must be 'cinn' or 'decinet'; got {self.model!r}."
            )
        expected = (
            "raw_observations" if self.model == "cinn" else "denoised_encoding"
        )
        if self.architecture.conditioning != expected:
            raise ConfigError(
                f"model '{self.model}' expects conditioning='{expected}', got "
                f"'{self.architecture.conditioning}'."
            )

    @property
    def n_input_columns(self) -> int:
        """Row width the model consumes: 23 for the cINN, 37 for DECI-Net."""
        if self.model == "cinn":
            return N_UNOBSERVED_INPUTS + N_OBSERVED_OUTPUTS
        return N_UNOBSERVED_INPUTS + 2 * N_OBSERVED_OUTPUTS

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        raw = load_yaml(path)
        _unknown_keys(
            raw, ["model", "architecture", "training", "inference"], str(path)
        )

        arch_raw = dict(_require(raw, "architecture", str(path)))
        autoencoder_raw = arch_raw.pop("autoencoder", None)
        autoencoder = (
            AutoencoderConfig(
                encoder_units=[int(u) for u in autoencoder_raw["encoder_units"]],
                bottleneck_size=int(autoencoder_raw["bottleneck_size"]),
                hidden_activation=autoencoder_raw.get("hidden_activation", "relu"),
                output_activation=autoencoder_raw.get("output_activation", "linear"),
                bottleneck_activation=autoencoder_raw.get(
                    "bottleneck_activation", "relu"
                ),
                kernel_initializer=autoencoder_raw.get(
                    "kernel_initializer", "glorot_normal"
                ),
                l2_regularization=float(
                    autoencoder_raw.get("l2_regularization", 1.0e-4)
                ),
            )
            if autoencoder_raw
            else None
        )

        architecture = ArchitectureConfig(
            num_coupling_blocks=int(arch_raw["num_coupling_blocks"]),
            num_layers=int(arch_raw["num_layers"]),
            num_neurons=int(arch_raw["num_neurons"]),
            conditioning=arch_raw["conditioning"],
            hidden_activation=arch_raw.get("hidden_activation", "relu"),
            translation_activation=arch_raw.get("translation_activation", "linear"),
            scale_activation=arch_raw.get("scale_activation", "tanh"),
            kernel_initializer=arch_raw.get("kernel_initializer", "glorot_normal"),
            l2_regularization=float(arch_raw.get("l2_regularization", 1.0e-4)),
            autoencoder=autoencoder,
        )

        train_raw = _require(raw, "training", str(path))
        training = TrainingConfig(
            epochs=int(train_raw["epochs"]),
            batch_size=int(train_raw["batch_size"]),
            learning_rate=float(train_raw["learning_rate"]),
            optimizer=train_raw.get("optimizer", "adam"),
            early_stopping_patience=int(train_raw.get("early_stopping_patience", 5)),
            validation_split=float(train_raw.get("validation_split", 0.1)),
            restore_best_weights=bool(train_raw.get("restore_best_weights", True)),
            shuffle=bool(train_raw.get("shuffle", True)),
            seed=int(train_raw.get("seed", 0)),
            lambda_reconstruction=float(train_raw.get("lambda_reconstruction", 1.0)),
        )

        infer_raw = raw.get("inference", {})
        inference = InferenceConfig(
            n_posterior_samples=int(infer_raw.get("n_posterior_samples", 1000)),
            point_estimate=infer_raw.get("point_estimate", "mean"),
            seed=int(infer_raw.get("seed", training.seed)),
        )

        return cls(
            model=_require(raw, "model", str(path)),
            architecture=architecture,
            training=training,
            inference=inference,
        )


def load_model_config(path: str | Path) -> ModelConfig:
    """Load a model configuration, accepting a bare name like ``"decinet"``.

    ``load_model_config("decinet")`` resolves to ``configs/decinet.yaml``.
    """
    path = Path(path)
    if not path.suffix:
        path = CONFIG_DIR / f"{path.name}.yaml"
    return ModelConfig.from_yaml(path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentSpec:
    """One row of Table 3: which augmented test version an experiment uses."""

    name: str
    label: str
    data_version: str


@dataclass(frozen=True)
class MetricsConfig:
    """Metric definitions (Eqs. 10-13)."""

    confidence_levels: List[float]
    calibration_definition: str
    sharpness_coverages: List[float]
    crps_estimator: str
    crps_max_samples: Optional[int] = None

    def __post_init__(self) -> None:
        if self.calibration_definition not in ("cdf", "interval"):
            raise ConfigError(
                "metrics: calibration_definition must be 'cdf' (Eq. 10) or "
                f"'interval'; got {self.calibration_definition!r}."
            )
        if self.crps_estimator not in ("ensemble", "gaussian"):
            raise ConfigError(
                "metrics: crps_estimator must be 'ensemble' or 'gaussian'; got "
                f"{self.crps_estimator!r}."
            )


@dataclass(frozen=True)
class TimingConfig:
    """Inference-time benchmark settings (Section 5.11)."""

    n_timed_observations: int = 500
    n_warmup: int = 50


@dataclass(frozen=True)
class ModelCalibrationConfig:
    """CVRMSE stage settings (Section 5.10, Eq. 14)."""

    target_meters: List[str]
    cvrmse_p: int = 1
    hourly_threshold_percent: float = 30.0
    clip_predictions_at_zero: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    """Everything in ``configs/evaluation.yaml``."""

    experiments: List[ExperimentSpec]
    metrics: MetricsConfig
    timing: TimingConfig
    model_calibration: ModelCalibrationConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvaluationConfig":
        raw = load_yaml(path)
        _unknown_keys(
            raw,
            ["experiments", "metrics", "timing", "model_calibration"],
            str(path),
        )

        experiments = [
            ExperimentSpec(
                name=item["name"],
                label=item["label"],
                data_version=item["data_version"],
            )
            for item in _require(raw, "experiments", str(path))
        ]

        metrics_raw = _require(raw, "metrics", str(path))
        metrics = MetricsConfig(
            confidence_levels=[float(v) for v in metrics_raw["confidence_levels"]],
            calibration_definition=metrics_raw.get("calibration_definition", "cdf"),
            sharpness_coverages=[
                float(v) for v in metrics_raw["sharpness_coverages"]
            ],
            crps_estimator=metrics_raw.get("crps_estimator", "ensemble"),
            crps_max_samples=(
                int(metrics_raw["crps_max_samples"])
                if metrics_raw.get("crps_max_samples")
                else None
            ),
        )

        timing_raw = raw.get("timing", {})
        timing = TimingConfig(
            n_timed_observations=int(timing_raw.get("n_timed_observations", 500)),
            n_warmup=int(timing_raw.get("n_warmup", 50)),
        )

        mc_raw = _require(raw, "model_calibration", str(path))
        model_calibration = ModelCalibrationConfig(
            target_meters=list(mc_raw["target_meters"]),
            cvrmse_p=int(mc_raw.get("cvrmse_p", 1)),
            hourly_threshold_percent=float(
                mc_raw.get("hourly_threshold_percent", 30.0)
            ),
            clip_predictions_at_zero=bool(mc_raw.get("clip_predictions_at_zero", True)),
        )

        return cls(experiments, metrics, timing, model_calibration)


# ---------------------------------------------------------------------------
# Default config locations
# ---------------------------------------------------------------------------
DEFAULT_DATA_GENERATION_CONFIG = CONFIG_DIR / "data_generation.yaml"
DEFAULT_EVALUATION_CONFIG = CONFIG_DIR / "evaluation.yaml"
