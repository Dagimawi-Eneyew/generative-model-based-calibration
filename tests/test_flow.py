"""Tests for the conditional normalizing flow and the two calibrator models.

The flow is only a valid generative model if it really is a bijection and if the
log-determinant it reports really is the log-determinant of its own Jacobian.
Both are checked directly here, which is the strongest available guard against
the coupling algebra of Eqs. 2-3 being subtly wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow is required for model tests")

from genphysical.config import (  # noqa: E402
    ArchitectureConfig,
    AutoencoderConfig,
    ConfigError,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from genphysical.constants import (  # noqa: E402
    N_OBSERVED_OUTPUTS,
    N_UNOBSERVED_INPUTS,
)
from genphysical.models.builder import build_model, compile_model  # noqa: E402
from genphysical.models.flow import alternating_masks  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: deliberately small models, so the suite stays fast
# ---------------------------------------------------------------------------
def _cinn_config(num_blocks: int = 4) -> ModelConfig:
    return ModelConfig(
        model="cinn",
        architecture=ArchitectureConfig(
            num_coupling_blocks=num_blocks,
            num_layers=2,
            num_neurons=16,
            conditioning="raw_observations",
        ),
        training=TrainingConfig(epochs=1, batch_size=32, learning_rate=1e-3, seed=0),
        inference=InferenceConfig(n_posterior_samples=16, seed=0),
    )


def _decinet_config(num_blocks: int = 4, bottleneck: int = 6) -> ModelConfig:
    return ModelConfig(
        model="decinet",
        architecture=ArchitectureConfig(
            num_coupling_blocks=num_blocks,
            num_layers=2,
            num_neurons=16,
            conditioning="denoised_encoding",
            autoencoder=AutoencoderConfig(
                encoder_units=[32, 16], bottleneck_size=bottleneck
            ),
        ),
        training=TrainingConfig(epochs=1, batch_size=32, learning_rate=1e-3, seed=0),
        inference=InferenceConfig(n_posterior_samples=16, seed=0),
    )


@pytest.fixture(params=["cinn", "decinet"])
def model_and_config(request):
    config = _cinn_config() if request.param == "cinn" else _decinet_config()
    model = build_model(config)
    compile_model(model, config)
    # Force lazy weight creation.
    model(tf.zeros((2, config.n_input_columns), dtype=tf.float32), training=False)
    return model, config


def _random_batch(config: ModelConfig, n_rows: int = 8, seed: int = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(size=(n_rows, config.n_input_columns)).astype(np.float32)


# ---------------------------------------------------------------------------
# Coupling masks
# ---------------------------------------------------------------------------
def test_masks_alternate_and_are_complementary():
    masks = alternating_masks(6, N_UNOBSERVED_INPUTS)
    assert masks.shape == (6, N_UNOBSERVED_INPUTS)
    for pair_start in range(0, 6, 2):
        np.testing.assert_allclose(masks[pair_start] + masks[pair_start + 1], 1.0)


def test_masks_require_an_even_block_count():
    with pytest.raises(ValueError, match="even"):
        alternating_masks(5, N_UNOBSERVED_INPUTS)


def test_config_rejects_an_odd_block_count():
    with pytest.raises(ConfigError, match="even"):
        ArchitectureConfig(
            num_coupling_blocks=7,
            num_layers=2,
            num_neurons=16,
            conditioning="raw_observations",
        )


# ---------------------------------------------------------------------------
# The bijection
# ---------------------------------------------------------------------------
def test_forward_and_inverse_are_exact_inverses(model_and_config):
    """f(f^-1(z; c); c) == z, to floating-point precision.

    If this fails, the flow is not a normalizing flow: the change-of-variables
    formula of Eq. 1 no longer applies and neither the likelihood nor the
    posterior sampling means anything.
    """
    model, config = model_and_config
    batch = _random_batch(config, n_rows=16)

    observations = tf.convert_to_tensor(
        batch[:, N_UNOBSERVED_INPUTS : N_UNOBSERVED_INPUTS + N_OBSERVED_OUTPUTS]
    )
    condition = model.condition_from_observations(observations)

    latent = tf.convert_to_tensor(
        np.random.default_rng(1).normal(size=(16, N_UNOBSERVED_INPUTS)).astype(np.float32)
    )

    estimated = model.inverse(latent, condition)
    recovered, _ = model.forward(estimated, condition)

    np.testing.assert_allclose(recovered.numpy(), latent.numpy(), atol=1e-4, rtol=1e-4)


def test_inverse_then_forward_round_trips_from_the_data_side(model_and_config):
    """The round trip also holds starting from x, not just from z."""
    model, config = model_and_config
    batch = _random_batch(config, n_rows=12, seed=2)

    x = tf.convert_to_tensor(batch[:, :N_UNOBSERVED_INPUTS])
    observations = tf.convert_to_tensor(
        batch[:, N_UNOBSERVED_INPUTS : N_UNOBSERVED_INPUTS + N_OBSERVED_OUTPUTS]
    )
    condition = model.condition_from_observations(observations)

    latent, _ = model.forward(x, condition)
    recovered = model.inverse(latent, condition)

    np.testing.assert_allclose(recovered.numpy(), x.numpy(), atol=1e-4, rtol=1e-4)


def test_log_determinant_matches_the_numerical_jacobian(model_and_config):
    """The reported log|det dz/dx| must equal the true Jacobian determinant.

    Checked against automatic differentiation of the forward map. An error here
    would bias the likelihood of Eq. 4 / Eq. 8 without breaking invertibility,
    so it would be invisible to the round-trip tests above.
    """
    model, config = model_and_config
    batch = _random_batch(config, n_rows=4, seed=3)

    observations = tf.convert_to_tensor(
        batch[:, N_UNOBSERVED_INPUTS : N_UNOBSERVED_INPUTS + N_OBSERVED_OUTPUTS]
    )
    condition = model.condition_from_observations(observations)
    x = tf.convert_to_tensor(batch[:, :N_UNOBSERVED_INPUTS])

    _, reported = model.forward(x, condition)

    for row in range(batch.shape[0]):
        single_x = tf.convert_to_tensor(x.numpy()[row : row + 1])
        single_condition = tf.convert_to_tensor(condition.numpy()[row : row + 1])

        with tf.GradientTape() as tape:
            tape.watch(single_x)
            latent, _ = model.forward(single_x, single_condition)
        jacobian = tape.jacobian(latent, single_x)
        jacobian = tf.reshape(
            jacobian, (N_UNOBSERVED_INPUTS, N_UNOBSERVED_INPUTS)
        ).numpy()

        sign, log_abs_det = np.linalg.slogdet(jacobian)
        assert sign > 0, "The flow must be orientation preserving."
        assert log_abs_det == pytest.approx(float(reported.numpy()[row]), abs=1e-3)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def test_losses_are_finite_and_reduce_during_training(model_and_config):
    """A few gradient steps on easy data must lower the loss."""
    model, config = model_and_config
    generator = np.random.default_rng(4)
    n_rows = 512

    # Observations that carry real information about the inputs, so the flow has
    # something learnable to latch onto.
    inputs = generator.normal(size=(n_rows, N_UNOBSERVED_INPUTS))
    observations = np.tile(inputs[:, :1], (1, N_OBSERVED_OUTPUTS)) + 0.1 * generator.normal(
        size=(n_rows, N_OBSERVED_OUTPUTS)
    )
    blocks = [inputs, observations]
    if config.model == "decinet":
        blocks.append(observations)
    batch = np.concatenate(blocks, axis=1).astype(np.float32)

    initial, _ = model.compute_losses(tf.convert_to_tensor(batch))
    assert np.isfinite(float(initial))

    history = model.fit(batch, batch_size=128, epochs=3, verbose=0)
    losses = history.history["loss"]
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]


def test_decinet_reports_both_loss_components():
    """Eq. 8 has two learnable terms; both must be tracked."""
    config = _decinet_config()
    model = build_model(config)
    compile_model(model, config)
    batch = _random_batch(config, n_rows=32, seed=5)

    total, components = model.compute_losses(tf.convert_to_tensor(batch))
    assert set(components) == {"nll", "reconstruction_loss"}
    assert np.isfinite(float(total))
    assert float(components["reconstruction_loss"]) >= 0.0


def test_decinet_lambda_scales_the_reconstruction_term():
    """The lambda of Eq. 8 must actually change the objective."""
    config = _decinet_config()
    batch = _random_batch(config, n_rows=32, seed=6)

    model = build_model(config)
    compile_model(model, config)
    total_default, components = model.compute_losses(tf.convert_to_tensor(batch))

    # Same weights, different lambda: the totals must differ by exactly
    # (new - old) * reconstruction_loss.
    model.lambda_reconstruction = 5.0
    total_scaled, _ = model.compute_losses(tf.convert_to_tensor(batch))

    expected_difference = 4.0 * float(components["reconstruction_loss"])
    assert float(total_scaled) - float(total_default) == pytest.approx(
        expected_difference, rel=1e-5
    )


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------
def test_cinn_conditions_on_the_raw_observations():
    config = _cinn_config()
    model = build_model(config)
    observations = tf.convert_to_tensor(
        np.random.default_rng(7).normal(size=(5, N_OBSERVED_OUTPUTS)).astype(np.float32)
    )
    condition = model.condition_from_observations(observations)
    np.testing.assert_allclose(condition.numpy(), observations.numpy())
    assert config.architecture.condition_dim == N_OBSERVED_OUTPUTS


def test_decinet_conditions_on_the_bottleneck():
    bottleneck = 6
    config = _decinet_config(bottleneck=bottleneck)
    model = build_model(config)
    observations = tf.convert_to_tensor(
        np.random.default_rng(8).normal(size=(5, N_OBSERVED_OUTPUTS)).astype(np.float32)
    )
    condition = model.condition_from_observations(observations)
    assert condition.shape == (5, bottleneck)
    assert config.architecture.condition_dim == bottleneck
    # The narrower conditioning vector is the point: it shrinks every coupling
    # sub-network relative to conditioning on all 14 observations.
    assert config.architecture.subnet_input_dim == N_UNOBSERVED_INPUTS + bottleneck


def test_decinet_inference_needs_only_the_encoder():
    """Section 4.6: the decoder is discarded at inference time."""
    config = _decinet_config()
    model = build_model(config)
    observations = tf.convert_to_tensor(
        np.random.default_rng(9).normal(size=(3, N_OBSERVED_OUTPUTS)).astype(np.float32)
    )
    # Conditioning must work from 14 observations alone - no clean-target block.
    condition = model.condition_from_observations(observations)
    latent = model.sample_latent(3)
    estimated = model.inverse(latent, condition)
    assert estimated.shape == (3, N_UNOBSERVED_INPUTS)
    assert np.all(np.isfinite(estimated.numpy()))


# ---------------------------------------------------------------------------
# Configuration guards
# ---------------------------------------------------------------------------
def test_model_and_conditioning_must_agree():
    with pytest.raises(ConfigError, match="expects conditioning"):
        ModelConfig(
            model="cinn",
            architecture=ArchitectureConfig(
                num_coupling_blocks=4,
                num_layers=2,
                num_neurons=16,
                conditioning="denoised_encoding",
                autoencoder=AutoencoderConfig(encoder_units=[8], bottleneck_size=4),
            ),
            training=TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3),
            inference=InferenceConfig(),
        )


def test_decinet_requires_an_autoencoder_section():
    with pytest.raises(ConfigError, match="requires an 'autoencoder' section"):
        ArchitectureConfig(
            num_coupling_blocks=4,
            num_layers=2,
            num_neurons=16,
            conditioning="denoised_encoding",
        )
