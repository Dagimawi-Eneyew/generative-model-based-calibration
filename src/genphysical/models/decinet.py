"""DECI-Net: Denoised-Encodings-Conditioned Invertible Neural Network.

The calibrator model proposed in Section 4.4 of the paper.  A denoising
autoencoder sits in front of the conditional invertible neural network: instead
of conditioning the coupling blocks on the raw observations ``y_o``, DECI-Net
conditions them on the autoencoder's bottleneck ``C_e``.

Two things follow from that (Section 4.4):

* **Robustness.**  The autoencoder is trained to reconstruct the *clean*
  observations from corrupted ones, so ``C_e`` is a representation that survives
  sensor noise and dropouts.  Section 5.9: "This network effectively mapped
  noisy data into latent conditions that were more resistant to noise and
  missing data."
* **Capacity.**  ``C_e`` is much narrower than ``y_o`` (8 versus 14 by default),
  which shrinks each coupling sub-network and reduces the depth needed for a
  high-dimensional conditioning input - the vanishing/exploding gradient concern
  raised via Köhler et al.

Training minimises the combined loss of Eq. 8:

    L_DECI-Net = (1/N) Σ_i [ ½‖f(x_i ; C_e_i, θ)‖² - log|J_i|
                             + λ (1/D)‖y_o_i - ŷ_o_i‖² ] + τ‖θ‖²

i.e. the flow's negative log-likelihood plus ``λ`` times the autoencoder's mean
squared reconstruction error, trained end to end in one pass.  The paper notes
this "avoids a two-step training process of similar architectures".

Input layout, per row (37 columns):

    [ 0: 9]   x         standardised unobserved model inputs
    [ 9:23]   y_o       standardised observations, possibly noisy/masked
                        (the autoencoder's *input*)
    [23:37]   y_o clean  standardised uncorrupted observations
                        (the autoencoder's reconstruction *target*)

The clean block is training supervision only.  At inference the decoder is
discarded entirely - Section 4.6: "During the inference stage of the DECI-Net
model, the decoder is no longer needed, and only the encoder part of the
denoising autoencoder will be used."
"""

from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

from ..config import ArchitectureConfig, AutoencoderConfig
from ..constants import N_OBSERVED_OUTPUTS, N_UNOBSERVED_INPUTS
from .flow import ConditionalRealNVP


def DenoisingAutoencoder(
    config: AutoencoderConfig,
    input_dim: int = N_OBSERVED_OUTPUTS,
    name: str = "denoising_autoencoder",
) -> keras.Model:
    """Build the denoising autoencoder conditioning network (Section 2.4).

    A symmetric encoder/decoder around a narrow bottleneck.  The encoder units
    are given by ``config.encoder_units`` and the decoder mirrors them.

    Parameters
    ----------
    config:
        Autoencoder section of the model config.
    input_dim:
        Width of the observation vector (14).
    name:
        Keras model name.  Kept stable, because it becomes part of the variable
        names inside a saved checkpoint.

    Returns
    -------
    keras.Model
        Maps ``(batch, 14)`` to ``[reconstruction, bottleneck]``.  The
        reconstruction feeds the MSE term of Eq. 8; the bottleneck is the
        conditioning vector ``C_e``.
    """
    regularizer = regularizers.l2(config.l2_regularization)
    inputs = keras.layers.Input(shape=(input_dim,), name="observations")

    hidden = inputs
    for layer_index, units in enumerate(config.encoder_units):
        hidden = keras.layers.Dense(
            units,
            activation=config.hidden_activation,
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=regularizer,
            name=f"encoder_{layer_index}",
        )(hidden)

    # C_e: the compressed, noise-robust representation used for conditioning.
    bottleneck = keras.layers.Dense(
        config.bottleneck_size,
        activation=config.bottleneck_activation,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=regularizer,
        name="bottleneck",
    )(hidden)

    hidden = bottleneck
    for layer_index, units in enumerate(reversed(config.encoder_units)):
        hidden = keras.layers.Dense(
            units,
            activation=config.hidden_activation,
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=regularizer,
            name=f"decoder_{layer_index}",
        )(hidden)

    reconstruction = keras.layers.Dense(
        input_dim,
        activation=config.output_activation,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=regularizer,
        name="reconstruction",
    )(hidden)

    return keras.Model(
        inputs=inputs, outputs=[reconstruction, bottleneck], name=name
    )


class DECINet(ConditionalRealNVP):
    """Conditional INN conditioned on a denoising autoencoder's bottleneck."""

    def __init__(
        self,
        config: ArchitectureConfig,
        lambda_reconstruction: float = 1.0,
        **kwargs,
    ):
        if config.conditioning != "denoised_encoding":
            raise ValueError(
                "DECINet requires conditioning='denoised_encoding'; got "
                f"{config.conditioning!r}."
            )
        if config.autoencoder is None:
            raise ValueError("DECINet requires an autoencoder configuration.")

        super().__init__(config, latent_dim=N_UNOBSERVED_INPUTS, **kwargs)

        self.autoencoder = DenoisingAutoencoder(config.autoencoder)
        # lambda of Eq. 8.  Stored as a constant so it is captured in the graph
        # and reported in the frozen config saved next to the checkpoint.
        self.lambda_reconstruction = float(lambda_reconstruction)

        self.reconstruction_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.nll_tracker = keras.metrics.Mean(name="nll")
        self.extra_trackers: Dict[str, keras.metrics.Metric] = {
            "reconstruction_loss": self.reconstruction_tracker,
            "nll": self.nll_tracker,
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.nll_tracker, self.reconstruction_tracker]

    # -- row layout ----------------------------------------------------------

    def split_batch(
        self, data: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Split a packed row into ``(x, y_o, y_o_clean)``.

      
        """
        observations_end = N_UNOBSERVED_INPUTS + N_OBSERVED_OUTPUTS
        return (
            data[:, :N_UNOBSERVED_INPUTS],
            data[:, N_UNOBSERVED_INPUTS:observations_end],
            data[:, observations_end : observations_end + N_OBSERVED_OUTPUTS],
        )

    def condition_from_observations(self, observations: tf.Tensor) -> tf.Tensor:
        """Encode observations into the conditioning vector ``C_e``.

        Only the encoder half of the autoencoder is exercised, which is exactly
        what Algorithm 1 prescribes at inference time.
        """
        _, bottleneck = self.autoencoder(observations)
        return bottleneck

    def encode_and_reconstruct(
        self, observations: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Return ``(reconstruction, bottleneck)`` for a batch of observations."""
        reconstruction, bottleneck = self.autoencoder(observations)
        return reconstruction, bottleneck

    # -- loss ----------------------------------------------------------------

    def compute_losses(self, data: tf.Tensor):
        """Eq. 8: NLL + lambda * reconstruction MSE + the L2 weight penalty."""
        inputs, observations, clean_observations = self.split_batch(data)

        reconstruction, condition = self.encode_and_reconstruct(observations)

        # First two terms of Eq. 8, via the exact flow log-likelihood.
        nll = self.negative_log_likelihood(inputs, condition)

        # Third term: mean squared error across the feature dimension between the
        # decoder output and the *uncorrupted* observations - Section 4.4, "a
        # reconstruction loss for the denoising autoencoder, calculated as the
        # mean squared error across the feature dimension of the autoencoder's
        # input".
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_mean(tf.square(clean_observations - reconstruction), axis=1)
        )

        weight_penalty = tf.add_n(self.losses) if self.losses else 0.0
        total = nll + self.lambda_reconstruction * reconstruction_loss + weight_penalty
        return total, {"nll": nll, "reconstruction_loss": reconstruction_loss}

    # -- inference -----------------------------------------------------------

    def call(self, inputs, training=False):
        """Keras forward pass: return the latent ``z`` for a packed batch."""
        x = inputs[:, :N_UNOBSERVED_INPUTS]
        observations = inputs[
            :, N_UNOBSERVED_INPUTS : N_UNOBSERVED_INPUTS + N_OBSERVED_OUTPUTS
        ]
        latent, _ = self.forward(x, self.condition_from_observations(observations))
        return latent
