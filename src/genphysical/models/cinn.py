"""Baseline calibrator: a conditional invertible neural network (cINN).

Implements the standard cINN of Section 2.3, used throughout Section 5 as the
comparison point for DECI-Net.  The coupling sub-networks are conditioned
directly on the 14 raw observed model outputs ``y_o``, so the model has to cope
with sensor noise and dropouts on its own - which, as Section 5.9 reports, is
where it loses ground to DECI-Net.

Training minimises Eq. 4:

    L_cINN = (1/N) Σ_i [ ½‖f(x_i ; c_i, θ)‖² - log|J_i| ] + τ‖θ‖²

The first term is the exact negative log-likelihood under a standard normal
base (see :meth:`ConditionalRealNVP.negative_log_likelihood`); the ``τ‖θ‖²``
term is the L2 weight regularisation attached to every Dense layer in the
coupling blocks, which Keras adds to the loss through ``self.losses``.

Input layout, per row (23 columns):

    [ 0: 9]   x     standardised unobserved model inputs
    [ 9:23]   y_o   standardised observed model outputs (the conditioning input)
"""

from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf

from ..config import ArchitectureConfig
from ..constants import N_OBSERVED_OUTPUTS, N_UNOBSERVED_INPUTS
from .flow import ConditionalRealNVP


class ConditionalINN(ConditionalRealNVP):
    """Conditional invertible neural network conditioned on raw observations."""

    def __init__(self, config: ArchitectureConfig, **kwargs):
        if config.conditioning != "raw_observations":
            raise ValueError(
                "ConditionalINN requires conditioning='raw_observations'; got "
                f"{config.conditioning!r}."
            )
        super().__init__(config, latent_dim=N_UNOBSERVED_INPUTS, **kwargs)
        self.extra_trackers: Dict[str, tf.keras.metrics.Metric] = {}

    # -- row layout ----------------------------------------------------------

    def split_batch(self, data: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Split a packed row into ``(x, y_o)``."""
        return (
            data[:, :N_UNOBSERVED_INPUTS],
            data[:, N_UNOBSERVED_INPUTS : N_UNOBSERVED_INPUTS + N_OBSERVED_OUTPUTS],
        )

    def condition_from_observations(self, observations: tf.Tensor) -> tf.Tensor:
        """The conditioning vector ``c``: the observations themselves.

        DECI-Net overrides this with an encoded representation; keeping the same
        method name lets the inference code treat both models identically.
        """
        return observations

    # -- loss ----------------------------------------------------------------

    def compute_losses(self, data: tf.Tensor):
        """Eq. 4: negative log-likelihood plus the L2 weight penalty."""
        inputs, observations = self.split_batch(data)
        condition = self.condition_from_observations(observations)

        nll = self.negative_log_likelihood(inputs, condition)
        # Keras collects the kernel_regularizer penalties of every coupling
        # layer here; adding them realises the tau*||theta||^2 term of Eq. 4.
        weight_penalty = tf.add_n(self.losses) if self.losses else 0.0
        return nll + weight_penalty, {"nll": nll}

    # -- inference -----------------------------------------------------------

    def call(self, inputs, training=False):
        """Keras forward pass: return the latent ``z`` for a packed batch.

        Provided so ``model(batch)`` and ``model.summary()`` behave sensibly.
        Calibration goes through :meth:`ConditionalRealNVP.inverse` instead - see
        :mod:`genphysical.evaluation.predict`.
        """
        x, observations = self.split_batch(inputs)
        latent, _ = self.forward(x, self.condition_from_observations(observations))
        return latent
