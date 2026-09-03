"""Affine coupling block (Section 2.2 of the paper).

An affine coupling block splits its input, transforms one half with a scale and
a translation predicted from the other half, and leaves that other half
untouched.  Because the untransformed half is what produced the scale and
shift, the whole map is invertible in closed form and its Jacobian is
triangular, so the log-determinant is just the sum of the log scales
(Eqs. 2 and 3).

This module builds the two sub-networks ``s(.)`` and ``t(.)`` of one block.  In
a *conditional* coupling block (Fig. 1) the conditioning vector ``c`` is
concatenated onto the sub-network input, which is why they are built as a single
Keras model taking ``[masked latent | condition]`` and emitting ``(s, t)``.

Table 1 fixes the activations: ReLU on the hidden layers, linear on ``t(.)`` and
tanh on ``s(.)``.  The tanh bound on the scale keeps ``exp(s)`` in a numerically
safe range, which is what makes a deep stack of blocks trainable at all.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

from ..config import ArchitectureConfig


def AffineCouplingBlock(
    input_dim: int,
    output_dim: int,
    num_layers: int = 2,
    num_neurons: int = 64,
    hidden_activation: str = "relu",
    scale_activation: str = "tanh",
    translation_activation: str = "linear",
    kernel_initializer: str = "glorot_normal",
    l2_regularization: float = 1e-4,
    name: str | None = None,
) -> keras.Model:
    """Build the ``s(.)`` / ``t(.)`` sub-networks of one coupling block.

    Parameters
    ----------
    input_dim:
        Width of the concatenated ``[masked latent | condition]`` input.
    output_dim:
        Width of the latent vector being transformed (9 here).
    num_layers:
        Hidden layers in each sub-network (Table 2: "Number of layers").
    num_neurons:
        Units per hidden layer (Table 2: "Number of neurons").
    hidden_activation, scale_activation, translation_activation:
        Activations from Table 1.
    kernel_initializer:
        Table 1 specifies Xavier/Glorot initialisation.
    l2_regularization:
        The ``tau`` of Eqs. 4 and 8, applied to every weight matrix in the block.
    name:
        Optional Keras model name.

    Returns
    -------
    keras.Model
        Maps ``(batch, input_dim)`` to ``[s, t]``, each ``(batch, output_dim)``.
    """
    regularizer = regularizers.l2(l2_regularization)
    inputs = keras.layers.Input(shape=(input_dim,), name="coupling_input")

    def subnetwork(output_activation: str, prefix: str) -> tf.Tensor:
        """Stack `num_layers` hidden layers, then project to `output_dim`."""
        hidden = inputs
        for layer_index in range(num_layers):
            hidden = keras.layers.Dense(
                num_neurons,
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=regularizer,
                name=f"{prefix}_hidden_{layer_index}",
            )(hidden)
        return keras.layers.Dense(
            output_dim,
            activation=output_activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=regularizer,
            name=f"{prefix}_out",
        )(hidden)

    # s(.) - the log-scale.  tanh-bounded so exp(s) stays in [e^-1, e^1].
    scale = subnetwork(scale_activation, "scale")
    # t(.) - the translation, unbounded.
    translation = subnetwork(translation_activation, "translation")

    return keras.Model(inputs=inputs, outputs=[scale, translation], name=name)


def build_coupling_blocks(config: ArchitectureConfig, latent_dim: int) -> list:
    """Build the full stack of coupling blocks described by ``config``."""
    return [
        AffineCouplingBlock(
            input_dim=config.subnet_input_dim,
            output_dim=latent_dim,
            num_layers=config.num_layers,
            num_neurons=config.num_neurons,
            hidden_activation=config.hidden_activation,
            scale_activation=config.scale_activation,
            translation_activation=config.translation_activation,
            kernel_initializer=config.kernel_initializer,
            l2_regularization=config.l2_regularization,
            name=f"coupling_block_{block_index}",
        )
        for block_index in range(config.num_coupling_blocks)
    ]
