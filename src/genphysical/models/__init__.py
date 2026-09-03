"""Generative calibrator models.

    coupling.py   affine coupling block (Section 2.2, Eqs. 2-3)
    flow.py       conditional RealNVP core shared by both models
    cinn.py       baseline conditional invertible neural network (Section 2.3, Eq. 4)
    decinet.py    DECI-Net: cINN conditioned on a denoising autoencoder (Section 4.4, Eq. 8)

:func:`build_model` is the single entry point used by both the training and the
evaluation stages, so a checkpoint can never be reloaded into a different
architecture than the one that produced it.
"""

from .builder import build_model, load_trained_model
from .cinn import ConditionalINN
from .coupling import AffineCouplingBlock
from .decinet import DECINet, DenoisingAutoencoder

__all__ = [
    "AffineCouplingBlock",
    "ConditionalINN",
    "DECINet",
    "DenoisingAutoencoder",
    "build_model",
    "load_trained_model",
]
