"""Conditional RealNVP core shared by both calibrator models.

Implements the normalizing flow of Sections 2.1-2.3.  A stack of conditional
affine coupling blocks defines a bijection between the 9 unobserved model inputs
``x`` and a latent ``z`` with a standard normal base distribution, conditioned on
``c``:

    forward   (normalizing)   z = f(x ; c)      used for the likelihood, Eq. 4
    inverse   (generative)    x = f^-1(z ; c)   used for calibration, Algorithm 1

Because the map is bijective, the density of ``x`` follows from the change of
variables theorem (Eq. 1) and the training loss is an exact negative
log-likelihood - no variational bound and no likelihood-model evaluation, which
is what Section 4.4 means by "exact likelihood computation, bypassing the need
for posterior distribution approximation".

**Masking convention.**  Eq. 2 describes a block as splitting its input into two
halves ``i1`` and ``i2``.  The implementation uses the equivalent binary-mask
formulation of Dinh et al.: a fixed 0/1 vector ``b`` selects the passthrough
components, ``b ⊙ x`` conditions the sub-networks, and the affine transform is
applied to the complement ``(1 - b) ⊙ x``.  Consecutive blocks alternate ``b``
and ``1 - b``, so every component is transformed by half the blocks and
conditions the other half.  Two chained blocks with complementary masks are
exactly the two coupling layers of Eq. 2.

One block, generative direction:

    s, t   = net([b ⊙ x, c])                      # masked to the complement
    x_out  = b ⊙ x + (1 - b) ⊙ (x ⊙ exp(s) + t)
    log|J| = Σ (1 - b) ⊙ s

and its exact inverse:

    s, t   = net([b ⊙ x, c])
    x_out  = b ⊙ x + (1 - b) ⊙ ((x - t) ⊙ exp(-s))
    log|J| = -Σ (1 - b) ⊙ s

with the blocks traversed in reverse order.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from ..config import ArchitectureConfig
from ..constants import N_UNOBSERVED_INPUTS
from .coupling import build_coupling_blocks


def alternating_masks(num_blocks: int, latent_dim: int) -> np.ndarray:
    """Build the alternating binary masks for a stack of coupling blocks.

    Block ``2k`` uses ``[0, 1, 0, 1, ...]`` and block ``2k + 1`` its complement,
    so each latent component is transformed by exactly half the blocks.

    Returns
    -------
    numpy.ndarray
        ``(num_blocks, latent_dim)`` of 0.0/1.0.
    """
    if num_blocks % 2 != 0:
        raise ValueError(
            f"num_blocks must be even so the masks alternate in pairs; got {num_blocks}."
        )
    even = np.array([index % 2 for index in range(latent_dim)], dtype="float32")
    odd = 1.0 - even
    return np.array([even, odd] * (num_blocks // 2), dtype="float32")


class ConditionalRealNVP(keras.Model):
    """A stack of conditional affine coupling blocks over a standard normal base.

    Subclasses supply the conditioning vector: the baseline cINN passes the raw
    observations, DECI-Net passes a denoising autoencoder's bottleneck.  Everything
    below - the bijection, the log-determinant and the likelihood - is shared.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        latent_dim: int = N_UNOBSERVED_INPUTS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.latent_dim = latent_dim

        # Standard normal base distribution p_Z (Eq. 1, with a Gaussian prior on z).
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dim, dtype=tf.float32),
            scale_diag=tf.ones(latent_dim, dtype=tf.float32),
        )

        # Fixed, non-trainable coupling masks.
        self.masks = tf.constant(
            alternating_masks(config.num_coupling_blocks, latent_dim),
            dtype=tf.float32,
        )

        self.coupling_blocks: List[keras.Model] = build_coupling_blocks(
            config, latent_dim
        )

        self.loss_tracker = keras.metrics.Mean(name="loss")

    # -- the bijection -------------------------------------------------------

    def _scale_and_translation(
        self, x: tf.Tensor, condition: tf.Tensor, block_index: int
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Evaluate one block's sub-networks on the passthrough half.

        Returns ``(passthrough, masked_scale, masked_translation)``.  Both
        outputs are zeroed on the passthrough components, so adding them there
        is a no-op and the transform touches only the complement.
        """
        mask = self.masks[block_index]
        passthrough = x * mask
        scale, translation = self.coupling_blocks[block_index](
            tf.concat([passthrough, condition], axis=1)
        )
        complement = 1.0 - mask
        return passthrough, scale * complement, translation * complement

    def forward(
        self, x: tf.Tensor, condition: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Normalizing direction ``z = f(x ; c)``.

        Returns
        -------
        (tf.Tensor, tf.Tensor)
            The latent ``z`` and the log-determinant ``log|det dz/dx|``, the
            quantity that enters the change-of-variables formula of Eq. 1.
        """
        log_det = tf.zeros(tf.shape(x)[0], dtype=x.dtype)
        # Reverse order: forward undoes the generative pass block by block.
        for block_index in reversed(range(self.config.num_coupling_blocks)):
            passthrough, scale, translation = self._scale_and_translation(
                x, condition, block_index
            )
            x = passthrough + (1.0 - self.masks[block_index]) * (
                (x - translation) * tf.exp(-scale)
            )
            log_det -= tf.reduce_sum(scale, axis=1)
        return x, log_det

    def inverse(self, z: tf.Tensor, condition: tf.Tensor) -> tf.Tensor:
        """Generative direction ``x = f^-1(z ; c)``.

        This is the direction used to solve the calibration problem: draw ``z``
        from the base distribution, condition on the building's observations and
        read off a sample of the unobserved model inputs (Algorithm 1, line 6).
        """
        x = z
        for block_index in range(self.config.num_coupling_blocks):
            passthrough, scale, translation = self._scale_and_translation(
                x, condition, block_index
            )
            x = passthrough + (1.0 - self.masks[block_index]) * (
                x * tf.exp(scale) + translation
            )
        return x

    # -- likelihood ----------------------------------------------------------

    def log_likelihood(self, x: tf.Tensor, condition: tf.Tensor) -> tf.Tensor:
        """Per-sample ``log p(x | c)`` under the flow (Eq. 1, log form)."""
        z, log_det = self.forward(x, condition)
        return self.distribution.log_prob(z) + log_det

    def negative_log_likelihood(
        self, x: tf.Tensor, condition: tf.Tensor
    ) -> tf.Tensor:
        """Mean NLL over the batch - the first term of Eqs. 4 and 8.

        Note that ``-log p(x|c) = 0.5‖z‖² - log|J| + const`` for a standard
        normal base, which is Eq. 4 up to the constant ``(d/2) log 2π`` that does
        not depend on the parameters.
        """
        return -tf.reduce_mean(self.log_likelihood(x, condition))

    # -- sampling ------------------------------------------------------------

    def sample_latent(
        self, n_samples: int, seed: Optional[int] = None
    ) -> tf.Tensor:
        """Draw ``n_samples`` from the standard normal base (Algorithm 1, line 2)."""
        return self.distribution.sample(n_samples, seed=seed)

    # -- Keras plumbing ------------------------------------------------------

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        return [self.loss_tracker]

    def split_batch(self, data: tf.Tensor):
        """Split a packed training row into its parts.

        Subclasses override this to describe their own row layout.
        """
        raise NotImplementedError

    def compute_losses(self, data: tf.Tensor):
        """Return ``(total_loss, {name: value})`` for one batch."""
        raise NotImplementedError

    def train_step(self, data):
        with tf.GradientTape() as tape:
            total_loss, components = self.compute_losses(data)
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return self._update_trackers(total_loss, components)

    def test_step(self, data):
        total_loss, components = self.compute_losses(data)
        return self._update_trackers(total_loss, components)

    def _update_trackers(self, total_loss, components):
        self.loss_tracker.update_state(total_loss)
        results = {"loss": self.loss_tracker.result()}
        for name, tracker in getattr(self, "extra_trackers", {}).items():
            tracker.update_state(components[name])
            results[name] = tracker.result()
        return results
