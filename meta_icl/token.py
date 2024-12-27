import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class TokenVocab(nn.Module):
    """Learnable Vocabulary with certain "token" size."""

    w_init: Optional[nn.initializers.Initializer]
    e_size: int = 128
    vocab_size: int = 60000

    def setup(self):
        """Initializes the vocabulary weights."""
        self.vocab = self.param("vocab", self.w_init, (self.vocab_size, 1, self.e_size))

    def __call__(self, x, logits=False):
        """Forward pass."""
        if logits:
            return jnp.einsum("...l,Vl->...V", x, jnp.squeeze(self.vocab))
        else:
            return jnp.take_along_axis(self.vocab, jnp.expand_dims(x, axis=-1), axis=0)
