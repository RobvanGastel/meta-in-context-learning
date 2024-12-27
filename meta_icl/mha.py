import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Multi-headed attention (MHA) module."""

    num_heads: int
    key_size: int
    model_size: int
    w_init: Optional[nn.initializers.Initializer] = nn.initializers.xavier_uniform()
    use_bias_p: bool = False
    use_softmax: bool = False
    use_non_lin_mix: bool = False
    sum_normalization: bool = False

    def _linear_projection(
        self, x: jnp.ndarray, head_size: int, name: str
    ) -> jnp.ndarray:
        """Linear projection for attention heads."""
        y = nn.Dense(
            self.num_heads * head_size,
            use_bias=self.use_bias_p,
            kernel_init=self.w_init,
            name=name,
        )(x)
        return y.reshape((*x.shape[:-1], self.num_heads, head_size))

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ):
        """Computes MHA with optional mask."""

        query_heads = self._linear_projection(query, self.key_size, "query")
        key_heads = self._linear_projection(key, self.key_size, "key")

        # TODO: Potential self.value_size instead of key_size
        value_heads = self._linear_projection(value, self.key_size, "value")

        if self.sum_normalization:
            query_heads = query_heads / (
                jnp.sum(query_heads, axis=-1, keepdims=True) + 1e-6
            )
            key_heads = key_heads / (jnp.sum(key_heads, axis=-1, keepdims=True) + 1e-6)

        # Attention logits
        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        if mask is not None:
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        # Apply softmax or custom non-linear mix
        if self.use_softmax:
            attn_weights = jax.nn.softmax(
                attn_logits / jnp.sqrt(self.key_size).astype(query.dtype)
            )
        elif self.use_non_lin_mix:
            y = nn.Dense(
                1, use_bias=False, kernel_init=self.w_init, name="non_lin_mix"
            )(jnp.array([1.0]))
            attn_weights = (
                jax.nn.softmax(
                    attn_logits / jnp.sqrt(self.key_size).astype(query.dtype)
                )
            ) * jax.nn.sigmoid(y * 10) + (1 - jax.nn.sigmoid(y * 10)) * attn_logits
        else:
            attn_weights = attn_logits

        # Weight the values and apply final projection
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = attn.reshape((*attn.shape[:-2], -1))  # Reshape to [T', H*V]

        final_projection = nn.Dense(
            self.model_size, kernel_init=self.w_init, use_bias=self.use_bias_p
        )
        attn = final_projection(attn)
        return attn, attn_weights
