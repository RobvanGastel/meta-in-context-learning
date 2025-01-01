from typing import Optional

import jax
import jax.numpy as jnp
from einops import rearrange

from flax import linen as nn


class MultiHeadAttention(nn.Module):
    """Multi-headed attention (MHA) module"""

    num_heads: int
    key_size: int
    model_size: int
    w_init: Optional[nn.initializers.Initializer] = nn.initializers.xavier_uniform()
    use_bias: bool = False
    use_softmax: bool = False
    use_non_lin_mix: bool = False
    sum_normalization: bool = False

    def _linear_projection(
        self, x: jnp.ndarray, head_size: int, name: str
    ) -> jnp.ndarray:

        y = nn.Dense(
            self.num_heads * head_size,
            use_bias=self.use_bias,
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
        # Computes MHA with optional mask

        key_heads = self._linear_projection(key, self.key_size, "key")
        query_heads = self._linear_projection(query, self.key_size, "query")
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
            self.model_size,
            kernel_init=self.w_init,
            use_bias=self.use_bias,
            name="final_proj",
        )
        attn = final_projection(attn)
        return attn, attn_weights


class SimplifiedMultiHeadAttention(nn.Module):
    """A version of Multi-headed attention (MHA) with less parameters"""

    emb_dim: int
    num_heads: int = 8
    d_head: int = 64

    @nn.compact
    def __call__(self, x):
        inner_dim = self.d_head * self.num_heads
        norm = nn.LayerNorm(epsilon=1e-5, use_bias=False)

        to_qkv = nn.Dense(inner_dim * 3, use_bias=False)
        to_out = nn.Dense(self.emb_dim, use_bias=False)

        x = norm(x)

        qkv = jnp.split(to_qkv(x), 3, axis=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            qkv,
        )

        dots = jnp.einsum("b h i d, b h j d -> b h i j", q, k) * self.d_head**-0.5

        attn = nn.softmax(dots, axis=-1)

        x = jnp.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(x, "b h n d -> b n (h d)")
        return to_out(out)
