import math

import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn

from meta_icl.mha import MultiHeadAttention


class MLP(nn.Module):
    """A multi-layer perceptron (MLP)."""

    output_dim: int = 0
    widening_factor: int = 4
    use_bias: bool = False
    second_layer: bool = False
    w_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        emb_dim = x.shape[-1]

        x = nn.Dense(
            self.widening_factor * emb_dim,
            use_bias=self.use_bias,
            kernel_init=self.w_init,
        )(x)
        x = jax.nn.gelu(x)

        if self.second_layer:
            x = nn.Dense(
                self.widening_factor * emb_dim,
                use_bias=self.use_bias,
                kernel_init=self.w_init,
            )(x)
            x = jax.nn.gelu(x)

        if self.output_dim == 0:
            return nn.Dense(
                emb_dim,
                use_bias=self.use_bias,
                kernel_init=self.w_init,
            )(x)
        else:
            return nn.Dense(
                self.output_dim,
                use_bias=self.use_bias,
                kernel_init=self.w_init,
            )(x)


class Transformer(nn.Module):
    """A flexible Transformer implementation."""

    num_heads: int = 1
    num_layers: int = 1
    key_size: int = 11
    output_size: int = 1
    embedding_size: int = 11
    widening_factor: int = 4
    in_context_length: int = 10
    in_context_length_test: int = 10
    only_attention: bool = True
    use_layer_norm: bool = True
    use_pe: bool = False
    pe_size: int = 6
    use_bias: bool = True
    deq: bool = True
    use_softmax: bool = False
    use_non_lin_mix: bool = False
    first_layer_sm: bool = False
    sum_norm: bool = False
    dampening: float = 1.0
    clip: float = 0.0
    flip: bool = False

    def setup(self):
        if self.pe_size > 0:
            self.p_enc = self.create_pos_encoding(
                self.in_context_length, self.pe_size, self.flip
            )
            self.p_enc_test = self.create_pos_encoding(
                self.in_context_length_test, self.pe_size, self.flip
            )
        else:
            self.p_enc = None

        # Recurrent Transformer
        if self.deq:
            self.attn_block = MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                model_size=self.key_size * self.num_heads,
                use_softmax=self.use_softmax,
                use_non_lin_mix=self.use_non_lin_mix,
                use_bias=self.use_bias,
                sum_normalization=self.sum_norm,
            )

            if not self.only_attention:
                self.dense_block = MLP(
                    widening_factor=self.widening_factor,
                    use_bias=self.use_bias,
                )

            if self.use_layer_norm:
                self.lnorm_1 = nn.LayerNorm(feature_axes=-1, use_scale=True)
                self.lnorm_2 = nn.LayerNorm(feature_axes=-1, use_scale=True)

    def trans_block(self, h):
        """Transformer block with attention and optionally MLP."""

        if self.deq:
            h_norm = self.lnorm_1(h) if self.use_layer_norm else h
            key = h_norm[:, :-1, :]
            value = h_norm[:, :-1, :]

            h_attn, att_map = self.attn_block(h_norm, key, value)
        else:
            # For non recurrent do not use a layer norm on the first layer.
            raise NotImplementedError

        # TODO: Ignore dropout?
        # h_attn = nn.Dropout(self.dropout_rate)(h_attn)

        h = h + self.dampening * h_attn
        if self.clip > 0:
            h = jnp.clip(h, -self.clip, self.clip)

        if not self.only_attention:
            if self.deq:
                h_inter_norm = self.lnorm_2(h) if self.use_layer_norm else h
                h_dense = self.dense_block(h_inter_norm)
            else:
                raise NotImplementedError

            # TODO: Ignore dropout?
            # h_dense = nn.Dropout(self.dropout_rate)(h_dense)

            h = h + self.dampening * h_dense
            if self.clip > 0:
                h = jnp.clip(h, -self.clip, self.clip)
        return h, att_map

    @nn.compact
    def __call__(self, x: jnp.ndarray, is_training: bool):
        # Positional encoding
        if self.use_pe:
            pos_enc = self.p_enc if is_training else self.p_enc_test
            x += pos_enc

        h = x
        for _ in range(self.num_layers):
            h, _ = self.trans_block(h)

        return h

    @staticmethod
    def create_pos_encoding(
        context_size: int, input_size: int, flip: bool = False
    ) -> jnp.ndarray:
        """Create constant positional encoding."""
        pe = np.zeros((context_size, input_size))
        position = np.arange(0, context_size, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, input_size, 2) * (-math.log(10000.0) / input_size)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        if flip:
            return jnp.flip(jax.numpy.squeeze(jax.device_put(pe), axis=0), 0)
        else:
            return jax.numpy.squeeze(jax.device_put(pe), axis=0)
