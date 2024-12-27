import math


import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn

from meta_icl.mha import MultiHeadAttention
from meta_icl.token import TokenVocab


class Transformer(nn.Module):
    """A flexible Transformer implementation."""

    num_heads: int = 2
    widening_factor: int = 4
    num_layers: int = 3
    key_size: int = 5
    embedding_size: int = 10
    output_size: int = 1
    in_context_length: int = 17
    in_context_length_test: int = 17
    only_attention: bool = True
    use_layer_norm: bool = True
    use_pe: bool = False
    pe_size: int = 6
    concat_pe: bool = False
    output_mapping: bool = False
    input_mapping: bool = True
    use_bias_p: bool = True
    zero_embeddings: bool = False
    deq: bool = True
    use_softmax: bool = False
    use_non_lin_mix: bool = False
    first_layer_sm: bool = False
    input_mlp: bool = False
    sum_norm: bool = False
    dampening: float = 1.0
    clip: float = 0.0
    flip: bool = False
    vocab_size: int = 0
    vocab_token_dim: int = 0
    vocab_init: int = 0.01
    return_logits: bool = False
    include_query: bool = False

    def setup(self):
        """Initializes the module layers."""
        if self.pe_size > 0:
            self.pos_encoding = create_pos_encoding(
                self.in_context_length, self.pe_size, self.flip
            )
            self.pos_encoding_test = create_pos_encoding(
                self.in_context_length_test, self.pe_size, self.flip
            )
        else:
            self.pos_encoding = None

        if not self.only_attention:
            self.dense_block = MLP(
                widening_factor=self.widening_factor,
                use_bias_p=self.use_bias_p,
            )

    def trans_block(self, h, nl):
        """Transformer block with attention and optionally MLP."""

        if self.deq:
            # TODO: Check this layer norm initialization
            h_norm = (
                nn.LayerNorm(feature_axes=-1, use_scale=True)(h)
                if self.use_layer_norm
                else h
            )

            key = h_norm[:, :-1, :] if not self.include_query else h_norm
            value = key

            h_attn, att_map = MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                model_size=self.key_size * self.num_heads,
            )(h_norm, key, value)
        else:
            h_norm = (
                nn.LayerNorm(feature_axes=-1, use_scale=True)(h)
                if self.use_layer_norm
                else h
            )
            h_attn, att_map = MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                model_size=self.key_size * self.num_heads,
            )(h_norm, h_norm, h_norm)

        # TODO: Ignore dropout?
        # h_attn = nn.Dropout(self.dropout_rate)(h_attn)

        h = h + self.dampening * h_attn
        if self.clip > 0:
            h = jnp.clip(h, -self.clip, self.clip)

        if not self.only_attention:
            if self.deq:
                # TODO: Layernorm
                h_dense = self.dense_block(h) if self.use_layer_norm else h
            else:
                # TODO: Layernorm
                h_dense = MLP(self.widening_factor, use_bias_p=self.use_bias_p)(h)

            # TODO: Ignore dropout?
            # h_dense = nn.Dropout(self.dropout_rate)(h_dense)
            h = h + self.dampening * h_dense
            if self.clip > 0:
                h = jnp.clip(h, -self.clip, self.clip)

        return h, att_map

    @nn.compact
    def __call__(self, x: jnp.ndarray, is_training: bool):
        if self.vocab_size > 0 and self.vocab_token_dim > 0:
            vocab = TokenVocab(self.vocab_size, self.vocab_token_dim, self.vocab_init)
            x = vocab(x)

        embeddings = x
        if self.input_mapping:
            embeddings = nn.Dense(self.embedding_size, use_bias=self.use_bias_p)(x)

        if self.input_mlp:
            input_mlp = MLP(self.widening_factor, use_bias_p=True)(embeddings)
            embeddings = embeddings + input_mlp(embeddings)

        if self.use_pe:
            pos_encoding = (
                self.pos_encoding_test if not is_training else self.pos_encoding
            )
            pos_encoding = jnp.repeat(
                pos_encoding[None, ...], embeddings.shape[0], axis=0
            )
            pos_encoding = pos_encoding * 0 if self.zero_embeddings else pos_encoding
            h = (
                jnp.concatenate([embeddings, pos_encoding], axis=2)
                if self.concat_pe
                else embeddings + pos_encoding
            )
        else:
            h = embeddings

        for nl in range(self.num_layers):
            h, att_map = self.trans_block(h, nl)

        out = nn.Dense(self.output_size)(h) if self.output_mapping else h
        if self.return_logits:
            out = vocab(out, logits=True)

        return out


def create_pos_encoding(
    context_size: int, input_size: int, flip: bool = False
) -> jnp.ndarray:
    """Create constant positional encoding."""
    pe = np.zeros((context_size, input_size))
    position = np.arange(0, context_size, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, input_size, 2) * (-math.log(10000.0) / input_size))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[None]
    if flip:
        return jnp.flip(jax.numpy.squeeze(jax.device_put(pe), axis=0), 0)
    else:
        return jax.numpy.squeeze(jax.device_put(pe), axis=0)


class MLP(nn.Module):
    """A multi-layer perceptron (MLP)."""

    w_init: nn.initializers.Initializer
    widening_factor: int = 4
    second_layer: bool = False
    use_bias_p: bool = False
    output_dim: int = 0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the MLP."""
        hiddens = x.shape[-1]

        x = nn.Dense(
            self.widening_factor * hiddens,
            use_bias=self.use_bias_p,
            kernel_init=self.w_init,
        )(x)
        x = jax.nn.gelu(x)

        if self.second_layer:
            x = nn.Dense(
                self.widening_factor * hiddens,
                use_bias=self.use_bias_p,
                kernel_init=self.w_init,
            )(x)
            x = jax.nn.gelu(x)

        if self.output_dim == 0:
            return nn.Dense(
                hiddens,
                use_bias=self.use_bias_p,
                kernel_init=self.w_init,
            )(x)
        else:
            return nn.Dense(
                self.output_dim,
                use_bias=self.use_bias_p,
                kernel_init=self.w_init,
            )(x)
