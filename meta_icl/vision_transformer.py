import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers

from meta_icl.mha import SimplifiedMultiHeadAttention


class FeedForward(nn.Module):
    dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(epsilon=1e-5, use_bias=False)(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim)(x)
        return x


class ViT(nn.Module):
    image_size: int
    num_classes: int
    patch_size: int
    emb_dim: int
    mlp_dim: int
    channels: int = 3
    num_heads: int = 1
    num_layers: int = 1
    d_head: int = 64

    def forward_features(self, x):
        # ViT Encoder
        layers = []
        for _ in range(self.num_layers):
            layers.append(
                [
                    SimplifiedMultiHeadAttention(
                        self.emb_dim,
                        num_heads=self.num_heads,
                        d_head=self.d_head,
                    ),
                    FeedForward(self.emb_dim, self.mlp_dim),
                ]
            )

        linear_head = nn.Sequential(
            [
                nn.LayerNorm(epsilon=1e-5, use_bias=False),
                nn.Dense(features=self.num_classes),
            ]
        )

        # TODO: Have pretrained weights
        # With pretrained weights attempt freezing the encoder

        for attn, ff in layers:
            x = attn(x) + x
            x = ff(x) + x

        x = x.mean(axis=1)
        return linear_head(x)

    @nn.compact
    def __call__(self, X, y):
        img_h, img_w = self.image_size, self.image_size
        patch_h, patch_w = self.patch_size[0], self.patch_size[1]

        assert (
            img_h % patch_h == 0 and img_w % patch_w == 0
        ), "Image dimensions must be divisible by the patch size."

        # TODO: Get right input format, should reshape to batch, sequence, emb_dim (28*28)
        b, *_ = X.shape

        # Prepare context vector
        y_one_hot = jax.nn.one_hot(y, num_classes=self.num_classes)
        y_emb = jnp.concatenate(
            [jnp.zeros((1, 1, self.num_classes)), y_one_hot], axis=0
        )
        y_emb = jnp.tile(y_emb[None, :, :], (b, 1, 1))

        # new shape: batch, sequence, (10 + 28*28)
        context = jnp.concatenate([X, y_emb], axis=-1)

        # Apply learned positional embeddings
        input_proj = nn.Dense(self.emb_dim)(context)
        pos_emb = self.param(
            "pos_emb", initializers.zeros, (1, 28, self.emb_dim)  # Initialize to zeros
        )

        context = input_proj(context) + pos_emb[:, : context.shape[1], :]

        sequence = self.forward_features(context)
        query = sequence[:, -1]

        output_proj = nn.Dense(self.num_classes, use_bias=False)
        return output_proj(query)
