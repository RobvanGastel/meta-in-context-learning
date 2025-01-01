import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn

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

    @nn.compact
    def __call__(self, x):
        img_h, img_w = self.image_size, self.image_size
        patch_h, patch_w = self.patch_size[0], self.patch_size[1]
        assert (
            img_h % patch_h == 0 and img_w % patch_w == 0
        ), "Image dimensions must be divisible by the patch size."

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

        x = rearrange(x, "b c (h p1) (w p2) -> b h w (p1 p2 c)", p1=patch_h, p2=patch_w)
        x = nn.Dense(features=self.emb_dim)(x)
        pe = self.create_pos_encoding(x)
        x = rearrange(x, "b ... d -> b (...) d") + pe

        for attn, ff in layers:
            x = attn(x) + x
            x = ff(x) + x

        x = x.mean(axis=1)
        return linear_head(x)

    @staticmethod
    def create_pos_encoding(patches: int) -> jnp.ndarray:
        # sin-cos positional embedding
        _, h, w, dim = patches.shape
        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

        y, x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
        omega = jnp.arange(dim // 4) / (dim // 4 - 1)
        # The 1.0 / (temperature ** omega)
        omega = 1.0 / (10000.0**omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = jnp.concatenate((jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)), axis=1)
        return pe
