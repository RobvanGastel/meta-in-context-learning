import jax
from PIL import Image
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from transformers import FlaxCLIPModel, CLIPProcessor

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
    seq_length: int
    channels: int = 3
    num_heads: int = 1
    num_layers: int = 1
    d_head: int = 64
    image_encoder: str = "openai/clip-vit-base-patch16"

    def setup(self):
        self.clip_model = FlaxCLIPModel.from_pretrained(self.image_encoder)
        self.preprocessor = CLIPProcessor.from_pretrained(self.image_encoder)

        self.pos_embedding = self.param(
            "pos_embedding",
            initializers.zeros,
            (1, self.seq_length, self.emb_dim),  # Initialize to zeros
        )

    def forward_features(self, x):
        inputs = self.preprocessor(
            images=x, return_tensors="np"
        )  # TODO: Check return tensors
        pixel_values = jnp.array(inputs["pixel_values"])
        image_features = self.clip_model.get_image_features(pixel_values)
        return image_features

    def forward_encoder(self, x):
        # Sequence-to-seuence Transformer Encoder
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
        ln = nn.LayerNorm(epsilon=1e-5, use_bias=False)
        #

        for attn, ff in layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    @nn.compact
    def __call__(self, X, y):
        # Input shape (B, S, H*W)
        b, s, *_ = X.shape

        # Prepare context vector
        y_one_hot = jax.nn.one_hot(y, num_classes=self.num_classes)
        y_emb = jnp.concatenate(
            [jnp.zeros((b, 1, self.num_classes)), y_one_hot], axis=1
        )

        # Use CLIP pre-trained weights to create image embeddings
        # (B, S, 512)
        X_bs = X.reshape(b * s, 1, 28, 28).astype(jnp.uint8)
        X_bs = jnp.tile(X_bs, (1, 3, 1, 1))
        X_emb = self.forward_features(X_bs)
        X_emb = X_emb.reshape(b, s, 512)

        # new shape: batch, sequence, (num_classes + H*W)
        context = jnp.concatenate([X, y_emb], axis=-1)

        # Apply learned positional embeddings
        context = nn.Dense(self.emb_dim)(context)
        context = context + self.pos_embedding[:, : context.shape[1], :]

        # Forward through the transformer
        sequence = self.forward_encoder(context)
        query = sequence[:, -1]

        output_proj = nn.Dense(self.num_classes, use_bias=False, name="out_proj")
        out = output_proj(query)
        return out
