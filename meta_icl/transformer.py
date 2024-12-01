from flax import linen as nn


class Transformer(nn.Module):
    num_heads: int
    embed_dim: int
    num_layers: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.embed_dim)(x)
        for _ in range(self.num_layers):
            x = nn.SelfAttention(num_heads=self.num_heads)(x)
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.output_dim)(x)
        return x
