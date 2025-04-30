import jax
import jax.numpy as jnp

@jax.jit
def augment_tasks(X, y, key, p=0.0):
    batch_size, seq, _, _ = X.shape
    key_A, key_perm, key_apply = jax.random.split(key, 3)

    # Linear projection A, A_ij \in N(0, 1/Nx)
    X_bar = jnp.reshape(X, (batch_size, seq, 28*28))
    A = jax.random.normal(key_A, (batch_size, 28*28), dtype=jnp.float32) / 28
    # Shape: (batch_size, seq_length, 28*28)
    X_bar = jnp.einsum('bsd,bd->bsd', X_bar, A)

    # Should have permutation \rho(y), to create a new mapping
    perm  = jax.random.permutation(key_perm, 10) # num_classes
    apply = jax.random.bernoulli(key_apply, p)
    y_bar = jnp.where(apply, perm[y], y)

    return X_bar, y_bar

