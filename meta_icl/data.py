import jax
import jax.numpy as jnp


def create_reg_dataset(rng, i_size, c_size, size_distract, input_range, w_scale):
    """Create a linear regression data set: X*w where x ~ U(-1, 1), w ~ N(0,1)."""

    rng, new_rng, new_rng2, new_rng3, new_rng4 = jax.random.split(rng, 5)
    w = jax.random.normal(rng, shape=[i_size]) * w_scale

    x = jax.random.uniform(
        new_rng, shape=[c_size, i_size], minval=-input_range / 2, maxval=input_range / 2
    )
    x_querry = jax.random.uniform(
        new_rng2, shape=[1, i_size], minval=-input_range / 2, maxval=input_range / 2
    )

    y_data = jnp.squeeze(x @ w)
    choice = jax.random.choice(new_rng4, c_size, shape=[size_distract], replace=False)
    y_data = y_data.at[choice].set(jax.random.normal(new_rng3, shape=[size_distract]))

    y_target = x_querry @ w
    y_target = y_target[..., None]

    seq = jnp.concatenate([x, y_data[..., None]], -1)
    target = jnp.concatenate([x_querry, y_target], -1)
    x_querry_init = -1 * x_querry.dot(jnp.ones_like(x_querry).T * 0.0)
    zero = jnp.concatenate([x_querry, x_querry_init], -1)
    seq = jnp.concatenate([seq, zero], 0)
    return jnp.squeeze(seq), jnp.squeeze(target), w
