import jax
import jax.numpy as jnp


def create_reg_dataset(
    rng,
    input_size,
    set_size,
    distract_size,
    input_range,
    w_scale,
):
    """Create a linear regression data set:
    X*w where x ~ U(-1, 1), w ~ N(0,1).
    """
    rng_1, rng_2, rng_3, rng_4, rng_5 = jax.random.split(rng, 5)
    w = jax.random.normal(rng_1, shape=[input_size]) * w_scale

    x = jax.random.uniform(
        rng_2,
        shape=[set_size, input_size],
        minval=-input_range / 2,
        maxval=input_range / 2,
    )
    x_query = jax.random.uniform(
        rng_3,
        shape=[1, input_size],
        minval=-input_range / 2,
        maxval=input_range / 2,
    )

    y_data = jnp.squeeze(x @ w)
    choice = jax.random.choice(rng_5, set_size, shape=[distract_size], replace=False)
    y_data = y_data.at[choice].set(jax.random.normal(rng_4, shape=[distract_size]))

    y_target = x_query @ w
    y_target = y_target[..., None]

    X = jnp.concatenate([x, y_data[..., None]], -1)
    target = jnp.concatenate([x_query, y_target], -1)

    x_query_init = -1 * x_query.dot(jnp.ones_like(x_query).T * 0.0)
    zero = jnp.concatenate([x_query, x_query_init], -1)
    X = jnp.concatenate([X, zero], 0)
    return jnp.squeeze(X), jnp.squeeze(target), w


def sample_regression_dataset(
    rng,
    input_size: int,
    set_size: int = 10,
    distract_size: int = 0,  # Not used?
    input_range: float = 1.0,
    w_scale: float = 1.0,
):
    data_creator = jax.vmap(
        create_reg_dataset,
        in_axes=(0, None, None, None, None, None),
        out_axes=0,
    )

    train_data = data_creator(
        jax.random.split(rng, num=10_000),
        input_size,
        set_size,
        distract_size,
        input_range,
        w_scale,
    )

    return train_data
