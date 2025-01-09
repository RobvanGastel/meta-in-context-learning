from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset


class FewShotDataset(Dataset):
    def __init__(self, dataset, train: bool, transform=None):
        self.dataset = dataset(root="./data", train=train, download=True)

        self.X = self.dataset.data
        self.y = self.dataset.targets  # convert to long
        self.transform = transform

        self.mean = self.X.float().mean() / 255
        self.std = self.X.float().std() / 255

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)

        x = x.float() / 255
        # Z-normalize
        x = (x - self.mean) / self.std

        return x, y

    def __len__(self):
        return len(self.X)


class FewShotBatchSampler:
    def __init__(self, y, n_way, k_shot, batch_size=5, shuffle=True):
        self.y = y.numpy()
        self.n_way = n_way
        self.k_shot = k_shot
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.class_indices = {
            c: np.where(self.y == c)[0].tolist() for c in np.unique(y)
        }

    def __iter__(self):
        classes = np.array(list(self.class_indices.keys()))

        for _ in range(len(classes) // self.n_way):
            tasks = []
            for _ in range(self.num_tasks):
                selected_classes = np.random.choice(classes, self.n_way, replace=False)
                batch = []
                for cls in selected_classes:
                    examples = np.random.choice(
                        self.class_indices[cls], self.k_shot, replace=False
                    ).tolist()
                    batch.extend(examples)

                if self.shuffle:
                    np.random.shuffle(batch)
                tasks.append(batch)

            yield tasks

    def __len__(self):
        return len(self.y) // (self.n_way * self.k_shot)


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_reg_dataset(
    rng,
    input_size: int,
    set_size: int,
    distract_size: int,
    input_range: float,
    w_scale: float,
):
    """Create a linear regression data set:
    X*w where x ~ U(-1, 1), w ~ N(0,1)

    Taken from the original paper (von Oswald et al., 2022) but restructured to make the
    code runnable again.
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
    batch_size: int = 10_000,
    set_size: int = 10,
    input_range: float = 1.0,
    w_scale: float = 1.0,
):
    data_creator = jax.vmap(
        create_reg_dataset,
        in_axes=(0, None, None, None, None, None),
        out_axes=0,
    )

    train_data = data_creator(
        jax.random.split(rng, num=batch_size),
        input_size,
        set_size,
        0,
        input_range,
        w_scale,
    )
    return train_data
