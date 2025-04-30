import pickle
import argparse

import optax
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import serialization
from flax.training import train_state
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST

from meta_icl.vision_transformer import ViT
from meta_icl.transforms import augment_tasks
from meta_icl.data import FewShotDataset, FewShotBatchSampler


def train_step(state, X, y, perm_prob, train_key, eval_key):
    def loss_fn(params, X, y):
        logits = state.apply_fn(params, X, y[:, :-1])
        # Compare against the last y label in the few-shot task and omit this y label during
        # the forward pass.
        y_one_hot = jax.nn.one_hot(y, 10)[:, -1].squeeze()
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_one_hot))
        return loss

    # Task augmentation
    X_bar, y_bar = augment_tasks(X, y, train_key, perm_prob)

    loss_grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = loss_grad_fn(state.params, X_bar, y_bar)
    state = state.apply_gradients(grads=grads)

    # Validating the performance with a different evaluation key
    X_bar, y_bar = augment_tasks(X, y, eval_key)
    logits = state.apply_fn(state.params, X_bar, y[:, :-1])
    y_one_hot = jax.nn.one_hot(y, 10)[:, -1].squeeze()
    acc = jnp.sum(y[:, -1] == jnp.argmax(logits, axis=-1))
    return state, loss, acc

class TrainState(train_state.TrainState):
    pass

def train_general_purpose_vit(config: argparse.Namespace):

    vit_model = ViT(
        image_size = 28,
        patch_size = (14, 14),
        num_classes = 10,
        emb_dim = 256,
        seq_length = config.sequence_length,
        channels = 1,
        num_layers = 4,
        num_heads = 8,
        mlp_dim = 512
    )
    
    key = jax.random.key(config.seed)
    key, x_key, y_key, rng_key = jax.random.split(key, 4)

    # Initialize with the keys
    X = jax.random.normal(x_key, (config.batch_size, config.sequence_length, 28*28))
    y = jax.random.normal(y_key, (config.batch_size, config.sequence_length-1))
    params = vit_model.init({'params': rng_key}, X, y)
    output = vit_model.apply(params, X, y, rngs={'params': rng_key})
    
    state = TrainState.create(
        apply_fn=vit_model.apply,
        params=params,
        tx=optax.adamw(learning_rate=config.lr)
    )
    
    train_dataset = FewShotDataset(dataset=config.task, train=True)
    data_loader = DataLoader(train_dataset, batch_sampler=FewShotBatchSampler(
        train_dataset.y, config.n_way, config.k_shot, batch_size=config.batch_size
        )
    )
    
    # Meta-training loop
    metrics = {
        "loss": [],
        "accuracy": []
    }
    for epoch in range(config.epochs):
        train_loss = 0
        accuracy = 0.0
    
        for X, y in data_loader:
            key, train_key, eval_key = jax.random.split(key, 3)
            state, loss, acc = train_step(
                state,
                X.numpy(),
                y.numpy(),
                config.permutation_probability,
                train_key,
                eval_key
            )
            train_loss += loss.mean()
            accuracy += (acc / config.batch_size)
            
        metrics["loss"].append(float(train_loss))
        metrics["accuracy"].append(float(accuracy))
    
        if epoch % 50 == 0:
            print(f"epoch {epoch}/{config.epochs}: loss: {train_loss}, accuracy: {accuracy}")
    
            # Save the weights
            with open(f"output/gpicl_e{epoch}.pkl", "wb") as f:
                f.write(pickle.dumps(serialization.to_state_dict(params)))
    
    with open(f"output/gpicl_mnist_metrics.json", "w") as f:
        json.dump(metrics, f)

                
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument(
        "--task",
        type=str,
        default="mnist",
        help="The task either MNIST or FashionMNIST",
    )
    parser.add_argument(
        "--permutation_probability",
        type=float,
        default=0.0,
        help="The label permutation probability of a task",
    )
    parser.add_argument(
        "--n_way",
        type=int,
        default=3,
        help="N-way (k-shot) few-shot task",
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=2,
        help="(N-way) k-shot few-shot task",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=6,
        help="sequence",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=750,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Finetuning batch size",
    )
    config = parser.parse_args()

    # Number of total samples
    config.sequence_length = config.k_shot * config.n_way
    config.task = MNIST if config.task == "mnist" else FashionMNIST
    
    train_general_purpose_vit(config)