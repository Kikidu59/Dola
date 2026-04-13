import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, random
import optax
import pickle
from typing import List, Tuple, Dict
import time

# ---------------------------------------------------------------------------
# Type aliases for readability
# ---------------------------------------------------------------------------
Params = List[Dict[str, jnp.ndarray]]  # list of {W, b} dicts, one per layer


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


def init_params(layer_sizes: List[int], key: jax.Array) -> Params:
    """Initialize network weights using Glorot uniform initialization.

    Args:
        layer_sizes: list of integers defining the width of each layer,
                     e.g. [784, 128, 64, 10].
        key: JAX random key.

    Returns:
        List of parameter dicts, each containing 'W' and 'b'.
    """
    params = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, subkey = random.split(key)
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        W = random.uniform(subkey, shape=(fan_in, fan_out), minval=-limit, maxval=limit)
        b = jnp.zeros((fan_out,))
        params.append({"W": W, "b": b})
    return params


def forward(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    """Compute a forward pass through the network.

    Applies ReLU activations on all hidden layers; no activation on the output.

    Args:
        params: list of layer parameter dicts.
        x: input array of shape (batch_size, input_dim).

    Returns:
        Logits of shape (batch_size, output_dim).
    """
    for layer in params[:-1]:
        x = x @ layer["W"] + layer["b"]
        x = jax.nn.relu(x)

    # Output layer: no activation
    x = x @ params[-1]["W"] + params[-1]["b"]
    return x


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def cross_entropy_loss(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute mean cross-entropy loss over a batch.

    Args:
        params: network parameters.
        x: input batch of shape (batch_size, input_dim).
        y: integer class labels of shape (batch_size,).

    Returns:
        Scalar loss value.
    """
    logits = forward(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def mse_loss(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute mean squared error loss over a batch.

    Args:
        params: network parameters.
        x: input batch of shape (batch_size, input_dim).
        y: target values of shape (batch_size, output_dim).

    Returns:
        Scalar loss value.
    """
    preds = forward(params, x)
    return jnp.mean((preds - y) ** 2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def accuracy(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute classification accuracy on a batch.

    Args:
        params: network parameters.
        x: input batch.
        y: integer class labels.

    Returns:
        Fraction of correctly classified samples.
    """
    logits = forward(params, x)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == y)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def make_train_step(optimizer, loss_fn):
    """Return a JIT-compiled training step function bound to an optimizer and loss.

    Args:
        optimizer: optax optimizer instance.
        loss_fn: loss function with signature (params, x, y) -> scalar.

    Returns:
        A JIT-compiled function (params, opt_state, x, y) -> (params, opt_state, loss).
    """

    @jit
    def train_step(params, opt_state, x, y):
        loss_value, grads = value_and_grad(loss_fn)(params, x, y)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_value

    return train_step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    params: Params,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    key: jax.Array,
) -> Tuple[Params, Dict]:
    """Full training loop with validation logging.

    Args:
        params: initial network parameters.
        x_train: training inputs.
        y_train: training labels.
        x_val: validation inputs.
        y_val: validation labels.
        num_epochs: number of passes over the training set.
        batch_size: number of samples per gradient update.
        learning_rate: step size for the optimizer.
        key: JAX random key for shuffling.

    Returns:
        Tuple of (trained_params, history) where history is a dict of
        lists containing loss and accuracy per epoch.
    """
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer, cross_entropy_loss)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    n_samples = x_train.shape[0]

    for epoch in range(num_epochs):
        start = time.time()

        # Shuffle training data at the start of each epoch
        key, subkey = random.split(key)
        perm = random.permutation(subkey, n_samples)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        # Mini-batch gradient descent
        epoch_loss = 0.0
        num_batches = n_samples // batch_size
        for i in range(num_batches):
            x_batch = x_shuffled[i * batch_size : (i + 1) * batch_size]
            y_batch = y_shuffled[i * batch_size : (i + 1) * batch_size]
            params, opt_state, batch_loss = train_step(
                params, opt_state, x_batch, y_batch
            )
            epoch_loss += batch_loss

        avg_loss = epoch_loss / num_batches
        val_loss = cross_entropy_loss(params, x_val, y_val)
        val_acc = accuracy(params, x_val, y_val)

        history["train_loss"].append(float(avg_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))

        elapsed = time.time() - start
        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | "
            f"train_loss: {avg_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f} | "
            f"time: {elapsed:.2f}s"
        )

    return params, history


# ---------------------------------------------------------------------------
# Save / load utilities
# ---------------------------------------------------------------------------


def save_params(params: Params, filepath: str) -> None:
    """Serialize and save parameters to disk using pickle.

    Args:
        params: network parameters to save.
        filepath: destination file path (e.g. 'results/model.pkl').
    """
    with open(filepath, "wb") as f:
        pickle.dump(params, f)
    print(f"Parameters saved to {filepath}")


def load_params(filepath: str) -> Params:
    """Load parameters previously saved with save_params.

    Args:
        filepath: path to the saved parameter file.

    Returns:
        Loaded network parameters.
    """
    with open(filepath, "rb") as f:
        params = pickle.load(f)
    print(f"Parameters loaded from {filepath}")
    return params


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    key = random.PRNGKey(42)

    # Network architecture: input 784, two hidden layers of 128, output 10
    layer_sizes = [784, 128, 128, 10]
    key, subkey = random.split(key)
    params = init_params(layer_sizes, subkey)

    # Dummy data for illustration
    key, subkey = random.split(key)
    x_train = random.normal(subkey, shape=(1000, 784))
    y_train = random.randint(subkey, shape=(1000,), minval=0, maxval=10)
    x_val = random.normal(subkey, shape=(200, 784))
    y_val = random.randint(subkey, shape=(200,), minval=0, maxval=10)

    trained_params, history = train(
        params=params,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-3,
        key=key,
    )

    save_params(trained_params, "results/model.pkl")
