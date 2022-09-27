import jax
import optax
import haiku as hk
import numpy as np
import jax.numpy as jnp
from absl import app
from tqdm import tqdm
from typing import NamedTuple, Tuple

from data import Batch, get_dataset
from model import Transformer

SEED = 42
LEARNING_RATE = 3e-4
MAX_STEPS = 1000
LOG_EVERY = 10

NUM_LAYERS = 2
NUM_HEADS = 4
EMB_DIM = 128

P = 10


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    rng: jnp.DeviceArray
    step: jnp.DeviceArray


class Metrics(NamedTuple):
    loss: jnp.ndarray


def main(_):
    def forward(tokens: jnp.ndarray) -> jnp.ndarray:
        transformer = Transformer(
            NUM_LAYERS,
            NUM_HEADS,
            EMB_DIM,
            num_tokens=P
        )

        return transformer(tokens)


    optimiser = optax.adam(LEARNING_RATE)

    @hk.transform
    def loss_fn(batch: Batch) -> jnp.ndarray:
        logits = forward(batch.inputs)[:, -1]
        targets = jax.nn.one_hot(batch.targets, num_classes=P)
        assert logits.shape == targets.shape

        log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
        return -jnp.sum(log_likelihood)

    @jax.jit
    def update(state: TrainingState, data: Batch) -> Tuple[TrainingState, Metrics]:
        rng, new_rng = jax.random.split(state.rng)
        loss_and_grad_fn = jax.value_and_grad(loss_fn.apply)
        loss, grads = loss_and_grad_fn(state.params, rng, data)
        
        updates, new_opt_state = optimiser.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)

        new_state = TrainingState(
            new_params,
            new_opt_state,
            new_rng,
            step=state.step + 1
        )

        metrics = Metrics(loss)

        return new_state, metrics

    @jax.jit
    def init(rng: jnp.ndarray, data: jnp.ndarray) -> TrainingState:
        rng, init_rng = jax.random.split(rng)
        initial_params = loss_fn.init(init_rng, data)
        initial_opt_state = optimiser.init(initial_params)

        return TrainingState(
            initial_params,
            initial_opt_state,
            rng,
            step=np.array([0])
        )

    train_data, test_data = get_dataset('x+y', 0.8, 1.0, P)

    rng = jax.random.PRNGKey(SEED)
    state = init(rng, train_data)

    # Training loop
    p_bar = tqdm(range(MAX_STEPS))
    for step in p_bar:
        state, metrics = update(state, train_data)
        p_bar.set_description(f'train/loss: {metrics.loss:.2f}')


if __name__ == '__main__':
    app.run(main)