import jax
import wandb
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
VAL_PERIOD = 10

NUM_LAYERS = 2
NUM_HEADS = 4
EMB_DIM = 128

P = 97
TRAIN_SPLIT = 0.5

class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    rng: jnp.DeviceArray
    step: jnp.DeviceArray


class Metrics(NamedTuple):
    loss: jnp.ndarray

def net_fn(tokens: jnp.ndarray) -> jnp.ndarray:
    transformer = Transformer(
        NUM_LAYERS,
        NUM_HEADS,
        EMB_DIM,
        num_tokens=P
    )

    return transformer(tokens)


def main(_):
    wandb.init('grokking')

    # Create network and optimiser
    network = hk.without_apply_rng(hk.transform(net_fn))
    optimiser = optax.adam(LEARNING_RATE)

    def loss_fn(params: hk.Params, batch: Batch) -> jnp.ndarray:
        logits = network.apply(params, batch.inputs)[:, -1]
        targets = jax.nn.one_hot(batch.targets, num_classes=P)
        assert logits.shape == targets.shape

        log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
        return -jnp.sum(log_likelihood)

    @jax.jit
    def evaluate(params: hk.Params, batch: Batch) -> jnp.ndarray:
        logits = network.apply(params, batch.inputs)[:, -1]
        preds = jnp.argmax(logits, axis=-1)
        return jnp.mean(preds == batch.targets)

    @jax.jit
    def update(state: TrainingState, data: Batch) -> Tuple[TrainingState, Metrics]:
        rng, new_rng = jax.random.split(state.rng)
        loss_and_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_and_grad_fn(state.params, data)
        
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
    def init(rng: jnp.ndarray, batch: Batch) -> TrainingState:
        rng, init_rng = jax.random.split(rng)
        initial_params = network.init(init_rng, batch.inputs)
        initial_opt_state = optimiser.init(initial_params)

        return TrainingState(
            initial_params,
            initial_opt_state,
            rng,
            step=np.array([0])
        )

    train_data, test_data = get_dataset('x+y', TRAIN_SPLIT, P)

    rng = jax.random.PRNGKey(SEED)
    state = init(rng, train_data)

    train_acc = 0
    test_acc = 0

    # Training loop
    p_bar = tqdm(range(MAX_STEPS))
    for step in p_bar:
        state, metrics = update(state, train_data)
        wandb.log({'train/loss': metrics.loss, 'step': step})

        p_bar.set_description(f'train/loss: {metrics.loss:.2f}, train/acc: {train_acc:.2f}, test/acc: {test_acc:.2f}')

        # Evaluate on test set
        if step % VAL_PERIOD == 0:
            train_acc = evaluate(state.params, train_data)
            test_acc = evaluate(state.params, test_data)
            test_loss = loss_fn(state.params, test_data)

            metrics = {
                'train/acc': train_acc,
                'test/acc': test_acc,
                'test/loss': test_loss,
                'step': step
            }
            wandb.log(metrics)


if __name__ == '__main__':
    app.run(main)