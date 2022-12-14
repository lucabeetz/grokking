import itertools
import numpy as np
from typing import NamedTuple

class Batch(NamedTuple):
    inputs: np.ndarray
    targets: np.ndarray

def get_operation(op_key: str, p: int):
    """Get matching lambda function for binary operation"""

    OPERATIONS = {
        'x+y': lambda x, y: (x, y, (x + y) % p)
    }

    return OPERATIONS[op_key]

def create_training_data(op_key: str, p: int, op_token: int, eq_token: int):
    """Create dataset for binary operation"""

    assert op_key in ['x+y']

    # Create all possible training pairs
    x = np.arange(p)
    y = np.arange(p)
    cart_prod = np.array(list(itertools.product(x, y)))
    x, y = cart_prod[:, 0], cart_prod[:, 1]

    # Calculate labels
    op_fn = get_operation(op_key, p)
    x, y, labels = op_fn(x, y)

    # Concatenate parts
    op_tokens = np.ones_like(x) * op_token
    eq_tokens = np.ones_like(y) * eq_token
    data = np.stack([x, op_tokens, y, eq_tokens, labels], axis=1)

    # Shuffle data
    np.random.shuffle(data)

    return data[:, :4], data[:, 4]

def get_dataset(op_key: str, train_split: float, p: int):
    """Get train and test splits for given `op_key`"""

    inputs, labels = create_training_data(op_key, p, p, p+1)

    train_size = int(len(inputs) * train_split)

    return Batch(inputs[:train_size], labels[:train_size]), Batch(inputs[train_size:], labels[train_size:])
