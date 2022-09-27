import random
import itertools
from typing import Iterable, Iterator, TypeVar

_T = TypeVar('_T')

def repeat_and_shuffle(iterable: Iterable[_T], buffer_size: int) -> Iterator[_T]:
    """Infinitely repeats, caches and shuffles data from `iterable`"""

    ds = itertools.cycle(iterable)
    buffer = [next(ds) for _ in range(buffer_size)]
    random.shuffle(buffer)
    for item in ds:
        idx = random.randint(0, buffer_size-1)
        result = buffer[idx]
        buffer[idx] = item
        yield result