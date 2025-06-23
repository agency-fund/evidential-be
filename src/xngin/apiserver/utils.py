import random
import secrets
from collections.abc import Sequence


def random_choice[T](choices: Sequence[T], seed: int | None = None) -> T:
    """Choose a random value from choices."""
    if seed:
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        # use a predictable random
        r = random.Random(seed)
        return r.choice(choices)
    # Use very strong random by default
    return secrets.choice(choices)
