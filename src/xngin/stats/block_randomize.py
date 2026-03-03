"""Permuted block randomization for balanced online arm assignment.

Provides a thread-safe in-memory block randomizer that guarantees perfect
balance over every block of consecutive draws for a given experiment,
while preserving marginal randomization probabilities.  Supports both
uniform and weighted arm allocation.
"""

from collections import deque
from collections.abc import Iterable, Sequence
from fractions import Fraction
from math import lcm
from threading import Lock

import numpy as np


def get_lcm_prob_denominators(probs: Iterable[float]) -> int:
    """Compute the LCM of the denominators of the given probabilities.

    Borrowed from https://github.com/manmartgarc/stochatreat
    """
    prob_denominators = (Fraction(prob).limit_denominator().denominator for prob in probs)
    return lcm(*prob_denominators)


# Block key is an internal type representing the tuple of:
#   (experiment identifier, the number of arms, optional arm weights)
# that maps to the in-memory block of arm assignments to dole out sequentially.
type BlockKey = tuple[str, int, tuple[float, ...] | None]


class BlockRandomize:
    """Thread-safe permuted block randomizer for balanced assignment to experiment arms.

    Maintains per-experiment shuffled blocks of arm indices, drawing without
    replacement within each block to guarantee balance over every block of
    consecutive assignments.

    For uniform allocation the base block has one entry per arm; for weighted
    allocation the base block is sized via LCM of weight-derived probability
    denominators so that each arm appears in exact proportion.  The base block
    is then tiled `block_multiple` times, subject to `max_block_size`.

    Args:
        block_multiple: Number of times the base treatment cycle is tiled
            per block.
        max_block_size: Upper bound on block length (number of ints stored).
            If `base_size * block_multiple` exceeds this, the effective
            multiple is reduced (minimum 1).
    """

    def __init__(self, block_multiple: int = 10, max_block_size: int = 1000) -> None:
        if block_multiple < 1:
            raise ValueError(f"block_multiple must be >= 1, got {block_multiple}")
        if max_block_size < 1:
            raise ValueError(f"max_block_size must be >= 1, got {max_block_size}")
        self._block_multiple = block_multiple
        self._max_block_size = max_block_size

        self._blocks: dict[BlockKey, deque[int]] = {}
        self._lock = Lock()

    def random_index_for(
        self,
        identifier: str,
        length: int,
        weights: Sequence[float] | None = None,
    ) -> int:
        """Return a random arm index in `[0, length)` via permuted block randomization.

        When *weights* is `None` every arm is equally likely within each
        block.  When provided, arms appear in proportion to their weights.

        Args:
            identifier: Experiment identifier (block-map key).
            length: Number of arms.  Must be >= 1.
            weights: Optional per-arm weights (need not sum to 1).
                Length must equal *length* and every weight must be > 0.

        Returns:
            An integer in `[0, length)`.

        Raises:
            ValueError: On invalid *length* or *weights*.
        """
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")

        weights_key: tuple[float, ...] | None = None
        if weights is not None:
            if len(weights) != length:
                raise ValueError(f"len(weights) must equal length ({length}), got {len(weights)}")
            if any(w <= 0 for w in weights):
                raise ValueError("all weights must be > 0")
            weights_key = tuple(weights)

        key = (identifier, length, weights_key)
        with self._lock:
            block = self._blocks.get(key)
            if not block:  # None or empty deque
                block = self._make_block(length, weights_key)
                self._blocks[key] = block
            return block.popleft()

    def _make_block(self, length: int, weights: tuple[float, ...] | None = None) -> deque[int]:
        if weights is not None:
            total = sum(weights)
            probs = [w / total for w in weights]
            lcm_denom = get_lcm_prob_denominators(probs)
            base_counts = [round(lcm_denom * p) for p in probs]
        else:
            base_counts = [1] * length

        base_size = sum(base_counts)
        effective_multiple = max(1, min(self._block_multiple, self._max_block_size // base_size))

        indices: list[int] = []
        for idx, count in enumerate(base_counts):
            indices.extend([idx] * (count * effective_multiple))

        np.random.default_rng().shuffle(indices)
        return deque(indices)
