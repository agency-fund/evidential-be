"""Permuted block randomization for balanced online arm assignment.

Provides a thread-safe in-memory block randomizer that guarantees perfect
balance over every `length * block_multiple` consecutive draws for a given
experiment, while preserving marginal randomization probabilities.
"""

from collections import deque
from threading import Lock

import numpy as np


class BlockRandomize:
    """Thread-safe permuted block randomizer for balanced assignment to experiment arms.

    Maintains per-experiment shuffled blocks of arm indices, drawing without
    replacement within each block to guarantee balance over every
    `length * block_multiple` consecutive assignments.

    Args:
        block_multiple: Number of complete treatment cycles per block.
            A block for `length` arms contains `length * block_multiple`
            entries (each index 0..length-1 repeated `block_multiple` times).
    """

    def __init__(self, block_multiple: int = 10) -> None:
        if block_multiple < 1:
            raise ValueError(f"block_multiple must be >= 1, got {block_multiple}")
        self._block_multiple = block_multiple
        self._blocks: dict[tuple[str, int], deque[int]] = {}
        self._lock = Lock()

    def random_index_for(self, identifier: str, length: int) -> int:
        """Return a random index in [0, length) via permuted block randomization.

        Within every `length * block_multiple` draws for the same
        `(identifier, length)` pair, each index appears exactly `block_multiple`
        times.

        Returns:
            An integer in `[0, length)`.

        Raises:
            ValueError: If `length` < 1.
        """
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")

        key = (identifier, length)
        with self._lock:
            block = self._blocks.get(key)
            if not block:  # None or empty deque
                block = self._make_block(length)
                self._blocks[key] = block
            return block.popleft()

    def _make_block(self, length: int) -> deque[int]:
        indices = list(range(length)) * self._block_multiple
        np.random.default_rng().shuffle(indices)
        return deque(indices)
