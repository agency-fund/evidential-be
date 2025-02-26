import time
from collections import OrderedDict
from threading import Lock
from typing import Generic, TypeVar
from collections.abc import Callable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class PydanticLRUCache(Generic[T]):
    """
    Thread-safe LRU cache for Pydantic models with string keys.
    """

    def __init__(self, max_size: int, max_age: float | None = None):
        """
        Initialize a new cache with a maximum size and optional max age.

        Args:
            max_size: Maximum number of items to store in the cache
            max_age: Maximum age of cache entries in seconds (None means no expiration)
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self.max_size = max_size
        self.max_age = max_age
        self.cache = OrderedDict()  # type: OrderedDict[str, Tuple[T, float]]
        self.locks = {}  # type: Dict[str, Lock]
        self.global_lock = Lock()

    def get(self, key: str, fn: Callable[[], T], /, refresh: bool = False) -> T:
        """
        Get a value by key, fetching it with fn if not in cache or if expired.

        Args:
            key: The cache key (string)
            fn: Function to call to get the value if not in cache or if ignore=True
            refresh: If True, always call fn and update the cache with the result

        Returns:
            The cached or newly fetched Pydantic model
        """
        current_time = time.time()

        # Get or create a lock for this key
        with self.global_lock:
            if key not in self.locks:
                self.locks[key] = Lock()

        # Acquire the lock for this specific key
        with self.locks[key]:
            # Check if we need to fetch a fresh value
            need_fresh_value = (
                refresh
                or key not in self.cache
                or (
                    self.max_age is not None
                    and current_time - self.cache[key][1] > self.max_age
                )
            )

            if need_fresh_value:
                # Fetch fresh value
                value = fn()

                # Ensure value is a Pydantic model
                if not isinstance(value, BaseModel):
                    raise TypeError(
                        f"Cache values must be Pydantic models, got {type(value)}"
                    )

                # Update cache with value and timestamp
                with self.global_lock:
                    self.cache[key] = (value, current_time)

                    # Move to end (most recently used)
                    self.cache.move_to_end(key)

                    # Evict oldest item if over capacity
                    if len(self.cache) > self.max_size:
                        self.cache.popitem(last=False)

                return value
            # Use cached value
            with self.global_lock:
                value, _ = self.cache[key]
                # Update timestamp to current time (access time)
                self.cache[key] = (value, current_time)
                # Move to end (most recently used)
                self.cache.move_to_end(key)

            return value

    def invalidate(self, key: str) -> None:
        """
        Remove a key from the cache.

        Args:
            key: The cache key to remove
        """
        with self.global_lock:
            if key in self.cache:
                del self.cache[key]

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            Number of entries removed
        """
        if self.max_age is None:
            return 0

        removed = 0
        current_time = time.time()

        with self.global_lock:
            # Create a list of keys to remove (can't modify during iteration)
            expired_keys = [
                k
                for k, (_, timestamp) in self.cache.items()
                if current_time - timestamp > self.max_age
            ]

            # Remove all expired keys
            for key in expired_keys:
                del self.cache[key]
                removed += 1

        return removed

    def clear(self) -> None:
        """
        Clear all items from the cache.
        """
        with self.global_lock:
            self.cache.clear()

    def __len__(self) -> int:
        """
        Return the current number of items in the cache.
        """
        with self.global_lock:
            return len(self.cache)

    def keys(self) -> list[str]:
        """
        Return a list of all keys in the cache.
        """
        with self.global_lock:
            return list(self.cache.keys())
