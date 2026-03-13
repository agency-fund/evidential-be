import dataclasses
import time
from collections.abc import Iterator
from contextlib import contextmanager

from loguru import logger


@dataclasses.dataclass(slots=True)
class TimingResult:
    elapsed: float = 0


@contextmanager
def timing(message: str | None = None) -> Iterator[TimingResult]:
    result = TimingResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed = time.perf_counter() - start
        if message:
            logger.info("Timing: {}: elapsed={}s", message, result.elapsed)
