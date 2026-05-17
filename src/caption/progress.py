"""Console progress helpers."""

import asyncio
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Any, Awaitable, Callable, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")
R = TypeVar("R")


def log_step(message: str, icon: str = "•") -> None:
    """
    Print a timestamped progress message without breaking progress bars.

    Parameters
    ----------
    message : str
        Message to print.
    icon : str
        Short status icon.
    """
    now = datetime.now().strftime("%H:%M:%S")
    tqdm.write(f"[{now}] {icon} {message}")


def progress_iter(items: Iterable[T], total: int, desc: str, unit: str) -> Iterator[T]:
    """
    Wrap an iterable with a tqdm progress bar.

    Parameters
    ----------
    items : Iterable[T]
        Items to iterate.
    total : int
        Total item count.
    desc : str
        Progress bar label.
    unit : str
        Unit label.

    Yields
    ------
    T
        Items from the input iterable.
    """
    yield from tqdm(items, total=total, desc=desc, unit=unit)


async def progress_gather(
    items: list[T], concurrency: int, desc: str, unit: str, worker: Callable[[T], Awaitable[R]]
) -> list[R]:
    """
    Run async work items concurrently while preserving result order.

    Parameters
    ----------
    items : list[T]
        Work items.
    concurrency : int
        Maximum number of concurrent workers.
    desc : str
        Progress bar label.
    unit : str
        Unit label.
    worker : Callable[[T], Awaitable[R]]
        Async function executed for each item.

    Returns
    -------
    list[R]
        Results ordered like the input items.
    """
    if not items:
        return []

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(index: int, item: T) -> tuple[int, R]:
        async with semaphore:
            return index, await worker(item)

    tasks = [asyncio.create_task(run_one(index, item)) for index, item in enumerate(items)]
    results: list[R | None] = [None] * len(items)
    with tqdm(total=len(items), desc=desc, unit=unit) as bar:
        for task in asyncio.as_completed(tasks):
            index, result = await task
            results[index] = result
            bar.update(1)

    return [result for result in results if result is not None]


class AsrProgress:
    """Convert mlx-qwen3-asr progress events into a tqdm progress bar."""

    def __init__(self) -> None:
        self._bar: tqdm | None = None
        self._completed = 0

    def __call__(self, event: dict[str, Any]) -> None:
        """
        Handle one ASR progress event.

        Parameters
        ----------
        event : dict[str, Any]
            Event emitted by mlx-qwen3-asr.
        """
        name = str(event.get("event", ""))
        if name == "chunks_prepared":
            total = int(event.get("total_chunks", 0))
            audio_seconds = float(event.get("audio_duration_sec", 0.0))
            log_step(f"ASR prepared {total} chunk(s), audio {audio_seconds:.1f}s", icon="🎧")
            self._bar = tqdm(total=total, desc="ASR", unit="chunk")
            self._completed = 0
            return

        if name == "chunk_completed" and self._bar is not None:
            chunk_index = int(event.get("chunk_index", 0))
            increment = max(0, chunk_index - self._completed)
            if increment:
                self._bar.update(increment)
                self._completed = chunk_index

            total = int(event.get("total_chunks", 0))
            if total and self._completed >= total:
                self._bar.close()
                self._bar = None
