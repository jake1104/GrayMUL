from typing import Any, Callable, Optional

ProgressCallback = Callable[[dict[str, Any]], Optional[bool]]

class GrayMULError(ValueError):
    category: str
    detail: str
    def __init__(self, category: str, detail: str) -> None: ...

def multiply(
    a: int,
    b: int,
    *,
    algorithm: str | None = None,
    threads: int | None = None,
    timeout_seconds: int | None = None,
    max_memory_mb: int | None = None,
    seed: int | None = None,
    max_cpu_seconds: int | None = None,
    perf_mode: bool = False,
) -> int: ...
def multiply_with_progress(
    a: int,
    b: int,
    callback: ProgressCallback,
    *,
    algorithm: str | None = None,
    threads: int | None = None,
    timeout_seconds: int | None = None,
    max_memory_mb: int | None = None,
    seed: int | None = None,
    max_cpu_seconds: int | None = None,
    perf_mode: bool = False,
) -> int: ...
def stage_timings(
    a: int,
    b: int,
    *,
    algorithm: str | None = None,
    threads: int | None = None,
    timeout_seconds: int | None = None,
    max_memory_mb: int | None = None,
    seed: int | None = None,
    max_cpu_seconds: int | None = None,
    perf_mode: bool = False,
) -> list[dict[str, Any]]: ...
def multiply_report(
    a: int,
    b: int,
    *,
    algorithm: str | None = None,
    threads: int | None = None,
    timeout_seconds: int | None = None,
    max_memory_mb: int | None = None,
    seed: int | None = None,
    max_cpu_seconds: int | None = None,
    perf_mode: bool = False,
) -> dict[str, Any]: ...
def version() -> str: ...
def multiply_batch(
    pairs: list[tuple[int, int]],
    *,
    fail_fast: bool = True,
    algorithm: str | None = None,
    threads: int | None = None,
    timeout_seconds: int | None = None,
    max_memory_mb: int | None = None,
    seed: int | None = None,
    max_cpu_seconds: int | None = None,
    perf_mode: bool = False,
) -> list[int | GrayMULError]: ...
