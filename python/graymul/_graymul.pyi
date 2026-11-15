from typing import Any, Callable, Optional

def multiply_decimal(
    a: str,
    b: str,
    algorithm: str | None = ...,
    threads: int | None = ...,
    timeout_seconds: int | None = ...,
    max_memory_mb: int | None = ...,
    seed: int | None = ...,
    max_cpu_seconds: int | None = ...,
    perf_mode: bool = ...,
) -> str: ...
def multiply_with_progress_decimal(
    a: str,
    b: str,
    callback: Callable[[dict[str, Any]], Optional[bool]] | None = ...,
    algorithm: str | None = ...,
    threads: int | None = ...,
    timeout_seconds: int | None = ...,
    max_memory_mb: int | None = ...,
    seed: int | None = ...,
    max_cpu_seconds: int | None = ...,
    perf_mode: bool = ...,
) -> str: ...
def stage_timings_decimal(
    a: str,
    b: str,
    algorithm: str | None = ...,
    threads: int | None = ...,
    timeout_seconds: int | None = ...,
    max_memory_mb: int | None = ...,
    seed: int | None = ...,
    max_cpu_seconds: int | None = ...,
    perf_mode: bool = ...,
) -> list[dict[str, Any]]: ...
def multiply_report_decimal(
    a: str,
    b: str,
    algorithm: str | None = ...,
    threads: int | None = ...,
    timeout_seconds: int | None = ...,
    max_memory_mb: int | None = ...,
    seed: int | None = ...,
    max_cpu_seconds: int | None = ...,
    perf_mode: bool = ...,
) -> dict[str, Any]: ...
def version() -> str: ...
