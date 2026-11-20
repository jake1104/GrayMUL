from __future__ import annotations

from typing import Any, Callable, Optional

from . import _graymul

__all__ = [
    "GrayMULError",
    "explain",
    "multiply",
    "multiply_batch",
    "multiply_report",
    "multiply_with_progress",
    "stage_timings",
    "version",
]

ProgressCallback = Callable[[dict[str, Any]], Optional[bool]]


class GrayMULError(ValueError):
    def __init__(self, category: str, detail: str) -> None:
        self.category = category
        self.detail = detail
        super().__init__(category + ": " + detail)


_GRAY_ERROR_PREFIXES = ("input: ", "resource-limit: ", "cancelled: ", "internal: ")


def _to_gray_error(exc: ValueError) -> ValueError:
    message = str(exc)
    for prefix in _GRAY_ERROR_PREFIXES:
        if message.startswith(prefix):
            category, _, detail = message.partition(": ")
            return GrayMULError(category, detail)
    if "Exceeds the limit" in message:
        return GrayMULError("input", message)
    return exc


def _validate_operand(value: int, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(name + " must be an integer")
    return str(value)


def _validate_seed(seed: int | None) -> None:
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if seed < 0:
        raise ValueError("input: seed must be a non-negative integer")


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
) -> int:
    """Multiply a by b; keywords mirror the documented surface (None means auto or unset)."""
    try:
        _validate_seed(seed)
        return int(
            _graymul.multiply_decimal(
                _validate_operand(a, "a"),
                _validate_operand(b, "b"),
                algorithm,
                threads,
                timeout_seconds,
                max_memory_mb,
                seed,
                max_cpu_seconds,
                perf_mode,
            )
        )
    except ValueError as exc:
        raise _to_gray_error(exc)


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
) -> int:
    """Multiply a by b like multiply, but call callback({stage, current, total, message}) at each stage boundary; the callback is required here (the raw binding accepts None and falls back to a plain multiply). Return a falsy value (e.g. False) from the callback to cancel the operation (surfaced as GrayMULError with category=cancelled); truthy values and None continue."""
    if not callable(callback):
        raise TypeError("callback must be callable")
    try:
        _validate_seed(seed)
        return int(
            _graymul.multiply_with_progress_decimal(
                _validate_operand(a, "a"),
                _validate_operand(b, "b"),
                callback,
                algorithm,
                threads,
                timeout_seconds,
                max_memory_mb,
                seed,
                max_cpu_seconds,
                perf_mode,
            )
        )
    except ValueError as exc:
        raise _to_gray_error(exc)


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
) -> list[dict[str, Any]]:
    """Return per-stage timing records [{stage, degree, elapsed_ms, detail}] for multiplying a by b without printing."""
    try:
        _validate_seed(seed)
        return [
            {
                "stage": str(entry["stage"]),
                "degree": None if entry["degree"] is None else int(entry["degree"]),
                "elapsed_ms": int(entry["elapsed_ms"]),
                "detail": str(entry["detail"]),
            }
            for entry in _graymul.stage_timings_decimal(
                _validate_operand(a, "a"),
                _validate_operand(b, "b"),
                algorithm,
                threads,
                timeout_seconds,
                max_memory_mb,
                seed,
                max_cpu_seconds,
                perf_mode,
            )
        ]
    except ValueError as exc:
        raise _to_gray_error(exc)


def version() -> str:
    return str(_graymul.version())


def explain(
    a: int,
    b: int,
    *,
    threads: int | None = None,
    timeout_seconds: int | None = None,
    max_memory_mb: int | None = None,
    max_cpu_seconds: int | None = None,
) -> dict[str, Any]:
    """Return the pure execution plan for multiplying a by b without multiplying: {operands, dispatch, resources, verification}. The dispatch section keeps the static band route and the warmed-cache override on separate lines."""
    try:
        raw = _graymul.explain_decimal(
            _validate_operand(a, "a"),
            _validate_operand(b, "b"),
            threads,
            timeout_seconds,
            max_memory_mb,
            max_cpu_seconds,
        )
    except ValueError as exc:
        raise _to_gray_error(exc)
    operands = raw["operands"]
    dispatch = raw["dispatch"]
    resources = raw["resources"]
    verification = raw["verification"]
    return {
        "operands": {
            "limbs_a": int(operands["limbs_a"]),
            "limbs_b": int(operands["limbs_b"]),
            "limbs_max": int(operands["limbs_max"]),
            "ratio": float(operands["ratio"]),
            "negative_a": bool(operands["negative_a"]),
            "negative_b": bool(operands["negative_b"]),
            "negative": bool(operands["negative"]),
        },
        "dispatch": {
            "static_route": str(dispatch["static_route"]),
            "selected": str(dispatch["selected"]),
            "reason": str(dispatch["reason"]),
            "warmed_override": None
            if dispatch["warmed_override"] is None
            else str(dispatch["warmed_override"]),
        },
        "resources": {
            "workers": int(resources["workers"]),
            "est_memory_bytes": int(resources["est_memory_bytes"]),
            "timeout_ms": None
            if resources["timeout_ms"] is None
            else int(resources["timeout_ms"]),
        },
        "verification": {
            "methods": [str(entry) for entry in verification["methods"]],
        },
    }


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
) -> list[int | GrayMULError]:
    """Multiply each (a, b) pair in pairs with the shared multiply keywords; fail_fast=True raises on the first failure while False records the GrayMULError in place."""
    results: list[int | GrayMULError] = []
    for pair in pairs:
        first, second = pair
        try:
            results.append(
                multiply(
                    first,
                    second,
                    algorithm=algorithm,
                    threads=threads,
                    timeout_seconds=timeout_seconds,
                    max_memory_mb=max_memory_mb,
                    seed=seed,
                    max_cpu_seconds=max_cpu_seconds,
                    perf_mode=perf_mode,
                )
            )
        except GrayMULError as exc:
            if fail_fast:
                raise
            results.append(exc)
    return results


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
) -> dict[str, Any]:
    """Multiply a by b once and return {product, method, limbs, elapsed_ms, timings}."""
    try:
        _validate_seed(seed)
        raw = _graymul.multiply_report_decimal(
            _validate_operand(a, "a"),
            _validate_operand(b, "b"),
            algorithm,
            threads,
            timeout_seconds,
            max_memory_mb,
            seed,
            max_cpu_seconds,
            perf_mode,
        )
    except ValueError as exc:
        raise _to_gray_error(exc)
    return {
        "product": int(raw["product"]),
        "method": str(raw["method"]),
        "limbs": int(raw["limbs"]),
        "elapsed_ms": int(raw["elapsed_ms"]),
        "timings": [
            {
                "stage": str(entry["stage"]),
                "degree": None if entry["degree"] is None else int(entry["degree"]),
                "elapsed_ms": int(entry["elapsed_ms"]),
                "detail": str(entry["detail"]),
            }
            for entry in raw["timings"]
        ],
    }
