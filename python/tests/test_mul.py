import pytest

import graymul

REAL_STAGES = {
    "pilot",
    "schoolbook",
    "karatsuba",
    "toom3",
    "toom4",
    "ssa",
    "furer",
    "hvh",
    "hvh-full",
    "trivial",
    "reclaim",
    "verify",
}

REAL_METHODS = {
    "schoolbook",
    "karatsuba",
    "toom3",
    "toom4",
    "ssa",
    "furer",
    "hvh",
    "hvh-full",
    "trivial",
}


def test_multiply_basic() -> None:
    assert graymul.multiply(12, 34) == 408
    assert graymul.multiply(123456789, 987654321) == 121932631112635269
    assert graymul.multiply(0, 999) == 0
    assert graymul.multiply(1, 1) == 1


def test_multiply_rejects_bool_with_type_error() -> None:
    with pytest.raises(TypeError):
        graymul.multiply(True, 3)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        graymul.multiply(3, False)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        graymul.multiply("12", 3)  # type: ignore[arg-type]


def test_multiply_rejects_negative_with_input_error() -> None:
    with pytest.raises(graymul.GrayMULError) as excinfo:
        graymul.multiply(-5, 3)
    assert excinfo.value.category == "input"


def test_multiply_unknown_algorithm_is_input_error() -> None:
    with pytest.raises(graymul.GrayMULError) as excinfo:
        graymul.multiply(12, 34, algorithm="gnfs")
    assert excinfo.value.category == "input"


def test_multiply_zero_timeout_is_resource_limit() -> None:
    with pytest.raises(graymul.GrayMULError) as excinfo:
        graymul.multiply(12, 34, timeout_seconds=0)
    assert excinfo.value.category == "resource-limit"


def test_progress_false_cancels() -> None:
    seen: list[dict] = []

    def cancel(event: dict) -> bool:
        seen.append(event)
        return False

    try:
        graymul.multiply_with_progress(12, 34, cancel)
    except graymul.GrayMULError as exc:
        assert exc.category == "cancelled"
        assert exc.detail == "the operation was cancelled"
        assert str(exc) == "cancelled: the operation was cancelled"
    else:
        raise AssertionError("expected GrayMULError")
    assert seen
    assert seen[0]["stage"] in REAL_STAGES


def test_progress_true_continues() -> None:
    assert graymul.multiply_with_progress(12, 34, lambda event: True) == 408


def test_progress_none_continues() -> None:
    assert graymul.multiply_with_progress(12, 34, lambda event: None) == 408


def test_progress_requires_callable() -> None:
    with pytest.raises(TypeError):
        graymul.multiply_with_progress(12, 34, False)  # type: ignore[arg-type]


def test_multiply_report_keys() -> None:
    report = graymul.multiply_report(12, 34)
    assert report["product"] == 408
    assert report["method"] in REAL_METHODS
    assert report["limbs"] == 1
    assert isinstance(report["elapsed_ms"], int)
    assert len(report["timings"]) >= 1
    for entry in report["timings"]:
        assert entry["stage"] in REAL_STAGES
        assert entry["degree"] is None
        assert isinstance(entry["elapsed_ms"], int)
        assert "limbs=" in entry["detail"]
    assert any(entry["stage"] == report["method"] for entry in report["timings"])


def test_explicit_hvh_method() -> None:
    assert graymul.multiply(12, 34, algorithm="hvh") == 408
    report = graymul.multiply_report(12, 34, algorithm="hvh")
    assert report["method"] == "hvh"
    assert report["product"] == 408


def test_stage_timings_shape() -> None:
    entries = graymul.stage_timings(12, 34)
    assert len(entries) >= 1
    assert set(entries[0].keys()) == {"stage", "degree", "elapsed_ms", "detail"}
    assert all(entry["stage"] in REAL_STAGES for entry in entries)
    assert any(entry["stage"] == "pilot" for entry in entries)


def test_multiply_batch_fail_fast_and_record() -> None:
    with pytest.raises(graymul.GrayMULError):
        graymul.multiply_batch([(2, 3), (-1, 5), (4, 5)])
    results = graymul.multiply_batch([(2, 3), (-1, 5), (4, 5)], fail_fast=False)
    assert results[0] == 6
    assert isinstance(results[1], graymul.GrayMULError)
    assert results[1].category == "input"
    assert results[2] == 20


def test_version() -> None:
    assert graymul.version() == "0.1.0"


def test_oversized_int_maps_to_input_category() -> None:
    with pytest.raises(graymul.GrayMULError) as info:
        graymul.multiply(10**5000, 3)
    assert info.value.category == "input"
    assert "Exceeds the limit" in info.value.detail
    with pytest.raises(graymul.GrayMULError) as product_info:
        graymul.multiply(10**2500, 10**2500)
    assert product_info.value.category == "input"


def test_docstrings_make_no_ignored_threads_claim() -> None:
    funcs = (
        graymul.multiply,
        graymul.multiply_with_progress,
        graymul.stage_timings,
        graymul.multiply_batch,
        graymul.multiply_report,
    )
    for func in funcs:
        assert func.__doc__ is not None
        assert "ignor" not in func.__doc__
