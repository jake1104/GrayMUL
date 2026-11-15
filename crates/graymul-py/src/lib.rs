use graymul_core::{MultiplyError, MultiplyOptions, ProgressEvent, StageTiming, TimingCollector};
use num_bigint::BigUint;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::str::FromStr;
use std::sync::atomic::AtomicBool;

fn error_category(error: &MultiplyError) -> &'static str {
    match error {
        MultiplyError::Input(_) => "input",
        MultiplyError::ResourceLimit(_) => "resource-limit",
        MultiplyError::Cancelled(_) => "cancelled",
        MultiplyError::Internal(_) => "internal",
    }
}

fn error_detail(error: &MultiplyError) -> String {
    match error {
        MultiplyError::Input(detail)
        | MultiplyError::ResourceLimit(detail)
        | MultiplyError::Cancelled(detail)
        | MultiplyError::Internal(detail) => detail.clone(),
    }
}

fn py_error_message(error: &MultiplyError) -> String {
    format!("{}: {}", error_category(error), error_detail(error))
}

fn to_py_error(error: MultiplyError) -> PyErr {
    PyValueError::new_err(py_error_message(&error))
}

fn stage_timing_to_py(py: Python<'_>, entry: StageTiming) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("stage", entry.stage)?;
    dict.set_item("degree", entry.degree)?;
    dict.set_item("elapsed_ms", entry.elapsed_ms)?;
    dict.set_item("detail", entry.detail)?;
    Ok(dict.unbind())
}

fn parse_operand(text: &str) -> Result<BigUint, MultiplyError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(MultiplyError::Input("empty integer".to_string()));
    }
    BigUint::from_str(trimmed).map_err(|_| MultiplyError::Input(invalid_integer_message(trimmed)))
}

fn invalid_integer_message(text: &str) -> String {
    const LIMIT: usize = 64;
    if text.len() <= LIMIT {
        return format!("invalid integer {text}");
    }
    let shown: String = text.chars().take(LIMIT).collect();
    format!("invalid integer {shown}...<{} chars>", text.chars().count())
}

#[allow(clippy::too_many_arguments)]
fn build_options(
    algorithm: Option<String>,
    threads: Option<usize>,
    timeout_seconds: Option<u64>,
    max_memory_mb: Option<usize>,
    seed: Option<u64>,
    max_cpu_seconds: Option<u64>,
    perf_mode: bool,
) -> MultiplyOptions {
    MultiplyOptions {
        algorithm,
        threads,
        max_duration: timeout_seconds.map(std::time::Duration::from_secs),
        max_memory_mb,
        max_cpu_seconds,
        seed: seed.unwrap_or(MultiplyOptions::default().seed),
        perf_mode,
    }
}

fn internal_error(error: PyErr) -> MultiplyError {
    MultiplyError::Internal(error.to_string())
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (a, b, algorithm=None, threads=None, timeout_seconds=None, max_memory_mb=None, seed=None, max_cpu_seconds=None, perf_mode=false))]
fn multiply_decimal(
    py: Python<'_>,
    a: &str,
    b: &str,
    algorithm: Option<String>,
    threads: Option<usize>,
    timeout_seconds: Option<u64>,
    max_memory_mb: Option<usize>,
    seed: Option<u64>,
    max_cpu_seconds: Option<u64>,
    perf_mode: bool,
) -> PyResult<String> {
    let options = build_options(
        algorithm,
        threads,
        timeout_seconds,
        max_memory_mb,
        seed,
        max_cpu_seconds,
        perf_mode,
    );
    let a_value = parse_operand(a).map_err(to_py_error)?;
    let b_value = parse_operand(b).map_err(to_py_error)?;
    let report = py
        .detach(|| graymul_core::multiply(&a_value, &b_value, &options))
        .map_err(to_py_error)?;
    Ok(report.product)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (a, b, callback=None, algorithm=None, threads=None, timeout_seconds=None, max_memory_mb=None, seed=None, max_cpu_seconds=None, perf_mode=false))]
fn multiply_with_progress_decimal(
    py: Python<'_>,
    a: &str,
    b: &str,
    callback: Option<Py<PyAny>>,
    algorithm: Option<String>,
    threads: Option<usize>,
    timeout_seconds: Option<u64>,
    max_memory_mb: Option<usize>,
    seed: Option<u64>,
    max_cpu_seconds: Option<u64>,
    perf_mode: bool,
) -> PyResult<String> {
    let options = build_options(
        algorithm,
        threads,
        timeout_seconds,
        max_memory_mb,
        seed,
        max_cpu_seconds,
        perf_mode,
    );
    if let Some(callback) = callback {
        py.detach(|| {
            let a_value = parse_operand(a).map_err(to_py_error)?;
            let b_value = parse_operand(b).map_err(to_py_error)?;
            let flag = AtomicBool::new(false);
            Python::attach(|py| {
                let mut bridge = |event: &ProgressEvent| -> Result<(), MultiplyError> {
                    let event_dict = PyDict::new(py);
                    event_dict
                        .set_item("stage", &event.stage)
                        .map_err(internal_error)?;
                    event_dict
                        .set_item("current", event.current)
                        .map_err(internal_error)?;
                    event_dict
                        .set_item("total", event.total)
                        .map_err(internal_error)?;
                    event_dict
                        .set_item("message", &event.message)
                        .map_err(internal_error)?;
                    callback
                        .call1(py, (event_dict,))
                        .map_err(internal_error)
                        .and_then(|outcome| {
                            let cont = outcome.is_none(py)
                                || outcome.is_truthy(py).map_err(internal_error)?;
                            if cont {
                                Ok(())
                            } else {
                                Err(MultiplyError::Cancelled(
                                    "the operation was cancelled".to_string(),
                                ))
                            }
                        })
                };
                graymul_core::multiply_with_progress(
                    &a_value,
                    &b_value,
                    &options,
                    &flag,
                    &mut bridge,
                    None,
                )
                .map(|report| report.product)
                .map_err(to_py_error)
            })
        })
    } else {
        multiply_decimal(
            py,
            a,
            b,
            options.algorithm,
            options.threads,
            options.max_duration.map(|span| span.as_secs()),
            options.max_memory_mb,
            Some(options.seed),
            options.max_cpu_seconds,
            options.perf_mode,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (a, b, algorithm=None, threads=None, timeout_seconds=None, max_memory_mb=None, seed=None, max_cpu_seconds=None, perf_mode=false))]
fn stage_timings_decimal(
    py: Python<'_>,
    a: &str,
    b: &str,
    algorithm: Option<String>,
    threads: Option<usize>,
    timeout_seconds: Option<u64>,
    max_memory_mb: Option<usize>,
    seed: Option<u64>,
    max_cpu_seconds: Option<u64>,
    perf_mode: bool,
) -> PyResult<Vec<Py<PyAny>>> {
    let options = build_options(
        algorithm,
        threads,
        timeout_seconds,
        max_memory_mb,
        seed,
        max_cpu_seconds,
        perf_mode,
    );
    let a_value = parse_operand(a).map_err(to_py_error)?;
    let b_value = parse_operand(b).map_err(to_py_error)?;
    let collector = TimingCollector::new();
    let flag = AtomicBool::new(false);
    let mut sink = |_: &ProgressEvent| Ok(());
    py.detach(|| {
        graymul_core::multiply_with_progress(
            &a_value,
            &b_value,
            &options,
            &flag,
            &mut sink,
            Some(&collector),
        )
    })
    .map_err(to_py_error)?;
    collector
        .snapshot()
        .into_iter()
        .map(|entry| Ok(stage_timing_to_py(py, entry)?.into_any()))
        .collect()
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (a, b, algorithm=None, threads=None, timeout_seconds=None, max_memory_mb=None, seed=None, max_cpu_seconds=None, perf_mode=false))]
fn multiply_report_decimal(
    py: Python<'_>,
    a: &str,
    b: &str,
    algorithm: Option<String>,
    threads: Option<usize>,
    timeout_seconds: Option<u64>,
    max_memory_mb: Option<usize>,
    seed: Option<u64>,
    max_cpu_seconds: Option<u64>,
    perf_mode: bool,
) -> PyResult<Py<PyDict>> {
    let options = build_options(
        algorithm,
        threads,
        timeout_seconds,
        max_memory_mb,
        seed,
        max_cpu_seconds,
        perf_mode,
    );
    let a_value = parse_operand(a).map_err(to_py_error)?;
    let b_value = parse_operand(b).map_err(to_py_error)?;
    let collector = TimingCollector::new();
    let flag = AtomicBool::new(false);
    let mut sink = |_: &ProgressEvent| Ok(());
    let report = py
        .detach(|| {
            graymul_core::multiply_with_progress(
                &a_value,
                &b_value,
                &options,
                &flag,
                &mut sink,
                Some(&collector),
            )
        })
        .map_err(to_py_error)?;
    let dict = PyDict::new(py);
    dict.set_item("product", report.product)?;
    dict.set_item("method", report.method)?;
    dict.set_item("limbs", report.limbs)?;
    dict.set_item("elapsed_ms", report.elapsed_ms)?;
    let entries: PyResult<Vec<Py<PyDict>>> = collector
        .snapshot()
        .into_iter()
        .map(|entry| stage_timing_to_py(py, entry))
        .collect();
    dict.set_item("timings", entries?)?;
    Ok(dict.unbind())
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_options() -> MultiplyOptions {
        build_options(None, None, None, None, None, None, false)
    }

    #[test]
    fn build_options_carries_memory_budget() {
        let options = build_options(None, None, None, Some(512), None, None, false);
        assert_eq!(options.max_memory_mb, Some(512));
        assert_eq!(options.memory_budget_bytes(), 512 * 1024 * 1024);
    }

    #[test]
    fn build_options_carries_cpu_budget() {
        let options = build_options(None, None, None, None, None, Some(30), false);
        assert_eq!(options.max_cpu_seconds, Some(30));
        assert_eq!(options.cpu_budget_ms(), Some(30_000));
    }

    #[test]
    fn build_options_defaults_seed_and_perf_mode() {
        let options = base_options();
        assert_eq!(options.seed, MultiplyOptions::default().seed);
        assert!(!options.perf_mode);
        assert!(options.validate().is_ok());
    }

    #[test]
    fn core_multiply_returns_verified_product() {
        let report = graymul_core::multiply(
            &BigUint::from(12u32),
            &BigUint::from(34u32),
            &base_options(),
        )
        .expect("multiply");
        assert_eq!(report.product, "408");
        assert!(graymul_core::Engine::from_str(&report.method).is_some());
    }

    #[test]
    fn core_multiply_rejects_bad_algorithm() {
        let options = build_options(
            Some("gnfs".to_string()),
            None,
            None,
            None,
            None,
            None,
            false,
        );
        assert!(matches!(
            graymul_core::multiply(&BigUint::from(12u32), &BigUint::from(34u32), &options),
            Err(MultiplyError::Input(_))
        ));
    }

    #[test]
    fn core_multiply_zero_timeout_expires() {
        let options = build_options(None, None, Some(0), None, None, None, false);
        assert!(matches!(
            graymul_core::multiply(&BigUint::from(12u32), &BigUint::from(34u32), &options),
            Err(MultiplyError::ResourceLimit(_))
        ));
    }

    #[test]
    fn core_multiply_zero_memory_budget_rejects() {
        let options = build_options(None, None, None, Some(0), None, None, false);
        assert!(matches!(
            graymul_core::multiply(&BigUint::from(12u32), &BigUint::from(34u32), &options),
            Err(MultiplyError::ResourceLimit(_))
        ));
    }

    #[test]
    fn report_timings_use_real_stages() {
        const STAGES: [&str; 12] = [
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
        ];
        let collector = TimingCollector::new();
        let flag = AtomicBool::new(false);
        let mut sink = |_: &ProgressEvent| Ok(());
        let report = graymul_core::multiply_with_progress(
            &BigUint::from(12u32),
            &BigUint::from(34u32),
            &base_options(),
            &flag,
            &mut sink,
            Some(&collector),
        )
        .expect("multiply with timings");
        assert!(graymul_core::Engine::from_str(&report.method).is_some());
        let records = collector.snapshot();
        assert!(!records.is_empty());
        for record in &records {
            assert!(STAGES.contains(&record.stage.as_str()));
        }
        assert!(records.iter().any(|record| record.stage == "pilot"));
        assert!(records.iter().any(|record| record.stage == report.method));
    }

    #[test]
    fn error_category_maps_input() {
        let error = MultiplyError::Input("bad digit".to_string());
        assert_eq!(error_category(&error), "input");
        assert!(py_error_message(&error).starts_with("input: "));
    }

    #[test]
    fn error_category_maps_resource_limit() {
        let error = MultiplyError::ResourceLimit("deadline".to_string());
        assert_eq!(error_category(&error), "resource-limit");
        assert!(py_error_message(&error).starts_with("resource-limit: "));
    }

    #[test]
    fn error_category_maps_cancelled() {
        let error = MultiplyError::Cancelled("the operation was cancelled".to_string());
        assert_eq!(error_category(&error), "cancelled");
        assert!(py_error_message(&error).starts_with("cancelled: "));
    }

    #[test]
    fn error_category_maps_internal() {
        let error = MultiplyError::Internal("verification failed".to_string());
        assert_eq!(error_category(&error), "internal");
        assert!(py_error_message(&error).starts_with("internal: "));
    }

    #[test]
    fn py_error_message_carries_detail() {
        let error = MultiplyError::Input("invalid integer abc".to_string());
        assert_eq!(py_error_message(&error), "input: invalid integer abc");
    }
}

#[pymodule]
fn _graymul(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(multiply_decimal, module)?)?;
    module.add_function(wrap_pyfunction!(multiply_with_progress_decimal, module)?)?;
    module.add_function(wrap_pyfunction!(stage_timings_decimal, module)?)?;
    module.add_function(wrap_pyfunction!(multiply_report_decimal, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}
