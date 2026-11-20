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

fn split_sign(text: &str) -> (bool, &str) {
    if let Some(rest) = text.strip_prefix('-') {
        (true, rest)
    } else if let Some(rest) = text.strip_prefix('+') {
        (false, rest)
    } else {
        (false, text)
    }
}

fn parse_operand(text: &str) -> Result<(bool, BigUint), MultiplyError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(MultiplyError::Input("empty integer".to_string()));
    }
    let (negative, digits) = split_sign(trimmed);
    if digits.is_empty() {
        return Err(MultiplyError::Input(invalid_integer_message(trimmed)));
    }
    BigUint::from_str(digits)
        .map(|value| (negative, value))
        .map_err(|_| MultiplyError::Input(invalid_integer_message(trimmed)))
}

fn apply_sign(product: String, negative: bool) -> String {
    if negative && product != "0" {
        let mut signed = String::with_capacity(product.len() + 1);
        signed.push('-');
        signed.push_str(&product);
        signed
    } else {
        product
    }
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
    let (neg_a, a_value) = match parse_operand(a) {
        Ok(pair) => pair,
        Err(error) => return Err(to_py_error(error)),
    };
    let (neg_b, b_value) = match parse_operand(b) {
        Ok(pair) => pair,
        Err(error) => return Err(to_py_error(error)),
    };
    let report = py
        .detach(|| graymul_core::multiply(&a_value, &b_value, &options))
        .map_err(to_py_error)?;
    Ok(apply_sign(report.product, neg_a != neg_b))
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
            let (neg_a, a_value) = parse_operand(a).map_err(to_py_error)?;
            let (neg_b, b_value) = parse_operand(b).map_err(to_py_error)?;
            let negative = neg_a != neg_b;
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
                .map(|report| apply_sign(report.product, negative))
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
    let (_, a_value) = parse_operand(a).map_err(to_py_error)?;
    let (_, b_value) = parse_operand(b).map_err(to_py_error)?;
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
    let (neg_a, a_value) = parse_operand(a).map_err(to_py_error)?;
    let (neg_b, b_value) = parse_operand(b).map_err(to_py_error)?;
    let negative = neg_a != neg_b;
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
    dict.set_item("product", apply_sign(report.product, negative))?;
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

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (a, b, threads=None, timeout_seconds=None, max_memory_mb=None, max_cpu_seconds=None))]
fn explain_decimal(
    py: Python<'_>,
    a: &str,
    b: &str,
    threads: Option<usize>,
    timeout_seconds: Option<u64>,
    max_memory_mb: Option<usize>,
    max_cpu_seconds: Option<u64>,
) -> PyResult<Py<PyDict>> {
    let options = MultiplyOptions {
        algorithm: None,
        threads,
        max_duration: timeout_seconds.map(std::time::Duration::from_secs),
        max_memory_mb,
        max_cpu_seconds,
        seed: MultiplyOptions::default().seed,
        perf_mode: false,
    };
    let (neg_a, a_value) = parse_operand(a).map_err(to_py_error)?;
    let (neg_b, b_value) = parse_operand(b).map_err(to_py_error)?;
    let plan = py.detach(|| {
        graymul_core::explain::explain_bigints(&a_value, &b_value, neg_a, neg_b, &options)
    });
    let dict = PyDict::new(py);
    let operands = PyDict::new(py);
    operands.set_item("limbs_a", plan.operands.limbs_a)?;
    operands.set_item("limbs_b", plan.operands.limbs_b)?;
    operands.set_item("limbs_max", plan.operands.limbs_max)?;
    operands.set_item("ratio", plan.operands.ratio)?;
    operands.set_item("negative_a", plan.operands.negative_a)?;
    operands.set_item("negative_b", plan.operands.negative_b)?;
    operands.set_item("negative", plan.operands.negative)?;
    let dispatch = PyDict::new(py);
    dispatch.set_item("static_route", plan.dispatch.static_route)?;
    dispatch.set_item("selected", plan.dispatch.selected)?;
    dispatch.set_item("reason", plan.dispatch.reason)?;
    dispatch.set_item("warmed_override", plan.dispatch.warmed_override)?;
    let resources = PyDict::new(py);
    resources.set_item("workers", plan.resources.workers)?;
    resources.set_item("est_memory_bytes", plan.resources.est_memory_bytes)?;
    resources.set_item("timeout_ms", plan.resources.timeout_ms)?;
    let verification = PyDict::new(py);
    verification.set_item("methods", plan.verification.methods)?;
    dict.set_item("operands", operands)?;
    dict.set_item("dispatch", dispatch)?;
    dict.set_item("resources", resources)?;
    dict.set_item("verification", verification)?;
    Ok(dict.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_options() -> MultiplyOptions {
        build_options(None, None, None, None, None, None, false)
    }

    #[test]
    fn parse_operand_accepts_sign_prefix() {
        assert_eq!(
            parse_operand("-5").expect("minus"),
            (true, BigUint::from(5u32))
        );
        assert_eq!(
            parse_operand("+5").expect("plus"),
            (false, BigUint::from(5u32))
        );
        assert_eq!(
            parse_operand("  -007  ").expect("padded"),
            (true, BigUint::from(7u32))
        );
        assert_eq!(
            parse_operand("42").expect("plain"),
            (false, BigUint::from(42u32))
        );
        for text in ["", "   ", "-", "+", "--5", "- 5", "12x34"] {
            assert!(parse_operand(text).is_err(), "text {text} should fail");
        }
    }

    #[test]
    fn apply_sign_prefixes_nonzero_negative_product() {
        assert_eq!(apply_sign("408".to_string(), true), "-408");
        assert_eq!(apply_sign("408".to_string(), false), "408");
        assert_eq!(apply_sign("0".to_string(), true), "0");
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
    module.add_function(wrap_pyfunction!(explain_decimal, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}
