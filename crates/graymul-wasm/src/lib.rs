use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use graymul_core::{MultiplyOptions, MultiplyReport, TimingCollector, multiply_with_progress};
use num_bigint::BigUint;
use serde_json::{Value, json};
use wasm_bindgen::prelude::*;

const MIN_BUDGET_MS: u32 = 1_000;
const MAX_BUDGET_MS: u32 = 10_000;
const MEMORY_LIMIT_MB: usize = 512;
const SEED: u64 = 0x9e3779b97f4a7c15;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[wasm_bindgen]
pub fn multiply(a_text: &str, b_text: &str, budget_ms: u32) -> String {
    let max_duration = if budget_ms == 0 {
        None
    } else {
        Some(Duration::from_millis(u64::from(
            budget_ms.clamp(MIN_BUDGET_MS, MAX_BUDGET_MS),
        )))
    };
    let parsed = match parse_decimal(a_text).and_then(|a| parse_decimal(b_text).map(|b| (a, b))) {
        Ok(pair) => pair,
        Err(error) => return failure(a_text, b_text, &error.to_string()),
    };
    let (a, b) = parsed;
    let options = MultiplyOptions {
        threads: Some(1),
        max_duration,
        max_memory_mb: Some(MEMORY_LIMIT_MB),
        seed: SEED,
        ..MultiplyOptions::default()
    };
    let collector = TimingCollector::new();
    let flag = AtomicBool::new(false);
    let mut sink = |_: &graymul_core::ProgressEvent| Ok(());
    let result = multiply_with_progress(&a, &b, &options, &flag, &mut sink, Some(&collector));
    match result {
        Ok(report) => success(a_text, b_text, &report, &collector.snapshot()),
        Err(error) => failure(a_text, b_text, &error.to_string()),
    }
}

fn parse_decimal(text: &str) -> Result<BigUint, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("input: empty integer".to_string());
    }
    BigUint::from_str(trimmed).map_err(|_| invalid_message(trimmed))
}

fn invalid_message(text: &str) -> String {
    const LIMIT: usize = 64;
    if text.len() <= LIMIT {
        return format!("input: invalid integer {text}");
    }
    let shown: String = text.chars().take(LIMIT).collect();
    format!(
        "input: invalid integer {shown}...<{} chars>",
        text.chars().count()
    )
}

fn success(
    a_text: &str,
    b_text: &str,
    report: &MultiplyReport,
    timings: &[graymul_core::StageTiming],
) -> String {
    let stages: Vec<Value> = timings
        .iter()
        .map(|timing| {
            json!({
                "stage": &timing.stage,
                "degree": timing.degree,
                "ms": millis(timing.elapsed_ms),
                "detail": &timing.detail,
            })
        })
        .collect();
    json!({
        "ok": true,
        "inputA": a_text,
        "inputB": b_text,
        "product": &report.product,
        "method": &report.method,
        "limbs": report.limbs,
        "elapsedMs": millis(report.elapsed_ms),
        "stages": stages,
    })
    .to_string()
}

fn failure(a_text: &str, b_text: &str, error: &str) -> String {
    json!({
        "ok": false,
        "inputA": a_text,
        "inputB": b_text,
        "error": error,
    })
    .to_string()
}

fn millis(value: u128) -> u64 {
    value.min(u128::from(u64::MAX)) as u64
}
