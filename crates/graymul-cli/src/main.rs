use clap::{CommandFactory, Parser, Subcommand};
use graymul_core::{
    CancellationToken, MultiplyError, MultiplyOptions, MultiplyReport, ProgressEvent, StageTiming,
    TimingCollector,
};
use num_bigint::BigUint;
use std::io::BufRead;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug)]
#[command(name = "graymul", version, about = "Multiply integers with GrayMUL")]
struct Cli {
    a: Option<String>,
    b: Option<String>,
    #[arg(
        long,
        default_value = "auto",
        help = "Multiply engine: auto, schoolbook, karatsuba, toom3, toom4, ssa, furer, hvh or hvh-full"
    )]
    algorithm: String,
    #[arg(long, help = "Print a per-stage timing breakdown")]
    timings: bool,
    #[arg(long, help = "Emit the result as a single JSON object on stdout")]
    json: bool,
    #[arg(
        long,
        help = "Thread budget (defaults to all logical cores): sizes engine memory budgets in this CLI; drives the ssa and split legs on bench naked/interleave tiers"
    )]
    threads: Option<usize>,
    #[arg(
        long,
        help = "Overall budget in seconds; on expiry the run ends with a clean ResourceLimit"
    )]
    timeout_seconds: Option<u64>,
    #[arg(
        long,
        help = "Cap estimated multiply memory in MiB (unset; 1 GiB budget)"
    )]
    max_memory_mb: Option<usize>,
    #[arg(long, help = "Process-wide CPU-time budget in seconds")]
    max_cpu_seconds: Option<u64>,
    #[arg(
        long,
        help = "Seed for the deterministic RNG (absent = 0x9e3779b97f4a7c15)"
    )]
    seed: Option<u64>,
    #[arg(
        long,
        help = "Mark timing records as performance-mode runs without changing behavior"
    )]
    perf_mode: bool,
    #[arg(long, help = "Stream per-event multiply progress to stderr")]
    progress: bool,
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Batch {
        #[arg(
            long,
            help = "Path to a text file holding one \"A B\" decimal pair per line"
        )]
        input: PathBuf,
        #[arg(long, help = "Emit JSON Lines records (the default batch encoding)")]
        jsonl: bool,
        #[arg(
            long,
            help = "Record per-line failures and keep going instead of aborting"
        )]
        continue_on_error: bool,
    },
}

fn options_from_cli(cli: &Cli) -> MultiplyOptions {
    MultiplyOptions {
        algorithm: if cli.algorithm == "auto" {
            None
        } else {
            Some(cli.algorithm.clone())
        },
        threads: cli.threads,
        max_duration: cli.timeout_seconds.map(std::time::Duration::from_secs),
        max_memory_mb: cli.max_memory_mb,
        max_cpu_seconds: cli.max_cpu_seconds,
        seed: cli.seed.unwrap_or(MultiplyOptions::default().seed),
        perf_mode: cli.perf_mode,
    }
}

fn parse_decimal(text: &str) -> Result<BigUint, MultiplyError> {
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

fn split_pair(line: &str) -> Result<(String, String), MultiplyError> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() != 2 {
        return Err(MultiplyError::Input(format!(
            "expected two integers per line, found {}",
            parts.len()
        )));
    }
    Ok((parts[0].to_string(), parts[1].to_string()))
}

fn multiply_pair(
    a_text: &str,
    b_text: &str,
    options: &MultiplyOptions,
    token: &CancellationToken,
    show_progress: bool,
) -> (Result<MultiplyReport, MultiplyError>, Vec<StageTiming>) {
    if token.is_cancelled() {
        return (
            Err(MultiplyError::Cancelled(
                "the operation was cancelled".to_string(),
            )),
            Vec::new(),
        );
    }
    let parsed = match parse_decimal(a_text).and_then(|a| parse_decimal(b_text).map(|b| (a, b))) {
        Ok(pair) => pair,
        Err(error) => return (Err(error), Vec::new()),
    };
    let (a, b) = parsed;
    let collector = TimingCollector::new();
    let mut on_event = |event: &ProgressEvent| {
        if show_progress {
            eprintln!("{}", format_progress(event));
        }
        Ok(())
    };
    let result = graymul_core::multiply_with_progress(
        &a,
        &b,
        options,
        token.as_atomic(),
        &mut on_event,
        Some(&collector),
    );
    (result, collector.snapshot())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match &cli.command {
        Some(Commands::Batch {
            input,
            continue_on_error,
            ..
        }) => run_batch(&cli, input, *continue_on_error),
        None => run_single(&cli),
    }
}

struct BatchRecord {
    input_a: String,
    input_b: String,
    product: Option<String>,
    method: Option<String>,
    status: String,
    elapsed_ms: u128,
    error: Option<String>,
}

fn batch_record_value(record: &BatchRecord) -> serde_json::Value {
    serde_json::json!({
        "input_a": record.input_a,
        "input_b": record.input_b,
        "product": record.product,
        "method": record.method,
        "status": record.status,
        "elapsed_ms": record.elapsed_ms,
        "error": record.error,
    })
}

fn multiply_batch_line(
    raw: &str,
    options: &MultiplyOptions,
    token: &CancellationToken,
    show_progress: bool,
) -> BatchRecord {
    let started = std::time::Instant::now();
    let (a_text, b_text) = match split_pair(raw) {
        Ok(pair) => pair,
        Err(error) => {
            let (category, detail) = error_parts(&error);
            return BatchRecord {
                input_a: raw.to_string(),
                input_b: String::new(),
                product: None,
                method: None,
                status: "error".to_string(),
                elapsed_ms: started.elapsed().as_millis(),
                error: Some(format!("{category}: {detail}")),
            };
        }
    };
    let (result, _) = multiply_pair(&a_text, &b_text, options, token, show_progress);
    match result {
        Ok(report) => BatchRecord {
            input_a: a_text,
            input_b: b_text,
            product: Some(report.product),
            method: Some(report.method),
            status: "ok".to_string(),
            elapsed_ms: started.elapsed().as_millis(),
            error: None,
        },
        Err(error) => {
            let (category, detail) = error_parts(&error);
            BatchRecord {
                input_a: a_text,
                input_b: b_text,
                product: None,
                method: None,
                status: "error".to_string(),
                elapsed_ms: started.elapsed().as_millis(),
                error: Some(format!("{category}: {detail}")),
            }
        }
    }
}

fn run_batch_lines(
    lines: &[String],
    options: &MultiplyOptions,
    token: &CancellationToken,
    show_progress: bool,
    continue_on_error: bool,
    emit: &mut dyn FnMut(&BatchRecord),
) -> bool {
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record = multiply_batch_line(trimmed, options, token, show_progress);
        let failed = record.status == "error";
        let cancelled = record
            .error
            .as_deref()
            .is_some_and(|text| text.starts_with("cancelled"));
        emit(&record);
        if cancelled || (failed && !continue_on_error) {
            return false;
        }
        if token.is_cancelled() {
            return false;
        }
    }
    true
}

fn batch_exit_code(failed: bool, cancelled: bool) -> ExitCode {
    if failed || cancelled {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn run_batch(cli: &Cli, input: &PathBuf, continue_on_error: bool) -> ExitCode {
    let file = match std::fs::File::open(input) {
        Ok(handle) => handle,
        Err(error) => {
            eprintln!(
                "error: input: cannot open batch file {}: {error}",
                input.display()
            );
            return ExitCode::FAILURE;
        }
    };
    let options = options_from_cli(cli);
    if let Err(error) = options.validate() {
        report_error(&error);
        return ExitCode::FAILURE;
    }
    let show_progress = cli.progress;
    let token = CancellationToken::new();
    let interrupt = token.clone();
    let _ = ctrlc::set_handler(move || {
        interrupt.cancel();
    });
    let reader = std::io::BufReader::new(file);
    let mut lines = Vec::new();
    for line in reader.lines() {
        match line {
            Ok(text) => lines.push(text),
            Err(error) => {
                eprintln!("error: input: cannot read batch line: {error}");
                return ExitCode::FAILURE;
            }
        }
    }
    let mut failed = false;
    let mut emit = |record: &BatchRecord| {
        if record.status == "error" {
            failed = true;
        }
        print_json(&batch_record_value(record));
    };
    run_batch_lines(
        &lines,
        &options,
        &token,
        show_progress,
        continue_on_error,
        &mut emit,
    );
    batch_exit_code(failed, token.is_cancelled())
}

fn run_single(cli: &Cli) -> ExitCode {
    let (a_text, b_text) = match (&cli.a, &cli.b) {
        (Some(a), Some(b)) => (a.clone(), b.clone()),
        _ => Cli::command()
            .error(
                clap::error::ErrorKind::MissingRequiredArgument,
                "the following required arguments were not provided:\n  <A> <B>",
            )
            .exit(),
    };
    let options = options_from_cli(cli);
    if let Err(error) = options.validate() {
        if cli.json {
            print_json(&json_error_value(&error, None));
        } else {
            report_error(&error);
        }
        return ExitCode::FAILURE;
    }
    let show_progress = cli.progress;
    let token = CancellationToken::new();
    let interrupt = token.clone();
    let _ = ctrlc::set_handler(move || {
        interrupt.cancel();
    });
    let (result, timings) = multiply_pair(&a_text, &b_text, &options, &token, show_progress);
    match result {
        Ok(report) => {
            if cli.json {
                let selected = cli.timings.then_some(timings.as_slice());
                print_json(&json_report_value(&report, &a_text, &b_text, selected));
            } else {
                println!("A: {a_text}");
                println!("B: {b_text}");
                println!("product: {}", report.product);
                println!("method: {}", report.method);
                println!("limbs: {}", report.limbs);
                println!("elapsed_ms: {}", report.elapsed_ms);
                if cli.timings {
                    print_timings(&timings);
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            if cli.json {
                let selected = cli.timings.then_some(timings.as_slice());
                print_json(&json_error_value(&error, selected));
            } else {
                if cli.timings {
                    print_timings(&timings);
                }
                report_error(&error);
            }
            ExitCode::FAILURE
        }
    }
}

fn print_json(value: &serde_json::Value) {
    println!("{value}");
}

fn format_progress(event: &ProgressEvent) -> String {
    format!(
        "progress: stage={} current={} total={} message={}",
        event.stage, event.current, event.total, event.message
    )
}

fn timings_json(entries: &[StageTiming]) -> serde_json::Value {
    serde_json::Value::Array(
        entries
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "stage": entry.stage,
                    "degree": entry.degree,
                    "elapsed_ms": entry.elapsed_ms,
                    "detail": entry.detail,
                })
            })
            .collect::<Vec<_>>(),
    )
}

fn json_report_value(
    report: &MultiplyReport,
    a_text: &str,
    b_text: &str,
    timings: Option<&[StageTiming]>,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "input_a": a_text,
        "input_b": b_text,
        "product": report.product,
        "method": report.method,
        "limbs": report.limbs,
        "elapsed_ms": report.elapsed_ms,
    });
    if let Some(entries) = timings {
        value["timings"] = timings_json(entries);
    }
    value
}

fn json_error_value(error: &MultiplyError, timings: Option<&[StageTiming]>) -> serde_json::Value {
    let (category, detail) = error_parts(error);
    let mut value = serde_json::json!({
        "error": {
            "category": category,
            "detail": detail,
        }
    });
    if let Some(entries) = timings {
        value["timings"] = timings_json(entries);
    }
    value
}

fn print_timings(entries: &[StageTiming]) {
    for entry in entries {
        let degree = entry
            .degree
            .map_or_else(|| "n/a".to_string(), |value| value.to_string());
        println!(
            "stage: {:<20} degree: {:<4} elapsed_ms: {:<8} {}",
            entry.stage, degree, entry.elapsed_ms, entry.detail
        );
    }
}

fn error_parts(error: &MultiplyError) -> (&'static str, String) {
    match error {
        MultiplyError::Input(detail) => ("input", detail.clone()),
        MultiplyError::ResourceLimit(detail) => ("resource-limit", detail.clone()),
        MultiplyError::Cancelled(detail) => ("cancelled", detail.clone()),
        MultiplyError::Internal(detail) => ("internal", detail.clone()),
    }
}

fn report_error(error: &MultiplyError) {
    let (category, detail) = error_parts(error);
    eprintln!("error: {category}: {detail}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use graymul_core::Engine;

    const REAL_STAGES: [&str; 12] = [
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

    fn is_real_engine_name(name: &str) -> bool {
        Engine::from_str(name).is_some() || name == "trivial"
    }

    fn sample_report() -> MultiplyReport {
        MultiplyReport {
            product: "121932631112635269".to_string(),
            method: "hvh".to_string(),
            limbs: 1,
            elapsed_ms: 3,
        }
    }

    #[test]
    fn parse_decimal_accepts_plain_and_padded_digits() {
        assert_eq!(parse_decimal("0").expect("zero"), BigUint::from(0u32));
        assert_eq!(
            parse_decimal("  12345  ").expect("padded"),
            BigUint::from(12345u32)
        );
        assert_eq!(parse_decimal("+12").expect("plus"), BigUint::from(12u32));
    }

    #[test]
    fn parse_decimal_rejects_blank_negative_and_alpha() {
        for text in ["", "   ", "-5", "abc", "12x34"] {
            assert!(parse_decimal(text).is_err(), "text {text} should fail");
        }
    }

    #[test]
    fn split_pair_accepts_two_tokens_and_rejects_other_arity() {
        let (a, b) = split_pair("12 34").expect("pair");
        assert_eq!((a.as_str(), b.as_str()), ("12", "34"));
        for line in ["12", "", "1 2 3", "   "] {
            assert!(split_pair(line).is_err(), "line {line:?} should fail");
        }
    }

    #[test]
    fn json_report_keys_match_schema_and_timings_stay_additive() {
        let report = sample_report();
        let value = json_report_value(&report, "12", "34", None);
        let mut keys: Vec<&str> = value
            .as_object()
            .expect("report object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "elapsed_ms",
                "input_a",
                "input_b",
                "limbs",
                "method",
                "product"
            ]
        );
        assert_eq!(value["input_a"], "12");
        assert_eq!(value["input_b"], "34");
        assert_eq!(value["product"], "121932631112635269");
        assert_eq!(value["method"], "hvh");
        assert!(value.get("timings").is_none());
        let text = value.to_string();
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("round trip");
        assert_eq!(parsed, value);
    }

    #[test]
    fn json_report_includes_timings_only_when_requested() {
        let report = sample_report();
        let entries = vec![StageTiming {
            stage: "hvh".to_string(),
            degree: None,
            elapsed_ms: 3,
            detail: "limbs=1".to_string(),
        }];
        let with = json_report_value(&report, "12", "34", Some(entries.as_slice()));
        let timings = with["timings"].as_array().expect("timings array");
        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0]["stage"], "hvh");
        assert!(timings[0]["degree"].is_null());
        assert_eq!(timings[0]["detail"], "limbs=1");
    }

    #[test]
    fn json_error_shape_has_category_and_detail_for_each_category() {
        let cases = [
            (MultiplyError::Input("bad digit".to_string()), "input"),
            (
                MultiplyError::ResourceLimit("deadline exceeded".to_string()),
                "resource-limit",
            ),
            (
                MultiplyError::Cancelled("the operation was cancelled".to_string()),
                "cancelled",
            ),
            (
                MultiplyError::Internal("verification failed".to_string()),
                "internal",
            ),
        ];
        for (error, category) in &cases {
            let value = json_error_value(error, None);
            assert_eq!(value["error"]["category"], *category);
            assert!(value["error"]["detail"].is_string());
            assert!(value.get("input_a").is_none());
            assert!(value.get("product").is_none());
        }
    }

    #[test]
    fn error_parts_taxonomy_matches_stderr_prefixes() {
        let (category, detail) = error_parts(&MultiplyError::Input("x".to_string()));
        assert_eq!(category, "input");
        assert_eq!(detail, "x");
        let (category, _) = error_parts(&MultiplyError::ResourceLimit("y".to_string()));
        assert_eq!(category, "resource-limit");
        let (category, detail) = error_parts(&MultiplyError::Cancelled(
            "the operation was cancelled".to_string(),
        ));
        assert_eq!(category, "cancelled");
        assert!(!detail.is_empty());
        let (category, _) = error_parts(&MultiplyError::Internal("z".to_string()));
        assert_eq!(category, "internal");
    }

    #[test]
    fn invalid_integer_message_bounds_huge_input() {
        assert_eq!(invalid_integer_message("abc"), "invalid integer abc");
        let huge = "z".repeat(10000);
        let detail = invalid_integer_message(&huge);
        assert!(detail.len() < 200);
        assert!(detail.starts_with("invalid integer zzzz"));
        assert!(detail.ends_with("<10000 chars>"));
        assert!(matches!(parse_decimal(&huge), Err(MultiplyError::Input(_))));
        let wide = "é".repeat(100);
        let wide_detail = invalid_integer_message(&wide);
        assert!(wide_detail.starts_with("invalid integer éé"));
        assert!(wide_detail.ends_with("<100 chars>"));
    }

    #[test]
    fn batch_error_record_bounds_huge_line() {
        let options = MultiplyOptions::default();
        let token = CancellationToken::new();
        let line = format!("12 {}", "z".repeat(100000));
        let record = multiply_batch_line(&line, &options, &token, false);
        assert_eq!(record.status, "error");
        assert!(record.product.is_none());
        assert!(record.method.is_none());
        let error = record.error.expect("error text");
        assert!(error.starts_with("input:"));
        assert!(error.len() < 500);
    }

    #[test]
    fn auto_method_is_real_engine_name() {
        let options = MultiplyOptions::default();
        let token = CancellationToken::new();
        let (result, _) = multiply_pair("12", "34", &options, &token, false);
        let report = result.expect("auto multiply");
        assert!(is_real_engine_name(&report.method), "{}", report.method);
        assert_eq!(report.product, "408");
        assert_eq!(report.limbs, 1);
    }

    #[test]
    fn zero_timeout_expires_before_compute() {
        let options = MultiplyOptions {
            max_duration: Some(std::time::Duration::from_secs(0)),
            ..MultiplyOptions::default()
        };
        let token = CancellationToken::new();
        let (result, _) = multiply_pair("12", "34", &options, &token, false);
        assert!(matches!(result, Err(MultiplyError::ResourceLimit(_))));
    }

    #[test]
    fn zero_memory_budget_rejects_before_compute() {
        let options = MultiplyOptions {
            max_memory_mb: Some(0),
            ..MultiplyOptions::default()
        };
        let token = CancellationToken::new();
        let (result, _) = multiply_pair("12", "34", &options, &token, false);
        assert!(matches!(result, Err(MultiplyError::ResourceLimit(_))));
    }

    #[test]
    fn precancelled_token_reports_cancelled() {
        let options = MultiplyOptions::default();
        let token = CancellationToken::new();
        token.cancel();
        let (result, _) = multiply_pair("12", "34", &options, &token, false);
        assert!(matches!(result, Err(MultiplyError::Cancelled(_))));
    }

    #[test]
    fn bogus_algorithm_parses_at_cli_but_fails_options_validate() {
        let cli = Cli::try_parse_from(["graymul", "12", "34", "--algorithm", "gnfs"])
            .expect("clap accepts free-form algorithm");
        assert_eq!(cli.algorithm, "gnfs");
        let options = options_from_cli(&cli);
        assert!(matches!(options.validate(), Err(MultiplyError::Input(_))));
    }

    #[test]
    fn batch_exit_code_truth_table() {
        assert_eq!(batch_exit_code(false, false), ExitCode::SUCCESS);
        assert_eq!(batch_exit_code(true, false), ExitCode::FAILURE);
        assert_eq!(batch_exit_code(false, true), ExitCode::FAILURE);
        assert_eq!(batch_exit_code(true, true), ExitCode::FAILURE);
    }

    #[test]
    fn batch_continue_golden_three_lines_one_failure() {
        let options = MultiplyOptions::default();
        let token = CancellationToken::new();
        let lines = vec![
            "123456789 987654321".to_string(),
            "abc def".to_string(),
            "7 8".to_string(),
        ];
        let mut records = Vec::new();
        let completed = run_batch_lines(&lines, &options, &token, false, true, &mut |record| {
            records.push(batch_record_value(record));
        });
        assert!(completed);
        assert_eq!(records.len(), 3);
        assert_eq!(records[0]["status"], "ok");
        assert_eq!(records[0]["input_a"], "123456789");
        assert_eq!(records[0]["input_b"], "987654321");
        assert_eq!(records[0]["product"], "121932631112635269");
        assert!(
            is_real_engine_name(records[0]["method"].as_str().expect("method string")),
            "{}",
            records[0]["method"]
        );
        assert!(records[0]["error"].is_null());
        assert_eq!(records[1]["status"], "error");
        assert!(records[1]["product"].is_null());
        assert!(records[1]["method"].is_null());
        let detail = records[1]["error"].as_str().expect("error string");
        assert!(detail.starts_with("input:"));
        assert_eq!(records[2]["status"], "ok");
        assert_eq!(records[2]["product"], "56");
        assert!(records[2]["error"].is_null());
        for value in &records {
            let mut keys: Vec<&str> = value
                .as_object()
                .expect("batch record object")
                .keys()
                .map(String::as_str)
                .collect();
            keys.sort_unstable();
            assert_eq!(
                keys,
                vec![
                    "elapsed_ms",
                    "error",
                    "input_a",
                    "input_b",
                    "method",
                    "product",
                    "status"
                ]
            );
            let text = value.to_string();
            let parsed: serde_json::Value = serde_json::from_str(&text).expect("round trip");
            assert_eq!(&parsed, value);
        }
    }

    #[test]
    fn batch_aborts_on_first_failure_without_continue_flag() {
        let options = MultiplyOptions::default();
        let token = CancellationToken::new();
        let lines = vec![
            "123456789 987654321".to_string(),
            "abc def".to_string(),
            "7 8".to_string(),
        ];
        let mut records = Vec::new();
        let completed = run_batch_lines(&lines, &options, &token, false, false, &mut |record| {
            records.push(batch_record_value(record));
        });
        assert!(!completed);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["status"], "ok");
        assert_eq!(records[1]["status"], "error");
    }

    #[test]
    fn batch_precancelled_token_aborts_after_first_record() {
        let options = MultiplyOptions::default();
        let token = CancellationToken::new();
        token.cancel();
        let lines = vec!["2 3".to_string(), "4 5".to_string()];
        let mut records = Vec::new();
        let completed = run_batch_lines(&lines, &options, &token, false, true, &mut |record| {
            records.push(batch_record_value(record));
        });
        assert!(!completed);
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn cli_batch_subcommand_parses_input_jsonl_continue_flags() {
        let cli = Cli::try_parse_from([
            "graymul",
            "batch",
            "--input",
            "pairs.txt",
            "--jsonl",
            "--continue-on-error",
        ])
        .expect("parse batch subcommand");
        match cli.command.expect("batch command present") {
            Commands::Batch {
                input,
                jsonl,
                continue_on_error,
            } => {
                assert_eq!(input, PathBuf::from("pairs.txt"));
                assert!(jsonl);
                assert!(continue_on_error);
            }
        }
        assert!(cli.a.is_none());
        assert!(cli.b.is_none());
    }

    #[test]
    fn cli_two_positionals_parse_with_full_flag_table() {
        let cli = Cli::try_parse_from([
            "graymul",
            "12",
            "34",
            "--algorithm",
            "karatsuba",
            "--timings",
            "--json",
            "--threads",
            "4",
            "--timeout-seconds",
            "30",
            "--max-memory-mb",
            "512",
            "--max-cpu-seconds",
            "60",
            "--seed",
            "7",
            "--perf-mode",
            "--progress",
        ])
        .expect("parse full flag table");
        assert_eq!(cli.a.as_deref(), Some("12"));
        assert_eq!(cli.b.as_deref(), Some("34"));
        assert_eq!(cli.algorithm, "karatsuba");
        assert!(cli.timings);
        assert!(cli.json);
        assert_eq!(cli.threads, Some(4));
        assert_eq!(cli.timeout_seconds, Some(30));
        assert_eq!(cli.max_memory_mb, Some(512));
        assert_eq!(cli.max_cpu_seconds, Some(60));
        assert_eq!(cli.seed, Some(7));
        assert!(cli.perf_mode);
        assert!(cli.progress);
        let options = options_from_cli(&cli);
        assert!(options.validate().is_ok());
        assert_eq!(options.seed, 7);
    }

    #[test]
    fn threads_help_states_measured_facts() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("--threads"));
        assert!(help.contains("defaults to all logical cores"));
        assert!(!help.contains("ignor"));
    }

    #[test]
    fn format_progress_keeps_stage_current_total_message_shape() {
        let event = ProgressEvent {
            stage: "hvh".to_string(),
            current: 1,
            total: 1,
            message: "multiplied 1 limbs".to_string(),
        };
        assert_eq!(
            format_progress(&event),
            "progress: stage=hvh current=1 total=1 message=multiplied 1 limbs"
        );
    }

    #[test]
    fn timings_stages_stay_in_vocabulary() {
        let options = MultiplyOptions::default();
        let token = CancellationToken::new();
        let (result, timings) = multiply_pair("123456789", "987654321", &options, &token, false);
        let report = result.expect("auto run");
        assert!(!timings.is_empty());
        for entry in &timings {
            assert!(
                REAL_STAGES.contains(&entry.stage.as_str()),
                "stage {}",
                entry.stage
            );
        }
        assert!(timings.iter().any(|entry| entry.stage == "pilot"));
        assert!(timings.iter().any(|entry| entry.stage == report.method));
    }
}
