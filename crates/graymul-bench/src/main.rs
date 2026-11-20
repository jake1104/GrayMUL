use clap::Parser;
use graymul_core::{Engine, MultiplyError, MultiplyOptions, ProgressEvent, verify_mod};
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(
    name = "graymul-bench",
    version,
    about = "Benchmark GrayMUL across multiply limb tiers"
)]
struct Cli {
    #[arg(long, default_value = "all")]
    tier: String,
    #[arg(long, default_value_t = 30)]
    timeout_seconds: u64,
    #[arg(
        long,
        help = "Worker count for naked/interleave ssa (>1 uses the parallel backend) and split legs (defaults to machine parallelism); memory-budget planning on the sweep tier"
    )]
    threads: Option<usize>,
    #[arg(long, default_value = "adhoc")]
    task_id: String,
    #[arg(long, default_value = "logs/graymul-benchmark.json")]
    output: PathBuf,
    #[arg(long, default_value_t = 0x9e3779b97f4a7c15)]
    seed: u64,
    #[arg(
        long,
        default_value = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384"
    )]
    sweep_limbs: String,
    #[arg(long)]
    reps: Option<usize>,
    #[arg(long)]
    warmup: Option<usize>,
    #[arg(long)]
    small_cell_ns: Option<u128>,
    #[arg(long)]
    small_reps: Option<usize>,
    #[arg(long, default_value = "auto")]
    engine: String,
    #[arg(long, default_value = "schoolbook,karatsuba,toom3,toom4,ssa,furer,hvh")]
    naked_engines: String,
    #[arg(long, default_value = "schoolbook,karatsuba,toom3,toom4,ssa,furer,hvh")]
    faceoff_engines: String,
    #[arg(long)]
    engines: Option<String>,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    verify_naked: bool,
    #[arg(long)]
    csv: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct Case {
    name: String,
    tier: String,
    a: BigUint,
    b: BigUint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Observation {
    case: String,
    tier: String,
    limbs_a: usize,
    limbs_b: usize,
    method: String,
    elapsed_ms: u128,
    verified: bool,
    status: String,
    error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BenchArchive {
    task_id: String,
    tier: String,
    timeout_seconds: u64,
    threads_requested: Option<usize>,
    threads_effective: usize,
    seeds: Vec<u64>,
    timestamp_unix_secs: u64,
    observations: Vec<Observation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sweep_limbs: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sweep_reps: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sweep_engine: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    naked_engines: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    faceoff_engines: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    faceoff_ratios: Option<Vec<FaceoffRatio>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    interleave_engines: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    interleave_pairs: Option<Vec<InterleavePair>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    interleave_summary: Option<Vec<InterleaveSummary>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    asym_cells: Option<Vec<AsymCell>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warmup: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    small_cell_ns: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    small_reps: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verify_naked: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sweep_records: Option<Vec<SweepRecord>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sweep_summary: Option<Vec<SweepSummary>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SweepRecord {
    limbs: usize,
    rep: usize,
    engine: String,
    method: String,
    elapsed_ns: u128,
    ns_per_limb: f64,
    verified: bool,
    verify_in_timer: bool,
    warmup: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SweepSummary {
    limbs: usize,
    engine: String,
    reps: usize,
    median_ns: f64,
    median_ns_per_limb: f64,
    cv: f64,
    verified_all: bool,
    warmup_discarded: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FaceoffRatio {
    limbs: usize,
    engine: String,
    vs: String,
    ratio: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AsymCell {
    small: usize,
    big: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InterleavePair {
    limbs: usize,
    rep: usize,
    delta_ns: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InterleaveSummary {
    limbs: usize,
    engine_a: String,
    engine_b: String,
    reps: usize,
    median_a_ns: f64,
    median_b_ns: f64,
    median_delta_ns: f64,
    std_delta_ns: f64,
    cv_delta: f64,
    ratio_a_over_b: f64,
    verified_all: bool,
    warmup_discarded: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leg {
    Core(Engine),
    Probe(ProbeId),
    Split,
    Chunk(Engine),
    NumBigint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeId {
    SchoolbookScalar,
    #[cfg(target_arch = "x86_64")]
    SchoolbookAdx,
    KaratsubaCore,
    SsaCrt,
    SsaSmall,
    SsaFlat,
    SsaBailey,
    SsaTrunc,
}

impl ProbeId {
    fn name(self) -> &'static str {
        match self {
            ProbeId::SchoolbookScalar => "schoolbook/scalar",
            #[cfg(target_arch = "x86_64")]
            ProbeId::SchoolbookAdx => "schoolbook/adx",
            ProbeId::KaratsubaCore => "karatsuba/core",
            ProbeId::SsaCrt => "ssa/crt",
            ProbeId::SsaSmall => "ssa/small",
            ProbeId::SsaFlat => "ssa/flat",
            ProbeId::SsaBailey => "ssa/bailey",
            ProbeId::SsaTrunc => "ssa/trunc",
        }
    }

    fn from_str(text: &str) -> Option<ProbeId> {
        match text {
            "schoolbook/scalar" => Some(ProbeId::SchoolbookScalar),
            #[cfg(target_arch = "x86_64")]
            "schoolbook/adx" => Some(ProbeId::SchoolbookAdx),
            "karatsuba/core" => Some(ProbeId::KaratsubaCore),
            "ssa/crt" => Some(ProbeId::SsaCrt),
            "ssa/small" => Some(ProbeId::SsaSmall),
            "ssa/flat" => Some(ProbeId::SsaFlat),
            "ssa/bailey" => Some(ProbeId::SsaBailey),
            "ssa/trunc" => Some(ProbeId::SsaTrunc),
            _ => None,
        }
    }

    fn valid_ids() -> Vec<&'static str> {
        let mut ids = vec!["schoolbook/scalar"];
        #[cfg(target_arch = "x86_64")]
        ids.push("schoolbook/adx");
        ids.extend([
            "karatsuba/core",
            "ssa/crt",
            "ssa/small",
            "ssa/flat",
            "ssa/bailey",
            "ssa/trunc",
        ]);
        ids
    }
}

impl Leg {
    fn name(self) -> String {
        match self {
            Leg::Core(engine) => engine.name().to_string(),
            Leg::Probe(probe) => probe.name().to_string(),
            Leg::Split => "split".to_string(),
            Leg::Chunk(engine) => format!("chunked/{}", engine.name()),
            Leg::NumBigint => "num-bigint".to_string(),
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if cli.tier == "sweep" {
        return run_sweep(&cli);
    }
    if cli.tier == "naked" {
        return run_naked(&cli);
    }
    if cli.tier == "faceoff" {
        return run_faceoff(&cli);
    }
    if cli.tier == "interleave" {
        return run_interleave(&cli);
    }
    if cli.tier == "asym" {
        return run_asym(&cli);
    }
    let tiers = match selected_tiers(&cli.tier) {
        Ok(tiers) => tiers,
        Err(error) => {
            eprintln!("error: input: {error}");
            return ExitCode::FAILURE;
        }
    };
    let threads_effective = cli.threads.unwrap_or(
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1),
    );
    let timestamp_unix_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let cases = build_cases(&tiers, cli.seed);
    let mut observations = Vec::with_capacity(cases.len());
    for case in &cases {
        observations.push(run_case(case, cli.timeout_seconds, cli.threads, cli.seed));
    }
    print_summary(&observations);
    let archive = BenchArchive {
        task_id: cli.task_id.clone(),
        tier: cli.tier.clone(),
        timeout_seconds: cli.timeout_seconds,
        threads_requested: cli.threads,
        threads_effective,
        seeds: vec![cli.seed],
        timestamp_unix_secs,
        observations,
        sweep_limbs: None,
        sweep_reps: None,
        sweep_engine: None,
        naked_engines: None,
        faceoff_engines: None,
        faceoff_ratios: None,
        warmup: None,
        small_cell_ns: None,
        small_reps: None,
        verify_naked: None,
        sweep_records: None,
        sweep_summary: None,
        interleave_engines: None,
        interleave_pairs: None,
        interleave_summary: None,
        asym_cells: None,
    };
    if let Some(parent) = cli.output.parent() {
        let created = fs::create_dir_all(parent);
        if let Err(error) = created {
            eprintln!("error: internal: cannot create output dir: {error}");
            return ExitCode::FAILURE;
        }
    }
    let payload = match serde_json::to_vec_pretty(&archive) {
        Ok(payload) => payload,
        Err(error) => {
            eprintln!("error: internal: cannot encode archive: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = fs::write(&cli.output, payload) {
        eprintln!("error: internal: cannot write output: {error}");
        return ExitCode::FAILURE;
    }
    println!("json: {}", cli.output.display());
    if archive.observations.iter().all(|item| item.verified) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn tier_limbs(tier: &str) -> Option<usize> {
    match tier {
        "tiny" => Some(1),
        "small" => Some(4),
        "medium" => Some(8),
        "large" => Some(32),
        "huge" => Some(128),
        _ => None,
    }
}

fn selected_tiers(tier: &str) -> Result<Vec<String>, String> {
    if tier == "all" {
        return Ok(vec![
            "tiny".to_string(),
            "small".to_string(),
            "medium".to_string(),
            "large".to_string(),
            "huge".to_string(),
        ]);
    }
    if tier_limbs(tier).is_some() {
        return Ok(vec![tier.to_string()]);
    }
    Err(format!("unknown tier: {tier}"))
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

fn seeded_operand(state: &mut u64, limbs: usize) -> BigUint {
    let mut words = Vec::with_capacity(limbs);
    for _ in 0..limbs {
        words.push(splitmix64(state));
    }
    words[limbs - 1] |= 1 << 63;
    let mut bytes = Vec::with_capacity(limbs * 8);
    for word in &words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    BigUint::from_bytes_le(&bytes)
}

fn build_cases(tiers: &[String], seed: u64) -> Vec<Case> {
    let mut state = seed;
    let mut cases = Vec::with_capacity(tiers.len());
    for tier in tiers {
        let limbs = tier_limbs(tier).unwrap_or(1);
        let (a, b) = if tier == "tiny" {
            (BigUint::from(12345u32), BigUint::from(67890u32))
        } else {
            (
                seeded_operand(&mut state, limbs),
                seeded_operand(&mut state, limbs),
            )
        };
        cases.push(Case {
            name: format!("{tier}_pair"),
            tier: tier.clone(),
            a,
            b,
        });
    }
    cases
}

fn run_case(case: &Case, timeout_seconds: u64, threads: Option<usize>, seed: u64) -> Observation {
    let limbs_a = graymul_core::from_bigint(&case.a).len();
    let limbs_b = graymul_core::from_bigint(&case.b).len();
    let mut observation = Observation {
        case: case.name.clone(),
        tier: case.tier.clone(),
        limbs_a,
        limbs_b,
        method: String::new(),
        elapsed_ms: 0,
        verified: false,
        status: String::new(),
        error: None,
    };
    let options = MultiplyOptions {
        threads,
        max_duration: Some(Duration::from_secs(timeout_seconds)),
        max_memory_mb: None,
        max_cpu_seconds: None,
        seed,
        perf_mode: false,
        ..MultiplyOptions::default()
    };
    let flag = AtomicBool::new(false);
    let mut sink = |_: &ProgressEvent| Ok(());
    match graymul_core::multiply_with_progress(&case.a, &case.b, &options, &flag, &mut sink, None) {
        Ok(report) => {
            observation.method = report.method;
            observation.elapsed_ms = report.elapsed_ms;
            observation.verified = true;
            observation.status = "ok".to_string();
        }
        Err(error) => {
            let (status, detail) = error_parts(&error);
            observation.status = status.to_string();
            observation.error = Some(format!("{status}: {detail}"));
        }
    }
    observation
}

fn error_parts(error: &MultiplyError) -> (&'static str, String) {
    match error {
        MultiplyError::Input(detail) => ("error", detail.clone()),
        MultiplyError::ResourceLimit(detail) => ("resource-limit", detail.clone()),
        MultiplyError::Cancelled(detail) => ("cancelled", detail.clone()),
        MultiplyError::Internal(detail) => ("error", detail.clone()),
    }
}

fn print_summary(observations: &[Observation]) {
    println!("case,tier,limbs_a,limbs_b,status,verified,method,elapsed_ms");
    for observation in observations {
        println!(
            "{},{},{},{},{},{},{},{}",
            observation.case,
            observation.tier,
            observation.limbs_a,
            observation.limbs_b,
            observation.status,
            observation.verified,
            observation.method,
            observation.elapsed_ms,
        );
    }
}

fn parse_sweep_limbs(text: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in text.split(',') {
        let trimmed = part.trim();
        match trimmed.parse::<usize>() {
            Ok(n) if n >= 1 => out.push(n),
            _ => return Err(format!("invalid sweep limbs entry: {trimmed}")),
        }
    }
    if out.is_empty() {
        return Err("sweep limbs list is empty".to_string());
    }
    Ok(out)
}

fn median_i64(values: &[i64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid] as f64)
    } else {
        Some((sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0)
    }
}

fn std_i64(values: &[i64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    Some(variance.sqrt().max(0.0))
}

fn cv_signed(values: &[i64]) -> Option<f64> {
    let median = median_i64(values)?;
    if median == 0.0 {
        return Some(0.0);
    }
    Some(std_i64(values).unwrap_or(0.0) / median.abs())
}

fn median_of(values: &[u128]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid] as f64)
    } else {
        Some((sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0)
    }
}

fn coeff_var(values: &[u128]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
    if mean <= 0.0 {
        return Some(0.0);
    }
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    Some((variance.sqrt() / mean).max(0.0))
}

fn build_sweep_cases(limbs_list: &[usize], seed: u64) -> Vec<Case> {
    let mut state = seed;
    let mut cases = Vec::with_capacity(limbs_list.len());
    for limbs in limbs_list {
        let a = seeded_operand(&mut state, *limbs);
        let b = seeded_operand(&mut state, *limbs);
        cases.push(Case {
            name: format!("sweep_{limbs}"),
            tier: "sweep".to_string(),
            a,
            b,
        });
    }
    cases
}

fn run_sweep_rep(
    case: &Case,
    engine: &str,
    rep: usize,
    timeout_seconds: u64,
    threads: Option<usize>,
    seed: u64,
    warmup: bool,
) -> SweepRecord {
    if engine == "num-bigint" {
        return run_control_rep(case, rep, warmup, true);
    }
    let limbs = graymul_core::from_bigint(&case.a)
        .len()
        .max(graymul_core::from_bigint(&case.b).len());
    let options = MultiplyOptions {
        algorithm: if engine == "auto" {
            None
        } else {
            Some(engine.to_string())
        },
        threads,
        max_duration: Some(Duration::from_secs(timeout_seconds)),
        max_memory_mb: None,
        max_cpu_seconds: None,
        seed,
        perf_mode: false,
    };
    let flag = AtomicBool::new(false);
    let mut sink = |_: &ProgressEvent| Ok(());
    let start = Instant::now();
    let outcome =
        graymul_core::multiply_with_progress(&case.a, &case.b, &options, &flag, &mut sink, None);
    let elapsed_ns = start.elapsed().as_nanos();
    let ns_per_limb = elapsed_ns as f64 / limbs.max(1) as f64;
    match outcome {
        Ok(report) => {
            let verified = BigUint::from_str(&report.product)
                .map(|product| product == &case.a * &case.b)
                .unwrap_or(false);
            SweepRecord {
                limbs,
                rep,
                engine: engine.to_string(),
                method: report.method,
                elapsed_ns,
                ns_per_limb,
                verified,
                verify_in_timer: true,
                warmup,
            }
        }
        Err(_) => SweepRecord {
            limbs,
            rep,
            engine: engine.to_string(),
            method: String::new(),
            elapsed_ns,
            ns_per_limb,
            verified: false,
            verify_in_timer: true,
            warmup,
        },
    }
}

fn run_control_rep(case: &Case, rep: usize, warmup: bool, verify: bool) -> SweepRecord {
    let a_limbs = graymul_core::from_bigint(&case.a);
    let b_limbs = graymul_core::from_bigint(&case.b);
    let limbs = a_limbs.len().max(b_limbs.len());
    let start = Instant::now();
    let product = &case.a * &case.b;
    let elapsed_ns = start.elapsed().as_nanos();
    let verified = verify && verify_mod(&a_limbs, &b_limbs, &graymul_core::from_bigint(&product));
    SweepRecord {
        limbs,
        rep,
        engine: "num-bigint".to_string(),
        method: "num-bigint".to_string(),
        elapsed_ns,
        ns_per_limb: elapsed_ns as f64 / limbs.max(1) as f64,
        verified,
        verify_in_timer: false,
        warmup,
    }
}

fn summarize_sweep(limbs: usize, engine: &str, records: &[SweepRecord]) -> SweepSummary {
    let timed: Vec<&SweepRecord> = records.iter().filter(|record| !record.warmup).collect();
    let times: Vec<u128> = timed.iter().map(|record| record.elapsed_ns).collect();
    let median_ns = median_of(&times).unwrap_or(0.0);
    SweepSummary {
        limbs,
        engine: engine.to_string(),
        reps: timed.len(),
        median_ns,
        median_ns_per_limb: median_ns / limbs.max(1) as f64,
        cv: coeff_var(&times).unwrap_or(0.0),
        verified_all: !timed.is_empty() && timed.iter().all(|record| record.verified),
        warmup_discarded: records.len() - timed.len(),
    }
}

fn sweep_csv(records: &[SweepRecord]) -> String {
    let mut text = String::from("limbs,rep,engine,method,elapsed_ns,ns_per_limb,verified,warmup\n");
    for record in records {
        text.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            record.limbs,
            record.rep,
            record.engine,
            record.method,
            record.elapsed_ns,
            record.ns_per_limb,
            record.verified,
            record.warmup
        ));
    }
    text
}

fn print_sweep_summary(summaries: &[SweepSummary]) {
    println!("limbs,engine,reps,median_ns,cv,verified_all,warmup_discarded");
    for summary in summaries {
        println!(
            "{},{},{},{},{},{},{}",
            summary.limbs,
            summary.engine,
            summary.reps,
            summary.median_ns,
            summary.cv,
            summary.verified_all,
            summary.warmup_discarded
        );
    }
}

const SWEEP_DEFAULT_REPS: usize = 3;
const NAKED_DEFAULT_REPS: usize = 15;
const WARMUP_DEFAULT_REPS: usize = 1;
const SMALL_CELL_NS_DEFAULT: u128 = 1_000_000;
const SMALL_REPS_DEFAULT: usize = 15;

fn effective_reps(tier: &str, reps: Option<usize>) -> usize {
    match reps {
        Some(n) => n,
        None => {
            if tier == "naked" || tier == "faceoff" || tier == "interleave" || tier == "asym" {
                NAKED_DEFAULT_REPS
            } else {
                SWEEP_DEFAULT_REPS
            }
        }
    }
}

fn effective_warmup(warmup: Option<usize>) -> usize {
    warmup.unwrap_or(WARMUP_DEFAULT_REPS)
}

fn effective_small_cell_ns(small_cell_ns: Option<u128>) -> u128 {
    small_cell_ns.unwrap_or(SMALL_CELL_NS_DEFAULT)
}

fn effective_small_reps(small_reps: Option<usize>) -> usize {
    small_reps.unwrap_or(SMALL_REPS_DEFAULT)
}

fn needs_extension(median_ns: f64, threshold_ns: u128) -> bool {
    median_ns.is_finite() && median_ns < threshold_ns as f64
}

fn extend_small_cell(
    records: &mut Vec<SweepRecord>,
    limbs: usize,
    engine: &str,
    small_ns: u128,
    small_reps: usize,
    mut make: impl FnMut(usize) -> SweepRecord,
) {
    let probe = summarize_sweep(limbs, engine, records);
    if needs_extension(probe.median_ns, small_ns) {
        let base = records.len();
        for extra in 0..small_reps {
            records.push(make(base + extra));
        }
    }
}

fn parse_interleave_engines(text: &str) -> Result<[Leg; 2], String> {
    let mut out = Vec::new();
    for part in text.split(',') {
        let trimmed = part.trim();
        if trimmed == "num-bigint" {
            out.push(Leg::NumBigint);
            continue;
        }
        if trimmed == "split" {
            out.push(Leg::Split);
            continue;
        }
        if trimmed.contains('/') {
            if let Some(rest) = trimmed.strip_prefix("chunked/") {
                match Engine::from_str(rest) {
                    Some(engine) => {
                        if engine == Engine::HvhFull {
                            return Err("hvh-full is correctness-only, excluded from interleave"
                                .to_string());
                        }
                        out.push(Leg::Chunk(engine));
                        continue;
                    }
                    None => return Err(format!("unknown engine: {trimmed}")),
                }
            }
            match ProbeId::from_str(trimmed) {
                Some(probe) => {
                    out.push(Leg::Probe(probe));
                    continue;
                }
                None => {
                    return Err(format!(
                        "unknown composite id: {trimmed}; valid ids: {}",
                        ProbeId::valid_ids().join(", ")
                    ));
                }
            }
        }
        match Engine::from_str(trimmed) {
            Some(engine) => {
                if engine == Engine::HvhFull {
                    return Err(
                        "hvh-full is correctness-only, excluded from interleave".to_string()
                    );
                }
                out.push(Leg::Core(engine));
            }
            None => return Err(format!("unknown engine: {trimmed}")),
        }
    }
    match out.len() {
        2 => Ok([out[0], out[1]]),
        _ => Err(format!("need exactly 2 engines, got {}", out.len())),
    }
}

fn parse_naked_engines(text: &str) -> Result<Vec<Leg>, String> {
    let mut out = Vec::new();
    for part in text.split(',') {
        let trimmed = part.trim();
        if trimmed == "split" {
            out.push(Leg::Split);
            continue;
        }
        match Engine::from_str(trimmed) {
            Some(engine) => {
                if engine == Engine::HvhFull {
                    return Err("hvh-full is correctness-only, excluded from naked".to_string());
                }
                out.push(Leg::Core(engine));
            }
            None => return Err(format!("unknown engine: {trimmed}")),
        }
    }
    if out.is_empty() {
        return Err("naked engine list is empty".to_string());
    }
    Ok(out)
}

fn asym_add_shifted(out: &mut Vec<u64>, add: &[u64], shift: usize) {
    if out.len() < shift + add.len() {
        out.resize(shift + add.len(), 0);
    }
    let mut carry: u128 = 0;
    let mut index = 0;
    while index < add.len() || carry > 0 {
        if shift + index >= out.len() {
            out.push(0);
        }
        let slot = &mut out[shift + index];
        let cur = u128::from(*slot) + u128::from(*add.get(index).unwrap_or(&0)) + carry;
        *slot = cur as u64;
        carry = cur >> 64;
        index += 1;
    }
}

fn asym_chunked_mul(a_limbs: &[u64], b_limbs: &[u64], engine: Engine) -> Vec<u64> {
    let (big, small) = if a_limbs.len() >= b_limbs.len() {
        (a_limbs, b_limbs)
    } else {
        (b_limbs, a_limbs)
    };
    let width = small.len().max(1);
    let chunks = big.len().div_ceil(width).max(1);
    let mut out = Vec::new();
    for index in 0..chunks {
        let lo = index * width;
        let hi = (lo + width).min(big.len());
        let sub = graymul_core::multiply_naked(engine, &big[lo..hi], small);
        asym_add_shifted(&mut out, &sub, lo);
    }
    graymul_core::normalize_limbs(&mut out);
    out
}

fn run_chunked_rep(
    a_limbs: &[u64],
    b_limbs: &[u64],
    engine: Engine,
    limbs: usize,
    rep: usize,
    warmup: bool,
    verify: bool,
) -> SweepRecord {
    let label = format!("chunked/{}", engine.name());
    let start = Instant::now();
    let out = asym_chunked_mul(a_limbs, b_limbs, engine);
    let elapsed_ns = start.elapsed().as_nanos();
    let verified = verify && verify_mod(a_limbs, b_limbs, &out);
    SweepRecord {
        limbs,
        rep,
        engine: label.clone(),
        method: label,
        elapsed_ns,
        ns_per_limb: elapsed_ns as f64 / limbs.max(1) as f64,
        verified,
        verify_in_timer: false,
        warmup,
    }
}

fn build_asym_cases(seed: u64) -> Vec<Case> {
    let mut state = seed;
    let mut cases = Vec::with_capacity(ASYM_SMALL_BIG.len());
    for (small, big) in ASYM_SMALL_BIG {
        let a = seeded_operand(&mut state, small);
        let b = seeded_operand(&mut state, big);
        cases.push(Case {
            name: format!("asym_{small}x{big}"),
            tier: "asym".to_string(),
            a,
            b,
        });
    }
    cases
}

fn run_naked_rep(
    a_limbs: &[u64],
    b_limbs: &[u64],
    engine: Engine,
    limbs: usize,
    rep: usize,
    warmup: bool,
    verify: bool,
) -> SweepRecord {
    let start = Instant::now();
    let out = graymul_core::multiply_naked(engine, a_limbs, b_limbs);
    let elapsed_ns = start.elapsed().as_nanos();
    let verified = verify && verify_mod(a_limbs, b_limbs, &out);
    SweepRecord {
        limbs,
        rep,
        engine: engine.name().to_string(),
        method: engine.name().to_string(),
        elapsed_ns,
        ns_per_limb: elapsed_ns as f64 / limbs.max(1) as f64,
        verified,
        verify_in_timer: false,
        warmup,
    }
}

fn run_probe_rep(
    a_limbs: &[u64],
    b_limbs: &[u64],
    probe: ProbeId,
    limbs: usize,
    rep: usize,
    warmup: bool,
    verify: bool,
) -> SweepRecord {
    let start = Instant::now();
    let out = match probe {
        ProbeId::SchoolbookScalar => graymul_core::schoolbook::measure_scalar(a_limbs, b_limbs),
        #[cfg(target_arch = "x86_64")]
        ProbeId::SchoolbookAdx => graymul_core::schoolbook::measure_adx(a_limbs, b_limbs),
        ProbeId::KaratsubaCore => graymul_core::karatsuba::mul_core(
            a_limbs,
            b_limbs,
            &mut graymul_core::karatsuba::WorkScratch::new(),
        ),
        ProbeId::SsaCrt => graymul_core::schonhage::mul_crt(a_limbs, b_limbs),
        ProbeId::SsaSmall => graymul_core::schonhage::mul_small(a_limbs, b_limbs),
        ProbeId::SsaFlat => graymul_core::small_ntt::measure_crt_flat(a_limbs, b_limbs),
        ProbeId::SsaBailey => graymul_core::small_ntt::measure_crt_bailey(a_limbs, b_limbs),
        ProbeId::SsaTrunc => graymul_core::small_ntt::measure_crt_trunc(a_limbs, b_limbs),
    };
    let elapsed_ns = start.elapsed().as_nanos();
    let verified = verify && verify_mod(a_limbs, b_limbs, &out);
    SweepRecord {
        limbs,
        rep,
        engine: probe.name().to_string(),
        method: probe.name().to_string(),
        elapsed_ns,
        ns_per_limb: elapsed_ns as f64 / limbs.max(1) as f64,
        verified,
        verify_in_timer: false,
        warmup,
    }
}

fn run_split_rep(
    a_limbs: &[u64],
    b_limbs: &[u64],
    limbs: usize,
    rep: usize,
    warmup: bool,
    verify: bool,
    threads: Option<usize>,
) -> SweepRecord {
    let workers = threads.unwrap_or_else(|| MultiplyOptions::default().effective_workers());
    let start = Instant::now();
    let out = graymul_core::mul_split_par(a_limbs, b_limbs, workers);
    let elapsed_ns = start.elapsed().as_nanos();
    let verified = verify && verify_mod(a_limbs, b_limbs, &out);
    SweepRecord {
        limbs,
        rep,
        engine: "split".to_string(),
        method: "split".to_string(),
        elapsed_ns,
        ns_per_limb: elapsed_ns as f64 / limbs.max(1) as f64,
        verified,
        verify_in_timer: false,
        warmup,
    }
}

fn run_ssa_par_rep(
    a_limbs: &[u64],
    b_limbs: &[u64],
    limbs: usize,
    rep: usize,
    warmup: bool,
    verify: bool,
    workers: usize,
) -> SweepRecord {
    let start = Instant::now();
    let out = graymul_core::schonhage::mul_par(a_limbs, b_limbs, workers);
    let elapsed_ns = start.elapsed().as_nanos();
    let verified = verify && verify_mod(a_limbs, b_limbs, &out);
    SweepRecord {
        limbs,
        rep,
        engine: "ssa".to_string(),
        method: "ssa".to_string(),
        elapsed_ns,
        ns_per_limb: elapsed_ns as f64 / limbs.max(1) as f64,
        verified,
        verify_in_timer: false,
        warmup,
    }
}

const ASYM_SMALL_BIG: [(usize, usize); 8] = [
    (32, 64),
    (32, 256),
    (32, 2048),
    (32, 16384),
    (128, 256),
    (128, 1024),
    (128, 8192),
    (128, 65536),
];

#[allow(clippy::too_many_arguments)]
fn run_naked_leg(
    a_limbs: &[u64],
    b_limbs: &[u64],
    leg: Leg,
    limbs: usize,
    rep: usize,
    warmup: bool,
    verify: bool,
    threads: Option<usize>,
) -> SweepRecord {
    match leg {
        Leg::Core(Engine::Ssa) if matches!(threads, Some(workers) if workers > 1) => {
            run_ssa_par_rep(
                a_limbs,
                b_limbs,
                limbs,
                rep,
                warmup,
                verify,
                threads.unwrap(),
            )
        }
        Leg::Core(engine) => run_naked_rep(a_limbs, b_limbs, engine, limbs, rep, warmup, verify),
        Leg::Chunk(engine) => run_chunked_rep(a_limbs, b_limbs, engine, limbs, rep, warmup, verify),
        Leg::Split => run_split_rep(a_limbs, b_limbs, limbs, rep, warmup, verify, threads),
        Leg::Probe(probe) => run_probe_rep(a_limbs, b_limbs, probe, limbs, rep, warmup, verify),
        Leg::NumBigint => unreachable!("naked tiers never parse control legs"),
    }
}

#[allow(clippy::too_many_arguments)]
fn faceoff_cell_records(
    a_limbs: &[u64],
    b_limbs: &[u64],
    limbs: usize,
    engines: &[Leg],
    reps: usize,
    warmup: usize,
    verify: bool,
    threads: Option<usize>,
) -> Vec<SweepRecord> {
    let mut out = Vec::with_capacity((warmup + reps) * engines.len());
    for rep in 0..(warmup + reps) {
        for (order, leg) in engines.iter().enumerate() {
            out.push(run_naked_leg(
                a_limbs,
                b_limbs,
                *leg,
                limbs,
                rep * engines.len() + order,
                rep < warmup,
                verify,
                threads,
            ));
        }
    }
    out
}

fn all_verified(records: &[SweepRecord], verify_enabled: bool) -> bool {
    records
        .iter()
        .all(|record| record.verified || !verify_enabled)
}

#[allow(clippy::too_many_arguments)]
fn run_leg(
    case: &Case,
    a_limbs: &[u64],
    b_limbs: &[u64],
    leg: Leg,
    limbs: usize,
    rep: usize,
    warmup: bool,
    verify: bool,
    threads: Option<usize>,
) -> SweepRecord {
    match leg {
        Leg::Core(Engine::Ssa) if matches!(threads, Some(workers) if workers > 1) => {
            run_ssa_par_rep(
                a_limbs,
                b_limbs,
                limbs,
                rep,
                warmup,
                verify,
                threads.unwrap(),
            )
        }
        Leg::Core(engine) => run_naked_rep(a_limbs, b_limbs, engine, limbs, rep, warmup, verify),
        Leg::Chunk(engine) => run_chunked_rep(a_limbs, b_limbs, engine, limbs, rep, warmup, verify),
        Leg::Probe(probe) => run_probe_rep(a_limbs, b_limbs, probe, limbs, rep, warmup, verify),
        Leg::Split => run_split_rep(a_limbs, b_limbs, limbs, rep, warmup, verify, threads),
        Leg::NumBigint => run_control_rep(case, rep, warmup, verify),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_paired_rep(
    case: &Case,
    a_limbs: &[u64],
    b_limbs: &[u64],
    limbs: usize,
    legs: &[Leg; 2],
    rep: usize,
    warmup: bool,
    verify: bool,
    threads: Option<usize>,
) -> (SweepRecord, SweepRecord, i64) {
    let order = if rep % 2 == 0 {
        [0usize, 1usize]
    } else {
        [1usize, 0usize]
    };
    let mut kept = [None, None];
    let mut times = [0u128, 0u128];
    for (slot, leg_index) in order.iter().enumerate() {
        let record = run_leg(
            case,
            a_limbs,
            b_limbs,
            legs[*leg_index],
            limbs,
            rep * 2 + slot,
            warmup,
            verify,
            threads,
        );
        times[*leg_index] = record.elapsed_ns;
        kept[*leg_index] = Some(record);
    }
    let first = kept[0].take().expect("paired leg a");
    let second = kept[1].take().expect("paired leg b");
    let delta = times[0] as i64 - times[1] as i64;
    (first, second, delta)
}

#[allow(clippy::too_many_arguments)]
fn run_interleave_cell(
    case: &Case,
    a_limbs: &[u64],
    b_limbs: &[u64],
    limbs: usize,
    legs: &[Leg; 2],
    reps: usize,
    warmup: usize,
    verify: bool,
    small_ns: u128,
    small_reps: usize,
    threads: Option<usize>,
) -> (Vec<SweepRecord>, Vec<InterleavePair>, InterleaveSummary) {
    let mut records = Vec::with_capacity(2 * (warmup + reps));
    let mut pair_index = 0usize;
    let mut pairs = Vec::with_capacity(reps);
    let mut rep = 0usize;
    while rep < warmup + reps {
        let (first, second, delta) = run_paired_rep(
            case,
            a_limbs,
            b_limbs,
            limbs,
            legs,
            rep,
            rep < warmup,
            verify,
            threads,
        );
        records.push(first);
        records.push(second);
        if rep >= warmup {
            pairs.push(InterleavePair {
                limbs,
                rep: pair_index,
                delta_ns: delta,
            });
            pair_index += 1;
        }
        rep += 1;
    }
    let timed_a: Vec<u128> = records
        .iter()
        .filter(|record| !record.warmup && record.engine == legs[0].name())
        .map(|record| record.elapsed_ns)
        .collect();
    let timed_b: Vec<u128> = records
        .iter()
        .filter(|record| !record.warmup && record.engine == legs[1].name())
        .map(|record| record.elapsed_ns)
        .collect();
    let median_a = median_of(&timed_a).unwrap_or(0.0);
    let median_b = median_of(&timed_b).unwrap_or(0.0);
    if needs_extension(median_a, small_ns) || needs_extension(median_b, small_ns) {
        for _ in 0..small_reps {
            let (first, second, delta) = run_paired_rep(
                case, a_limbs, b_limbs, limbs, legs, rep, false, verify, threads,
            );
            records.push(first);
            records.push(second);
            pairs.push(InterleavePair {
                limbs,
                rep: pair_index,
                delta_ns: delta,
            });
            pair_index += 1;
            rep += 1;
        }
    }
    let final_a: Vec<u128> = records
        .iter()
        .filter(|record| !record.warmup && record.engine == legs[0].name())
        .map(|record| record.elapsed_ns)
        .collect();
    let final_b: Vec<u128> = records
        .iter()
        .filter(|record| !record.warmup && record.engine == legs[1].name())
        .map(|record| record.elapsed_ns)
        .collect();
    let deltas: Vec<i64> = pairs.iter().map(|pair| pair.delta_ns).collect();
    let summary_median_a = median_of(&final_a).unwrap_or(0.0);
    let summary_median_b = median_of(&final_b).unwrap_or(0.0);
    let summary = InterleaveSummary {
        limbs,
        engine_a: legs[0].name(),
        engine_b: legs[1].name(),
        reps: pairs.len(),
        median_a_ns: summary_median_a,
        median_b_ns: summary_median_b,
        median_delta_ns: median_i64(&deltas).unwrap_or(0.0),
        std_delta_ns: std_i64(&deltas).unwrap_or(0.0),
        cv_delta: cv_signed(&deltas).unwrap_or(0.0),
        ratio_a_over_b: if summary_median_b > 0.0 {
            summary_median_a / summary_median_b
        } else {
            0.0
        },
        verified_all: !pairs.is_empty()
            && records
                .iter()
                .filter(|record| !record.warmup)
                .all(|record| record.verified),
        warmup_discarded: records.len() - records.iter().filter(|record| !record.warmup).count(),
    };
    (records, pairs, summary)
}

fn run_sweep(cli: &Cli) -> ExitCode {
    let limbs_list = match parse_sweep_limbs(&cli.sweep_limbs) {
        Ok(list) => list,
        Err(error) => {
            eprintln!("error: input: {error}");
            return ExitCode::FAILURE;
        }
    };
    if cli.engine != "auto" && cli.engine != "num-bigint" && Engine::from_str(&cli.engine).is_none()
    {
        eprintln!("error: input: unknown engine: {}", cli.engine);
        return ExitCode::FAILURE;
    }
    if cli.reps == Some(0) {
        eprintln!("error: input: reps must be at least 1");
        return ExitCode::FAILURE;
    }
    let reps = effective_reps("sweep", cli.reps);
    let warmup = effective_warmup(cli.warmup);
    let small_ns = effective_small_cell_ns(cli.small_cell_ns);
    let small_reps = effective_small_reps(cli.small_reps);
    let threads_effective = cli.threads.unwrap_or(
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1),
    );
    let timestamp_unix_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let cases = build_sweep_cases(&limbs_list, cli.seed);
    let mut records = Vec::new();
    let mut summaries = Vec::new();
    for case in &cases {
        let limbs = graymul_core::from_bigint(&case.a)
            .len()
            .max(graymul_core::from_bigint(&case.b).len());
        let mut cell_records = Vec::with_capacity(warmup + reps);
        for rep in 0..(warmup + reps) {
            cell_records.push(run_sweep_rep(
                case,
                &cli.engine,
                rep,
                cli.timeout_seconds,
                cli.threads,
                cli.seed,
                rep < warmup,
            ));
        }
        extend_small_cell(
            &mut cell_records,
            limbs,
            &cli.engine,
            small_ns,
            small_reps,
            |rep| {
                run_sweep_rep(
                    case,
                    &cli.engine,
                    rep,
                    cli.timeout_seconds,
                    cli.threads,
                    cli.seed,
                    false,
                )
            },
        );
        summaries.push(summarize_sweep(limbs, &cli.engine, &cell_records));
        records.extend(cell_records);
    }
    print_sweep_summary(&summaries);
    let archive = BenchArchive {
        task_id: cli.task_id.clone(),
        tier: "sweep".to_string(),
        timeout_seconds: cli.timeout_seconds,
        threads_requested: cli.threads,
        threads_effective,
        seeds: vec![cli.seed],
        timestamp_unix_secs,
        observations: Vec::new(),
        sweep_limbs: Some(limbs_list),
        sweep_reps: Some(reps),
        sweep_engine: Some(cli.engine.clone()),
        naked_engines: None,
        faceoff_engines: None,
        faceoff_ratios: None,
        warmup: Some(warmup),
        small_cell_ns: Some(small_ns),
        small_reps: Some(small_reps),
        verify_naked: None,
        sweep_records: Some(records.clone()),
        sweep_summary: Some(summaries),
        interleave_engines: None,
        interleave_pairs: None,
        interleave_summary: None,
        asym_cells: None,
    };
    if let Some(parent) = cli.output.parent() {
        let created = fs::create_dir_all(parent);
        if let Err(error) = created {
            eprintln!("error: internal: cannot create output dir: {error}");
            return ExitCode::FAILURE;
        }
    }
    let payload = match serde_json::to_vec_pretty(&archive) {
        Ok(payload) => payload,
        Err(error) => {
            eprintln!("error: internal: cannot encode archive: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = fs::write(&cli.output, payload) {
        eprintln!("error: internal: cannot write output: {error}");
        return ExitCode::FAILURE;
    }
    println!("json: {}", cli.output.display());
    if let Some(csv_path) = &cli.csv {
        if let Some(parent) = csv_path.parent() {
            if !parent.as_os_str().is_empty() {
                let created = fs::create_dir_all(parent);
                if let Err(error) = created {
                    eprintln!("error: internal: cannot create csv dir: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        if let Err(error) = fs::write(csv_path, sweep_csv(&records)) {
            eprintln!("error: internal: cannot write csv: {error}");
            return ExitCode::FAILURE;
        }
        println!("csv: {}", csv_path.display());
    }
    if all_verified(archive.sweep_records.as_deref().unwrap_or(&[]), true) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn run_naked(cli: &Cli) -> ExitCode {
    let limbs_list = match parse_sweep_limbs(&cli.sweep_limbs) {
        Ok(list) => list,
        Err(error) => {
            eprintln!("error: input: {error}");
            return ExitCode::FAILURE;
        }
    };
    let engines = match parse_naked_engines(&cli.naked_engines) {
        Ok(engines) => engines,
        Err(error) => {
            eprintln!("error: input: {error}");
            return ExitCode::FAILURE;
        }
    };
    if cli.reps == Some(0) {
        eprintln!("error: input: reps must be at least 1");
        return ExitCode::FAILURE;
    }
    let reps = effective_reps("naked", cli.reps);
    let warmup = effective_warmup(cli.warmup);
    let small_ns = effective_small_cell_ns(cli.small_cell_ns);
    let small_reps = effective_small_reps(cli.small_reps);
    let threads_effective = cli.threads.unwrap_or(
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1),
    );
    let timestamp_unix_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let cases = build_sweep_cases(&limbs_list, cli.seed);
    let mut records = Vec::new();
    let mut summaries = Vec::new();
    for case in &cases {
        let a_limbs = graymul_core::from_bigint(&case.a);
        let b_limbs = graymul_core::from_bigint(&case.b);
        let limbs = a_limbs.len().max(b_limbs.len());
        for engine in &engines {
            let mut cell_records = Vec::with_capacity(warmup + reps);
            for rep in 0..(warmup + reps) {
                cell_records.push(run_naked_leg(
                    &a_limbs,
                    &b_limbs,
                    *engine,
                    limbs,
                    rep,
                    rep < warmup,
                    cli.verify_naked,
                    cli.threads,
                ));
            }
            extend_small_cell(
                &mut cell_records,
                limbs,
                &engine.name(),
                small_ns,
                small_reps,
                |rep| {
                    run_naked_leg(
                        &a_limbs,
                        &b_limbs,
                        *engine,
                        limbs,
                        rep,
                        false,
                        cli.verify_naked,
                        cli.threads,
                    )
                },
            );
            summaries.push(summarize_sweep(limbs, &engine.name(), &cell_records));
            records.extend(cell_records);
        }
    }
    print_sweep_summary(&summaries);
    let archive = BenchArchive {
        task_id: cli.task_id.clone(),
        tier: "naked".to_string(),
        timeout_seconds: cli.timeout_seconds,
        threads_requested: cli.threads,
        threads_effective,
        seeds: vec![cli.seed],
        timestamp_unix_secs,
        observations: Vec::new(),
        sweep_limbs: Some(limbs_list),
        sweep_reps: Some(reps),
        sweep_engine: None,
        naked_engines: Some(engines.iter().map(|engine| engine.name()).collect()),
        faceoff_engines: None,
        faceoff_ratios: None,
        warmup: Some(warmup),
        small_cell_ns: Some(small_ns),
        small_reps: Some(small_reps),
        verify_naked: Some(cli.verify_naked),
        sweep_records: Some(records.clone()),
        sweep_summary: Some(summaries),
        interleave_engines: None,
        interleave_pairs: None,
        interleave_summary: None,
        asym_cells: None,
    };
    if let Some(parent) = cli.output.parent() {
        let created = fs::create_dir_all(parent);
        if let Err(error) = created {
            eprintln!("error: internal: cannot create output dir: {error}");
            return ExitCode::FAILURE;
        }
    }
    let payload = match serde_json::to_vec_pretty(&archive) {
        Ok(payload) => payload,
        Err(error) => {
            eprintln!("error: internal: cannot encode archive: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = fs::write(&cli.output, payload) {
        eprintln!("error: internal: cannot write output: {error}");
        return ExitCode::FAILURE;
    }
    println!("json: {}", cli.output.display());
    if let Some(csv_path) = &cli.csv {
        if let Some(parent) = csv_path.parent() {
            if !parent.as_os_str().is_empty() {
                let created = fs::create_dir_all(parent);
                if let Err(error) = created {
                    eprintln!("error: internal: cannot create csv dir: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        if let Err(error) = fs::write(csv_path, sweep_csv(&records)) {
            eprintln!("error: internal: cannot write csv: {error}");
            return ExitCode::FAILURE;
        }
        println!("csv: {}", csv_path.display());
    }
    if all_verified(
        archive.sweep_records.as_deref().unwrap_or(&[]),
        cli.verify_naked,
    ) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn run_faceoff(cli: &Cli) -> ExitCode {
    let limbs_list = match parse_sweep_limbs(&cli.sweep_limbs) {
        Ok(list) => list,
        Err(error) => {
            eprintln!("error: input: {error}");
            return ExitCode::FAILURE;
        }
    };
    let engines = match parse_naked_engines(&cli.faceoff_engines) {
        Ok(engines) => engines,
        Err(error) => {
            eprintln!("error: input: {error}");
            return ExitCode::FAILURE;
        }
    };
    if cli.reps == Some(0) {
        eprintln!("error: input: reps must be at least 1");
        return ExitCode::FAILURE;
    }
    let reps = effective_reps("faceoff", cli.reps);
    let warmup = effective_warmup(cli.warmup);
    let small_ns = effective_small_cell_ns(cli.small_cell_ns);
    let small_reps = effective_small_reps(cli.small_reps);
    let threads_effective = cli.threads.unwrap_or(
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1),
    );
    let timestamp_unix_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let cases = build_sweep_cases(&limbs_list, cli.seed);
    let mut records = Vec::new();
    let mut summaries = Vec::new();
    let mut ratios = Vec::new();
    for case in &cases {
        let a_limbs = graymul_core::from_bigint(&case.a);
        let b_limbs = graymul_core::from_bigint(&case.b);
        let limbs = a_limbs.len().max(b_limbs.len());
        let mut cell_records = faceoff_cell_records(
            &a_limbs,
            &b_limbs,
            limbs,
            &engines,
            reps,
            warmup,
            cli.verify_naked,
            cli.threads,
        );
        for engine in &engines {
            let subset: Vec<SweepRecord> = cell_records
                .iter()
                .filter(|record| record.engine == engine.name())
                .cloned()
                .collect();
            let probe = summarize_sweep(limbs, &engine.name(), &subset);
            if needs_extension(probe.median_ns, small_ns) {
                let base = cell_records.len();
                for extra in 0..small_reps {
                    cell_records.push(run_naked_leg(
                        &a_limbs,
                        &b_limbs,
                        *engine,
                        limbs,
                        base + extra,
                        false,
                        cli.verify_naked,
                        cli.threads,
                    ));
                }
            }
        }
        let mut medians = Vec::with_capacity(engines.len());
        for engine in &engines {
            let subset: Vec<SweepRecord> = cell_records
                .iter()
                .filter(|record| record.engine == engine.name())
                .cloned()
                .collect();
            let summary = summarize_sweep(limbs, &engine.name(), &subset);
            medians.push(summary.median_ns);
            summaries.push(summary);
        }
        if medians[0] > 0.0 {
            for (engine, median) in engines.iter().zip(medians.iter()).skip(1) {
                ratios.push(FaceoffRatio {
                    limbs,
                    engine: engine.name(),
                    vs: engines[0].name(),
                    ratio: median / medians[0],
                });
            }
        }
        records.extend(cell_records);
    }
    print_sweep_summary(&summaries);
    println!("limbs,engine,vs,ratio");
    for entry in &ratios {
        println!(
            "{},{},{},{}",
            entry.limbs, entry.engine, entry.vs, entry.ratio
        );
    }
    let archive = BenchArchive {
        task_id: cli.task_id.clone(),
        tier: "faceoff".to_string(),
        timeout_seconds: cli.timeout_seconds,
        threads_requested: cli.threads,
        threads_effective,
        seeds: vec![cli.seed],
        timestamp_unix_secs,
        observations: Vec::new(),
        sweep_limbs: Some(limbs_list),
        sweep_reps: Some(reps),
        sweep_engine: None,
        naked_engines: None,
        faceoff_engines: Some(engines.iter().map(|engine| engine.name()).collect()),
        faceoff_ratios: Some(ratios),
        warmup: Some(warmup),
        small_cell_ns: Some(small_ns),
        small_reps: Some(small_reps),
        verify_naked: Some(cli.verify_naked),
        sweep_records: Some(records.clone()),
        sweep_summary: Some(summaries),
        interleave_engines: None,
        interleave_pairs: None,
        interleave_summary: None,
        asym_cells: None,
    };
    if let Some(parent) = cli.output.parent() {
        let created = fs::create_dir_all(parent);
        if let Err(error) = created {
            eprintln!("error: internal: cannot create output dir: {error}");
            return ExitCode::FAILURE;
        }
    }
    let payload = match serde_json::to_vec_pretty(&archive) {
        Ok(payload) => payload,
        Err(error) => {
            eprintln!("error: internal: cannot encode archive: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = fs::write(&cli.output, payload) {
        eprintln!("error: internal: cannot write output: {error}");
        return ExitCode::FAILURE;
    }
    println!("json: {}", cli.output.display());
    if let Some(csv_path) = &cli.csv {
        if let Some(parent) = csv_path.parent() {
            if !parent.as_os_str().is_empty() {
                let created = fs::create_dir_all(parent);
                if let Err(error) = created {
                    eprintln!("error: internal: cannot create csv dir: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        if let Err(error) = fs::write(csv_path, sweep_csv(&records)) {
            eprintln!("error: internal: cannot write csv: {error}");
            return ExitCode::FAILURE;
        }
        println!("csv: {}", csv_path.display());
    }
    if all_verified(
        archive.sweep_records.as_deref().unwrap_or(&[]),
        cli.verify_naked,
    ) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn run_interleave(cli: &Cli) -> ExitCode {
    let limbs_list = match parse_sweep_limbs(&cli.sweep_limbs) {
        Ok(list) => list,
        Err(error) => {
            eprintln!("error: input: {error}");
            return ExitCode::FAILURE;
        }
    };
    let engines_text = match &cli.engines {
        Some(text) => text.clone(),
        None => {
            eprintln!("error: input: --engines <A,B> is required for the interleave tier");
            return ExitCode::FAILURE;
        }
    };
    let legs = match parse_interleave_engines(&engines_text) {
        Ok(legs) => legs,
        Err(error) => {
            eprintln!("error: input: {error}");
            return ExitCode::FAILURE;
        }
    };
    if cli.reps == Some(0) {
        eprintln!("error: input: reps must be at least 1");
        return ExitCode::FAILURE;
    }
    let reps = effective_reps("interleave", cli.reps);
    let warmup = effective_warmup(cli.warmup);
    let small_ns = effective_small_cell_ns(cli.small_cell_ns);
    let small_reps = effective_small_reps(cli.small_reps);
    let threads_effective = cli.threads.unwrap_or(
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1),
    );
    let timestamp_unix_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let cases = build_sweep_cases(&limbs_list, cli.seed);
    let mut records = Vec::new();
    let mut summaries = Vec::new();
    let mut pairs = Vec::new();
    for case in &cases {
        let a_limbs = graymul_core::from_bigint(&case.a);
        let b_limbs = graymul_core::from_bigint(&case.b);
        let limbs = a_limbs.len().max(b_limbs.len());
        let (cell_records, cell_pairs, summary) = run_interleave_cell(
            case,
            &a_limbs,
            &b_limbs,
            limbs,
            &legs,
            reps,
            warmup,
            cli.verify_naked,
            small_ns,
            small_reps,
            cli.threads,
        );
        summaries.push(summary);
        records.extend(cell_records);
        pairs.extend(cell_pairs);
    }
    println!(
        "limbs,engine_a,engine_b,reps,median_a_ns,median_b_ns,median_delta_ns,std_delta_ns,ratio,verified_all"
    );
    for summary in &summaries {
        println!(
            "{},{},{},{},{},{},{},{},{},{},{}",
            summary.limbs,
            summary.engine_a,
            summary.engine_b,
            summary.reps,
            summary.median_a_ns,
            summary.median_b_ns,
            summary.median_delta_ns,
            summary.std_delta_ns,
            summary.ratio_a_over_b,
            summary.cv_delta,
            summary.verified_all
        );
    }
    println!("limbs,rep,delta_ns");
    for pair in &pairs {
        println!("{},{},{}", pair.limbs, pair.rep, pair.delta_ns);
    }
    let archive = BenchArchive {
        task_id: cli.task_id.clone(),
        tier: "interleave".to_string(),
        timeout_seconds: cli.timeout_seconds,
        threads_requested: cli.threads,
        threads_effective,
        seeds: vec![cli.seed],
        timestamp_unix_secs,
        observations: Vec::new(),
        sweep_limbs: Some(limbs_list),
        sweep_reps: Some(reps),
        sweep_engine: None,
        naked_engines: None,
        faceoff_engines: None,
        faceoff_ratios: None,
        interleave_engines: Some(legs.iter().map(|leg| leg.name()).collect()),
        interleave_pairs: Some(pairs),
        interleave_summary: Some(summaries),
        asym_cells: None,
        warmup: Some(warmup),
        small_cell_ns: Some(small_ns),
        small_reps: Some(small_reps),
        verify_naked: Some(cli.verify_naked),
        sweep_records: Some(records.clone()),
        sweep_summary: None,
    };
    if let Some(parent) = cli.output.parent() {
        let created = fs::create_dir_all(parent);
        if let Err(error) = created {
            eprintln!("error: internal: cannot create output dir: {error}");
            return ExitCode::FAILURE;
        }
    }
    let payload = match serde_json::to_vec_pretty(&archive) {
        Ok(payload) => payload,
        Err(error) => {
            eprintln!("error: internal: cannot encode archive: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = fs::write(&cli.output, payload) {
        eprintln!("error: internal: cannot write output: {error}");
        return ExitCode::FAILURE;
    }
    println!("json: {}", cli.output.display());
    if let Some(csv_path) = &cli.csv {
        if let Some(parent) = csv_path.parent() {
            if !parent.as_os_str().is_empty() {
                let created = fs::create_dir_all(parent);
                if let Err(error) = created {
                    eprintln!("error: internal: cannot create csv dir: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        if let Err(error) = fs::write(csv_path, sweep_csv(&records)) {
            eprintln!("error: internal: cannot write csv: {error}");
            return ExitCode::FAILURE;
        }
        println!("csv: {}", csv_path.display());
    }
    if all_verified(
        archive.sweep_records.as_deref().unwrap_or(&[]),
        cli.verify_naked,
    ) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn run_asym(cli: &Cli) -> ExitCode {
    let engines_text = match &cli.engines {
        Some(text) => text.clone(),
        None => {
            eprintln!("error: input: --engines <A,B> is required for the asym tier");
            return ExitCode::FAILURE;
        }
    };
    let legs = match parse_interleave_engines(&engines_text) {
        Ok(legs) => legs,
        Err(error) => {
            eprintln!("error: input: {error}");
            return ExitCode::FAILURE;
        }
    };
    if cli.reps == Some(0) {
        eprintln!("error: input: reps must be at least 1");
        return ExitCode::FAILURE;
    }
    let reps = effective_reps("asym", cli.reps);
    let warmup = effective_warmup(cli.warmup);
    let small_ns = effective_small_cell_ns(cli.small_cell_ns);
    let small_reps = effective_small_reps(cli.small_reps);
    let threads_effective = cli.threads.unwrap_or(
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1),
    );
    let timestamp_unix_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let cases = build_asym_cases(cli.seed);
    let mut records = Vec::new();
    let mut summaries = Vec::new();
    let mut pairs = Vec::new();
    for case in &cases {
        let a_limbs = graymul_core::from_bigint(&case.a);
        let b_limbs = graymul_core::from_bigint(&case.b);
        let limbs = a_limbs.len().max(b_limbs.len());
        let (cell_records, cell_pairs, summary) = run_interleave_cell(
            case,
            &a_limbs,
            &b_limbs,
            limbs,
            &legs,
            reps,
            warmup,
            cli.verify_naked,
            small_ns,
            small_reps,
            cli.threads,
        );
        summaries.push(summary);
        records.extend(cell_records);
        pairs.extend(cell_pairs);
    }
    println!(
        "limbs,engine_a,engine_b,reps,median_a_ns,median_b_ns,median_delta_ns,std_delta_ns,ratio,verified_all"
    );
    for summary in &summaries {
        println!(
            "{},{},{},{},{},{},{},{},{},{},{}",
            summary.limbs,
            summary.engine_a,
            summary.engine_b,
            summary.reps,
            summary.median_a_ns,
            summary.median_b_ns,
            summary.median_delta_ns,
            summary.std_delta_ns,
            summary.ratio_a_over_b,
            summary.cv_delta,
            summary.verified_all
        );
    }
    println!("limbs,rep,delta_ns");
    for pair in &pairs {
        println!("{},{},{}", pair.limbs, pair.rep, pair.delta_ns);
    }
    let asym_cells: Vec<AsymCell> = ASYM_SMALL_BIG
        .iter()
        .map(|(small, big)| AsymCell {
            small: *small,
            big: *big,
        })
        .collect();
    let archive = BenchArchive {
        task_id: cli.task_id.clone(),
        tier: "asym".to_string(),
        timeout_seconds: cli.timeout_seconds,
        threads_requested: cli.threads,
        threads_effective,
        seeds: vec![cli.seed],
        timestamp_unix_secs,
        observations: Vec::new(),
        sweep_limbs: None,
        sweep_reps: Some(reps),
        sweep_engine: None,
        naked_engines: None,
        faceoff_engines: None,
        faceoff_ratios: None,
        interleave_engines: Some(legs.iter().map(|leg| leg.name()).collect()),
        interleave_pairs: Some(pairs),
        interleave_summary: Some(summaries),
        asym_cells: Some(asym_cells),
        warmup: Some(warmup),
        small_cell_ns: Some(small_ns),
        small_reps: Some(small_reps),
        verify_naked: Some(cli.verify_naked),
        sweep_records: Some(records.clone()),
        sweep_summary: None,
    };
    if let Some(parent) = cli.output.parent() {
        let created = fs::create_dir_all(parent);
        if let Err(error) = created {
            eprintln!("error: internal: cannot create output dir: {error}");
            return ExitCode::FAILURE;
        }
    }
    let payload = match serde_json::to_vec_pretty(&archive) {
        Ok(payload) => payload,
        Err(error) => {
            eprintln!("error: internal: cannot encode archive: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = fs::write(&cli.output, payload) {
        eprintln!("error: internal: cannot write output: {error}");
        return ExitCode::FAILURE;
    }
    println!("json: {}", cli.output.display());
    if let Some(csv_path) = &cli.csv {
        if let Some(parent) = csv_path.parent() {
            if !parent.as_os_str().is_empty() {
                let created = fs::create_dir_all(parent);
                if let Err(error) = created {
                    eprintln!("error: internal: cannot write csv dir: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        if let Err(error) = fs::write(csv_path, sweep_csv(&records)) {
            eprintln!("error: internal: cannot write csv: {error}");
            return ExitCode::FAILURE;
        }
        println!("csv: {}", csv_path.display());
    }
    if all_verified(
        archive.sweep_records.as_deref().unwrap_or(&[]),
        cli.verify_naked,
    ) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graymul_core::{Engine, from_bigint, to_bigint};

    #[test]
    fn tier_table_covers_five_limbs_scales() {
        assert_eq!(tier_limbs("tiny"), Some(1));
        assert_eq!(tier_limbs("small"), Some(4));
        assert_eq!(tier_limbs("medium"), Some(8));
        assert_eq!(tier_limbs("large"), Some(32));
        assert_eq!(tier_limbs("huge"), Some(128));
        assert_eq!(tier_limbs("nope"), None);
    }

    #[test]
    fn tier_selection_all_or_single_or_error() {
        assert_eq!(selected_tiers("all").expect("all").len(), 5);
        assert_eq!(
            selected_tiers("tiny").expect("tiny"),
            vec!["tiny".to_string()]
        );
        assert!(selected_tiers("nope").is_err());
    }

    #[test]
    fn splitmix64_stream_matches_reference_values() {
        let mut state = 0x9e3779b97f4a7c15u64;
        assert_eq!(splitmix64(&mut state), 7960286522194355700);
        assert_eq!(splitmix64(&mut state), 487617019471545679);
    }

    #[test]
    fn seeded_operands_hit_exact_limb_counts_deterministically() {
        let first = build_cases(&["small".to_string()], 7);
        let second = build_cases(&["small".to_string()], 7);
        assert_eq!(first[0].a, second[0].a);
        assert_eq!(first[0].b, second[0].b);
        assert_eq!(from_bigint(&first[0].a).len(), 4);
        assert_eq!(from_bigint(&first[0].b).len(), 4);
        let other = build_cases(&["small".to_string()], 8);
        assert!(other[0].a != first[0].a || other[0].b != first[0].b);
    }

    #[test]
    fn tiny_case_uses_fixed_pair_and_verifies() {
        let cases = build_cases(&["tiny".to_string()], 99);
        assert_eq!(cases[0].a, BigUint::from(12345u32));
        assert_eq!(cases[0].b, BigUint::from(67890u32));
        let observation = run_case(&cases[0], 30, None, 99);
        assert_eq!(observation.status, "ok");
        assert!(observation.verified);
        assert!(Engine::from_str(&observation.method).is_some());
        assert_eq!(&cases[0].a * &cases[0].b, BigUint::from(838102050u32));
    }

    #[test]
    fn zero_timeout_reports_resource_limit_without_abort() {
        let cases = build_cases(&["tiny".to_string()], 99);
        let observation = run_case(&cases[0], 0, None, 99);
        assert_eq!(observation.status, "resource-limit");
        assert!(!observation.verified);
        assert!(
            observation
                .error
                .as_deref()
                .is_some_and(|text| text.starts_with("resource-limit:"))
        );
    }

    #[test]
    fn archive_envelope_carries_provenance_keys() {
        let cases = build_cases(&["tiny".to_string()], 99);
        let archive = BenchArchive {
            task_id: "0.0.0.1.4".to_string(),
            tier: "tiny".to_string(),
            timeout_seconds: 30,
            threads_requested: None,
            threads_effective: 4,
            seeds: vec![99],
            timestamp_unix_secs: 1,
            observations: vec![run_case(&cases[0], 30, None, 99)],
            sweep_limbs: None,
            sweep_reps: None,
            sweep_engine: None,
            naked_engines: None,
            faceoff_engines: None,
            faceoff_ratios: None,
            warmup: None,
            small_cell_ns: None,
            small_reps: None,
            verify_naked: None,
            sweep_records: None,
            sweep_summary: None,
            interleave_engines: None,
            interleave_pairs: None,
            interleave_summary: None,
            asym_cells: None,
        };
        let value = serde_json::to_value(&archive).expect("serializes");
        for key in [
            "task_id",
            "tier",
            "timeout_seconds",
            "threads_requested",
            "threads_effective",
            "seeds",
            "timestamp_unix_secs",
            "observations",
        ] {
            assert!(value.get(key).is_some(), "missing key {key}");
        }
        let text = serde_json::to_string(&value).expect("round trip");
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("parses");
        assert_eq!(parsed, value);
    }

    #[test]
    fn classic_envelope_omits_sweep_keys() {
        let cases = build_cases(&["tiny".to_string()], 99);
        let archive = BenchArchive {
            task_id: "0.0.0.1.4".to_string(),
            tier: "tiny".to_string(),
            timeout_seconds: 30,
            threads_requested: None,
            threads_effective: 4,
            seeds: vec![99],
            timestamp_unix_secs: 1,
            observations: vec![run_case(&cases[0], 30, None, 99)],
            sweep_limbs: None,
            sweep_reps: None,
            sweep_engine: None,
            naked_engines: None,
            faceoff_engines: None,
            faceoff_ratios: None,
            warmup: None,
            small_cell_ns: None,
            small_reps: None,
            verify_naked: None,
            sweep_records: None,
            sweep_summary: None,
            interleave_engines: None,
            interleave_pairs: None,
            interleave_summary: None,
            asym_cells: None,
        };
        let value = serde_json::to_value(&archive).expect("serializes");
        for key in [
            "sweep_limbs",
            "sweep_reps",
            "sweep_engine",
            "sweep_records",
            "sweep_summary",
            "naked_engines",
            "faceoff_engines",
            "faceoff_ratios",
            "warmup",
            "small_cell_ns",
            "small_reps",
            "verify_naked",
            "interleave_engines",
            "interleave_pairs",
            "interleave_summary",
        ] {
            assert!(value.get(key).is_none(), "leaked key {key}");
        }
    }

    #[test]
    fn sweep_flag_parsing_defaults_and_custom() {
        let cli = Cli::try_parse_from(["graymul-bench", "--tier", "sweep"]).expect("parse sweep");
        assert_eq!(cli.tier, "sweep");
        assert_eq!(cli.reps, None);
        assert_eq!(effective_reps("sweep", cli.reps), 3);
        assert_eq!(effective_reps("naked", cli.reps), 15);
        assert_eq!(effective_reps("interleave", cli.reps), 15);
        assert_eq!(cli.engine, "auto");
        assert_eq!(cli.csv, None);
        assert_eq!(
            parse_sweep_limbs(&cli.sweep_limbs).expect("defaults"),
            vec![
                1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
            ]
        );
        let custom = Cli::try_parse_from([
            "graymul-bench",
            "--tier",
            "sweep",
            "--sweep-limbs",
            "4, 8",
            "--reps",
            "2",
            "--engine",
            "karatsuba",
            "--csv",
            "out.csv",
        ])
        .expect("parse custom");
        assert_eq!(
            parse_sweep_limbs(&custom.sweep_limbs).expect("custom"),
            vec![4, 8]
        );
        assert_eq!(custom.reps, Some(2));
        assert!(Engine::from_str(&custom.engine).is_some());
        assert_eq!(custom.csv, Some(std::path::PathBuf::from("out.csv")));
    }

    #[test]
    fn sweep_rejects_bad_engine_and_bad_lists() {
        assert!(parse_sweep_limbs("").is_err());
        assert!(parse_sweep_limbs("abc").is_err());
        assert!(parse_sweep_limbs("0").is_err());
        assert!(parse_sweep_limbs("4,,8").is_err());
        assert!(parse_sweep_limbs("4,-8").is_err());
        assert!(Engine::from_str("karatsuba").is_some());
        assert!(Engine::from_str("wat").is_none());
    }

    #[test]
    fn median_odd_even_and_empty() {
        assert_eq!(median_of(&[3, 1, 2]), Some(2.0));
        assert_eq!(median_of(&[4, 1, 3, 2]), Some(2.5));
        assert_eq!(median_of(&[7]), Some(7.0));
        assert_eq!(median_of(&[]), None);
    }

    #[test]
    fn coeff_var_zero_mean_guarded() {
        assert_eq!(coeff_var(&[]), None);
        assert_eq!(coeff_var(&[0, 0, 0]), Some(0.0));
        assert_eq!(coeff_var(&[5, 5, 5, 5]), Some(0.0));
        let spread = coeff_var(&[0, 10]).expect("spread");
        assert!((spread - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sweep_generation_is_deterministic() {
        let first = build_sweep_cases(&[4, 8], 7);
        let second = build_sweep_cases(&[4, 8], 7);
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].name, "sweep_4");
        assert_eq!(first[0].a, second[0].a);
        assert_eq!(first[0].b, second[0].b);
        assert_eq!(first[1].a, second[1].a);
        assert_eq!(from_bigint(&first[0].a).len(), 4);
        assert_eq!(from_bigint(&first[1].a).len(), 8);
    }

    #[test]
    fn sweep_record_verifies_product() {
        let cases = build_sweep_cases(&[2], 7);
        let record = run_sweep_rep(&cases[0], "karatsuba", 0, 30, None, 7, false);
        assert!(record.verified);
        assert_eq!(record.method, "karatsuba");
        assert_eq!(record.limbs, 2);
        assert_eq!(record.rep, 0);
    }

    #[test]
    fn control_rep_times_and_verifies() {
        let cases = build_sweep_cases(&[2], 7);
        let record = run_sweep_rep(&cases[0], "num-bigint", 0, 30, None, 7, false);
        assert!(record.verified);
        assert_eq!(record.engine, "num-bigint");
        assert_eq!(record.method, "num-bigint");
        assert_eq!(record.limbs, 2);
        assert_eq!(record.rep, 0);
    }

    #[test]
    fn sweep_summary_reports_median_and_cv() {
        let records = vec![
            SweepRecord {
                limbs: 4,
                rep: 0,
                engine: "karatsuba".to_string(),
                method: "karatsuba".to_string(),
                elapsed_ns: 100,
                ns_per_limb: 25.0,
                verified: true,
                verify_in_timer: true,
                warmup: false,
            },
            SweepRecord {
                limbs: 4,
                rep: 1,
                engine: "karatsuba".to_string(),
                method: "karatsuba".to_string(),
                elapsed_ns: 300,
                ns_per_limb: 75.0,
                verified: true,
                verify_in_timer: true,
                warmup: false,
            },
            SweepRecord {
                limbs: 4,
                rep: 2,
                engine: "karatsuba".to_string(),
                method: "karatsuba".to_string(),
                elapsed_ns: 200,
                ns_per_limb: 50.0,
                verified: true,
                verify_in_timer: true,
                warmup: false,
            },
        ];
        let summary = summarize_sweep(4, "karatsuba", &records);
        assert_eq!(summary.median_ns, 200.0);
        assert_eq!(summary.median_ns_per_limb, 50.0);
        assert_eq!(summary.reps, 3);
        assert!(summary.verified_all);
        assert!(summary.cv >= 0.0);
    }

    #[test]
    fn naked_shares_sweep_operand_stream() {
        for limbs in [1usize, 4, 8] {
            let first = build_sweep_cases(&[limbs], 0x9e3779b97f4a7c15);
            let second = build_sweep_cases(&[limbs], 0x9e3779b97f4a7c15);
            assert_eq!(first[0].a, second[0].a);
            assert_eq!(first[0].b, second[0].b);
            let a_limbs = from_bigint(&first[0].a);
            let b_limbs = from_bigint(&first[0].b);
            assert_eq!(a_limbs.len(), limbs);
            assert_eq!(b_limbs.len(), limbs);
            assert_ne!(a_limbs[limbs - 1] >> 63, 0);
            assert_ne!(b_limbs[limbs - 1] >> 63, 0);
        }
    }

    #[test]
    fn naked_rep_verifies_posthoc_without_timer_verify() {
        let cases = build_sweep_cases(&[4], 7);
        let a_limbs = from_bigint(&cases[0].a);
        let b_limbs = from_bigint(&cases[0].b);
        let record = run_naked_rep(&a_limbs, &b_limbs, Engine::Schoolbook, 4, 0, false, true);
        assert!(record.verified);
        assert!(!record.verify_in_timer);
        assert_eq!(record.method, "schoolbook");
        assert_eq!(record.limbs, 4);
    }

    #[test]
    fn naked_rejects_bad_engine_list() {
        assert!(parse_naked_engines("schoolbook,karatsuba").is_ok());
        assert!(parse_naked_engines("wat").is_err());
        assert!(parse_naked_engines("").is_err());
        assert!(parse_naked_engines("hvh-full").is_err());
        assert!(parse_naked_engines("schoolbook, hvh-full").is_err());
        let defaults = parse_naked_engines("schoolbook,karatsuba,toom3,toom4,ssa,furer,hvh")
            .expect("defaults");
        assert_eq!(defaults.len(), 7);
        assert!(!defaults.contains(&Leg::Core(Engine::HvhFull)));
    }

    #[test]
    fn naked_envelope_carries_provenance_keys() {
        let archive = BenchArchive {
            task_id: "0.0.0.12.1".to_string(),
            tier: "naked".to_string(),
            timeout_seconds: 30,
            threads_requested: None,
            threads_effective: 4,
            seeds: vec![7],
            timestamp_unix_secs: 1,
            observations: Vec::new(),
            sweep_limbs: Some(vec![4]),
            sweep_reps: Some(2),
            sweep_engine: None,
            naked_engines: Some(vec!["schoolbook".to_string()]),
            faceoff_engines: None,
            faceoff_ratios: None,
            warmup: None,
            small_cell_ns: None,
            small_reps: None,
            verify_naked: None,
            sweep_records: Some(Vec::new()),
            sweep_summary: Some(Vec::new()),
            interleave_engines: None,
            interleave_pairs: None,
            interleave_summary: None,
            asym_cells: None,
        };
        let value = serde_json::to_value(&archive).expect("serializes");
        for key in [
            "task_id",
            "tier",
            "timeout_seconds",
            "threads_requested",
            "threads_effective",
            "seeds",
            "timestamp_unix_secs",
            "observations",
            "naked_engines",
        ] {
            assert!(value.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(
            value.get("tier").and_then(|tier| tier.as_str()),
            Some("naked")
        );
    }

    #[test]
    fn naked_kernel_beats_sweep_wall_same_product() {
        for (limbs, engine) in [(4usize, "schoolbook"), (8usize, "karatsuba")] {
            let cases = build_sweep_cases(&[limbs], 7);
            let a_limbs = from_bigint(&cases[0].a);
            let b_limbs = from_bigint(&cases[0].b);
            let parsed = Engine::from_str(engine).expect("engine");
            let mut naked_ns = Vec::new();
            for rep in 0..3 {
                let record = run_naked_rep(&a_limbs, &b_limbs, parsed, limbs, rep, false, true);
                assert!(record.verified);
                naked_ns.push(record.elapsed_ns);
                let product = to_bigint(&graymul_core::multiply_naked(parsed, &a_limbs, &b_limbs));
                assert_eq!(product, &cases[0].a * &cases[0].b);
            }
            let mut sweep_ns = Vec::new();
            for rep in 0..3 {
                let record = run_sweep_rep(&cases[0], engine, rep, 30, None, 7, false);
                assert!(record.verified);
                sweep_ns.push(record.elapsed_ns);
            }
            let naked_med = median_of(&naked_ns).expect("naked median");
            let sweep_med = median_of(&sweep_ns).expect("sweep median");
            assert!(
                naked_med < sweep_med,
                "naked {naked_med} not below sweep {sweep_med} at {limbs} limbs"
            );
        }
    }

    fn warmup_record(elapsed_ns: u128, warmup: bool) -> SweepRecord {
        SweepRecord {
            limbs: 4,
            rep: 0,
            engine: "schoolbook".to_string(),
            method: "schoolbook".to_string(),
            elapsed_ns,
            ns_per_limb: elapsed_ns as f64 / 4.0,
            verified: true,
            verify_in_timer: false,
            warmup,
        }
    }

    #[test]
    fn warmup_discard_shapes_stats() {
        let records = vec![
            warmup_record(10_000, true),
            warmup_record(100, false),
            warmup_record(200, false),
            warmup_record(300, false),
        ];
        let summary = summarize_sweep(4, "schoolbook", &records);
        assert_eq!(summary.median_ns, 200.0);
        assert_eq!(summary.reps, 3);
        assert_eq!(summary.warmup_discarded, 1);
        assert!(summary.verified_all);
        let plain = vec![
            warmup_record(100, false),
            warmup_record(200, false),
            warmup_record(300, false),
        ];
        let unwarmed = summarize_sweep(4, "schoolbook", &plain);
        assert_eq!(unwarmed.median_ns, 200.0);
        assert_eq!(unwarmed.warmup_discarded, 0);
    }

    #[test]
    fn warmup_and_extension_flags_parse() {
        let cli = Cli::try_parse_from(["graymul-bench", "--tier", "naked"]).expect("parse naked");
        assert_eq!(cli.warmup, None);
        assert_eq!(effective_warmup(cli.warmup), 1);
        assert_eq!(effective_warmup(Some(0)), 0);
        assert_eq!(effective_warmup(Some(2)), 2);
        assert_eq!(effective_small_cell_ns(None), 1_000_000);
        assert_eq!(effective_small_reps(None), 15);
        assert!(needs_extension(500.0, 1_000_000));
        assert!(!needs_extension(2_000_000.0, 1_000_000));
        assert!(!needs_extension(1_000_000.0, 1_000_000));
        let custom = Cli::try_parse_from([
            "graymul-bench",
            "--tier",
            "faceoff",
            "--warmup",
            "2",
            "--small-cell-ns",
            "500",
            "--small-reps",
            "7",
        ])
        .expect("parse faceoff flags");
        assert_eq!(custom.warmup, Some(2));
        assert_eq!(custom.small_cell_ns, Some(500));
        assert_eq!(custom.small_reps, Some(7));
    }

    #[test]
    fn faceoff_interleaves_engines_per_rep() {
        let cases = build_sweep_cases(&[4], 7);
        let a_limbs = from_bigint(&cases[0].a);
        let b_limbs = from_bigint(&cases[0].b);
        let engines = vec![Leg::Core(Engine::Schoolbook), Leg::Core(Engine::Karatsuba)];
        let records = faceoff_cell_records(&a_limbs, &b_limbs, 4, &engines, 2, 1, true, None);
        assert_eq!(records.len(), 6);
        let names: Vec<&str> = records
            .iter()
            .map(|record| record.engine.as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                "schoolbook",
                "karatsuba",
                "schoolbook",
                "karatsuba",
                "schoolbook",
                "karatsuba"
            ]
        );
        let reps: Vec<usize> = records.iter().map(|record| record.rep).collect();
        assert_eq!(reps, vec![0, 1, 2, 3, 4, 5]);
        for (index, record) in records.iter().enumerate() {
            assert_eq!(record.rep % engines.len(), index % engines.len());
            assert_eq!(record.warmup, index < engines.len());
            assert!(record.verified);
            assert!(!record.verify_in_timer);
        }
    }

    #[test]
    fn faceoff_first_k_match_sweep() {
        for limbs in [4usize, 8] {
            let sweep = build_sweep_cases(&[limbs], 0x9e3779b97f4a7c15);
            let faceoff = build_sweep_cases(&[limbs], 0x9e3779b97f4a7c15);
            assert_eq!(sweep[0].a, faceoff[0].a);
            assert_eq!(sweep[0].b, faceoff[0].b);
            assert_eq!(from_bigint(&faceoff[0].a).len(), limbs);
            assert_eq!(from_bigint(&faceoff[0].b).len(), limbs);
        }
        let defaults = parse_naked_engines("schoolbook,karatsuba,toom3,toom4,ssa,furer,hvh")
            .expect("faceoff defaults");
        assert_eq!(defaults.len(), 7);
    }

    #[test]
    fn verify_flag_parses_default_on() {
        let cli = Cli::try_parse_from(["graymul-bench", "--tier", "naked"]).expect("parse naked");
        assert!(cli.verify_naked);
        let off = Cli::try_parse_from([
            "graymul-bench",
            "--tier",
            "naked",
            "--verify-naked",
            "false",
        ])
        .expect("parse opt-out");
        assert!(!off.verify_naked);
    }

    #[test]
    fn naked_rep_verify_skip_marks_unverified() {
        let cases = build_sweep_cases(&[4], 7);
        let a_limbs = from_bigint(&cases[0].a);
        let b_limbs = from_bigint(&cases[0].b);
        let checked = run_naked_rep(&a_limbs, &b_limbs, Engine::Schoolbook, 4, 0, false, true);
        assert!(checked.verified);
        assert!(!checked.verify_in_timer);
        let skipped = run_naked_rep(&a_limbs, &b_limbs, Engine::Schoolbook, 4, 0, false, false);
        assert!(!skipped.verified);
        assert!(!skipped.verify_in_timer);
        assert!(!all_verified(std::slice::from_ref(&skipped), true));
        assert!(all_verified(std::slice::from_ref(&skipped), false));
        assert!(all_verified(std::slice::from_ref(&checked), true));
    }

    #[test]
    fn interleave_shares_sweep_operand_stream() {
        for limbs in [1usize, 4, 8] {
            let first = build_sweep_cases(&[limbs], 0x9e3779b97f4a7c15);
            let second = build_sweep_cases(&[limbs], 0x9e3779b97f4a7c15);
            assert_eq!(first[0].a, second[0].a);
            assert_eq!(first[0].b, second[0].b);
            let a_limbs = from_bigint(&first[0].a);
            let b_limbs = from_bigint(&first[0].b);
            assert_eq!(a_limbs.len(), limbs);
            assert_eq!(b_limbs.len(), limbs);
            assert_ne!(a_limbs[limbs - 1] >> 63, 0);
            assert_ne!(b_limbs[limbs - 1] >> 63, 0);
        }
    }

    #[test]
    fn interleave_rejects_bad_engine_lists() {
        assert!(parse_interleave_engines("schoolbook").is_err());
        assert!(parse_interleave_engines("schoolbook,karatsuba,toom3").is_err());
        assert!(parse_interleave_engines("wat,schoolbook").is_err());
        assert!(parse_interleave_engines("").is_err());
        assert!(parse_interleave_engines("hvh-full,schoolbook").is_err());
        assert!(parse_interleave_engines("schoolbook,hvh-full").is_err());
        let pair = parse_interleave_engines("schoolbook,karatsuba").expect("pair");
        assert_eq!(pair.len(), 2);
        let control = parse_interleave_engines("schoolbook,num-bigint").expect("control pair");
        assert_eq!(control[1], Leg::NumBigint);
    }

    #[test]
    fn split_leg_parses_everywhere() {
        assert_eq!(Leg::Split.name(), "split");
        let naked = parse_naked_engines("schoolbook,split").expect("naked split");
        assert_eq!(naked, vec![Leg::Core(Engine::Schoolbook), Leg::Split]);
        let pair = parse_interleave_engines("split,schoolbook").expect("split pair");
        assert_eq!(pair, [Leg::Split, Leg::Core(Engine::Schoolbook)]);
        assert!(parse_interleave_engines("split").is_err());
    }

    #[test]
    fn split_rep_matches_schoolbook_product() {
        let cases = build_sweep_cases(&[128], 0x9e3779b97f4a7c15);
        let a_limbs = from_bigint(&cases[0].a);
        let b_limbs = from_bigint(&cases[0].b);
        let record = run_split_rep(&a_limbs, &b_limbs, 128, 0, false, true, None);
        assert!(record.verified);
        assert_eq!(record.engine, "split");
        let plain = run_naked_rep(&a_limbs, &b_limbs, Engine::Schoolbook, 128, 0, false, true);
        assert!(plain.verified);
        let split_product = graymul_core::mul_split_par(&a_limbs, &b_limbs, 2);
        let schoolbook_product =
            graymul_core::multiply_naked(Engine::Schoolbook, &a_limbs, &b_limbs);
        assert_eq!(split_product, schoolbook_product);
    }

    #[test]
    fn threads_override_routes_par_entries() {
        let cases = build_sweep_cases(&[128], 0x9e3779b97f4a7c15);
        let a_limbs = from_bigint(&cases[0].a);
        let b_limbs = from_bigint(&cases[0].b);
        let serial = run_naked_leg(
            &a_limbs,
            &b_limbs,
            Leg::Core(Engine::Ssa),
            128,
            0,
            false,
            true,
            None,
        );
        assert!(serial.verified);
        assert_eq!(serial.engine, "ssa");
        let threaded = run_naked_leg(
            &a_limbs,
            &b_limbs,
            Leg::Core(Engine::Ssa),
            128,
            0,
            false,
            true,
            Some(4),
        );
        assert!(threaded.verified);
        assert_eq!(threaded.engine, "ssa");
        let expected = graymul_core::schonhage::mul_par(&a_limbs, &b_limbs, 4);
        let reference = graymul_core::multiply_naked(Engine::Ssa, &a_limbs, &b_limbs);
        assert_eq!(expected, reference);
        let split_default = run_split_rep(&a_limbs, &b_limbs, 128, 0, false, true, None);
        let split_two = run_split_rep(&a_limbs, &b_limbs, 128, 0, false, true, Some(2));
        assert!(split_default.verified);
        assert!(split_two.verified);
        assert_eq!(
            graymul_core::mul_split_par(&a_limbs, &b_limbs, 16),
            graymul_core::mul_split_par(&a_limbs, &b_limbs, 1)
        );
    }

    #[test]
    fn interleave_accepts_composite_ids() {
        let expected = vec![
            "schoolbook/scalar",
            #[cfg(target_arch = "x86_64")]
            "schoolbook/adx",
            "karatsuba/core",
            "ssa/crt",
            "ssa/small",
            "ssa/flat",
            "ssa/bailey",
            "ssa/trunc",
        ];
        assert_eq!(ProbeId::valid_ids(), expected);
        for id in ProbeId::valid_ids() {
            let probe = ProbeId::from_str(id).expect("valid id");
            assert_eq!(probe.name(), id);
        }
        let pair =
            parse_interleave_engines("schoolbook/scalar,schoolbook").expect("composite pair");
        assert_eq!(pair[0], Leg::Probe(ProbeId::SchoolbookScalar));
        assert_eq!(pair[1], Leg::Core(Engine::Schoolbook));
        assert_eq!(pair[0].name(), "schoolbook/scalar");
    }

    #[test]
    fn interleave_rejects_bad_composite_ids() {
        for bad in [
            "schoolbook/nope",
            "ssa/",
            "/scalar",
            "schoolbook/",
            "toom3/core",
            "schoolbook/scalar/extra",
        ] {
            let error = parse_interleave_engines(bad).unwrap_err();
            assert!(
                error.contains("valid ids:"),
                "missing id listing for {bad}: {error}"
            );
        }
        assert!(parse_interleave_engines("schoolbook/scalar").is_err());
        assert!(parse_interleave_engines("ssa/small,").is_err());
        assert!(parse_interleave_engines("num-bigint,schoolbook/scalar").is_ok());
    }

    #[test]
    fn composite_ids_match_plain_engine_products() {
        let cases = build_sweep_cases(&[128], 0x9e3779b97f4a7c15);
        let a_limbs = from_bigint(&cases[0].a);
        let b_limbs = from_bigint(&cases[0].b);
        assert_eq!(a_limbs.len(), 128);
        let plain_schoolbook = graymul_core::multiply_naked(Engine::Schoolbook, &a_limbs, &b_limbs);
        let plain_karatsuba = graymul_core::multiply_naked(Engine::Karatsuba, &a_limbs, &b_limbs);
        let plain_ssa = graymul_core::multiply_naked(Engine::Ssa, &a_limbs, &b_limbs);
        assert_eq!(
            graymul_core::schoolbook::measure_scalar(&a_limbs, &b_limbs),
            plain_schoolbook
        );
        #[cfg(target_arch = "x86_64")]
        assert_eq!(
            graymul_core::schoolbook::measure_adx(&a_limbs, &b_limbs),
            plain_schoolbook
        );
        assert_eq!(
            graymul_core::karatsuba::mul_core(
                &a_limbs,
                &b_limbs,
                &mut graymul_core::karatsuba::WorkScratch::new()
            ),
            plain_karatsuba
        );
        assert_eq!(
            graymul_core::schonhage::mul_crt(&a_limbs, &b_limbs),
            plain_ssa
        );
        assert_eq!(
            graymul_core::schonhage::mul_small(&a_limbs, &b_limbs),
            plain_ssa
        );
        assert_eq!(
            graymul_core::small_ntt::measure_crt_flat(&a_limbs, &b_limbs),
            plain_ssa
        );
        assert_eq!(
            graymul_core::small_ntt::measure_crt_bailey(&a_limbs, &b_limbs),
            plain_ssa
        );
        assert_eq!(
            graymul_core::small_ntt::measure_crt_trunc(&a_limbs, &b_limbs),
            plain_ssa
        );
    }

    #[test]
    fn composite_identical_path_delta_near_zero() {
        let cases = build_sweep_cases(&[8], 7);
        let a_limbs = from_bigint(&cases[0].a);
        let b_limbs = from_bigint(&cases[0].b);
        let legs = [
            Leg::Probe(ProbeId::SchoolbookScalar),
            Leg::Probe(ProbeId::SchoolbookScalar),
        ];
        let (records, pairs, summary) = run_interleave_cell(
            &cases[0], &a_limbs, &b_limbs, 8, &legs, 10, 1, true, 1_000_000, 15, None,
        );
        assert_eq!(pairs.len(), summary.reps);
        assert!(records.iter().all(|record| record.verified));
        assert!(summary.verified_all);
        assert_eq!(summary.engine_a, "schoolbook/scalar");
        assert_eq!(summary.engine_b, "schoolbook/scalar");
        assert!(summary.median_delta_ns.abs() <= 3.0 * summary.std_delta_ns);
    }

    #[test]
    fn interleave_identical_path_delta_near_zero() {
        let cases = build_sweep_cases(&[8], 7);
        let a_limbs = from_bigint(&cases[0].a);
        let b_limbs = from_bigint(&cases[0].b);
        let legs = [Leg::Core(Engine::Schoolbook), Leg::Core(Engine::Schoolbook)];
        let (records, pairs, summary) = run_interleave_cell(
            &cases[0], &a_limbs, &b_limbs, 8, &legs, 10, 1, true, 1_000_000, 15, None,
        );
        assert_eq!(pairs.len(), summary.reps);
        assert!(records.iter().all(|record| record.verified));
        assert!(summary.verified_all);
        assert_eq!(summary.engine_a, "schoolbook");
        assert_eq!(summary.engine_b, "schoolbook");
        assert!(summary.median_delta_ns.abs() <= 3.0 * summary.std_delta_ns);
        assert_eq!(summary.warmup_discarded, 2);
    }

    #[test]
    fn asym_cells_cover_eight_ratio_cells_deterministically() {
        let first = build_asym_cases(0x9e3779b97f4a7c15);
        let second = build_asym_cases(0x9e3779b97f4a7c15);
        assert_eq!(first.len(), ASYM_SMALL_BIG.len());
        for (index, (small, big)) in ASYM_SMALL_BIG.iter().enumerate() {
            assert_eq!(from_bigint(&first[index].a).len(), *small);
            assert_eq!(from_bigint(&first[index].b).len(), *big);
            assert_eq!(first[index].name, format!("asym_{small}x{big}"));
            assert_eq!(first[index].a, second[index].a);
            assert_eq!(first[index].b, second[index].b);
        }
        let other = build_asym_cases(8);
        assert!(other[0].a != first[0].a || other[0].b != first[0].b);
        assert_eq!(
            effective_reps("asym", None),
            effective_reps("interleave", None)
        );
    }

    #[test]
    fn asym_chunked_leg_matches_serial_bit_identical() {
        let pair = parse_interleave_engines("chunked/schoolbook,schoolbook").expect("chunked pair");
        assert_eq!(pair[0].name(), "chunked/schoolbook");
        assert_eq!(pair[1].name(), "schoolbook");
        assert!(parse_interleave_engines("chunked/hvh-full,schoolbook").is_err());
        assert!(parse_interleave_engines("chunked/wat,schoolbook").is_err());
        let cases = build_asym_cases(0x9e3779b97f4a7c15);
        for case in cases.iter().take(6) {
            let a_limbs = from_bigint(&case.a);
            let b_limbs = from_bigint(&case.b);
            let limbs = a_limbs.len().max(b_limbs.len());
            let record = run_chunked_rep(
                &a_limbs,
                &b_limbs,
                Engine::Schoolbook,
                limbs,
                0,
                false,
                true,
            );
            assert!(record.verified);
            assert_eq!(record.engine, "chunked/schoolbook");
            assert_eq!(
                asym_chunked_mul(&a_limbs, &b_limbs, Engine::Schoolbook),
                graymul_core::multiply_naked(Engine::Schoolbook, &a_limbs, &b_limbs)
            );
        }
    }

    #[test]
    fn interleave_envelope_carries_provenance_keys() {
        let archive = BenchArchive {
            task_id: "0.0.0.17.1".to_string(),
            tier: "interleave".to_string(),
            timeout_seconds: 30,
            threads_requested: None,
            threads_effective: 4,
            seeds: vec![7],
            timestamp_unix_secs: 1,
            observations: Vec::new(),
            sweep_limbs: Some(vec![8]),
            sweep_reps: Some(2),
            sweep_engine: None,
            naked_engines: None,
            faceoff_engines: None,
            faceoff_ratios: None,
            warmup: Some(1),
            small_cell_ns: None,
            small_reps: None,
            verify_naked: Some(true),
            sweep_records: Some(Vec::new()),
            sweep_summary: None,
            interleave_engines: Some(vec!["schoolbook".to_string(), "karatsuba".to_string()]),
            interleave_pairs: Some(Vec::new()),
            interleave_summary: Some(Vec::new()),
            asym_cells: None,
        };
        let value = serde_json::to_value(&archive).expect("serializes");
        for key in [
            "task_id",
            "tier",
            "observations",
            "interleave_engines",
            "interleave_pairs",
            "interleave_summary",
        ] {
            assert!(value.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(
            value.get("tier").and_then(|tier| tier.as_str()),
            Some("interleave")
        );
    }

    #[test]
    fn interleave_warmup_excluded_from_pairs() {
        let cases = build_sweep_cases(&[8], 7);
        let a_limbs = from_bigint(&cases[0].a);
        let b_limbs = from_bigint(&cases[0].b);
        let legs = [Leg::Core(Engine::Schoolbook), Leg::Core(Engine::Karatsuba)];
        let (records, pairs, summary) = run_interleave_cell(
            &cases[0],
            &a_limbs,
            &b_limbs,
            8,
            &legs,
            3,
            2,
            true,
            u128::MAX,
            0,
            None,
        );
        assert_eq!(pairs.len(), 3);
        assert_eq!(summary.reps, 3);
        assert_eq!(summary.warmup_discarded, 4);
        assert_eq!(records.len(), 10);
        assert!(records.iter().filter(|record| record.warmup).count() == 4);
        for (index, pair) in pairs.iter().enumerate() {
            assert_eq!(pair.rep, index);
        }
    }
}
