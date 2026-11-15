use crate::cascade::{
    DISPATCH_FRACTION, ReclaimStep, TrancheState, cap_slice, margin_clears_variance, plan_reclaim,
    project_cost_ms, project_cost_raw_ns, reclaim_note, should_switch, tranche_first_box,
};
use crate::dispatch::{Engine, MulSignals};
use crate::error::MultiplyError;
use crate::limb;
use crate::options::MultiplyOptions;
use crate::report::{MultiplyReport, ProgressEvent, TimingCollector};
use crate::verify::verify_mod;
use num_bigint::BigUint;
use num_traits::Zero;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub const PILOT_LIMBS: usize = 64;
pub const TIE_BAND: f64 = 0.01;

pub(crate) fn run_engine(engine: Engine, a: &[u64], b: &[u64]) -> Vec<u64> {
    match engine {
        Engine::Schoolbook => crate::schoolbook::mul(a, b),
        Engine::Karatsuba => crate::karatsuba::mul(a, b),
        Engine::Toom3 => crate::toom::mul(a, b),
        Engine::Toom4 => crate::toom::mul_toom4(a, b),
        Engine::Ssa => crate::schonhage::mul(a, b),
        Engine::Furer => crate::furer::mul(a, b),
        Engine::Hvh => crate::hvh::mul(a, b),
        Engine::HvhFull => crate::hvh_full::mul(a, b),
    }
}

pub(crate) fn run_engine_cancelled(
    engine: Engine,
    a: &[u64],
    b: &[u64],
    cancelled: &AtomicBool,
) -> Result<Vec<u64>, MultiplyError> {
    match engine {
        Engine::Schoolbook => crate::schoolbook::mul_cancelled(a, b, cancelled),
        Engine::Karatsuba => crate::karatsuba::mul_cancelled(a, b, cancelled),
        Engine::Toom3 => crate::toom::mul_cancelled(a, b, cancelled),
        Engine::Toom4 => crate::toom::mul_toom4_cancelled(a, b, cancelled),
        Engine::Ssa => crate::schonhage::mul_cancelled(a, b, cancelled),
        Engine::Furer => Ok(crate::furer::mul(a, b)),
        Engine::Hvh => Ok(crate::hvh::mul(a, b)),
        Engine::HvhFull => Ok(crate::hvh_full::mul(a, b)),
    }
}

pub fn multiply_naked(engine: Engine, a_limbs: &[u64], b_limbs: &[u64]) -> Vec<u64> {
    run_engine(engine, a_limbs, b_limbs)
}

pub(crate) fn pick_winner(costs: &[(Engine, u128)]) -> Engine {
    let mut best = Engine::Hvh;
    let mut best_cost = u128::MAX;
    for (engine, cost) in costs {
        if *cost < best_cost {
            best = *engine;
            best_cost = *cost;
        }
    }
    if best != Engine::Hvh {
        let hvh_cost = costs
            .iter()
            .find(|(engine, _)| *engine == Engine::Hvh)
            .map(|(_, cost)| *cost);
        if let Some(hvh_cost) = hvh_cost {
            let band = best_cost as f64 * TIE_BAND;
            if (hvh_cost as f64 - best_cost as f64).abs() <= band {
                return Engine::Hvh;
            }
        }
    }
    best
}

fn check_running(deadline: Option<Instant>, cancelled: &AtomicBool) -> Result<(), MultiplyError> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(MultiplyError::Cancelled(
            "the operation was cancelled".to_string(),
        ));
    }
    if let Some(limit) = deadline
        && Instant::now() >= limit
    {
        return Err(MultiplyError::ResourceLimit(
            "deadline exceeded".to_string(),
        ));
    }
    Ok(())
}

fn pilot_limbs(limbs: &[u64]) -> &[u64] {
    &limbs[..limbs.len().min(PILOT_LIMBS)]
}

#[allow(clippy::too_many_arguments)]
fn finish_run(
    engine: Engine,
    a: &[u64],
    b: &[u64],
    out: Vec<u64>,
    elapsed_ns: u128,
    winner: bool,
    tranche: u64,
    signals: &mut MulSignals,
    timings: Option<&TimingCollector>,
    on_event: &mut dyn FnMut(&ProgressEvent) -> Result<(), MultiplyError>,
    current: usize,
    total: usize,
) -> Result<Vec<u64>, MultiplyError> {
    let elapsed_ms = elapsed_ns.div_ceil(1_000_000);
    signals.report_engine(engine.name(), a.len().max(b.len()), elapsed_ns);
    crate::rate_cache::observe(engine.name(), a.len().max(b.len()), elapsed_ns);
    let stage = if winner { engine.name() } else { "pilot" };
    if let Some(collector) = timings {
        collector.record_elapsed(
            stage,
            None,
            elapsed_ms,
            format!(
                "engine={} limbs={} winner={winner} tranche={tranche}",
                engine.name(),
                a.len().max(b.len())
            ),
        );
    }
    on_event(&ProgressEvent {
        stage: engine.name().to_string(),
        current,
        total,
        message: format!("{} finished {} limbs", engine.name(), a.len().max(b.len())),
    })?;
    let verify_start = Instant::now();
    let verified = verify_mod(a, b, &out);
    let verify_ns = verify_start.elapsed().as_nanos();
    if let Some(collector) = timings {
        collector.record_elapsed(
            "verify",
            None,
            verify_ns.div_ceil(1_000_000),
            format!(
                "engine={} limbs={} winner={winner} tranche={tranche} verify_ns={verify_ns}",
                engine.name(),
                a.len().max(b.len())
            ),
        );
    }
    if !verified {
        return Err(MultiplyError::Internal(format!(
            "{} verification failed",
            engine.name()
        )));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn run_tranche(
    engine: Engine,
    a: &[u64],
    b: &[u64],
    box_ms: Option<u128>,
    tranche: u64,
    winner: bool,
    signals: &mut MulSignals,
    timings: Option<&TimingCollector>,
    on_event: &mut dyn FnMut(&ProgressEvent) -> Result<(), MultiplyError>,
    current: usize,
    total: usize,
    cancelled: &AtomicBool,
) -> Result<Vec<u64>, MultiplyError> {
    match box_ms {
        None => {
            let start = Instant::now();
            let out = run_engine_cancelled(engine, a, b, cancelled)?;
            finish_run(
                engine,
                a,
                b,
                out,
                start.elapsed().as_nanos(),
                winner,
                tranche,
                signals,
                timings,
                on_event,
                current,
                total,
            )
        }
        Some(box_ms) => {
            let start = Instant::now();
            let work_a = a.to_vec();
            let work_b = b.to_vec();
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(run_engine(engine, &work_a, &work_b));
            });
            let wait = Duration::from_millis(box_ms.min(u64::MAX as u128) as u64);
            match rx.recv_timeout(wait) {
                Ok(out) => finish_run(
                    engine,
                    a,
                    b,
                    out,
                    start.elapsed().as_nanos(),
                    winner,
                    tranche,
                    signals,
                    timings,
                    on_event,
                    current,
                    total,
                ),
                Err(mpsc::RecvTimeoutError::Timeout) => Err(MultiplyError::ResourceLimit(
                    "tranche box exceeded".to_string(),
                )),
                Err(mpsc::RecvTimeoutError::Disconnected) => Err(MultiplyError::Internal(format!(
                    "{} worker failed",
                    engine.name()
                ))),
            }
        }
    }
}

pub(crate) fn needs_second_tranche(
    first_result: &Result<Vec<u64>, MultiplyError>,
    remaining_ms: Option<u128>,
) -> bool {
    if !matches!(first_result, Err(MultiplyError::ResourceLimit(_))) {
        return false;
    }
    match remaining_ms {
        Some(remaining) => remaining > 0,
        None => true,
    }
}

fn try_race_skip(full_limbs: usize, signals: &mut MulSignals) -> Option<(Engine, f64, f64)> {
    try_race_skip_tier(full_limbs, signals, false)
}

fn try_race_skip_full(full_limbs: usize, signals: &mut MulSignals) -> Option<(Engine, f64, f64)> {
    try_race_skip_tier(full_limbs, signals, true)
}

fn try_race_skip_tier(
    full_limbs: usize,
    signals: &mut MulSignals,
    require_all: bool,
) -> Option<(Engine, f64, f64)> {
    let mut raws: Vec<(Engine, f64)> = Vec::new();
    let mut max_cv = 0.0f64;
    let mut cached = 0usize;
    for engine in Engine::all() {
        if crate::rate_cache::sample_count(engine.name()) < 3 {
            continue;
        }
        let Some(rate) = crate::rate_cache::get(engine.name()) else {
            continue;
        };
        let Some(spread) = crate::rate_cache::cv(engine.name()) else {
            continue;
        };
        let Some(raw) = project_cost_raw_ns(engine, full_limbs, rate) else {
            continue;
        };
        signals.measured_ns_per_limb.insert(engine.name(), rate);
        if spread > max_cv {
            max_cv = spread;
        }
        cached += 1;
        raws.push((engine, raw));
    }
    if require_all && cached != Engine::all().len() {
        return None;
    }
    if raws.len() < 2 {
        return None;
    }
    raws.sort_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let best = raws[0].1;
    let second = raws[1].1;
    if !best.is_finite() || !second.is_finite() || best <= 0.0 {
        return None;
    }
    let margin = (second - best) / best;
    if margin_clears_variance(margin, max_cv) {
        Some((raws[0].0, margin, max_cv))
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn pilot_race(
    a_limbs: &[u64],
    b_limbs: &[u64],
    signals: &mut MulSignals,
    deadline: Option<Instant>,
    cancelled: &AtomicBool,
    timings: Option<&TimingCollector>,
    on_event: &mut dyn FnMut(&ProgressEvent) -> Result<(), MultiplyError>,
    budget_bytes: u64,
) -> Result<(Engine, Vec<u64>), MultiplyError> {
    let total = Engine::all().len() + 1;
    let mut current = 0;
    let pilot_a = pilot_limbs(a_limbs);
    let pilot_b = pilot_limbs(b_limbs);
    let pilot_budget_ms =
        (signals.effective_remaining_ms(deadline) as f64 * DISPATCH_FRACTION) as u128;
    let pilot_start = Instant::now();
    let full_gated = try_race_skip_full(signals.limbs_max, signals);
    let partial_gated = if full_gated.is_none() {
        try_race_skip(signals.limbs_max, signals)
    } else {
        None
    };
    let winner: Engine = if let Some((gated, margin, spread)) = full_gated {
        if let Some(collector) = timings {
            collector.record_elapsed(
                "pilot",
                None,
                0,
                format!(
                    "race-skipped-full winner={} margin={margin:.3} cv={spread:.3} limbs={} tranche=1",
                    gated.name(),
                    signals.limbs_max
                ),
            );
        }
        signals.report_pilot(
            pilot_a.len() + pilot_b.len(),
            pilot_start.elapsed().as_millis(),
        );
        gated
    } else if let Some((gated, margin, spread)) = partial_gated {
        if let Some(collector) = timings {
            collector.record_elapsed(
                "pilot",
                None,
                0,
                format!(
                    "race-skipped winner={} margin={margin:.3} cv={spread:.3} limbs={} tranche=1",
                    gated.name(),
                    signals.limbs_max
                ),
            );
        }
        signals.report_pilot(
            pilot_a.len() + pilot_b.len(),
            pilot_start.elapsed().as_millis(),
        );
        gated
    } else if let Some(table_engine) = crate::routing_table::route_for_limbs(signals.limbs_max) {
        if let Some(collector) = timings {
            let band = crate::routing_table::band_for_limbs(signals.limbs_max)
                .map(|(min, max)| {
                    if max == u64::MAX {
                        format!("{min}-inf")
                    } else {
                        format!("{min}-{max}")
                    }
                })
                .unwrap_or_else(|| "unknown".to_string());
            collector.record_elapsed(
                "pilot",
                None,
                0,
                format!(
                    "static-route engine={} band={band} limbs={} tranche=1",
                    table_engine.name(),
                    signals.limbs_max
                ),
            );
        }
        signals.report_pilot(
            pilot_a.len() + pilot_b.len(),
            pilot_start.elapsed().as_millis(),
        );
        table_engine
    } else {
        let mut piloted = 0u64;
        let mut skipped = 0u64;
        for engine in Engine::all() {
            check_running(deadline, cancelled)?;
            if pilot_start.elapsed().as_millis() > pilot_budget_ms {
                break;
            }
            current += 1;
            if crate::rate_cache::sample_count(engine.name()) >= crate::rate_cache::CACHED_SAMPLES
                && let Some(rate) = crate::rate_cache::get(engine.name())
            {
                signals.measured_ns_per_limb.insert(engine.name(), rate);
                skipped += 1;
                on_event(&ProgressEvent {
                    stage: engine.name().to_string(),
                    current,
                    total,
                    message: format!(
                        "{} cached {} limbs",
                        engine.name(),
                        pilot_a.len().max(pilot_b.len())
                    ),
                })?;
                continue;
            }
            piloted += 1;
            run_tranche(
                engine, pilot_a, pilot_b, None, 1, false, signals, timings, on_event, current,
                total, cancelled,
            )?;
        }
        if piloted == 0 && skipped > 0 {
            if let Some(collector) = timings {
                collector.record_elapsed(
                    "pilot",
                    None,
                    0,
                    format!(
                        "all-cached skipped={skipped} limbs={} tranche=1",
                        signals.limbs_max
                    ),
                );
            }
        }
        signals.report_pilot(
            pilot_a.len() + pilot_b.len(),
            pilot_start.elapsed().as_millis(),
        );
        let full = signals.limbs_max;
        let mut costs = Vec::new();
        for engine in Engine::all() {
            if let Some(cost) = project_cost_ms(engine, full, signals) {
                costs.push((engine, cost));
            }
        }
        pick_winner(&costs)
    };
    let plan = plan_reclaim(signals.limbs_max, winner, signals.workers, budget_bytes);
    let winner = plan.engine;
    signals.workers = plan.workers;
    if plan.step != ReclaimStep::None {
        if let Some(collector) = timings {
            collector.record_elapsed("reclaim", None, 0, reclaim_note(&plan));
        }
    }
    let mut tranche = TrancheState::default();
    let live_ms = signals.effective_remaining_ms(deadline);
    let first_box = tranche_first_box(Some(live_ms), signals.remaining_ms(deadline));
    check_running(deadline, cancelled)?;
    current += 1;
    let tranche_start = Instant::now();
    match run_tranche(
        winner, a_limbs, b_limbs, first_box, 1, true, signals, timings, on_event, current, total,
        cancelled,
    ) {
        Ok(out) => {
            tranche.record_attempt(tranche_start.elapsed().as_millis(), true);
            tranche.winner = Some(winner.name().to_string());
            check_running(deadline, cancelled)?;
            Ok((winner, out))
        }
        Err(error) => {
            let spent = tranche_start.elapsed().as_millis();
            tranche.record_attempt(spent, false);
            let remaining = signals.remaining_ms(deadline);
            let failed: Result<Vec<u64>, MultiplyError> = Err(error);
            if !needs_second_tranche(&failed, remaining) {
                return failed.map(|out| (winner, out));
            }
            if let Some(collector) = timings {
                collector.record_elapsed(
                    winner.name(),
                    None,
                    spent,
                    format!(
                        "limbs={} winner=false tranche=1 miss=true",
                        signals.limbs_max
                    ),
                );
            }
            signals.report_engine(
                winner.name(),
                signals.limbs_max,
                first_box.unwrap_or(0).saturating_mul(1_000_000),
            );
            let mut fresh = Vec::new();
            for engine in Engine::all() {
                if let Some(cost) = project_cost_ms(engine, signals.limbs_max, signals) {
                    fresh.push((engine, cost));
                }
            }
            let fresh_names: Vec<(&str, u128)> = fresh
                .iter()
                .map(|(engine, cost)| (engine.name(), *cost))
                .collect();
            let second = should_switch(winner.name(), &fresh_names)
                .and_then(Engine::from_str)
                .unwrap_or_else(|| pick_winner(&fresh));
            let planned = fresh
                .iter()
                .find(|(engine, _)| *engine == second)
                .map(|(_, cost)| *cost);
            let box_two = cap_slice(signals.remaining_ms(deadline), planned, tranche.spent_ms);
            current += 1;
            let total_two = total + tranche.attempts as usize;
            let second_start = Instant::now();
            let out = run_tranche(
                second, a_limbs, b_limbs, box_two, 2, true, signals, timings, on_event, current,
                total_two, cancelled,
            )?;
            tranche.record_attempt(second_start.elapsed().as_millis(), true);
            tranche.winner = Some(second.name().to_string());
            check_running(deadline, cancelled)?;
            Ok((second, out))
        }
    }
}

pub fn multiply_with_progress(
    a: &BigUint,
    b: &BigUint,
    options: &MultiplyOptions,
    cancelled: &AtomicBool,
    on_event: &mut dyn FnMut(&ProgressEvent) -> Result<(), MultiplyError>,
    timings: Option<&TimingCollector>,
) -> Result<MultiplyReport, MultiplyError> {
    options.validate()?;
    if options.memory_budget_bytes() == 0 {
        return Err(MultiplyError::ResourceLimit(
            "memory budget is zero".to_string(),
        ));
    }
    let started = Instant::now();
    let deadline = options.effective_deadline(started);
    check_running(deadline, cancelled)?;
    let a_limbs = limb::from_bigint(a);
    let b_limbs = limb::from_bigint(b);
    if limb::is_zero(&a_limbs) || limb::is_zero(&b_limbs) {
        return Ok(MultiplyReport {
            product: BigUint::zero().to_string(),
            method: "trivial".to_string(),
            limbs: 1,
            elapsed_ms: started.elapsed().as_millis(),
        });
    }
    if limb::is_one(&a_limbs) {
        return Ok(MultiplyReport {
            product: b.to_string(),
            method: "trivial".to_string(),
            limbs: b_limbs.len(),
            elapsed_ms: started.elapsed().as_millis(),
        });
    }
    if limb::is_one(&b_limbs) {
        return Ok(MultiplyReport {
            product: a.to_string(),
            method: "trivial".to_string(),
            limbs: a_limbs.len(),
            elapsed_ms: started.elapsed().as_millis(),
        });
    }
    let mut signals = MulSignals::collect_with_options(
        a_limbs.len(),
        b_limbs.len(),
        limb::limb_bits(&a_limbs).max(limb::limb_bits(&b_limbs)),
        options,
        deadline,
    );
    if let Some(name) = &options.algorithm {
        if name != "auto" {
            let engine = Engine::from_str(name)
                .ok_or_else(|| MultiplyError::Input(format!("unsupported algorithm {name}")))?;
            check_running(deadline, cancelled)?;
            let out = run_tranche(
                engine,
                &a_limbs,
                &b_limbs,
                None,
                0,
                true,
                &mut signals,
                timings,
                on_event,
                1,
                1,
                cancelled,
            )?;
            check_running(deadline, cancelled)?;
            return Ok(MultiplyReport {
                product: limb::to_bigint(&out).to_string(),
                method: engine.name().to_string(),
                limbs: signals.limbs_max,
                elapsed_ms: started.elapsed().as_millis(),
            });
        }
    }
    let (winner, out) = pilot_race(
        &a_limbs,
        &b_limbs,
        &mut signals,
        deadline,
        cancelled,
        timings,
        on_event,
        options.memory_budget_bytes(),
    )?;
    Ok(MultiplyReport {
        product: limb::to_bigint(&out).to_string(),
        method: winner.name().to_string(),
        limbs: signals.limbs_max,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

pub fn multiply_cancelled(
    a: &BigUint,
    b: &BigUint,
    options: &MultiplyOptions,
    cancelled: &AtomicBool,
) -> Result<MultiplyReport, MultiplyError> {
    let mut sink = |_: &ProgressEvent| Ok(());
    multiply_with_progress(a, b, options, cancelled, &mut sink, None)
}

pub fn multiply(
    a: &BigUint,
    b: &BigUint,
    options: &MultiplyOptions,
) -> Result<MultiplyReport, MultiplyError> {
    let flag = AtomicBool::new(false);
    multiply_cancelled(a, b, options, &flag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate_cache::{lock_for_tests, reset_for_tests};
    use crate::report::TimingCollector;
    use crate::schoolbook;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn fixture(rng: &mut ChaCha8Rng, limbs: usize) -> (BigUint, BigUint) {
        let a: Vec<u64> = (0..limbs).map(|_| rng.next_u64()).collect();
        let b: Vec<u64> = (0..limbs).map(|_| rng.next_u64()).collect();
        (limb::to_bigint(&a), limb::to_bigint(&b))
    }

    fn auto_options() -> MultiplyOptions {
        MultiplyOptions::default()
    }

    #[test]
    fn explicit_routing_per_engine() {
        let _guard = lock_for_tests();
        let a = BigUint::from(12345u32);
        let b = BigUint::from(67890u32);
        for name in [
            "schoolbook",
            "karatsuba",
            "toom3",
            "toom4",
            "ssa",
            "furer",
            "hvh",
            "hvh-full",
        ] {
            let options = MultiplyOptions {
                algorithm: Some(name.to_string()),
                ..MultiplyOptions::default()
            };
            let report = multiply(&a, &b, &options).expect("explicit run");
            assert_eq!(report.product, "838102050");
            assert_eq!(report.method, name);
        }
    }

    #[test]
    fn explicit_toom4_routes_and_verifies() {
        let _guard = lock_for_tests();
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c57);
        let (a, b) = fixture(&mut rng, 256);
        let options = MultiplyOptions {
            algorithm: Some("toom4".to_string()),
            ..MultiplyOptions::default()
        };
        let report = multiply(&a, &b, &options).expect("toom4 run");
        assert_eq!(report.method, "toom4");
        let expected = limb::to_bigint(&schoolbook::mul(
            &limb::from_bigint(&a),
            &limb::from_bigint(&b),
        ))
        .to_string();
        assert_eq!(report.product, expected);
    }

    #[test]
    fn products_identical_across_all_engines() {
        let _guard = lock_for_tests();
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c51);
        let (a, b) = fixture(&mut rng, 40);
        let expected = limb::to_bigint(&schoolbook::mul(
            &limb::from_bigint(&a),
            &limb::from_bigint(&b),
        ))
        .to_string();
        for engine in [
            Engine::Schoolbook,
            Engine::Karatsuba,
            Engine::Toom3,
            Engine::Toom4,
            Engine::Ssa,
            Engine::Furer,
            Engine::Hvh,
            Engine::HvhFull,
        ] {
            let options = MultiplyOptions {
                algorithm: Some(engine.name().to_string()),
                ..MultiplyOptions::default()
            };
            let report = multiply(&a, &b, &options).expect("engine run");
            assert_eq!(report.product, expected);
            assert_eq!(report.method, engine.name());
        }
    }

    #[test]
    fn auto_race_three_fixture_sizes() {
        let _guard = lock_for_tests();
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c52);
        for limbs in [4usize, 70, 2100] {
            let (a, b) = fixture(&mut rng, limbs);
            let report = multiply(&a, &b, &auto_options()).expect("auto run");
            let expected = limb::to_bigint(&schoolbook::mul(
                &limb::from_bigint(&a),
                &limb::from_bigint(&b),
            ))
            .to_string();
            assert_eq!(report.product, expected);
            assert!(Engine::from_str(&report.method).is_some());
            assert_eq!(report.limbs, limbs);
        }
    }

    #[test]
    fn auto_timings_record_stages() {
        let _guard = lock_for_tests();
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c53);
        let (a, b) = fixture(&mut rng, 40);
        let collector = TimingCollector::new();
        let flag = AtomicBool::new(false);
        let mut events = Vec::new();
        let mut on_event = |event: &ProgressEvent| {
            events.push(event.stage.clone());
            Ok(())
        };
        let report = multiply_with_progress(
            &a,
            &b,
            &auto_options(),
            &flag,
            &mut on_event,
            Some(&collector),
        )
        .expect("auto with timings");
        let records = collector.snapshot();
        assert!(!records.is_empty());
        assert!(records.iter().any(|entry| entry.stage == report.method));
        assert!(!events.is_empty());
    }

    #[test]
    fn trivial_zero_and_one() {
        let big = BigUint::from(987654321u64);
        let zero = multiply(&BigUint::zero(), &big, &auto_options()).expect("zero");
        assert_eq!(zero.product, "0");
        assert_eq!(zero.method, "trivial");
        let one = multiply(&BigUint::from(1u32), &big, &auto_options()).expect("one");
        assert_eq!(one.product, "987654321");
        assert_eq!(one.method, "trivial");
        let one_right = multiply(&big, &BigUint::from(1u32), &auto_options()).expect("one right");
        assert_eq!(one_right.product, "987654321");
    }

    #[test]
    fn invalid_algorithm_is_input() {
        let options = MultiplyOptions {
            algorithm: Some("gnfs".to_string()),
            ..MultiplyOptions::default()
        };
        assert!(matches!(
            multiply(&BigUint::from(2u32), &BigUint::from(3u32), &options),
            Err(MultiplyError::Input(_))
        ));
    }

    #[test]
    fn zero_timeout_is_resource_limit() {
        let options = MultiplyOptions {
            max_duration: Some(std::time::Duration::from_secs(0)),
            ..MultiplyOptions::default()
        };
        assert!(matches!(
            multiply(&BigUint::from(2u32), &BigUint::from(3u32), &options),
            Err(MultiplyError::ResourceLimit(_))
        ));
    }

    #[test]
    fn precancelled_is_cancelled() {
        let flag = AtomicBool::new(true);
        assert!(matches!(
            multiply_cancelled(
                &BigUint::from(2u32),
                &BigUint::from(3u32),
                &auto_options(),
                &flag
            ),
            Err(MultiplyError::Cancelled(_))
        ));
    }

    #[test]
    fn precancelled_ssa_and_toom4_are_cancelled() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c70);
        let (a, b) = fixture(&mut rng, 2048);
        for name in ["ssa", "toom4"] {
            let flag = AtomicBool::new(true);
            let options = MultiplyOptions {
                algorithm: Some(name.to_string()),
                ..MultiplyOptions::default()
            };
            assert!(matches!(
                multiply_cancelled(&a, &b, &options, &flag),
                Err(MultiplyError::Cancelled(_))
            ));
        }
    }

    #[test]
    fn mid_run_cancel_aborts_ssa_32768() {
        let _guard = lock_for_tests();
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c71);
        let (a, b) = fixture(&mut rng, 32768);
        let options = MultiplyOptions {
            algorithm: Some("ssa".to_string()),
            ..MultiplyOptions::default()
        };
        let flag = AtomicBool::new(false);
        let (result, latency_ms) = std::thread::scope(|s| {
            let worker = s.spawn(|| multiply_cancelled(&a, &b, &options, &flag));
            std::thread::sleep(std::time::Duration::from_millis(500));
            let mark = std::time::Instant::now();
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
            let result = worker.join().unwrap();
            (result, mark.elapsed().as_millis())
        });
        eprintln!("mid_run_cancel_latency_ms={latency_ms}");
        assert!(matches!(result, Err(MultiplyError::Cancelled(_))));
    }

    #[test]
    fn callback_error_propagates() {
        let _guard = lock_for_tests();
        let flag = AtomicBool::new(false);
        let mut on_event = |_: &ProgressEvent| Err(MultiplyError::Cancelled("stop".to_string()));
        assert!(matches!(
            multiply_with_progress(
                &BigUint::from(12u32),
                &BigUint::from(34u32),
                &auto_options(),
                &flag,
                &mut on_event,
                None
            ),
            Err(MultiplyError::Cancelled(_))
        ));
    }

    #[test]
    fn tie_break_prefers_hvh() {
        assert_eq!(
            pick_winner(&[(Engine::Schoolbook, 100), (Engine::Hvh, 100)]),
            Engine::Hvh
        );
        assert_eq!(
            pick_winner(&[(Engine::Schoolbook, 100), (Engine::Hvh, 101)]),
            Engine::Hvh
        );
        assert_eq!(
            pick_winner(&[(Engine::Schoolbook, 100), (Engine::Hvh, 102)]),
            Engine::Schoolbook
        );
        assert_eq!(
            pick_winner(&[(Engine::Hvh, 50), (Engine::Schoolbook, 100)]),
            Engine::Hvh
        );
        assert_eq!(pick_winner(&[]), Engine::Hvh);
    }

    #[test]
    fn second_tranche_only_on_resourcelimit_with_budget() {
        let ok: Result<Vec<u64>, MultiplyError> = Ok(vec![1]);
        assert!(!needs_second_tranche(&ok, Some(100)));
        let internal: Result<Vec<u64>, MultiplyError> =
            Err(MultiplyError::Internal("bad".to_string()));
        assert!(!needs_second_tranche(&internal, Some(100)));
        let cancelled: Result<Vec<u64>, MultiplyError> =
            Err(MultiplyError::Cancelled("stop".to_string()));
        assert!(!needs_second_tranche(&cancelled, Some(100)));
        let miss: Result<Vec<u64>, MultiplyError> =
            Err(MultiplyError::ResourceLimit("box".to_string()));
        assert!(needs_second_tranche(&miss, Some(100)));
        assert!(!needs_second_tranche(&miss, Some(0)));
        assert!(needs_second_tranche(&miss, None));
    }

    #[test]
    fn auto_stages_stay_in_vocabulary() {
        let _guard = lock_for_tests();
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
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c56);
        let (a, b) = fixture(&mut rng, 40);
        let collector = TimingCollector::new();
        let flag = AtomicBool::new(false);
        let mut sink = |_: &ProgressEvent| Ok(());
        let report =
            multiply_with_progress(&a, &b, &auto_options(), &flag, &mut sink, Some(&collector))
                .expect("auto run");
        let records = collector.snapshot();
        assert!(!records.is_empty());
        for record in &records {
            assert!(
                STAGES.contains(&record.stage.as_str()),
                "stage {}",
                record.stage
            );
        }
        assert!(records.iter().any(|record| record.stage == "pilot"));
        assert!(records.iter().any(|record| record.stage == report.method));
    }

    #[test]
    fn auto_timings_carry_tranche_details() {
        let _guard = lock_for_tests();
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c54);
        let (a, b) = fixture(&mut rng, 40);
        let collector = TimingCollector::new();
        let flag = AtomicBool::new(false);
        let mut sink = |_: &ProgressEvent| Ok(());
        multiply_with_progress(&a, &b, &auto_options(), &flag, &mut sink, Some(&collector))
            .expect("auto run");
        let records = collector.snapshot();
        assert!(!records.is_empty());
        for record in &records {
            assert!(record.detail.contains("tranche="), "{}", record.detail);
        }
        assert!(
            records
                .iter()
                .any(|record| record.detail.contains("tranche=1"))
        );
    }

    #[test]
    fn zero_memory_budget_is_resource_limit() {
        let options = MultiplyOptions {
            max_memory_mb: Some(0),
            ..MultiplyOptions::default()
        };
        assert!(matches!(
            multiply(&BigUint::from(12u32), &BigUint::from(34u32), &options),
            Err(MultiplyError::ResourceLimit(_))
        ));
    }

    #[test]
    fn over_budget_demotes_without_misfire() {
        let _guard = lock_for_tests();
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c55);
        let (a, b) = fixture(&mut rng, 9000);
        let options = MultiplyOptions {
            max_memory_mb: Some(1),
            threads: Some(2),
            ..MultiplyOptions::default()
        };
        let report = multiply(&a, &b, &options).expect("no resource misfire");
        let expected = limb::to_bigint(&schoolbook::mul(
            &limb::from_bigint(&a),
            &limb::from_bigint(&b),
        ))
        .to_string();
        assert_eq!(report.product, expected);
        assert!(Engine::from_str(&report.method).is_some());
    }

    fn pilot_runs(records: &[crate::report::StageTiming]) -> usize {
        records
            .iter()
            .filter(|record| {
                record.stage == "pilot"
                    && !record.detail.contains("all-cached")
                    && !record.detail.contains("race-skipped")
                    && !record.detail.contains("static-route")
            })
            .count()
    }

    #[test]
    fn rate_cache_first_call_runs_all_pilots() {
        let _guard = lock_for_tests();
        reset_for_tests();
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c58);
        let (a, b) = fixture(&mut rng, 128);
        let expected = limb::to_bigint(&schoolbook::mul(
            &limb::from_bigint(&a),
            &limb::from_bigint(&b),
        ))
        .to_string();
        let collector = TimingCollector::new();
        let flag = AtomicBool::new(false);
        let mut sink = |_: &ProgressEvent| Ok(());
        let report =
            multiply_with_progress(&a, &b, &auto_options(), &flag, &mut sink, Some(&collector))
                .expect("auto run");
        assert_eq!(report.product, expected);
        assert_eq!(report.method, "schoolbook");
        let records = collector.snapshot();
        assert_eq!(pilot_runs(&records), 0);
        assert!(
            records
                .iter()
                .any(|record| record.detail.contains("static-route"))
        );
    }

    #[test]
    fn rate_cache_seeded_call_runs_zero_pilots() {
        let _guard = lock_for_tests();
        reset_for_tests();
        for engine in Engine::all() {
            crate::rate_cache::observe(engine.name(), 64, 6400);
            crate::rate_cache::observe(engine.name(), 64, 6400);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c59);
        let (a, b) = fixture(&mut rng, 128);
        let expected = limb::to_bigint(&schoolbook::mul(
            &limb::from_bigint(&a),
            &limb::from_bigint(&b),
        ))
        .to_string();
        let collector = TimingCollector::new();
        let flag = AtomicBool::new(false);
        let mut sink = |_: &ProgressEvent| Ok(());
        let report =
            multiply_with_progress(&a, &b, &auto_options(), &flag, &mut sink, Some(&collector))
                .expect("auto run");
        assert_eq!(report.product, expected);
        let records = collector.snapshot();
        assert_eq!(pilot_runs(&records), 0);
        assert!(
            records
                .iter()
                .any(|record| record.detail.contains("static-route"))
        );
    }

    #[test]
    fn rate_cache_third_call_runs_zero_pilots() {
        let _guard = lock_for_tests();
        reset_for_tests();
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c60);
        let (a, b) = fixture(&mut rng, 128);
        let mut counts = Vec::new();
        for _ in 0..3 {
            let collector = TimingCollector::new();
            let flag = AtomicBool::new(false);
            let mut sink = |_: &ProgressEvent| Ok(());
            multiply_with_progress(&a, &b, &auto_options(), &flag, &mut sink, Some(&collector))
                .expect("auto run");
            counts.push(pilot_runs(&collector.snapshot()));
        }
        assert_eq!(counts, vec![0, 0, 0]);
    }

    fn seed_schoolbook_cache() {
        crate::rate_cache::observe("schoolbook", 100, 20_000);
        crate::rate_cache::observe("schoolbook", 100, 20_000);
        crate::rate_cache::observe("schoolbook", 100, 20_000);
        for engine in Engine::all() {
            if engine == Engine::Schoolbook {
                continue;
            }
            crate::rate_cache::observe(engine.name(), 100, 1_900_000);
            crate::rate_cache::observe(engine.name(), 100, 2_000_000);
            crate::rate_cache::observe(engine.name(), 100, 2_100_000);
        }
    }

    fn run_auto(a: &BigUint, b: &BigUint, collector: &TimingCollector) -> MultiplyReport {
        let flag = AtomicBool::new(false);
        let mut sink = |_: &ProgressEvent| Ok(());
        multiply_with_progress(a, b, &auto_options(), &flag, &mut sink, Some(collector))
            .expect("auto run")
    }

    #[test]
    fn race_skip_gate_closed_when_cold() {
        let _guard = lock_for_tests();
        reset_for_tests();
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c61);
        let (a, b) = fixture(&mut rng, 128);
        let collector = TimingCollector::new();
        run_auto(&a, &b, &collector);
        let records = collector.snapshot();
        assert_eq!(pilot_runs(&records), 0);
        assert!(
            records
                .iter()
                .any(|record| record.detail.contains("static-route"))
        );
        assert!(
            !records
                .iter()
                .any(|record| record.detail.contains("race-skipped"))
        );
    }

    #[test]
    fn race_skip_gate_skips_small_inputs() {
        let _guard = lock_for_tests();
        for limbs in [1usize, 2] {
            reset_for_tests();
            seed_schoolbook_cache();
            let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c62 + limbs as u64);
            let (a, b) = fixture(&mut rng, limbs);
            let expected = limb::to_bigint(&schoolbook::mul(
                &limb::from_bigint(&a),
                &limb::from_bigint(&b),
            ))
            .to_string();
            let collector = TimingCollector::new();
            let report = run_auto(&a, &b, &collector);
            assert_eq!(report.product, expected);
            let records = collector.snapshot();
            assert_eq!(pilot_runs(&records), 0);
            assert!(
                records
                    .iter()
                    .any(|record| record.detail.contains("race-skipped"))
            );
        }
    }

    fn seed_full_cache_schoolbook_dominant() {
        for _ in 0..3 {
            crate::rate_cache::observe("schoolbook", 100, 10_000);
        }
        for engine in Engine::all() {
            if engine == Engine::Schoolbook {
                continue;
            }
            crate::rate_cache::observe(engine.name(), 100, 1_900_000);
            crate::rate_cache::observe(engine.name(), 100, 2_000_000);
            crate::rate_cache::observe(engine.name(), 100, 2_100_000);
        }
    }

    #[test]
    fn race_skip_full_gate_skips_large_inputs() {
        let _guard = lock_for_tests();
        reset_for_tests();
        seed_full_cache_schoolbook_dominant();
        for engine in Engine::all() {
            assert!(crate::rate_cache::sample_count(engine.name()) >= 3);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c67);
        let (a, b) = fixture(&mut rng, 1024);
        let expected = limb::to_bigint(&schoolbook::mul(
            &limb::from_bigint(&a),
            &limb::from_bigint(&b),
        ))
        .to_string();
        let collector = TimingCollector::new();
        let report = run_auto(&a, &b, &collector);
        assert_eq!(report.product, expected);
        assert_eq!(report.method, "schoolbook");
        let records = collector.snapshot();
        assert_eq!(pilot_runs(&records), 0);
        assert!(
            records
                .iter()
                .any(|record| record.detail.contains("race-skipped-full"))
        );
        assert!(
            !records
                .iter()
                .any(|record| record.detail.contains("all-cached"))
        );
    }

    fn verify_ns_sum(records: &[crate::report::StageTiming]) -> u128 {
        let mut total = 0u128;
        for record in records {
            if record.stage == "verify" {
                let marker = "verify_ns=";
                if let Some(start) = record.detail.find(marker) {
                    let rest = &record.detail[start + marker.len()..];
                    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                    total += digits.parse::<u128>().unwrap_or(0);
                }
            }
        }
        total
    }

    fn forced_schoolbook(a: &BigUint, b: &BigUint) -> (Vec<crate::report::StageTiming>, u128) {
        let options = MultiplyOptions {
            algorithm: Some("schoolbook".to_string()),
            ..MultiplyOptions::default()
        };
        let collector = TimingCollector::new();
        let flag = AtomicBool::new(false);
        let mut sink = |_: &ProgressEvent| Ok(());
        multiply_with_progress(a, b, &options, &flag, &mut sink, Some(&collector))
            .expect("forced run");
        let records = collector.snapshot();
        let verify = verify_ns_sum(&records);
        (records, verify)
    }

    fn kernel_median_ns(a_limbs: &[u64], b_limbs: &[u64]) -> f64 {
        let mut samples = Vec::with_capacity(3);
        for _ in 0..3 {
            let start = Instant::now();
            let _ = crate::schoolbook::mul(a_limbs, b_limbs);
            samples.push(start.elapsed().as_nanos() as f64);
        }
        samples.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        samples[1]
    }

    #[test]
    fn prod_verify_stage_present_on_default_call() {
        let _guard = lock_for_tests();
        reset_for_tests();
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c65);
        let (a, b) = fixture(&mut rng, 40);
        let collector = TimingCollector::new();
        run_auto(&a, &b, &collector);
        let records = collector.snapshot();
        assert!(records.iter().any(|record| record.stage == "verify"));
        assert!(
            records
                .iter()
                .filter(|record| record.stage == "verify")
                .all(|record| record.detail.contains("verify_ns=")
                    && record.detail.contains("tranche="))
        );
        assert!(verify_ns_sum(&records) > 0);
    }

    #[test]
    fn verify_share_falls_with_size() {
        let _guard = lock_for_tests();
        reset_for_tests();
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c66);
        let mut shares = Vec::new();
        let mut absolutes = Vec::new();
        for limbs in [32usize, 512] {
            let (a, b) = fixture(&mut rng, limbs);
            let a_limbs = limb::from_bigint(&a);
            let b_limbs = limb::from_bigint(&b);
            let (_, verify_ns) = forced_schoolbook(&a, &b);
            assert!(verify_ns > 0);
            absolutes.push(verify_ns as f64);
            let kernel = kernel_median_ns(&a_limbs, &b_limbs);
            assert!(kernel > 0.0);
            shares.push(verify_ns as f64 / (verify_ns as f64 + kernel));
        }
        assert!(absolutes[1] > absolutes[0]);
        assert!(shares[1] < shares[0], "shares {shares:?}");
    }

    fn seed_full_cache_hvh_dominant() {
        for _ in 0..3 {
            crate::rate_cache::observe("hvh", 100, 10_000);
        }
        for engine in Engine::all() {
            if engine == Engine::Hvh {
                continue;
            }
            crate::rate_cache::observe(engine.name(), 100, 2_000_000);
            crate::rate_cache::observe(engine.name(), 100, 2_000_000);
            crate::rate_cache::observe(engine.name(), 100, 2_000_000);
        }
    }

    #[test]
    fn warmed_override_beats_static_table() {
        let _guard = lock_for_tests();
        reset_for_tests();
        seed_full_cache_hvh_dominant();
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c68);
        let (a, b) = fixture(&mut rng, 128);
        let expected = limb::to_bigint(&schoolbook::mul(
            &limb::from_bigint(&a),
            &limb::from_bigint(&b),
        ))
        .to_string();
        let collector = TimingCollector::new();
        let report = run_auto(&a, &b, &collector);
        assert_eq!(report.product, expected);
        assert_eq!(report.method, "hvh");
        let records = collector.snapshot();
        assert_eq!(pilot_runs(&records), 0);
        assert!(
            records
                .iter()
                .any(|record| record.detail.contains("race-skipped-full"))
        );
        assert!(
            !records
                .iter()
                .any(|record| record.detail.contains("static-route"))
        );
    }

    #[test]
    fn cold_auto_winners_match_pin_cells() {
        use std::str::FromStr;
        let text = include_str!("../../../bench/pins.json");
        let parsed: serde_json::Value = serde_json::from_str(text).expect("pins.json parses");
        let cells = parsed
            .get("cells")
            .and_then(|value| value.as_array())
            .expect("cells");
        assert_eq!(cells.len(), 5);
        for cell in cells {
            let _guard = lock_for_tests();
            reset_for_tests();
            let a = BigUint::from_str(
                cell.get("input_a")
                    .and_then(|value| value.as_str())
                    .expect("input_a"),
            )
            .expect("decimal a");
            let b = BigUint::from_str(
                cell.get("input_b")
                    .and_then(|value| value.as_str())
                    .expect("input_b"),
            )
            .expect("decimal b");
            let expected_limbs = cell
                .get("expected")
                .and_then(|value| value.get("limbs"))
                .and_then(|value| value.as_u64())
                .expect("limbs") as usize;
            let collector = TimingCollector::new();
            let report = run_auto(&a, &b, &collector);
            assert_eq!(report.product, (a * b).to_string());
            assert_eq!(report.method, "schoolbook");
            assert_eq!(report.limbs, expected_limbs);
        }
    }

    #[test]
    fn naked_kernel_matches_engine_mul() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c63);
        let (a, b) = fixture(&mut rng, 24);
        let a_limbs = limb::from_bigint(&a);
        let b_limbs = limb::from_bigint(&b);
        for engine in Engine::all() {
            let naked = multiply_naked(engine, &a_limbs, &b_limbs);
            let direct = match engine {
                Engine::Schoolbook => crate::schoolbook::mul(&a_limbs, &b_limbs),
                Engine::Karatsuba => crate::karatsuba::mul(&a_limbs, &b_limbs),
                Engine::Toom3 => crate::toom::mul(&a_limbs, &b_limbs),
                Engine::Toom4 => crate::toom::mul_toom4(&a_limbs, &b_limbs),
                Engine::Ssa => crate::schonhage::mul(&a_limbs, &b_limbs),
                Engine::Furer => crate::furer::mul(&a_limbs, &b_limbs),
                Engine::Hvh => crate::hvh::mul(&a_limbs, &b_limbs),
                Engine::HvhFull => crate::hvh_full::mul(&a_limbs, &b_limbs),
            };
            assert_eq!(naked, direct);
        }
    }

    #[test]
    fn naked_path_leaves_production_verify_intact() {
        let _guard = lock_for_tests();
        reset_for_tests();
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c64);
        let (a, b) = fixture(&mut rng, 24);
        let a_limbs = limb::from_bigint(&a);
        let b_limbs = limb::from_bigint(&b);
        let out = multiply_naked(Engine::Toom3, &a_limbs, &b_limbs);
        assert!(crate::verify::verify_mod(&a_limbs, &b_limbs, &out));
        let report = multiply(&a, &b, &auto_options()).expect("production run");
        assert!(Engine::from_str(&report.method).is_some());
        let expected = limb::to_bigint(&crate::schoolbook::mul(&a_limbs, &b_limbs)).to_string();
        assert_eq!(report.product, expected);
    }
}
