use crate::dispatch::{Engine, MulSignals};

pub const DISPATCH_FRACTION: f64 = 0.2;
pub const SIGMA_MULT: f64 = 3.0;

fn predicted_work(engine: Engine, limbs: usize) -> f64 {
    match engine {
        Engine::Schoolbook => crate::schoolbook::predicted_work_limbs(limbs),
        Engine::Karatsuba => crate::karatsuba::predicted_work_limbs(limbs),
        Engine::Toom3 => crate::toom::predicted_work_limbs(limbs),
        Engine::Toom4 => crate::toom::predicted_work_limbs_toom4(limbs),
        Engine::Ssa => crate::schonhage::predicted_work_limbs(limbs),
        Engine::Furer => crate::furer::predicted_work_limbs(limbs),
        Engine::Hvh => crate::hvh::predicted_work_limbs(limbs),
        Engine::HvhFull => crate::hvh_full::predicted_work_limbs(limbs),
    }
}

pub fn project_cost_ms(engine: Engine, limbs: usize, signals: &MulSignals) -> Option<u128> {
    let rate = *signals.measured_ns_per_limb.get(engine.name())?;
    let work = predicted_work(engine, limbs);
    if !rate.is_finite() || rate <= 0.0 || !work.is_finite() || work <= 0.0 {
        return None;
    }
    let cost_ms = rate * work / 1_000_000.0;
    if cost_ms.is_finite() {
        Some(cost_ms.ceil() as u128)
    } else {
        None
    }
}

pub fn project_cost_raw_ns(engine: Engine, limbs: usize, rate_ns_per_limb: f64) -> Option<f64> {
    let work = predicted_work(engine, limbs);
    if !rate_ns_per_limb.is_finite() || rate_ns_per_limb <= 0.0 || !work.is_finite() || work <= 0.0
    {
        return None;
    }
    let cost_ns = rate_ns_per_limb * work;
    if cost_ns.is_finite() && cost_ns > 0.0 {
        Some(cost_ns)
    } else {
        None
    }
}

pub fn projected_margin(costs: &[(String, u128)]) -> Option<f64> {
    if costs.len() < 2 {
        return None;
    }
    let mut ordered: Vec<u128> = costs.iter().map(|(_, cost)| *cost).collect();
    ordered.sort_unstable();
    let best = ordered[0] as f64;
    let second = ordered[1] as f64;
    if !best.is_finite() || !second.is_finite() || best <= 0.0 {
        return None;
    }
    Some((second - best) / best)
}

pub fn projected_margin_f64(costs: &[(String, f64)]) -> Option<f64> {
    if costs.len() < 2 {
        return None;
    }
    let mut ordered: Vec<f64> = costs.iter().map(|(_, cost)| *cost).collect();
    ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let best = ordered[0];
    let second = ordered[1];
    if !best.is_finite() || !second.is_finite() || best <= 0.0 {
        return None;
    }
    let margin = (second - best) / best;
    if margin.is_finite() && margin >= 0.0 {
        Some(margin)
    } else {
        None
    }
}

pub fn margin_clears_variance(margin: f64, cv: f64) -> bool {
    if !margin.is_finite() || !cv.is_finite() {
        return false;
    }
    if margin < 0.0 || cv < 0.0 {
        return false;
    }
    margin > SIGMA_MULT * cv
}

pub fn cost_shares<'a>(costs: &[(&'a str, u128)]) -> Vec<(&'a str, f64)> {
    if costs.is_empty() {
        return Vec::new();
    }
    if costs.iter().any(|(_, cost)| *cost == 0) {
        let zeros = costs.iter().filter(|(_, cost)| *cost == 0).count() as f64;
        return costs
            .iter()
            .map(|(name, cost)| (*name, if *cost == 0 { 1.0 / zeros } else { 0.0 }))
            .collect();
    }
    let total: f64 = costs.iter().map(|(_, cost)| 1.0 / *cost as f64).sum();
    if !total.is_finite() || total <= 0.0 {
        let even = 1.0 / costs.len() as f64;
        return costs.iter().map(|(name, _)| (*name, even)).collect();
    }
    costs
        .iter()
        .map(|(name, cost)| (*name, (1.0 / *cost as f64) / total))
        .collect()
}

pub fn should_switch<'a>(current: &str, candidate_costs: &[(&'a str, u128)]) -> Option<&'a str> {
    let current_cost = candidate_costs.iter().find(|(name, _)| *name == current)?.1;
    let (best_name, best_cost) = candidate_costs.iter().min_by_key(|(_, cost)| *cost)?;
    if *best_name != current && *best_cost < current_cost {
        Some(*best_name)
    } else {
        None
    }
}

pub fn tranche_first_box(budget_ms: Option<u128>, live_remaining_ms: Option<u128>) -> Option<u128> {
    let box_ms = budget_ms?;
    let half = box_ms / 2;
    match live_remaining_ms {
        Some(live) => Some(half.min(live / 2)),
        None => Some(half),
    }
}

pub fn cap_slice(
    live_remaining_ms: Option<u128>,
    planned_ms: Option<u128>,
    spent_ms: u128,
) -> Option<u128> {
    const NEGLIGIBLE_BURN_DIVISOR: u128 = 32;
    let live = live_remaining_ms?;
    let planned = planned_ms?;
    if spent_ms.saturating_mul(NEGLIGIBLE_BURN_DIVISOR) < live {
        return Some(planned);
    }
    Some(planned.saturating_sub(spent_ms).min(live / 2))
}

pub fn worker_split(
    total: usize,
    shares: &[(String, f64)],
    engines: &[Engine],
) -> Vec<(Engine, usize)> {
    let mut out = Vec::with_capacity(engines.len());
    let mut remaining = total;
    for (index, engine) in engines.iter().enumerate() {
        let share = shares
            .iter()
            .find(|(name, _)| *name == engine.name())
            .map(|(_, share)| *share)
            .unwrap_or(0.0);
        let reserve = engines.len() - index - 1;
        let cap = remaining.saturating_sub(reserve);
        let workers = ((total as f64 * share).floor() as usize).max(1).min(cap);
        remaining = remaining.saturating_sub(workers);
        out.push((*engine, workers));
    }
    out
}

#[derive(Clone, Debug, Default)]
pub struct TrancheState {
    pub attempts: u64,
    pub spent_ms: u128,
    pub winner: Option<String>,
}

impl TrancheState {
    pub fn record_attempt(&mut self, ms: u128, _won: bool) {
        self.attempts += 1;
        self.spent_ms = self.spent_ms.saturating_add(ms);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReclaimStep {
    None,
    DemoteEngine,
    HalveWorkers,
}

impl ReclaimStep {
    pub fn name(self) -> &'static str {
        match self {
            ReclaimStep::None => "none",
            ReclaimStep::DemoteEngine => "demote_engine",
            ReclaimStep::HalveWorkers => "halve_workers",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReclaimPlan {
    pub step: ReclaimStep,
    pub engine: Engine,
    pub workers: usize,
    pub live_bytes: u64,
    pub budget_bytes: u64,
}

pub fn estimate_live_bytes(limbs: usize, engine_name: &str, workers: usize) -> u64 {
    const PROCESS_BASE_BYTES: u64 = 4 << 20;
    const SCHOOLBOOK_BYTES_PER_LIMB: u64 = 24;
    const KARATSUBA_BYTES_PER_LIMB: u64 = 48;
    const TOOM_BYTES_PER_LIMB: u64 = 256;
    const SMALL_NTT_BYTES_PER_LIMB: u64 = 320;
    const HVH_BYTES_PER_LIMB: u64 = 512;
    let width = limbs.max(1) as u64;
    let threads = workers.max(1) as u64;
    let scaled = match engine_name {
        "schoolbook" => width
            .saturating_mul(SCHOOLBOOK_BYTES_PER_LIMB)
            .saturating_mul(threads),
        "karatsuba" => width
            .saturating_mul(KARATSUBA_BYTES_PER_LIMB)
            .saturating_mul(threads),
        "toom3" | "toom4" => width
            .saturating_mul(TOOM_BYTES_PER_LIMB)
            .saturating_mul(threads),
        "ssa" | "furer" => width
            .saturating_mul(SMALL_NTT_BYTES_PER_LIMB)
            .saturating_mul(threads),
        _ => width
            .saturating_mul(HVH_BYTES_PER_LIMB)
            .saturating_mul(threads),
    };
    PROCESS_BASE_BYTES.saturating_add(scaled)
}

pub fn demote_engine(name: &str) -> &str {
    match name {
        "hvh-full" => "hvh",
        "hvh" => "furer",
        "furer" => "ssa",
        "ssa" => "toom3",
        "toom4" => "toom3",
        "toom3" => "karatsuba",
        "karatsuba" => "schoolbook",
        _ => "schoolbook",
    }
}

pub fn next_reclaim_step(
    live_bytes: u64,
    budget_bytes: u64,
    demoted: bool,
    halved: bool,
) -> ReclaimStep {
    if live_bytes <= budget_bytes {
        return ReclaimStep::None;
    }
    if !demoted {
        return ReclaimStep::DemoteEngine;
    }
    if !halved {
        return ReclaimStep::HalveWorkers;
    }
    ReclaimStep::None
}

pub fn plan_reclaim(
    limbs: usize,
    engine: Engine,
    workers: usize,
    budget_bytes: u64,
) -> ReclaimPlan {
    let mut current = engine;
    let mut thread_count = workers.max(1);
    let mut live = estimate_live_bytes(limbs, current.name(), thread_count);
    let mut step = ReclaimStep::None;
    let mut demoted = false;
    let mut halved = false;
    loop {
        match next_reclaim_step(live, budget_bytes, demoted, halved) {
            ReclaimStep::None => break,
            ReclaimStep::DemoteEngine => {
                if let Some(next) = Engine::from_str(demote_engine(current.name())) {
                    current = next;
                }
                demoted = true;
                live = estimate_live_bytes(limbs, current.name(), thread_count);
                step = ReclaimStep::DemoteEngine;
            }
            ReclaimStep::HalveWorkers => {
                let cut = (thread_count / 2).max(1);
                if cut < thread_count {
                    live = live.saturating_mul(cut as u64) / (thread_count.max(1) as u64);
                    thread_count = cut;
                }
                halved = true;
                step = ReclaimStep::HalveWorkers;
            }
        }
    }
    ReclaimPlan {
        step,
        engine: current,
        workers: thread_count,
        live_bytes: live,
        budget_bytes,
    }
}

pub fn reclaim_note(plan: &ReclaimPlan) -> String {
    format!(
        "reclaim={} engine={} workers={} live_bytes={} budget_bytes={}",
        plan.step.name(),
        plan.engine.name(),
        plan.workers,
        plan.live_bytes,
        plan.budget_bytes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::MulSignals;

    fn measured(rate: f64) -> MulSignals {
        let mut signals = MulSignals::collect(64, 64, 4096, None);
        for engine in Engine::all() {
            signals.report_engine(engine.name(), 1000, (rate * 1000.0) as u128);
        }
        signals
    }

    #[test]
    fn unknown_engine_projects_no_cost() {
        let signals = MulSignals::collect(64, 64, 4096, None);
        assert_eq!(project_cost_ms(Engine::Hvh, 64, &signals), None);
        assert_eq!(project_cost_ms(Engine::HvhFull, 64, &signals), None);
    }

    #[test]
    fn measured_rate_projects_scaled_cost() {
        let signals = measured(1000.0);
        let limbs = 64usize;
        let cost = project_cost_ms(Engine::Schoolbook, limbs, &signals).expect("cost");
        let work = crate::schoolbook::predicted_work_limbs(limbs);
        assert_eq!(cost, (1000.0 * work / 1_000_000.0).ceil() as u128);
    }

    #[test]
    fn toom4_estimator_arm_fires() {
        let mut signals = MulSignals::collect(64, 64, 4096, None);
        signals.report_engine("toom4", 1000, 1_000_000);
        let limbs = 256usize;
        let cost = project_cost_ms(Engine::Toom4, limbs, &signals).expect("cost");
        let work = crate::toom::predicted_work_limbs_toom4(limbs);
        assert_eq!(cost, (1000.0 * work / 1_000_000.0).ceil() as u128);
        assert_ne!(work, crate::toom::predicted_work_limbs(limbs));
    }

    #[test]
    fn fast_engine_wins_shares_and_sums_to_one() {
        let shares = cost_shares(&[("schoolbook", 100), ("hvh", 25)]);
        assert_eq!(shares.len(), 2);
        let hvh = shares
            .iter()
            .find(|(name, _)| *name == "hvh")
            .expect("hvh")
            .1;
        let schoolbook = shares
            .iter()
            .find(|(name, _)| *name == "schoolbook")
            .expect("schoolbook")
            .1;
        assert!(hvh > schoolbook);
        let total: f64 = shares.iter().map(|(_, share)| share).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn shares_handle_empty_and_zero() {
        assert!(cost_shares(&[]).is_empty());
        let even = cost_shares(&[("a", 0), ("b", 0)]);
        assert_eq!(even, vec![("a", 0.5), ("b", 0.5)]);
        let zero_wins = cost_shares(&[("a", 0), ("b", 50)]);
        assert_eq!(zero_wins, vec![("a", 1.0), ("b", 0.0)]);
    }

    #[test]
    fn switch_moves_to_cheaper_and_stays_on_winner() {
        let costs = [("schoolbook", 100u128), ("hvh", 25u128)];
        assert_eq!(should_switch("schoolbook", &costs), Some("hvh"));
        assert_eq!(should_switch("hvh", &costs), None);
        assert_eq!(should_switch("ssa", &costs), None);
        assert_eq!(should_switch("schoolbook", &[]), None);
    }

    #[test]
    fn tranche_first_box_clamps_to_live() {
        assert_eq!(tranche_first_box(Some(700), Some(800)), Some(350));
        assert_eq!(tranche_first_box(Some(700), Some(400)), Some(200));
        assert_eq!(tranche_first_box(Some(700), None), Some(350));
        assert_eq!(tranche_first_box(None, Some(400)), None);
    }

    #[test]
    fn cap_slice_clamps_after_burn() {
        assert_eq!(cap_slice(Some(1000), Some(800), 100), Some(500));
        assert_eq!(cap_slice(Some(10000), Some(800), 100), Some(800));
        assert_eq!(cap_slice(None, Some(800), 100), None);
        assert_eq!(cap_slice(Some(1000), None, 100), None);
        assert_eq!(cap_slice(Some(1000), Some(800), 900), Some(0));
    }

    #[test]
    fn worker_split_proportional_bounded_unit() {
        let shares = vec![("schoolbook".to_string(), 0.25), ("hvh".to_string(), 0.75)];
        let engines = [Engine::Schoolbook, Engine::Hvh];
        let split = worker_split(16, &shares, &engines);
        assert_eq!(split, vec![(Engine::Schoolbook, 4), (Engine::Hvh, 12)]);
        let unlisted = worker_split(8, &[], &engines);
        let sum: usize = unlisted.iter().map(|(_, workers)| workers).sum();
        assert!(sum <= 8);
        assert!(unlisted.iter().all(|(_, workers)| *workers >= 1));
        assert!(worker_split(4, &[], &[]).is_empty());
    }

    #[test]
    fn tranche_state_records_attempts() {
        let mut state = TrancheState::default();
        assert_eq!(state.attempts, 0);
        assert_eq!(state.spent_ms, 0);
        assert_eq!(state.winner, None);
        state.record_attempt(120, false);
        state.record_attempt(30, true);
        state.winner = Some("hvh".to_string());
        assert_eq!(state.attempts, 2);
        assert_eq!(state.spent_ms, 150);
        assert_eq!(state.winner.as_deref(), Some("hvh"));
    }

    #[test]
    fn demote_walks_down_ladder() {
        assert_eq!(demote_engine("hvh-full"), "hvh");
        assert_eq!(demote_engine("hvh"), "furer");
        assert_eq!(demote_engine("furer"), "ssa");
        assert_eq!(demote_engine("ssa"), "toom3");
        assert_eq!(demote_engine("toom4"), "toom3");
        assert_eq!(demote_engine("toom3"), "karatsuba");
        assert_eq!(demote_engine("karatsuba"), "schoolbook");
        assert_eq!(demote_engine("schoolbook"), "schoolbook");
    }

    #[test]
    fn reclaim_order_demotes_before_halving() {
        assert_eq!(
            next_reclaim_step(2000, 1000, false, false),
            ReclaimStep::DemoteEngine
        );
        assert_eq!(
            next_reclaim_step(2000, 1000, true, false),
            ReclaimStep::HalveWorkers
        );
        assert_eq!(next_reclaim_step(2000, 1000, true, true), ReclaimStep::None);
        assert_eq!(
            next_reclaim_step(500, 1000, false, false),
            ReclaimStep::None
        );
        assert_eq!(
            next_reclaim_step(1000, 1000, false, false),
            ReclaimStep::None
        );
    }

    #[test]
    fn estimate_monotonic_in_limbs() {
        for name in [
            "schoolbook",
            "karatsuba",
            "toom3",
            "ssa",
            "furer",
            "hvh",
            "hvh-full",
        ] {
            let small = estimate_live_bytes(100, name, 4);
            let large = estimate_live_bytes(1000, name, 4);
            assert!(large > small);
            let narrow = estimate_live_bytes(1000, name, 1);
            assert!(large >= narrow);
        }
    }

    #[test]
    fn estimate_covers_base_and_orders_ntt_top() {
        assert!(estimate_live_bytes(1, "schoolbook", 1) >= (4 << 20));
        let limbs = 32768usize;
        let schoolbook = estimate_live_bytes(limbs, "schoolbook", 1);
        let karatsuba = estimate_live_bytes(limbs, "karatsuba", 1);
        let toom4 = estimate_live_bytes(limbs, "toom4", 1);
        let ssa = estimate_live_bytes(limbs, "ssa", 1);
        let hvh = estimate_live_bytes(limbs, "hvh", 1);
        assert!(karatsuba > schoolbook);
        assert!(toom4 > karatsuba);
        assert!(ssa > toom4);
        assert!(hvh > ssa);
        assert_eq!(
            estimate_live_bytes(limbs, "hvh-full", 1),
            estimate_live_bytes(limbs, "hvh", 1)
        );
    }

    #[test]
    fn tiny_budget_demotes_ssa_to_toom3_and_halves() {
        let plan = plan_reclaim(16384, Engine::Ssa, 16, 1024 * 1024);
        assert_eq!(plan.engine, Engine::Toom3);
        assert_eq!(plan.workers, 8);
        assert_eq!(plan.step, ReclaimStep::HalveWorkers);
    }

    #[test]
    fn over_budget_fixture_demotes_and_halves() {
        let plan = plan_reclaim(10_000, Engine::Hvh, 16, 1024);
        assert_eq!(plan.engine, Engine::Furer);
        assert_eq!(plan.step, ReclaimStep::HalveWorkers);
        assert_eq!(plan.workers, 8);
        assert!(reclaim_note(&plan).contains("reclaim=halve_workers"));
    }

    #[test]
    fn over_budget_mib_fixture_demotes() {
        let plan = plan_reclaim(9000, Engine::Hvh, 2, 1024 * 1024);
        assert_eq!(plan.engine, Engine::Furer);
        assert_eq!(plan.workers, 1);
    }

    #[test]
    fn schoolbook_never_demotes() {
        let plan = plan_reclaim(10_000, Engine::Schoolbook, 4, 1024);
        assert_eq!(plan.engine, Engine::Schoolbook);
        assert!(reclaim_note(&plan).contains("engine=schoolbook"));
    }

    #[test]
    fn projected_margin_needs_two_costs() {
        assert_eq!(projected_margin(&[]), None);
        assert_eq!(projected_margin(&[("schoolbook".to_string(), 100)]), None);
        assert_eq!(
            projected_margin(&[("schoolbook".to_string(), 100), ("hvh".to_string(), 150)]),
            Some(0.5)
        );
    }

    #[test]
    fn projected_margin_sorts_and_guards_zero_best() {
        let margin = projected_margin(&[("hvh".to_string(), 150), ("schoolbook".to_string(), 100)])
            .expect("margin");
        assert!((margin - 0.5).abs() < 1e-12);
        assert_eq!(
            projected_margin(&[("a".to_string(), 0), ("b".to_string(), 50)]),
            None
        );
        assert_eq!(
            projected_margin(&[("a".to_string(), 100), ("b".to_string(), 100)]),
            Some(0.0)
        );
    }

    #[test]
    fn margin_clears_variance_pins() {
        assert!(margin_clears_variance(0.1, 0.01));
        assert!(!margin_clears_variance(0.02, 0.01));
        assert!(!margin_clears_variance(0.03, 0.01));
        assert!(!margin_clears_variance(f64::NAN, 0.01));
        assert!(!margin_clears_variance(0.5, f64::INFINITY));
    }

    #[test]
    fn raw_projection_keeps_sub_ms_resolution() {
        let fine = project_cost_raw_ns(Engine::Schoolbook, 1, 200.0).expect("raw");
        assert!((fine - 200.0).abs() < 1e-9);
        let margin = projected_margin_f64(&[
            ("schoolbook".to_string(), 200.0),
            ("hvh".to_string(), 20_000.0),
        ])
        .expect("raw margin");
        assert!((margin - 99.0).abs() < 1e-9);
        assert_eq!(projected_margin_f64(&[]), None);
    }
}
