use std::collections::HashMap;

use crate::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Engine {
    Schoolbook,
    Karatsuba,
    Toom3,
    Toom4,
    Ssa,
    Furer,
    Hvh,
    HvhFull,
}

impl Engine {
    pub fn name(self) -> &'static str {
        match self {
            Engine::Schoolbook => "schoolbook",
            Engine::Karatsuba => "karatsuba",
            Engine::Toom3 => "toom3",
            Engine::Toom4 => "toom4",
            Engine::Ssa => "ssa",
            Engine::Furer => "furer",
            Engine::Hvh => "hvh",
            Engine::HvhFull => "hvh-full",
        }
    }

    pub fn all() -> [Engine; 7] {
        [
            Engine::Schoolbook,
            Engine::Karatsuba,
            Engine::Toom3,
            Engine::Toom4,
            Engine::Ssa,
            Engine::Furer,
            Engine::Hvh,
        ]
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str) -> Option<Engine> {
        match text {
            "schoolbook" => Some(Engine::Schoolbook),
            "karatsuba" => Some(Engine::Karatsuba),
            "toom3" => Some(Engine::Toom3),
            "toom4" => Some(Engine::Toom4),
            "ssa" => Some(Engine::Ssa),
            "furer" => Some(Engine::Furer),
            "hvh" => Some(Engine::Hvh),
            "hvh-full" => Some(Engine::HvhFull),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MulSignals {
    pub limbs_a: usize,
    pub limbs_b: usize,
    pub limbs_max: usize,
    pub bits: u64,
    pub workers: usize,
    pub measured_ns_per_limb: HashMap<&'static str, f64>,
    pub measured_coeff: HashMap<(&'static str, u64), f64>,
    pub pilot_limbs_per_ms: Option<f64>,
    pub remaining_ms_at_collect: Option<u128>,
}

impl MulSignals {
    pub fn collect(a_limbs: usize, b_limbs: usize, bits: u64, deadline: Option<Instant>) -> Self {
        let workers = crate::par::available_workers();
        let remaining_ms_at_collect =
            deadline.map(|limit| limit.saturating_duration_since(Instant::now()).as_millis());
        Self {
            limbs_a: a_limbs,
            limbs_b: b_limbs,
            limbs_max: a_limbs.max(b_limbs),
            bits,
            workers,
            measured_ns_per_limb: HashMap::new(),
            measured_coeff: HashMap::new(),
            pilot_limbs_per_ms: None,
            remaining_ms_at_collect,
        }
    }

    pub fn observe_workers(&mut self, count: usize) {
        self.workers = count.max(1);
    }

    pub fn observe_options(
        &mut self,
        options: &crate::options::MultiplyOptions,
        deadline: Option<Instant>,
    ) {
        self.workers = options.effective_workers();
        if let Some(limit) = deadline {
            self.remaining_ms_at_collect =
                Some(limit.saturating_duration_since(Instant::now()).as_millis());
        } else if let Some(span) = options.max_duration {
            self.remaining_ms_at_collect = Some(span.as_millis());
        }
    }

    pub fn collect_with_options(
        a_limbs: usize,
        b_limbs: usize,
        bits: u64,
        options: &crate::options::MultiplyOptions,
        deadline: Option<Instant>,
    ) -> Self {
        let resolved = deadline.or_else(|| options.effective_deadline(Instant::now()));
        let mut signals = Self::collect(a_limbs, b_limbs, bits, resolved);
        signals.observe_options(options, resolved);
        signals
    }

    pub fn report_engine(&mut self, engine_name: &'static str, limbs: usize, elapsed_ns: u128) {
        if limbs == 0 {
            return;
        }
        let rate = elapsed_ns as f64 / limbs as f64;
        if rate.is_finite() && rate > 0.0 {
            self.measured_ns_per_limb.insert(engine_name, rate);
        }
        if let Some(engine) = Engine::from_str(engine_name)
            && let Some(coeff) = crate::cascade::coeff_for_observation(engine, limbs, elapsed_ns)
        {
            self.record_coeff(engine_name, limbs, coeff);
        }
    }

    pub fn record_coeff(&mut self, engine_name: &'static str, limbs: usize, coeff: f64) {
        if !coeff.is_finite() || coeff <= 0.0 {
            return;
        }
        let band = crate::rate_cache::band_key_for_limbs(limbs);
        self.measured_coeff.insert((engine_name, band), coeff);
    }

    pub fn coeff_for(&self, engine_name: &str, limbs: usize) -> Option<f64> {
        let band = crate::rate_cache::band_key_for_limbs(limbs);
        if let Some(coeff) = self.measured_coeff.get(&(engine_name, band)) {
            return Some(*coeff);
        }
        let mut fallback: Option<(u64, f64)> = None;
        for ((name, entry_band), coeff) in self.measured_coeff.iter() {
            if *name != engine_name {
                continue;
            }
            let distance = entry_band.abs_diff(band);
            let replace = match fallback {
                None => true,
                Some((current_band, _)) => {
                    distance < current_band.abs_diff(band)
                        || (distance == current_band.abs_diff(band) && *entry_band < current_band)
                }
            };
            if replace {
                fallback = Some((*entry_band, *coeff));
            }
        }
        fallback.map(|(_, coeff)| coeff)
    }

    pub fn report_pilot(&mut self, limbs: usize, elapsed_ms: u128) {
        let rate = limbs as f64 / elapsed_ms.max(1) as f64;
        if rate.is_finite() && rate > 0.0 {
            self.pilot_limbs_per_ms = Some(rate);
        }
    }

    pub fn remaining_ms(&self, deadline: Option<Instant>) -> Option<u128> {
        deadline.map(|limit| limit.saturating_duration_since(Instant::now()).as_millis())
    }

    pub fn effective_remaining_ms(&self, deadline: Option<Instant>) -> u128 {
        self.remaining_ms(deadline)
            .unwrap_or_else(|| Self::probe_budget_ms(self.bits))
    }

    pub fn probe_budget_ms(bits: u64) -> u128 {
        crate::options::probe_budget_ms_for_bits(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::MultiplyOptions;

    #[test]
    fn collect_observe_report_round_trip() {
        let mut signals = MulSignals::collect(10, 20, 1280, None);
        assert_eq!(signals.limbs_a, 10);
        assert_eq!(signals.limbs_b, 20);
        assert_eq!(signals.limbs_max, 20);
        assert_eq!(signals.bits, 1280);
        assert!(signals.workers >= 1);
        assert!(signals.measured_ns_per_limb.is_empty());
        assert_eq!(signals.pilot_limbs_per_ms, None);
        assert_eq!(signals.remaining_ms_at_collect, None);
        signals.observe_workers(4);
        assert_eq!(signals.workers, 4);
        signals.observe_workers(0);
        assert_eq!(signals.workers, 1);
        signals.report_engine("schoolbook", 100, 5000);
        assert_eq!(signals.measured_ns_per_limb.get("schoolbook"), Some(&50.0));
        assert_eq!(
            signals.coeff_for("schoolbook", 100),
            Some(5000.0 / crate::schoolbook::predicted_work_limbs(100))
        );
        assert_eq!(
            signals.coeff_for("schoolbook", 4096),
            signals.coeff_for("schoolbook", 100)
        );
        signals.report_engine("karatsuba", 0, 999);
        assert!(!signals.measured_ns_per_limb.contains_key("karatsuba"));
        signals.report_pilot(200, 4);
        assert_eq!(signals.pilot_limbs_per_ms, Some(50.0));
        assert_eq!(signals.remaining_ms(None), None);
        assert_eq!(
            signals.effective_remaining_ms(None),
            MulSignals::probe_budget_ms(1280)
        );
    }

    #[test]
    fn collect_with_options_carries_workers_and_deadline() {
        let options = MultiplyOptions {
            threads: Some(3),
            max_duration: Some(std::time::Duration::from_secs(30)),
            ..MultiplyOptions::default()
        };
        let signals = MulSignals::collect_with_options(5, 7, 640, &options, None);
        assert_eq!(signals.workers, 3);
        assert_eq!(signals.limbs_max, 7);
        let remaining = signals.remaining_ms_at_collect.expect("deadline set");
        assert!(remaining <= 30_000);
    }

    #[test]
    fn engine_names_cover_method_vocabulary() {
        let names: Vec<&str> = Engine::all().iter().map(|engine| engine.name()).collect();
        assert_eq!(
            names,
            vec![
                "schoolbook",
                "karatsuba",
                "toom3",
                "toom4",
                "ssa",
                "furer",
                "hvh"
            ]
        );
        for engine in Engine::all() {
            assert_eq!(Engine::from_str(engine.name()), Some(engine));
        }
        assert_eq!(Engine::from_str("hvh-full"), Some(Engine::HvhFull));
        assert_eq!(Engine::from_str("gnfs"), None);
        assert!(!Engine::all().contains(&Engine::HvhFull));
    }
}
