use serde::Serialize;

use crate::options::MultiplyOptions;

pub const REASON_STATIC_BAND: &str = "static-band";
pub const REASON_WARMED_OVERRIDE: &str = "warmed-override";
pub const REASON_RATIO_RULE: &str = "ratio-rule";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExplainOperands {
    pub limbs_a: usize,
    pub limbs_b: usize,
    pub limbs_max: usize,
    pub ratio: f64,
    pub negative_a: bool,
    pub negative_b: bool,
    pub negative: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExplainDispatch {
    pub static_route: String,
    pub selected: String,
    pub reason: String,
    pub warmed_override: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExplainResources {
    pub workers: usize,
    pub est_memory_bytes: u64,
    pub timeout_ms: Option<u128>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExplainVerification {
    pub methods: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Plan {
    pub operands: ExplainOperands,
    pub dispatch: ExplainDispatch,
    pub resources: ExplainResources,
    pub verification: ExplainVerification,
}

impl Plan {
    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("plan serializes to json")
    }

    pub fn report_lines(&self, a_text: &str, b_text: &str) -> Vec<String> {
        let warmed = self.dispatch.warmed_override.as_deref().unwrap_or("none");
        let timeout = self
            .resources
            .timeout_ms
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string());
        vec![
            format!("input_a: {a_text}"),
            format!("input_b: {b_text}"),
            format!(
                "operands: limbs_a={} limbs_b={} limbs_max={} ratio={:.3} negative={}",
                self.operands.limbs_a,
                self.operands.limbs_b,
                self.operands.limbs_max,
                self.operands.ratio,
                self.operands.negative
            ),
            format!(
                "dispatch: static_route={} selected={} reason={}",
                self.dispatch.static_route, self.dispatch.selected, self.dispatch.reason
            ),
            format!("warmed_override: {warmed}"),
            format!(
                "resources: workers={} est_memory_bytes={} timeout_ms={}",
                self.resources.workers, self.resources.est_memory_bytes, timeout
            ),
            format!(
                "verification: methods={}",
                self.verification.methods.join(",")
            ),
        ]
    }
}

fn warmed_override(limbs_max: usize, static_route: &str) -> Option<String> {
    let width = limbs_max.max(1);
    let mut best: Option<(String, f64)> = None;
    for engine in crate::dispatch::Engine::all() {
        if crate::rate_cache::sample_count_for_limbs(engine.name(), width)
            < crate::rate_cache::CACHED_SAMPLES
        {
            continue;
        }
        let coeff = match crate::rate_cache::get_for_limbs(engine.name(), width) {
            Some(value) => value,
            None => continue,
        };
        let cost = match crate::cascade::project_cost_raw_ns(engine, width, coeff) {
            Some(value) => value,
            None => continue,
        };
        let replace = match &best {
            None => true,
            Some((_, current)) => cost < *current,
        };
        if replace {
            best = Some((engine.name().to_string(), cost));
        }
    }
    match best {
        Some((name, _)) if name != static_route => Some(name),
        _ => None,
    }
}

pub fn explain_signed(
    a_limbs: usize,
    b_limbs: usize,
    negative_a: bool,
    negative_b: bool,
    options: &MultiplyOptions,
) -> Plan {
    let limbs_max = a_limbs.max(b_limbs);
    let denom = a_limbs.min(b_limbs).max(1);
    let ratio = limbs_max.max(1) as f64 / denom as f64;
    let static_route = crate::routing_table::route_for_limbs(limbs_max)
        .map(|engine| engine.name().to_string())
        .unwrap_or_else(|| "schoolbook".to_string());
    let warmed = warmed_override(limbs_max, &static_route);
    let (selected, reason) = match warmed.clone() {
        Some(engine) => (engine, REASON_WARMED_OVERRIDE.to_string()),
        None => (static_route.clone(), REASON_STATIC_BAND.to_string()),
    };
    let workers = options.effective_workers();
    let est_memory_bytes = crate::cascade::estimate_live_bytes(limbs_max, &selected, workers);
    let timeout_ms = Some(match options.max_duration {
        Some(span) => span.as_millis(),
        None => crate::options::probe_budget_ms_for_bits(limbs_max.max(1) as u64 * 64),
    });
    Plan {
        operands: ExplainOperands {
            limbs_a: a_limbs,
            limbs_b: b_limbs,
            limbs_max,
            ratio,
            negative_a,
            negative_b,
            negative: negative_a != negative_b,
        },
        dispatch: ExplainDispatch {
            static_route,
            selected,
            reason,
            warmed_override: warmed,
        },
        resources: ExplainResources {
            workers,
            est_memory_bytes,
            timeout_ms,
        },
        verification: ExplainVerification {
            methods: vec!["mod-check".to_string(), "length-pin".to_string()],
        },
    }
}

pub fn explain_plan(a_limbs: usize, b_limbs: usize, options: &MultiplyOptions) -> Plan {
    explain_signed(a_limbs, b_limbs, false, false, options)
}

pub fn explain_bigints(
    a: &num_bigint::BigUint,
    b: &num_bigint::BigUint,
    negative_a: bool,
    negative_b: bool,
    options: &MultiplyOptions,
) -> Plan {
    explain_signed(
        crate::limb::from_bigint(a).len(),
        crate::limb::from_bigint(b).len(),
        negative_a,
        negative_b,
        options,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_keys(value: &serde_json::Value) -> Vec<String> {
        let mut keys: Vec<String> = value
            .as_object()
            .expect("json object")
            .keys()
            .cloned()
            .collect();
        keys.sort();
        keys
    }

    fn cold_options() -> MultiplyOptions {
        MultiplyOptions {
            threads: Some(1),
            ..MultiplyOptions::default()
        }
    }

    #[test]
    fn json_schema_pins_exact_key_sets() {
        let _guard = crate::rate_cache::lock_for_tests();
        crate::rate_cache::reset_for_tests();
        let plan = explain_plan(64, 128, &cold_options());
        let value = plan.to_json_value();
        assert_eq!(
            sorted_keys(&value),
            vec![
                "dispatch".to_string(),
                "operands".to_string(),
                "resources".to_string(),
                "verification".to_string(),
            ]
        );
        assert_eq!(
            sorted_keys(&value["operands"]),
            vec![
                "limbs_a".to_string(),
                "limbs_b".to_string(),
                "limbs_max".to_string(),
                "negative".to_string(),
                "negative_a".to_string(),
                "negative_b".to_string(),
                "ratio".to_string(),
            ]
        );
        assert_eq!(
            sorted_keys(&value["dispatch"]),
            vec![
                "reason".to_string(),
                "selected".to_string(),
                "static_route".to_string(),
                "warmed_override".to_string(),
            ]
        );
        assert_eq!(
            sorted_keys(&value["resources"]),
            vec![
                "est_memory_bytes".to_string(),
                "timeout_ms".to_string(),
                "workers".to_string(),
            ]
        );
        assert_eq!(
            sorted_keys(&value["verification"]),
            vec!["methods".to_string()]
        );
        let reason = value["dispatch"]["reason"].as_str().expect("reason string");
        assert!(
            reason == REASON_STATIC_BAND
                || reason == REASON_WARMED_OVERRIDE
                || reason == REASON_RATIO_RULE,
            "reason {reason}"
        );
        let methods = value["verification"]["methods"]
            .as_array()
            .expect("methods array");
        assert!(!methods.is_empty());
        let text = value.to_string();
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("round trip");
        assert_eq!(parsed, value);
        let lines = plan.report_lines("12", "34");
        assert_eq!(lines.len(), 7);
        assert!(lines[0].starts_with("input_a: "));
        assert!(lines[1].starts_with("input_b: "));
        assert!(lines[2].starts_with("operands: "));
        assert!(lines[3].starts_with("dispatch: "));
        assert!(lines[4].starts_with("warmed_override: "));
        assert!(lines[5].starts_with("resources: "));
        assert!(lines[6].starts_with("verification: "));
    }

    #[test]
    fn plan_is_deterministic() {
        let _guard = crate::rate_cache::lock_for_tests();
        crate::rate_cache::reset_for_tests();
        let first = explain_plan(128, 1024, &cold_options());
        let second = explain_plan(128, 1024, &cold_options());
        assert_eq!(first, second);
        assert_eq!(first.to_json_value(), second.to_json_value());
    }

    #[test]
    fn explain_avoids_compute() {
        let _guard = crate::rate_cache::lock_for_tests();
        crate::rate_cache::reset_for_tests();
        let started = std::time::Instant::now();
        let plan = explain_plan(16384, 16384, &cold_options());
        let elapsed = started.elapsed();
        assert_eq!(plan.operands.limbs_max, 16384);
        assert!(
            elapsed < std::time::Duration::from_millis(10),
            "explain took {elapsed:?}"
        );
    }

    #[test]
    fn warmed_override_reported_as_separate_line() {
        let _guard = crate::rate_cache::lock_for_tests();
        crate::rate_cache::reset_for_tests();
        let cold = explain_plan(1024, 128, &cold_options());
        assert_eq!(cold.dispatch.reason, REASON_STATIC_BAND);
        assert_eq!(cold.dispatch.warmed_override, None);
        assert_eq!(cold.dispatch.selected, cold.dispatch.static_route);
        crate::rate_cache::observe("schoolbook", 1024, 1024);
        crate::rate_cache::observe("schoolbook", 1024, 1024);
        for name in ["karatsuba", "toom3", "toom4", "ssa", "furer", "hvh"] {
            crate::rate_cache::observe(name, 1024, 100_000_000);
            crate::rate_cache::observe(name, 1024, 100_000_000);
        }
        let warmed = explain_plan(1024, 128, &cold_options());
        assert_eq!(warmed.dispatch.static_route, cold.dispatch.static_route);
        assert_eq!(warmed.dispatch.reason, REASON_WARMED_OVERRIDE);
        assert_eq!(
            warmed.dispatch.warmed_override,
            Some("schoolbook".to_string())
        );
        assert_eq!(warmed.dispatch.selected, "schoolbook");
        let lines = warmed.report_lines("1", "2");
        let dispatch_line = lines
            .iter()
            .find(|line| line.starts_with("dispatch: "))
            .expect("dispatch line");
        let override_line = lines
            .iter()
            .find(|line| line.starts_with("warmed_override: "))
            .expect("override line");
        assert!(dispatch_line.contains(&warmed.dispatch.static_route));
        assert!(override_line.contains("schoolbook"));
        assert_ne!(dispatch_line, override_line);
        crate::rate_cache::reset_for_tests();
    }

    #[test]
    fn negative_inputs_report_sign_and_magnitude() {
        let _guard = crate::rate_cache::lock_for_tests();
        crate::rate_cache::reset_for_tests();
        let plan = explain_signed(1, 2, true, false, &cold_options());
        assert!(plan.operands.negative_a);
        assert!(!plan.operands.negative_b);
        assert!(plan.operands.negative);
        assert_eq!(plan.operands.limbs_a, 1);
        assert_eq!(plan.operands.limbs_b, 2);
        let even = explain_signed(3, 3, true, true, &cold_options());
        assert!(!even.operands.negative);
        let value = plan.to_json_value();
        assert_eq!(value["operands"]["negative"], true);
        let text = value.to_string();
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("round trip");
        assert_eq!(parsed, value);
    }

    #[test]
    fn ratio_rule_never_fires() {
        let _guard = crate::rate_cache::lock_for_tests();
        crate::rate_cache::reset_for_tests();
        let pairs = [
            (128, 256),
            (128, 1024),
            (128, 8192),
            (128, 65536),
            (1, 512),
            (64, 64),
            (1, 1),
            (16384, 16384),
        ];
        for (a, b) in pairs {
            let plan = explain_plan(a, b, &cold_options());
            assert_ne!(plan.dispatch.reason, REASON_RATIO_RULE, "limbs {a} {b}");
            assert!(
                plan.dispatch.reason == REASON_STATIC_BAND
                    || plan.dispatch.reason == REASON_WARMED_OVERRIDE,
                "limbs {a} {b}"
            );
        }
    }

    #[test]
    fn explain_reads_cache_without_writing() {
        let _guard = crate::rate_cache::lock_for_tests();
        crate::rate_cache::reset_for_tests();
        crate::rate_cache::observe("toom3", 100, 50_000);
        crate::rate_cache::observe("toom3", 100, 50_000);
        let before_count = crate::rate_cache::sample_count("toom3");
        let before_rate = crate::rate_cache::get("toom3");
        let _ = explain_plan(512, 512, &cold_options());
        let _ = explain_plan(1024, 64, &cold_options());
        assert_eq!(crate::rate_cache::sample_count("toom3"), before_count);
        assert_eq!(crate::rate_cache::get("toom3"), before_rate);
        crate::rate_cache::reset_for_tests();
    }
}
