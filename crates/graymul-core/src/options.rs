use std::time::Duration;

use crate::time::Instant;

pub const DEFAULT_MEMORY_BUDGET_BYTES: u64 = 1 << 30;
pub const PROBE_BUDGET_MIN_MS: f64 = 1_000.0;
pub const PROBE_BUDGET_MAX_MS: f64 = 3_600_000.0;

pub fn probe_budget_ms_for_bits(bits: u64) -> u128 {
    let ln_n = (bits as f64 * std::f64::consts::LN_2).max(1.0);
    let ln_ln_n = ln_n.ln().max(f64::EPSILON);
    let exponent =
        ln_n.powf(1.0 / 3.0) * ln_ln_n.powf(2.0 / 3.0) * (64.0_f64 / 9.0).powf(1.0 / 3.0);
    exponent
        .exp()
        .clamp(PROBE_BUDGET_MIN_MS, PROBE_BUDGET_MAX_MS) as u128
}

pub fn estimate_ntt_bytes(limbs: usize, workers: usize) -> u64 {
    let width = limbs.max(1) as u64;
    width
        .saturating_mul(16)
        .saturating_mul(workers.max(1) as u64)
}

#[derive(Clone, Debug)]
pub struct MultiplyOptions {
    pub algorithm: Option<String>,
    pub threads: Option<usize>,
    pub max_duration: Option<Duration>,
    pub max_memory_mb: Option<usize>,
    pub max_cpu_seconds: Option<u64>,
    pub seed: u64,
    pub perf_mode: bool,
}

impl Default for MultiplyOptions {
    fn default() -> Self {
        Self {
            algorithm: None,
            threads: None,
            max_duration: None,
            max_memory_mb: None,
            max_cpu_seconds: None,
            seed: 0x9e3779b97f4a7c15,
            perf_mode: false,
        }
    }
}

impl MultiplyOptions {
    pub fn memory_budget_bytes(&self) -> u64 {
        match self.max_memory_mb {
            Some(mb) => (mb as u64).saturating_mul(1024 * 1024),
            None => DEFAULT_MEMORY_BUDGET_BYTES,
        }
    }

    pub fn validate(&self) -> Result<(), crate::MultiplyError> {
        if let Some(algorithm) = &self.algorithm {
            const KNOWN: [&str; 9] = [
                "schoolbook",
                "karatsuba",
                "toom3",
                "toom4",
                "ssa",
                "furer",
                "hvh",
                "hvh-full",
                "auto",
            ];
            if !KNOWN.contains(&algorithm.as_str()) {
                return Err(crate::MultiplyError::Input(format!(
                    "unsupported algorithm {algorithm}"
                )));
            }
        }
        Ok(())
    }

    pub fn effective_workers(&self) -> usize {
        let available = crate::par::available_workers();
        self.threads.unwrap_or(available).max(1)
    }

    pub fn effective_deadline(&self, from: Instant) -> Option<Instant> {
        self.max_duration.map(|span| from + span)
    }

    pub fn cpu_budget_ms(&self) -> Option<u128> {
        self.max_cpu_seconds.map(|secs| u128::from(secs) * 1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_budget_defaults_to_one_gibibyte() {
        let options = MultiplyOptions::default();
        assert_eq!(options.memory_budget_bytes(), DEFAULT_MEMORY_BUDGET_BYTES);
    }

    #[test]
    fn unknown_algorithm_is_invalid() {
        let options = MultiplyOptions {
            algorithm: Some("gnfs".to_string()),
            ..MultiplyOptions::default()
        };
        assert!(options.validate().is_err());
    }

    #[test]
    fn known_algorithms_validate() {
        for name in [
            "schoolbook",
            "karatsuba",
            "toom3",
            "toom4",
            "ssa",
            "furer",
            "hvh",
            "hvh-full",
            "auto",
        ] {
            let options = MultiplyOptions {
                algorithm: Some(name.to_string()),
                ..MultiplyOptions::default()
            };
            assert!(options.validate().is_ok());
        }
    }
}
