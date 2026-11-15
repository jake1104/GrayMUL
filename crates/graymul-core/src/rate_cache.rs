use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

pub const EMA_ALPHA: f64 = 0.3;
pub const CACHED_SAMPLES: u64 = 2;

#[derive(Clone, Copy, Debug)]
struct EngineRate {
    samples: u64,
    ns_per_limb: f64,
    mean: f64,
    m2: f64,
}

static RATE_CACHE: OnceLock<RwLock<HashMap<&'static str, EngineRate>>> = OnceLock::new();

fn cache() -> &'static RwLock<HashMap<&'static str, EngineRate>> {
    RATE_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn observe(engine: &'static str, limbs: usize, elapsed_ns: u128) {
    if limbs == 0 || elapsed_ns == 0 {
        return;
    }
    let rate = elapsed_ns as f64 / limbs as f64;
    if !rate.is_finite() || rate <= 0.0 {
        return;
    }
    if let Ok(mut guard) = cache().write() {
        guard
            .entry(engine)
            .and_modify(|entry| {
                entry.samples = entry.samples.saturating_add(1);
                entry.ns_per_limb = EMA_ALPHA * rate + (1.0 - EMA_ALPHA) * entry.ns_per_limb;
                let count = entry.samples as f64;
                let delta = rate - entry.mean;
                entry.mean += delta / count;
                entry.m2 += delta * (rate - entry.mean);
            })
            .or_insert(EngineRate {
                samples: 1,
                ns_per_limb: rate,
                mean: rate,
                m2: 0.0,
            });
    }
}

pub fn get(engine: &str) -> Option<f64> {
    cache()
        .read()
        .map(|guard| guard.get(engine).map(|entry| entry.ns_per_limb))
        .unwrap_or(None)
}

pub fn sample_count(engine: &str) -> u64 {
    cache()
        .read()
        .map(|guard| guard.get(engine).map(|entry| entry.samples).unwrap_or(0))
        .unwrap_or(0)
}

pub fn variance(engine: &str) -> Option<f64> {
    let guard = cache().read().ok()?;
    let entry = guard.get(engine)?;
    if entry.samples < 3 {
        return None;
    }
    let var = entry.m2 / (entry.samples as f64 - 1.0);
    if var.is_finite() && var >= 0.0 {
        Some(var)
    } else {
        None
    }
}

pub fn cv(engine: &str) -> Option<f64> {
    let guard = cache().read().ok()?;
    let entry = guard.get(engine)?;
    if entry.samples < 3 {
        return None;
    }
    if !entry.mean.is_finite() || entry.mean <= 0.0 {
        return None;
    }
    let var = variance(engine)?;
    let cv = var.sqrt() / entry.mean;
    if cv.is_finite() && cv >= 0.0 {
        Some(cv)
    } else {
        None
    }
}

#[cfg(test)]
pub fn reset_for_tests() {
    if let Ok(mut guard) = cache().write() {
        guard.clear();
    }
}

#[cfg(test)]
static CACHE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub fn lock_for_tests() -> std::sync::MutexGuard<'static, ()> {
    CACHE_SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_blends_samples() {
        let _guard = lock_for_tests();
        reset_for_tests();
        observe("schoolbook", 100, 10_000);
        assert_eq!(get("schoolbook"), Some(100.0));
        assert_eq!(sample_count("schoolbook"), 1);
        observe("schoolbook", 100, 20_000);
        let blended = get("schoolbook").expect("blended rate");
        assert!((blended - 130.0).abs() < 1e-9);
        assert_eq!(sample_count("schoolbook"), 2);
    }

    #[test]
    fn zero_samples_ignored() {
        let _guard = lock_for_tests();
        reset_for_tests();
        observe("karatsuba", 0, 5000);
        observe("karatsuba", 100, 0);
        assert_eq!(get("karatsuba"), None);
        assert_eq!(sample_count("karatsuba"), 0);
    }

    #[test]
    fn unknown_engine_reads_empty() {
        let _guard = lock_for_tests();
        reset_for_tests();
        assert_eq!(get("nope"), None);
        assert_eq!(sample_count("nope"), 0);
    }

    #[test]
    fn reset_clears_state() {
        let _guard = lock_for_tests();
        reset_for_tests();
        observe("ssa", 50, 5000);
        assert_eq!(sample_count("ssa"), 1);
        reset_for_tests();
        assert_eq!(sample_count("ssa"), 0);
        assert_eq!(get("ssa"), None);
    }

    #[test]
    fn constant_stream_yields_zero_cv() {
        let _guard = lock_for_tests();
        reset_for_tests();
        observe("toom3", 100, 10_000);
        observe("toom3", 100, 10_000);
        assert_eq!(variance("toom3"), None);
        assert_eq!(cv("toom3"), None);
        observe("toom3", 100, 10_000);
        assert_eq!(variance("toom3"), Some(0.0));
        assert_eq!(cv("toom3"), Some(0.0));
    }

    #[test]
    fn spread_stream_yields_expected_variance() {
        let _guard = lock_for_tests();
        reset_for_tests();
        observe("hvh", 100, 10_000);
        observe("hvh", 100, 20_000);
        observe("hvh", 100, 30_000);
        let var = variance("hvh").expect("variance");
        assert!((var - 10_000.0).abs() < 1e-6);
        let spread = cv("hvh").expect("cv");
        assert!((spread - 0.5).abs() < 1e-9);
    }
}
