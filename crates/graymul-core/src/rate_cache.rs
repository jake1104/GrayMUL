use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

pub const EMA_ALPHA: f64 = 0.3;
pub const CACHED_SAMPLES: u64 = 2;

#[derive(Clone, Copy, Debug)]
struct EngineRate {
    samples: u64,
    coeff: f64,
    mean: f64,
    m2: f64,
}

static RATE_CACHE: OnceLock<RwLock<HashMap<(&'static str, u64), EngineRate>>> = OnceLock::new();

fn cache() -> &'static RwLock<HashMap<(&'static str, u64), EngineRate>> {
    RATE_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn band_key_for_limbs(limbs: usize) -> u64 {
    match crate::routing_table::band_for_limbs(limbs) {
        Some((min, _)) => min,
        None => crate::routing_table::ROUTING_TABLE[0].0,
    }
}

pub fn observe(engine: &'static str, limbs: usize, elapsed_ns: u128) {
    if limbs == 0 || elapsed_ns == 0 {
        return;
    }
    let engine_kind = match crate::dispatch::Engine::from_str(engine) {
        Some(kind) => kind,
        None => return,
    };
    let coeff = match crate::cascade::coeff_for_observation(engine_kind, limbs, elapsed_ns) {
        Some(value) => value,
        None => return,
    };
    if !coeff.is_finite() || coeff <= 0.0 {
        return;
    }
    let band = band_key_for_limbs(limbs);
    if let Ok(mut guard) = cache().write() {
        guard
            .entry((engine, band))
            .and_modify(|entry| {
                entry.samples = entry.samples.saturating_add(1);
                entry.coeff = EMA_ALPHA * coeff + (1.0 - EMA_ALPHA) * entry.coeff;
                let count = entry.samples as f64;
                let delta = coeff - entry.mean;
                entry.mean += delta / count;
                entry.m2 += delta * (coeff - entry.mean);
            })
            .or_insert(EngineRate {
                samples: 1,
                coeff,
                mean: coeff,
                m2: 0.0,
            });
    }
}

pub fn get_for_limbs(engine: &str, limbs: usize) -> Option<f64> {
    let band = band_key_for_limbs(limbs);
    cache()
        .read()
        .map(|guard| guard.get(&(engine, band)).map(|entry| entry.coeff))
        .unwrap_or(None)
}

pub fn sample_count_for_limbs(engine: &str, limbs: usize) -> u64 {
    let band = band_key_for_limbs(limbs);
    cache()
        .read()
        .map(|guard| {
            guard
                .get(&(engine, band))
                .map(|entry| entry.samples)
                .unwrap_or(0)
        })
        .unwrap_or(0)
}

pub fn variance_for_limbs(engine: &str, limbs: usize) -> Option<f64> {
    let band = band_key_for_limbs(limbs);
    let guard = cache().read().ok()?;
    let entry = guard.get(&(engine, band))?;
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

pub fn cv_for_limbs(engine: &str, limbs: usize) -> Option<f64> {
    let band = band_key_for_limbs(limbs);
    let guard = cache().read().ok()?;
    let entry = guard.get(&(engine, band))?;
    if entry.samples < 3 {
        return None;
    }
    if !entry.mean.is_finite() || entry.mean <= 0.0 {
        return None;
    }
    let var = variance_for_limbs(engine, limbs)?;
    let cv = var.sqrt() / entry.mean;
    if cv.is_finite() && cv >= 0.0 {
        Some(cv)
    } else {
        None
    }
}

pub fn get(engine: &str) -> Option<f64> {
    let guard = cache().read().ok()?;
    let mut total = 0u64;
    let mut weighted = 0.0;
    for ((name, _), entry) in guard.iter() {
        if *name == engine {
            total = total.saturating_add(entry.samples);
            weighted += entry.samples as f64 * entry.coeff;
        }
    }
    if total == 0 {
        return None;
    }
    let pooled = weighted / total as f64;
    if pooled.is_finite() && pooled > 0.0 {
        Some(pooled)
    } else {
        None
    }
}

pub fn sample_count(engine: &str) -> u64 {
    cache()
        .read()
        .map(|guard| {
            guard
                .iter()
                .filter(|((name, _), _)| *name == engine)
                .map(|(_, entry)| entry.samples)
                .sum()
        })
        .unwrap_or(0)
}

pub fn variance(engine: &str) -> Option<f64> {
    let guard = cache().read().ok()?;
    let mut total = 0u64;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for ((name, _), entry) in guard.iter() {
        if *name != engine || entry.samples == 0 {
            continue;
        }
        let count = entry.samples as f64;
        let combined = total as f64 + count;
        let delta = entry.mean - mean;
        mean += delta * count / combined;
        m2 += entry.m2 + delta * delta * total as f64 * count / combined;
        total = total.saturating_add(entry.samples);
    }
    if total < 3 {
        return None;
    }
    let var = m2 / (total as f64 - 1.0);
    if var.is_finite() && var >= 0.0 {
        Some(var)
    } else {
        None
    }
}

pub fn cv(engine: &str) -> Option<f64> {
    let guard = cache().read().ok()?;
    let mut total = 0u64;
    let mut weighted = 0.0;
    for ((name, _), entry) in guard.iter() {
        if *name == engine {
            total = total.saturating_add(entry.samples);
            weighted += entry.samples as f64 * entry.mean;
        }
    }
    if total < 3 {
        return None;
    }
    let mean = weighted / total as f64;
    if !mean.is_finite() || mean <= 0.0 {
        return None;
    }
    let var = variance(engine)?;
    let cv = var.sqrt() / mean;
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
        assert_eq!(get("schoolbook"), Some(1.0));
        assert_eq!(sample_count("schoolbook"), 1);
        observe("schoolbook", 100, 20_000);
        let blended = get("schoolbook").expect("blended coeff");
        assert!((blended - 1.3).abs() < 1e-9);
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
    fn unknown_engine_observations_dropped() {
        let _guard = lock_for_tests();
        reset_for_tests();
        observe("nope", 100, 5000);
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
        observe("schoolbook", 100, 10_000);
        observe("schoolbook", 100, 20_000);
        observe("schoolbook", 100, 30_000);
        let var = variance("schoolbook").expect("variance");
        assert!((var - 1.0).abs() < 1e-9);
        let spread = cv("schoolbook").expect("cv");
        assert!((spread - 0.5).abs() < 1e-9);
        let banded = variance_for_limbs("schoolbook", 100).expect("banded variance");
        assert!((banded - 1.0).abs() < 1e-9);
    }

    #[test]
    fn band_separation_same_engine_sizes_do_not_share_mean() {
        let _guard = lock_for_tests();
        reset_for_tests();
        observe("schoolbook", 64, 64 * 64 * 10);
        observe("schoolbook", 4096, 4096 * 4096 * 90);
        let narrow = get_for_limbs("schoolbook", 64).expect("narrow band coeff");
        let wide = get_for_limbs("schoolbook", 4096).expect("wide band coeff");
        assert!((narrow - 10.0).abs() < 1e-9);
        assert!((wide - 90.0).abs() < 1e-9);
        assert_eq!(sample_count_for_limbs("schoolbook", 64), 1);
        assert_eq!(sample_count_for_limbs("schoolbook", 4096), 1);
    }

    #[test]
    fn normalization_recovers_synthetic_coefficient() {
        let _guard = lock_for_tests();
        reset_for_tests();
        let school_c = 7.25;
        let school_work = crate::schoolbook::predicted_work_limbs(100);
        observe("schoolbook", 100, (school_c * school_work) as u128);
        let recovered = get_for_limbs("schoolbook", 100).expect("schoolbook coeff");
        assert!((recovered - school_c).abs() < 1e-3);
        let kara_c = 3.5;
        let kara_work = crate::karatsuba::predicted_work_limbs(200);
        observe("karatsuba", 200, (kara_c * kara_work) as u128);
        let kara_recovered = get_for_limbs("karatsuba", 200).expect("karatsuba coeff");
        assert!((kara_recovered - kara_c).abs() < 1e-3);
    }

    #[test]
    fn band_keys_follow_routing_table_edges() {
        let _guard = lock_for_tests();
        assert_eq!(band_key_for_limbs(0), 1);
        assert_eq!(band_key_for_limbs(1), 1);
        assert_eq!(band_key_for_limbs(63), 1);
        assert_eq!(band_key_for_limbs(64), 64);
        assert_eq!(band_key_for_limbs(255), 64);
        assert_eq!(band_key_for_limbs(256), 256);
        assert_eq!(band_key_for_limbs(1023), 256);
        assert_eq!(band_key_for_limbs(1024), 1024);
        assert_eq!(band_key_for_limbs(2047), 1024);
        assert_eq!(band_key_for_limbs(2048), 2048);
        assert_eq!(band_key_for_limbs(8191), 2048);
        assert_eq!(band_key_for_limbs(8192), 8192);
        assert_eq!(band_key_for_limbs(16383), 8192);
        assert_eq!(band_key_for_limbs(16384), 16384);
        assert_eq!(band_key_for_limbs(1_000_000), 16384);
    }
}
