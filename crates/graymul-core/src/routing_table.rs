use crate::dispatch::Engine;

pub const ROUTING_TABLE_VERSION: &str = "0.1.1";

pub const ROUTING_TABLE: [(u64, u64, &str); 7] = [
    (1, 64, "schoolbook"),
    (64, 256, "schoolbook"),
    (256, 1024, "schoolbook"),
    (1024, 2048, "toom3"),
    (2048, 8192, "toom4"),
    (8192, 16384, "toom4"),
    (16384, u64::MAX, "toom4"),
];

pub fn band_for_limbs(limbs: usize) -> Option<(u64, u64)> {
    let query = limbs as u64;
    for (min, max, _) in ROUTING_TABLE {
        if query >= min && (max == u64::MAX || query < max) {
            return Some((min, max));
        }
    }
    None
}

pub fn route_for_limbs(limbs: usize) -> Option<Engine> {
    let query = limbs as u64;
    for (min, max, name) in ROUTING_TABLE {
        if query >= min && (max == u64::MAX || query < max) {
            return Engine::from_str(name);
        }
    }
    if query < ROUTING_TABLE[0].0 {
        return Engine::from_str(ROUTING_TABLE[0].2);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_version_matches_routing_json() {
        let text = include_str!("../../../bench/routing.json");
        let parsed: serde_json::Value = serde_json::from_str(text).expect("routing.json parses");
        let version = parsed
            .get("version")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        assert_eq!(ROUTING_TABLE_VERSION, version);
        let bands = parsed
            .get("bands")
            .and_then(|value| value.as_array())
            .expect("bands");
        assert_eq!(ROUTING_TABLE.len(), bands.len());
    }

    #[test]
    fn table_engines_belong_to_race_set() {
        for (_, _, name) in ROUTING_TABLE {
            let engine = Engine::from_str(name).expect("known engine");
            assert!(Engine::all().contains(&engine));
            assert_ne!(engine, Engine::HvhFull);
        }
    }

    #[test]
    fn route_covers_all_sizes() {
        for limbs in [
            0usize, 1, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 8191, 8192,
            8193, 16383, 16384, 16385, 65536, 1_000_000,
        ] {
            assert!(route_for_limbs(limbs).is_some());
        }
        let mut limbs = 0usize;
        while limbs <= 1_000_000 {
            assert!(route_for_limbs(limbs).is_some());
            limbs += 99991;
        }
    }
}
