use crate::dispatch::Engine;
use crate::limb::normalize_limbs;

pub fn mul_split_par(a: &[u64], b: &[u64], workers: usize) -> Vec<u64> {
    let split_a = a.len() >= b.len();
    let big_len = a.len().max(b.len());
    if workers < 2 || big_len < 2 {
        return sub_product(a, b);
    }
    let half = big_len / 2;
    #[cfg(target_arch = "wasm32")]
    let (lo, hi) = {
        if split_a {
            let (alo, ahi) = a.split_at(half);
            (sub_product(alo, b), sub_product(ahi, b))
        } else {
            let (blo, bhi) = b.split_at(half);
            (sub_product(a, blo), sub_product(a, bhi))
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    let (lo, hi) = std::thread::scope(|s| {
        if split_a {
            let (alo, ahi) = a.split_at(half);
            let hi_handle = s.spawn(move || sub_product(ahi, b));
            let lo = sub_product(alo, b);
            (lo, hi_handle.join().unwrap())
        } else {
            let (blo, bhi) = b.split_at(half);
            let hi_handle = s.spawn(move || sub_product(a, bhi));
            let lo = sub_product(a, blo);
            (lo, hi_handle.join().unwrap())
        }
    });
    let mut out = lo;
    add_shifted(&mut out, &hi, half);
    normalize_limbs(&mut out);
    out
}

fn sub_product(x: &[u64], other: &[u64]) -> Vec<u64> {
    let size = x.len().max(other.len());
    let engine = crate::routing_table::route_for_limbs(size).unwrap_or(Engine::Schoolbook);
    crate::multiply::run_engine(engine, x, other)
}

fn add_shifted(out: &mut Vec<u64>, add: &[u64], shift: usize) {
    if out.len() < shift + add.len() {
        out.resize(shift + add.len(), 0);
    }
    let mut carry: u128 = 0;
    let mut i = 0;
    while i < add.len() || carry > 0 {
        if shift + i >= out.len() {
            out.push(0);
        }
        let slot = &mut out[shift + i];
        let cur = u128::from(*slot) + u128::from(*add.get(i).unwrap_or(&0)) + carry;
        *slot = cur as u64;
        carry = cur >> 64;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn split_case_limbs(seed: u64, la: usize, lb: usize) -> (Vec<u64>, Vec<u64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut build = |len: usize| {
            let mut v: Vec<u64> = (0..len).map(|_| rng.next_u64() % 1_000_000).collect();
            if len > 0 {
                v[len - 1] |= 1u64 << 63;
            }
            v
        };
        (build(la), build(lb))
    }

    #[test]
    fn split_matches_serial_all_counts() {
        let sizes = [
            (1usize, 1usize),
            (0, 5),
            (5, 0),
            (1, 9),
            (3, 3),
            (64, 64),
            (65, 65),
            (129, 100),
            (100, 129),
            (300, 300),
            (2049, 2049),
        ];
        for (index, (la, lb)) in sizes.iter().enumerate() {
            let (a, b) = split_case_limbs(100u64 + index as u64, *la, *lb);
            let serial = sub_product(&a, &b);
            assert!(crate::verify::verify_mod(&a, &b, &serial));
            if *la <= 129 && *lb <= 129 {
                assert_eq!(serial, crate::schoolbook::mul(&a, &b));
            }
            for workers in [0usize, 1, 2, 3, 8] {
                assert_eq!(mul_split_par(&a, &b, workers), serial);
            }
            assert_eq!(mul_split_par(&a, &b, 2), mul_split_par(&a, &b, 2));
        }
    }

    #[test]
    fn split_single_level_subcalls_stay_serial() {
        let (a, b) = split_case_limbs(200u64, 300, 200);
        let serial = sub_product(&a, &b);
        assert_eq!(mul_split_par(&a, &b, 64), serial);
        let half = a.len().max(b.len()) / 2;
        let (alo, ahi) = a.split_at(half);
        assert_eq!(
            sub_product(alo, &b),
            crate::multiply::multiply_naked(
                crate::routing_table::route_for_limbs(alo.len().max(b.len()))
                    .unwrap_or(Engine::Schoolbook),
                alo,
                &b
            )
        );
        assert_eq!(
            sub_product(ahi, &b),
            crate::multiply::multiply_naked(
                crate::routing_table::route_for_limbs(ahi.len().max(b.len()))
                    .unwrap_or(Engine::Schoolbook),
                ahi,
                &b
            )
        );
    }
}
