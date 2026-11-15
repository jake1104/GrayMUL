use crate::limb::normalize_limbs;
use crate::ntt;
use crate::scratch::WorkScratch;
use crate::toom;

pub const SSA_CUTOFF: usize = 512;
pub const SMALL_NTT_CUTOFF: usize = 512;
pub const RECURSION_CUTOFF: usize = 8192;
pub const RECURSION_CHUNKS: usize = 8;
pub const MAX_DEPTH: u32 = 2;

pub fn backend_for(limbs: usize) -> &'static str {
    if limbs < SMALL_NTT_CUTOFF {
        "crt"
    } else if limbs <= RECURSION_CUTOFF {
        "small"
    } else {
        "small-flat"
    }
}

pub fn mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.len().min(b.len()) <= SSA_CUTOFF {
        return toom::mul(a, b);
    }
    if backend_for(a.len().min(b.len())) != "crt" {
        return crate::small_ntt::crt_convolution_small(a, b);
    }
    mul_pool(a, b, &mut WorkScratch::new())
}

pub fn mul_recursive(a: &[u64], b: &[u64]) -> Vec<u64> {
    rec_depth(a, b, 0)
}

fn rec_depth(a: &[u64], b: &[u64], depth: u32) -> Vec<u64> {
    if depth >= MAX_DEPTH {
        return crate::small_ntt::crt_convolution_small(a, b);
    }
    let size = a.len().max(b.len()).div_ceil(RECURSION_CHUNKS);
    let ac = chunk_ranges(a, size);
    let bc = chunk_ranges(b, size);
    let mut out = vec![0u64];
    for (i, (astart, aend)) in ac.iter().enumerate() {
        for (j, (bstart, bend)) in bc.iter().enumerate() {
            let part = rec_depth(&a[*astart..*aend], &b[*bstart..*bend], depth + 1);
            add_shifted(&mut out, &part, (i + j) * size);
        }
    }
    normalize_limbs(&mut out);
    out
}

fn chunk_ranges(value: &[u64], size: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < value.len() {
        let end = (pos + size).min(value.len());
        out.push((pos, end));
        pos = end;
    }
    out
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

pub fn mul_small(a: &[u64], b: &[u64]) -> Vec<u64> {
    crate::small_ntt::crt_convolution_small(a, b)
}

pub(crate) fn mul_small_cancelled(
    a: &[u64],
    b: &[u64],
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    crate::small_ntt::crt_convolution_small_cancelled(a, b, cancelled)
}

pub(crate) fn mul_cancelled(
    a: &[u64],
    b: &[u64],
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    if a.len().min(b.len()) <= SSA_CUTOFF {
        return Ok(toom::mul(a, b));
    }
    if backend_for(a.len().min(b.len())) != "crt" {
        return mul_small_cancelled(a, b, cancelled);
    }
    Ok(mul_pool(a, b, &mut WorkScratch::new()))
}

pub fn mul_par(a: &[u64], b: &[u64], workers: usize) -> Vec<u64> {
    crate::small_ntt::convolution_parallel(a, b, workers)
}

pub fn mul_crt(a: &[u64], b: &[u64]) -> Vec<u64> {
    mul_pool(a, b, &mut WorkScratch::new())
}

fn mul_pool(a: &[u64], b: &[u64], pool: &mut WorkScratch) -> Vec<u64> {
    let mut ea = pool.take();
    expand_into(&mut ea, a);
    let mut eb = pool.take();
    expand_into(&mut eb, b);
    let conv = ntt::crt_convolution(&ea, &eb);
    pool.give(ea);
    pool.give(eb);
    let mut carried = pool.take();
    carry_into(&mut carried, &conv);
    pool.give(conv);
    let mut out = pool.take();
    fold_into(&mut out, &carried);
    pool.give(carried);
    normalize_limbs(&mut out);
    out
}

pub fn mul_buf(a: &[u64], b: &[u64], work: &mut Vec<u64>) -> Vec<u64> {
    if a.len().min(b.len()) <= SSA_CUTOFF {
        return toom::mul_buf(a, b, work);
    }
    if backend_for(a.len().min(b.len())) != "crt" {
        return crate::small_ntt::crt_convolution_small(a, b);
    }
    let mut pool = WorkScratch::new();
    pool.give(std::mem::take(work));
    let out = mul_pool(a, b, &mut pool);
    *work = pool.take();
    out
}

pub fn predicted_work_limbs(n: usize) -> f64 {
    let width = n as f64;
    let log = width.ln().max(1.0);
    width * log * log.ln().max(1.0)
}

fn expand_into(out: &mut Vec<u64>, limbs: &[u64]) {
    out.clear();
    out.reserve(limbs.len() * 4);
    for limb in limbs {
        out.push(limb & 0xFFFF);
        out.push((limb >> 16) & 0xFFFF);
        out.push((limb >> 32) & 0xFFFF);
        out.push((limb >> 48) & 0xFFFF);
    }
}

fn carry_into(out: &mut Vec<u64>, digits: &[u64]) {
    out.clear();
    out.reserve(digits.len() + 4);
    let mut carry: u128 = 0;
    let mut i = 0;
    while i < digits.len() || carry > 0 {
        let cur = u128::from(*digits.get(i).unwrap_or(&0)) + carry;
        out.push((cur & 0xFFFF) as u64);
        carry = cur >> 16;
        i += 1;
    }
}

fn fold_into(out: &mut Vec<u64>, carried: &[u64]) {
    out.clear();
    out.reserve(carried.len().div_ceil(4));
    let mut i = 0;
    while i < carried.len() {
        let mut limb = 0u64;
        for k in 0..4 {
            limb |= (*carried.get(i + k).unwrap_or(&0)) << (16 * k);
        }
        out.push(limb);
        i += 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schoolbook;
    use crate::verify::verify_mod;
    use rand::Rng;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn random_limbs(rng: &mut ChaCha8Rng, len: usize) -> Vec<u64> {
        (0..len).map(|_| rng.next_u64()).collect()
    }

    #[test]
    fn precancelled_small_is_cancelled() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c72);
        let a = random_limbs(&mut rng, 600);
        let b = random_limbs(&mut rng, 600);
        let flag = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            mul_small_cancelled(&a, &b, &flag),
            Err(crate::MultiplyError::Cancelled(_))
        ));
        assert!(matches!(
            mul_cancelled(&a, &b, &flag),
            Err(crate::MultiplyError::Cancelled(_))
        ));
        let clear = std::sync::atomic::AtomicBool::new(false);
        let result = mul_small_cancelled(&a, &b, &clear).expect("runs");
        assert_eq!(result, mul_small(&a, &b));
        assert!(verify_mod(&a, &b, &result));
    }

    #[test]
    fn large_cross_validation_vs_schoolbook() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c24);
        for len in [512usize, 1024, 2048] {
            let a = random_limbs(&mut rng, len);
            let b = random_limbs(&mut rng, len);
            let result = mul(&a, &b);
            assert_eq!(result, schoolbook::mul(&a, &b));
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn asymmetric_large_cross_validation() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c25);
        for (la, lb) in [(300usize, 500usize), (2048, 300)] {
            let a = random_limbs(&mut rng, la);
            let b = random_limbs(&mut rng, lb);
            assert_eq!(mul(&a, &b), schoolbook::mul(&a, &b));
        }
    }

    #[test]
    fn below_cutoff_delegates_to_toom() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c26);
        for len in [10usize, 200, SSA_CUTOFF] {
            let a = random_limbs(&mut rng, len);
            let b = random_limbs(&mut rng, len + 3);
            assert_eq!(mul(&a, &b), toom::mul(&a, &b));
        }
    }

    #[test]
    fn matches_schoolbook_below_cutoff() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c28);
        let a = random_limbs(&mut rng, SSA_CUTOFF - 1);
        let b = random_limbs(&mut rng, SSA_CUTOFF - 1);
        assert_eq!(mul(&a, &b), schoolbook::mul(&a, &b));
    }

    #[test]
    fn carry_chain_all_max() {
        let limbs = vec![u64::MAX; 300];
        let result = mul(&limbs, &limbs);
        assert_eq!(result, schoolbook::mul(&limbs, &limbs));
        assert!(verify_mod(&limbs, &limbs, &result));
    }

    #[test]
    fn zero_and_one_identities() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c27);
        let big = random_limbs(&mut rng, 300);
        assert_eq!(mul(&vec![0; 300], &big), vec![0]);
        assert_eq!(mul(&big, &vec![0; 300]), vec![0]);
        assert_eq!(mul(&[1], &big), big);
        assert_eq!(mul(&big, &[1]), big);
        assert_eq!(mul(&[], &big), vec![0]);
    }

    #[test]
    fn predicted_work_matches_ssa_shape() {
        assert_eq!(predicted_work_limbs(0), 0.0);
        assert_eq!(predicted_work_limbs(1), 1.0);
        let width = 1024f64;
        let expected = width * width.ln() * width.ln().ln();
        assert_eq!(predicted_work_limbs(1024), expected);
        assert!(expected > 1024.0 && expected < 1024.0 * 1024.0);
    }

    #[test]
    fn scratch_buffer_matches_fresh_allocation() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c28);
        let a = random_limbs(&mut rng, 300);
        let b = random_limbs(&mut rng, 300);
        let mut work = vec![0xDEAD_BEEF_DEAD_BEEFu64; 16];
        assert_eq!(mul_buf(&a, &b, &mut work), mul(&a, &b));
    }

    #[test]
    fn buf_matches_mul_across_cutoff() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c33);
        for _ in 0..30 {
            let la = rng.random_range(0..=1500);
            let lb = rng.random_range(0..=1500);
            let a = random_limbs(&mut rng, la);
            let b = random_limbs(&mut rng, lb);
            let expected = mul(&a, &b);
            let mut work = Vec::new();
            assert_eq!(mul_buf(&a, &b, &mut work), expected);
            assert!(verify_mod(&a, &b, &expected));
        }
    }

    #[test]
    fn backend_for_routes_at_entry_size() {
        assert_eq!(backend_for(0), "crt");
        assert_eq!(backend_for(SMALL_NTT_CUTOFF - 1), "crt");
        assert_eq!(backend_for(SMALL_NTT_CUTOFF), "small");
        assert_eq!(backend_for(SMALL_NTT_CUTOFF + 1), "small");
        assert_eq!(backend_for(RECURSION_CUTOFF), "small");
        assert_eq!(backend_for(RECURSION_CUTOFF + 1), "small-flat");
        assert_eq!(backend_for(16384), "small-flat");
    }

    #[test]
    fn mul_routes_flat_above_recursion_cutoff() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c39);
        let a = random_limbs(&mut rng, RECURSION_CUTOFF + 512);
        let b = random_limbs(&mut rng, RECURSION_CUTOFF + 512);
        let expected = mul_small(&a, &b);
        assert_eq!(mul(&a, &b), expected);
        let mut work = Vec::new();
        assert_eq!(mul_buf(&a, &b, &mut work), expected);
        assert!(verify_mod(&a, &b, &expected));
    }

    #[test]
    fn small_backend_matches_crt_backend() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c34);
        for _ in 0..30 {
            let la = rng.random_range(64..=4096);
            let lb = rng.random_range(64..=4096);
            let a = random_limbs(&mut rng, la);
            let b = random_limbs(&mut rng, lb);
            let expected = mul_crt(&a, &b);
            assert_eq!(mul_small(&a, &b), expected);
            assert!(verify_mod(&a, &b, &expected));
        }
    }

    #[test]
    fn recursive_matches_flat_small() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c35);
        for _ in 0..20 {
            let la = rng.random_range(100..=1500);
            let lb = rng.random_range(100..=1500);
            let a = random_limbs(&mut rng, la);
            let b = random_limbs(&mut rng, lb);
            let expected = mul_small(&a, &b);
            assert_eq!(mul_recursive(&a, &b), expected);
            assert!(verify_mod(&a, &b, &expected));
        }
    }

    #[test]
    fn recursive_depth_cap_degenerates() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c36);
        for len in [100usize, 900] {
            let a = random_limbs(&mut rng, len);
            let b = random_limbs(&mut rng, len + 7);
            let flat = mul_small(&a, &b);
            assert_eq!(rec_depth(&a, &b, MAX_DEPTH), flat);
            assert_eq!(rec_depth(&a, &b, MAX_DEPTH + 1), flat);
        }
    }

    #[test]
    fn recursive_matches_schoolbook_8k() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c37);
        let a = random_limbs(&mut rng, 8192);
        let b = random_limbs(&mut rng, 8192);
        let result = mul_recursive(&a, &b);
        assert_eq!(result, schoolbook::mul(&a, &b));
        assert!(verify_mod(&a, &b, &result));
    }

    #[test]
    fn recursive_matches_flat_16k() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c38);
        let a = random_limbs(&mut rng, 16384);
        let b = random_limbs(&mut rng, 16384);
        let expected = mul_small(&a, &b);
        assert_eq!(mul_recursive(&a, &b), expected);
        assert!(verify_mod(&a, &b, &expected));
    }
}
