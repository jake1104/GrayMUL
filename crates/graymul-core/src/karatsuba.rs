use crate::limb::normalize_limbs;
use crate::schoolbook;

pub use crate::scratch::WorkScratch;

pub const CUTOFF: usize = 1024;
const CORE_BASE: usize = 8;

pub fn mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    rec(a, b, &mut WorkScratch::new(), CUTOFF)
}

pub fn mul_core(a: &[u64], b: &[u64], scratch: &mut WorkScratch) -> Vec<u64> {
    rec(a, b, scratch, CORE_BASE)
}

pub(crate) fn mul_cancelled(
    a: &[u64],
    b: &[u64],
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    rec_cancelled(a, b, &mut WorkScratch::new(), CUTOFF, cancelled)
}

pub(crate) fn mul_scratch_cancelled(
    a: &[u64],
    b: &[u64],
    scratch: &mut WorkScratch,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    rec_cancelled(a, b, scratch, CUTOFF, cancelled)
}

pub fn mul_buf(a: &[u64], b: &[u64], work: &mut Vec<u64>) -> Vec<u64> {
    let mut scratch = WorkScratch::new();
    scratch.give(std::mem::take(work));
    let out = rec(a, b, &mut scratch, CUTOFF);
    *work = scratch.take();
    out
}

pub fn predicted_work_limbs(n: usize) -> f64 {
    (n as f64).powf(1.585)
}

fn rec(a: &[u64], b: &[u64], scratch: &mut WorkScratch, base: usize) -> Vec<u64> {
    if a.len().min(b.len()) <= base {
        return schoolbook::mul(a, b);
    }
    let m = a.len().max(b.len()) / 2;
    let (a0, a1) = a.split_at(m.min(a.len()));
    let (b0, b1) = b.split_at(m.min(b.len()));
    let mut sa = scratch.take();
    add_into(&mut sa, a0, a1);
    let mut sb = scratch.take();
    add_into(&mut sb, b0, b1);
    let mut z1 = rec(&sa, &sb, scratch, base);
    scratch.give(sa);
    scratch.give(sb);
    let z0 = rec(a0, b0, scratch, base);
    let z2 = rec(a1, b1, scratch, base);
    sub_limbs_assign(&mut z1, &z0);
    sub_limbs_assign(&mut z1, &z2);
    let mut out = z0;
    add_shifted(&mut out, &z1, m);
    add_shifted(&mut out, &z2, 2 * m);
    normalize_limbs(&mut out);
    out
}

fn rec_cancelled(
    a: &[u64],
    b: &[u64],
    scratch: &mut WorkScratch,
    base: usize,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(crate::MultiplyError::cancelled());
    }
    if a.len().min(b.len()) <= base {
        return Ok(schoolbook::mul(a, b));
    }
    let m = a.len().max(b.len()) / 2;
    let (a0, a1) = a.split_at(m.min(a.len()));
    let (b0, b1) = b.split_at(m.min(b.len()));
    let mut sa = scratch.take();
    add_into(&mut sa, a0, a1);
    let mut sb = scratch.take();
    add_into(&mut sb, b0, b1);
    let mut z1 = rec_cancelled(&sa, &sb, scratch, base, cancelled)?;
    scratch.give(sa);
    scratch.give(sb);
    let z0 = rec_cancelled(a0, b0, scratch, base, cancelled)?;
    let z2 = rec_cancelled(a1, b1, scratch, base, cancelled)?;
    sub_limbs_assign(&mut z1, &z0);
    sub_limbs_assign(&mut z1, &z2);
    let mut out = z0;
    add_shifted(&mut out, &z1, m);
    add_shifted(&mut out, &z2, 2 * m);
    normalize_limbs(&mut out);
    Ok(out)
}

fn add_into(out: &mut Vec<u64>, a: &[u64], b: &[u64]) {
    out.clear();
    let width = a.len().max(b.len());
    out.reserve(width + 1);
    let mut carry: u128 = 0;
    for i in 0..width {
        let x = u128::from(*a.get(i).unwrap_or(&0));
        let y = u128::from(*b.get(i).unwrap_or(&0));
        let cur = x + y + carry;
        out.push(cur as u64);
        carry = cur >> 64;
    }
    if carry > 0 {
        out.push(carry as u64);
    }
}

fn sub_limbs_assign(target: &mut Vec<u64>, other: &[u64]) {
    if target.len() < other.len() {
        target.resize(other.len(), 0);
    }
    let mut borrow: i128 = 0;
    for (i, slot) in target.iter_mut().enumerate() {
        let x = *slot as i128 - borrow;
        let y = other.get(i).copied().unwrap_or(0) as i128;
        let diff = x - y;
        if diff < 0 {
            *slot = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            *slot = diff as u64;
            borrow = 0;
        }
    }
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
    use crate::schoolbook;
    use crate::verify::verify_mod;
    use rand::Rng;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn pattern_limbs(len: usize, salt: u64) -> Vec<u64> {
        (0..len)
            .map(|i| {
                (i as u64)
                    .wrapping_mul(0x9e3779b97f4a7c15)
                    .wrapping_add(salt)
            })
            .collect()
    }

    #[test]
    fn cutoff_boundary_matches_schoolbook() {
        for len in [CUTOFF - 1, CUTOFF, CUTOFF + 1, CUTOFF + 2] {
            let a = pattern_limbs(len, 1);
            let b = pattern_limbs(len, 2);
            assert_eq!(mul(&a, &b), schoolbook::mul(&a, &b));
            let wide = pattern_limbs(len + 5, 3);
            assert_eq!(mul(&a, &wide), schoolbook::mul(&a, &wide));
            assert_eq!(mul(&wide, &a), schoolbook::mul(&wide, &a));
        }
    }

    #[test]
    fn delegates_to_schoolbook_below_cutoff() {
        let len = CUTOFF - 1;
        let a = pattern_limbs(len, 21);
        let b = pattern_limbs(len, 22);
        assert_eq!(mul(&a, &b), schoolbook::mul(&a, &b));
    }

    #[test]
    fn odd_length_splits_match_schoolbook() {
        let shapes = [
            (7usize, 9usize),
            (33, 33),
            (65, 5),
            (33, 64),
            (1, 100),
            (100, 1),
        ];
        for (la, lb) in shapes {
            let a = pattern_limbs(la, 11);
            let b = pattern_limbs(lb, 17);
            assert_eq!(mul(&a, &b), schoolbook::mul(&a, &b));
            assert!(verify_mod(&a, &b, &mul(&a, &b)));
        }
    }

    #[test]
    fn zero_and_one_identities() {
        let big = pattern_limbs(50, 5);
        assert_eq!(mul(&vec![0; 50], &big), vec![0]);
        assert_eq!(mul(&big, &vec![0; 50]), vec![0]);
        assert_eq!(mul(&[1], &big), big);
        assert_eq!(mul(&big, &[1]), big);
        assert_eq!(mul(&[], &big), vec![0]);
    }

    #[test]
    fn commutativity_spot() {
        let a = pattern_limbs(40, 21);
        let b = pattern_limbs(63, 22);
        assert_eq!(mul(&a, &b), mul(&b, &a));
    }

    #[test]
    fn predicted_work_is_subquadratic() {
        assert_eq!(predicted_work_limbs(0), 0.0);
        assert_eq!(predicted_work_limbs(1), 1.0);
        let mid = predicted_work_limbs(64);
        assert_eq!(mid, 64f64.powf(1.585));
        assert!(mid > 64.0 && mid < 4096.0);
    }

    #[test]
    fn precancelled_mul_is_cancelled() {
        let a = pattern_limbs(300, 31);
        let b = pattern_limbs(300, 32);
        let flag = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            mul_cancelled(&a, &b, &flag),
            Err(crate::MultiplyError::Cancelled(_))
        ));
        let clear = std::sync::atomic::AtomicBool::new(false);
        assert_eq!(mul_cancelled(&a, &b, &clear).expect("runs"), mul(&a, &b));
    }

    #[test]
    fn randomized_cross_validation_vs_schoolbook() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c16);
        for _ in 0..200 {
            let la = rng.random_range(0..=70);
            let lb = rng.random_range(0..=70);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64()).collect();
            let result = mul(&a, &b);
            assert_eq!(result, schoolbook::mul(&a, &b));
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn scratch_buffer_matches_fresh_allocation() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c17);
        for _ in 0..50 {
            let la = rng.random_range(0..=70);
            let lb = rng.random_range(0..=70);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64()).collect();
            let mut work = vec![0xDEAD_BEEF_DEAD_BEEFu64; 16];
            assert_eq!(mul_buf(&a, &b, &mut work), mul(&a, &b));
        }
    }

    #[test]
    fn core_buf_paths_match_dispatched_mul() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c18);
        let mut shapes = Vec::with_capacity(54);
        for _ in 0..50 {
            shapes.push((rng.random_range(0..=2500), rng.random_range(0..=2500)));
        }
        shapes.push((1025, 1025));
        shapes.push((2048, 2048));
        shapes.push((3000, 1500));
        shapes.push((4096, 4096));
        for (la, lb) in shapes {
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64()).collect();
            let expected = mul(&a, &b);
            let mut work = Vec::new();
            assert_eq!(mul_buf(&a, &b, &mut work), expected);
            let mut scratch = WorkScratch::new();
            assert_eq!(mul_core(&a, &b, &mut scratch), expected);
            assert!(verify_mod(&a, &b, &expected));
        }
    }
}
