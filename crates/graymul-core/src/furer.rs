use crate::limb::normalize_limbs;
use crate::schonhage;

pub const FURER_CUTOFF: usize = 2048;
pub const FURER_TRUNC_CUTOFF: usize = 4096;
pub const CHUNKS: usize = 8;

pub fn mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.len().min(b.len()) <= FURER_CUTOFF {
        return schonhage::mul(a, b);
    }
    if a.len().min(b.len()) <= FURER_TRUNC_CUTOFF {
        return mul_work(a, b, &mut Vec::new());
    }
    mul_trunc(a, b)
}

fn mul_work(a: &[u64], b: &[u64], work: &mut Vec<u64>) -> Vec<u64> {
    if a.len().min(b.len()) <= FURER_CUTOFF {
        return schonhage::mul_buf(a, b, work);
    }
    let size = a.len().max(b.len()).div_ceil(CHUNKS);
    let ac = chunk_ranges(a, size);
    let bc = chunk_ranges(b, size);
    let mut out = vec![0u64];
    for (i, (astart, aend)) in ac.iter().enumerate() {
        for (j, (bstart, bend)) in bc.iter().enumerate() {
            let part = schonhage::mul_buf(&a[*astart..*aend], &b[*bstart..*bend], work);
            add_shifted(&mut out, &part, (i + j) * size);
        }
    }
    normalize_limbs(&mut out);
    out
}

pub fn mul_buf(a: &[u64], b: &[u64], work: &mut Vec<u64>) -> Vec<u64> {
    if a.len().min(b.len()) <= FURER_CUTOFF {
        return schonhage::mul_buf(a, b, work);
    }
    if a.len().min(b.len()) <= FURER_TRUNC_CUTOFF {
        return mul_work(a, b, work);
    }
    mul_trunc(a, b)
}

pub fn mul_trunc(a: &[u64], b: &[u64]) -> Vec<u64> {
    let size = a.len().max(b.len()).div_ceil(CHUNKS);
    let ac = chunk_ranges(a, size);
    let bc = chunk_ranges(b, size);
    let mut out = vec![0u64];
    for (i, (astart, aend)) in ac.iter().enumerate() {
        for (j, (bstart, bend)) in bc.iter().enumerate() {
            let part = chunk_product_trunc(&a[*astart..*aend], &b[*bstart..*bend]);
            add_shifted(&mut out, &part, (i + j) * size);
        }
    }
    normalize_limbs(&mut out);
    out
}

fn chunk_product_trunc(x: &[u64], y: &[u64]) -> Vec<u64> {
    if x.is_empty() || y.is_empty() {
        return vec![0];
    }
    let xd = expand32(x);
    let yd = expand32(y);
    let m = xd.len().max(yd.len());
    let n = crate::ntt::next_pow2((2 * m).saturating_sub(1).max(1));
    let signed = negacyclic_convolution_small(&xd, &yd, n);
    let mut carry: i128 = 0;
    let mut out32: Vec<u64> = Vec::with_capacity(signed.len() + 2);
    for value in signed {
        let cur = value + carry;
        out32.push((cur & 0xFFFF_FFFF) as u64);
        carry = cur >> 32;
    }
    while carry > 0 {
        out32.push((carry & 0xFFFF_FFFF) as u64);
        carry >>= 32;
    }
    debug_assert!(carry == 0);
    let mut out = Vec::with_capacity(out32.len().div_ceil(2));
    let mut i = 0;
    while i < out32.len() {
        let lo = out32[i];
        let hi = if i + 1 < out32.len() { out32[i + 1] } else { 0 };
        out.push(lo | (hi << 32));
        i += 2;
    }
    normalize_limbs(&mut out);
    out
}

pub fn negacyclic_convolution_small(a: &[u64], b: &[u64], n: usize) -> Vec<i128> {
    if a.is_empty() || b.is_empty() {
        return vec![0; n.max(1)];
    }
    let r0 = crate::small_ntt::convolution_f64_full(a, b, crate::small_ntt::P0);
    let r1 = crate::small_ntt::convolution_f64_full(a, b, crate::small_ntt::P1);
    let need = a.len() + b.len() - 1;
    let digits = crt_combine_nocarry(&r0, &r1, need);
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let low = if k < need { digits[k] as i128 } else { 0 };
        let high = if k + n < need {
            digits[k + n] as i128
        } else {
            0
        };
        out.push(low - high);
    }
    out
}

fn crt_combine_nocarry(r0: &[u64], r1: &[u64], need: usize) -> Vec<u128> {
    let p0 = crate::small_ntt::P0 as u128;
    let p1 = crate::small_ntt::P1 as u128;
    let inv_p0 = crate::ntt::mod_pow(
        crate::small_ntt::P0 % crate::small_ntt::P1,
        crate::small_ntt::P1 - 2,
        crate::small_ntt::P1,
    ) as u128;
    let mut out = Vec::with_capacity(need);
    for i in 0..need {
        let x0 = r0[i] as u128;
        let x1 = r1[i] as u128;
        let diff = (x1 + p1 - x0 % p1) % p1;
        let t = diff * inv_p0 % p1;
        out.push(x0 + p0 * t);
    }
    out
}

fn expand32(limbs: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(limbs.len() * 2);
    for limb in limbs {
        out.push(limb & 0xFFFF_FFFF);
        out.push(limb >> 32);
    }
    out
}

pub fn predicted_work_limbs(n: usize) -> f64 {
    let width = n as f64;
    width * width.ln().max(1.0)
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
    fn depth_zero_degenerates_to_ssa() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c29);
        for len in [10usize, 500, FURER_CUTOFF] {
            let a = random_limbs(&mut rng, len);
            let b = random_limbs(&mut rng, len + 7);
            assert_eq!(mul(&a, &b), schonhage::mul(&a, &b));
        }
    }

    #[test]
    fn ssa_identity_above_cutoff() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c30);
        for (la, lb) in [(2100usize, 2100usize), (2500, 2200), (3000, 3000)] {
            let a = random_limbs(&mut rng, la);
            let b = random_limbs(&mut rng, lb);
            let result = mul(&a, &b);
            assert_eq!(result, schonhage::mul(&a, &b));
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn matches_schoolbook_below_cutoff() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c33);
        let a = random_limbs(&mut rng, FURER_CUTOFF - 1);
        let b = random_limbs(&mut rng, FURER_CUTOFF - 1);
        assert_eq!(mul(&a, &b), schoolbook::mul(&a, &b));
    }

    #[test]
    fn large_spot_matches_schoolbook() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c31);
        let a = random_limbs(&mut rng, 1200);
        let b = random_limbs(&mut rng, 1200);
        assert_eq!(mul(&a, &b), schoolbook::mul(&a, &b));
    }

    #[test]
    fn buf_matches_mul_across_cutoff() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c35);
        for _ in 0..30 {
            let la = rng.random_range(0..=2500);
            let lb = rng.random_range(0..=2500);
            let a = random_limbs(&mut rng, la);
            let b = random_limbs(&mut rng, lb);
            let expected = mul(&a, &b);
            let mut work = Vec::new();
            assert_eq!(mul_buf(&a, &b, &mut work), expected);
            assert!(verify_mod(&a, &b, &expected));
        }
    }

    #[test]
    fn randomized_cross_validation_vs_ssa() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c32);
        for _ in 0..200 {
            let la = rng.random_range(0..=1500);
            let lb = rng.random_range(0..=1500);
            let a = random_limbs(&mut rng, la);
            let b = random_limbs(&mut rng, lb);
            let result = mul(&a, &b);
            assert_eq!(result, schonhage::mul(&a, &b));
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn zero_and_one_identities() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c33);
        let big = random_limbs(&mut rng, 1200);
        assert_eq!(mul(&vec![0; 1200], &big), vec![0]);
        assert_eq!(mul(&big, &vec![0; 1200]), vec![0]);
        assert_eq!(mul(&[1], &big), big);
        assert_eq!(mul(&big, &[1]), big);
        assert_eq!(mul(&[], &big), vec![0]);
    }

    #[test]
    fn predicted_work_matches_furer_shape() {
        assert_eq!(predicted_work_limbs(0), 0.0);
        assert_eq!(predicted_work_limbs(1), 1.0);
        let width = 4096f64;
        let expected = width * width.ln();
        assert_eq!(predicted_work_limbs(4096), expected);
        assert!(expected > 4096.0 && expected < 4096.0 * 4096.0);
    }

    #[test]
    fn scratch_buffer_matches_fresh_allocation() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c34);
        for len in [100usize, 1200] {
            let a = random_limbs(&mut rng, len);
            let b = random_limbs(&mut rng, len);
            let mut work = vec![0xDEAD_BEEF_DEAD_BEEFu64; 16];
            assert_eq!(mul_buf(&a, &b, &mut work), mul(&a, &b));
        }
    }

    fn schoolbook_digits(a: &[u64], b: &[u64]) -> Vec<i128> {
        if a.is_empty() || b.is_empty() {
            return vec![0];
        }
        let mut out = vec![0i128; a.len() + b.len() - 1];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                out[i + j] += (*x as i128) * (*y as i128);
            }
        }
        out
    }

    #[test]
    fn negacyclic_fold_identity_at_degenerate_length() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c40);
        for _ in 0..60 {
            let la = rng.random_range(1..=300);
            let lb = rng.random_range(1..=300);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64() % 65536).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64() % 65536).collect();
            let need = la + lb - 1;
            let mut n = 1usize;
            while n < need {
                n *= 2;
            }
            let folded = negacyclic_convolution_small(&a, &b, n);
            let linear = schoolbook_digits(&a, &b);
            assert_eq!(folded.len(), n);
            for (k, value) in linear.iter().enumerate() {
                assert_eq!(folded[k], *value);
            }
            for value in folded.iter().skip(need) {
                assert_eq!(*value, 0);
            }
        }
    }

    #[test]
    fn negacyclic_wrap_matches_folded_reference() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c41);
        for _ in 0..40 {
            let n = 1usize << rng.random_range(2..=4);
            let la = rng.random_range(1..=2 * n);
            let lb = rng.random_range(1..=2 * n);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64() % 65536).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64() % 65536).collect();
            let linear = schoolbook_digits(&a, &b);
            let need = la + lb - 1;
            let folded = negacyclic_convolution_small(&a, &b, n);
            assert_eq!(folded.len(), n);
            for k in 0..n {
                let low = if k < need { linear[k] } else { 0 };
                let high = if k + n < need { linear[k + n] } else { 0 };
                assert_eq!(folded[k], low - high);
            }
        }
    }

    #[test]
    fn negacyclic_operating_point_matches_reference() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c42);
        for _ in 0..20 {
            let la = rng.random_range(900..=1024);
            let lb = rng.random_range(900..=1024);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64() % 65536).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64() % 65536).collect();
            let folded = negacyclic_convolution_small(&a, &b, 2048);
            let linear = schoolbook_digits(&a, &b);
            for (k, value) in linear.iter().enumerate() {
                assert_eq!(folded[k], *value);
            }
        }
    }

    #[test]
    fn trunc_matches_schoolbook_to_400_limbs() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c43);
        for _ in 0..200 {
            let la = rng.random_range(0..=400);
            let lb = rng.random_range(0..=400);
            let a = random_limbs(&mut rng, la);
            let b = random_limbs(&mut rng, lb);
            let result = mul_trunc(&a, &b);
            assert_eq!(result, schoolbook::mul(&a, &b));
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn trunc_matches_ssa_at_routed_sizes() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c44);
        for len in [4096usize, 5000, 6000, 8192] {
            for _ in 0..2 {
                let a = random_limbs(&mut rng, len);
                let b = random_limbs(&mut rng, len);
                let result = mul_trunc(&a, &b);
                assert_eq!(result, schonhage::mul(&a, &b));
                assert!(verify_mod(&a, &b, &result));
            }
        }
    }

    #[test]
    fn trunc_matches_old_chunk_ring() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c45);
        for len in [4096usize, 5000, 6000, 8192] {
            for _ in 0..2 {
                let a = random_limbs(&mut rng, len);
                let b = random_limbs(&mut rng, len);
                let mut work = Vec::new();
                assert_eq!(mul_trunc(&a, &b), mul_work(&a, &b, &mut work));
            }
        }
    }

    #[test]
    fn routing_pins_trunc_band() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c46);
        for len in [4097usize, 5000, 8192] {
            let a = random_limbs(&mut rng, len);
            let b = random_limbs(&mut rng, len);
            assert_eq!(mul(&a, &b), mul_trunc(&a, &b));
            let mut work = Vec::new();
            assert_eq!(mul_buf(&a, &b, &mut work), mul_trunc(&a, &b));
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c47);
        let a = random_limbs(&mut rng, 3000);
        let b = random_limbs(&mut rng, 3000);
        let mut work = Vec::new();
        assert_eq!(mul(&a, &b), mul_work(&a, &b, &mut work));
    }
}
