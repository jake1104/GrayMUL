use crate::limb::normalize_limbs;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_addcarry_u64;

pub fn mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    mul_scalar(a, b)
}

fn mul_scalar(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::new();
    accumulate_scalar(a, b, &mut out);
    out
}

pub fn measure_scalar(a: &[u64], b: &[u64]) -> Vec<u64> {
    mul_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
pub fn measure_adx(a: &[u64], b: &[u64]) -> Vec<u64> {
    mul_adx_safe(a, b)
}

pub(crate) fn mul_cancelled(
    a: &[u64],
    b: &[u64],
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    use std::sync::atomic::Ordering;
    let mut out = Vec::new();
    if a.is_empty() || b.is_empty() {
        out.push(0);
        return Ok(out);
    }
    out.clear();
    out.resize(a.len() + b.len(), 0);
    for (i, x) in a.iter().enumerate() {
        if cancelled.load(Ordering::Relaxed) {
            return Err(crate::MultiplyError::cancelled());
        }
        let mut carry: u128 = 0;
        for (j, y) in b.iter().enumerate() {
            let slot = &mut out[i + j];
            let cur = u128::from(*x) * u128::from(*y) + u128::from(*slot) + carry;
            *slot = cur as u64;
            carry = cur >> 64;
        }
        let mut k = i + b.len();
        while carry > 0 {
            let cur = u128::from(out[k]) + carry;
            out[k] = cur as u64;
            carry = cur >> 64;
            k += 1;
        }
    }
    normalize_limbs(&mut out);
    Ok(out)
}

pub fn mul_buf(a: &[u64], b: &[u64], work: &mut Vec<u64>) -> Vec<u64> {
    accumulate_scalar(a, b, work);
    work.clone()
}

pub fn predicted_work_limbs(n: usize) -> f64 {
    (n as f64).powi(2)
}

fn accumulate_scalar(a: &[u64], b: &[u64], out: &mut Vec<u64>) {
    if a.is_empty() || b.is_empty() {
        out.clear();
        out.push(0);
        return;
    }
    out.clear();
    out.resize(a.len() + b.len(), 0);
    accumulate_scalar_inner(a, b, out);
    normalize_limbs(out);
}

fn accumulate_scalar_inner(a: &[u64], b: &[u64], out: &mut [u64]) {
    for (i, x) in a.iter().enumerate() {
        let mut carry: u128 = 0;
        for (j, y) in b.iter().enumerate() {
            let slot = &mut out[i + j];
            let cur = u128::from(*x) * u128::from(*y) + u128::from(*slot) + carry;
            *slot = cur as u64;
            carry = cur >> 64;
        }
        let mut k = i + b.len();
        while carry > 0 {
            let cur = u128::from(out[k]) + carry;
            out[k] = cur as u64;
            carry = cur >> 64;
            k += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn mul_adx_safe(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::new();
    if a.is_empty() || b.is_empty() {
        out.push(0);
        return out;
    }
    out.resize(a.len() + b.len(), 0);
    let use_asm =
        std::arch::is_x86_feature_detected!("adx") && std::arch::is_x86_feature_detected!("bmi2");
    unsafe {
        if use_asm {
            accumulate_adx_asm(a, b, &mut out);
        } else {
            accumulate_adx(a, b, &mut out);
        }
    }
    normalize_limbs(&mut out);
    out
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(unused_assignments)]
unsafe fn accumulate_adx_asm(a: &[u64], b: &[u64], out: &mut [u64]) {
    let m = b.len();
    for (i, x) in a.iter().enumerate() {
        let main = m & !3;
        let mut clo: u8 = 0;
        let mut chi: u8 = 0;
        if main > 0 {
            let mut b_ptr = b.as_ptr();
            let mut o_ptr = out.as_mut_ptr().add(i);
            let mut groups = (main / 4) as u64;
            let mut seed: u64 = 0;
            core::arch::asm!(
                "xor {seed}, {seed}",
                "2:",
                "mulx {h}, {l}, qword ptr [{b}]",
                "mov {t}, qword ptr [{o}]",
                "adcx {t}, {l}",
                "mov qword ptr [{o}], {t}",
                "mov {t}, qword ptr [{o} + 8]",
                "adox {t}, {h}",
                "mov qword ptr [{o} + 8], {t}",
                "mulx {h}, {l}, qword ptr [{b} + 8]",
                "mov {t}, qword ptr [{o} + 8]",
                "adcx {t}, {l}",
                "mov qword ptr [{o} + 8], {t}",
                "mov {t}, qword ptr [{o} + 16]",
                "adox {t}, {h}",
                "mov qword ptr [{o} + 16], {t}",
                "mulx {h}, {l}, qword ptr [{b} + 16]",
                "mov {t}, qword ptr [{o} + 16]",
                "adcx {t}, {l}",
                "mov qword ptr [{o} + 16], {t}",
                "mov {t}, qword ptr [{o} + 24]",
                "adox {t}, {h}",
                "mov qword ptr [{o} + 24], {t}",
                "mulx {h}, {l}, qword ptr [{b} + 24]",
                "mov {t}, qword ptr [{o} + 24]",
                "adcx {t}, {l}",
                "mov qword ptr [{o} + 24], {t}",
                "mov {t}, qword ptr [{o} + 32]",
                "adox {t}, {h}",
                "mov qword ptr [{o} + 32], {t}",
                "lea {b}, [{b} + 32]",
                "lea {o}, [{o} + 32]",
                "lea rcx, [rcx - 1]",
                "jrcxz 3f",
                "jmp 2b",
                "3:",
                "setc {clo}",
                "seto {chi}",
                in("rdx") *x,
                b = inout(reg) b_ptr,
                o = inout(reg) o_ptr,
                inout("rcx") groups,
                seed = inout(reg) seed,
                t = out(reg) _,
                h = out(reg) _,
                l = out(reg) _,
                clo = lateout(reg_byte) clo,
                chi = lateout(reg_byte) chi,
            );
        }
        let mut c0 = clo as u64;
        let mut c1 = chi as u64;
        let mut j = main;
        while j < m {
            let prod = (*x as u128) * (b[j] as u128);
            let cur = out[i + j] as u128 + (prod as u64) as u128 + c0 as u128;
            out[i + j] = cur as u64;
            c0 = (cur >> 64) as u64;
            let cur = out[i + j + 1] as u128 + ((prod >> 64) as u64) as u128 + c1 as u128;
            out[i + j + 1] = cur as u64;
            c1 = (cur >> 64) as u64;
            j += 1;
        }
        let mut k = i + m;
        let mut carry = c0;
        while carry > 0 {
            let cur = out[k] as u128 + carry as u128;
            out[k] = cur as u64;
            carry = (cur >> 64) as u64;
            k += 1;
        }
        if c1 != 0 {
            let mut p = i + m + 1;
            let mut ripple = c1;
            while ripple > 0 {
                let cur = out[p] as u128 + ripple as u128;
                out[p] = cur as u64;
                ripple = (cur >> 64) as u64;
                p += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "adx")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn accumulate_adx(a: &[u64], b: &[u64], out: &mut [u64]) {
    let m = b.len();
    let ptr = out.as_mut_ptr();
    for (i, x) in a.iter().enumerate() {
        let mut carry_lo: u8 = 0;
        for (j, y) in b.iter().enumerate() {
            let prod = (*x as u128) * (*y as u128);
            carry_lo = _addcarry_u64(
                carry_lo,
                ptr.add(i + j).read(),
                prod as u64,
                &mut *ptr.add(i + j),
            );
        }
        let mut carry_hi: u8 = 0;
        for (j, y) in b.iter().enumerate() {
            let prod = (*x as u128) * (*y as u128);
            carry_hi = _addcarry_u64(
                carry_hi,
                ptr.add(i + j + 1).read(),
                (prod >> 64) as u64,
                &mut *ptr.add(i + j + 1),
            );
        }
        let mut k = i + m;
        let mut carry = carry_lo as u64;
        while carry > 0 {
            let cur = out[k] as u128 + carry as u128;
            out[k] = cur as u64;
            carry = (cur >> 64) as u64;
            k += 1;
        }
        if carry_hi != 0 {
            let mut p = i + m + 1;
            let mut ripple = carry_hi as u64;
            while ripple > 0 {
                let cur = out[p] as u128 + ripple as u128;
                out[p] = cur as u64;
                ripple = (cur >> 64) as u64;
                p += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limb::{from_bigint, to_bigint};
    use crate::verify::verify_mod;
    use num_bigint::BigUint;
    use num_traits::{One, Zero};
    use rand::Rng;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn oracle(a: &[u64], b: &[u64]) -> Vec<u64> {
        from_bigint(&(to_bigint(a) * to_bigint(b)))
    }

    #[test]
    fn zero_times_anything_is_zero() {
        assert_eq!(mul(&[0], &[5, 6, 7]), vec![0]);
        assert_eq!(mul(&[5, 6, 7], &[0]), vec![0]);
        assert_eq!(mul(&[], &[1, 2]), vec![0]);
        assert_eq!(mul(&[1, 2], &[]), vec![0]);
        assert_eq!(mul(&[], &[]), vec![0]);
        assert_eq!(mul(&[0, 0], &[0]), vec![0]);
    }

    #[test]
    fn one_is_identity() {
        let value = vec![9, 8, 7, 6, 5];
        assert_eq!(mul(&[1], &value), value);
        assert_eq!(mul(&value, &[1]), value);
        assert_eq!(mul(&[1], &[1]), vec![1]);
    }

    #[test]
    fn max_times_max_matches_oracle() {
        let result = mul(&[u64::MAX], &[u64::MAX]);
        let expected = oracle(&[u64::MAX], &[u64::MAX]);
        assert_eq!(result, expected);
        assert_eq!(result, vec![1, u64::MAX - 1]);
        assert_eq!(
            to_bigint(&result),
            BigUint::from(u64::MAX) * BigUint::from(u64::MAX)
        );
    }

    #[test]
    fn carry_chain_four_max_squared() {
        let limbs = vec![u64::MAX; 4];
        let result = mul(&limbs, &limbs);
        assert_eq!(result, oracle(&limbs, &limbs));
        assert!(verify_mod(&limbs, &limbs, &result));
    }

    #[test]
    fn asymmetric_one_by_eight() {
        let short = vec![0x123456789ABCDEFu64];
        let long = vec![3, 1, 4, 1, 5, 9, 2, 6];
        assert_eq!(mul(&short, &long), oracle(&short, &long));
        assert_eq!(mul(&long, &short), oracle(&long, &short));
    }

    #[test]
    fn commutativity_spot() {
        let pairs = [
            (vec![0u64], vec![0u64]),
            (vec![7u64], vec![11u64, 13u64]),
            (vec![u64::MAX, 1], vec![2, 3, u64::MAX]),
        ];
        for (a, b) in &pairs {
            assert_eq!(mul(a, b), mul(b, a));
        }
    }

    #[test]
    fn scratch_buffer_matches_fresh_allocation() {
        let cases = [
            (vec![0u64], vec![9u64, 9u64]),
            (vec![u64::MAX; 3], vec![u64::MAX; 5]),
            (vec![1u64, 2, 3], vec![4u64]),
        ];
        for (a, b) in &cases {
            let mut work = vec![0xDEAD_BEEF_DEAD_BEEFu64; 100];
            assert_eq!(mul_buf(a, b, &mut work), mul(a, b));
        }
    }

    #[test]
    fn predicted_work_is_quadratic() {
        assert_eq!(predicted_work_limbs(0), 0.0);
        assert_eq!(predicted_work_limbs(1), 1.0);
        assert_eq!(predicted_work_limbs(4), 16.0);
        assert_eq!(predicted_work_limbs(8), 64.0);
    }

    #[test]
    fn randomized_cross_validation_vs_bigint_oracle() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c15);
        for _ in 0..200 {
            let la = rng.random_range(0..=8);
            let lb = rng.random_range(0..=8);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64()).collect();
            let result = mul(&a, &b);
            assert_eq!(result, oracle(&a, &b));
            assert!(verify_mod(&a, &b, &result));
            let mut work = Vec::new();
            assert_eq!(mul_buf(&a, &b, &mut work), result);
        }
    }

    #[test]
    fn oracle_agrees_with_shift_identities() {
        let base = (BigUint::one() << 130) + BigUint::from(12345u32);
        let limbs = from_bigint(&base);
        let doubled = mul(&limbs, &[2]);
        assert_eq!(to_bigint(&doubled), &base + &base);
        assert!(!to_bigint(&doubled).is_zero());
        assert_eq!(to_bigint(&mul(&limbs, &[0])), BigUint::zero());
    }

    #[test]
    fn precancelled_mul_is_cancelled() {
        let a = vec![7u64; 100];
        let b = vec![9u64; 100];
        let flag = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            mul_cancelled(&a, &b, &flag),
            Err(crate::MultiplyError::Cancelled(_))
        ));
        let clear = std::sync::atomic::AtomicBool::new(false);
        assert_eq!(mul_cancelled(&a, &b, &clear).expect("runs"), mul(&a, &b));
    }

    #[test]
    fn scalar_fallback_matches_oracle_vectors() {
        assert_eq!(mul_scalar(&[0], &[5, 6, 7]), vec![0]);
        assert_eq!(mul_scalar(&[], &[1, 2]), vec![0]);
        assert_eq!(mul_scalar(&[1], &[9, 8, 7]), vec![9, 8, 7]);
        assert_eq!(mul_scalar(&[u64::MAX], &[u64::MAX]), vec![1, u64::MAX - 1]);
        let limbs = vec![u64::MAX; 4];
        assert_eq!(mul_scalar(&limbs, &limbs), oracle(&limbs, &limbs));
    }

    #[test]
    fn adx_path_matches_scalar() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c16);
        for _ in 0..300 {
            let la = rng.random_range(0..=64);
            let lb = rng.random_range(0..=64);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64()).collect();
            let expected = oracle(&a, &b);
            assert_eq!(mul_scalar(&a, &b), expected);
            assert_eq!(mul(&a, &b), expected);
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("adx") {
                assert_eq!(mul_adx_safe(&a, &b), expected);
            }
        }
        for limbs in [128usize, 256, 400] {
            let a: Vec<u64> = (0..limbs)
                .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15))
                .collect();
            let b: Vec<u64> = (0..limbs)
                .map(|i| (i as u64).wrapping_mul(0xbf58476d1ce4e5b9))
                .collect();
            let expected = oracle(&a, &b);
            assert_eq!(mul_scalar(&a, &b), expected);
            assert_eq!(mul(&a, &b), expected);
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("adx") {
                assert_eq!(mul_adx_safe(&a, &b), expected);
            }
        }
    }

    #[test]
    fn adx_asm_matches_scalar_saturated() {
        for limbs in [1usize, 3, 4, 5, 7, 8, 9, 127, 128, 129, 256] {
            let a = vec![u64::MAX; limbs];
            let b = vec![u64::MAX; limbs];
            let expected = oracle(&a, &b);
            assert_eq!(mul_scalar(&a, &b), expected);
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("adx")
                && std::arch::is_x86_feature_detected!("bmi2")
            {
                assert_eq!(mul_adx_safe(&a, &b), expected);
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c18);
        for _ in 0..200 {
            let la = rng.random_range(1..=130);
            let lb = rng.random_range(1..=130);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64()).collect();
            #[cfg(target_arch = "x86_64")]
            assert_eq!(mul_adx_safe(&a, &b), mul_scalar(&a, &b));
            assert_eq!(mul(&a, &b), oracle(&a, &b));
        }
    }

    #[test]
    fn mul_routes_to_scalar() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c17);
        for _ in 0..20 {
            let la = rng.random_range(0..=600);
            let lb = rng.random_range(0..=600);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64()).collect();
            assert_eq!(mul(&a, &b), mul_scalar(&a, &b));
        }
    }

    #[test]
    fn adx_detection_report() {
        #[cfg(target_arch = "x86_64")]
        println!(
            "adx_detected={}",
            std::arch::is_x86_feature_detected!("adx")
        );
        #[cfg(not(target_arch = "x86_64"))]
        println!("adx_detected=false-non-x86_64");
    }
}
