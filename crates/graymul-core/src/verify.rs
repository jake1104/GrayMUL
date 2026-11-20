use crate::limb;
use num_bigint::BigUint;

const MODULI: [u64; 3] = [1_000_000_007, 1_000_000_009, 2_305_843_009_213_693_951];

fn mod_limbs(limbs: &[u64], modulus: u64) -> u64 {
    let mut acc: u128 = 0;
    for limb in limbs.iter().rev() {
        acc = ((acc << 64) | u128::from(*limb)) % u128::from(modulus);
    }
    acc as u64
}

pub fn verify_mod(a: &[u64], b: &[u64], product: &[u64]) -> bool {
    if limb::is_zero(a) || limb::is_zero(b) {
        return limb::is_zero(product);
    }
    if product.is_empty() {
        return false;
    }
    if product.len() > a.len() + b.len() {
        return false;
    }
    for modulus in MODULI {
        let lhs = (u128::from(mod_limbs(a, modulus)) * u128::from(mod_limbs(b, modulus)))
            % u128::from(modulus);
        if lhs as u64 != mod_limbs(product, modulus) {
            return false;
        }
    }
    true
}

pub fn verify_bigint(a: &BigUint, b: &BigUint, product: &BigUint) -> bool {
    verify_mod(
        &limb::from_bigint(a),
        &limb::from_bigint(b),
        &limb::from_bigint(product),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_correct_product() {
        let a = limb::from_bigint(&BigUint::from(123456789u64));
        let b = limb::from_bigint(&BigUint::from(987654321u64));
        let p = limb::from_bigint(&(BigUint::from(123456789u64) * BigUint::from(987654321u64)));
        assert!(verify_mod(&a, &b, &p));
    }

    #[test]
    fn rejects_wrong_product() {
        let a = limb::from_bigint(&BigUint::from(123456789u64));
        let b = limb::from_bigint(&BigUint::from(987654321u64));
        let mut p = limb::from_bigint(&(BigUint::from(123456789u64) * BigUint::from(987654321u64)));
        p[0] = p[0].wrapping_add(1);
        assert!(!verify_mod(&a, &b, &p));
    }

    #[test]
    fn zero_product_rule() {
        assert!(verify_mod(&[0], &[5], &[0]));
        assert!(!verify_mod(&[0], &[5], &[1]));
    }

    #[test]
    fn rejects_overlong_product() {
        let m = BigUint::from(1_000_000_007u64) * BigUint::from(1_000_000_009u64);
        let a = limb::from_bigint(&BigUint::from(2u32));
        let b = limb::from_bigint(&BigUint::from(2u32));
        let m2 = &m * &m;
        let bad_value = BigUint::from(4u32) + &m2 * &m;
        let bad = limb::from_bigint(&bad_value);
        assert!(bad.len() > a.len() + b.len());
        for modulus in [1_000_000_007u64, 1_000_000_009u64] {
            let lhs = (u128::from(mod_limbs(&a, modulus)) * u128::from(mod_limbs(&b, modulus)))
                % u128::from(modulus);
            assert_eq!(lhs as u64, mod_limbs(&bad, modulus));
        }
        assert!(!verify_mod(&a, &b, &bad));
    }

    #[test]
    fn mersenne_catches_old_moduli_blind_spot() {
        let m12 = BigUint::from(1_000_000_007u64) * BigUint::from(1_000_000_009u64);
        let m61 = 2_305_843_009_213_693_951u64;
        let a = limb::from_bigint(&BigUint::from(123456789u32));
        let b = limb::from_bigint(&BigUint::from(987654321u32));
        let p = BigUint::from(123456789u32) * BigUint::from(987654321u32);
        let mut found = None;
        for k in 1u32..=16 {
            let cand_value = &p + &m12 * BigUint::from(k);
            let cand = limb::from_bigint(&cand_value);
            if cand.len() > a.len() + b.len() {
                continue;
            }
            let old_ok = [1_000_000_007u64, 1_000_000_009u64].iter().all(|modulus| {
                let lhs = (u128::from(mod_limbs(&a, *modulus))
                    * u128::from(mod_limbs(&b, *modulus)))
                    % u128::from(*modulus);
                lhs as u64 == mod_limbs(&cand, *modulus)
            });
            let lhs61 =
                (u128::from(mod_limbs(&a, m61)) * u128::from(mod_limbs(&b, m61))) % u128::from(m61);
            if old_ok && lhs61 as u64 != mod_limbs(&cand, m61) {
                found = Some(cand);
                break;
            }
        }
        let tampered = found.expect("search must find a tamper invisible to the old moduli");
        assert!(!verify_mod(&a, &b, &tampered));
    }

    #[test]
    fn mod_limbs_matches_bigint_for_m61() {
        let m61 = 2_305_843_009_213_693_951u64;
        let m61_big = BigUint::from(m61);
        let mut x = 0x9E3779B97F4A7C15u64;
        for len in 0usize..8 {
            let mut limbs = Vec::new();
            for _ in 0..len {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                limbs.push(x);
            }
            if len == 3 {
                limbs[1] = u64::MAX;
            }
            let expected = limb::to_bigint(&limbs) % &m61_big;
            assert_eq!(BigUint::from(mod_limbs(&limbs, m61)), expected);
        }
    }
}
