use crate::limb;
use num_bigint::BigUint;

const MODULI: [u64; 2] = [1_000_000_007, 1_000_000_009];

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
}
