use num_bigint::BigUint;
use num_traits::{One, Zero};

pub fn from_bigint(value: &BigUint) -> Vec<u64> {
    if value.is_zero() {
        return vec![0];
    }
    let bytes = value.to_bytes_le();
    let mut limbs = Vec::with_capacity(bytes.len().div_ceil(8));
    for chunk in bytes.chunks(8) {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        limbs.push(u64::from_le_bytes(word));
    }
    normalize_limbs(&mut limbs);
    limbs
}

pub fn to_bigint(limbs: &[u64]) -> BigUint {
    let mut bytes = Vec::with_capacity(limbs.len() * 8);
    for limb in limbs {
        bytes.extend_from_slice(&limb.to_le_bytes());
    }
    BigUint::from_bytes_le(&bytes)
}

pub fn normalize_limbs(limbs: &mut Vec<u64>) {
    while limbs.len() > 1 && limbs.last() == Some(&0) {
        limbs.pop();
    }
    if limbs.is_empty() {
        limbs.push(0);
    }
}

pub fn limb_bits(limbs: &[u64]) -> u64 {
    let mut top = limbs.len();
    while top > 0 && limbs[top - 1] == 0 {
        top -= 1;
    }
    if top == 0 {
        return 0;
    }
    (top as u64 - 1) * 64 + (64 - limbs[top - 1].leading_zeros() as u64)
}

pub fn is_zero(limbs: &[u64]) -> bool {
    limbs.iter().all(|limb| *limb == 0)
}

pub fn is_one(limbs: &[u64]) -> bool {
    !limbs.is_empty() && limbs[0] == 1 && limbs[1..].iter().all(|limb| *limb == 0)
}

pub fn one() -> BigUint {
    BigUint::one()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_round_trip() {
        let limbs = from_bigint(&BigUint::zero());
        assert_eq!(limbs, vec![0]);
        assert_eq!(to_bigint(&limbs), BigUint::zero());
        assert!(is_zero(&limbs));
    }

    #[test]
    fn carry_chain_round_trip() {
        let max = BigUint::from(u64::MAX);
        let limbs = from_bigint(&max);
        assert_eq!(limbs, vec![u64::MAX]);
        assert_eq!(to_bigint(&limbs), max);
        assert_eq!(limb_bits(&limbs), 64);
    }

    #[test]
    fn multi_limb_round_trip() {
        let value = (BigUint::one() << 130) + BigUint::from(12345u32);
        let limbs = from_bigint(&value);
        assert_eq!(limbs.len(), 3);
        assert_eq!(to_bigint(&limbs), value);
        assert_eq!(limb_bits(&limbs), 131);
    }

    #[test]
    fn one_detection() {
        assert!(is_one(&[1]));
        assert!(is_one(&[1, 0, 0]));
        assert!(!is_one(&[2]));
        assert!(!is_one(&[0]));
    }
}
