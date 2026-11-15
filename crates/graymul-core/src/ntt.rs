use crate::limb::normalize_limbs;

pub const P1: u64 = 998_244_353;
pub const P2: u64 = 1_004_535_809;
pub const P1_ROOT: u64 = 3;
pub const P2_ROOT: u64 = 3;
pub const P1_MAX_POW: u32 = 23;
pub const P2_MAX_POW: u32 = 21;

pub fn mod_pow(base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut out: u128 = 1;
    let m = u128::from(modulus);
    let mut b = u128::from(base % modulus);
    while exp > 0 {
        if exp & 1 == 1 {
            out = out * b % m;
        }
        b = b * b % m;
        exp >>= 1;
    }
    out as u64
}

pub fn primitive_root_pow(prime: u64, root: u64, n: usize) -> u64 {
    debug_assert!((prime - 1) % (n as u64) == 0);
    mod_pow(root, (prime - 1) / (n as u64), prime)
}

pub fn ntt(a: &mut [u32], invert: bool, prime: u64, root: u64) {
    if a.is_empty() {
        return;
    }
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let wlen = primitive_root_pow(prime, root, len);
        let wlen = if invert {
            mod_pow(wlen, prime - 2, prime)
        } else {
            wlen
        };
        for i in (0..n).step_by(len) {
            let mut w = 1u64;
            for k in 0..len / 2 {
                let u = u64::from(a[i + k]);
                let v = u64::from(a[i + k + len / 2]) * w % prime;
                a[i + k] = ((u + v) % prime) as u32;
                a[i + k + len / 2] = ((u + prime - v) % prime) as u32;
                w = w * wlen % prime;
            }
        }
        len *= 2;
    }
    if invert {
        let n_inv = mod_pow(n as u64, prime - 2, prime);
        for slot in a.iter_mut() {
            *slot = ((u64::from(*slot) * n_inv) % prime) as u32;
        }
    }
}

pub fn convolution_mod(a: &[u64], b: &[u64], prime: u64, root: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }
    let n = next_pow2(a.len() + b.len() - 1);
    let mut fa = vec![0u32; n];
    let mut fb = vec![0u32; n];
    for (i, limb) in a.iter().enumerate() {
        fa[i] = (limb % prime) as u32;
    }
    for (i, limb) in b.iter().enumerate() {
        fb[i] = (limb % prime) as u32;
    }
    ntt(&mut fa, false, prime, root);
    ntt(&mut fb, false, prime, root);
    for i in 0..n {
        fa[i] = ((u64::from(fa[i]) * u64::from(fb[i])) % prime) as u32;
    }
    ntt(&mut fa, true, prime, root);
    fa.into_iter().map(u64::from).collect()
}

pub fn crt_convolution(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }
    let r1 = convolution_mod(a, b, P1, P1_ROOT);
    let r2 = convolution_mod(a, b, P2, P2_ROOT);
    let inv_p1 = mod_pow(P1 % P2, P2 - 2, P2);
    let mut out = Vec::with_capacity(r1.len());
    for i in 0..r1.len() {
        let x1 = r1[i];
        let x2 = r2[i];
        let diff = (x2 + P2 - x1 % P2) % P2;
        let t = (diff as u128 * inv_p1 as u128) % P2 as u128;
        let value = x1 as u128 + P1 as u128 * t;
        out.push(value as u64);
    }
    normalize_limbs(&mut out);
    out
}

pub fn next_pow2(n: usize) -> usize {
    n.max(1).next_power_of_two()
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

    #[test]
    fn roots_have_exact_order() {
        for (prime, root, max_pow) in [(P1, P1_ROOT, P1_MAX_POW), (P2, P2_ROOT, P2_MAX_POW)] {
            for pow in 1..=max_pow {
                let n = 1usize << pow;
                let w = primitive_root_pow(prime, root, n);
                assert_eq!(mod_pow(w, n as u64, prime), 1);
                assert_eq!(mod_pow(w, (n / 2) as u64, prime), prime - 1);
            }
        }
    }

    #[test]
    fn forward_inverse_is_identity() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c21);
        for (prime, root) in [(P1, P1_ROOT), (P2, P2_ROOT)] {
            let mut size = 1usize;
            while size <= 2048 {
                let original: Vec<u32> =
                    (0..size).map(|_| (rng.next_u64() % prime) as u32).collect();
                let mut work = original.clone();
                ntt(&mut work, false, prime, root);
                ntt(&mut work, true, prime, root);
                assert_eq!(work, original);
                size *= 2;
            }
        }
    }

    #[test]
    fn empty_transform_returns_silently() {
        let mut empty: Vec<u32> = Vec::new();
        ntt(&mut empty, false, P1, P1_ROOT);
        ntt(&mut empty, true, P2, P2_ROOT);
        assert!(empty.is_empty());
        assert_eq!(convolution_mod(&[], &[1, 2], P1, P1_ROOT), vec![0]);
        assert_eq!(crt_convolution(&[1], &[]), vec![0]);
    }

    #[test]
    fn convolution_mod_matches_schoolbook_residues() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c22);
        for _ in 0..100 {
            let la = rng.random_range(0..=64);
            let lb = rng.random_range(0..=64);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64() % 1000).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64() % 1000).collect();
            let expected = schoolbook::mul(&a, &b);
            for (prime, root) in [(P1, P1_ROOT), (P2, P2_ROOT)] {
                let residues = convolution_mod(&a, &b, prime, root);
                assert_eq!(&residues[..expected.len()], expected.as_slice());
                assert!(residues[expected.len()..].iter().all(|limb| *limb == 0));
            }
        }
    }

    #[test]
    fn crt_convolution_matches_schoolbook() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c23);
        for _ in 0..100 {
            let la = rng.random_range(0..=64);
            let lb = rng.random_range(0..=64);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64() % 1000).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64() % 1000).collect();
            let result = crt_convolution(&a, &b);
            assert_eq!(result, schoolbook::mul(&a, &b));
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn next_pow2_table() {
        assert_eq!(next_pow2(0), 1);
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(127), 128);
        assert_eq!(next_pow2(128), 128);
    }
}
