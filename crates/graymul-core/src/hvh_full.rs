use crate::hvh;
use crate::limb::normalize_limbs;
use std::collections::HashMap;
use std::sync::OnceLock;

pub const HVH_FULL_CUTOFF: usize = 64;
pub const MAX_DEPTH: usize = 3;
pub const CHUNKS_DEPTH0: usize = 4;
pub const CHUNKS_DEPTH1: usize = 4;
pub const CHUNKS_DEPTH2: usize = 4;
pub const DEPTH2_CUTOFF: usize = 2048;
pub const TABLE_MAX_NEED: usize = 64;

pub fn mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    mul_work(a, b, &mut Vec::new())
}

fn mul_work(a: &[u64], b: &[u64], work: &mut Vec<u64>) -> Vec<u64> {
    mul_depth(a, b, 0, work, depth_cap(a.len().max(b.len())))
}

pub fn mul_buf(a: &[u64], b: &[u64], work: &mut Vec<u64>) -> Vec<u64> {
    mul_work(a, b, work)
}

fn check_cancelled(cancelled: &std::sync::atomic::AtomicBool) -> Result<(), crate::MultiplyError> {
    if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(crate::MultiplyError::cancelled());
    }
    Ok(())
}

pub(crate) fn mul_cancelled(
    a: &[u64],
    b: &[u64],
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    check_cancelled(cancelled)?;
    mul_depth_cancelled(
        a,
        b,
        0,
        &mut Vec::new(),
        depth_cap(a.len().max(b.len())),
        cancelled,
    )
}

fn mul_depth_cancelled(
    a: &[u64],
    b: &[u64],
    depth: usize,
    work: &mut Vec<u64>,
    maxdepth: usize,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    if depth >= maxdepth || a.len().min(b.len()) <= HVH_FULL_CUTOFF {
        return leaf_mul_cancelled(a, b, work, cancelled);
    }
    let chunks = if depth == 0 {
        CHUNKS_DEPTH0
    } else if depth == 1 {
        CHUNKS_DEPTH1
    } else {
        CHUNKS_DEPTH2
    };
    let size = a.len().max(b.len()).div_ceil(chunks);
    let ac = chunk_ranges(a, size);
    let bc = chunk_ranges(b, size);
    let mut out = vec![0u64];
    for (i, (astart, aend)) in ac.iter().enumerate() {
        for (j, (bstart, bend)) in bc.iter().enumerate() {
            check_cancelled(cancelled)?;
            let part = mul_depth_cancelled(
                &a[*astart..*aend],
                &b[*bstart..*bend],
                depth + 1,
                work,
                maxdepth,
                cancelled,
            )?;
            add_shifted(&mut out, &part, (i + j) * size);
        }
    }
    normalize_limbs(&mut out);
    Ok(out)
}

fn leaf_mul_cancelled(
    a: &[u64],
    b: &[u64],
    _work: &mut Vec<u64>,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    check_cancelled(cancelled)?;
    if a.is_empty() || b.is_empty() {
        return Ok(vec![0]);
    }
    if a.len() + b.len() <= 32 {
        return Ok(short_mul_limbs(a, b));
    }
    hvh::mul_cancelled(a, b, cancelled)
}

pub fn predicted_work_limbs(n: usize) -> f64 {
    hvh::predicted_work_limbs(n)
}

fn depth_cap(max_limbs: usize) -> usize {
    if max_limbs <= DEPTH2_CUTOFF {
        2
    } else {
        MAX_DEPTH
    }
}

fn mul_depth(a: &[u64], b: &[u64], depth: usize, work: &mut Vec<u64>, maxdepth: usize) -> Vec<u64> {
    if depth >= maxdepth || a.len().min(b.len()) <= HVH_FULL_CUTOFF {
        return leaf_mul(a, b, work);
    }
    let chunks = if depth == 0 {
        CHUNKS_DEPTH0
    } else if depth == 1 {
        CHUNKS_DEPTH1
    } else {
        CHUNKS_DEPTH2
    };
    let size = a.len().max(b.len()).div_ceil(chunks);
    let ac = chunk_ranges(a, size);
    let bc = chunk_ranges(b, size);
    let mut out = vec![0u64];
    for (i, (astart, aend)) in ac.iter().enumerate() {
        for (j, (bstart, bend)) in bc.iter().enumerate() {
            let part = mul_depth(
                &a[*astart..*aend],
                &b[*bstart..*bend],
                depth + 1,
                work,
                maxdepth,
            );
            add_shifted(&mut out, &part, (i + j) * size);
        }
    }
    normalize_limbs(&mut out);
    out
}

fn leaf_mul(a: &[u64], b: &[u64], work: &mut Vec<u64>) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }
    if a.len() + b.len() <= 32 {
        return short_mul_limbs(a, b);
    }
    hvh::mul_buf(a, b, work)
}

fn short_mul_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    let xd = expand32(a);
    let yd = expand32(b);
    let r0 = short_convolution_prime(&xd, &yd, crate::small_ntt::P0);
    let r1 = short_convolution_prime(&xd, &yd, crate::small_ntt::P1);
    let need = xd.len() + yd.len() - 1;
    let combined = crt_combine_nocarry(&r0, &r1, need);
    let mut carry: u128 = 0;
    let mut out32: Vec<u64> = Vec::with_capacity(need + 2);
    for value in combined {
        let cur = value + carry;
        out32.push((cur & 0xFFFF_FFFF) as u64);
        carry = cur >> 32;
    }
    while carry > 0 {
        out32.push((carry & 0xFFFF_FFFF) as u64);
        carry >>= 32;
    }
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

fn short_convolution_prime(a: &[u64], b: &[u64], prime: u64) -> Vec<u64> {
    let need = a.len() + b.len() - 1;
    let mut n = 1usize;
    while n < need {
        n *= 2;
    }
    debug_assert!(n <= TABLE_MAX_NEED);
    let tables = short_tables();
    let table = &tables[&(prime, n)];
    let mut fa = vec![0u64; n];
    let mut fb = vec![0u64; n];
    fa[..a.len()].copy_from_slice(a);
    fb[..b.len()].copy_from_slice(b);
    let mut xa = dft_apply(&table.fwd, &fa, n, prime);
    let xb = dft_apply(&table.fwd, &fb, n, prime);
    for i in 0..n {
        xa[i] = ((xa[i] as u128 * xb[i] as u128) % prime as u128) as u64;
    }
    let mut out = dft_apply(&table.inv, &xa, n, prime);
    let ninv = crate::ntt::mod_pow(n as u64, prime - 2, prime);
    for slot in out.iter_mut() {
        *slot = ((*slot as u128 * ninv as u128) % prime as u128) as u64;
    }
    out
}

struct ShortTables {
    fwd: Vec<u64>,
    inv: Vec<u64>,
}

fn short_tables() -> &'static HashMap<(u64, usize), ShortTables> {
    static TABLES: OnceLock<HashMap<(u64, usize), ShortTables>> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut map = HashMap::new();
        for prime in [crate::small_ntt::P0, crate::small_ntt::P1] {
            for log in 0..=6 {
                let n = 1usize << log;
                let root = crate::small_ntt::primitive_root_2pow(prime, log);
                let inv_root = crate::ntt::mod_pow(root, prime - 2, prime);
                map.insert(
                    (prime, n),
                    ShortTables {
                        fwd: dft_matrix(prime, n, root),
                        inv: dft_matrix(prime, n, inv_root),
                    },
                );
            }
        }
        map
    })
}

fn dft_matrix(prime: u64, n: usize, root: u64) -> Vec<u64> {
    let mut out = vec![0u64; n * n];
    for k in 0..n {
        let mut pw = 1u64;
        let step = crate::ntt::mod_pow(root, k as u64, prime);
        for j in 0..n {
            out[k * n + j] = pw;
            pw = ((pw as u128 * step as u128) % prime as u128) as u64;
        }
    }
    out
}

fn dft_apply(matrix: &[u64], x: &[u64], n: usize, prime: u64) -> Vec<u64> {
    let mut out = vec![0u64; n];
    for k in 0..n {
        let mut acc: u128 = 0;
        for j in 0..n {
            acc = (acc + x[j] as u128 * matrix[k * n + j] as u128) % prime as u128;
        }
        out[k] = acc as u64;
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
    use crate::verify::verify_mod;
    use rand::Rng;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn random_limbs(rng: &mut ChaCha8Rng, len: usize) -> Vec<u64> {
        (0..len).map(|_| rng.next_u64()).collect()
    }

    #[test]
    fn depth_zero_degenerates_to_hvh() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c45);
        for len in [1usize, 10, HVH_FULL_CUTOFF] {
            let a = random_limbs(&mut rng, len);
            let b = random_limbs(&mut rng, len + 3);
            assert_eq!(mul(&a, &b), hvh::mul(&a, &b));
        }
    }

    #[test]
    fn hvh_identity_small_sizes() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c46);
        for _ in 0..50 {
            let la = rng.random_range(1..=200);
            let lb = rng.random_range(1..=200);
            let a = random_limbs(&mut rng, la);
            let b = random_limbs(&mut rng, lb);
            let result = mul(&a, &b);
            assert_eq!(result, hvh::mul(&a, &b));
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn spot_512_matches_hvh() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c47);
        let a = random_limbs(&mut rng, 512);
        let b = random_limbs(&mut rng, 512);
        let result = mul(&a, &b);
        assert_eq!(result, hvh::mul(&a, &b));
        assert!(verify_mod(&a, &b, &result));
    }

    #[test]
    fn mod_check_large_sizes() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c48);
        for len in [1024usize, 2048, 4096] {
            let a = random_limbs(&mut rng, len);
            let b = random_limbs(&mut rng, len);
            let result = mul(&a, &b);
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn zero_and_one_identities() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c49);
        let big = random_limbs(&mut rng, 300);
        assert_eq!(mul(&vec![0; 300], &big), vec![0]);
        assert_eq!(mul(&big, &vec![0; 300]), vec![0]);
        assert_eq!(mul(&[1], &big), big);
        assert_eq!(mul(&big, &[1]), big);
        assert_eq!(mul(&[], &big), vec![0]);
    }

    #[test]
    fn depth_cap_falls_back_to_hvh() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c50);
        let a = random_limbs(&mut rng, 300);
        let b = random_limbs(&mut rng, 300);
        assert_eq!(
            mul_depth(&a, &b, MAX_DEPTH, &mut Vec::new(), MAX_DEPTH),
            hvh::mul(&a, &b)
        );
        assert_eq!(
            mul_depth(&a, &b, MAX_DEPTH + 1, &mut Vec::new(), MAX_DEPTH),
            hvh::mul(&a, &b)
        );
    }

    #[test]
    fn depth_policy_pins() {
        assert_eq!(depth_cap(1), 2);
        assert_eq!(depth_cap(DEPTH2_CUTOFF), 2);
        assert_eq!(depth_cap(DEPTH2_CUTOFF + 1), 3);
        assert_eq!(depth_cap(MAX_DEPTH), 2);
    }

    #[test]
    fn table_matches_direct_dft() {
        let tables = short_tables();
        for prime in [crate::small_ntt::P0, crate::small_ntt::P1] {
            for log in 0..=6 {
                let n = 1usize << log;
                let table = &tables[&(prime, n)];
                assert_eq!(table.fwd.len(), n * n);
                assert_eq!(table.inv.len(), n * n);
                let root = crate::small_ntt::primitive_root_2pow(prime, log);
                let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c51);
                let x: Vec<u64> = (0..n).map(|_| rng.next_u64() % prime).collect();
                let mut direct = vec![0u64; n];
                for (k, slot) in direct.iter_mut().enumerate() {
                    let mut acc: u128 = 0;
                    for (j, value) in x.iter().enumerate() {
                        let wjk = crate::ntt::mod_pow(root, ((k * j) % n) as u64, prime);
                        acc = (acc + *value as u128 * wjk as u128) % prime as u128;
                    }
                    *slot = acc as u64;
                }
                assert_eq!(dft_apply(&table.fwd, &x, n, prime), direct);
                let back = dft_apply(&table.inv, &direct, n, prime);
                let ninv = crate::ntt::mod_pow(n as u64, prime - 2, prime);
                for (k, slot) in back.iter().enumerate() {
                    let scaled = ((*slot as u128 * ninv as u128) % prime as u128) as u64;
                    assert_eq!(scaled, x[k]);
                }
            }
        }
    }

    #[test]
    fn short_leaf_matches_hvh() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c52);
        for _ in 0..30 {
            let la = rng.random_range(0..=20);
            let lb = rng.random_range(0..=20);
            if la + lb > 32 || (la == 0 && lb == 0) {
                continue;
            }
            let a = random_limbs(&mut rng, la);
            let b = random_limbs(&mut rng, lb);
            assert_eq!(leaf_mul(&a, &b, &mut Vec::new()), hvh::mul(&a, &b));
            assert!(verify_mod(&a, &b, &leaf_mul(&a, &b, &mut Vec::new())));
        }
        assert_eq!(leaf_mul(&[], &[1, 2], &mut Vec::new()), vec![0]);
    }

    #[test]
    fn hvh_contrast_100_cases() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c53);
        for _ in 0..100 {
            let la = rng.random_range(1..=200);
            let lb = rng.random_range(1..=200);
            let a = random_limbs(&mut rng, la);
            let b = random_limbs(&mut rng, lb);
            let result = mul(&a, &b);
            assert_eq!(result, hvh::mul(&a, &b));
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn adaptive_depth_matches_hvh_across_bands() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c54);
        for len in [300usize, 1500, 3000, 6000] {
            let a = random_limbs(&mut rng, len);
            let b = random_limbs(&mut rng, len);
            let result = mul(&a, &b);
            assert_eq!(result, hvh::mul(&a, &b));
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn precancelled_mul_is_cancelled() {
        use std::sync::atomic::AtomicBool;
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c64);
        let a = random_limbs(&mut rng, 300);
        let b = random_limbs(&mut rng, 300);
        let flag = AtomicBool::new(true);
        assert!(matches!(
            mul_cancelled(&a, &b, &flag),
            Err(crate::MultiplyError::Cancelled(_))
        ));
    }

    #[test]
    fn cancelled_clear_matches_mul() {
        use std::sync::atomic::AtomicBool;
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c65);
        let a = random_limbs(&mut rng, 300);
        let b = random_limbs(&mut rng, 300);
        let flag = AtomicBool::new(false);
        let expected = mul(&a, &b);
        assert_eq!(mul_cancelled(&a, &b, &flag).expect("runs"), expected);
        assert!(verify_mod(&a, &b, &expected));
    }
}
