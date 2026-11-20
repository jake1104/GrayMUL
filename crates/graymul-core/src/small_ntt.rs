use crate::limb::normalize_limbs;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256d, _CMP_GE_OQ, _CMP_LT_OQ, _mm256_add_pd, _mm256_blendv_pd, _mm256_cmp_pd,
    _mm256_floor_pd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_max_pd, _mm256_mul_pd,
    _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd,
};

pub const P0: u64 = 63 * (1u64 << 44) + 1;
pub const P1: u64 = 75 * (1u64 << 43) + 1;
pub const P0_MAX_POW: u32 = 44;
pub const P1_MAX_POW: u32 = 43;
pub const TRUNCATE_RATIO: usize = 2;
pub const BAILEY_MIN: usize = 65536;

pub fn should_truncate(need: usize, full: usize) -> bool {
    need > 0 && full / need >= TRUNCATE_RATIO
}

pub fn inv_prime(prime: u64) -> f64 {
    1.0 / prime as f64
}

#[cfg(test)]
fn mod_mul_u64(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 * b as u128) % modulus as u128) as u64
}

#[cfg(test)]
fn is_probable_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for small in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n == small {
            return true;
        }
        if n % small == 0 {
            return false;
        }
    }
    let mut d = n - 1;
    let mut s = 0u32;
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    for base in [2u64, 325, 9375, 28178, 450775, 9780504, 1795265022] {
        let a = base % n;
        if a == 0 {
            continue;
        }
        let mut x = crate::ntt::mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        let mut composite = true;
        for _ in 1..s {
            x = mod_mul_u64(x, x, n);
            if x == n - 1 {
                composite = false;
                break;
            }
        }
        if composite {
            return false;
        }
    }
    true
}

fn distinct_prime_factors(mut n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut f = 2u64;
    while f * f <= n {
        if n % f == 0 {
            out.push(f);
            while n % f == 0 {
                n /= f;
            }
        }
        f += if f == 2 { 1 } else { 2 };
    }
    if n > 1 {
        out.push(n);
    }
    out
}

fn primitive_root(prime: u64) -> u64 {
    let factors = distinct_prime_factors(prime - 1);
    let mut g = 2u64;
    loop {
        if factors
            .iter()
            .all(|q| crate::ntt::mod_pow(g, (prime - 1) / q, prime) != 1)
        {
            return g;
        }
        g += 1;
    }
}

pub fn primitive_root_2pow(prime: u64, l: u32) -> u64 {
    let g = primitive_root(prime);
    crate::ntt::mod_pow(g, (prime - 1) >> l, prime)
}

fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let z = s - a;
    (s, (a - (s - z)) + (b - z))
}

pub fn mulmod_f64(a: f64, b: f64, p: f64, inv_p: f64) -> f64 {
    let (prod_hi, prod_lo) = two_prod(a, b);
    let q = (prod_hi * inv_p).floor().max(0.0);
    let (qp_hi, qp_lo) = two_prod(q, p);
    let (s1, e1) = two_sum(prod_hi, -qp_hi);
    let (s2, e2) = two_sum(s1, -qp_lo);
    let mut r = s2 + ((e1 + e2) + prod_lo);
    while r < 0.0 {
        r += p;
    }
    while r >= p {
        r -= p;
    }
    r
}

fn addmod_f64(a: f64, b: f64, p: f64) -> f64 {
    let s = a + b;
    if s >= p { s - p } else { s }
}

fn submod_f64(a: f64, b: f64, p: f64) -> f64 {
    let d = a - b;
    if d < 0.0 { d + p } else { d }
}

pub fn ntt_f64(a: &mut [f64], invert: bool, prime: u64, root: u64) {
    if a.len() >= BAILEY_MIN && a.len().is_power_of_two() {
        ntt_f64_bailey(a, invert, prime, root);
    } else {
        ntt_f64_flat(a, invert, prime, root);
    }
}

pub fn bailey_split(n: usize) -> (usize, usize) {
    let log = n.trailing_zeros();
    let log1 = log / 2;
    (1usize << log1, 1usize << (log - log1))
}

fn transpose_f64(dst: &mut [f64], src: &[f64], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

pub fn ntt_f64_bailey(a: &mut [f64], invert: bool, prime: u64, root: u64) {
    let n = a.len();
    if n < 4 || !n.is_power_of_two() {
        ntt_f64_flat(a, invert, prime, root);
        return;
    }
    let (n1, n2) = bailey_split(n);
    let log = n.trailing_zeros();
    let log1 = log / 2;
    let log2 = log - log1;
    let root1 = primitive_root_2pow(prime, log1.max(1));
    let root2 = primitive_root_2pow(prime, log2.max(1));
    let base = if invert {
        crate::ntt::mod_pow(root, prime - 2, prime)
    } else {
        root
    };
    let mut tmp = vec![0.0f64; n];
    transpose_f64(&mut tmp, a, n1, n2);
    for row in tmp.chunks_mut(n1) {
        ntt_f64(row, invert, prime, root1);
    }
    let p = prime as f64;
    let inv_p = inv_prime(prime);
    let mut wk1 = 1u64;
    for k1 in 0..n1 {
        let mut f = 1u64;
        for j2 in 0..n2 {
            let v = tmp[j2 * n1 + k1];
            a[k1 * n2 + j2] = mulmod_f64(v, f as f64, p, inv_p);
            f = ((f as u128 * wk1 as u128) % prime as u128) as u64;
        }
        wk1 = ((wk1 as u128 * base as u128) % prime as u128) as u64;
    }
    for row in a.chunks_mut(n2) {
        ntt_f64(row, invert, prime, root2);
    }
    transpose_f64(&mut tmp, a, n1, n2);
    a.copy_from_slice(&tmp);
}

pub fn ntt_f64_flat(a: &mut [f64], invert: bool, prime: u64, root: u64) {
    if a.is_empty() {
        return;
    }
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    debug_assert!(n <= (1usize << P1_MAX_POW));
    bit_reverse(a);
    let mut bound = 1.0f64;
    for (width, radix4) in stage_widths(n) {
        if radix4 {
            stage_radix4(a, n, width, invert, prime, root, &mut bound);
        } else {
            stage_radix2(a, n, width, invert, prime, root);
        }
    }
    if invert {
        let p = prime as f64;
        let inv_p = inv_prime(prime);
        reduce_pass(a, p, inv_p);
        let n_inv = crate::ntt::mod_pow(n as u64, prime - 2, prime) as f64;
        for slot in a.iter_mut() {
            *slot = mulmod_f64(*slot, n_inv, p, inv_p);
        }
    } else {
        let p = prime as f64;
        let inv_p = inv_prime(prime);
        reduce_pass(a, p, inv_p);
    }
}

const MIN_RADIX4_LEN: usize = 256;

fn stage_widths(n: usize) -> Vec<(usize, bool)> {
    let mut out = Vec::new();
    if n < MIN_RADIX4_LEN {
        let mut len = 2usize;
        while len <= n {
            out.push((len, false));
            len *= 2;
        }
        return out;
    }
    let mut log = 0u32;
    while (1usize << log) < n {
        log += 1;
    }
    if log % 2 == 1 {
        out.push((2, false));
    }
    let mut len = if log % 2 == 1 { 8usize } else { 4usize };
    while len <= n {
        out.push((len, true));
        len *= 4;
    }
    out
}

fn bit_reverse(a: &mut [f64]) {
    let n = a.len();
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
}

fn stage_wlen(prime: u64, root: u64, n: usize, len: usize, invert: bool) -> u64 {
    let wlen = crate::ntt::mod_pow(root, (n / len) as u64, prime);
    if invert {
        crate::ntt::mod_pow(wlen, prime - 2, prime)
    } else {
        wlen
    }
}

fn reduce_pass(a: &mut [f64], p: f64, inv_p: f64) {
    for slot in a.iter_mut() {
        let q = (*slot * inv_p).floor();
        let mut r = *slot - q * p;
        if r >= p {
            r -= p;
        }
        if r < 0.0 {
            r += p;
        }
        *slot = r;
    }
}

fn stage_radix2(a: &mut [f64], n: usize, len: usize, invert: bool, prime: u64, root: u64) {
    let p = prime as f64;
    let inv_p = inv_prime(prime);
    let wlen = stage_wlen(prime, root, n, len, invert) as f64;
    for i in (0..n).step_by(len) {
        let mut w = 1.0f64;
        for k in 0..len / 2 {
            let u = a[i + k];
            let v = mulmod_f64(a[i + k + len / 2], w, p, inv_p);
            a[i + k] = addmod_f64(u, v, p);
            a[i + k + len / 2] = submod_f64(u, v, p);
            w = mulmod_f64(w, wlen, p, inv_p);
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn simd_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn mulmod_vec(a: __m256d, b: __m256d, p: __m256d, inv_p: __m256d, zero: __m256d) -> __m256d {
    let prod_hi = _mm256_mul_pd(a, b);
    let prod_lo = _mm256_fmadd_pd(a, b, _mm256_sub_pd(zero, prod_hi));
    let q = _mm256_max_pd(_mm256_floor_pd(_mm256_mul_pd(prod_hi, inv_p)), zero);
    let qp_hi = _mm256_mul_pd(q, p);
    let qp_lo = _mm256_fmadd_pd(q, p, _mm256_sub_pd(zero, qp_hi));
    let s1 = _mm256_add_pd(prod_hi, _mm256_sub_pd(zero, qp_hi));
    let z1 = _mm256_sub_pd(s1, prod_hi);
    let e1 = _mm256_add_pd(
        _mm256_sub_pd(prod_hi, _mm256_sub_pd(s1, z1)),
        _mm256_sub_pd(_mm256_sub_pd(zero, qp_hi), z1),
    );
    let s2 = _mm256_add_pd(s1, _mm256_sub_pd(zero, qp_lo));
    let z2 = _mm256_sub_pd(s2, s1);
    let e2 = _mm256_add_pd(
        _mm256_sub_pd(s1, _mm256_sub_pd(s2, z2)),
        _mm256_sub_pd(_mm256_sub_pd(zero, qp_lo), z2),
    );
    let mut r = _mm256_add_pd(s2, _mm256_add_pd(_mm256_add_pd(e1, e2), prod_lo));
    for _ in 0..6 {
        r = _mm256_blendv_pd(r, _mm256_add_pd(r, p), _mm256_cmp_pd(r, zero, _CMP_LT_OQ));
    }
    for _ in 0..6 {
        r = _mm256_blendv_pd(r, _mm256_sub_pd(r, p), _mm256_cmp_pd(r, p, _CMP_GE_OQ));
    }
    r
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn butterfly_block4(
    ptr: *mut f64,
    base: usize,
    m: usize,
    x0v: __m256d,
    x1v: __m256d,
    x2v: __m256d,
    x3v: __m256d,
    w2v: __m256d,
    wjv: __m256d,
    wjmv: __m256d,
    p: __m256d,
    inv_p: __m256d,
    zero: __m256d,
) {
    let ax1 = mulmod_vec(x1v, w2v, p, inv_p, zero);
    let ax3 = mulmod_vec(x3v, w2v, p, inv_p, zero);
    let t0 = _mm256_add_pd(x0v, ax1);
    let t1 = _mm256_blendv_pd(
        _mm256_sub_pd(x0v, ax1),
        _mm256_add_pd(_mm256_sub_pd(x0v, ax1), p),
        _mm256_cmp_pd(_mm256_sub_pd(x0v, ax1), zero, _CMP_LT_OQ),
    );
    let t2 = _mm256_add_pd(x2v, ax3);
    let t3 = _mm256_blendv_pd(
        _mm256_sub_pd(x2v, ax3),
        _mm256_add_pd(_mm256_sub_pd(x2v, ax3), p),
        _mm256_cmp_pd(_mm256_sub_pd(x2v, ax3), zero, _CMP_LT_OQ),
    );
    let u2 = mulmod_vec(wjv, t2, p, inv_p, zero);
    let u3 = mulmod_vec(wjmv, t3, p, inv_p, zero);
    let down0 = _mm256_blendv_pd(
        _mm256_sub_pd(t0, u2),
        _mm256_add_pd(_mm256_sub_pd(t0, u2), p),
        _mm256_cmp_pd(_mm256_sub_pd(t0, u2), zero, _CMP_LT_OQ),
    );
    let down1 = _mm256_blendv_pd(
        _mm256_sub_pd(t1, u3),
        _mm256_add_pd(_mm256_sub_pd(t1, u3), p),
        _mm256_cmp_pd(_mm256_sub_pd(t1, u3), zero, _CMP_LT_OQ),
    );
    _mm256_storeu_pd(ptr.add(base), _mm256_add_pd(t0, u2));
    _mm256_storeu_pd(ptr.add(base + m), _mm256_add_pd(t1, u3));
    _mm256_storeu_pd(ptr.add(base + 2 * m), down0);
    _mm256_storeu_pd(ptr.add(base + 3 * m), down1);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn stage_radix4_simd(
    a: &mut [f64],
    n: usize,
    len: usize,
    invert: bool,
    prime: u64,
    root: u64,
) {
    let p = prime as f64;
    let inv_p = inv_prime(prime);
    let pv = _mm256_set1_pd(p);
    let inv_pv = _mm256_set1_pd(inv_p);
    let zero = _mm256_setzero_pd();
    let m = len / 4;
    let wlen_int = stage_wlen(prime, root, n, len, invert);
    let wlen = wlen_int as f64;
    let w2step = mulmod_f64(wlen, wlen, p, inv_p);
    let wm = crate::ntt::mod_pow(wlen_int, m as u64, prime) as f64;
    let w4step = mulmod_f64(w2step, w2step, p, inv_p);
    let w8step = mulmod_f64(w4step, w4step, p, inv_p);
    let ptr = a.as_mut_ptr();
    debug_assert!(m >= 4 && m % 4 == 0);
    for i in (0..n).step_by(len) {
        let w2seed = mulmod_f64(wlen, wlen, p, inv_p);
        let w3seed = mulmod_f64(w2seed, wlen, p, inv_p);
        let mut wjv = _mm256_loadu_pd([1.0, wlen, w2seed, w3seed].as_ptr());
        let mut w2v = _mm256_loadu_pd(
            [
                1.0,
                w2step,
                mulmod_f64(w2step, w2step, p, inv_p),
                mulmod_f64(mulmod_f64(w2step, w2step, p, inv_p), w2step, p, inv_p),
            ]
            .as_ptr(),
        );
        let mut wjmv = _mm256_loadu_pd(
            [
                wm,
                mulmod_f64(wm, wlen, p, inv_p),
                mulmod_f64(mulmod_f64(wm, wlen, p, inv_p), wlen, p, inv_p),
                mulmod_f64(
                    mulmod_f64(mulmod_f64(wm, wlen, p, inv_p), wlen, p, inv_p),
                    wlen,
                    p,
                    inv_p,
                ),
            ]
            .as_ptr(),
        );
        let w4v = _mm256_set1_pd(w4step);
        let w8v = _mm256_set1_pd(w8step);
        let mut j = 0usize;
        while j + 4 <= m {
            butterfly_block4(
                ptr,
                i + j,
                m,
                _mm256_loadu_pd(ptr.add(i + j)),
                _mm256_loadu_pd(ptr.add(i + j + m)),
                _mm256_loadu_pd(ptr.add(i + j + 2 * m)),
                _mm256_loadu_pd(ptr.add(i + j + 3 * m)),
                w2v,
                wjv,
                wjmv,
                pv,
                inv_pv,
                zero,
            );
            wjv = mulmod_vec(wjv, w4v, pv, inv_pv, zero);
            w2v = mulmod_vec(w2v, w8v, pv, inv_pv, zero);
            wjmv = mulmod_vec(wjmv, w4v, pv, inv_pv, zero);
            j += 4;
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn run_radix4_stage(a: &mut [f64], n: usize, len: usize, invert: bool, prime: u64, root: u64) {
    if simd_available() && len >= 16 {
        unsafe {
            stage_radix4_simd(a, n, len, invert, prime, root);
        }
    } else {
        stage_radix4_scalar_fallback(a, n, len, invert, prime, root);
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn run_radix4_stage(a: &mut [f64], n: usize, len: usize, invert: bool, prime: u64, root: u64) {
    stage_radix4_scalar_fallback(a, n, len, invert, prime, root);
}

#[allow(clippy::too_many_arguments)]
fn stage_radix4_scalar_fallback(
    a: &mut [f64],
    n: usize,
    len: usize,
    invert: bool,
    prime: u64,
    root: u64,
) {
    let p = prime as f64;
    let inv_p = inv_prime(prime);
    let m = len / 4;
    let wlen_int = stage_wlen(prime, root, n, len, invert);
    let wlen = wlen_int as f64;
    let w2step = mulmod_f64(wlen, wlen, p, inv_p);
    let wm = crate::ntt::mod_pow(wlen_int, m as u64, prime) as f64;
    for i in (0..n).step_by(len) {
        let mut wj = 1.0f64;
        let mut w2 = 1.0f64;
        for j in 0..m {
            let x0 = a[i + j];
            let x1 = a[i + j + m];
            let x2 = a[i + j + 2 * m];
            let x3 = a[i + j + 3 * m];
            let ax1 = mulmod_f64(x1, w2, p, inv_p);
            let ax3 = mulmod_f64(x3, w2, p, inv_p);
            let t0 = x0 + ax1;
            let mut t1 = x0 - ax1;
            if t1 < 0.0 {
                t1 += p;
            }
            let t2 = x2 + ax3;
            let mut t3 = x2 - ax3;
            if t3 < 0.0 {
                t3 += p;
            }
            let wjm = mulmod_f64(wj, wm, p, inv_p);
            let u2 = mulmod_f64(wj, t2, p, inv_p);
            let u3 = mulmod_f64(wjm, t3, p, inv_p);
            a[i + j] = t0 + u2;
            let mut down0 = t0 - u2;
            if down0 < 0.0 {
                down0 += p;
            }
            a[i + j + 2 * m] = down0;
            a[i + j + m] = t1 + u3;
            let mut down1 = t1 - u3;
            if down1 < 0.0 {
                down1 += p;
            }
            a[i + j + 3 * m] = down1;
            wj = mulmod_f64(wj, wlen, p, inv_p);
            w2 = mulmod_f64(w2, w2step, p, inv_p);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn stage_radix4(
    a: &mut [f64],
    n: usize,
    len: usize,
    invert: bool,
    prime: u64,
    root: u64,
    bound: &mut f64,
) {
    if *bound > 4.0 {
        reduce_pass(a, prime as f64, inv_prime(prime));
        *bound = 1.0;
    }
    run_radix4_stage(a, n, len, invert, prime, root);
    *bound += 2.0;
}

#[cfg(test)]
fn ntt_f64_radix2(a: &mut [f64], invert: bool, prime: u64, root: u64) {
    if a.is_empty() {
        return;
    }
    let n = a.len();
    bit_reverse(a);
    let mut len = 2usize;
    while len <= n {
        stage_radix2(a, n, len, invert, prime, root);
        len *= 2;
    }
    if invert {
        let p = prime as f64;
        let inv_p = inv_prime(prime);
        let n_inv = crate::ntt::mod_pow(n as u64, prime - 2, prime) as f64;
        for slot in a.iter_mut() {
            *slot = mulmod_f64(*slot, n_inv, p, inv_p);
        }
    }
}

#[cfg(test)]
fn ntt_f64_force_scalar(a: &mut [f64], invert: bool, prime: u64, root: u64) {
    if a.is_empty() {
        return;
    }
    let n = a.len();
    bit_reverse(a);
    let mut bound = 1.0f64;
    for (width, radix4) in stage_widths(n) {
        if radix4 {
            let p = prime as f64;
            let inv_p = inv_prime(prime);
            if bound > 4.0 {
                reduce_pass(a, p, inv_p);
                bound = 1.0;
            }
            stage_radix4_scalar_fallback(a, n, width, invert, prime, root);
            bound += 2.0;
        } else {
            stage_radix2(a, n, width, invert, prime, root);
        }
    }
    if invert {
        let p = prime as f64;
        let inv_p = inv_prime(prime);
        reduce_pass(a, p, inv_p);
        let n_inv = crate::ntt::mod_pow(n as u64, prime - 2, prime) as f64;
        for slot in a.iter_mut() {
            *slot = mulmod_f64(*slot, n_inv, p, inv_p);
        }
    } else {
        let p = prime as f64;
        let inv_p = inv_prime(prime);
        reduce_pass(a, p, inv_p);
    }
}

fn bit_rev(index: usize, bits: u32) -> usize {
    if bits == 0 {
        return 0;
    }
    index.reverse_bits() >> (usize::BITS - bits)
}

pub fn ntt_f64_forward_truncated(a: &mut [f64], nonzero: usize, prime: u64, root: u64) {
    let n = a.len();
    if n == 0 {
        return;
    }
    if nonzero >= n {
        ntt_f64(a, false, prime, root);
        return;
    }
    if nonzero == 0 {
        for slot in a.iter_mut() {
            *slot = 0.0;
        }
        return;
    }
    bit_reverse(a);
    let mut log = 0u32;
    while (1usize << log) < n {
        log += 1;
    }
    let mut kept = vec![false; n];
    for p in 0..nonzero {
        kept[bit_rev(p, log)] = true;
    }
    let p = prime as f64;
    let inv_p = inv_prime(prime);
    let mut bound = 1.0f64;
    for (width, radix4) in stage_widths(n) {
        if radix4 {
            if bound > 4.0 {
                reduce_pass(a, p, inv_p);
                bound = 1.0;
            }
            let m = width / 4;
            let wlen_int = stage_wlen(prime, root, n, width, false);
            let wlen = wlen_int as f64;
            let w2step = mulmod_f64(wlen, wlen, p, inv_p);
            let wm = crate::ntt::mod_pow(wlen_int, m as u64, prime) as f64;
            for i in (0..n).step_by(width) {
                let mut wj = 1.0f64;
                let mut w2 = 1.0f64;
                for j in 0..m {
                    let j0 = i + j;
                    let j1 = j0 + m;
                    let j2 = j0 + 2 * m;
                    let j3 = j0 + 3 * m;
                    if kept[j0] || kept[j1] || kept[j2] || kept[j3] {
                        let x0 = a[j0];
                        let x1 = a[j1];
                        let x2 = a[j2];
                        let x3 = a[j3];
                        let ax1 = mulmod_f64(x1, w2, p, inv_p);
                        let ax3 = mulmod_f64(x3, w2, p, inv_p);
                        let t0 = x0 + ax1;
                        let mut t1 = x0 - ax1;
                        if t1 < 0.0 {
                            t1 += p;
                        }
                        let t2 = x2 + ax3;
                        let mut t3 = x2 - ax3;
                        if t3 < 0.0 {
                            t3 += p;
                        }
                        let wjm = mulmod_f64(wj, wm, p, inv_p);
                        let u2 = mulmod_f64(wj, t2, p, inv_p);
                        let u3 = mulmod_f64(wjm, t3, p, inv_p);
                        a[j0] = t0 + u2;
                        let mut down0 = t0 - u2;
                        if down0 < 0.0 {
                            down0 += p;
                        }
                        a[j2] = down0;
                        a[j1] = t1 + u3;
                        let mut down1 = t1 - u3;
                        if down1 < 0.0 {
                            down1 += p;
                        }
                        a[j3] = down1;
                        kept[j0] = true;
                        kept[j1] = true;
                        kept[j2] = true;
                        kept[j3] = true;
                    }
                    wj = mulmod_f64(wj, wlen, p, inv_p);
                    w2 = mulmod_f64(w2, w2step, p, inv_p);
                }
            }
            bound += 2.0;
        } else {
            let wlen = stage_wlen(prime, root, n, width, false) as f64;
            for i in (0..n).step_by(width) {
                let mut w = 1.0f64;
                for k in 0..width / 2 {
                    let j0 = i + k;
                    let j1 = j0 + width / 2;
                    if kept[j0] || kept[j1] {
                        let u = a[j0];
                        let v = mulmod_f64(a[j1], w, p, inv_p);
                        a[j0] = addmod_f64(u, v, p);
                        a[j1] = submod_f64(u, v, p);
                        kept[j0] = true;
                        kept[j1] = true;
                    }
                    w = mulmod_f64(w, wlen, p, inv_p);
                }
            }
        }
    }
    reduce_pass(a, p, inv_p);
}

pub fn ntt_f64_inverse_truncated(a: &mut [f64], keep: usize, prime: u64, root: u64) {
    let n = a.len();
    if n == 0 {
        return;
    }
    if keep >= n {
        ntt_f64(a, true, prime, root);
        return;
    }
    if keep == 0 {
        for slot in a.iter_mut() {
            *slot = 0.0;
        }
        return;
    }
    let stages = stage_widths(n);
    let mut after = vec![false; n];
    for slot in after.iter_mut().take(keep) {
        *slot = true;
    }
    let mut masks: Vec<Vec<bool>> = Vec::with_capacity(stages.len());
    for (width, radix4) in stages.iter().rev() {
        masks.push(after.clone());
        let mut before = vec![false; n];
        if *radix4 {
            let m = width / 4;
            for i in (0..n).step_by(*width) {
                for j in 0..m {
                    let g = [i + j, i + j + m, i + j + 2 * m, i + j + 3 * m];
                    if g.iter().any(|t| after[*t]) {
                        for t in g {
                            before[t] = true;
                        }
                    }
                }
            }
        } else {
            for i in (0..n).step_by(*width) {
                for k in 0..width / 2 {
                    let j0 = i + k;
                    let j1 = j0 + width / 2;
                    if after[j0] || after[j1] {
                        before[j0] = true;
                        before[j1] = true;
                    }
                }
            }
        }
        after = before;
    }
    masks.reverse();
    bit_reverse(a);
    let p = prime as f64;
    let inv_p = inv_prime(prime);
    let mut bound = 1.0f64;
    for (s, (width, radix4)) in stages.iter().enumerate() {
        let mask = &masks[s];
        if *radix4 {
            if bound > 4.0 {
                reduce_pass(a, p, inv_p);
                bound = 1.0;
            }
            let m = width / 4;
            let wlen_int = stage_wlen(prime, root, n, *width, true);
            let wlen = wlen_int as f64;
            let w2step = mulmod_f64(wlen, wlen, p, inv_p);
            let wm = crate::ntt::mod_pow(wlen_int, m as u64, prime) as f64;
            for i in (0..n).step_by(*width) {
                let mut wj = 1.0f64;
                let mut w2 = 1.0f64;
                for j in 0..m {
                    let j0 = i + j;
                    let j1 = j0 + m;
                    let j2 = j0 + 2 * m;
                    let j3 = j0 + 3 * m;
                    if mask[j0] || mask[j1] || mask[j2] || mask[j3] {
                        let x0 = a[j0];
                        let x1 = a[j1];
                        let x2 = a[j2];
                        let x3 = a[j3];
                        let ax1 = mulmod_f64(x1, w2, p, inv_p);
                        let ax3 = mulmod_f64(x3, w2, p, inv_p);
                        let t0 = x0 + ax1;
                        let mut t1 = x0 - ax1;
                        if t1 < 0.0 {
                            t1 += p;
                        }
                        let t2 = x2 + ax3;
                        let mut t3 = x2 - ax3;
                        if t3 < 0.0 {
                            t3 += p;
                        }
                        let wjm = mulmod_f64(wj, wm, p, inv_p);
                        let u2 = mulmod_f64(wj, t2, p, inv_p);
                        let u3 = mulmod_f64(wjm, t3, p, inv_p);
                        a[j0] = t0 + u2;
                        let mut down0 = t0 - u2;
                        if down0 < 0.0 {
                            down0 += p;
                        }
                        a[j2] = down0;
                        a[j1] = t1 + u3;
                        let mut down1 = t1 - u3;
                        if down1 < 0.0 {
                            down1 += p;
                        }
                        a[j3] = down1;
                    }
                    wj = mulmod_f64(wj, wlen, p, inv_p);
                    w2 = mulmod_f64(w2, w2step, p, inv_p);
                }
            }
            bound += 2.0;
        } else {
            let wlen = stage_wlen(prime, root, n, *width, true) as f64;
            for i in (0..n).step_by(*width) {
                let mut w = 1.0f64;
                for k in 0..width / 2 {
                    let j0 = i + k;
                    let j1 = j0 + width / 2;
                    if mask[j0] || mask[j1] {
                        let u = a[j0];
                        let v = mulmod_f64(a[j1], w, p, inv_p);
                        a[j0] = addmod_f64(u, v, p);
                        a[j1] = submod_f64(u, v, p);
                    }
                    w = mulmod_f64(w, wlen, p, inv_p);
                }
            }
        }
    }
    reduce_pass(a, p, inv_p);
    let n_inv = crate::ntt::mod_pow(n as u64, prime - 2, prime) as f64;
    for slot in a.iter_mut().take(keep) {
        *slot = mulmod_f64(*slot, n_inv, p, inv_p);
    }
    for slot in a.iter_mut().skip(keep) {
        *slot = 0.0;
    }
}

pub fn convolution_f64(a: &[u64], b: &[u64], prime: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }
    let need = a.len() + b.len() - 1;
    let full = crate::ntt::next_pow2(need);
    if should_truncate(need, full) {
        convolution_f64_truncated(a, b, prime)
    } else {
        convolution_f64_full(a, b, prime)
    }
}

pub fn convolution_f64_truncated(a: &[u64], b: &[u64], prime: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }
    let n = crate::ntt::next_pow2(a.len() + b.len() - 1);
    let p = prime as f64;
    let mut fa = vec![0.0f64; n];
    let mut fb = vec![0.0f64; n];
    for (i, digit) in a.iter().enumerate() {
        fa[i] = (digit % prime) as f64;
    }
    for (i, digit) in b.iter().enumerate() {
        fb[i] = (digit % prime) as f64;
    }
    let mut power = 0u32;
    while (1usize << power) < n {
        power += 1;
    }
    let root = primitive_root_2pow(prime, power);
    let inv_p = inv_prime(prime);
    ntt_f64_forward_truncated(&mut fa, a.len(), prime, root);
    ntt_f64_forward_truncated(&mut fb, b.len(), prime, root);
    for i in 0..n {
        fa[i] = mulmod_f64(fa[i], fb[i], p, inv_p);
    }
    ntt_f64_inverse_truncated(&mut fa, a.len() + b.len() - 1, prime, root);
    fa.into_iter().map(|value| value as u64).collect()
}

pub fn convolution_f64_full(a: &[u64], b: &[u64], prime: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }
    let n = crate::ntt::next_pow2(a.len() + b.len() - 1);
    let p = prime as f64;
    let mut fa = vec![0.0f64; n];
    let mut fb = vec![0.0f64; n];
    for (i, digit) in a.iter().enumerate() {
        fa[i] = (digit % prime) as f64;
    }
    for (i, digit) in b.iter().enumerate() {
        fb[i] = (digit % prime) as f64;
    }
    let mut power = 0u32;
    while (1usize << power) < n {
        power += 1;
    }
    let root = primitive_root_2pow(prime, power);
    let inv_p = inv_prime(prime);
    ntt_f64_flat(&mut fa, false, prime, root);
    ntt_f64_flat(&mut fb, false, prime, root);
    for i in 0..n {
        fa[i] = mulmod_f64(fa[i], fb[i], p, inv_p);
    }
    ntt_f64_flat(&mut fa, true, prime, root);
    fa.into_iter().map(|value| value as u64).collect()
}

fn expand32(limbs: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(limbs.len() * 2);
    for limb in limbs {
        out.push(limb & 0xFFFF_FFFF);
        out.push(limb >> 32);
    }
    out
}

pub fn crt_convolution_small(a: &[u64], b: &[u64]) -> Vec<u64> {
    crt_convolve_with(a, b, convolution_f64)
}

pub fn measure_crt_flat(a: &[u64], b: &[u64]) -> Vec<u64> {
    crt_convolve_with(a, b, convolution_f64_full)
}

pub fn measure_crt_bailey(a: &[u64], b: &[u64]) -> Vec<u64> {
    crt_convolve_with(a, b, convolution_bailey)
}

pub fn measure_crt_trunc(a: &[u64], b: &[u64]) -> Vec<u64> {
    crt_convolve_with(a, b, convolution_f64_truncated)
}

pub fn convolution_parallel(a: &[u64], b: &[u64], workers: usize) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }
    if workers <= 1 {
        return measure_crt_flat(a, b);
    }
    let da = expand32(a);
    let db = expand32(b);
    let need = da.len() + db.len() - 1;
    let prime_b_workers = workers / 2;
    #[cfg(target_arch = "wasm32")]
    let (r0, r1) = {
        let _ = prime_b_workers;
        let r0 = convolution_prime_par(&da, &db, P0, 1);
        let r1 = convolution_prime_par(&da, &db, P1, 1);
        (r0, r1)
    };
    #[cfg(not(target_arch = "wasm32"))]
    let (r0, r1) = std::thread::scope(|s| {
        let prime_b = s.spawn(|| convolution_prime_par(&da, &db, P1, prime_b_workers));
        let prime_a = convolution_prime_par(&da, &db, P0, workers - prime_b_workers);
        (prime_a, prime_b.join().unwrap())
    });
    crt_combine_raw(&r0, &r1, need)
}

fn convolution_prime_par(a: &[u64], b: &[u64], prime: u64, workers: usize) -> Vec<u64> {
    let n = crate::ntt::next_pow2(a.len() + b.len() - 1);
    let p = prime as f64;
    let mut fa = vec![0.0f64; n];
    let mut fb = vec![0.0f64; n];
    for (i, digit) in a.iter().enumerate() {
        fa[i] = (digit % prime) as f64;
    }
    for (i, digit) in b.iter().enumerate() {
        fb[i] = (digit % prime) as f64;
    }
    let mut power = 0u32;
    while (1usize << power) < n {
        power += 1;
    }
    let root = primitive_root_2pow(prime, power);
    let inv_p = inv_prime(prime);
    if workers <= 1 {
        ntt_f64(&mut fa, false, prime, root);
        ntt_f64(&mut fb, false, prime, root);
    } else {
        let back_workers = workers / 2;
        let front_workers = workers - back_workers;
        #[cfg(target_arch = "wasm32")]
        let (done_fa, done_fb) = {
            let _ = (front_workers, back_workers);
            ntt_parallel(&mut fa, false, prime, root, 1);
            ntt_parallel(&mut fb, false, prime, root, 1);
            (fa, fb)
        };
        #[cfg(not(target_arch = "wasm32"))]
        let (done_fa, done_fb) = std::thread::scope(|s| {
            let front = s.spawn(move || {
                let mut front = fa;
                ntt_parallel(&mut front, false, prime, root, front_workers);
                front
            });
            let mut back = fb;
            ntt_parallel(&mut back, false, prime, root, back_workers);
            (front.join().unwrap(), back)
        });
        fa = done_fa;
        fb = done_fb;
    }
    for i in 0..n {
        fa[i] = mulmod_f64(fa[i], fb[i], p, inv_p);
    }
    ntt_parallel(&mut fa, true, prime, root, workers);
    fa.into_iter().map(|value| value as u64).collect()
}

fn ntt_parallel(a: &mut [f64], invert: bool, prime: u64, root: u64, workers: usize) {
    if workers <= 1 || !a.len().is_power_of_two() || a.len() < BAILEY_MIN {
        ntt_f64(a, invert, prime, root);
    } else {
        ntt_f64_bailey_par(a, invert, prime, root, workers);
    }
}

fn ntt_f64_bailey_par(a: &mut [f64], invert: bool, prime: u64, root: u64, workers: usize) {
    let n = a.len();
    if workers <= 1 || n < 4 || !n.is_power_of_two() {
        ntt_f64_bailey(a, invert, prime, root);
        return;
    }
    let (n1, n2) = bailey_split(n);
    let log = n.trailing_zeros();
    let log1 = log / 2;
    let log2 = log - log1;
    let root1 = primitive_root_2pow(prime, log1.max(1));
    let root2 = primitive_root_2pow(prime, log2.max(1));
    let base = if invert {
        crate::ntt::mod_pow(root, prime - 2, prime)
    } else {
        root
    };
    let mut tmp = vec![0.0f64; n];
    transpose_f64(&mut tmp, a, n1, n2);
    transform_rows(&mut tmp, n1, n2, invert, prime, root1, workers);
    let p = prime as f64;
    let inv_p = inv_prime(prime);
    let mut wk1 = 1u64;
    for k1 in 0..n1 {
        let mut f = 1u64;
        for j2 in 0..n2 {
            let v = tmp[j2 * n1 + k1];
            a[k1 * n2 + j2] = mulmod_f64(v, f as f64, p, inv_p);
            f = ((f as u128 * wk1 as u128) % prime as u128) as u64;
        }
        wk1 = ((wk1 as u128 * base as u128) % prime as u128) as u64;
    }
    transform_rows(a, n2, n1, invert, prime, root2, workers);
    transpose_f64(&mut tmp, a, n1, n2);
    a.copy_from_slice(&tmp);
}

#[allow(clippy::too_many_arguments)]
fn transform_rows(
    buf: &mut [f64],
    row_len: usize,
    row_count: usize,
    invert: bool,
    prime: u64,
    root: u64,
    workers: usize,
) {
    debug_assert_eq!(buf.len(), row_len * row_count);
    #[cfg(target_arch = "wasm32")]
    {
        let _ = workers;
        for row in buf.chunks_mut(row_len) {
            ntt_f64(row, invert, prime, root);
        }
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let lanes = workers.max(1).min(row_count);
        if lanes <= 1 {
            for row in buf.chunks_mut(row_len) {
                ntt_f64(row, invert, prime, root);
            }
            return;
        }
        let chunk = row_count.div_ceil(lanes);
        let mut blocks: Vec<&mut [f64]> = Vec::with_capacity(lanes);
        let mut rest = &mut buf[..];
        for k in 0..lanes {
            let take = chunk.min(row_count - k * chunk) * row_len;
            let (head, tail) = rest.split_at_mut(take);
            blocks.push(head);
            rest = tail;
        }
        std::thread::scope(|s| {
            for block in blocks {
                s.spawn(move || {
                    for row in block.chunks_mut(row_len) {
                        ntt_f64(row, invert, prime, root);
                    }
                });
            }
        });
    }
}

fn convolution_bailey(a: &[u64], b: &[u64], prime: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }
    let n = crate::ntt::next_pow2(a.len() + b.len() - 1);
    let p = prime as f64;
    let mut fa = vec![0.0f64; n];
    let mut fb = vec![0.0f64; n];
    for (i, digit) in a.iter().enumerate() {
        fa[i] = (digit % prime) as f64;
    }
    for (i, digit) in b.iter().enumerate() {
        fb[i] = (digit % prime) as f64;
    }
    let mut power = 0u32;
    while (1usize << power) < n {
        power += 1;
    }
    let root = primitive_root_2pow(prime, power);
    let inv_p = inv_prime(prime);
    ntt_f64_bailey(&mut fa, false, prime, root);
    ntt_f64_bailey(&mut fb, false, prime, root);
    for i in 0..n {
        fa[i] = mulmod_f64(fa[i], fb[i], p, inv_p);
    }
    ntt_f64_bailey(&mut fa, true, prime, root);
    fa.into_iter().map(|value| value as u64).collect()
}

fn crt_convolve_with(a: &[u64], b: &[u64], conv: fn(&[u64], &[u64], u64) -> Vec<u64>) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }
    let da = expand32(a);
    let db = expand32(b);
    let r0 = conv(&da, &db, P0);
    let r1 = conv(&da, &db, P1);
    let need = da.len() + db.len() - 1;
    crt_combine_raw(&r0, &r1, need)
}

fn crt_combine_raw(r0: &[u64], r1: &[u64], need: usize) -> Vec<u64> {
    let inv_p0 = crate::ntt::mod_pow(P0 % P1, P1 - 2, P1);
    let mut out32: Vec<u64> = Vec::with_capacity(need + 2);
    let mut carry: u128 = 0;
    for i in 0..need {
        let x0 = r0[i] as u128;
        let x1 = r1[i] as u128;
        let diff = (x1 + P1 as u128 - x0 % P1 as u128) % P1 as u128;
        let t = diff * inv_p0 as u128 % P1 as u128;
        let value = x0 + P0 as u128 * t + carry;
        out32.push((value & 0xFFFF_FFFF) as u64);
        carry = value >> 32;
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

fn check_cancelled(cancelled: &std::sync::atomic::AtomicBool) -> Result<(), crate::MultiplyError> {
    if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(crate::MultiplyError::cancelled());
    }
    Ok(())
}

fn ntt_f64_flat_cancelled(
    a: &mut [f64],
    invert: bool,
    prime: u64,
    root: u64,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<(), crate::MultiplyError> {
    if a.is_empty() {
        return Ok(());
    }
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    debug_assert!(n <= (1usize << P1_MAX_POW));
    bit_reverse(a);
    let mut bound = 1.0f64;
    for (width, radix4) in stage_widths(n) {
        check_cancelled(cancelled)?;
        if radix4 {
            stage_radix4(a, n, width, invert, prime, root, &mut bound);
        } else {
            stage_radix2(a, n, width, invert, prime, root);
        }
    }
    if invert {
        let p = prime as f64;
        let inv_p = inv_prime(prime);
        reduce_pass(a, p, inv_p);
        let n_inv = crate::ntt::mod_pow(n as u64, prime - 2, prime) as f64;
        for slot in a.iter_mut() {
            *slot = mulmod_f64(*slot, n_inv, p, inv_p);
        }
    } else {
        let p = prime as f64;
        let inv_p = inv_prime(prime);
        reduce_pass(a, p, inv_p);
    }
    Ok(())
}

fn convolution_prime_cancelled(
    a: &[u64],
    b: &[u64],
    prime: u64,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    let n = crate::ntt::next_pow2(a.len() + b.len() - 1);
    let p = prime as f64;
    let mut fa = vec![0.0f64; n];
    let mut fb = vec![0.0f64; n];
    for (i, digit) in a.iter().enumerate() {
        fa[i] = (digit % prime) as f64;
    }
    for (i, digit) in b.iter().enumerate() {
        fb[i] = (digit % prime) as f64;
    }
    let mut power = 0u32;
    while (1usize << power) < n {
        power += 1;
    }
    let root = primitive_root_2pow(prime, power);
    let inv_p = inv_prime(prime);
    check_cancelled(cancelled)?;
    ntt_f64_flat_cancelled(&mut fa, false, prime, root, cancelled)?;
    ntt_f64_flat_cancelled(&mut fb, false, prime, root, cancelled)?;
    for i in 0..n {
        fa[i] = mulmod_f64(fa[i], fb[i], p, inv_p);
    }
    ntt_f64_flat_cancelled(&mut fa, true, prime, root, cancelled)?;
    Ok(fa.into_iter().map(|value| value as u64).collect())
}

fn convolution_f64_cancelled(
    a: &[u64],
    b: &[u64],
    prime: u64,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    if a.is_empty() || b.is_empty() {
        return Ok(vec![0]);
    }
    let need = a.len() + b.len() - 1;
    let full = crate::ntt::next_pow2(need);
    if should_truncate(need, full) {
        Ok(convolution_f64_truncated(a, b, prime))
    } else {
        convolution_prime_cancelled(a, b, prime, cancelled)
    }
}

fn crt_convolve_with_cancelled(
    a: &[u64],
    b: &[u64],
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    if a.is_empty() || b.is_empty() {
        return Ok(vec![0]);
    }
    check_cancelled(cancelled)?;
    let da = expand32(a);
    let db = expand32(b);
    let r0 = convolution_f64_cancelled(&da, &db, P0, cancelled)?;
    let r1 = convolution_f64_cancelled(&da, &db, P1, cancelled)?;
    let need = da.len() + db.len() - 1;
    Ok(crt_combine_raw(&r0, &r1, need))
}

pub(crate) fn crt_convolution_small_cancelled(
    a: &[u64],
    b: &[u64],
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    crt_convolve_with_cancelled(a, b, cancelled)
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
    fn primes_have_form_order_and_primality() {
        assert_eq!(P0 % (1u64 << 44), 1);
        assert_eq!((P0 - 1) >> 44, 63);
        assert_eq!(P1 % (1u64 << 43), 1);
        assert_eq!((P1 - 1) >> 43, 75);
        assert!(is_probable_prime(P0));
        assert!(is_probable_prime(P1));
        assert!(!is_probable_prime(P0 - 1));
        assert!(!is_probable_prime(P1 - 1));
        assert!(!is_probable_prime(1));
    }

    #[test]
    fn roots_have_exact_two_power_order() {
        for (prime, max_pow) in [(P0, P0_MAX_POW), (P1, P1_MAX_POW)] {
            for l in 1..=8u32 {
                let w = primitive_root_2pow(prime, l);
                assert_eq!(crate::ntt::mod_pow(w, 1u64 << (l - 1), prime), prime - 1);
                assert_eq!(crate::ntt::mod_pow(w, 1u64 << l, prime), 1);
            }
            let w = primitive_root_2pow(prime, max_pow);
            assert_eq!(
                crate::ntt::mod_pow(w, 1u64 << (max_pow - 1), prime),
                prime - 1
            );
            assert_eq!(crate::ntt::mod_pow(w, 1u64 << max_pow, prime), 1);
        }
    }

    #[test]
    fn mulmod_f64_matches_u128_oracle() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c70);
        for prime in [P0, P1] {
            let p = prime as f64;
            let inv_p = inv_prime(prime);
            for _ in 0..10_000 {
                let a = rng.next_u64() % prime;
                let b = rng.next_u64() % prime;
                let expected = ((a as u128 * b as u128) % prime as u128) as u64;
                assert_eq!(mulmod_f64(a as f64, b as f64, p, inv_p) as u64, expected);
            }
            assert_eq!(mulmod_f64(0.0, 0.0, p, inv_p), 0.0);
            assert_eq!(mulmod_f64(1.0, 1.0, p, inv_p), 1.0);
            assert_eq!(
                mulmod_f64((prime - 1) as f64, (prime - 1) as f64, p, inv_p) as u64,
                1
            );
        }
    }

    #[test]
    fn forward_inverse_is_identity() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c71);
        for prime in [P0, P1] {
            let mut size = 1usize;
            while size <= 65536 {
                let original: Vec<f64> =
                    (0..size).map(|_| (rng.next_u64() % prime) as f64).collect();
                let mut work = original.clone();
                let mut power = 0u32;
                while (1usize << power) < size {
                    power += 1;
                }
                let root = primitive_root_2pow(prime, power.max(1));
                ntt_f64(&mut work, false, prime, root);
                ntt_f64(&mut work, true, prime, root);
                assert_eq!(work, original);
                size *= 2;
            }
        }
    }

    #[test]
    fn convolution_matches_schoolbook_residues() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c72);
        for _ in 0..50 {
            let la = rng.random_range(1..=128);
            let lb = rng.random_range(1..=128);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64() % 1000).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64() % 1000).collect();
            let expected = schoolbook::mul(&a, &b);
            for prime in [P0, P1] {
                let residues = convolution_f64(&a, &b, prime);
                assert_eq!(&residues[..expected.len()], expected.as_slice());
                assert!(residues[expected.len()..].iter().all(|limb| *limb == 0));
            }
        }
    }

    #[test]
    fn crt_convolution_small_matches_schoolbook() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c73);
        for _ in 0..30 {
            let la = rng.random_range(1..=64);
            let lb = rng.random_range(1..=64);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64()).collect();
            let result = crt_convolution_small(&a, &b);
            assert_eq!(result, schoolbook::mul(&a, &b));
            assert!(verify_mod(&a, &b, &result));
        }
        assert_eq!(crt_convolution_small(&[], &[1, 2]), vec![0]);
        assert_eq!(crt_convolution_small(&[1], &[]), vec![0]);
    }

    #[test]
    fn precancelled_convolution_is_cancelled() {
        let a = vec![3u64; 600];
        let b = vec![5u64; 600];
        let flag = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            crt_convolution_small_cancelled(&a, &b, &flag),
            Err(crate::MultiplyError::Cancelled(_))
        ));
        let clear = std::sync::atomic::AtomicBool::new(false);
        let result = crt_convolution_small_cancelled(&a, &b, &clear).expect("runs");
        assert_eq!(result, crt_convolution_small(&a, &b));
    }

    #[test]
    fn truncate_gate_pins() {
        assert_eq!(TRUNCATE_RATIO, 2);
        assert!(!should_truncate(0, 0));
        assert!(!should_truncate(0, 8));
        assert!(!should_truncate(1, 1));
        assert!(!should_truncate(5, 8));
        assert!(!should_truncate(7, 8));
        assert!(!should_truncate(4099, 8192));
        assert!(should_truncate(1, 2));
        assert!(should_truncate(3, 8));
        assert!(should_truncate(4, 8));
    }

    #[test]
    fn pruned_matches_full_bit_exact() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c77);
        for case in 0..50u32 {
            let prime = if case % 2 == 0 { P0 } else { P1 };
            let need = match case % 5 {
                0 => rng.random_range(1..=64),
                1 => rng.random_range(65..=300),
                2 => rng.random_range(301..=1200),
                3 => rng.random_range(1201..=2000),
                _ => rng.random_range(2001..=3000),
            };
            let base = crate::ntt::next_pow2(need);
            let full = if case % 3 == 0 { base } else { base * 2 };
            let input: Vec<f64> = (0..full)
                .map(|i| {
                    if i < need {
                        (rng.next_u64() % prime) as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            let mut power = 0u32;
            while (1usize << power) < full {
                power += 1;
            }
            let root = primitive_root_2pow(prime, power.max(1));
            let mut fast = input.clone();
            ntt_f64_forward_truncated(&mut fast, need, prime, root);
            let mut reference = input.clone();
            ntt_f64(&mut reference, false, prime, root);
            assert_eq!(fast, reference);
            let spectrum: Vec<f64> = (0..full).map(|_| (rng.next_u64() % prime) as f64).collect();
            let mut pruned = spectrum.clone();
            ntt_f64_inverse_truncated(&mut pruned, need, prime, root);
            let mut full_inv = spectrum;
            ntt_f64(&mut full_inv, true, prime, root);
            assert_eq!(&pruned[..need], &full_inv[..need]);
            assert!(pruned[need..].iter().all(|value| *value == 0.0));
            let la = rng.random_range(1..=need);
            let lb = need - la + 1;
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64() % 1000).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64() % 1000).collect();
            let truncated = convolution_f64_truncated(&a, &b, prime);
            let expected = convolution_f64_full(&a, &b, prime);
            let want = a.len() + b.len() - 1;
            assert_eq!(&truncated[..want], &expected[..want]);
        }
    }

    #[test]
    fn driver_matches_reference_twin() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c78);
        for (la, lb) in [(1usize, 1usize), (5, 7), (300, 700), (2050, 2050)] {
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64() % 1000).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64() % 1000).collect();
            for prime in [P0, P1] {
                assert_eq!(
                    convolution_f64(&a, &b, prime),
                    convolution_f64_full(&a, &b, prime)
                );
            }
        }
        assert_eq!(convolution_f64(&[], &[1, 2], P0), vec![0]);
        assert_eq!(convolution_f64_truncated(&[], &[1, 2], P0), vec![0]);
    }

    #[test]
    fn max_length_transform_runs() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c74);
        let n = 1usize << 20;
        let mut work: Vec<f64> = (0..n).map(|_| (rng.next_u64() % P0) as f64).collect();
        let root = primitive_root_2pow(P0, 20);
        ntt_f64(&mut work, false, P0, root);
        assert!(work.iter().all(|value| *value >= 0.0 && *value < P0 as f64));
        ntt_f64(&mut work, true, P0, root);
        assert_eq!(work.len(), n);
    }

    #[test]
    fn fast_matches_radix2_bit_exact() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c75);
        let mut cases = 0usize;
        for (size, count) in [(256usize, 80u32), (1024, 60), (4096, 40), (16384, 20)] {
            for _ in 0..count {
                for prime in [P0, P1] {
                    let original: Vec<f64> =
                        (0..size).map(|_| (rng.next_u64() % prime) as f64).collect();
                    let mut power = 0u32;
                    while (1usize << power) < size {
                        power += 1;
                    }
                    let root = primitive_root_2pow(prime, power.max(1));
                    let mut fast = original.clone();
                    ntt_f64(&mut fast, false, prime, root);
                    let mut reference = original.clone();
                    ntt_f64_radix2(&mut reference, false, prime, root);
                    assert_eq!(fast, reference);
                    ntt_f64(&mut fast, true, prime, root);
                    assert_eq!(fast, original);
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 400);
    }

    #[test]
    fn stage_twiddles_match_integer_roots() {
        for prime in [P0, P1] {
            for n in [4usize, 8, 256, 512, 1024, 4096, 16384, 65536] {
                let mut log = 0u32;
                while (1usize << log) < n {
                    log += 1;
                }
                let root = primitive_root_2pow(prime, log.max(1));
                for (width, _) in stage_widths(n) {
                    for invert in [false, true] {
                        let wlen = stage_wlen(prime, root, n, width, invert);
                        assert_eq!(crate::ntt::mod_pow(wlen, width as u64, prime), 1);
                        if width >= 2 {
                            assert_eq!(
                                crate::ntt::mod_pow(wlen, (width / 2) as u64, prime),
                                prime - 1
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn lazy_max_residue_round_trip_and_convolution() {
        let p0f = P0 as f64;
        let size = 1usize << 16;
        let original = vec![p0f - 1.0; size];
        let mut work = original.clone();
        let root = primitive_root_2pow(P0, 16);
        ntt_f64(&mut work, false, P0, root);
        assert!(work.iter().all(|value| value.fract() == 0.0));
        ntt_f64(&mut work, true, P0, root);
        assert_eq!(work, original);
        let limbs = vec![u64::MAX; 128];
        let expected = schoolbook::mul(&limbs, &limbs);
        let digits: Vec<u64> = limbs
            .iter()
            .flat_map(|limb| [limb & 0xFFFF_FFFF, limb >> 32])
            .collect();
        for prime in [P0, P1] {
            let residues = convolution_f64(&digits, &digits, prime);
            for (index, residue) in residues.iter().enumerate().take(digits.len() * 2 - 1) {
                let mut total: u128 = 0;
                let lo = index.saturating_sub(digits.len() - 1);
                let hi = index.min(digits.len() - 1);
                for j in lo..=hi {
                    total += digits[j] as u128 * digits[index - j] as u128;
                }
                assert_eq!(*residue as u128, total % prime as u128);
            }
        }
        assert_eq!(crt_convolution_small(&limbs, &limbs), expected);
    }

    #[test]
    fn simd_matches_scalar_bit_exact() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c76);
        let mut cases = 0u32;
        for (size, count) in [(256usize, 60u32), (1024, 60), (4096, 50), (16384, 30)] {
            for _ in 0..count {
                for prime in [P0, P1] {
                    let original: Vec<f64> =
                        (0..size).map(|_| (rng.next_u64() % prime) as f64).collect();
                    let mut power = 0u32;
                    while (1usize << power) < size {
                        power += 1;
                    }
                    let root = primitive_root_2pow(prime, power.max(1));
                    let mut dispatched = original.clone();
                    ntt_f64(&mut dispatched, false, prime, root);
                    let mut forced = original.clone();
                    ntt_f64_force_scalar(&mut forced, false, prime, root);
                    assert_eq!(dispatched, forced);
                    ntt_f64(&mut dispatched, true, prime, root);
                    ntt_f64_force_scalar(&mut forced, true, prime, root);
                    assert_eq!(dispatched, forced);
                    assert_eq!(dispatched, original);
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 400);
    }

    #[test]
    fn bailey_split_pins() {
        assert_eq!(BAILEY_MIN, 65536);
        assert_eq!(bailey_split(1024), (32, 32));
        assert_eq!(bailey_split(4096), (64, 64));
        assert_eq!(bailey_split(65536), (256, 256));
        assert_eq!(bailey_split(131072), (256, 512));
        assert_eq!(bailey_split(1usize << 20), (1024, 1024));
        for n in [1024usize, 4096, 65536, 131072, 1usize << 20] {
            let (n1, n2) = bailey_split(n);
            assert_eq!(n1 * n2, n);
            assert!(n1.is_power_of_two() && n2.is_power_of_two());
            assert!(n1 <= n2 && n2 / n1 <= 2);
        }
    }

    #[test]
    fn bailey_matches_flat_small() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c79);
        let mut cases = 0u32;
        for size in [1024usize, 4096, 16384] {
            for _ in 0..10 {
                for prime in [P0, P1] {
                    let original: Vec<f64> =
                        (0..size).map(|_| (rng.next_u64() % prime) as f64).collect();
                    let mut power = 0u32;
                    while (1usize << power) < size {
                        power += 1;
                    }
                    let root = primitive_root_2pow(prime, power.max(1));
                    let mut fast = original.clone();
                    ntt_f64_bailey(&mut fast, false, prime, root);
                    let mut reference = original.clone();
                    ntt_f64_flat(&mut reference, false, prime, root);
                    assert_eq!(fast, reference);
                    ntt_f64_bailey(&mut fast, true, prime, root);
                    assert_eq!(fast, original);
                    let mut spectrum: Vec<f64> =
                        (0..size).map(|_| (rng.next_u64() % prime) as f64).collect();
                    let expected = spectrum.clone();
                    ntt_f64_bailey(&mut spectrum, true, prime, root);
                    let mut flat_inv = expected.clone();
                    ntt_f64_flat(&mut flat_inv, true, prime, root);
                    assert_eq!(spectrum, flat_inv);
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 60);
    }

    #[test]
    fn bailey_matches_flat_large() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c7a);
        let mut cases = 0u32;
        for (size, count) in [(1usize << 16, 50u32), (1usize << 17, 20u32)] {
            for _ in 0..count {
                let prime = if cases % 2 == 0 { P0 } else { P1 };
                let original: Vec<f64> =
                    (0..size).map(|_| (rng.next_u64() % prime) as f64).collect();
                let mut power = 0u32;
                while (1usize << power) < size {
                    power += 1;
                }
                let root = primitive_root_2pow(prime, power.max(1));
                let mut fast = original.clone();
                ntt_f64_bailey(&mut fast, false, prime, root);
                let mut reference = original.clone();
                ntt_f64_flat(&mut reference, false, prime, root);
                assert_eq!(fast, reference);
                ntt_f64_bailey(&mut fast, true, prime, root);
                assert_eq!(fast, original);
                cases += 1;
            }
        }
        assert_eq!(cases, 70);
    }

    #[test]
    fn bailey_driver_and_convolution_pins() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c7b);
        for (size, prime) in [(1usize << 16, P0), (1usize << 17, P1)] {
            let original: Vec<f64> = (0..size).map(|_| (rng.next_u64() % prime) as f64).collect();
            let mut power = 0u32;
            while (1usize << power) < size {
                power += 1;
            }
            let root = primitive_root_2pow(prime, power.max(1));
            let mut driven = original.clone();
            ntt_f64(&mut driven, false, prime, root);
            let mut flat = original.clone();
            ntt_f64_flat(&mut flat, false, prime, root);
            assert_eq!(driven, flat);
            ntt_f64(&mut driven, true, prime, root);
            assert_eq!(driven, original);
        }
        let la = 20_000usize;
        let lb = 20_000usize;
        let a: Vec<u64> = (0..la).map(|_| rng.next_u64() % 1000).collect();
        let b: Vec<u64> = (0..lb).map(|_| rng.next_u64() % 1000).collect();
        assert_eq!(crate::ntt::next_pow2(la + lb - 1), 1usize << 16);
        for prime in [P0, P1] {
            assert_eq!(
                convolution_f64(&a, &b, prime),
                convolution_f64_full(&a, &b, prime)
            );
        }
    }

    #[test]
    fn simd_detection_report() {
        #[cfg(target_arch = "x86_64")]
        println!(
            "avx2_detected={} fma_detected={}",
            std::arch::is_x86_feature_detected!("avx2"),
            std::arch::is_x86_feature_detected!("fma")
        );
        #[cfg(not(target_arch = "x86_64"))]
        println!("simd_detected=false-non-x86_64");
    }

    fn parallel_case_limbs(seed: u64, limbs: usize) -> (Vec<u64>, Vec<u64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut build = || {
            let mut v: Vec<u64> = (0..limbs).map(|_| rng.next_u64() % 1_000_000).collect();
            v[limbs - 1] |= 1u64 << 63;
            v
        };
        (build(), build())
    }

    #[test]
    fn parallel_matches_serial_all_counts() {
        for (seed, limbs) in [(11u64, 64usize), (12, 1500), (13, 16384)] {
            let (a, b) = parallel_case_limbs(seed, limbs);
            let serial = measure_crt_flat(&a, &b);
            assert_eq!(crate::schonhage::mul_small(&a, &b), serial);
            assert_eq!(crate::schonhage::mul_par(&a, &b, 0), serial);
            assert_eq!(convolution_parallel(&a, &b, 1), serial);
            for workers in [2usize, 4, 8] {
                assert_eq!(convolution_parallel(&a, &b, workers), serial);
                assert_eq!(crate::schonhage::mul_par(&a, &b, workers), serial);
            }
            assert_eq!(
                convolution_parallel(&a, &b, 8),
                convolution_parallel(&a, &b, 8)
            );
            let auto = crate::options::MultiplyOptions::default().effective_workers();
            assert_eq!(convolution_parallel(&a, &b, auto), serial);
        }
    }

    #[test]
    fn parallel_bailey_rows_match_serial() {
        let mut rng = ChaCha8Rng::seed_from_u64(14u64);
        for prime in [P0, P1] {
            let size = 1usize << 16;
            let original: Vec<f64> = (0..size).map(|_| (rng.next_u64() % prime) as f64).collect();
            let mut power = 0u32;
            while (1usize << power) < size {
                power += 1;
            }
            let root = primitive_root_2pow(prime, power.max(1));
            let mut serial = original.clone();
            ntt_f64_bailey(&mut serial, false, prime, root);
            for workers in [2usize, 3, 5, 8] {
                let mut threaded = original.clone();
                ntt_f64_bailey_par(&mut threaded, false, prime, root, workers);
                assert_eq!(threaded, serial);
                ntt_f64_bailey_par(&mut threaded, true, prime, root, workers);
                assert_eq!(threaded, original);
            }
        }
    }

    #[test]
    fn parallel_smoke_timing() {
        let (a, b) = parallel_case_limbs(15u64, 16384);
        let start = std::time::Instant::now();
        let one = convolution_parallel(&a, &b, 1);
        let ms1 = start.elapsed().as_millis();
        let start = std::time::Instant::now();
        let four = convolution_parallel(&a, &b, 4);
        let ms4 = start.elapsed().as_millis();
        assert_eq!(one, four);
        eprintln!("parallel_smoke_16384 ms1={ms1} ms4={ms4}");
    }
}
