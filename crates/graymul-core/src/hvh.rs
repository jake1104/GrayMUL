use crate::furer;
use crate::limb::normalize_limbs;
use crate::ntt;
use crate::scratch::WorkScratch;

pub const HVH_CUTOFF: usize = 4096;
pub const SHORT_MAX: usize = 64;

pub fn mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.len().min(b.len()) <= HVH_CUTOFF {
        return furer::mul(a, b);
    }
    mul_pool(a, b, &mut WorkScratch::new())
}

fn mul_pool(a: &[u64], b: &[u64], pool: &mut WorkScratch) -> Vec<u64> {
    let mut ea = pool.take();
    expand_into(&mut ea, a);
    let mut eb = pool.take();
    expand_into(&mut eb, b);
    let conv = crt_two_prime(&ea, &eb);
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
    if a.len().min(b.len()) <= HVH_CUTOFF {
        return furer::mul_buf(a, b, work);
    }
    let mut pool = WorkScratch::new();
    pool.give(std::mem::take(work));
    let out = mul_pool(a, b, &mut pool);
    *work = pool.take();
    out
}

pub fn predicted_work_limbs(n: usize) -> f64 {
    let width = n as f64;
    width * width.ln().max(1.0)
}

pub fn bluestein_convolution(a: &[u64], b: &[u64], prime: u64, root: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }
    let need = a.len() + b.len() - 1;
    match next_chirp_len(need, prime) {
        Some(n) => bluestein_convolution_at(a, b, prime, root, n),
        None => ntt::convolution_mod(a, b, prime, root),
    }
}

fn next_chirp_len(need: usize, prime: u64) -> Option<usize> {
    let mut n = need.max(1);
    while (n as u64) < prime {
        if bluestein_feasible(n, prime) {
            return Some(n);
        }
        n += 1;
    }
    None
}

fn bluestein_feasible(n: usize, prime: u64) -> bool {
    if n <= 1 {
        return true;
    }
    if n % 2 == 1 {
        (prime - 1) % (n as u64) == 0
    } else {
        (prime - 1) % (2 * n as u64) == 0
    }
}

fn chirp_root(prime: u64, root: u64, n: usize, w: u64, invert: bool) -> u64 {
    if n % 2 == 1 {
        ntt::mod_pow(w, (n as u64).div_ceil(2), prime)
    } else {
        debug_assert!((prime - 1) % (2 * n as u64) == 0);
        let forward = ntt::mod_pow(root, (prime - 1) / (2 * n as u64), prime);
        if invert {
            ntt::mod_pow(forward, prime - 2, prime)
        } else {
            forward
        }
    }
}

fn bluestein_dft(a: &mut [u32], invert: bool, prime: u64, root: u64) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    let w = ntt::primitive_root_pow(prime, root, n);
    let w = if invert {
        ntt::mod_pow(w, prime - 2, prime)
    } else {
        w
    };
    let s = chirp_root(prime, root, n, w, invert);
    let s2 = (s as u128 * s as u128 % prime as u128) as u64;
    let mut pw = vec![1u64; n];
    let mut odd = s;
    for i in 0..n - 1 {
        pw[i + 1] = pw[i] * odd % prime;
        odd = odd * s2 % prime;
    }
    let s_inv = ntt::mod_pow(s, prime - 2, prime);
    let s_inv2 = (s_inv as u128 * s_inv as u128 % prime as u128) as u64;
    let mut ipw = vec![1u64; n];
    let mut iodd = s_inv;
    for i in 0..n - 1 {
        ipw[i + 1] = ipw[i] * iodd % prime;
        iodd = iodd * s_inv2 % prime;
    }
    let m = ntt::next_pow2(2 * n - 1);
    let mut fa = vec![0u32; m];
    let mut fb = vec![0u32; m];
    for (j, slot) in fa.iter_mut().enumerate().take(n) {
        *slot = ((u64::from(a[j]) * pw[j]) % prime) as u32;
    }
    fb[0] = 1;
    for t in 1..n {
        fb[t] = ipw[t] as u32;
        fb[m - t] = ipw[t] as u32;
    }
    ntt::ntt(&mut fa, false, prime, root);
    ntt::ntt(&mut fb, false, prime, root);
    for i in 0..m {
        fa[i] = ((u64::from(fa[i]) * u64::from(fb[i])) % prime) as u32;
    }
    ntt::ntt(&mut fa, true, prime, root);
    for k in 0..n {
        a[k] = ((u64::from(fa[k]) * pw[k]) % prime) as u32;
    }
    if invert {
        let n_inv = ntt::mod_pow(n as u64, prime - 2, prime);
        for slot in a.iter_mut() {
            *slot = ((u64::from(*slot) * n_inv) % prime) as u32;
        }
    }
}

fn bluestein_convolution_at(a: &[u64], b: &[u64], prime: u64, root: u64, n: usize) -> Vec<u64> {
    let mut fa = vec![0u32; n];
    let mut fb = vec![0u32; n];
    for (i, limb) in a.iter().enumerate() {
        fa[i] = (limb % prime) as u32;
    }
    for (i, limb) in b.iter().enumerate() {
        fb[i] = (limb % prime) as u32;
    }
    bluestein_dft(&mut fa, false, prime, root);
    bluestein_dft(&mut fb, false, prime, root);
    for i in 0..n {
        fa[i] = ((u64::from(fa[i]) * u64::from(fb[i])) % prime) as u32;
    }
    bluestein_dft(&mut fa, true, prime, root);
    fa.into_iter().map(u64::from).collect()
}

fn dft_naive(a: &mut [u32], invert: bool, prime: u64, root: u64) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    let w = ntt::primitive_root_pow(prime, root, n);
    let w = if invert {
        ntt::mod_pow(w, prime - 2, prime)
    } else {
        w
    };
    let mut wp = vec![1u64; n];
    for k in 1..n {
        wp[k] = wp[k - 1] * w % prime;
    }
    let mut out = vec![0u64; n];
    for (k, slot) in out.iter_mut().enumerate() {
        let mut acc = 0u64;
        for (j, value) in a.iter().enumerate() {
            acc = (acc + u64::from(*value) * wp[(j * k) % n]) % prime;
        }
        *slot = acc;
    }
    if invert {
        let n_inv = ntt::mod_pow(n as u64, prime - 2, prime);
        for value in out.iter_mut() {
            *value = *value * n_inv % prime;
        }
    }
    for (i, value) in out.into_iter().enumerate() {
        a[i] = value as u32;
    }
}

fn dft_forward(v: &mut [u32], prime: u64, root: u64) {
    let n = v.len();
    if n <= 1 {
        return;
    }
    if n <= SHORT_MAX {
        dft_naive(v, false, prime, root);
    } else if n.is_power_of_two() {
        ntt::ntt(v, false, prime, root);
    } else {
        debug_assert!(bluestein_feasible(n, prime));
        bluestein_dft(v, false, prime, root);
    }
}

fn dft_inverse(v: &mut [u32], prime: u64, root: u64) {
    let n = v.len();
    if n <= 1 {
        return;
    }
    if n <= SHORT_MAX {
        dft_naive(v, true, prime, root);
    } else if n.is_power_of_two() {
        ntt::ntt(v, true, prime, root);
    } else {
        debug_assert!(bluestein_feasible(n, prime));
        bluestein_dft(v, true, prime, root);
    }
}

fn mod_inv(a: u64, m: u64) -> u64 {
    let mut t = 0i128;
    let mut nt = 1i128;
    let mut r = m as i128;
    let mut nr = a as i128 % m as i128;
    while nr != 0 {
        let q = r / nr;
        let tmp = nt;
        nt = t - q * nt;
        t = tmp;
        let tmpr = nr;
        nr = r - q * nr;
        r = tmpr;
    }
    if t < 0 {
        t += m as i128;
    }
    t as u64
}

const ODD_CANDIDATES: [usize; 5] = [1, 7, 17, 119, 479];

fn pfa_shape(need: usize, prime: u64) -> (usize, usize) {
    let mut best = (ntt::next_pow2(need.max(1)), 1usize);
    for cols in ODD_CANDIDATES {
        if (prime - 1) % (cols as u64) != 0 {
            continue;
        }
        let rows = ntt::next_pow2(need.max(1).div_ceil(cols));
        if rows > (1usize << 21) {
            continue;
        }
        if rows * cols < best.0 * best.1 {
            best = (rows, cols);
        }
    }
    best
}

fn pfa_index_in(n1: usize, n2: usize, rows: usize, cols: usize) -> usize {
    (n1 * cols + n2 * rows) % (rows * cols)
}

fn pfa_index_out(k1: usize, k2: usize, rows: usize, cols: usize, c_r: u64, r_c: u64) -> usize {
    let n = (rows * cols) as u64;
    let first = k1 as u64 * cols as u64 % n * c_r % n;
    let second = k2 as u64 * rows as u64 % n * r_c % n;
    ((first + second) % n) as usize
}

fn pfa_forward(f: &mut [u32], rows: usize, cols: usize, prime: u64, root: u64) {
    let n = rows * cols;
    debug_assert_eq!(f.len(), n);
    let c_r = mod_inv(cols as u64 % rows as u64, rows as u64);
    let r_c = mod_inv(rows as u64 % cols as u64, cols as u64);
    let mut grid = vec![0u32; n];
    for n1 in 0..rows {
        for n2 in 0..cols {
            grid[n1 * cols + n2] = f[pfa_index_in(n1, n2, rows, cols)];
        }
    }
    for r in 0..rows {
        dft_forward(&mut grid[r * cols..(r + 1) * cols], prime, root);
    }
    let mut col = vec![0u32; rows];
    for c in 0..cols {
        for r in 0..rows {
            col[r] = grid[r * cols + c];
        }
        dft_forward(&mut col, prime, root);
        for r in 0..rows {
            grid[r * cols + c] = col[r];
        }
    }
    let mut out = vec![0u32; n];
    for k1 in 0..rows {
        for k2 in 0..cols {
            out[pfa_index_out(k1, k2, rows, cols, c_r, r_c)] = grid[k1 * cols + k2];
        }
    }
    f.copy_from_slice(&out);
}

fn pfa_inverse(f: &mut [u32], rows: usize, cols: usize, prime: u64, root: u64) {
    let n = rows * cols;
    debug_assert_eq!(f.len(), n);
    let c_r = mod_inv(cols as u64 % rows as u64, rows as u64);
    let r_c = mod_inv(rows as u64 % cols as u64, cols as u64);
    let mut grid = vec![0u32; n];
    for k1 in 0..rows {
        for k2 in 0..cols {
            grid[k1 * cols + k2] = f[pfa_index_out(k1, k2, rows, cols, c_r, r_c)];
        }
    }
    let mut col = vec![0u32; rows];
    for c in 0..cols {
        for r in 0..rows {
            col[r] = grid[r * cols + c];
        }
        dft_inverse(&mut col, prime, root);
        for r in 0..rows {
            grid[r * cols + c] = col[r];
        }
    }
    for r in 0..rows {
        dft_inverse(&mut grid[r * cols..(r + 1) * cols], prime, root);
    }
    let mut out = vec![0u32; n];
    for n1 in 0..rows {
        for n2 in 0..cols {
            out[pfa_index_in(n1, n2, rows, cols)] = grid[n1 * cols + n2];
        }
    }
    f.copy_from_slice(&out);
}

fn convolution_pfa(a: &[u64], b: &[u64], prime: u64, root: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }
    let (rows, cols) = pfa_shape(a.len() + b.len() - 1, prime);
    let total = rows * cols;
    let mut fa = vec![0u32; total];
    let mut fb = vec![0u32; total];
    for (i, limb) in a.iter().enumerate() {
        fa[i] = (limb % prime) as u32;
    }
    for (i, limb) in b.iter().enumerate() {
        fb[i] = (limb % prime) as u32;
    }
    pfa_forward(&mut fa, rows, cols, prime, root);
    pfa_forward(&mut fb, rows, cols, prime, root);
    for i in 0..total {
        fa[i] = ((u64::from(fa[i]) * u64::from(fb[i])) % prime) as u32;
    }
    pfa_inverse(&mut fa, rows, cols, prime, root);
    fa.into_iter().map(u64::from).collect()
}

fn crt_two_prime(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }
    let r1 = convolution_pfa(a, b, ntt::P1, ntt::P1_ROOT);
    let r2 = convolution_pfa(a, b, ntt::P2, ntt::P2_ROOT);
    let inv_p1 = ntt::mod_pow(ntt::P1 % ntt::P2, ntt::P2 - 2, ntt::P2);
    let need = a.len() + b.len() - 1;
    let mut out = Vec::with_capacity(need);
    for i in 0..need {
        let x1 = r1[i];
        let x2 = r2[i];
        let diff = (x2 + ntt::P2 - x1 % ntt::P2) % ntt::P2;
        let t = (diff as u128 * inv_p1 as u128) % ntt::P2 as u128;
        let value = x1 as u128 + ntt::P1 as u128 * t;
        out.push(value as u64);
    }
    normalize_limbs(&mut out);
    out
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
    use crate::schonhage;
    use crate::schoolbook;
    use crate::verify::verify_mod;
    use rand::Rng;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn random_limbs(rng: &mut ChaCha8Rng, len: usize) -> Vec<u64> {
        (0..len).map(|_| rng.next_u64()).collect()
    }

    fn random_small(rng: &mut ChaCha8Rng, len: usize, bound: u64) -> Vec<u64> {
        (0..len).map(|_| rng.next_u64() % bound).collect()
    }

    #[test]
    fn short_path_matches_ntt() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c35);
        for (prime, root) in [(ntt::P1, ntt::P1_ROOT), (ntt::P2, ntt::P2_ROOT)] {
            let mut size = 1usize;
            while size <= SHORT_MAX {
                let original: Vec<u32> =
                    (0..size).map(|_| (rng.next_u64() % prime) as u32).collect();
                let mut short = original.clone();
                dft_naive(&mut short, false, prime, root);
                let mut full = original.clone();
                ntt::ntt(&mut full, false, prime, root);
                assert_eq!(short, full);
                dft_naive(&mut short, true, prime, root);
                assert_eq!(short, original);
                size *= 2;
            }
        }
    }

    #[test]
    fn bluestein_matches_naive_on_feasible_lengths() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c36);
        for (prime, root, lengths) in [
            (ntt::P1, ntt::P1_ROOT, vec![7usize, 14, 17, 28, 34, 56]),
            (ntt::P2, ntt::P2_ROOT, vec![479usize]),
        ] {
            for n in lengths {
                let original: Vec<u32> = (0..n).map(|_| (rng.next_u64() % prime) as u32).collect();
                let mut expect = original.clone();
                dft_naive(&mut expect, false, prime, root);
                let mut got = original.clone();
                bluestein_dft(&mut got, false, prime, root);
                assert_eq!(got, expect);
                bluestein_dft(&mut got, true, prime, root);
                assert_eq!(got, original);
            }
        }
    }

    #[test]
    fn bluestein_convolution_non_pow2_lengths() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c37);
        let p1_needs = [
            7usize, 14, 17, 28, 34, 56, 68, 119, 136, 238, 272, 476, 544, 952, 1088, 1904, 2176,
            3808, 4352, 7616, 8704, 15232, 17408, 30464, 34816,
        ];
        for need in p1_needs {
            let la = need / 2 + 1;
            let lb = need - la + 1;
            let a = random_small(&mut rng, la, 100);
            let b = random_small(&mut rng, lb, 100);
            let residues = bluestein_convolution(&a, &b, ntt::P1, ntt::P1_ROOT);
            let expected = schoolbook::mul(&a, &b);
            assert_eq!(&residues[..expected.len()], expected.as_slice());
            assert!(residues[expected.len()..].iter().all(|limb| *limb == 0));
        }
        let p2_shapes = [
            (240usize, 240usize),
            (100, 380),
            (479, 1),
            (500, 459),
            (300, 659),
            (958, 2),
            (700, 259),
            (1000, 917),
            (400, 1517),
            (1916, 1),
            (1500, 417),
            (958, 958),
            (2000, 1917),
            (100, 1817),
            (3832, 1),
            (2500, 1333),
            (3000, 833),
            (1916, 1916),
            (500, 3333),
            (1200, 2633),
            (3832, 100),
            (7664, 1),
            (4000, 3665),
            (1916, 5749),
            (7664, 7664),
        ];
        for (la, lb) in p2_shapes {
            let a = random_small(&mut rng, la, 100);
            let b = random_small(&mut rng, lb, 100);
            let residues = bluestein_convolution(&a, &b, ntt::P2, ntt::P2_ROOT);
            let expected = schoolbook::mul(&a, &b);
            assert_eq!(&residues[..expected.len()], expected.as_slice());
            assert!(residues[expected.len()..].iter().all(|limb| *limb == 0));
        }
    }

    #[test]
    fn pfa_forward_matches_naive() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c44);
        for (prime, root, shapes) in [
            (
                ntt::P1,
                ntt::P1_ROOT,
                vec![
                    (2usize, 7usize),
                    (2, 17),
                    (8, 7),
                    (4, 17),
                    (1, 119),
                    (2, 119),
                ],
            ),
            (ntt::P2, ntt::P2_ROOT, vec![(1usize, 479usize), (2, 479)]),
        ] {
            for (rows, cols) in shapes {
                let n = rows * cols;
                let original: Vec<u32> = (0..n).map(|_| (rng.next_u64() % prime) as u32).collect();
                let mut expect = original.clone();
                dft_naive(&mut expect, false, prime, root);
                let mut got = original.clone();
                pfa_forward(&mut got, rows, cols, prime, root);
                assert_eq!(got, expect);
                pfa_inverse(&mut got, rows, cols, prime, root);
                assert_eq!(got, original);
            }
        }
    }

    #[test]
    fn pfa_convolution_matches_plain() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c38);
        for (prime, root) in [(ntt::P1, ntt::P1_ROOT), (ntt::P2, ntt::P2_ROOT)] {
            for _ in 0..20 {
                let la = rng.random_range(1..=300);
                let lb = rng.random_range(1..=300);
                let a = random_small(&mut rng, la, 1000);
                let b = random_small(&mut rng, lb, 1000);
                let residues = convolution_pfa(&a, &b, prime, root);
                let expected = ntt::convolution_mod(&a, &b, prime, root);
                let need = la + lb - 1;
                assert_eq!(&residues[..need], &expected[..need]);
                assert!(residues[need..].iter().all(|limb| *limb == 0));
                assert!(expected[need..].iter().all(|limb| *limb == 0));
            }
        }
    }

    #[test]
    fn below_cutoff_delegates_to_furer() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c39);
        for len in [10usize, 1000, HVH_CUTOFF] {
            let a = random_limbs(&mut rng, len);
            let b = random_limbs(&mut rng, len + 5);
            assert_eq!(mul(&a, &b), furer::mul(&a, &b));
        }
    }

    #[test]
    fn ssa_identity_above_cutoff() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c40);
        for (la, lb) in [(4100usize, 4100usize), (4200, 4100), (4500, 4500)] {
            let a = random_limbs(&mut rng, la);
            let b = random_limbs(&mut rng, lb);
            let result = mul(&a, &b);
            assert_eq!(result, schonhage::mul(&a, &b));
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn matches_schoolbook_below_cutoff() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c43);
        let a = random_limbs(&mut rng, HVH_CUTOFF - 1);
        let b = random_limbs(&mut rng, HVH_CUTOFF - 1);
        assert_eq!(mul(&a, &b), schoolbook::mul(&a, &b));
    }

    #[test]
    fn randomized_cross_validation_vs_ssa() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c41);
        for _ in 0..100 {
            let la = rng.random_range(0..=3000);
            let lb = rng.random_range(0..=3000);
            let a = random_limbs(&mut rng, la);
            let b = random_limbs(&mut rng, lb);
            let result = mul(&a, &b);
            assert_eq!(result, schonhage::mul(&a, &b));
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn zero_and_one_identities() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c42);
        let big = random_limbs(&mut rng, 2200);
        assert_eq!(mul(&vec![0; 2200], &big), vec![0]);
        assert_eq!(mul(&big, &vec![0; 2200]), vec![0]);
        assert_eq!(mul(&[1], &big), big);
        assert_eq!(mul(&big, &[1]), big);
        assert_eq!(mul(&[], &big), vec![0]);
    }

    #[test]
    fn predicted_work_matches_hvh_shape() {
        assert_eq!(predicted_work_limbs(0), 0.0);
        assert_eq!(predicted_work_limbs(1), 1.0);
        let width = 4096f64;
        let expected = width * width.ln();
        assert_eq!(predicted_work_limbs(4096), expected);
    }

    #[test]
    fn scratch_buffer_matches_fresh_allocation() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c43);
        for len in [100usize, 2200] {
            let a = random_limbs(&mut rng, len);
            let b = random_limbs(&mut rng, len);
            let mut work = vec![0xDEAD_BEEF_DEAD_BEEFu64; 16];
            assert_eq!(mul_buf(&a, &b, &mut work), mul(&a, &b));
        }
    }

    #[test]
    fn buf_matches_mul_across_cutoff() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c44);
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
        for len in [4200usize, 4500] {
            let a = random_limbs(&mut rng, len);
            let b = random_limbs(&mut rng, len);
            let expected = mul(&a, &b);
            let mut work = Vec::new();
            assert_eq!(mul_buf(&a, &b, &mut work), expected);
            assert!(verify_mod(&a, &b, &expected));
        }
    }
}
