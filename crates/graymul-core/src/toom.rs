use crate::karatsuba;
use crate::limb::normalize_limbs;
use std::cmp::Ordering;

pub const TOOM3_CUTOFF: usize = 64;
pub const TOOM4_CUTOFF: usize = 512;

pub fn mul_toom3(a: &[u64], b: &[u64]) -> Vec<u64> {
    mul_toom3_work(a, b, &mut Vec::new())
}

fn mul_toom3_work(a: &[u64], b: &[u64], work: &mut Vec<u64>) -> Vec<u64> {
    if a.len().min(b.len()) <= TOOM3_CUTOFF {
        return karatsuba::mul_buf(a, b, work);
    }
    let m = a.len().max(b.len()).div_ceil(3);
    let (a0, a1, a2) = split3(a, m);
    let (b0, b1, b2) = split3(b, m);
    toom3_raw(a0, a1, a2, b0, b1, b2, m, work)
}

pub(crate) fn mul_cancelled(
    a: &[u64],
    b: &[u64],
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    if a.len().min(b.len()) <= TOOM3_CUTOFF {
        return karatsuba::mul_cancelled(a, b, cancelled);
    }
    if a.len().max(b.len()) >= TOOM4_CUTOFF {
        return mul_toom4_cancelled(a, b, cancelled);
    }
    mul_toom3_cancelled(a, b, cancelled)
}

pub(crate) fn mul_toom3_cancelled(
    a: &[u64],
    b: &[u64],
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    mul_toom3_work_cancelled(a, b, &mut Vec::new(), cancelled)
}

fn mul_toom3_work_cancelled(
    a: &[u64],
    b: &[u64],
    work: &mut Vec<u64>,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(crate::MultiplyError::cancelled());
    }
    if a.len().min(b.len()) <= TOOM3_CUTOFF {
        return karatsuba::mul_cancelled(a, b, cancelled);
    }
    let m = a.len().max(b.len()).div_ceil(3);
    let (a0, a1, a2) = split3(a, m);
    let (b0, b1, b2) = split3(b, m);
    toom3_raw_cancelled(a0, a1, a2, b0, b1, b2, m, work, cancelled)
}

pub fn mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.len().min(b.len()) <= TOOM3_CUTOFF {
        return karatsuba::mul(a, b);
    }
    if a.len().max(b.len()) >= TOOM4_CUTOFF {
        return mul_toom4(a, b);
    }
    mul_toom3(a, b)
}

pub fn mul_buf(a: &[u64], b: &[u64], work: &mut Vec<u64>) -> Vec<u64> {
    if a.len().min(b.len()) <= TOOM3_CUTOFF {
        return karatsuba::mul_buf(a, b, work);
    }
    if a.len().max(b.len()) >= TOOM4_CUTOFF {
        return mul_toom4_work(a, b, work);
    }
    mul_toom3_work(a, b, work)
}

pub fn predicted_work_limbs(n: usize) -> f64 {
    (n as f64).powf(1.465)
}

pub fn predicted_work_limbs_toom4(n: usize) -> f64 {
    (n as f64).powf(1.404)
}

struct Signed {
    neg: bool,
    mag: Vec<u64>,
}

fn mag_is_zero(mag: &[u64]) -> bool {
    mag.iter().all(|limb| *limb == 0)
}

fn signed_new(neg: bool, mut mag: Vec<u64>) -> Signed {
    normalize_limbs(&mut mag);
    let neg = neg && !mag_is_zero(&mag);
    Signed { neg, mag }
}

fn negate(x: &Signed) -> Signed {
    Signed {
        neg: !x.neg && !mag_is_zero(&x.mag),
        mag: x.mag.clone(),
    }
}

fn cmp_mag(a: &[u64], b: &[u64]) -> Ordering {
    let mut la = a.len();
    while la > 0 && a[la - 1] == 0 {
        la -= 1;
    }
    let mut lb = b.len();
    while lb > 0 && b[lb - 1] == 0 {
        lb -= 1;
    }
    if la != lb {
        return la.cmp(&lb);
    }
    for i in (0..la).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    Ordering::Equal
}

fn unsigned_add(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(a.len().max(b.len()) + 1);
    let mut carry: u128 = 0;
    for i in 0..a.len().max(b.len()) {
        let cur = u128::from(*a.get(i).unwrap_or(&0)) + u128::from(*b.get(i).unwrap_or(&0)) + carry;
        out.push(cur as u64);
        carry = cur >> 64;
    }
    if carry > 0 {
        out.push(carry as u64);
    }
    out
}

fn sub_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = a.to_vec();
    let mut borrow: i128 = 0;
    for (i, slot) in out.iter_mut().enumerate() {
        let x = *slot as i128 - borrow;
        let y = b.get(i).copied().unwrap_or(0) as i128;
        let diff = x - y;
        if diff < 0 {
            *slot = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            *slot = diff as u64;
            borrow = 0;
        }
    }
    debug_assert_eq!(borrow, 0);
    normalize_limbs(&mut out);
    out
}

fn add_signed(x: &Signed, y: &Signed) -> Signed {
    if x.neg == y.neg {
        signed_new(x.neg, unsigned_add(&x.mag, &y.mag))
    } else {
        match cmp_mag(&x.mag, &y.mag) {
            Ordering::Greater => signed_new(x.neg, sub_mag(&x.mag, &y.mag)),
            Ordering::Less => signed_new(y.neg, sub_mag(&y.mag, &x.mag)),
            Ordering::Equal => signed_new(false, vec![0]),
        }
    }
}

fn sub_signed(x: &Signed, y: &Signed) -> Signed {
    add_signed(x, &negate(y))
}

fn mul_signed_buf(x: &Signed, y: &Signed, work: &mut Vec<u64>) -> Signed {
    signed_new(x.neg != y.neg, karatsuba::mul_buf(&x.mag, &y.mag, work))
}

fn mul_signed_buf_cancelled(
    x: &Signed,
    y: &Signed,
    scratch: &mut karatsuba::WorkScratch,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Signed, crate::MultiplyError> {
    if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(crate::MultiplyError::cancelled());
    }
    Ok(signed_new(
        x.neg != y.neg,
        karatsuba::mul_scratch_cancelled(&x.mag, &y.mag, scratch, cancelled)?,
    ))
}

fn mul_small_mag(value: &[u64], k: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(value.len() + 1);
    let mut carry: u128 = 0;
    for limb in value {
        let cur = u128::from(*limb) * u128::from(k) + carry;
        out.push(cur as u64);
        carry = cur >> 64;
    }
    if carry > 0 {
        out.push(carry as u64);
    }
    normalize_limbs(&mut out);
    out
}

fn div_small_assign(mag: &mut Vec<u64>, d: u64) {
    let mut rem: u128 = 0;
    for i in (0..mag.len()).rev() {
        let cur = (rem << 64) | u128::from(mag[i]);
        mag[i] = (cur / u128::from(d)) as u64;
        rem = cur % u128::from(d);
    }
    debug_assert_eq!(rem, 0);
    normalize_limbs(mag);
}

fn div_signed_exact(x: &Signed, d: u64) -> Signed {
    let mut mag = x.mag.clone();
    div_small_assign(&mut mag, d);
    signed_new(x.neg, mag)
}

fn add_shifted_assign(out: &mut Vec<u64>, add: &[u64], shift: usize) {
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

fn split3(value: &[u64], m: usize) -> (&[u64], &[u64], &[u64]) {
    let c1 = m.min(value.len());
    let c2 = (2 * m).min(value.len());
    (&value[..c1], &value[c1..c2], &value[c2..])
}

fn eval_toom3(p0: &[u64], p1: &[u64], p2: &[u64]) -> (Signed, Signed, Signed, Signed, Signed) {
    let at0 = signed_new(false, p0.to_vec());
    let at1 = signed_new(false, unsigned_add(&unsigned_add(p0, p1), p2));
    let atm1 = sub_signed(
        &signed_new(false, unsigned_add(p0, p2)),
        &signed_new(false, p1.to_vec()),
    );
    let at2 = signed_new(
        false,
        unsigned_add(
            &unsigned_add(p0, &mul_small_mag(p1, 2)),
            &mul_small_mag(p2, 4),
        ),
    );
    let atinf = signed_new(false, p2.to_vec());
    (at0, at1, atm1, at2, atinf)
}

fn interpolate(
    v0: &Signed,
    v1: &Signed,
    vm1: &Signed,
    v2: &Signed,
    vinf: &Signed,
) -> [Vec<u64>; 5] {
    let w0 = v0.mag.clone();
    let w4 = vinf.mag.clone();
    let half = div_signed_exact(&add_signed(v1, vm1), 2);
    let w2s = sub_signed(&sub_signed(&half, v0), vinf);
    let t = div_signed_exact(&sub_signed(v1, vm1), 2);
    let term_w2 = mul_small_mag(&w2s.mag, 4);
    let term_w4 = mul_small_mag(&w4, 16);
    let term_t = Signed {
        neg: t.neg,
        mag: mul_small_mag(&t.mag, 2),
    };
    let u = sub_signed(
        &sub_signed(
            &sub_signed(&sub_signed(v2, v0), &signed_new(false, term_w2)),
            &signed_new(false, term_w4),
        ),
        &term_t,
    );
    let w3s = div_signed_exact(&div_signed_exact(&u, 2), 3);
    let w1s = sub_signed(&t, &w3s);
    debug_assert!(!w1s.neg && !w2s.neg && !w3s.neg);
    [w0, w1s.mag, w2s.mag, w3s.mag, w4]
}

#[allow(clippy::too_many_arguments)]
fn toom3_raw(
    a0: &[u64],
    a1: &[u64],
    a2: &[u64],
    b0: &[u64],
    b1: &[u64],
    b2: &[u64],
    m: usize,
    work: &mut Vec<u64>,
) -> Vec<u64> {
    let (aa0, aa1, aam1, aa2, aainf) = eval_toom3(a0, a1, a2);
    let (bb0, bb1, bbm1, bb2, bbinf) = eval_toom3(b0, b1, b2);
    let v0 = mul_signed_buf(&aa0, &bb0, work);
    let v1 = mul_signed_buf(&aa1, &bb1, work);
    let vm1 = mul_signed_buf(&aam1, &bbm1, work);
    let v2 = mul_signed_buf(&aa2, &bb2, work);
    let vinf = mul_signed_buf(&aainf, &bbinf, work);
    let parts = interpolate(&v0, &v1, &vm1, &v2, &vinf);
    let mut out = parts[0].clone();
    add_shifted_assign(&mut out, &parts[1], m);
    add_shifted_assign(&mut out, &parts[2], 2 * m);
    add_shifted_assign(&mut out, &parts[3], 3 * m);
    add_shifted_assign(&mut out, &parts[4], 4 * m);
    normalize_limbs(&mut out);
    out
}

#[allow(clippy::too_many_arguments)]
fn toom3_raw_cancelled(
    a0: &[u64],
    a1: &[u64],
    a2: &[u64],
    b0: &[u64],
    b1: &[u64],
    b2: &[u64],
    m: usize,
    work: &mut Vec<u64>,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(crate::MultiplyError::cancelled());
    }
    let (aa0, aa1, aam1, aa2, aainf) = eval_toom3(a0, a1, a2);
    let (bb0, bb1, bbm1, bb2, bbinf) = eval_toom3(b0, b1, b2);
    let mut scratch = karatsuba::WorkScratch::new();
    scratch.give(std::mem::take(work));
    let v0 = mul_signed_buf_cancelled(&aa0, &bb0, &mut scratch, cancelled)?;
    let v1 = mul_signed_buf_cancelled(&aa1, &bb1, &mut scratch, cancelled)?;
    let vm1 = mul_signed_buf_cancelled(&aam1, &bbm1, &mut scratch, cancelled)?;
    let v2 = mul_signed_buf_cancelled(&aa2, &bb2, &mut scratch, cancelled)?;
    let vinf = mul_signed_buf_cancelled(&aainf, &bbinf, &mut scratch, cancelled)?;
    *work = scratch.take();
    let parts = interpolate(&v0, &v1, &vm1, &v2, &vinf);
    let mut out = parts[0].clone();
    add_shifted_assign(&mut out, &parts[1], m);
    add_shifted_assign(&mut out, &parts[2], 2 * m);
    add_shifted_assign(&mut out, &parts[3], 3 * m);
    add_shifted_assign(&mut out, &parts[4], 4 * m);
    normalize_limbs(&mut out);
    Ok(out)
}

pub fn mul_toom4(a: &[u64], b: &[u64]) -> Vec<u64> {
    mul_toom4_work(a, b, &mut Vec::new())
}

fn mul_toom4_work(a: &[u64], b: &[u64], work: &mut Vec<u64>) -> Vec<u64> {
    if a.len().min(b.len()) <= TOOM3_CUTOFF {
        return karatsuba::mul_buf(a, b, work);
    }
    let m = a.len().max(b.len()).div_ceil(4);
    let (a0, a1, a2, a3) = split4(a, m);
    let (b0, b1, b2, b3) = split4(b, m);
    toom4_raw(a0, a1, a2, a3, b0, b1, b2, b3, m, work)
}

pub(crate) fn mul_toom4_cancelled(
    a: &[u64],
    b: &[u64],
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    mul_toom4_work_cancelled(a, b, &mut Vec::new(), cancelled)
}

fn mul_toom4_work_cancelled(
    a: &[u64],
    b: &[u64],
    work: &mut Vec<u64>,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(crate::MultiplyError::cancelled());
    }
    if a.len().min(b.len()) <= TOOM3_CUTOFF {
        return karatsuba::mul_cancelled(a, b, cancelled);
    }
    let m = a.len().max(b.len()).div_ceil(4);
    let (a0, a1, a2, a3) = split4(a, m);
    let (b0, b1, b2, b3) = split4(b, m);
    toom4_raw_cancelled(a0, a1, a2, a3, b0, b1, b2, b3, m, work, cancelled)
}

fn mul_signed_toom3(x: &Signed, y: &Signed, work: &mut Vec<u64>) -> Signed {
    let size = x.mag.len().max(y.mag.len());
    if size >= TOOM4_CUTOFF {
        signed_new(x.neg != y.neg, mul_toom4_work(&x.mag, &y.mag, work))
    } else {
        signed_new(x.neg != y.neg, mul_toom3_work(&x.mag, &y.mag, work))
    }
}

fn mul_signed_toom3_cancelled(
    x: &Signed,
    y: &Signed,
    work: &mut Vec<u64>,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Signed, crate::MultiplyError> {
    if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(crate::MultiplyError::cancelled());
    }
    let size = x.mag.len().max(y.mag.len());
    if size >= TOOM4_CUTOFF {
        Ok(signed_new(
            x.neg != y.neg,
            mul_toom4_work_cancelled(&x.mag, &y.mag, work, cancelled)?,
        ))
    } else {
        Ok(signed_new(
            x.neg != y.neg,
            mul_toom3_work_cancelled(&x.mag, &y.mag, work, cancelled)?,
        ))
    }
}

fn split4(value: &[u64], m: usize) -> (&[u64], &[u64], &[u64], &[u64]) {
    let c1 = m.min(value.len());
    let c2 = (2 * m).min(value.len());
    let c3 = (3 * m).min(value.len());
    (&value[..c1], &value[c1..c2], &value[c2..c3], &value[c3..])
}

fn eval_toom4(
    p0: &[u64],
    p1: &[u64],
    p2: &[u64],
    p3: &[u64],
) -> (Signed, Signed, Signed, Signed, Signed, Signed, Signed) {
    let at0 = signed_new(false, p0.to_vec());
    let at1 = signed_new(
        false,
        unsigned_add(&unsigned_add(&unsigned_add(p0, p1), p2), p3),
    );
    let atm1 = sub_signed(
        &signed_new(false, unsigned_add(p0, p2)),
        &signed_new(false, unsigned_add(p1, p3)),
    );
    let at2 = signed_new(
        false,
        unsigned_add(
            &unsigned_add(
                &unsigned_add(p0, &mul_small_mag(p1, 2)),
                &mul_small_mag(p2, 4),
            ),
            &mul_small_mag(p3, 8),
        ),
    );
    let atm2 = sub_signed(
        &signed_new(false, unsigned_add(p0, &mul_small_mag(p2, 4))),
        &signed_new(
            false,
            unsigned_add(&mul_small_mag(p1, 2), &mul_small_mag(p3, 8)),
        ),
    );
    let at3 = signed_new(
        false,
        unsigned_add(
            &unsigned_add(
                &unsigned_add(p0, &mul_small_mag(p1, 3)),
                &mul_small_mag(p2, 9),
            ),
            &mul_small_mag(p3, 27),
        ),
    );
    let atinf = signed_new(false, p3.to_vec());
    (at0, at1, atm1, at2, atm2, at3, atinf)
}

fn interpolate4(
    v0: &Signed,
    v1: &Signed,
    vm1: &Signed,
    v2: &Signed,
    vm2: &Signed,
    v3: &Signed,
    vinf: &Signed,
) -> [Vec<u64>; 7] {
    let w0 = v0.mag.clone();
    let w6 = vinf.mag.clone();
    let t1 = sub_signed(&sub_signed(v1, v0), vinf);
    let tm1 = sub_signed(&sub_signed(vm1, v0), vinf);
    let w6x64 = mul_small_mag(&w6, 64);
    let t2 = sub_signed(&sub_signed(v2, v0), &signed_new(false, w6x64.clone()));
    let tm2 = sub_signed(&sub_signed(vm2, v0), &signed_new(false, w6x64));
    let w6x729 = mul_small_mag(&w6, 729);
    let t3 = sub_signed(&sub_signed(v3, v0), &signed_new(false, w6x729));
    let e1 = div_signed_exact(&add_signed(&t1, &tm1), 2);
    let o1 = div_signed_exact(&sub_signed(&t1, &tm1), 2);
    let e2 = div_signed_exact(&add_signed(&t2, &tm2), 2);
    let o2 = div_signed_exact(&sub_signed(&t2, &tm2), 2);
    let w4s = div_signed_exact(
        &sub_signed(&e2, &signed_new(false, mul_small_mag(&e1.mag, 4))),
        12,
    );
    let w2s = sub_signed(&e1, &w4s);
    let u35 = div_signed_exact(
        &sub_signed(&o2, &signed_new(false, mul_small_mag(&o1.mag, 2))),
        6,
    );
    let nine_e1 = mul_small_mag(&e1.mag, 9);
    let sev2_w4 = mul_small_mag(&w4s.mag, 72);
    let s139 = div_signed_exact(
        &sub_signed(
            &sub_signed(&t3, &signed_new(false, nine_e1)),
            &signed_new(false, sev2_w4),
        ),
        3,
    );
    let s3p10 = div_signed_exact(&sub_signed(&s139, &o1), 8);
    let w5s = div_signed_exact(&sub_signed(&s3p10, &u35), 5);
    let w3s = sub_signed(&u35, &signed_new(false, mul_small_mag(&w5s.mag, 5)));
    let w1s = sub_signed(&sub_signed(&o1, &w3s), &w5s);
    debug_assert!(!w1s.neg && !w2s.neg && !w3s.neg && !w4s.neg && !w5s.neg);
    [w0, w1s.mag, w2s.mag, w3s.mag, w4s.mag, w5s.mag, w6]
}

#[allow(clippy::too_many_arguments)]
fn toom4_raw(
    a0: &[u64],
    a1: &[u64],
    a2: &[u64],
    a3: &[u64],
    b0: &[u64],
    b1: &[u64],
    b2: &[u64],
    b3: &[u64],
    m: usize,
    work: &mut Vec<u64>,
) -> Vec<u64> {
    let (aa0, aa1, aam1, aa2, aam2, aa3, aainf) = eval_toom4(a0, a1, a2, a3);
    let (bb0, bb1, bbm1, bb2, bbm2, bb3, bbinf) = eval_toom4(b0, b1, b2, b3);
    let v0 = mul_signed_toom3(&aa0, &bb0, work);
    let v1 = mul_signed_toom3(&aa1, &bb1, work);
    let vm1 = mul_signed_toom3(&aam1, &bbm1, work);
    let v2 = mul_signed_toom3(&aa2, &bb2, work);
    let vm2 = mul_signed_toom3(&aam2, &bbm2, work);
    let v3 = mul_signed_toom3(&aa3, &bb3, work);
    let vinf = mul_signed_toom3(&aainf, &bbinf, work);
    let parts = interpolate4(&v0, &v1, &vm1, &v2, &vm2, &v3, &vinf);
    let mut out = parts[0].clone();
    add_shifted_assign(&mut out, &parts[1], m);
    add_shifted_assign(&mut out, &parts[2], 2 * m);
    add_shifted_assign(&mut out, &parts[3], 3 * m);
    add_shifted_assign(&mut out, &parts[4], 4 * m);
    add_shifted_assign(&mut out, &parts[5], 5 * m);
    add_shifted_assign(&mut out, &parts[6], 6 * m);
    normalize_limbs(&mut out);
    out
}

#[allow(clippy::too_many_arguments)]
fn toom4_raw_cancelled(
    a0: &[u64],
    a1: &[u64],
    a2: &[u64],
    a3: &[u64],
    b0: &[u64],
    b1: &[u64],
    b2: &[u64],
    b3: &[u64],
    m: usize,
    work: &mut Vec<u64>,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<Vec<u64>, crate::MultiplyError> {
    if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(crate::MultiplyError::cancelled());
    }
    let (aa0, aa1, aam1, aa2, aam2, aa3, aainf) = eval_toom4(a0, a1, a2, a3);
    let (bb0, bb1, bbm1, bb2, bbm2, bb3, bbinf) = eval_toom4(b0, b1, b2, b3);
    let v0 = mul_signed_toom3_cancelled(&aa0, &bb0, work, cancelled)?;
    let v1 = mul_signed_toom3_cancelled(&aa1, &bb1, work, cancelled)?;
    let vm1 = mul_signed_toom3_cancelled(&aam1, &bbm1, work, cancelled)?;
    let v2 = mul_signed_toom3_cancelled(&aa2, &bb2, work, cancelled)?;
    let vm2 = mul_signed_toom3_cancelled(&aam2, &bbm2, work, cancelled)?;
    let v3 = mul_signed_toom3_cancelled(&aa3, &bb3, work, cancelled)?;
    let vinf = mul_signed_toom3_cancelled(&aainf, &bbinf, work, cancelled)?;
    let parts = interpolate4(&v0, &v1, &vm1, &v2, &vm2, &v3, &vinf);
    let mut out = parts[0].clone();
    add_shifted_assign(&mut out, &parts[1], m);
    add_shifted_assign(&mut out, &parts[2], 2 * m);
    add_shifted_assign(&mut out, &parts[3], 3 * m);
    add_shifted_assign(&mut out, &parts[4], 4 * m);
    add_shifted_assign(&mut out, &parts[5], 5 * m);
    add_shifted_assign(&mut out, &parts[6], 6 * m);
    normalize_limbs(&mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schoolbook;
    use crate::verify::verify_mod;
    use num_bigint::BigUint;
    use num_traits::One;
    use rand::Rng;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn trimmed(mut value: Vec<u64>) -> Vec<u64> {
        normalize_limbs(&mut value);
        value
    }

    #[test]
    fn eval_interp_round_trip() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c18);
        for _ in 0..50 {
            let m = rng.random_range(1..=10);
            let p0: Vec<u64> = (0..m).map(|_| rng.next_u64()).collect();
            let p1: Vec<u64> = (0..m).map(|_| rng.next_u64()).collect();
            let p2: Vec<u64> = (0..m).map(|_| rng.next_u64()).collect();
            let (e0, e1, em1, e2, _) = eval_toom3(&p0, &p1, &p2);
            let zero = signed_new(false, vec![0]);
            let w = interpolate(&e0, &e1, &em1, &e2, &zero);
            assert_eq!(w[0], trimmed(p0.clone()));
            assert_eq!(w[1], trimmed(p1.clone()));
            assert_eq!(w[2], trimmed(p2.clone()));
            assert_eq!(w[3], vec![0]);
            assert_eq!(w[4], vec![0]);
        }
    }

    #[test]
    fn div_small_exact_paths() {
        let base = BigUint::from(3u32) * ((BigUint::one() << 100) + BigUint::from(7u32));
        let mut limbs = crate::limb::from_bigint(&base);
        div_small_assign(&mut limbs, 3);
        assert_eq!(
            limbs,
            crate::limb::from_bigint(&((BigUint::one() << 100) + BigUint::from(7u32)))
        );
        let mut even = vec![u64::MAX - 1, u64::MAX];
        div_small_assign(&mut even, 2);
        assert_eq!(even, vec![u64::MAX, u64::MAX >> 1]);
        let mut six = vec![36u64, 0, 12];
        div_small_assign(&mut six, 6);
        assert_eq!(six, vec![6, 0, 2]);
    }

    #[test]
    fn cutoff_boundary_matches_karatsuba() {
        for len in [
            TOOM3_CUTOFF - 5,
            TOOM3_CUTOFF - 1,
            TOOM3_CUTOFF,
            TOOM3_CUTOFF + 1,
            TOOM3_CUTOFF + 5,
        ] {
            let a: Vec<u64> = (0..len)
                .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15))
                .collect();
            let b: Vec<u64> = (0..len)
                .map(|i| (i as u64).wrapping_mul(0xbf58476d1ce4e5b9).wrapping_add(7))
                .collect();
            assert_eq!(mul(&a, &b), karatsuba::mul(&a, &b));
        }
    }

    #[test]
    fn odd_and_non_multiple_of_three_shapes() {
        let shapes = [
            (65usize, 65usize),
            (100, 37),
            (63, 63),
            (64, 64),
            (66, 66),
            (65, 200),
            (200, 3),
        ];
        for (la, lb) in shapes {
            let a: Vec<u64> = (0..la)
                .map(|i| (i as u64).wrapping_mul(0x94d049bb133111eb).wrapping_add(1))
                .collect();
            let b: Vec<u64> = (0..lb)
                .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(3))
                .collect();
            assert_eq!(mul(&a, &b), schoolbook::mul(&a, &b));
            assert!(verify_mod(&a, &b, &mul(&a, &b)));
        }
    }

    #[test]
    fn zero_and_one_identities() {
        let big: Vec<u64> = (0..100).map(|i| (i as u64).wrapping_add(5)).collect();
        assert_eq!(mul(&vec![0; 100], &big), vec![0]);
        assert_eq!(mul(&big, &vec![0; 100]), vec![0]);
        assert_eq!(mul(&[1], &big), big);
        assert_eq!(mul(&big, &[1]), big);
        assert_eq!(mul(&[], &big), vec![0]);
    }

    #[test]
    fn commutativity_spot() {
        let a: Vec<u64> = (0..90)
            .map(|i| (i as u64).wrapping_mul(3).wrapping_add(1))
            .collect();
        let b: Vec<u64> = (0..150)
            .map(|i| (i as u64).wrapping_mul(7).wrapping_add(2))
            .collect();
        assert_eq!(mul(&a, &b), mul(&b, &a));
    }

    #[test]
    fn predicted_work_matches_toom3_exponent() {
        assert_eq!(predicted_work_limbs(0), 0.0);
        assert_eq!(predicted_work_limbs(1), 1.0);
        let mid = predicted_work_limbs(128);
        assert_eq!(mid, 128f64.powf(1.465));
        assert!(mid > 128.0 && mid < 16384.0);
    }

    #[test]
    fn randomized_cross_validation_vs_schoolbook() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c19);
        for index in 0..200 {
            let (la, lb) = if index % 2 == 0 {
                (rng.random_range(0..=200), rng.random_range(0..=200))
            } else {
                (
                    rng.random_range(TOOM3_CUTOFF.saturating_sub(5)..=TOOM3_CUTOFF + 5),
                    rng.random_range(TOOM3_CUTOFF.saturating_sub(5)..=TOOM3_CUTOFF + 5),
                )
            };
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64()).collect();
            let result = mul(&a, &b);
            assert_eq!(result, schoolbook::mul(&a, &b));
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn scratch_buffer_matches_fresh_allocation() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c20);
        for _ in 0..50 {
            let la = rng.random_range(0..=200);
            let lb = rng.random_range(0..=200);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64()).collect();
            let mut work = vec![0xDEAD_BEEF_DEAD_BEEFu64; 16];
            assert_eq!(mul_buf(&a, &b, &mut work), mul(&a, &b));
        }
    }

    #[test]
    fn buf_matches_mul_across_cutoffs() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c21);
        for _ in 0..30 {
            let la = rng.random_range(0..=1200);
            let lb = rng.random_range(0..=1200);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64()).collect();
            let expected = mul(&a, &b);
            let mut work = Vec::new();
            assert_eq!(mul_buf(&a, &b, &mut work), expected);
            assert_eq!(mul_toom4(&a, &b), schoolbook::mul(&a, &b));
            assert!(verify_mod(&a, &b, &expected));
        }
    }

    #[test]
    fn eval_interp_round_trip_toom4() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c60);
        for _ in 0..50 {
            let m = rng.random_range(1..=10);
            let p0: Vec<u64> = (0..m).map(|_| rng.next_u64()).collect();
            let p1: Vec<u64> = (0..m).map(|_| rng.next_u64()).collect();
            let p2: Vec<u64> = (0..m).map(|_| rng.next_u64()).collect();
            let p3: Vec<u64> = (0..m).map(|_| rng.next_u64()).collect();
            let (e0, e1, em1, e2, em2, e3, _) = eval_toom4(&p0, &p1, &p2, &p3);
            let zero = signed_new(false, vec![0]);
            let w = interpolate4(&e0, &e1, &em1, &e2, &em2, &e3, &zero);
            assert_eq!(w[0], trimmed(p0.clone()));
            assert_eq!(w[1], trimmed(p1.clone()));
            assert_eq!(w[2], trimmed(p2.clone()));
            assert_eq!(w[3], trimmed(p3.clone()));
            assert_eq!(w[4], vec![0]);
            assert_eq!(w[5], vec![0]);
            assert_eq!(w[6], vec![0]);
        }
    }

    #[test]
    fn div_small_toom4_paths() {
        let mut twelve = vec![36u64, 0, 24];
        div_small_assign(&mut twelve, 12);
        assert_eq!(twelve, vec![3, 0, 2]);
        let mut eight = vec![u64::MAX - 7, 9];
        div_small_assign(&mut eight, 8);
        assert_eq!(
            eight,
            crate::limb::from_bigint(
                &((BigUint::from(u64::MAX - 7) + (BigUint::from(9u32) << 64))
                    / BigUint::from(8u32))
            )
        );
        let mut five = vec![25u64, 0, 5];
        div_small_assign(&mut five, 5);
        assert_eq!(five, vec![5, 0, 1]);
    }

    #[test]
    fn cutoff_boundary_matches_toom3() {
        for len in [
            TOOM4_CUTOFF - 5,
            TOOM4_CUTOFF - 1,
            TOOM4_CUTOFF,
            TOOM4_CUTOFF + 1,
            TOOM4_CUTOFF + 5,
        ] {
            let a: Vec<u64> = (0..len)
                .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15))
                .collect();
            let b: Vec<u64> = (0..len)
                .map(|i| (i as u64).wrapping_mul(0xbf58476d1ce4e5b9).wrapping_add(7))
                .collect();
            assert_eq!(mul(&a, &b), schoolbook::mul(&a, &b));
            assert_eq!(mul_toom4(&a, &b), schoolbook::mul(&a, &b));
        }
    }

    #[test]
    fn toom4_routes_below_cutoff() {
        let len = TOOM4_CUTOFF - 1;
        let a: Vec<u64> = (0..len)
            .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .collect();
        let b: Vec<u64> = (0..len)
            .map(|i| (i as u64).wrapping_mul(0xbf58476d1ce4e5b9).wrapping_add(7))
            .collect();
        assert_eq!(mul(&a, &b), schoolbook::mul(&a, &b));
        assert_eq!(mul_toom4(&a, &b), schoolbook::mul(&a, &b));
    }

    #[test]
    fn odd_and_non_multiple_of_four_shapes() {
        let shapes = [
            (65usize, 65usize),
            (100, 37),
            (193, 193),
            (200, 5),
            (300, 300),
        ];
        for (la, lb) in shapes {
            let a: Vec<u64> = (0..la)
                .map(|i| (i as u64).wrapping_mul(0x94d049bb133111eb).wrapping_add(1))
                .collect();
            let b: Vec<u64> = (0..lb)
                .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(3))
                .collect();
            assert_eq!(mul(&a, &b), schoolbook::mul(&a, &b));
            assert!(verify_mod(&a, &b, &mul(&a, &b)));
        }
    }

    #[test]
    fn toom4_zero_and_one_identities() {
        let big: Vec<u64> = (0..250).map(|i| (i as u64).wrapping_add(11)).collect();
        assert_eq!(mul_toom4(&vec![0; 250], &big), vec![0]);
        assert_eq!(mul_toom4(&big, &vec![0; 250]), vec![0]);
        assert_eq!(mul_toom4(&[1], &big), big);
        assert_eq!(mul_toom4(&big, &[1]), big);
        assert_eq!(mul_toom4(&[], &big), vec![0]);
    }

    #[test]
    fn toom4_commutativity_spot() {
        let a: Vec<u64> = (0..200)
            .map(|i| (i as u64).wrapping_mul(3).wrapping_add(1))
            .collect();
        let b: Vec<u64> = (0..210)
            .map(|i| (i as u64).wrapping_mul(7).wrapping_add(2))
            .collect();
        assert_eq!(mul_toom4(&a, &b), mul_toom4(&b, &a));
    }

    #[test]
    fn predicted_work_matches_toom4_exponent() {
        assert_eq!(predicted_work_limbs_toom4(0), 0.0);
        assert_eq!(predicted_work_limbs_toom4(1), 1.0);
        let mid = predicted_work_limbs_toom4(128);
        assert_eq!(mid, 128f64.powf(1.404));
        assert!(mid > 128.0 && mid < 16384.0);
        assert!(predicted_work_limbs_toom4(512) < predicted_work_limbs(512));
    }

    #[test]
    fn precancelled_toom4_is_cancelled() {
        let a: Vec<u64> = (0..600)
            .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .collect();
        let b: Vec<u64> = (0..600)
            .map(|i| (i as u64).wrapping_mul(0xbf58476d1ce4e5b9).wrapping_add(7))
            .collect();
        let flag = std::sync::atomic::AtomicBool::new(true);
        assert!(matches!(
            mul_toom4_cancelled(&a, &b, &flag),
            Err(crate::MultiplyError::Cancelled(_))
        ));
        assert!(matches!(
            mul_cancelled(&a, &b, &flag),
            Err(crate::MultiplyError::Cancelled(_))
        ));
        let clear = std::sync::atomic::AtomicBool::new(false);
        let result = mul_toom4_cancelled(&a, &b, &clear).expect("runs");
        assert_eq!(result, schoolbook::mul(&a, &b));
        assert!(verify_mod(&a, &b, &result));
    }

    #[test]
    fn randomized_cross_validation_toom4_vs_schoolbook() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c61);
        for index in 0..300 {
            let (la, lb) = if index % 2 == 0 {
                (rng.random_range(0..=400), rng.random_range(0..=400))
            } else {
                (
                    rng.random_range(TOOM4_CUTOFF.saturating_sub(5)..=TOOM4_CUTOFF + 5),
                    rng.random_range(TOOM4_CUTOFF.saturating_sub(5)..=TOOM4_CUTOFF + 5),
                )
            };
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64()).collect();
            let result = mul(&a, &b);
            assert_eq!(result, schoolbook::mul(&a, &b));
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn toom4_large_mod_check() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c62);
        for len in [512usize, 1024, 2048, 4096] {
            let a: Vec<u64> = (0..len).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..len).map(|_| rng.next_u64()).collect();
            let result = mul_toom4(&a, &b);
            assert!(verify_mod(&a, &b, &result));
        }
    }

    #[test]
    fn toom4_scratch_buffer_matches_fresh_allocation() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c63);
        for _ in 0..20 {
            let la = rng.random_range(0..=400);
            let lb = rng.random_range(0..=400);
            let a: Vec<u64> = (0..la).map(|_| rng.next_u64()).collect();
            let b: Vec<u64> = (0..lb).map(|_| rng.next_u64()).collect();
            let mut work = vec![0xDEAD_BEEF_DEAD_BEEFu64; 16];
            assert_eq!(mul_buf(&a, &b, &mut work), mul(&a, &b));
        }
    }
}
