use std::arch::x86_64::{
    _mm256_add_pd as add_f64x4, _mm256_sub_pd as sub_f64x4, _mm256_broadcast_sd as broadcast_f64,
    _mm256_loadu_pd as loadu_f64x4, _mm256_mul_pd as mul_f64x4, _mm256_storeu_pd as storeu_f64x4,
    _mm256_fmadd_pd as fmadd_f64x4, _mm256_fmsub_pd as fmsub_f64x4,
};
use crate::fft::Radix4Fft;

pub(crate) struct FftAvxFma;

impl Radix4Fft for FftAvxFma {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fwd_transform(&self, wr: &Vec<f64>, wi: &Vec<f64>, sr: &mut Vec<f64>, si: &mut Vec<f64>, k: usize) {
        if k == 0 {
            return;
        }
        if k == 1 {
            (sr[0], sr[1]) = (sr[0] + sr[1], sr[0] - sr[1]);
            (si[0], si[1]) = (si[0] + si[1], si[0] - si[1]);
            return;
        }
        if (k & 1) == 1 {
            let mid = 1_usize << (k - 1);
            let (mut p0r, mut p0i) = (sr.as_mut_ptr(), si.as_mut_ptr());
            let (mut p1r, mut p1i) = (p0r.add(mid), p0i.add(mid));

            for _ in (0..mid).step_by(4) {
                let (x0r, x0i) = (loadu_f64x4(p0r), loadu_f64x4(p0i));
                let (x1r, x1i) = (loadu_f64x4(p1r), loadu_f64x4(p1i));

                storeu_f64x4(p0r, add_f64x4(x0r, x1r));
                storeu_f64x4(p0i, add_f64x4(x0i, x1i));
                storeu_f64x4(p1r, sub_f64x4(x0r, x1r));
                storeu_f64x4(p1i, sub_f64x4(x0i, x1i));

                (p0r, p0i) = (p0r.add(4), p0i.add(4));
                (p1r, p1i) = (p1r.add(4), p1i.add(4));
            }
        }
        let mut u: usize = (k & 1) + 1;
        let mut v: usize = 1_usize << (k - 2 - (k & 1));
        while v >= 4 {
            let (mut p0r, mut p0i) = (sr.as_mut_ptr(), si.as_mut_ptr());
            let (mut p1r, mut p1i) = (p0r.add(v), p0i.add(v));
            let (mut p2r, mut p2i) = (p1r.add(v), p1i.add(v));
            let (mut p3r, mut p3i) = (p2r.add(v), p2i.add(v));

            for _ in (0..v).step_by(4) {
                let (x0r, x0i) = (loadu_f64x4(p0r), loadu_f64x4(p0i));
                let (x1r, x1i) = (loadu_f64x4(p1r), loadu_f64x4(p1i));
                let (x2r, x2i) = (loadu_f64x4(p2r), loadu_f64x4(p2i));
                let (x3r, x3i) = (loadu_f64x4(p3r), loadu_f64x4(p3i));

                let (y0r, y0i) = (add_f64x4(x0r, x2r), add_f64x4(x0i, x2i));
                let (y1r, y1i) = (add_f64x4(x1r, x3r), add_f64x4(x1i, x3i));
                let (y2r, y2i) = (sub_f64x4(x0r, x2r), sub_f64x4(x0i, x2i));
                let (y3r, y3i) = (sub_f64x4(x1i, x3i), sub_f64x4(x3r, x1r));

                storeu_f64x4(p0r, add_f64x4(y0r, y1r));
                storeu_f64x4(p0i, add_f64x4(y0i, y1i));
                storeu_f64x4(p1r, sub_f64x4(y0r, y1r));
                storeu_f64x4(p1i, sub_f64x4(y0i, y1i));
                storeu_f64x4(p2r, add_f64x4(y2r, y3r));
                storeu_f64x4(p2i, add_f64x4(y2i, y3i));
                storeu_f64x4(p3r, sub_f64x4(y2r, y3r));
                storeu_f64x4(p3i, sub_f64x4(y2i, y3i));

                (p0r, p0i) = (p0r.add(4), p0i.add(4));
                (p1r, p1i) = (p1r.add(4), p1i.add(4));
                (p2r, p2i) = (p2r.add(4), p2i.add(4));
                (p3r, p3i) = (p3r.add(4), p3i.add(4));
            }
            for h in 1..u {
                let (w1r, w1i) = (
                    broadcast_f64(&wr[h << 1]),
                    broadcast_f64(&wi[h << 1]),
                );
                let (w2r, w2i) = (
                    broadcast_f64(&wr[h]),
                    broadcast_f64(&wi[h])
                );
                let (w3r, w3i) = (
                    fmsub_f64x4(w1r, w2r, mul_f64x4(w1i, w2i)),
                    fmadd_f64x4(w1r, w2i, mul_f64x4(w1i, w2r)),
                );

                let st = 4 * h * v;
                (p0r, p0i) = (sr.as_mut_ptr().add(st), si.as_mut_ptr().add(st));
                (p1r, p1i) = (p0r.add(v), p0i.add(v));
                (p2r, p2i) = (p1r.add(v), p1i.add(v));
                (p3r, p3i) = (p2r.add(v), p2i.add(v));

                for _ in (0..v).step_by(4) {
                    let (t0r, t0i) = (loadu_f64x4(p0r), loadu_f64x4(p0i));
                    let (t1r, t1i) = (loadu_f64x4(p1r), loadu_f64x4(p1i));
                    let (t2r, t2i) = (loadu_f64x4(p2r), loadu_f64x4(p2i));
                    let (t3r, t3i) = (loadu_f64x4(p3r), loadu_f64x4(p3i));

                    let (x0r, x0i) = (t0r, t0i);
                    let (x1r, x1i) = (
                        fmsub_f64x4(t1r, w1r, mul_f64x4(t1i, w1i)),
                        fmadd_f64x4(t1r, w1i, mul_f64x4(t1i, w1r)),
                    );
                    let (x2r, x2i) = (
                        fmsub_f64x4(t2r, w2r, mul_f64x4(t2i, w2i)),
                        fmadd_f64x4(t2r, w2i, mul_f64x4(t2i, w2r)),
                    );
                    let (x3r, x3i) = (
                        fmsub_f64x4(t3r, w3r, mul_f64x4(t3i, w3i)),
                        fmadd_f64x4(t3r, w3i, mul_f64x4(t3i, w3r)),
                    );

                    let (y0r, y0i) = (add_f64x4(x0r, x2r), add_f64x4(x0i, x2i));
                    let (y1r, y1i) = (add_f64x4(x1r, x3r), add_f64x4(x1i, x3i));
                    let (y2r, y2i) = (sub_f64x4(x0r, x2r), sub_f64x4(x0i, x2i));
                    let (y3r, y3i) = (sub_f64x4(x1i, x3i), sub_f64x4(x3r, x1r));

                    storeu_f64x4(p0r, add_f64x4(y0r, y1r));
                    storeu_f64x4(p0i, add_f64x4(y0i, y1i));
                    storeu_f64x4(p1r, sub_f64x4(y0r, y1r));
                    storeu_f64x4(p1i, sub_f64x4(y0i, y1i));
                    storeu_f64x4(p2r, add_f64x4(y2r, y3r));
                    storeu_f64x4(p2i, add_f64x4(y2i, y3i));
                    storeu_f64x4(p3r, sub_f64x4(y2r, y3r));
                    storeu_f64x4(p3i, sub_f64x4(y2i, y3i));

                    (p0r, p0i) = (p0r.add(4), p0i.add(4));
                    (p1r, p1i) = (p1r.add(4), p1i.add(4));
                    (p2r, p2i) = (p2r.add(4), p2i.add(4));
                    (p3r, p3i) = (p3r.add(4), p3i.add(4));
                }
            }
            u <<= 2;
            v >>= 2;
        }
        for h in 0..u {
            let (w1r, w1i) = (wr[h << 1], wi[h << 1]);
            let (w2r, w2i) = (wr[h], wi[h]);
            let (w3r, w3i) = (w1r.mul_add(w2r, -w1i * w2i), w1r.mul_add(w2i, w1i * w2r));

            let p0 = h << 2;
            let (p1, p2, p3) = (p0 + 1, p0 + 2, p0 + 3);
            let (x0r, x0i) = (sr[p0], si[p0]);
            let (x1r, x1i) = (sr[p1].mul_add(w1r, -si[p1] * w1i), sr[p1].mul_add(w1i, si[p1] * w1r));
            let (x2r, x2i) = (sr[p2].mul_add(w2r, -si[p2] * w2i), sr[p2].mul_add(w2i, si[p2] * w2r));
            let (x3r, x3i) = (sr[p3].mul_add(w3r, -si[p3] * w3i), sr[p3].mul_add(w3i, si[p3] * w3r));

            let (y0r, y0i) = (x0r + x2r, x0i + x2i);
            let (y1r, y1i) = (x1r + x3r, x1i + x3i);
            let (y2r, y2i) = (x0r - x2r, x0i - x2i);
            let (y3r, y3i) = (x1i - x3i, x3r - x1r);

            (sr[p0], si[p0]) = (y0r + y1r, y0i + y1i);
            (sr[p1], si[p1]) = (y0r - y1r, y0i - y1i);
            (sr[p2], si[p2]) = (y2r + y3r, y2i + y3i);
            (sr[p3], si[p3]) = (y2r - y3r, y2i - y3i);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn inv_transform(&self, wr: &Vec<f64>, wi: &Vec<f64>, sr: &mut Vec<f64>, si: &mut Vec<f64>, k: usize) {
        if k == 0 {
            return;
        }
        if k == 1 {
            (sr[0], sr[1]) = (sr[0] + sr[1], sr[0] - sr[1]);
            (si[0], si[1]) = (si[0] + si[1], si[0] - si[1]);
            return;
        }
        let mut u: usize = 1_usize << (k - 2);
        for h in 0..u {
            let (w1r, w1i) = (wr[h << 1], -wi[h << 1]);
            let (w2r, w2i) = (wr[h], -wi[h]);
            let (w3r, w3i) = (w1r.mul_add(w2r, -w1i * w2i), w1r.mul_add(w2i, w1i * w2r));

            let p0 = h << 2;
            let (p1, p2, p3) = (p0 + 1, p0 + 2, p0 + 3);
            let (x0r, x0i) = (sr[p0] + sr[p1], si[p0] + si[p1]);
            let (x1r, x1i) = (sr[p0] - sr[p1], si[p0] - si[p1]);
            let (x2r, x2i) = (sr[p2] + sr[p3], si[p2] + si[p3]);
            let (x3r, x3i) = (si[p3] - si[p2], sr[p2] - sr[p3]);

            let (y0r, y0i) = (x0r + x2r, x0i + x2i);
            let (y1r, y1i) = (x1r + x3r, x1i + x3i);
            let (y2r, y2i) = (x0r - x2r, x0i - x2i);
            let (y3r, y3i) = (x1r - x3r, x1i - x3i);

            (sr[p0], si[p0]) = (y0r, y0i);
            (sr[p1], si[p1]) = (y1r.mul_add(w1r, -y1i * w1i), y1r.mul_add(w1i, y1i * w1r));
            (sr[p2], si[p2]) = (y2r.mul_add(w2r, -y2i * w2i), y2r.mul_add(w2i, y2i * w2r));
            (sr[p3], si[p3]) = (y3r.mul_add(w3r, -y3i * w3i), y3r.mul_add(w3i, y3i * w3r));
        }
        u >>= 2;
        let mut v: usize = 4;
        while u > 0 {
            let (mut p0r, mut p0i) = (sr.as_mut_ptr(), si.as_mut_ptr());
            let (mut p1r, mut p1i) = (p0r.add(v), p0i.add(v));
            let (mut p2r, mut p2i) = (p1r.add(v), p1i.add(v));
            let (mut p3r, mut p3i) = (p2r.add(v), p2i.add(v));

            for _ in (0..v).step_by(4) {
                let (x0r, x0i) = (loadu_f64x4(p0r), loadu_f64x4(p0i));
                let (x1r, x1i) = (loadu_f64x4(p1r), loadu_f64x4(p1i));
                let (x2r, x2i) = (loadu_f64x4(p2r), loadu_f64x4(p2i));
                let (x3r, x3i) = (loadu_f64x4(p3r), loadu_f64x4(p3i));

                let (y0r, y0i) = (add_f64x4(x0r, x1r), add_f64x4(x0i, x1i));
                let (y1r, y1i) = (sub_f64x4(x0r, x1r), sub_f64x4(x0i, x1i));
                let (y2r, y2i) = (add_f64x4(x2r, x3r), add_f64x4(x2i, x3i));
                let (y3r, y3i) = (sub_f64x4(x3i, x2i), sub_f64x4(x2r, x3r));

                storeu_f64x4(p0r, add_f64x4(y0r, y2r));
                storeu_f64x4(p0i, add_f64x4(y0i, y2i));
                storeu_f64x4(p1r, add_f64x4(y1r, y3r));
                storeu_f64x4(p1i, add_f64x4(y1i, y3i));
                storeu_f64x4(p2r, sub_f64x4(y0r, y2r));
                storeu_f64x4(p2i, sub_f64x4(y0i, y2i));
                storeu_f64x4(p3r, sub_f64x4(y1r, y3r));
                storeu_f64x4(p3i, sub_f64x4(y1i, y3i));

                (p0r, p0i) = (p0r.add(4), p0i.add(4));
                (p1r, p1i) = (p1r.add(4), p1i.add(4));
                (p2r, p2i) = (p2r.add(4), p2i.add(4));
                (p3r, p3i) = (p3r.add(4), p3i.add(4));
            }
            for h in 1..u {
                let (w1r, w1i) = (
                    broadcast_f64(&wr[h << 1]),
                    broadcast_f64(&-wi[h << 1]),
                );
                let (w2r, w2i) = (
                    broadcast_f64(&wr[h]),
                    broadcast_f64(&-wi[h]),
                );
                let (w3r, w3i) = (
                    fmsub_f64x4(w1r, w2r, mul_f64x4(w1i, w2i)),
                    fmadd_f64x4(w1r, w2i, mul_f64x4(w1i, w2r)),
                );

                let st = 4 * h * v;
                (p0r, p0i) = (sr.as_mut_ptr().add(st), si.as_mut_ptr().add(st));
                (p1r, p1i) = (p0r.add(v), p0i.add(v));
                (p2r, p2i) = (p1r.add(v), p1i.add(v));
                (p3r, p3i) = (p2r.add(v), p2i.add(v));

                for _ in (0..v).step_by(4) {
                    let (t0r, t0i) = (loadu_f64x4(p0r), loadu_f64x4(p0i));
                    let (t1r, t1i) = (loadu_f64x4(p1r), loadu_f64x4(p1i));
                    let (t2r, t2i) = (loadu_f64x4(p2r), loadu_f64x4(p2i));
                    let (t3r, t3i) = (loadu_f64x4(p3r), loadu_f64x4(p3i));

                    let (x0r, x0i) = (add_f64x4(t0r, t1r), add_f64x4(t0i, t1i));
                    let (x1r, x1i) = (sub_f64x4(t0r, t1r), sub_f64x4(t0i, t1i));
                    let (x2r, x2i) = (add_f64x4(t2r, t3r), add_f64x4(t2i, t3i));
                    let (x3r, x3i) = (sub_f64x4(t3i, t2i), sub_f64x4(t2r, t3r));

                    let (y0r, y0i) = (add_f64x4(x0r, x2r), add_f64x4(x0i, x2i));
                    let (y1r, y1i) = (add_f64x4(x1r, x3r), add_f64x4(x1i, x3i));
                    let (y2r, y2i) = (sub_f64x4(x0r, x2r), sub_f64x4(x0i, x2i));
                    let (y3r, y3i) = (sub_f64x4(x1r, x3r), sub_f64x4(x1i, x3i));

                    storeu_f64x4(p0r, y0r);
                    storeu_f64x4(p0i, y0i);
                    storeu_f64x4(p1r, fmsub_f64x4(y1r, w1r, mul_f64x4(y1i, w1i)));
                    storeu_f64x4(p1i, fmadd_f64x4(y1r, w1i, mul_f64x4(y1i, w1r)));
                    storeu_f64x4(p2r, fmsub_f64x4(y2r, w2r, mul_f64x4(y2i, w2i)));
                    storeu_f64x4(p2i, fmadd_f64x4(y2r, w2i, mul_f64x4(y2i, w2r)));
                    storeu_f64x4(p3r, fmsub_f64x4(y3r, w3r, mul_f64x4(y3i, w3i)));
                    storeu_f64x4(p3i, fmadd_f64x4(y3r, w3i, mul_f64x4(y3i, w3r)));

                    (p0r, p0i) = (p0r.add(4), p0i.add(4));
                    (p1r, p1i) = (p1r.add(4), p1i.add(4));
                    (p2r, p2i) = (p2r.add(4), p2i.add(4));
                    (p3r, p3i) = (p3r.add(4), p3i.add(4));
                }
            }
            u >>= 2;
            v <<= 2;
        }
        if (k & 1) == 1 {
            let mid = 1_usize << (k - 1);
            let (mut p0r, mut p0i) = (sr.as_mut_ptr(), si.as_mut_ptr());
            let (mut p1r, mut p1i) = (p0r.add(mid), p0i.add(mid));

            for _ in (0..mid).step_by(4) {
                let (x0r, x0i) = (loadu_f64x4(p0r), loadu_f64x4(p0i));
                let (x1r, x1i) = (loadu_f64x4(p1r), loadu_f64x4(p1i));

                storeu_f64x4(p0r, add_f64x4(x0r, x1r));
                storeu_f64x4(p0i, add_f64x4(x0i, x1i));
                storeu_f64x4(p1r, sub_f64x4(x0r, x1r));
                storeu_f64x4(p1i, sub_f64x4(x0i, x1i));

                (p0r, p0i) = (p0r.add(4), p0i.add(4));
                (p1r, p1i) = (p1r.add(4), p1i.add(4));
            }
        }
    }
}
