use crate::sleekfft::fft::Radix4Fft;

pub(crate) struct FftGeneric;

impl Radix4Fft for FftGeneric {
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
            for i in 0..mid {
                (sr[i], sr[i + mid]) = (sr[i] + sr[i + mid], sr[i] - sr[i + mid]);
                (si[i], si[i + mid]) = (si[i] + si[i + mid], si[i] - si[i + mid]);
            }
        }
        let mut u: usize = (k & 1) + 1;
        let mut v: usize = 1_usize << (k - 2 - (k & 1));
        while v > 0 {
            let (mut p1, mut p2, mut p3) = (v, 2 * v, 3 * v);

            for p0 in 0..v {
                let (x0r, x0i) = (sr[p0] + sr[p2], si[p0] + si[p2]);
                let (x1r, x1i) = (sr[p1] + sr[p3], si[p1] + si[p3]);
                let (x2r, x2i) = (sr[p0] - sr[p2], si[p0] - si[p2]);
                let (x3r, x3i) = (si[p1] - si[p3], sr[p3] - sr[p1]);

                (sr[p0], si[p0]) = (x0r + x1r, x0i + x1i);
                (sr[p1], si[p1]) = (x0r - x1r, x0i - x1i);
                (sr[p2], si[p2]) = (x2r + x3r, x2i + x3i);
                (sr[p3], si[p3]) = (x2r - x3r, x2i - x3i);

                p1 += 1;
                p2 += 1;
                p3 += 1;
            }
            for h in 1..u {
                let (w1r, w1i) = (wr[h << 1], wi[h << 1]);
                let (w2r, w2i) = (wr[h], wi[h]);
                let (w3r, w3i) = (w1r * w2r - w1i * w2i, w1r * w2i + w1i * w2r);

                let mut p0 = 4 * h * v;
                let (mut p1, mut p2, mut p3) = (p0 + v, p0 + 2 * v, p0 + 3 * v);

                for _ in 0..v {
                    let (x0r, x0i) = (sr[p0], si[p0]);
                    let (x1r, x1i) = (sr[p1] * w1r - si[p1] * w1i, sr[p1] * w1i + si[p1] * w1r);
                    let (x2r, x2i) = (sr[p2] * w2r - si[p2] * w2i, sr[p2] * w2i + si[p2] * w2r);
                    let (x3r, x3i) = (sr[p3] * w3r - si[p3] * w3i, sr[p3] * w3i + si[p3] * w3r);

                    let (y0r, y0i) = (x0r + x2r, x0i + x2i);
                    let (y1r, y1i) = (x1r + x3r, x1i + x3i);
                    let (y2r, y2i) = (x0r - x2r, x0i - x2i);
                    let (y3r, y3i) = (x1i - x3i, x3r - x1r);

                    (sr[p0], si[p0]) = (y0r + y1r, y0i + y1i);
                    (sr[p1], si[p1]) = (y0r - y1r, y0i - y1i);
                    (sr[p2], si[p2]) = (y2r + y3r, y2i + y3i);
                    (sr[p3], si[p3]) = (y2r - y3r, y2i - y3i);

                    p0 += 1;
                    p1 += 1;
                    p2 += 1;
                    p3 += 1;
                }
            }
            u <<= 2;
            v >>= 2;
        }
    }

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
        let mut v: usize = 1;
        while u > 0 {
            let (mut p1, mut p2, mut p3) = (v, 2 * v, 3 * v);

            for p0 in 0..v {
                let (x0r, x0i) = (sr[p0] + sr[p1], si[p0] + si[p1]);
                let (x1r, x1i) = (sr[p0] - sr[p1], si[p0] - si[p1]);
                let (x2r, x2i) = (sr[p2] + sr[p3], si[p2] + si[p3]);
                let (x3r, x3i) = (si[p3] - si[p2], sr[p2] - sr[p3]);

                (sr[p0], si[p0]) = (x0r + x2r, x0i + x2i);
                (sr[p1], si[p1]) = (x1r + x3r, x1i + x3i);
                (sr[p2], si[p2]) = (x0r - x2r, x0i - x2i);
                (sr[p3], si[p3]) = (x1r - x3r, x1i - x3i);

                p1 += 1;
                p2 += 1;
                p3 += 1;
            }
            for h in 1..u {
                let (w1r, w1i) = (wr[h << 1], -wi[h << 1]);
                let (w2r, w2i) = (wr[h], -wi[h]);
                let (w3r, w3i) = (w1r * w2r - w1i * w2i, w1r * w2i + w1i * w2r);

                let mut p0 = 4 * h * v;
                let (mut p1, mut p2, mut p3) = (p0 + v, p0 + 2 * v, p0 + 3 * v);

                for _ in 0..v {
                    let (x0r, x0i) = (sr[p0] + sr[p1], si[p0] + si[p1]);
                    let (x1r, x1i) = (sr[p0] - sr[p1], si[p0] - si[p1]);
                    let (x2r, x2i) = (sr[p2] + sr[p3], si[p2] + si[p3]);
                    let (x3r, x3i) = (si[p3] - si[p2], sr[p2] - sr[p3]);

                    let (y0r, y0i) = (x0r + x2r, x0i + x2i);
                    let (y1r, y1i) = (x1r + x3r, x1i + x3i);
                    let (y2r, y2i) = (x0r - x2r, x0i - x2i);
                    let (y3r, y3i) = (x1r - x3r, x1i - x3i);

                    (sr[p0], si[p0]) = (y0r, y0i);
                    (sr[p1], si[p1]) = (y1r * w1r - y1i * w1i, y1r * w1i + y1i * w1r);
                    (sr[p2], si[p2]) = (y2r * w2r - y2i * w2i, y2r * w2i + y2i * w2r);
                    (sr[p3], si[p3]) = (y3r * w3r - y3i * w3i, y3r * w3i + y3i * w3r);

                    p0 += 1;
                    p1 += 1;
                    p2 += 1;
                    p3 += 1;
                }
            }
            u >>= 2;
            v <<= 2;
        }
        if (k & 1) == 1 {
            let mid = 1_usize << (k - 1);
            let mut p1 = mid;

            for p0 in 0..mid {
                (sr[p0], sr[p1]) = (sr[p0] + sr[p1], sr[p0] - sr[p1]);
                (si[p0], si[p1]) = (si[p0] + si[p1], si[p0] - si[p1]);

                p1 += 1;
            }
        }
    }
}