pub(crate) trait Radix4Fft {
    unsafe fn fwd_transform(&self, wr: &Vec<f64>, wi: &Vec<f64>, sr: &mut Vec<f64>, si: &mut Vec<f64>, k: usize);
    unsafe fn inv_transform(&self, wr: &Vec<f64>, wi: &Vec<f64>, sr: &mut Vec<f64>, si: &mut Vec<f64>, k: usize);
}

#[macro_export]
macro_rules! impl_radix4fft_f64 {
    (
        $($target_arch: literal),+ $(,)?;
        $($target_feature: literal),+ $(,)?;
        $name: ident;
        $loadu: ident,
        $storeu: ident,
        $add: ident,
        $sub: ident,
        $mul_add: expr,
        $mul_sub: expr,
        $broadcast: expr,
        $offset: expr,
    ) => {
        impl Radix4Fft for $name {
            #[cfg(any($(target_arch = $target_arch), *))]
            #[target_feature($(enable = $target_feature), *)]
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

                    for _ in (0..mid).step_by($offset) {
                        let (x0r, x0i) = ($loadu(p0r), $loadu(p0i));
                        let (x1r, x1i) = ($loadu(p1r), $loadu(p1i));

                        $storeu(p0r, $add(x0r, x1r));
                        $storeu(p0i, $add(x0i, x1i));
                        $storeu(p1r, $sub(x0r, x1r));
                        $storeu(p1i, $sub(x0i, x1i));

                        (p0r, p0i) = (p0r.add($offset), p0i.add($offset));
                        (p1r, p1i) = (p1r.add($offset), p1i.add($offset));
                    }
                }
                let mut u: usize = (k & 1) + 1;
                let mut v: usize = 1_usize << (k - 2 - (k & 1));
                while v >= $offset {
                    let (mut p0r, mut p0i) = (sr.as_mut_ptr(), si.as_mut_ptr());
                    let (mut p1r, mut p1i) = (p0r.add(v), p0i.add(v));
                    let (mut p2r, mut p2i) = (p1r.add(v), p1i.add(v));
                    let (mut p3r, mut p3i) = (p2r.add(v), p2i.add(v));

                    for _ in (0..v).step_by($offset) {
                        let (x0r, x0i) = ($loadu(p0r), $loadu(p0i));
                        let (x1r, x1i) = ($loadu(p1r), $loadu(p1i));
                        let (x2r, x2i) = ($loadu(p2r), $loadu(p2i));
                        let (x3r, x3i) = ($loadu(p3r), $loadu(p3i));

                        let (y0r, y0i) = ($add(x0r, x2r), $add(x0i, x2i));
                        let (y1r, y1i) = ($add(x1r, x3r), $add(x1i, x3i));
                        let (y2r, y2i) = ($sub(x0r, x2r), $sub(x0i, x2i));
                        let (y3r, y3i) = ($sub(x1i, x3i), $sub(x3r, x1r));

                        $storeu(p0r, $add(y0r, y1r));
                        $storeu(p0i, $add(y0i, y1i));
                        $storeu(p1r, $sub(y0r, y1r));
                        $storeu(p1i, $sub(y0i, y1i));
                        $storeu(p2r, $add(y2r, y3r));
                        $storeu(p2i, $add(y2i, y3i));
                        $storeu(p3r, $sub(y2r, y3r));
                        $storeu(p3i, $sub(y2i, y3i));

                        (p0r, p0i) = (p0r.add($offset), p0i.add($offset));
                        (p1r, p1i) = (p1r.add($offset), p1i.add($offset));
                        (p2r, p2i) = (p2r.add($offset), p2i.add($offset));
                        (p3r, p3i) = (p3r.add($offset), p3i.add($offset));
                    }
                    for h in 1..u {
                        let (w1r, w1i) = ($broadcast(&wr[h << 1]), $broadcast(&wi[h << 1]));
                        let (w2r, w2i) = ($broadcast(&wr[h]), $broadcast(&wi[h]));
                        let (w3r, w3i) = ($mul_sub(w1r, w2r, w1i, w2i), $mul_add(w1r, w2i, w1i, w2r));

                        let st = 4 * h * v;
                        (p0r, p0i) = (sr.as_mut_ptr().add(st), si.as_mut_ptr().add(st));
                        (p1r, p1i) = (p0r.add(v), p0i.add(v));
                        (p2r, p2i) = (p1r.add(v), p1i.add(v));
                        (p3r, p3i) = (p2r.add(v), p2i.add(v));

                        for _ in (0..v).step_by($offset) {
                            let (t0r, t0i) = ($loadu(p0r), $loadu(p0i));
                            let (t1r, t1i) = ($loadu(p1r), $loadu(p1i));
                            let (t2r, t2i) = ($loadu(p2r), $loadu(p2i));
                            let (t3r, t3i) = ($loadu(p3r), $loadu(p3i));

                            let (x0r, x0i) = (t0r, t0i);
                            let (x1r, x1i) = ($mul_sub(t1r, w1r, t1i, w1i), $mul_add(t1r, w1i, t1i, w1r));
                            let (x2r, x2i) = ($mul_sub(t2r, w2r, t2i, w2i), $mul_add(t2r, w2i, t2i, w2r));
                            let (x3r, x3i) = ($mul_sub(t3r, w3r, t3i, w3i), $mul_add(t3r, w3i, t3i, w3r));

                            let (y0r, y0i) = ($add(x0r, x2r), $add(x0i, x2i));
                            let (y1r, y1i) = ($add(x1r, x3r), $add(x1i, x3i));
                            let (y2r, y2i) = ($sub(x0r, x2r), $sub(x0i, x2i));
                            let (y3r, y3i) = ($sub(x1i, x3i), $sub(x3r, x1r));

                            $storeu(p0r, $add(y0r, y1r));
                            $storeu(p0i, $add(y0i, y1i));
                            $storeu(p1r, $sub(y0r, y1r));
                            $storeu(p1i, $sub(y0i, y1i));
                            $storeu(p2r, $add(y2r, y3r));
                            $storeu(p2i, $add(y2i, y3i));
                            $storeu(p3r, $sub(y2r, y3r));
                            $storeu(p3i, $sub(y2i, y3i));

                            (p0r, p0i) = (p0r.add($offset), p0i.add($offset));
                            (p1r, p1i) = (p1r.add($offset), p1i.add($offset));
                            (p2r, p2i) = (p2r.add($offset), p2i.add($offset));
                            (p3r, p3i) = (p3r.add($offset), p3i.add($offset));
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

            #[cfg(any($(target_arch = $target_arch), *))]
            #[target_feature($(enable = $target_feature), *)]
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

                    for _ in (0..v).step_by($offset) {
                        let (x0r, x0i) = ($loadu(p0r), $loadu(p0i));
                        let (x1r, x1i) = ($loadu(p1r), $loadu(p1i));
                        let (x2r, x2i) = ($loadu(p2r), $loadu(p2i));
                        let (x3r, x3i) = ($loadu(p3r), $loadu(p3i));

                        let (y0r, y0i) = ($add(x0r, x1r), $add(x0i, x1i));
                        let (y1r, y1i) = ($sub(x0r, x1r), $sub(x0i, x1i));
                        let (y2r, y2i) = ($add(x2r, x3r), $add(x2i, x3i));
                        let (y3r, y3i) = ($sub(x3i, x2i), $sub(x2r, x3r));

                        $storeu(p0r, $add(y0r, y2r));
                        $storeu(p0i, $add(y0i, y2i));
                        $storeu(p1r, $add(y1r, y3r));
                        $storeu(p1i, $add(y1i, y3i));
                        $storeu(p2r, $sub(y0r, y2r));
                        $storeu(p2i, $sub(y0i, y2i));
                        $storeu(p3r, $sub(y1r, y3r));
                        $storeu(p3i, $sub(y1i, y3i));

                        (p0r, p0i) = (p0r.add($offset), p0i.add($offset));
                        (p1r, p1i) = (p1r.add($offset), p1i.add($offset));
                        (p2r, p2i) = (p2r.add($offset), p2i.add($offset));
                        (p3r, p3i) = (p3r.add($offset), p3i.add($offset));
                    }
                    for h in 1..u {
                        let (w1r, w1i) = ($broadcast(&wr[h << 1]), $broadcast(&-wi[h << 1]));
                        let (w2r, w2i) = ($broadcast(&wr[h]), $broadcast(&-wi[h]));
                        let (w3r, w3i) = ($mul_sub(w1r, w2r, w1i, w2i), $mul_add(w1r, w2i, w1i, w2r));

                        let st = 4 * h * v;
                        (p0r, p0i) = (sr.as_mut_ptr().add(st), si.as_mut_ptr().add(st));
                        (p1r, p1i) = (p0r.add(v), p0i.add(v));
                        (p2r, p2i) = (p1r.add(v), p1i.add(v));
                        (p3r, p3i) = (p2r.add(v), p2i.add(v));

                        for _ in (0..v).step_by($offset) {
                            let (t0r, t0i) = ($loadu(p0r), $loadu(p0i));
                            let (t1r, t1i) = ($loadu(p1r), $loadu(p1i));
                            let (t2r, t2i) = ($loadu(p2r), $loadu(p2i));
                            let (t3r, t3i) = ($loadu(p3r), $loadu(p3i));

                            let (x0r, x0i) = ($add(t0r, t1r), $add(t0i, t1i));
                            let (x1r, x1i) = ($sub(t0r, t1r), $sub(t0i, t1i));
                            let (x2r, x2i) = ($add(t2r, t3r), $add(t2i, t3i));
                            let (x3r, x3i) = ($sub(t3i, t2i), $sub(t2r, t3r));

                            let (y0r, y0i) = ($add(x0r, x2r), $add(x0i, x2i));
                            let (y1r, y1i) = ($add(x1r, x3r), $add(x1i, x3i));
                            let (y2r, y2i) = ($sub(x0r, x2r), $sub(x0i, x2i));
                            let (y3r, y3i) = ($sub(x1r, x3r), $sub(x1i, x3i));

                            $storeu(p0r, y0r);
                            $storeu(p0i, y0i);
                            $storeu(p1r, $mul_sub(y1r, w1r, y1i, w1i));
                            $storeu(p1i, $mul_add(y1r, w1i, y1i, w1r));
                            $storeu(p2r, $mul_sub(y2r, w2r, y2i, w2i));
                            $storeu(p2i, $mul_add(y2r, w2i, y2i, w2r));
                            $storeu(p3r, $mul_sub(y3r, w3r, y3i, w3i));
                            $storeu(p3i, $mul_add(y3r, w3i, y3i, w3r));

                            (p0r, p0i) = (p0r.add($offset), p0i.add($offset));
                            (p1r, p1i) = (p1r.add($offset), p1i.add($offset));
                            (p2r, p2i) = (p2r.add($offset), p2i.add($offset));
                            (p3r, p3i) = (p3r.add($offset), p3i.add($offset));
                        }
                    }
                    u >>= 2;
                    v <<= 2;
                }
                if (k & 1) == 1 {
                    let mid = 1_usize << (k - 1);
                    let (mut p0r, mut p0i) = (sr.as_mut_ptr(), si.as_mut_ptr());
                    let (mut p1r, mut p1i) = (p0r.add(mid), p0i.add(mid));

                    for _ in (0..mid).step_by($offset) {
                        let (x0r, x0i) = ($loadu(p0r), $loadu(p0i));
                        let (x1r, x1i) = ($loadu(p1r), $loadu(p1i));

                        $storeu(p0r, $add(x0r, x1r));
                        $storeu(p0i, $add(x0i, x1i));
                        $storeu(p1r, $sub(x0r, x1r));
                        $storeu(p1i, $sub(x0i, x1i));

                        (p0r, p0i) = (p0r.add($offset), p0i.add($offset));
                        (p1r, p1i) = (p1r.add($offset), p1i.add($offset));
                    }
                }
            }
        }
    };
}