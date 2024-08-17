use std::f64::consts::PI;
use crate::generic::fft_gen::FftGeneric;
use crate::x86::{
    fft_sse::FftSse,
    fft_avx::FftAvx,
    fft_avx_fma::FftAvxFma,
};

pub(crate) trait Radix4Fft {
    unsafe fn fwd_transform(&self, wr: &Vec<f64>, wi: &Vec<f64>, sr: &mut Vec<f64>, si: &mut Vec<f64>, k: usize);
    unsafe fn inv_transform(&self, wr: &Vec<f64>, wi: &Vec<f64>, sr: &mut Vec<f64>, si: &mut Vec<f64>, k: usize);
}

pub struct Fft {
    wr: Vec<f64>,
    wi: Vec<f64>,
    algo: Box<dyn Radix4Fft>,
}

impl Fft {

    /// Creates a new `Fft` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use sleekfft_rs::Fft;
    /// let fft = Fft::new();
    /// ```
    pub fn new() -> Fft {
        let supported_algo: Box<dyn Radix4Fft> =
            if is_x86_feature_detected!("avx") {
                if is_x86_feature_detected!("fma") {
                    Box::new(FftAvxFma)
                } else {
                    Box::new(FftAvx)
                }
            } else if is_x86_feature_detected!("sse2") {
                Box::new(FftSse)
            } else {
                Box::new(FftGeneric)
            };
        Fft {
            wr: vec![],
            wi: vec![],
            algo: supported_algo,
        }
    }

    fn reserve(&mut self, k: usize) {
        let m = 1_usize << (k - 1);
        if self.wr.len() >= m {
            return;
        }
        self.wr = vec![0.0; m];
        self.wi = vec![0.0; m];
        self.wr[0] = 1.0;
        let theta: f64 = -PI / m as f64;
        let (mut i, mut j) = (1_usize, m >> 1);
        while j > 0 {
            let arg: f64 = theta * j as f64;
            self.wr[i] = arg.cos();
            self.wi[i] = arg.sin();
            i <<= 1;
            j >>= 1;
        }
        for i in 1..m {
            let x: usize = i & (i - 1);
            let y: usize = 1_usize << i.trailing_zeros();
            self.wr[i] = self.wr[x].mul_add(self.wr[y], -self.wi[x] * self.wi[y]);
            self.wi[i] = self.wr[x].mul_add(self.wi[y], self.wi[x] * self.wr[y]);
        }
    }

    /// This function performs fast fourier transform of the signal and replaces the original
    /// input signal. Input signal always considered as in normal order.
    ///
    /// # Parameters
    ///
    /// * `sr` Real part of the seperated signal
    /// * `si` Imaginary part of the seperated signal
    /// * `out_ord_normal` if this is true the transformed signal is placed in normal
    /// order otherwise signal is placed in bit-reversed order.
    ///
    /// # Important
    ///
    /// * Length of the two input vectors must be same.
    /// * The length must be power of 2.
    /// * It will be slightly faster if the two signals are properly aligned according to the
    /// maximum SIMD instruction set supported by the cpu (avx --> 32 byte, sse2 --> 16 byte etc.).
    ///
    /// # Example
    ///
    /// ```
    /// use sleekfft_rs::Fft;
    /// let n: usize = 1024;
    /// let mut a = vec![0.0; n];
    /// let mut b = vec![0.0; n];
    /// for i in 0..n {
    ///     a[i] = i as f64;
    ///     b[i] = (n - 1 - i) as f64;
    /// }
    /// let mut c = vec![0.0; n];
    /// let mut d = vec![0.0; n];
    /// let arg = -2.0 * std::f64::consts::PI / n as f64;
    /// for i in 0..n {
    ///     for j in 0..n {
    ///         let theta = arg * i as f64 * j as f64;
    ///         let w = (theta.cos(), theta.sin());
    ///         c[i] += a[j] * w.0 - b[j] * w.1;
    ///         d[i] += a[j] * w.1 + b[j] * w.0;
    ///     }
    /// }
    /// Fft::new().fwd_transform(&mut a, &mut b, true);
    /// for i in 0..n {
    ///     assert!((a[i] - c[i]).abs() < 1e-7);
    ///     assert!((b[i] - d[i]).abs() < 1e-7);
    /// }
    /// ```
    ///
    pub fn fwd_transform(&mut self, sr: &mut Vec<f64>, si: &mut Vec<f64>, out_ord_normal: bool) {
        debug_assert_eq!(sr.len(), si.len(), "Length of two vectors should be equal");
        debug_assert!(sr.len().count_ones() == 1, "Length is not power of 2");
        let k: usize = sr.len().trailing_zeros() as usize;
        self.reserve(k);
        unsafe { self.algo.fwd_transform(&self.wr, &self.wi, sr, si, k); }
        if out_ord_normal {
            for i in 0.. sr.len() {
                let j = i.reverse_bits() >> (size_of::<usize>() * 8 - k);
                if i < j {
                    sr.swap(i, j);
                    si.swap(i, j);
                }
            }
        }
    }

    /// This function performs inverse fast fourier transform of the signal and replaces the
    /// original input signal. Transformed signal is always placed in normal order.
    ///
    /// # Parameters
    ///
    /// * `sr` Real part of the seperated signal
    /// * `si` Imaginary part of the seperated signal
    /// * `in_ord_normal` if this is true the input signal is assumed to be in normal
    /// order otherwise signal is considered as in bit-reversed order.
    ///
    /// # Important
    ///
    /// * Length of the two input vectors must be same.
    /// * The length must be power of 2.
    /// * It will be slightly faster if the two signals are properly aligned according to the
    /// maximum SIMD instruction set supported by the cpu (avx --> 32 byte, sse2 --> 16 byte etc.).
    ///
    /// # Example
    ///
    /// ```
    /// use sleekfft_rs::Fft;
    /// let n: usize = 1024;
    /// let mut a = vec![0.0; n];
    /// let mut b = vec![0.0; n];
    /// for i in 0..n {
    ///     a[i] = i as f64;
    ///     b[i] = (n - 1 - i) as f64;
    /// }
    /// let mut fft = Fft::new();
    /// fft.fwd_transform(&mut a, &mut b, false);
    /// fft.inv_transform(&mut a, &mut b, false);
    /// for i in 0..n {
    ///     a[i] /= n as f64;
    ///     b[i] /= n as f64;
    /// }
    /// for i in 0..n {
    ///     assert_eq!(a[i].round() as usize, i);
    ///     assert_eq!(b[i].round() as usize, n - 1 - i);
    /// }
    /// ```
    ///
    pub fn inv_transform(&mut self, sr: &mut Vec<f64>, si: &mut Vec<f64>, in_ord_normal: bool) {
        debug_assert_eq!(sr.len(), si.len(), "Length of two vectors should be equal");
        debug_assert!(sr.len().count_ones() == 1, "Length is not power of 2");
        let k: usize = sr.len().trailing_zeros() as usize;
        if in_ord_normal {
            for i in 0.. sr.len() {
                let j = i.reverse_bits() >> (size_of::<usize>() * 8 - k);
                if i < j {
                    sr.swap(i, j);
                    si.swap(i, j);
                }
            }
        }
        self.reserve(k);
        unsafe { self.algo.inv_transform(&self.wr, &self.wi, sr, si, k); }
    }

    ///
    /// This function performs convolution between two real signals and put the resulting signal
    /// in alternating halves of the input signals. For example, if we perform convolution between
    /// two signals a, b of length n, the result will be contained in a\[0\], b\[0\], a\[1\], b\[1\],
    /// a\[2\], b\[2\] ..... a\[n/2 - 1\], b\[n/2 - 1\]
    ///
    /// # Parameters
    ///
    /// * `sr1` A real signal
    /// * `sr2` Another real signal
    ///
    /// # Important
    ///
    /// * Length of the two input vectors must be same.
    /// * The length must be power of 2.
    /// * It will be slightly faster if the two signals are properly aligned according to the
    /// maximum SIMD instruction set supported by the cpu (avx --> 32 byte, sse2 --> 16 byte etc.).
    ///
    /// # Example
    ///
    /// ```
    /// let n1: usize = 300;
    /// let n2: usize = 1000;
    /// let mut a = Vec::with_capacity(n1);
    /// let mut b = Vec::with_capacity(n2);
    /// for i in 1..=n1 {
    ///     a.push(i as f64);
    /// }
    /// for i in 1..=n2 {
    ///     b.push(i as f64);
    /// }
    /// let mut c = vec![0.0; n1 + n2 - 1];
    /// for i in 0..n1 {
    ///     for j in 0..n2 {
    ///         c[i + j] += a[i] * b[j];
    ///     }
    /// }
    /// let mut n = n1 + n2 - 1;
    /// let mut k = 63 - n.leading_zeros();
    /// if n.count_ones() > 1 { k += 1; }
    /// n = 1_usize << k;
    /// a.resize(n, 0.0);
    /// b.resize(n, 0.0);
    /// sleekfft_rs::Fft::new().cyclic_convolution(&mut a, &mut b);
    /// for i in 0..(n1 + n2 - 1) {
    ///     if (i & 1) == 0 {
    ///         assert_eq!(c[i].round() as u64, a[i >> 1].round() as u64);
    ///     } else {
    ///         assert_eq!(c[i].round() as u64, b[i >> 1].round() as u64);
    ///     }
    /// }
    /// ```
    ///
    pub fn cyclic_convolution(&mut self, sr1: &mut Vec<f64>, sr2: &mut Vec<f64>) {
        debug_assert_eq!(sr1.len(), sr2.len(), "Length of two vectors should be equal");
        debug_assert!(sr1.len().count_ones() == 1, "Length is not power of 2");
        let n: usize = sr1.len();
        let k: usize = n.trailing_zeros() as usize;
        self.reserve(k);
        unsafe { self.algo.fwd_transform(&self.wr, &self.wi, sr1, sr2, k); }
        (sr2[0], sr2[1]) = (4.0 * sr1[0] * sr2[0], 4.0 * sr1[1] * sr2[1]);
        (sr1[0], sr1[1]) = (0.0, 0.0);
        let (mut i, mut j) = (2_usize, 4_usize);
        while i < n {
            for x in (i..j).step_by(2) {
                let y = x ^ (i - 1);
                let (ar, ai) = (sr1[x] - sr1[y], sr2[x] + sr2[y]);
                let (br, bi) = (sr1[x] + sr1[y], sr2[x] - sr2[y]);
                (sr1[x], sr2[x]) = (ar.mul_add(br, -ai * bi), ar.mul_add(bi, ai * br));
                (sr1[y], sr2[y]) = (-sr1[x], sr2[x]);
            }
            i <<= 1;
            j <<= 1;
        }
        let fc = 0.25_f64 / n as f64;
        for i in (0..n).step_by(2) {
            let (j, x) = (i | 1, i >> 1);
            let (ar, ai) = (sr1[i] + sr1[j], sr2[i] + sr2[j]);
            let (mut br, mut bi) = (sr1[i] - sr1[j], sr2[i] - sr2[j]);
            (br, bi) = (
                br.mul_add(self.wr[x], bi * self.wi[x]),
                bi.mul_add(self.wr[x], -br * self.wi[x]),
            );
            (sr1[x], sr2[x]) = ((br + ai) * fc, (bi - ar) * fc);
        }
        unsafe { self.algo.inv_transform(&self.wr, &self.wi, sr1, sr2, k - 1); }
    }

}

