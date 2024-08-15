use std::f64::consts::PI;
use crate::sleekfft::generic::fft_gen::FftGeneric;
use crate::sleekfft::x86::{
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

    pub fn fwd_transform(&mut self, sr: &mut Vec<f64>, si: &mut Vec<f64>) {
        debug_assert_eq!(sr.len(), si.len(), "Length of two vectors should be equal");
        debug_assert!(sr.len().count_ones() == 1, "Length is not power of 2");
        let k: usize = sr.len().trailing_zeros() as usize;
        self.reserve(k);
        unsafe { self.algo.fwd_transform(&self.wr, &self.wi, sr, si, k); }
    }

    pub fn inv_transform(&mut self, sr: &mut Vec<f64>, si: &mut Vec<f64>) {
        debug_assert_eq!(sr.len(), si.len(), "Length of two vectors should be equal");
        debug_assert!(sr.len().count_ones() == 1, "Length is not power of 2");
        let k: usize = sr.len().trailing_zeros() as usize;
        self.reserve(k);
        unsafe { self.algo.inv_transform(&self.wr, &self.wi, sr, si, k); }
    }

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

