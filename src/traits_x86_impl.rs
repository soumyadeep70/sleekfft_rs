use crate::impl_radix4fft_f64;
use crate::traits::Radix4Fft;
use std::arch::x86_64::*;
pub(crate) struct FftSse;

impl_radix4fft_f64!(
    "x86", "x86_64";
    "sse2";
    FftSse;
    _mm_loadu_pd,
    _mm_storeu_pd,
    _mm_add_pd,
    _mm_sub_pd,
    |a: __m128d, b: __m128d, c: __m128d, d: __m128d| -> __m128d {
        _mm_add_pd(_mm_mul_pd(a, b), _mm_mul_pd(c, d))
    },
    |a: __m128d, b: __m128d, c: __m128d, d: __m128d| -> __m128d {
        _mm_sub_pd(_mm_mul_pd(a, b), _mm_mul_pd(c, d))
    },
    |a: &f64| -> __m128d {
        _mm_set1_pd(*a)
    },
    2,
);

pub(crate) struct FftAvx;

impl_radix4fft_f64!(
    "x86", "x86_64";
    "avx";
    FftAvx;
    _mm256_loadu_pd,
    _mm256_storeu_pd,
    _mm256_add_pd,
    _mm256_sub_pd,
    |a: __m256d, b: __m256d, c: __m256d, d: __m256d| -> __m256d {
        _mm256_add_pd(_mm256_mul_pd(a, b), _mm256_mul_pd(c, d))
    },
    |a: __m256d, b: __m256d, c: __m256d, d: __m256d| -> __m256d {
        _mm256_sub_pd(_mm256_mul_pd(a, b), _mm256_mul_pd(c, d))
    },
    |a: &f64| -> __m256d {
        _mm256_broadcast_sd(a)
    },
    4,
);

pub(crate) struct FftAvxFma;

impl_radix4fft_f64!(
    "x86", "x86_64";
    "avx", "fma";
    FftAvxFma;
    _mm256_loadu_pd,
    _mm256_storeu_pd,
    _mm256_add_pd,
    _mm256_sub_pd,
    |a: __m256d, b: __m256d, c: __m256d, d: __m256d| -> __m256d {
        _mm256_add_pd(_mm256_mul_pd(a, b), _mm256_mul_pd(c, d))
    },
    |a: __m256d, b: __m256d, c: __m256d, d: __m256d| -> __m256d {
        _mm256_sub_pd(_mm256_mul_pd(a, b), _mm256_mul_pd(c, d))
    },
    |a: &f64| -> __m256d {
        _mm256_broadcast_sd(a)
    },
    4,
);