//! # sleekfft_rs
//!
//! `sleekfft_rs` is a module written in pure Rust which provides efficient functions
//! to perform fast fourier transform of power of two sized signals. It also provides
//! a convolution function which performs convolution between two real signals.
//!
//! This module only provides fft functions for double precision floating point numbers.
//!
//! It utilizes hardware SIMD instructions to accelerate the algorithms. No special code is
//! needed to utilize SIMD. The `Fft` struct will choose the best SIMD instruction available
//! dynamically. This crate often beats FFTW and RustFFT in terms of performance.
//!
//! Currently only x86 SIMD instruction set (SSE2, AVX, FMA) is supported. On any other
//! platform it will use scalar operations.
//!

mod traits;
mod traits_impl_generic;
mod traits_x86_impl;
mod fft;

pub use self::fft::Fft;