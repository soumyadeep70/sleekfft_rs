pub mod sleekfft;

#[cfg(test)]
mod tests {
    use super::*;

    fn test_convolution(n1: usize, n2: usize) {
        let mut a = Vec::with_capacity(n1);
        let mut b = Vec::with_capacity(n2);
        for i in 1..=n1 {
            a.push(i as f64);
        }
        for i in 1..=n2 {
            b.push(i as f64);
        }
        let mut c = vec![0.0; n1 + n2 - 1];
        for i in 0..n1 {
            for j in 0..n2 {
                c[i + j] += a[i] * b[j];
            }
        }
        let mut n = n1 + n2 - 1;
        let mut k = 63 - n.leading_zeros();
        if n.count_ones() > 1 { k += 1; }
        n = 1_usize << k;
        a.resize(n, 0.0);
        b.resize(n, 0.0);
        sleekfft::Fft::new().cyclic_convolution(&mut a, &mut b);
        for i in 0..(n1 + n2 - 1) {
            if (i & 1) == 0 {
                assert_eq!(c[i].round() as u64, a[i >> 1].round() as u64);
            } else {
                assert_eq!(c[i].round() as u64, b[i >> 1].round() as u64);
            }
        }
    }

    #[test]
    fn test_conv() {
        test_convolution(335, 4014);
        test_convolution(5450, 9412);
        test_convolution(50000, 50000);
    }
}
