//! `av1::cpu::force_scalar` turns the run-time kernel choices off for the
//! whole process, so it has a test binary of its own: the entries then
//! give what they gave with the kernels on.

use cutlass_simd::av1::{cpu, mc};

#[test]
fn the_entries_give_the_same_samples_with_the_kernels_forced_off() {
    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 33) as u32
    };
    let stride = 150;
    let src: Vec<u8> = (0..stride * 140).map(|_| (next() % 256) as u8).collect();
    let filters: [[i8; 8]; 3] = [
        [0, 2, -14, 110, 38, -10, 2, 0],
        [-4, 12, -24, 80, 80, -24, 12, -4],
        [0, 0, 0, 64, 64, 0, 0, 0],
    ];
    let run = |w: usize, h: usize, fh: &[i8; 8], fv: &[i8; 8]| {
        let b = mc::Block {
            src: &src,
            start: 11,
            src_stride: stride,
            w,
            h,
            fh,
            fv,
        };
        let mut prep = vec![0i16; 3 * w * h];
        let mut put = vec![0u8; 3 * w * h];
        let n = w * h;
        assert!(mc::prep_hv_u8(&mut prep[..n], w, &b));
        assert!(mc::prep_h_u8(&mut prep[n..2 * n], w, &b));
        assert!(mc::prep_v_u8(&mut prep[2 * n..], w, &b));
        assert!(mc::put_hv_u8(&mut put[..n], w, &b));
        assert!(mc::put_h_u8(&mut put[n..2 * n], w, &b));
        assert!(mc::put_v_u8(&mut put[2 * n..], w, &b));
        (prep, put)
    };
    let cases = [(8, 8), (16, 16), (32, 7), (64, 64), (128, 9)];
    let with: Vec<_> = cases
        .iter()
        .flat_map(|&(w, h)| filters.iter().map(move |f| (w, h, f)))
        .map(|(w, h, f)| run(w, h, f, &filters[(w + h) % 3]))
        .collect();
    cpu::force_scalar(true);
    assert!(!cpu::avx2());
    assert!(!mc::accelerated() || cfg!(target_feature = "dotprod"));
    let without: Vec<_> = cases
        .iter()
        .flat_map(|&(w, h)| filters.iter().map(move |f| (w, h, f)))
        .map(|(w, h, f)| run(w, h, f, &filters[(w + h) % 3]))
        .collect();
    cpu::force_scalar(false);
    assert_eq!(with, without);
    assert_eq!(cpu::avx2(), cutlass_simd::wide::has_avx2());
}
