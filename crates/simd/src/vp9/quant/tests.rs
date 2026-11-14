use super::{dz, dz_scalar};

#[test]
fn matches_the_scalar_twin() {
    let mut seed = 5u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        seed >> 8
    };
    for case in 0..500 {
        let n = [16, 64, 256, 1024, 15, 3, 1][case % 7];
        let scale = [1.0f32, 30.0, 1000.0, 1e7][case % 4];
        let coef: Vec<f32> = (0..n)
            .map(|_| ((rnd() % 20001) as f32 - 10000.0) / 10000.0 * scale)
            .collect();
        let (inv, step, round) = (1.0 / (4.0 + (rnd() % 400) as f32), 8.0, 0.33);
        let mut a = vec![0i32; n];
        let mut b = vec![0i32; n];
        let x = dz(&coef, &mut a, inv, step, round);
        let y = dz_scalar(&coef, &mut b, inv, step, round);
        assert_eq!(a, b, "case {case}");
        assert_eq!(x.bits, y.bits);
        assert_eq!(x.nonzero, y.nonzero);
        let close = |p: f32, q: f32| (p - q).abs() <= 1e-3 * p.abs().max(q.abs()) + 1e-3;
        assert!(
            close(x.err, y.err) && close(x.energy, y.energy),
            "{x:?} {y:?}"
        );
    }
}
