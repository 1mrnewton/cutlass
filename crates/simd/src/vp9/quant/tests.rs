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

#[test]
fn scan_matches_the_scalar_twin() {
    use super::{Scan, Trial, scan, scan_scalar};
    let mut seed = 7u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        seed >> 8
    };
    for case in 0..400 {
        let n = [16, 64, 256, 1024, 24][case % 5];
        let scale = [1.0f32, 30.0, 1000.0, 60000.0][case % 4];
        let coef: Vec<f32> = (0..n)
            .map(|_| ((rnd() % 20001) as f32 - 10000.0) / 10000.0 * scale)
            .collect();
        // A permutation for the scan.
        let mut iscan: Vec<u16> = (0..n as u16).collect();
        for i in (1..n).rev() {
            iscan.swap(i, rnd() as usize % (i + 1));
        }
        let inv = 1.0 / (4.0 + (rnd() % 400) as f32);
        let th = [0.5f32, 0.67][case % 2];
        let with = case % 3 != 0;
        let mut out = [(); 2].map(|()| (vec![0f32; n], vec![0u32; n], vec![0u8; n]));
        let mut res = [Scan::default(); 2];
        for (j, (sizes, levels, energy)) in out.iter_mut().enumerate() {
            let mut t = Trial {
                levels,
                energy,
                round: 0.33,
                step: 8.0,
                last_band: [21, 13][case % 2],
                band_cost: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            };
            let t = with.then_some(&mut t);
            res[j] = if j == 0 {
                scan(&coef, &iscan, sizes, inv, th, t)
            } else {
                scan_scalar(&coef, &iscan, sizes, inv, th, t)
            };
        }
        assert_eq!(out[0], out[1], "case {case}");
        assert_eq!(res[0].energy, res[1].energy, "case {case}");
        assert_eq!(res[0].last, res[1].last, "case {case}");
        let (p, q) = (res[0].gain, res[1].gain);
        assert!(
            (p - q).abs() <= 1e-3 * p.abs().max(q.abs()) + 1e-3,
            "case {case}: {p} {q}"
        );
    }
}
