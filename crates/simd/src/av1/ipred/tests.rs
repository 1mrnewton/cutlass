use super::{FilterTaps, filter_intra, scalar};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

/// Weights like the specification's: summing to 16, at least -12
/// negative, at most -6 each.
fn taps(rng: &mut Rng) -> FilterTaps {
    core::array::from_fn(|_| {
        let mut t = [0i8; 7];
        let mut neg = 0i32;
        for v in &mut t {
            let x = (rng.next() % 19) as i32 - 6;
            let x = if x < 0 && neg + x < -12 { 0 } else { x };
            neg += x.min(0);
            *v = x as i8;
        }
        let s: i32 = t.iter().map(|&v| i32::from(v)).sum();
        // Put the rest on the largest weight.
        let i = (0..7).max_by_key(|&i| t[i]).unwrap_or(0);
        t[i] = (i32::from(t[i]) + 16 - s).clamp(-6, 127) as i8;
        t
    })
}

#[test]
fn the_filter_matches_the_scalar_one() {
    let mut rng = Rng(7);
    for _ in 0..3000 {
        let w = 4 << (rng.next() % 4);
        let h = 2 << (rng.next() % 5).min(4);
        let (w, h) = (w.min(32), h.min(32));
        let stride = w + (rng.next() % 9) as usize;
        let t = taps(&mut rng);
        let above: Vec<u8> = (0..=w).map(|_| rng.next() as u8).collect();
        let left: Vec<u8> = (0..h).map(|_| rng.next() as u8).collect();
        let init: Vec<u8> = (0..stride * h).map(|_| rng.next() as u8).collect();
        let (mut a, mut b) = (init.clone(), init);
        filter_intra(&mut a, stride, w, h, &t, &above, &left);
        scalar::filter_intra(&mut b, stride, w, h, &t, &above, &left);
        assert_eq!(a, b, "{w}x{h} {t:?}");
    }
}
