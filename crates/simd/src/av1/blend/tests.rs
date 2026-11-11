use super::{mask_blend8, scalar};

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

/// The specification's blend of one sample.
fn spec(a: i16, b: i16, m: u8) -> u8 {
    let (a, b, m) = (i32::from(a), i32::from(b), i32::from(m));
    ((m * a + (64 - m) * b + 512) >> 10).clamp(0, 255) as u8
}

#[test]
fn the_blend_matches_the_specification_for_8_bit_intermediates() {
    let mut rng = Rng(0xB1E4D);
    for round in 0..50_000 {
        // Intermediates of 8-bit predictions, with the ends of the range
        // now and then.
        let value = |rng: &mut Rng| -> i16 {
            match rng.next() % 16 {
                0 => -5131,
                1 => 9212,
                _ => (rng.next() % 14_344) as i16 - 5131,
            }
        };
        let a: [i16; 8] = core::array::from_fn(|_| value(&mut rng));
        let b: [i16; 8] = core::array::from_fn(|_| value(&mut rng));
        let m: [u8; 8] = core::array::from_fn(|i| match (round + i) % 9 {
            0 => 0,
            1 => 64,
            _ => (rng.next() % 65) as u8,
        });
        let want: [u8; 8] = core::array::from_fn(|i| spec(a[i], b[i], m[i]));
        assert_eq!(scalar::mask_blend8(&a, &b, &m), want, "{a:?} {b:?} {m:?}");
        assert_eq!(mask_blend8(&a, &b, &m), want, "{a:?} {b:?} {m:?}");
    }
}

#[test]
fn the_kernel_wraps_as_its_twin_on_any_values() {
    let mut rng = Rng(7);
    for _ in 0..50_000 {
        let a: [i16; 8] = core::array::from_fn(|_| rng.next() as i16);
        let b: [i16; 8] = core::array::from_fn(|_| rng.next() as i16);
        let m: [u8; 8] = core::array::from_fn(|_| (rng.next() % 65) as u8);
        assert_eq!(mask_blend8(&a, &b, &m), scalar::mask_blend8(&a, &b, &m));
    }
}
