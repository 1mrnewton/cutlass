use super::{Filters, prep8x8, put8x8, scalar};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// A value in `-n..=n`.
    fn signed(&mut self, n: u32) -> i32 {
        (self.next() % (2 * n + 1)) as i32 - n as i32
    }
}

/// A table of filters that each sum to 128, their taps' absolute values
/// to at most 255, some at that bound.
fn table(rng: &mut Rng) -> [[i8; 8]; 193] {
    core::array::from_fn(|i| {
        loop {
            let spread = if i % 7 == 0 { 64 } else { 24 };
            let mut f: [i32; 8] = core::array::from_fn(|_| rng.signed(spread));
            f[3] = 0;
            f[3] = 128 - f.iter().sum::<i32>();
            let fits = f.iter().all(|&t| (-128..=127).contains(&t));
            if fits && f.iter().map(|t| t.abs()).sum::<i32>() <= 255 {
                break f.map(|t| t as i8);
            }
        }
    })
}

#[test]
fn the_warp_matches_its_twin() {
    let mut rng = Rng(0x3A4);
    for round in 0..2_000 {
        let table = table(&mut rng);
        let rows: [[u8; 16]; 15] = core::array::from_fn(|_| {
            core::array::from_fn(|_| match rng.next() % 8 {
                0 => 0,
                1 => 255,
                _ => rng.next() as u8,
            })
        });
        let rows = core::array::from_fn(|r| &rows[r]);
        // Parameters that keep every filter inside the table.
        let f = Filters {
            table: &table,
            h: [rng.signed(32_768), rng.signed(2_048), rng.signed(2_048)],
            v: [rng.signed(32_768), rng.signed(4_096), rng.signed(4_096)],
        };
        assert_eq!(put8x8(rows, &f), scalar::put8x8(rows, &f), "round {round}");
        assert_eq!(
            prep8x8(rows, &f),
            scalar::prep8x8(rows, &f),
            "round {round}"
        );
    }
}
