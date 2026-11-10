use super::{direction_u8, direction_u16, scalar};

/// A small fixed-seed generator: the same blocks on every run.
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

/// Blocks that reach the extremes of the sums, then textured random ones.
fn blocks(bd: u32, n: usize) -> Vec<[[u16; 8]; 8]> {
    let max = (1u16 << bd) - 1;
    let mut out = vec![[[0; 8]; 8], [[max; 8]; 8], [[max / 2 + 1; 8]; 8]];
    // One edge in each direction, both polarities.
    for k in 0..16 {
        let mut b = [[0u16; 8]; 8];
        for (i, row) in b.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                let side = match k % 8 {
                    0 => i + j < 8,
                    1 => 2 * i + j < 12,
                    2 => i < 4,
                    3 => 2 * i + 7 - j < 12,
                    4 => i + 7 - j < 8,
                    5 => i + 2 * (7 - j) < 12,
                    6 => j < 4,
                    _ => i + 2 * j < 12,
                };
                *v = if side == (k < 8) { max } else { 0 };
            }
        }
        out.push(b);
    }
    let mut rng = Rng(0x00C0_FFEE ^ u64::from(bd));
    for t in 0..n {
        let mut b = [[0u16; 8]; 8];
        // Narrow ranges make near ties between directions.
        let span = [u32::from(max) + 1, 2, 4, 37][t % 4];
        let base = rng.next() % (u32::from(max) + 2 - span);
        for row in &mut b {
            for v in row {
                *v = (base + rng.next() % span) as u16;
            }
        }
        out.push(b);
    }
    out
}

#[test]
fn direction_u8_matches_the_specification() {
    for b in blocks(8, 20_000) {
        let b8 = b.map(|r| r.map(|v| v as u8));
        assert_eq!(direction_u8(&b8), scalar::direction(&b, 8), "{b:?}");
    }
}

#[test]
fn direction_u16_matches_the_specification_at_every_depth() {
    for bd in [8, 10, 12] {
        for b in blocks(bd, 20_000) {
            assert_eq!(
                direction_u16(&b, bd),
                scalar::direction(&b, bd),
                "{bd} {b:?}"
            );
        }
    }
}

#[test]
fn a_flat_block_has_no_direction_and_no_variance() {
    assert_eq!(direction_u8(&[[77; 8]; 8]), (0, 0));
    // Every cost is equal, so the first direction stays.
    assert_eq!(direction_u16(&[[1023; 8]; 8], 10), (0, 0));
}

#[test]
fn an_edge_gives_its_direction() {
    // Every row is flat, so the sums along the rows stand out: direction 2.
    let mut rows = [[0u8; 8]; 8];
    for (i, r) in rows.iter_mut().enumerate() {
        *r = [if i < 4 { 255 } else { 0 }; 8];
    }
    let (dir, var) = direction_u8(&rows);
    assert_eq!(dir, 2);
    assert!(var > 0);
    // Every column is flat: direction 6.
    let cols = [[255, 255, 255, 255, 0, 0, 0, 0]; 8];
    assert_eq!(direction_u8(&cols).0, 6);
}

#[test]
fn a_depth_outside_the_range_is_clamped() {
    let b = blocks(12, 4);
    assert_eq!(direction_u16(&b[5], 99), direction_u16(&b[5], 12));
    assert_eq!(
        direction_u16(&[[200; 8]; 8], 0),
        direction_u16(&[[200; 8]; 8], 8)
    );
}
