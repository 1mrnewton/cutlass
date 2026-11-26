use super::{EdgeUnits, Edges, edge_codes, levels, levels_odd, scalar};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: u32) -> u8 {
        (self.next() % n) as u8
    }
}

/// A transform size of 4 to 128 samples on each side.
fn dims(rng: &mut Rng) -> u8 {
    rng.below(6) | rng.below(6) << 3
}

#[test]
fn the_codes_match_one_unit_at_a_time() {
    let mut rng = Rng(0x0DE6E);
    let mut on = 0;
    for round in 0..20_000 {
        let u = EdgeUnits {
            dims: core::array::from_fn(|_| dims(&mut rng)),
            across: core::array::from_fn(|_| dims(&mut rng)),
            // Many zero levels, so both choices of level are taken.
            lvl: core::array::from_fn(|_| rng.below(64).saturating_sub(16)),
            lvl_across: core::array::from_fn(|_| rng.below(64).saturating_sub(16)),
        };
        let edges = if round % 2 == 0 {
            Edges::Left {
                first: rng.below(256),
            }
        } else {
            Edges::Top {
                zeros: rng.below(8),
            }
        };
        let luma = round % 4 < 2;
        let (got, mask) = edge_codes(&u, edges, luma);
        assert_eq!(mask, scalar::edge_codes(&u, edges, luma).1);
        for (l, &g) in got.iter().enumerate() {
            assert_eq!(mask >> l & 1 == 1, g != 0);
            assert_eq!(
                g,
                scalar::edge_code(&u, l, edges, luma),
                "{u:?} {edges:?} luma {luma} lane {l}"
            );
            on += usize::from(g != 0);
        }
    }
    assert!(on > 50_000, "{on} edges");
}

#[test]
fn the_sizes_are_those_of_the_specification() {
    // A unit at the top of an 8x8 transform (dims 1 | 1 << 3) in row 2,
    // below transforms 4 to 32 samples high.
    let code = |across: u8, luma: bool| {
        let u = EdgeUnits {
            dims: [1 | 1 << 3; 16],
            across: [across << 3; 16],
            lvl: [9; 16],
            lvl_across: [0; 16],
        };
        edge_codes(&u, Edges::Top { zeros: 1 }, luma).0[0]
    };
    assert_eq!(code(0, true), 4 << 8 | 9);
    assert_eq!(code(1, true), 8 << 8 | 9);
    assert_eq!(code(3, true), 8 << 8 | 9);
    assert_eq!(code(0, false), 4 << 8 | 9);
    assert_eq!(code(2, false), 6 << 8 | 9);
    // Column 9 is inside the transform: no edge.
    let u = EdgeUnits {
        dims: [1; 16],
        across: [1; 16],
        lvl: [9; 16],
        lvl_across: [9; 16],
    };
    let c = edge_codes(&u, Edges::Left { first: 8 }, true).0;
    assert!(c[0] != 0 && c[1] == 0 && c[2] != 0);
}

#[test]
fn levels_pick_their_byte() {
    let mut rng = Rng(5);
    for i in 0..4 {
        let r: [[u8; 4]; 32] = core::array::from_fn(|_| core::array::from_fn(|_| rng.below(256)));
        let first: [[u8; 4]; 16] = core::array::from_fn(|k| r[k]);
        assert_eq!(levels(&first, i), scalar::levels(&first, i));
        assert_eq!(levels(&first, i), core::array::from_fn(|k| r[k][i] & 63));
        assert_eq!(levels_odd(&r, i), scalar::levels_odd(&r, i));
        assert_eq!(
            levels_odd(&r, i),
            core::array::from_fn(|k| r[2 * k + 1][i] & 63)
        );
    }
}

#[test]
fn every_column_row_and_size_gives_the_code_of_the_specification() {
    // Every first column (the count wraps at 256), every count of zero
    // bits, every pair of sizes, with levels on one side, the other, both
    // and neither, and bytes whose unused high bits are set.
    for luma in [true, false] {
        for first in 0..=255u8 {
            for round in 0..4u8 {
                let u = EdgeUnits {
                    dims: core::array::from_fn(|l| {
                        ((l as u8).wrapping_add(first) % 6)
                            | ((l as u8 + round) % 6) << 3
                            | round << 6
                    }),
                    across: core::array::from_fn(|l| {
                        ((l as u8 * 5 + round) % 6)
                            | ((l as u8 * 3).wrapping_add(first) % 6) << 3
                            | 0xC0
                    }),
                    lvl: core::array::from_fn(|l| {
                        if (l as u8 + round).is_multiple_of(3) {
                            0
                        } else {
                            1 + (first.wrapping_mul(7).wrapping_add(l as u8) % 63)
                        }
                    }),
                    lvl_across: core::array::from_fn(|l| {
                        if (l as u8).wrapping_add(first).is_multiple_of(4) {
                            0
                        } else {
                            63 - (l as u8 + round) % 5
                        }
                    }),
                };
                for edges in [Edges::Left { first }, Edges::Top { zeros: first % 9 }] {
                    let (got, mask) = edge_codes(&u, edges, luma);
                    let want: [u16; 16] =
                        core::array::from_fn(|l| scalar::edge_code(&u, l, edges, luma));
                    assert_eq!(got, want, "{u:?} {edges:?} luma {luma}");
                    assert_eq!(mask, scalar::edge_codes(&u, edges, luma).1);
                }
            }
        }
    }
}

#[test]
fn units_without_an_edge_or_a_level_give_no_code_and_no_bit() {
    let none = EdgeUnits {
        dims: [4 | 4 << 3; 16],
        across: [4 | 4 << 3; 16],
        lvl: [40; 16],
        lvl_across: [40; 16],
    };
    // Columns 1 to 16 and row 1 are inside 64x64 transforms, but for
    // the unit at column 16.
    let (codes, mask) = edge_codes(&none, Edges::Left { first: 1 }, true);
    assert_eq!(mask, 1 << 15);
    assert!(codes[..15].iter().all(|&c| c == 0) && codes[15] == 16 << 8 | 40);
    assert_eq!(
        edge_codes(&none, Edges::Top { zeros: 0 }, true),
        ([0; 16], 0)
    );
    let level = EdgeUnits {
        dims: [0; 16],
        across: [0; 16],
        lvl: [0; 16],
        lvl_across: [0; 16],
    };
    assert_eq!(
        edge_codes(&level, Edges::Left { first: 0 }, false),
        ([0; 16], 0)
    );
    assert_eq!(
        edge_codes(&level, Edges::Top { zeros: 7 }, true),
        ([0; 16], 0)
    );
    // Every unit an edge.
    let all = EdgeUnits {
        lvl: [63; 16],
        ..level
    };
    assert_eq!(
        edge_codes(&all, Edges::Top { zeros: 0 }, true),
        ([4 << 8 | 63; 16], 0xFFFF)
    );
}

#[test]
fn levels_take_any_byte_index_by_its_low_two_bits() {
    let r: [[u8; 4]; 32] = core::array::from_fn(|k| {
        core::array::from_fn(|i| {
            (k as u8)
                .wrapping_mul(37)
                .wrapping_add((i as u8).wrapping_mul(101))
                | 0x80
        })
    });
    let first: [[u8; 4]; 16] = core::array::from_fn(|k| r[k]);
    for i in 0..12 {
        assert_eq!(levels(&first, i), scalar::levels(&first, i & 3));
        assert_eq!(levels_odd(&r, i), scalar::levels_odd(&r, i & 3));
    }
}
