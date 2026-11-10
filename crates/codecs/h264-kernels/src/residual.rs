//! The inverse transform of a 4×4 residual block added to the picture,
//! clause 8.5.12.
//!
//! The arithmetic is 32-bit, as wide as the decoder's own, so the vector
//! version gives the same samples as the plain one for any coefficients,
//! not only ones a conforming stream can hold.

/// Inverse-transforms `block`, scaled coefficients in raster order, and
/// adds it to the 4×4 samples of `plane` whose first is at `at`, with rows
/// `down` apart. `false` means the samples are not all in the plane and
/// nothing was done.
pub fn idct4_add(plane: &mut [u8], at: usize, down: usize, block: &[i32; 16]) -> bool {
    kernel!(idct4_add(plane, at, down, block))
}

/// Adds `dc` to each sample of the `size`×`size` block of `plane` whose
/// first sample is at `at`, with rows `down` apart, and holds each to 0 to
/// 255. `size` is 4 or 8. `false` means it is another size or the samples
/// are not all in the plane, and nothing was done.
pub fn dc_add(plane: &mut [u8], at: usize, down: usize, dc: i32, size: usize) -> bool {
    kernel!(dc_add(plane, at, down, dc, size))
}

/// The plane reads and writes around whichever `idct4_rows` is in scope: it
/// takes the block and the four rows of four samples, and gives the rows
/// back with the residual added.
macro_rules! residual_kernels {
    ($visibility:vis, $(#[$feature:meta])*) => {
        /// See [`crate::residual::idct4_add`].
        $(#[$feature])*
        $visibility fn idct4_add(plane: &mut [u8], at: usize, down: usize, block: &[i32; 16]) -> bool {
            if at + 3 * down + 4 > plane.len() {
                return false;
            }
            let mut rows = [[0u8; 4]; 4];
            for (index, row) in rows.iter_mut().enumerate() {
                let start = at + index * down;
                row.copy_from_slice(&plane[start..start + 4]);
            }
            let rows = idct4_rows(block, rows);
            for (index, row) in rows.iter().enumerate() {
                let start = at + index * down;
                plane[start..start + 4].copy_from_slice(row);
            }
            true
        }

        /// See [`crate::residual::dc_add`]. `dc_row` adds to the eight
        /// samples in a `u64`.
        $(#[$feature])*
        $visibility fn dc_add(plane: &mut [u8], at: usize, down: usize, dc: i32, size: usize) -> bool {
            if !(size == 4 || size == 8) {
                return false;
            }
            let Some(rest) = plane.len().checked_sub(at) else {
                return false;
            };
            if (size - 1).checked_mul(down).is_none_or(|rows| rows + size > rest) {
                return false;
            }
            for index in 0..size {
                let row = &mut plane[at + index * down..];
                if size == 8 {
                    if let Some(row) = row.first_chunk_mut::<8>() {
                        *row = dc_row(u64::from_le_bytes(*row), dc).to_le_bytes();
                    }
                } else if let Some(row) = row.first_chunk_mut::<4>() {
                    let samples = u64::from(u32::from_le_bytes(*row));
                    *row = (dc_row(samples, dc) as u32).to_le_bytes();
                }
            }
            true
        }
    };
}
pub(crate) use residual_kernels;

#[cfg(test)]
mod tests {
    use crate::scalar;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }
    }

    #[test]
    fn the_dc_add_matches_the_plain_one() {
        let mut rng = Lcg(11);
        for round in 0..20_000 {
            let down = 9 + (rng.next() % 8) as usize;
            let before: Vec<u8> = (0..down * 12).map(|_| rng.next() as u8).collect();
            let size = [4, 8, 5][round % 3];
            // Mostly inside the plane, sometimes off its end.
            let at = (rng.next() as usize) % (down * 6);
            let dc = (rng.next() % 1201) as i32 - 600;
            let mut fast = before.clone();
            let mut plain = before.clone();
            let done = super::dc_add(&mut fast, at, down, dc, size);
            let wanted = scalar::dc_add(&mut plain, at, down, dc, size);
            assert_eq!(done, wanted, "round {round}");
            assert_eq!(fast, plain, "round {round}: dc {dc}, size {size}");
            if !done {
                assert_eq!(fast, before);
            }
        }
    }

    #[test]
    fn the_transform_and_add_match_the_plain_ones() {
        let mut rng = Lcg(21);
        let stride = 12usize;
        for round in 0..60_000u32 {
            let plane: Vec<u8> = (0..stride * 8)
                .map(|_| match rng.next() % 6 {
                    0 => 0,
                    1 => 255,
                    _ => rng.next() as u8,
                })
                .collect();
            // Small coefficients, ones as large as a stream can scale to,
            // and ones far past that.
            let span = [16, 1 << 9, 1 << 15, 1 << 24][(round % 4) as usize];
            let fill = 1 + rng.next() % 16;
            let mut block = [0i32; 16];
            for level in block.iter_mut().take(fill as usize) {
                *level = (rng.next() % (2 * span)) as i32 - span as i32;
            }
            let at = (rng.next() % 4) as usize * stride + (rng.next() % 8) as usize;
            let (mut fast, mut plain) = (plane.clone(), plane.clone());
            assert!(super::idct4_add(&mut fast, at, stride, &block));
            assert!(scalar::idct4_add(&mut plain, at, stride, &block));
            assert_eq!(fast, plain, "round {round} block {block:?}");
        }
    }

    #[test]
    fn a_lone_dc_adds_the_same_value_everywhere() {
        let mut plane = vec![100u8; 16];
        let mut block = [0i32; 16];
        block[0] = 640;
        assert!(super::idct4_add(&mut plane, 0, 4, &block));
        assert_eq!(plane, vec![110u8; 16]);
        block[0] = -64_000;
        assert!(super::idct4_add(&mut plane, 0, 4, &block));
        assert_eq!(plane, vec![0u8; 16]);
        // A block that does not fit is refused.
        assert!(!super::idct4_add(&mut plane, 4, 4, &block));
    }
}
