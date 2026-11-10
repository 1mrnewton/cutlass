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
