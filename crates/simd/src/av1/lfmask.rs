//! Where the deblocking filter runs (AV1 specification 7.14.2 to
//! 7.14.5), sixteen 4x4 units of a row at a time: whether each unit's
//! left or top side is an edge to filter, with what filter and level.
//!
//! A unit is described by bytes the decoder keeps for it: its transform
//! size, and an 8-byte record of its block whose first four bytes are
//! the block's filter levels.

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod scalar;
#[cfg(test)]
mod tests;

/// A unit's transform size: the base-2 logarithms of its width and
/// height in 4x4 units, in the low three bits and the three above them.
pub type Dims = u8;

/// Sixteen units of a row and the units across their edges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EdgeUnits {
    /// The transform size of each unit.
    pub dims: [Dims; 16],
    /// For horizontal edges, the transform size of the unit above each
    /// unit. For vertical edges, that of the sixteen units before these
    /// in the row, the last of them left of the first unit.
    pub across: [Dims; 16],
    /// The filter level of each unit's block.
    pub lvl: [u8; 16],
    /// The filter level of the block above each unit, or of the units
    /// before these, as for [`Self::across`].
    pub lvl_across: [u8; 16],
}

/// The edges [`edge_codes`] looks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edges {
    /// The left sides of units in columns `first, first + 1, ...` (only
    /// the low five bits count): a unit's left side is an edge when its
    /// column is a multiple of its transform's width.
    Left { first: u8 },
    /// The top sides of units in a row whose index has `zeros` trailing
    /// zero bits: a unit's top side is an edge when the row is a multiple
    /// of its transform's height.
    Top { zeros: u8 },
}

/// For each of the sixteen units, 0 when its side is no edge or both
/// levels are 0, else the filter size times 256 plus the level: the
/// unit's own level, or when that is 0 the level across the edge. Also
/// the units that are not 0, as bits from the lowest.
///
/// The filter size follows from the smaller transform size `m` (in 4x4
/// units, as a base-2 logarithm) on the two sides: for luma 4, 8 or 16
/// samples for `m` of 0, 1 and more; for chroma 4, else 6 (the 6-tap
/// filter of filter size 8).
#[inline]
#[must_use]
pub fn edge_codes(u: &EdgeUnits, edges: Edges, luma: bool) -> ([u16; 16], u16) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the `cfg` above proves NEON is enabled for this build,
        // and the kernel names AES only where the build has it too.
        #[allow(unsafe_code)]
        unsafe {
            neon::edge_codes(u, edges, luma)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::edge_codes(u, edges, luma)
}

/// The low 6 bits of byte `i` (0 to 3, a filter level) of each of
/// sixteen records.
#[inline]
#[must_use]
pub fn levels(records: &[[u8; 4]; 16], i: usize) -> [u8; 16] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: as in `edge_codes`.
        #[allow(unsafe_code)]
        unsafe {
            neon::levels(records, i & 3)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::levels(records, i & 3)
}

/// [`levels`] of each odd record of thirty-two: the records of the luma
/// units holding the bottom-right corners of sixteen chroma units, when
/// chroma has half the columns.
#[inline]
#[must_use]
pub fn levels_odd(records: &[[u8; 4]; 32], i: usize) -> [u8; 16] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: as in `edge_codes`.
        #[allow(unsafe_code)]
        unsafe {
            neon::levels_odd(records, i & 3)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::levels_odd(records, i & 3)
}
