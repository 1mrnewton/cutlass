//! Sample depth for an 8-bit plane and a 10-bit or 12-bit plane.
//!
//! Eight-bit reconstruction stays on `u8`. Wider pictures use [`Wide`] so the
//! 8-bit kernels are not rewritten onto 16-bit storage.

/// One reconstructed sample.
pub trait Pix: Copy + Default + PartialEq + 'static {
    /// `BitDepth - 8`. Loop-filter thresholds shift by this.
    const SHIFT: u32;
    const DEPTH: u8;
    const MAX: i32;
    /// Skip the 16-bit DCT truncation used by the 8-bit kernels.
    const HIGH: bool;

    fn get(self) -> i32;
    fn put(value: i32) -> Self;

    fn put_i64(value: i64) -> Self {
        if value <= 0 {
            return Self::put(0);
        }
        if value >= i64::from(Self::MAX) {
            return Self::put(Self::MAX);
        }
        Self::put(i32::try_from(value).unwrap_or(0))
    }

    /// Unavailable above sample, `(1 << (BitDepth - 1)) - 1`.
    fn above() -> Self {
        Self::put((1 << (u32::from(Self::DEPTH) - 1)) - 1)
    }

    /// Unavailable left sample, `(1 << (BitDepth - 1)) + 1`.
    fn left() -> Self {
        Self::put((1 << (u32::from(Self::DEPTH) - 1)) + 1)
    }

    /// DC when both edges are missing, `1 << (BitDepth - 1)`.
    fn mid() -> Self {
        Self::put(1 << (u32::from(Self::DEPTH) - 1))
    }
}

impl Pix for u8 {
    const SHIFT: u32 = 0;
    const DEPTH: u8 = 8;
    const MAX: i32 = 255;
    const HIGH: bool = false;

    fn get(self) -> i32 {
        i32::from(self)
    }

    fn put(value: i32) -> Self {
        u8::try_from(value.clamp(0, 255)).unwrap_or(0)
    }
}

/// A 10-bit or 12-bit sample. `BD` is 10 or 12.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Wide<const BD: u8>(pub u16);

impl<const BD: u8> Pix for Wide<BD> {
    const SHIFT: u32 = BD as u32 - 8;
    const DEPTH: u8 = BD;
    const MAX: i32 = (1 << (BD as u32)) - 1;
    const HIGH: bool = true;

    fn get(self) -> i32 {
        i32::from(self.0)
    }

    fn put(value: i32) -> Self {
        let clipped = value.clamp(0, Self::MAX);
        Wide(u16::try_from(clipped).unwrap_or(0))
    }
}

/// Packed bytes for a cropped plane. 8-bit is one byte. 10-bit and 12-bit are
/// little-endian `u16`.
pub fn pack_row<P: Pix>(samples: &[P], out: &mut Vec<u8>) {
    if P::DEPTH == 8 {
        for sample in samples {
            out.push(u8::try_from(sample.get()).unwrap_or(0));
        }
        return;
    }
    for sample in samples {
        let bits = u16::try_from(sample.get().clamp(0, P::MAX)).unwrap_or(0);
        out.extend_from_slice(&bits.to_le_bytes());
    }
}

/// Inverse of [`pack_row`] for a tight plane.
pub fn unpack_plane<P: Pix>(bytes: &[u8], count: usize) -> Result<Vec<P>, ()> {
    if P::DEPTH == 8 {
        if bytes.len() != count {
            return Err(());
        }
        return Ok(bytes
            .iter()
            .copied()
            .map(|byte| P::put(i32::from(byte)))
            .collect());
    }
    if bytes.len() != count.saturating_mul(2) {
        return Err(());
    }
    let mut out = Vec::with_capacity(count);
    for pair in bytes.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes(*pair);
        out.push(P::put(i32::from(bits)));
    }
    Ok(out)
}
