//! High dynamic range metadata a video stream carries beside its samples.
//!
//! The values are stored as the stream codes them. Nothing here converts a
//! picture.

/// Content light level (CTA-861.3): the brightest sample and the brightest
/// frame average of the content, in candelas per square metre.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ContentLight {
    /// Maximum content light level.
    pub max_cll: u16,
    /// Maximum frame-average light level.
    pub max_fall: u16,
}

/// Mastering display colour volume (SMPTE ST 2086), in the fixed-point
/// units of an AV1 metadata OBU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct MasteringDisplay {
    /// CIE 1931 x and y of red, green and blue, each 0.16 fixed point.
    pub primaries: [[u16; 2]; 3],
    /// CIE 1931 x and y of the white point, each 0.16 fixed point.
    pub white_point: [u16; 2],
    /// Maximum luminance in candelas per square metre, 24.8 fixed point.
    pub max_luminance: u32,
    /// Minimum luminance in candelas per square metre, 18.14 fixed point.
    pub min_luminance: u32,
}

impl MasteringDisplay {
    /// A chromaticity coordinate as a fraction of one.
    #[must_use]
    pub fn chromaticity(coordinate: u16) -> f64 {
        f64::from(coordinate) / 65536.0
    }

    /// Maximum luminance in candelas per square metre.
    #[must_use]
    pub fn max_nits(&self) -> f64 {
        f64::from(self.max_luminance) / 256.0
    }

    /// Minimum luminance in candelas per square metre.
    #[must_use]
    pub fn min_nits(&self) -> f64 {
        f64::from(self.min_luminance) / 16384.0
    }
}

/// One ITU-T T.35 message registered to a country and, past that, to a
/// provider: HDR10+ dynamic metadata is one. The payload is carried as it
/// was coded and is not interpreted here.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct ItuT35 {
    pub country_code: u8,
    /// Present when `country_code` is `0xff`.
    pub country_code_extension: Option<u8>,
    pub payload: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::MasteringDisplay;

    #[test]
    fn the_fixed_point_values_read_as_their_units() {
        let display = MasteringDisplay {
            primaries: [[0; 2]; 3],
            white_point: [0x5000, 0x5443],
            max_luminance: 1000 << 8,
            min_luminance: 82,
        };
        assert!((display.max_nits() - 1000.0).abs() < 1e-9);
        assert!((display.min_nits() - 0.005).abs() < 1e-4);
        // D65 is x 0.3127, y 0.3290.
        assert!((MasteringDisplay::chromaticity(display.white_point[0]) - 0.3125).abs() < 1e-9);
        assert!((MasteringDisplay::chromaticity(display.white_point[1]) - 0.3291).abs() < 1e-4);
    }
}
