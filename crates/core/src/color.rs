//! Colour a container can attach to a video stream.
//!
//! [`ColorTags`] are the code points in Rec. ITU-T H.273. [`MasteringDisplay`]
//! and [`ContentLight`] are the integers from the `mdcv` and `clli` boxes.

/// Primaries, transfer, and matrix for one video stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorTags {
    pub primaries: u16,
    pub transfer: u16,
    pub matrix: u16,
    /// `Some` when the container states the range. `nclc` leaves it out.
    pub full_range: Option<bool>,
}

impl ColorTags {
    /// PQ is 16 and HLG is 18 in H.273.
    pub const fn is_hdr(self) -> bool {
        matches!(self.transfer, 16 | 18)
    }
}

/// Chromaticity and luminance from an `mdcv` box.
///
/// `primaries_x` and `primaries_y` are green, then blue, then red, in units
/// of 0.00002. The white point uses the same units. Luminance is in units of
/// 0.0001 cd/m². The integers are what the container stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MasteringDisplay {
    pub primaries_x: [u16; 3],
    pub primaries_y: [u16; 3],
    pub white_x: u16,
    pub white_y: u16,
    pub max_luminance: u32,
    pub min_luminance: u32,
}

/// Content light level from a `clli` box.
///
/// Both numbers are cd/m², as the container stored them. Zero stays zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentLight {
    /// MaxCLL.
    pub max_content_light_level: u16,
    /// MaxFALL.
    pub max_pic_average_light_level: u16,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq_and_hlg_are_hdr() {
        let tags = ColorTags {
            primaries: 9,
            transfer: 16,
            matrix: 9,
            full_range: Some(false),
        };
        assert!(tags.is_hdr());
        assert!(
            ColorTags {
                transfer: 18,
                ..tags
            }
            .is_hdr()
        );
        assert!(
            !ColorTags {
                transfer: 1,
                ..tags
            }
            .is_hdr()
        );
    }
}
