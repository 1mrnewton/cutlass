//! Colour tags a container can attach to a video stream.
//!
//! The numbers are the code points in Rec. ITU-T H.273.

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
