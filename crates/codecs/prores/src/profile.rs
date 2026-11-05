//! The six ProRes profiles and their sample-entry codes.

/// A ProRes profile, named by the four-character code a movie file stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    /// `apco`, 4:2:2 10-bit.
    Proxy,
    /// `apcs`, 4:2:2 10-bit.
    Lt,
    /// `apcn`, 4:2:2 10-bit.
    Standard,
    /// `apch`, 4:2:2 10-bit.
    Hq,
    /// `ap4h`, 4:4:4 12-bit with optional alpha.
    P4444,
    /// `ap4x`, 4:4:4 12-bit with optional alpha.
    Xq,
}

impl Profile {
    pub const ALL: [Self; 6] = [
        Self::Proxy,
        Self::Lt,
        Self::Standard,
        Self::Hq,
        Self::P4444,
        Self::Xq,
    ];

    /// The sample-entry code.
    #[must_use]
    pub const fn fourcc(self) -> [u8; 4] {
        match self {
            Self::Proxy => *b"apco",
            Self::Lt => *b"apcs",
            Self::Standard => *b"apcn",
            Self::Hq => *b"apch",
            Self::P4444 => *b"ap4h",
            Self::Xq => *b"ap4x",
        }
    }

    /// The profile a sample-entry code names. `None` for any other code.
    #[must_use]
    pub fn from_fourcc(code: &[u8]) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|profile| code.get(..4) == Some(&profile.fourcc()[..]))
    }

    /// True for the 4:4:4 profiles.
    #[must_use]
    pub const fn is_444(self) -> bool {
        matches!(self, Self::P4444 | Self::Xq)
    }

    /// Bits per sample a decoder outputs for this profile: 12 for 4:4:4, else 10.
    #[must_use]
    pub const fn depth(self) -> u8 {
        if self.is_444() { 12 } else { 10 }
    }

    /// Human name.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Proxy => "ProRes 422 Proxy",
            Self::Lt => "ProRes 422 LT",
            Self::Standard => "ProRes 422",
            Self::Hq => "ProRes 422 HQ",
            Self::P4444 => "ProRes 4444",
            Self::Xq => "ProRes 4444 XQ",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Profile;

    #[test]
    fn codes_name_their_profiles() {
        for profile in Profile::ALL {
            assert_eq!(Profile::from_fourcc(&profile.fourcc()), Some(profile));
        }
        assert_eq!(Profile::from_fourcc(b"avc1"), None);
        assert_eq!(Profile::from_fourcc(b"ap4"), None);
        assert_eq!(Profile::Hq.depth(), 10);
        assert_eq!(Profile::Xq.depth(), 12);
    }
}
