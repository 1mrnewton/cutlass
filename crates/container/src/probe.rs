//! A family accepts or refuses a header window.
//!
//! The window length is the family's decision. A short slice is not a
//! refusal and not an acceptance. This module does not recognize a file
//! format. The family supplies the verdict for a window that is long enough.

use cutlass_codec::ContainerId;

/// What a family concluded about a header that was long enough to judge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderVerdict {
    /// The bytes are this family, including a brand open may still reject.
    Match,
    /// The bytes are not this family.
    Unrecognized,
    /// The bytes are this family and the brand is unsupported.
    ///
    /// Probe still accepts the family. Open returns [`crate::OpenError::UnsupportedBrand`].
    UnsupportedBrand,
}

/// The result of looking at a fixed number of header bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeResult {
    /// `examined` bytes name this family.
    Accept {
        family: ContainerId,
        examined: usize,
    },
    /// `examined` bytes are not this family.
    Refuse { examined: usize },
    /// The caller has not provided the window this family needs.
    NeedBytes { have: usize, need: usize },
}

impl ProbeResult {
    #[must_use]
    pub const fn examined(&self) -> Option<usize> {
        match self {
            Self::Accept { examined, .. } | Self::Refuse { examined } => Some(*examined),
            Self::NeedBytes { .. } => None,
        }
    }
}

/// Decides from the window length and a verdict the family already made.
///
/// A window shorter than `need` is [`ProbeResult::NeedBytes`] even when
/// `verdict` says the family matches. `need` of zero refuses every input,
/// so an empty requirement cannot swallow another family's file.
#[must_use]
pub fn decide_probe(
    header: &[u8],
    need: usize,
    family: ContainerId,
    verdict: HeaderVerdict,
) -> ProbeResult {
    if need == 0 {
        return ProbeResult::Refuse { examined: 0 };
    }
    if header.len() < need {
        return ProbeResult::NeedBytes {
            have: header.len(),
            need,
        };
    }
    match verdict {
        HeaderVerdict::Match | HeaderVerdict::UnsupportedBrand => ProbeResult::Accept {
            family,
            examined: need,
        },
        HeaderVerdict::Unrecognized => ProbeResult::Refuse { examined: need },
    }
}

/// Header recognition for one container family.
///
/// `matches` is only meaningful once `header.len() >= HEADER_BYTES`.
/// [`Probe::probe`] checks the length first.
pub trait Probe {
    /// How many header bytes this family needs before it will accept or refuse.
    const HEADER_BYTES: usize;

    fn family(&self) -> ContainerId;

    /// Called with at least [`Self::HEADER_BYTES`] bytes.
    fn matches(&self, header: &[u8]) -> HeaderVerdict;

    fn probe(&self, header: &[u8]) -> ProbeResult {
        if needs_more(header, Self::HEADER_BYTES) {
            return decide_probe(
                header,
                Self::HEADER_BYTES,
                self.family(),
                HeaderVerdict::Unrecognized,
            );
        }
        decide_probe(
            header,
            Self::HEADER_BYTES,
            self.family(),
            self.matches(header),
        )
    }
}

fn needs_more(header: &[u8], need: usize) -> bool {
    need == 0 || header.len() < need
}

#[cfg(test)]
mod tests {
    use super::{HeaderVerdict, Probe, ProbeResult, decide_probe};
    use cutlass_codec::ContainerId;

    struct Family {
        id: ContainerId,
        magic: &'static [u8],
    }

    impl Probe for Family {
        const HEADER_BYTES: usize = 4;

        fn family(&self) -> ContainerId {
            self.id
        }

        fn matches(&self, header: &[u8]) -> HeaderVerdict {
            if header.len() >= 4
                && self.magic.len() >= 4
                && header[..3] == self.magic[..3]
                && header[3] == 0xff
            {
                return HeaderVerdict::UnsupportedBrand;
            }
            if header.starts_with(self.magic) {
                HeaderVerdict::Match
            } else {
                HeaderVerdict::Unrecognized
            }
        }
    }

    fn movie() -> Family {
        Family {
            id: ContainerId::Isobmff,
            magic: b"MOV!",
        }
    }

    fn still() -> Family {
        Family {
            id: ContainerId::Still,
            magic: b"STL!",
        }
    }

    #[test]
    fn a_short_window_is_not_an_accept_or_a_refuse() {
        let result = decide_probe(b"MOV", 4, ContainerId::Isobmff, HeaderVerdict::Match);
        assert_eq!(result, ProbeResult::NeedBytes { have: 3, need: 4 });
        assert_eq!(
            movie().probe(b"MO"),
            ProbeResult::NeedBytes { have: 2, need: 4 }
        );
    }

    #[test]
    fn an_empty_requirement_refuses_every_input() {
        let result = decide_probe(b"MOV!", 0, ContainerId::Still, HeaderVerdict::Match);
        assert_eq!(result, ProbeResult::Refuse { examined: 0 });
    }

    #[test]
    fn a_movie_window_does_not_open_as_a_still() {
        assert_eq!(
            movie().probe(b"MOV!....extra"),
            ProbeResult::Accept {
                family: ContainerId::Isobmff,
                examined: 4
            }
        );
        assert_eq!(still().probe(b"MOV!"), ProbeResult::Refuse { examined: 4 });
        assert_eq!(movie().probe(b"STL!"), ProbeResult::Refuse { examined: 4 });
        assert_eq!(
            still().probe(b"STL!"),
            ProbeResult::Accept {
                family: ContainerId::Still,
                examined: 4
            }
        );
    }

    #[test]
    fn an_unsupported_brand_still_names_the_family() {
        assert_eq!(
            movie().probe(b"MOV\xff"),
            ProbeResult::Accept {
                family: ContainerId::Isobmff,
                examined: 4
            }
        );
    }

    #[test]
    fn probe_examines_the_window_and_not_the_rest_of_the_slice() {
        assert_eq!(
            movie().probe(b"MOV! and a lot more bytes").examined(),
            Some(4)
        );
    }
}
