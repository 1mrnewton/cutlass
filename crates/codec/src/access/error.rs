/// Failure from a codec or from a container framing that codec.
///
/// `at_bit` is the bit offset in the buffer being parsed, when the parser
/// has one. It is not a copy of the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodecError {
    /// The byte stream ended inside a frame that the caller has not finished.
    NeedMoreData,
    /// The caller presented a complete unit that ran out of bits early.
    Truncated { at_bit: u64, what: &'static str },
    /// The bits were present and do not form a legal value.
    Malformed { at_bit: u64, what: &'static str },
    /// The configuration names a tool or profile this build does not run.
    Unsupported { what: &'static str },
    /// A length, a search, or a buffer would pass a documented limit.
    Limit { what: &'static str },
    /// The call does not match the codec's state machine.
    InvalidApi { what: &'static str },
    /// PCM contained NaN or an infinity.
    NonFinite,
    /// A finite PCM sample was outside the format's full scale.
    OutOfRange,
    /// Channel planes in one push did not share a length.
    UnevenChannels,
    /// The caller set the cancel flag before this frame finished.
    Cancelled,
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NeedMoreData => write!(f, "need more data"),
            Self::Truncated { at_bit, what } => {
                write!(f, "truncated {what} at bit {at_bit}")
            }
            Self::Malformed { at_bit, what } => {
                write!(f, "malformed {what} at bit {at_bit}")
            }
            Self::Unsupported { what } => write!(f, "unsupported {what}"),
            Self::Limit { what } => write!(f, "limit exceeded: {what}"),
            Self::InvalidApi { what } => write!(f, "invalid call: {what}"),
            Self::NonFinite => write!(f, "non-finite sample"),
            Self::OutOfRange => write!(f, "sample out of range"),
            Self::UnevenChannels => write!(f, "uneven channel lengths"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for CodecError {}
