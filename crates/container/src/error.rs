use std::fmt;

/// A failed probe, open, read, seek, or write.
///
/// Truncation, a size that runs past its parent, an unsupported brand, and a
/// seek that has no index are different variants. Library paths return these
/// values and do not panic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    NotThisFamily,
    Truncated { at: u64, needed: u64 },
    HostileSize { at: u64, claimed: u64 },
    Nesting { depth: u32 },
    Limit { what: &'static str, value: u64 },
    UnsupportedBrand { major: [u8; 4] },
    Unsupported { what: &'static str },
    Missing { what: &'static str },
    SampleExtendsPastMedia { offset: u64, size: u32 },
    SeekLimited { reason: &'static str },
    ExternalReference,
    CodecRejected { sample_entry: [u8; 4] },
    Incomplete,
    Io { at: u64, kind: std::io::ErrorKind },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotThisFamily => write!(f, "not this container family"),
            Self::Truncated { at, needed } => {
                write!(f, "truncated at {at}, needed {needed}")
            }
            Self::HostileSize { at, claimed } => {
                write!(f, "size {claimed} at {at} exceeds its parent")
            }
            Self::Nesting { depth } => write!(f, "box nesting {depth} exceeds the limit"),
            Self::Limit { what, value } => write!(f, "{what} limit exceeded ({value})"),
            Self::UnsupportedBrand { major } => {
                write!(f, "unsupported brand {}", fourcc(major))
            }
            Self::Unsupported { what } => write!(f, "unsupported {what}"),
            Self::Missing { what } => write!(f, "missing {what}"),
            Self::SampleExtendsPastMedia { offset, size } => {
                write!(f, "sample at {offset} size {size} extends past media data")
            }
            Self::SeekLimited { reason } => write!(f, "seek limited: {reason}"),
            Self::ExternalReference => write!(f, "media data is in another file"),
            Self::CodecRejected { sample_entry } => {
                write!(f, "codec {} cannot be carried", fourcc(sample_entry))
            }
            Self::Incomplete => write!(f, "the file index was not finished"),
            Self::Io { at, kind } => write!(f, "io error at {at}: {kind:?}"),
        }
    }
}

impl std::error::Error for Error {}

fn fourcc(code: &[u8; 4]) -> String {
    code.iter()
        .map(|byte| {
            if byte.is_ascii_graphic() || *byte == b' ' {
                char::from(*byte).to_string()
            } else {
                format!("\\x{byte:02x}")
            }
        })
        .collect()
}
