//! Failures from reading a PNG file.

/// Why a file could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The signature is not the PNG signature.
    #[error("the file is not PNG")]
    Signature,
    /// The file ended before the chunk did.
    #[error("file ended at byte {at} while reading {need} bytes")]
    Truncated { at: u64, need: u64 },
    /// The CRC of a chunk does not match its type and data.
    #[error("the {name} chunk checksum does not match")]
    Checksum { name: ChunkKind },
    /// The image data is not a zlib stream of the scanlines.
    #[error("the image data is not valid zlib")]
    Inflate,
    /// Width, height, or the decoded buffer is past the decoder limit.
    #[error("the picture is larger than the decoder allows")]
    TooLarge,
    /// A field uses a value this decoder does not read.
    #[error("{0}")]
    Unsupported(&'static str),
    /// The file breaks a rule this decoder enforces.
    #[error("{0}")]
    Inconsistent(&'static str),
    /// The chunk is critical and this decoder does not know it.
    #[error("the critical chunk {name} is not supported")]
    Critical { name: ChunkKind },
}

/// A four-byte PNG chunk type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkKind(pub [u8; 4]);

impl std::fmt::Display for ChunkKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            if byte.is_ascii_graphic() {
                write!(f, "{}", char::from(byte))?;
            } else {
                write!(f, "\\x{byte:02x}")?;
            }
        }
        Ok(())
    }
}
