//! The digest a finished file has to match.
//!
//! A preview may be published with no digest until the host names one for
//! that URL. A package is published only when these bytes match.

use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// SHA-256 of a file's bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Checksum([u8; 32]);

impl Checksum {
    /// The digest of `bytes`.
    pub fn of(bytes: &[u8]) -> Self {
        Self::from_hasher(Sha256::new().chain_update(bytes))
    }

    /// Lowercase hex, 64 characters.
    #[must_use]
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut text = String::with_capacity(64);
        for byte in self.0 {
            text.push(char::from(HEX[usize::from(byte >> 4)]));
            text.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        text
    }

    /// Parse lowercase or uppercase hex.
    pub fn parse(text: &str) -> Result<Self, ChecksumError> {
        let bytes = text.as_bytes();
        if bytes.len() != 64 {
            return Err(ChecksumError::Length { found: bytes.len() });
        }
        let mut out = [0u8; 32];
        for (index, slot) in out.iter_mut().enumerate() {
            let high = hex_digit(bytes[index * 2])?;
            let low = hex_digit(bytes[index * 2 + 1])?;
            *slot = (high << 4) | low;
        }
        Ok(Self(out))
    }

    fn from_hasher(hasher: Sha256) -> Self {
        let digest = hasher.finalize();
        let mut bytes = [0u8; 32];
        for (slot, byte) in bytes.iter_mut().zip(digest.iter()) {
            *slot = *byte;
        }
        Self(bytes)
    }
}

impl fmt::Debug for Checksum {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

impl fmt::Display for Checksum {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

impl Serialize for Checksum {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Checksum {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

/// Why a digest could not be read or parsed.
#[derive(Debug)]
pub enum ChecksumError {
    /// The hex text is not 64 characters.
    Length { found: usize },
    /// A character is not hexadecimal.
    Hex,
    /// The path is a directory or a symlink, so it is not the file's bytes.
    NotAFile { path: PathBuf },
    /// The filesystem refused the read.
    Io { path: PathBuf, message: String },
}

impl fmt::Display for ChecksumError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length { found } => {
                write!(formatter, "a checksum is 64 hex characters, found {found}")
            }
            Self::Hex => formatter.write_str("a checksum has a character that is not hex"),
            Self::NotAFile { path } => {
                write!(formatter, "{} is not a file", path.display())
            }
            Self::Io { path, message } => {
                write!(formatter, "could not read {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for ChecksumError {}

/// Hash `path` in chunks. A directory or a symlink is refused.
pub fn hash_file(path: &Path) -> Result<(Checksum, u64), ChecksumError> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| ChecksumError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    if !metadata.file_type().is_file() {
        return Err(ChecksumError::NotAFile {
            path: path.to_path_buf(),
        });
    }
    let mut file = File::open(path).map_err(|error| ChecksumError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut size = 0u64;
    loop {
        let read = file.read(&mut buffer).map_err(|error| ChecksumError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        let Ok(count) = u64::try_from(read) else {
            return Err(ChecksumError::Io {
                path: path.to_path_buf(),
                message: "a read was larger than a file size".to_owned(),
            });
        };
        size = size.saturating_add(count);
    }
    Ok((Checksum::from_hasher(hasher), size))
}

fn hex_digit(byte: u8) -> Result<u8, ChecksumError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(ChecksumError::Hex),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::{Checksum, ChecksumError, hash_file};

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("cutlass-download-{}-{n}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn the_empty_file_and_abc_match_sha256() {
        let empty = Checksum::of(b"");
        assert_eq!(
            empty.to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let abc = Checksum::of(b"abc");
        assert_eq!(
            abc.to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(Checksum::parse(&empty.to_hex()).unwrap(), empty);
        assert_eq!(
            Checksum::parse(&abc.to_hex().to_ascii_uppercase()).unwrap(),
            abc
        );
    }

    #[test]
    fn a_checksum_that_is_not_hex_is_refused() {
        let error = Checksum::parse("").unwrap_err();
        assert!(matches!(error, ChecksumError::Length { found: 0 }));
        let error = Checksum::parse(&"ab".repeat(31)).unwrap_err();
        assert!(matches!(error, ChecksumError::Length { found: 62 }));
        let error = Checksum::parse(&"zz".repeat(32)).unwrap_err();
        assert!(matches!(error, ChecksumError::Hex));
        let error = Checksum::parse(&format!("gg{}", "ab".repeat(31))).unwrap_err();
        assert!(matches!(error, ChecksumError::Hex));
    }

    #[test]
    fn a_checksum_round_trips_as_hex_text() {
        let checksum = Checksum::of(b"preview");
        let json = serde_json::to_value(checksum).unwrap();
        assert_eq!(json, serde_json::Value::String(checksum.to_hex()));
        let back: Checksum = serde_json::from_value(json).unwrap();
        assert_eq!(back, checksum);
        let error = serde_json::from_str::<Checksum>("12").unwrap_err();
        assert!(!error.to_string().is_empty());
        let error = serde_json::from_str::<Checksum>("\"\"").unwrap_err();
        assert!(!error.to_string().is_empty());
    }

    #[test]
    fn a_file_hashes_to_the_same_digest_as_its_bytes() {
        let temp = Temp::new();
        let path = temp.path.join("clip.mp4");
        fs::write(&path, b"abc").unwrap();
        let (checksum, size) = hash_file(&path).unwrap();
        assert_eq!(checksum, Checksum::of(b"abc"));
        assert_eq!(size, 3);
        let missing = hash_file(&temp.path.join("gone")).unwrap_err();
        assert!(matches!(missing, ChecksumError::Io { .. }));
        let directory = hash_file(&temp.path).unwrap_err();
        assert!(matches!(directory, ChecksumError::NotAFile { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_is_not_hashed() {
        let temp = Temp::new();
        let target = temp.path.join("bytes");
        fs::write(&target, b"abc").unwrap();
        let link = temp.path.join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let error = hash_file(&link).unwrap_err();
        assert!(matches!(error, ChecksumError::NotAFile { .. }));
        assert_eq!(fs::read(&target).unwrap(), b"abc");
    }
}
