//! The wire: what a server says about a file, and a byte range of it.

/// What a server says about one URL before the bytes are fetched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Head {
    /// The file's length, when the server gives one.
    pub length: Option<u64>,
    /// Whether the server answers a byte range with just those bytes.
    pub ranges: bool,
}

/// Fetches bytes for the downloader. A test double stands in for the
/// network.
pub trait Transport: Send + Sync + 'static {
    /// The file's length and whether ranges work.
    fn head(&self, url: &str) -> Result<Head, String>;

    /// Bytes `range.0..range.1` of the file, or all of it for `None`, handed
    /// to `sink` as they arrive. `sink` returns `false` to stop early.
    fn get(
        &self,
        url: &str,
        range: Option<(u64, u64)>,
        sink: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<(), String>;
}
