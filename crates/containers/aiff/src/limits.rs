//! Bounds for one header. They are not a cap on the sound data.

/// Bytes `probe` needs before it will accept or refuse.
pub(crate) const PROBE_BYTES: usize = 12;

/// Chunks in one FORM, including `COMM` and `SSND`.
pub(crate) const MAX_CHUNKS: u32 = 4096;

/// Channels stored in one frame.
pub(crate) const MAX_CHANNELS: u16 = 64;

/// Largest `COMM` payload this crate will read.
pub(crate) const MAX_COMM: u32 = 4096;
