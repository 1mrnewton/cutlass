//! Tests that need more than one module: oracle comparisons, real files,
//! malformed input.
//!
//! Oracles are libwebp's `cwebp`, `dwebp` and `webpinfo`, macOS ImageIO
//! through `sips`, ffmpeg's native `webp` decoder, and ImageMagick. Each is
//! looked up on `PATH`. A test whose oracle is missing returns early, unless
//! `WEBP_REQUIRE_ORACLES` is set, in which case it fails. Nothing here runs
//! when the crate is built as a library.

mod alpha;
mod animation;
mod encoder;
mod lossless_encode;
mod lossy;
mod oracle;
pub(crate) use oracle::synthetic_rgba as oracle_synthetic;
mod stills;
