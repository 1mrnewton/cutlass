//! Export a timeline range.
//!
//! The picture is `eval::frame`. The samples are `eval::audio_slice` on the
//! export clock. Packets are encoded through the encode factory and written
//! through the container mux trait. This crate does not decode, mix, or
//! composite on its own.
//!
//! Styled caption motion is not written to a sidecar. ASS is refused. Eval
//! does not draw text yet, so a burned caption glyph is not in the frame.

#![forbid(unsafe_code)]

mod config;
mod run;
mod sidecars;

pub use config::{
    AudioExport, ExportError, ExportRequest, GIF_LONG_SIDES, MAX_FRAMES, SidecarKind,
    SidecarRequest, VIDEO_LONG_SIDES, output_size, validate,
};
pub use run::{
    ExportResult, ExportRun, ExportedFrame, export, frame_indexes, same_software_pixels,
};
pub use sidecars::{sidecar_text, write_sidecar};
