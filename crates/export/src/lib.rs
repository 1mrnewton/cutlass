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
mod movie;
mod pictures;
mod prerender;
mod proxy;
mod run;
mod sidecars;
mod sound;
mod still;

pub use config::{
    AudioExport, ExportError, ExportRequest, GIF_SHORT_SIDES, MAX_EDGE, MAX_FRAMES, ProresProfile,
    SidecarKind, SidecarRequest, VIDEO_SHORT_SIDES, coded_size, output_size, validate,
};
pub use movie::{
    MOVIE_SOUND, MovieSources, part_path, video_bitrate, write_animation, write_movie, write_sound,
};
pub use pictures::{PictureMovie, write_pictures};
pub use prerender::{
    PREVIEW_SIDE, read_render_name, render_key, render_name, render_request, render_side, stand_in,
};
pub use proxy::{
    DEFAULT_PROXY_SIDE, PROXY_SIDES, proxy_bitrate, proxy_draft, proxy_name, proxy_request,
    proxy_side, wants_proxy,
};
pub use run::{
    ExportResult, ExportRun, ExportedFrame, export, export_sound, frame_indexes,
    same_software_pixels,
};
pub use sidecars::{sidecar_text, write_sidecar};
pub use sound::file_audio;
pub use still::{StillFormat, still_picture, write_still};
