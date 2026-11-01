//! The preview session.
//!
//! The editor sends a playhead time. Seeking and scrubbing call
//! [`cutlass_eval::frame`] for that time and keep the CPU readback. Playback
//! does not register a container. [`Player`] holds the reader factory `frame`
//! opens and keeps those files open across seeks. The engine maps time and
//! caches decoded pictures.
//!
//! Linux has no output device. Playback stays silent until one exists. That
//! silence is not a missing build, and it is not a failed open. Pictures
//! still present. A missing device is [`AudioGap`], not a panic.
//!
//! WASAPI and AAudio are not in this pass.

#![forbid(unsafe_code)]

mod apple;
mod playhead;
mod present;
mod preview;

pub use apple::{AudioGap, AudioHost, AudioOutput, CopyOutcome};
pub use playhead::{Player, PlayheadError, Readback, Tick};
pub use present::{FrameCache, PresentError, Presentation};
pub use preview::{PLAY_FRAMES, PreviewMix, preview_frames, queue_mix};

#[cfg(test)]
pub(crate) fn color_session(duration_us: i64, segment_us: i64) -> cutlass_session::Session {
    use cutlass_model::{Draft, MaterialColorConfig, Micros, Segment, TimeRange};

    let mut draft = Draft::empty("draft");
    draft.canvas_config.width = 2;
    draft.canvas_config.height = 1;
    draft.duration = Micros(duration_us);
    draft.materials.material_colors.push(MaterialColorConfig {
        id: "color".into(),
        is_color_clip: true,
        solid_color: "0xFF336699".to_owned(),
        ..MaterialColorConfig::default()
    });
    draft.tracks[0].segments.push(Segment::new(
        "segment",
        "color",
        TimeRange::new(0, segment_us),
    ));
    cutlass_session::Session::open(draft)
}
