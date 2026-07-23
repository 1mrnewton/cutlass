//! Frame grab and video export on the engine host thread.
//!
//! Agents use [`do_get_frame`] to visually verify edits (PNG) and
//! [`do_export_video`] to mux the whole timeline. Both run synchronously on
//! the engine thread — accepted v1 behavior for a single agent session.

use std::path::PathBuf;

use cutlass_commands::{Command, ProjectCommand};
use cutlass_engine::{ApplyOutcome, Engine};
use cutlass_models::{Rational, RationalTime};
use cutlass_render::encode_png;
use serde::Serialize;

use super::{eng_err, require_engine_mut};

/// PNG frame grab for agent visual verification.
#[derive(Debug, Serialize)]
pub struct FrameGrab {
    /// Encoded PNG bytes (not base64 — the tool layer encodes for MCP).
    #[serde(skip)]
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Frame-snapped timeline time in seconds (nearest project frame to the
    /// request — matches the pixels, not the raw input).
    pub seconds: f64,
}

/// Successful whole-timeline export.
#[derive(Debug, Serialize)]
pub struct ExportDone {
    pub path: String,
    pub frames: u64,
}

pub(super) fn do_get_frame(
    slot: &mut Option<Engine>,
    seconds: f64,
    max_dim: u32,
) -> Result<FrameGrab, String> {
    let engine = require_engine_mut(slot)?;
    if !seconds.is_finite() {
        return Err(format!("time must be a finite number (got {seconds})"));
    }
    if seconds < 0.0 {
        return Err(format!("time must not be negative (got {seconds}s)"));
    }
    let rate = engine.project().timeline().frame_rate;
    let time = seconds_to_timeline_time(seconds, rate)?;
    let image = engine
        .get_frame_fit(time, max_dim, max_dim)
        .map_err(eng_err)?;
    let width = image.width;
    let height = image.height;
    let png = encode_png(&image).map_err(eng_err)?;
    // Report the snapped frame time so the caption matches the pixels.
    let snapped = time.value as f64 * f64::from(rate.den) / f64::from(rate.num);
    Ok(FrameGrab {
        png,
        width,
        height,
        seconds: snapped,
    })
}

pub(super) fn do_export_video(
    slot: &mut Option<Engine>,
    path: PathBuf,
) -> Result<ExportDone, String> {
    let engine = require_engine_mut(slot)?;
    // Exporter empty-timeline errors are opaque; reject early with a clear tip.
    if engine.project().timeline().duration().value <= 0 {
        return Err("timeline is empty — add clips before export_video (nothing to render)".into());
    }
    let path_str = path.display().to_string();
    match engine
        .apply(Command::Project(ProjectCommand::Export { path }))
        .map_err(eng_err)?
    {
        ApplyOutcome::Exported { frames } => Ok(ExportDone {
            path: path_str,
            frames,
        }),
        other => Err(format!("unexpected export outcome: {other:?}")),
    }
}

/// Frame-snap seconds to the project rate — same formula as
/// `cutlass_ai::validate::time::seconds_to_ticks` / `timeline_time`
/// (nearest-frame `round`, finite already checked by the caller).
///
/// Also mirrors the ±2^53 ticks bound so a huge finite `seconds` gets a
/// readable rejection instead of saturating on `as i64`.
fn seconds_to_timeline_time(seconds: f64, rate: Rational) -> Result<RationalTime, String> {
    let ticks = seconds * f64::from(rate.num) / f64::from(rate.den);
    if !(-(2f64.powi(53))..=2f64.powi(53)).contains(&ticks) {
        return Err(format!("time of {seconds}s is out of range"));
    }
    Ok(RationalTime::new(ticks.round() as i64, rate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::Rational;

    #[test]
    fn seconds_to_timeline_time_rejects_out_of_range() {
        let err =
            seconds_to_timeline_time(1e18, Rational::FPS_30).expect_err("huge finite seconds");
        assert!(
            err.contains("out of range"),
            "expected readable rejection, got: {err}"
        );
    }
}
