//! The tracking job: follow a box through a video clip's frames and hand
//! back its path on the canvas, for `Request::Track` to write.
//!
//! The box is drawn on one frame, in the clip's own frame (-1 to 1, y up,
//! as masks read it). Frames are decoded at the project's frame rate from
//! there to the clip's end, then back to its start. Going back, a stretch
//! of frames is decoded forward and kept as small grey copies, then
//! stepped through in reverse, so the decoder never seeks once a frame.
//! Each quad is placed on the canvas through the clip's transform at that
//! time, keys included. Nothing here writes the draft.

use std::path::Path;

use cutlass_engine::{ClockKind, DisplayQuarter, ReaderFactory, decoded_picture, open_segment_in};
use cutlass_model::{Draft, Micros, Segment, SegmentId, TrackMode, TrackPath, TrackSample};
use cutlass_motion::{Gray, Mode, MotionError, Quad, Step, Tracker};
use cutlass_picture::Picture;

/// Frames decoded forward at a time when tracking back.
const BACK_STRETCH: usize = 32;

/// Why a track has no path.
#[derive(Clone, Debug, PartialEq)]
pub enum TrackError {
    /// The clip is not a video clip on the timeline, or the time is not on it.
    Clip { reason: String },
    /// A frame could not be opened or decoded.
    Frame { reason: String },
    /// The tracker refused the box or a frame.
    Motion(MotionError),
    /// The callback asked to stop.
    Stopped,
}

impl std::fmt::Display for TrackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Clip { reason } | Self::Frame { reason } => f.write_str(reason),
            Self::Motion(error) => write!(f, "{error}"),
            Self::Stopped => f.write_str("tracking was stopped"),
        }
    }
}

impl std::error::Error for TrackError {}

/// Which way from the drawn frame to track.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrackSpan {
    /// To the clip's end and back to its start.
    #[default]
    Both,
    /// Only to the clip's end.
    After,
    /// Only back to the clip's start.
    Before,
}

/// What to track: a box in a clip's own frame at one timeline time.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackAsk {
    pub segment_id: SegmentId,
    /// The timeline time the box was drawn at.
    pub at: Micros,
    /// The box's corners in the clip's own frame: -1 to 1, y up.
    pub corners: [(f64, f64); 4],
    pub mode: TrackMode,
    pub span: TrackSpan,
}

/// Follow `ask`'s box through its clip. `progress` hears the share of the
/// clip's frames read so far after each one, and stops the job by
/// answering `false`. Frames where the target is hidden have no sample;
/// the path ends where the target was lost, on each side.
pub fn track_clip(
    readers: &ReaderFactory,
    draft: &Draft,
    root: Option<&Path>,
    ask: &TrackAsk,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Result<TrackPath, TrackError> {
    let TrackAsk {
        segment_id,
        at,
        corners,
        mode,
        span,
    } = ask;
    let (at, corners, mode) = (*at, *corners, *mode);
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)
        .ok_or_else(|| TrackError::Clip {
            reason: "the clip is not on the timeline".into(),
        })?;
    let source = open_segment_in(readers, draft, segment_id, ClockKind::Export, 1, root).map_err(
        |error| TrackError::Frame {
            reason: format!("the clip's file could not be opened: {error}"),
        },
    )?;
    let display = source.info().display;
    let mut frame_at = |time: Micros| -> Result<Gray, TrackError> {
        let unreadable = |what: &str| TrackError::Frame {
            reason: format!("the clip's frame at {} µs {what}", time.0),
        };
        let decoded = source
            .decoded_at(time)
            .map_err(|_| unreadable("could not be decoded"))?;
        let Ok(Picture::Cpu(picture)) = decoded_picture(&decoded) else {
            return Err(unreadable("is not in memory"));
        };
        let gray = Gray::from_rgba(picture.width(), picture.height(), picture.rgba())
            .map_err(TrackError::Motion)?;
        upright(gray, display)
    };
    let canvas = (
        u32::try_from(draft.canvas_config.width.max(1)).unwrap_or(1),
        u32::try_from(draft.canvas_config.height.max(1)).unwrap_or(1),
    );
    let place = |time: Micros, quad: Quad| on_canvas(draft, segment, time, canvas, quad);
    track_frames(
        &frame_times(draft, segment, at)?.only(*span),
        &mut frame_at,
        &place,
        corners,
        mode,
        progress,
    )
}

/// The clip's frame times at the project's frame rate, through `at`:
/// those before it, `at` itself, and those after it.
pub(crate) struct FrameTimes {
    pub(crate) before: Vec<Micros>,
    pub(crate) at: Micros,
    pub(crate) after: Vec<Micros>,
}

impl FrameTimes {
    /// These times on `span`'s side of `at`.
    fn only(mut self, span: TrackSpan) -> Self {
        match span {
            TrackSpan::Both => {}
            TrackSpan::After => self.before.clear(),
            TrackSpan::Before => self.after.clear(),
        }
        self
    }
}

fn frame_times(draft: &Draft, segment: &Segment, at: Micros) -> Result<FrameTimes, TrackError> {
    let (start, end) = (segment.target.start.0, segment.target.end().0);
    if !(start..end).contains(&at.0) {
        return Err(TrackError::Clip {
            reason: "move the playhead onto the clip".into(),
        });
    }
    let fps = if draft.fps.is_finite() && (1.0..=240.0).contains(&draft.fps) {
        draft.fps
    } else {
        30.0
    };
    let step = 1_000_000.0 / fps;
    let time = |k: i64| Micros(at.0 + (k as f64 * step).round() as i64);
    let after = (1..).map(time).take_while(|t| t.0 < end).collect();
    let before = (1..)
        .map(|k| time(-k))
        .take_while(|t| t.0 >= start)
        .collect();
    Ok(FrameTimes { before, at, after })
}

/// Track through `times`, reading each frame with `frame_at` and placing
/// each quad with `place`.
pub(crate) fn track_frames(
    times: &FrameTimes,
    frame_at: &mut dyn FnMut(Micros) -> Result<Gray, TrackError>,
    place: &dyn Fn(Micros, Quad) -> Option<[(f64, f64); 4]>,
    corners: [(f64, f64); 4],
    mode: TrackMode,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Result<TrackPath, TrackError> {
    let total = (times.before.len() + times.after.len() + 1) as f32;
    let mut read = 1_usize;
    let mut tell = |read: usize| progress(read as f32 / total);
    let unplaced = || TrackError::Clip {
        reason: "the clip is not drawn on the canvas there".into(),
    };
    let first = frame_at(times.at)?;
    let box_quad = Quad(corners.map(|(u, v)| ((u + 1.0) / 2.0, (1.0 - v) / 2.0)));
    let motion_mode = match mode {
        TrackMode::Object => Mode::Object,
        TrackMode::Plane => Mode::Plane,
    };
    let mut forward = Tracker::start(&first, box_quad, motion_mode).map_err(TrackError::Motion)?;
    let mut backward = Tracker::start(&first, box_quad, motion_mode).map_err(TrackError::Motion)?;
    let start = place(times.at, box_quad).ok_or_else(unplaced)?;
    if !tell(read) {
        return Err(TrackError::Stopped);
    }

    let mut after = Vec::new();
    for &time in &times.after {
        let frame = frame_at(time)?;
        read += 1;
        match forward.step(&frame).map_err(TrackError::Motion)? {
            Step::Tracked { quad, .. } => after.push(TrackSample {
                time,
                corners: place(time, quad).ok_or_else(unplaced)?,
            }),
            Step::Hidden => {}
            Step::Lost => break,
        }
        if !tell(read) {
            return Err(TrackError::Stopped);
        }
    }

    // Back from `at`: decode each stretch forward, then step it in reverse.
    let mut before = Vec::new();
    'back: for stretch in times.before.chunks(BACK_STRETCH) {
        let mut ascending: Vec<Micros> = stretch.to_vec();
        ascending.reverse();
        let mut frames = Vec::with_capacity(ascending.len());
        for &time in &ascending {
            frames.push((time, frame_at(time)?));
        }
        for (time, frame) in frames.into_iter().rev() {
            read += 1;
            match backward.step(&frame).map_err(TrackError::Motion)? {
                Step::Tracked { quad, .. } => before.push(TrackSample {
                    time,
                    corners: place(time, quad).ok_or_else(unplaced)?,
                }),
                Step::Hidden => {}
                Step::Lost => break 'back,
            }
            if !tell(read) {
                return Err(TrackError::Stopped);
            }
        }
    }
    before.reverse();
    before.push(TrackSample {
        time: times.at,
        corners: start,
    });
    before.extend(after);
    Ok(TrackPath {
        mode,
        samples: before,
    })
}

/// `path`, in canvas units (1 is half the canvas, y up), in clip
/// `segment`'s own frame at each sample's time: -1 to 1 across and up, the
/// frame its mask is drawn in. `None` when the clip is not on the canvas
/// at some sample.
#[must_use]
pub fn path_in_clip(draft: &Draft, segment: &Segment, path: &TrackPath) -> Option<TrackPath> {
    let canvas = (
        u32::try_from(draft.canvas_config.width).ok()?,
        u32::try_from(draft.canvas_config.height).ok()?,
    );
    let (hw, hh) = (f64::from(canvas.0) / 2.0, f64::from(canvas.1) / 2.0);
    let mut samples = Vec::with_capacity(path.samples.len());
    for sample in &path.samples {
        let [a, b, c, d, e, f] = crate::media::clip_frame(draft, segment, sample.time, canvas)?.0;
        let corners = sample.corners.map(|(x, y)| {
            let (px, py) = (hw + x * hw, hh - y * hh);
            (a * px + b * py + c, d * px + e * py + f)
        });
        if !corners.iter().all(|p| p.0.is_finite() && p.1.is_finite()) {
            return None;
        }
        samples.push(TrackSample {
            time: sample.time,
            corners,
        });
    }
    Some(TrackPath {
        mode: path.mode,
        samples,
    })
}

/// `quad`, in unit coordinates of the clip's upright picture, on the
/// canvas at `time` in transform units (1 is half the canvas, y up).
fn on_canvas(
    draft: &Draft,
    segment: &Segment,
    time: Micros,
    canvas: (u32, u32),
    quad: Quad,
) -> Option<[(f64, f64); 4]> {
    let [a, b, c, d, e, f] = crate::media::clip_frame(draft, segment, time, canvas)?.0;
    // Invert the canvas-pixel-to-clip map.
    let det = a * e - b * d;
    if det.abs() < 1e-12 || !det.is_finite() {
        return None;
    }
    let (hw, hh) = (f64::from(canvas.0) / 2.0, f64::from(canvas.1) / 2.0);
    let mut out = [(0.0, 0.0); 4];
    for (corner, &(x, y)) in out.iter_mut().zip(&quad.0) {
        let (u, v) = (2.0 * x - 1.0, 1.0 - 2.0 * y);
        let (du, dv) = (u - c, v - f);
        let px = (e * du - b * dv) / det;
        let py = (a * dv - d * du) / det;
        *corner = ((px - hw) / hw, (hh - py) / hh);
    }
    out.iter()
        .all(|p| p.0.is_finite() && p.1.is_finite())
        .then_some(out)
}

/// `gray` turned by the file's display quarter, so it is upright.
fn upright(gray: Gray, display: DisplayQuarter) -> Result<Gray, TrackError> {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    if display == DisplayQuarter::Zero {
        return Ok(gray);
    }
    let (out_w, out_h) = match display {
        DisplayQuarter::Clockwise90 | DisplayQuarter::Clockwise270 => (h, w),
        DisplayQuarter::Zero | DisplayQuarter::Half => (w, h),
    };
    let values = gray.data();
    let mut data = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let (ox, oy) = match display {
                DisplayQuarter::Zero => (x, y),
                DisplayQuarter::Clockwise90 => (h - 1 - y, x),
                DisplayQuarter::Half => (w - 1 - x, h - 1 - y),
                DisplayQuarter::Clockwise270 => (y, w - 1 - x),
            };
            data[oy * out_w + ox] = values[y * w + x];
        }
    }
    Gray::new(out_w as u32, out_h as u32, data).map_err(TrackError::Motion)
}

#[cfg(test)]
mod tests;
