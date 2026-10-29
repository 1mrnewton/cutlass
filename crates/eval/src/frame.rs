use std::fmt;

use cutlass_compositor::{Device, Extent, Frame, submit};
use cutlass_generated::PictureSource;
use cutlass_model::{Draft, Micros};

use crate::sources::solid_sources;

/// A draft borrowed for evaluation. The draft is not written.
#[derive(Clone, Copy)]
pub struct Snapshot<'a> {
    draft: &'a Draft,
}

impl<'a> Snapshot<'a> {
    #[must_use]
    pub fn new(draft: &'a Draft) -> Self {
        Self { draft }
    }

    #[must_use]
    pub fn draft(self) -> &'a Draft {
        self.draft
    }
}

/// Why a snapshot did not produce a frame.
#[derive(Debug, PartialEq, Eq)]
pub enum EvalError<E> {
    /// The canvas width or height is not a framebuffer size.
    Canvas,
    /// The compositor or the device refused the frame.
    Compose(cutlass_compositor::ComposeError<E>),
    /// The export clock did not accept the picture.
    Wait,
}

impl<E: fmt::Display> fmt::Display for EvalError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Canvas => formatter.write_str("the canvas size is not a frame"),
            Self::Compose(error) => write!(formatter, "{error}"),
            Self::Wait => formatter.write_str("export did not accept the frame"),
        }
    }
}

impl<E: std::error::Error> std::error::Error for EvalError<E> {}

fn canvas_extent(draft: &Draft) -> Option<Extent> {
    let width = u32::try_from(draft.canvas_config.width).ok()?;
    let height = u32::try_from(draft.canvas_config.height).ok()?;
    let extent = Extent { width, height };
    extent.rgba_len()?;
    Some(extent)
}

/// The composited frame at `time`. The snapshot's draft is not changed.
pub fn frame<D: Device>(
    snapshot: &Snapshot<'_>,
    time: Micros,
    device: &mut D,
) -> Result<Frame, EvalError<D::Error>> {
    let draft = snapshot.draft();
    let extent = canvas_extent(draft).ok_or(EvalError::Canvas)?;
    let solids = solid_sources(draft, extent.width, extent.height);
    let mut sources: Vec<&dyn PictureSource> = Vec::new();
    for solid in &solids {
        sources.push(solid);
    }
    submit(device, extent, time, &sources).map_err(EvalError::Compose)
}

/// Keeps `frame` for export. A late arrival is still the picture.
///
/// This calls the engine export clock. Playback's late drop is a different clock.
pub fn wait_export_frame<E>(
    requested: Micros,
    ready_at: Micros,
    frame: Frame,
) -> Result<Frame, EvalError<E>> {
    let clock = cutlass_engine::MediaClock::new(cutlass_engine::ClockKind::Export);
    let extent = frame.extent;
    let rgba = clock
        .export_rgba(requested, ready_at, extent.width, extent.height, frame.rgba)
        .map_err(|_| EvalError::Wait)?;
    Ok(Frame { extent, rgba })
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialColorConfig, Micros, Segment, TimeRange, TrackAttribute};
    use cutlass_renderer_software::SoftwareDevice;

    use super::*;

    fn color_clip(color: &str, gradient: bool) -> Draft {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "color".into(),
            is_color_clip: true,
            is_gradient: gradient,
            solid_color: color.to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.tracks[0].segments.push(Segment::new(
            "segment",
            "color",
            TimeRange::new(0, 1_000_000),
        ));
        draft
    }

    fn pixels(draft: &Draft, time: i64) -> Vec<u8> {
        let mut device = SoftwareDevice::new();
        frame(&Snapshot::new(draft), Micros(time), &mut device)
            .unwrap()
            .rgba
    }

    #[test]
    fn the_same_snapshot_at_the_same_microsecond_returns_the_same_pixels() {
        let draft = color_clip("0xFF336699", false);
        let before = draft.clone();
        let mut device = SoftwareDevice::new();
        let snapshot = Snapshot::new(&draft);
        let first = frame(&snapshot, Micros(10), &mut device).unwrap();
        let second = frame(&snapshot, Micros(10), &mut device).unwrap();
        assert_eq!(first.rgba, second.rgba);
        assert_eq!(first.rgba, [0x33, 0x66, 0x99, 0xFF].repeat(2));
        assert_eq!(draft, before);
    }

    #[test]
    fn a_time_outside_the_span_a_gradient_and_a_hidden_track_draw_nothing() {
        let draft = color_clip("0xFF336699", false);
        let before = draft.clone();
        assert_eq!(pixels(&draft, 1_000_000), vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(pixels(&draft, -1), vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(draft, before);

        let gradient = color_clip("0xFF336699", true);
        let before_gradient = gradient.clone();
        assert_eq!(pixels(&gradient, 10), vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(gradient, before_gradient);

        let mut hidden = color_clip("0xFF336699", false);
        hidden.tracks[0].attribute = TrackAttribute(TrackAttribute::HIDE);
        let before_hidden = hidden.clone();
        assert_eq!(pixels(&hidden, 10), vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(hidden, before_hidden);

        let mut invisible = color_clip("0xFF336699", false);
        invisible.tracks[0].segments[0].visible = false;
        let before_invisible = invisible.clone();
        assert_eq!(pixels(&invisible, 10), vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(invisible, before_invisible);
    }

    #[test]
    fn an_empty_draft_and_a_bad_color_leave_a_clear_frame() {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 1;
        draft.canvas_config.height = 1;
        let before = draft.clone();
        assert_eq!(pixels(&draft, 0), vec![0, 0, 0, 0]);
        assert_eq!(draft, before);

        let named = color_clip("white", false);
        let before_named = named.clone();
        assert_eq!(pixels(&named, 10), vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(named, before_named);

        let mut not_a_clip = color_clip("0xFF336699", false);
        not_a_clip.materials.material_colors[0].is_color_clip = false;
        let before_clip = not_a_clip.clone();
        assert_eq!(pixels(&not_a_clip, 10), vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(not_a_clip, before_clip);
    }

    #[test]
    fn a_higher_render_index_is_drawn_on_top() {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 1;
        draft.canvas_config.height = 1;
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "blue".into(),
            is_color_clip: true,
            solid_color: "0xFF0000FF".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "red".into(),
            is_color_clip: true,
            solid_color: "0xFFFF0000".to_owned(),
            ..MaterialColorConfig::default()
        });
        let mut under = Segment::new("under", "blue", TimeRange::new(0, 1_000));
        under.render_index = 1;
        let mut over = Segment::new("over", "red", TimeRange::new(0, 1_000));
        over.render_index = 2;
        draft.tracks[0].segments.push(over);
        draft.tracks[0].segments.push(under);
        let before = draft.clone();
        assert_eq!(pixels(&draft, 0), vec![255, 0, 0, 255]);
        assert_eq!(draft, before);
    }

    #[test]
    fn a_negative_canvas_is_refused_and_the_draft_stays() {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = -1;
        let before = draft.clone();
        let mut device = SoftwareDevice::new();
        let error = frame(&Snapshot::new(&draft), Micros(0), &mut device).unwrap_err();
        assert_eq!(error, EvalError::Canvas);
        assert_eq!(draft, before);
    }

    #[test]
    fn export_keeps_a_frame_that_arrives_after_its_request() {
        use cutlass_compositor::Extent;

        let frame = Frame {
            extent: Extent {
                width: 1,
                height: 1,
            },
            rgba: vec![4, 5, 6, 7],
        };
        let kept = wait_export_frame::<()>(Micros(10), Micros(80), frame).expect("wait");
        assert_eq!(kept.rgba, vec![4, 5, 6, 7]);
        assert_eq!(kept.extent.width, 1);
    }
}
