//! The choices a project's settings offer: frame rates, color spaces and
//! canvas shapes, with their labels and the step to the next one.

use cutlass_model::{Draft, FrameRate};
use cutlass_session::{CANVAS_SHAPES, CanvasShape, EditError, Request, canvas_for};

/// The project frame rates, slowest first.
pub const PROJECT_RATES: [FrameRate; 9] = [
    FrameRate::FPS_23_976,
    FrameRate::FPS_24,
    FrameRate::FPS_25,
    FrameRate::FPS_29_97,
    FrameRate::FPS_30,
    FrameRate::FPS_50,
    FrameRate::FPS_59_94,
    FrameRate::FPS_60,
    FrameRate::FPS_120,
];

/// The stored color-space ints and their names.
pub const COLOR_SPACES: [(i32, &str); 5] = [
    (-1, "Unknown"),
    (0, "None"),
    (2, "SDR 2020"),
    (67, "HLG"),
    (2084, "PQ"),
];

/// The rate after `current`; an unknown rate steps to the first.
#[must_use]
pub fn next_rate(current: Option<FrameRate>) -> FrameRate {
    let index = current
        .and_then(|rate| PROJECT_RATES.iter().position(|item| *item == rate))
        .map_or(0, |index| index.saturating_add(1));
    PROJECT_RATES[index % PROJECT_RATES.len()]
}

/// Drop-frame timecode exists for 29.97 and 59.94.
#[must_use]
pub fn is_drop_rate(rate: FrameRate) -> bool {
    matches!(
        (rate.numerator, rate.denominator),
        (30_000, 1001) | (60_000, 1001)
    )
}

/// The frames per second the draft stores for `rate`.
#[must_use]
pub fn rate_fps(rate: FrameRate) -> f64 {
    f64::from(rate.numerator) / f64::from(rate.denominator)
}

/// `29.97 fps`, `30 fps`.
#[must_use]
pub fn rate_label(rate: FrameRate) -> String {
    let name = match (rate.numerator, rate.denominator) {
        (24_000, 1001) => "23.976",
        (30_000, 1001) => "29.97",
        (60_000, 1001) => "59.94",
        (numerator, 1) => return format!("{numerator} fps"),
        _ => "Frame rate",
    };
    format!("{name} fps")
}

/// The name of a stored color space; an int outside the list shows as itself.
#[must_use]
pub fn color_space_name(space: i32) -> String {
    COLOR_SPACES
        .iter()
        .find(|item| item.0 == space)
        .map_or_else(|| space.to_string(), |item| item.1.to_owned())
}

/// The color space after `current`; an unknown int steps to the first.
#[must_use]
pub fn next_color_space(current: i32) -> i32 {
    let index = COLOR_SPACES
        .iter()
        .position(|item| item.0 == current)
        .map_or(0, |index| index.saturating_add(1));
    COLOR_SPACES[index % COLOR_SPACES.len()].0
}

/// The canvas shape after the one this frame matches. A new project is
/// 1920×1080, so the next shape is 9:16.
#[must_use]
pub fn next_canvas(width: i32, height: i32) -> CanvasShape {
    let index = CANVAS_SHAPES
        .iter()
        .position(|shape| shape.width == width && shape.height == height)
        .map_or(0, |index| index.saturating_add(1));
    CANVAS_SHAPES[index % CANVAS_SHAPES.len()]
}

/// The canvas shapes the project dialog offers, as their stored ratio.
/// `original` takes the first picture's size.
pub const PROJECT_RATIOS: [(&str, &str); 6] = [
    ("original", "Original"),
    ("16:9", "16:9"),
    ("9:16", "9:16"),
    ("1:1", "1:1"),
    ("4:5", "4:5"),
    ("4:3", "4:3"),
];

/// The resolutions the project dialog offers, as the short side of the
/// canvas. `None` keeps the shape's own size.
pub const PROJECT_RESOLUTIONS: [(Option<i32>, &str); 5] = [
    (None, "Adapted"),
    (Some(720), "720p"),
    (Some(1080), "1080p"),
    (Some(1440), "2K"),
    (Some(2160), "4K"),
];

/// Everything the project dialog sets, saved together.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectSettings {
    pub name: String,
    /// A ratio from [`PROJECT_RATIOS`], or the one the project stores.
    pub ratio: String,
    /// The canvas's short side; `None` keeps the shape's own size.
    pub short_side: Option<i32>,
    pub rate: FrameRate,
    /// Only kept for 29.97 and 59.94.
    pub drop_frame: bool,
    pub color_space: i32,
}

/// The settings `draft` has now. A short side that is one of
/// [`PROJECT_RESOLUTIONS`] is named; any other size is Adapted.
#[must_use]
pub fn project_settings(draft: &Draft) -> ProjectSettings {
    let canvas = &draft.canvas_config;
    let short = canvas.width.min(canvas.height);
    ProjectSettings {
        name: draft.name.clone(),
        ratio: canvas.ratio.clone(),
        short_side: PROJECT_RESOLUTIONS
            .iter()
            .find_map(|(side, _)| side.filter(|side| *side == short)),
        rate: draft.frame_rate().unwrap_or(FrameRate::FPS_30),
        drop_frame: draft.is_drop_frame_timecode,
        color_space: draft.color_space,
    }
}

/// The canvas of `ratio` with its short side `short_side`, both edges even.
pub fn project_canvas(
    draft: &Draft,
    ratio: &str,
    short_side: Option<i32>,
) -> Result<CanvasShape, EditError> {
    let shape = canvas_for(draft, ratio)?;
    let Some(side) = short_side else {
        return Ok(shape);
    };
    let short = shape.width.min(shape.height);
    if short <= 0 || side <= 0 {
        return Err(EditError::CanvasRatio);
    }
    let scale = |edge: i32| {
        let scaled = i64::from(edge) * i64::from(side) / i64::from(short);
        i32::try_from(scaled / 2 * 2).unwrap_or(i32::MAX)
    };
    Ok(CanvasShape {
        width: scale(shape.width),
        height: scale(shape.height),
        ..shape
    })
}

/// The requests that bring `draft` to `settings`, only for what changed, so
/// the dialog saves as one undo step. A blank name is kept blank.
pub fn project_requests(
    draft: &Draft,
    settings: &ProjectSettings,
) -> Result<Vec<Request>, EditError> {
    let now = project_settings(draft);
    let mut requests = Vec::new();
    let name = settings.name.trim();
    if name != draft.name {
        requests.push(Request::SetProjectName {
            name: name.to_owned(),
        });
    }
    if settings.ratio != now.ratio || settings.short_side != now.short_side {
        let shape = project_canvas(draft, &settings.ratio, settings.short_side)?;
        let canvas = &draft.canvas_config;
        if shape.width != canvas.width
            || shape.height != canvas.height
            || shape.ratio != canvas.ratio
        {
            requests.push(Request::SetCanvasSize {
                width: shape.width,
                height: shape.height,
                ratio: shape.ratio.to_owned(),
            });
        }
    }
    let drop_frame = settings.drop_frame && is_drop_rate(settings.rate);
    if draft.frame_rate() != Some(settings.rate) {
        requests.push(Request::SetFrameRate {
            fps: rate_fps(settings.rate),
            drop_frame,
        });
    } else if drop_frame != draft.is_drop_frame_timecode {
        requests.push(Request::SetDropFrame { on: drop_frame });
    }
    if settings.color_space != draft.color_space {
        requests.push(Request::SetColorSpace {
            space: settings.color_space,
        });
    }
    Ok(requests)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dialog_saves_only_what_changed() {
        let draft = Draft::empty("Trip");
        let now = project_settings(&draft);
        assert_eq!(project_requests(&draft, &now).unwrap(), Vec::new());
        let settings = ProjectSettings {
            name: "  Beach ".to_owned(),
            ratio: "9:16".to_owned(),
            short_side: Some(720),
            rate: FrameRate::FPS_29_97,
            drop_frame: true,
            color_space: 67,
        };
        let requests = project_requests(&draft, &settings).unwrap();
        assert_eq!(
            requests,
            vec![
                Request::SetProjectName {
                    name: "Beach".to_owned()
                },
                Request::SetCanvasSize {
                    width: 720,
                    height: 1280,
                    ratio: "9:16".to_owned()
                },
                Request::SetFrameRate {
                    fps: 30_000.0 / 1001.0,
                    drop_frame: true
                },
                Request::SetColorSpace { space: 67 },
            ]
        );
        // Drop frame is dropped for a rate that has none.
        let plain = ProjectSettings {
            rate: FrameRate::FPS_25,
            drop_frame: true,
            ..now
        };
        assert_eq!(
            project_requests(&draft, &plain).unwrap(),
            vec![Request::SetFrameRate {
                fps: 25.0,
                drop_frame: false
            }]
        );
    }

    #[test]
    fn a_resolution_sets_the_short_side_with_even_edges() {
        let draft = Draft::empty("D");
        let shape = project_canvas(&draft, "4:5", Some(2160)).unwrap();
        assert_eq!((shape.width, shape.height), (2160, 2700));
        let shape = project_canvas(&draft, "16:9", Some(720)).unwrap();
        assert_eq!((shape.width, shape.height), (1280, 720));
        let kept = project_canvas(&draft, "1:1", None).unwrap();
        assert_eq!((kept.width, kept.height), (1080, 1080));
        assert!(project_canvas(&draft, "7:3", None).is_err());
    }

    #[test]
    fn rates_step_and_wrap_and_label() {
        assert_eq!(next_rate(None), FrameRate::FPS_23_976);
        assert_eq!(next_rate(Some(FrameRate::FPS_29_97)), FrameRate::FPS_30);
        assert_eq!(next_rate(Some(FrameRate::FPS_120)), FrameRate::FPS_23_976);
        assert_eq!(rate_label(FrameRate::FPS_59_94), "59.94 fps");
        assert_eq!(rate_label(FrameRate::FPS_25), "25 fps");
        assert!(is_drop_rate(FrameRate::FPS_29_97));
        assert!(!is_drop_rate(FrameRate::FPS_23_976));
        assert!((rate_fps(FrameRate::FPS_29_97) - 29.97).abs() < 0.001);
    }

    #[test]
    fn color_spaces_and_canvas_step() {
        assert_eq!(color_space_name(67), "HLG");
        assert_eq!(color_space_name(9), "9");
        assert_eq!(next_color_space(2084), -1);
        assert_eq!(next_color_space(9), -1);
        let after = next_canvas(1920, 1080);
        assert_eq!((after.width, after.height), (1080, 1920));
    }
}
