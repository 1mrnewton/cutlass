//! The preview stage: where a picture clip sits on the canvas, which clip a
//! finger lands on, and how a drag, a pinch, and a twist become its
//! position, scale, and rotation.
//!
//! Points are canvas pixels, x to the right and y down, as a screen draws.
//! A turn is degrees clockwise on screen. The draft stores the position as
//! fractions of half the canvas with y up, and rotation counter-clockwise;
//! this module converts both ways. Keyframes are not read yet: the box is
//! the clip's stored placement.
//!
//! A video or photo fits inside the canvas. A title keeps the size its text
//! is drawn at, which only a host with the faces knows: it answers through
//! [`TextSizes`].

use cutlass_model::{Crop, Draft, MaterialRef, Segment, TrackAttribute};
use cutlass_session::{Micros, Request, SegmentId};

/// A centre or an edge snaps within this share of the canvas's short side.
pub const SNAP_REACH: f64 = 0.02;
/// A turn snaps to a quarter within this many degrees.
pub const TURN_REACH: f64 = 5.0;
/// A scale snaps to 1, the fitted size, within this share.
pub const SCALE_REACH: f64 = 0.04;
/// The smallest scale a pinch reaches.
pub const MIN_SCALE: f64 = 0.05;
/// The largest scale a pinch reaches.
pub const MAX_SCALE: f64 = 20.0;

/// A picture clip's box on the canvas.
#[derive(Clone, Debug, PartialEq)]
pub struct StageBox {
    pub segment_id: SegmentId,
    pub center_x: f64,
    pub center_y: f64,
    /// Drawn width before the turn.
    pub width: f64,
    /// Drawn height before the turn.
    pub height: f64,
    /// Degrees clockwise on screen.
    pub turn: f64,
}

impl StageBox {
    /// Whether the canvas point `(x, y)` is inside the turned box.
    #[must_use]
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let (sin, cos) = self.turn.to_radians().sin_cos();
        let (dx, dy) = (x - self.center_x, y - self.center_y);
        // Turn the point back by the box's turn.
        let local_x = dx * cos + dy * sin;
        let local_y = -dx * sin + dy * cos;
        local_x.abs() <= self.width / 2.0 && local_y.abs() <= self.height / 2.0
    }

    /// Top left, top right, bottom right, bottom left, after the turn.
    #[must_use]
    pub fn corners(&self) -> [(f64, f64); 4] {
        let (sin, cos) = self.turn.to_radians().sin_cos();
        let (w, h) = (self.width / 2.0, self.height / 2.0);
        [(-w, -h), (w, -h), (w, h), (-w, h)].map(|(x, y)| {
            (
                self.center_x + x * cos - y * sin,
                self.center_y + x * sin + y * cos,
            )
        })
    }
}

/// What the draft stores for a picture's placement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Fractions of half the canvas width, to the right.
    pub x: f64,
    /// Fractions of half the canvas height, up.
    pub y: f64,
    pub scale: f64,
    /// Degrees counter-clockwise, from 0 up to 360.
    pub rotation: f64,
}

/// The canvas centre lines a snapped clip sits on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageGuides {
    /// The clip's centre is on the canvas's upright centre line.
    pub vertical: bool,
    /// The clip's centre is on the canvas's level centre line.
    pub horizontal: bool,
}

/// One moment of a gesture: the placement, its box, and the guides to show.
#[derive(Clone, Debug, PartialEq)]
pub struct StageStep {
    pub pose: Pose,
    pub stage: StageBox,
    pub guides: StageGuides,
}

/// A drag, pinch, and twist on one clip, measured from where it started.
#[derive(Clone, Debug, PartialEq)]
pub struct StageGesture {
    segment_id: SegmentId,
    canvas: (f64, f64),
    /// The picture fitted in the canvas at scale 1.
    fitted: (f64, f64),
    start: Pose,
}

/// The size a title's text picture is drawn at, before placement.
pub trait TextSizes {
    fn text_size(&self, segment: &Segment) -> Option<(f64, f64)>;
}

/// No faces: titles have no box.
pub struct NoText;

impl TextSizes for NoText {
    fn text_size(&self, _segment: &Segment) -> Option<(f64, f64)> {
        None
    }
}

/// The canvas size from the draft. A size that is not positive is 1920×1080.
#[must_use]
pub fn canvas_size(draft: &Draft) -> (f64, f64) {
    let config = &draft.canvas_config;
    if config.width > 0 && config.height > 0 {
        (f64::from(config.width), f64::from(config.height))
    } else {
        (1920.0, 1080.0)
    }
}

/// The box of the picture or title clip `segment_id`. Other clips, and
/// ones with no size, have none.
#[must_use]
pub fn stage_box(draft: &Draft, segment_id: &SegmentId, sizes: &dyn TextSizes) -> Option<StageBox> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| segment.id == *segment_id)?;
    let canvas = canvas_size(draft);
    let fitted = fitted(draft, segment, canvas, sizes)?;
    Some(box_at(segment_id, canvas, fitted, pose_of(segment)))
}

/// The topmost picture or title clip under the canvas point `(x, y)` at
/// `time`. Hidden rows and clips, locked rows, and clear or empty pictures
/// are passed over.
#[must_use]
pub fn stage_pick(
    draft: &Draft,
    time: Micros,
    (x, y): (f64, f64),
    sizes: &dyn TextSizes,
) -> Option<SegmentId> {
    let mut found: Option<((i32, usize, usize), SegmentId)> = None;
    for (track_index, track) in draft.tracks.iter().enumerate() {
        if track.attribute.contains(TrackAttribute::HIDE)
            || track.attribute.contains(TrackAttribute::LOCK)
        {
            continue;
        }
        for (segment_index, segment) in track.segments.iter().enumerate() {
            let start = segment.target.start.0;
            let end = start.saturating_add(segment.target.duration.0);
            if !segment.visible || segment.clip.alpha <= 0.0 || !(start..end).contains(&time.0) {
                continue;
            }
            let key = (segment.render_index, track_index, segment_index);
            if found.as_ref().is_some_and(|(top, _)| *top > key) {
                continue;
            }
            if stage_box(draft, &segment.id, sizes).is_some_and(|stage| stage.contains(x, y)) {
                found = Some((key, segment.id.clone()));
            }
        }
    }
    found.map(|(_, id)| id)
}

/// Start a gesture on the picture or title clip `segment_id`. Other clips,
/// and ones on a locked row, have none.
#[must_use]
pub fn stage_gesture(
    draft: &Draft,
    segment_id: &SegmentId,
    sizes: &dyn TextSizes,
) -> Option<StageGesture> {
    let (track, segment) = draft.tracks.iter().find_map(|track| {
        let segment = track.segments.iter().find(|s| s.id == *segment_id)?;
        Some((track, segment))
    })?;
    if track.attribute.contains(TrackAttribute::LOCK) {
        return None;
    }
    let canvas = canvas_size(draft);
    Some(StageGesture {
        segment_id: segment_id.clone(),
        canvas,
        fitted: fitted(draft, segment, canvas, sizes)?,
        start: pose_of(segment),
    })
}

impl StageGesture {
    #[must_use]
    pub fn segment_id(&self) -> &SegmentId {
        &self.segment_id
    }

    /// The placement the clip had when the gesture started.
    #[must_use]
    pub const fn start(&self) -> Pose {
        self.start
    }

    /// Where the clip is after the fingers moved `(shift_x, shift_y)` canvas
    /// pixels, pinched by `pinch` (1 is no change), and turned `turn`
    /// degrees clockwise, all since the start. The centre snaps to the
    /// canvas centre lines, a quarter turn edge to the canvas edges, the
    /// turn to quarters, and the scale to 1.
    #[must_use]
    pub fn step(&self, shift_x: f64, shift_y: f64, pinch: f64, turn: f64) -> StageStep {
        let (width, height) = self.canvas;
        let reach = width.min(height) * SNAP_REACH;
        let pinch = if pinch.is_finite() && pinch > 0.0 {
            pinch
        } else {
            1.0
        };
        let mut scale = (self.start.scale * pinch).clamp(MIN_SCALE, MAX_SCALE);
        if (scale - 1.0).abs() <= SCALE_REACH {
            scale = 1.0;
        }
        let turn = if turn.is_finite() { turn } else { 0.0 };
        let mut rotation = (self.start.rotation - turn).rem_euclid(360.0);
        let quarter = (rotation / 90.0).round() * 90.0;
        let square = (rotation - quarter).abs() <= TURN_REACH;
        if square {
            rotation = quarter.rem_euclid(360.0);
        }
        let (start_x, start_y) = centre_of(self.canvas, self.start);
        let finite = |value: f64| if value.is_finite() { value } else { 0.0 };
        let mut center_x = start_x + finite(shift_x);
        let mut center_y = start_y + finite(shift_y);
        // The drawn half sizes along the canvas axes when the turn is square.
        let (mut half_w, mut half_h) = (self.fitted.0 * scale / 2.0, self.fitted.1 * scale / 2.0);
        if square && (rotation == 90.0 || rotation == 270.0) {
            (half_w, half_h) = (half_h, half_w);
        }
        let mut guides = StageGuides::default();
        if let Some(snapped) = snap_axis(center_x, width, half_w, square, reach) {
            guides.vertical = snapped == width / 2.0;
            center_x = snapped;
        }
        if let Some(snapped) = snap_axis(center_y, height, half_h, square, reach) {
            guides.horizontal = snapped == height / 2.0;
            center_y = snapped;
        }
        let pose = Pose {
            x: (center_x - width / 2.0) / (width / 2.0),
            y: (height / 2.0 - center_y) / (height / 2.0),
            scale,
            rotation,
        };
        StageStep {
            stage: box_at(&self.segment_id, self.canvas, self.fitted, pose),
            pose,
            guides,
        }
    }

    /// The requests that store `pose` on the clip.
    #[must_use]
    pub fn requests(&self, pose: Pose) -> Vec<Request> {
        vec![
            Request::SetPosition {
                segment_id: self.segment_id.clone(),
                x: pose.x,
                y: pose.y,
            },
            Request::SetScale {
                segment_id: self.segment_id.clone(),
                scale: pose.scale,
            },
            Request::SetRotation {
                segment_id: self.segment_id.clone(),
                degrees: pose.rotation,
            },
        ]
    }
}

/// The nearest snap for a centre at `center` on an axis `length` long: the
/// middle, or with a square turn an edge of the picture on an edge of the
/// canvas.
fn snap_axis(center: f64, length: f64, half: f64, square: bool, reach: f64) -> Option<f64> {
    let mut targets = vec![length / 2.0];
    if square {
        targets.extend([half, length - half]);
    }
    targets
        .into_iter()
        .map(|target| (target, (center - target).abs()))
        .filter(|(_, distance)| *distance <= reach)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(target, _)| target)
}

fn pose_of(segment: &Segment) -> Pose {
    let clip = &segment.clip;
    let scale = if segment.uniform_scale.on {
        segment.uniform_scale.value
    } else {
        clip.scale.x
    };
    let finite = |value: f64, default: f64| if value.is_finite() { value } else { default };
    Pose {
        x: finite(clip.transform.x, 0.0),
        y: finite(clip.transform.y, 0.0),
        scale: finite(scale, 1.0).abs(),
        rotation: finite(clip.rotation, 0.0).rem_euclid(360.0),
    }
}

fn centre_of(canvas: (f64, f64), pose: Pose) -> (f64, f64) {
    let (width, height) = canvas;
    (
        width / 2.0 + pose.x * width / 2.0,
        height / 2.0 - pose.y * height / 2.0,
    )
}

fn box_at(segment_id: &SegmentId, canvas: (f64, f64), fitted: (f64, f64), pose: Pose) -> StageBox {
    let (center_x, center_y) = centre_of(canvas, pose);
    StageBox {
        segment_id: segment_id.clone(),
        center_x,
        center_y,
        width: fitted.0 * pose.scale,
        height: fitted.1 * pose.scale,
        // Adding zero turns a negative zero into zero.
        turn: (-pose.rotation).rem_euclid(360.0) + 0.0,
    }
}

/// The size drawn at scale 1. A picture fits inside the canvas, keeping
/// the cropped shape; a title keeps its own size unless it is larger than
/// the canvas, when it fits too.
fn fitted(
    draft: &Draft,
    segment: &Segment,
    canvas: (f64, f64),
    sizes: &dyn TextSizes,
) -> Option<(f64, f64)> {
    let (width, height, native) = match draft.materials.get(segment.material_id.as_str())? {
        MaterialRef::Video(video) => {
            segment.source_range.as_ref()?;
            let (width, height) = (f64::from(video.width), f64::from(video.height));
            if width <= 0.0 || height <= 0.0 {
                return None;
            }
            let (width, height) = cropped(video.crop, width, height);
            (width, height, false)
        }
        MaterialRef::Text(_) => {
            let (width, height) = sizes.text_size(segment)?;
            (width, height, true)
        }
        _ => return None,
    };
    if !(width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0) {
        return None;
    }
    if native && width <= canvas.0 && height <= canvas.1 {
        return Some((width, height));
    }
    let fit = (canvas.0 / width).min(canvas.1 / height);
    Some((width * fit, height * fit))
}

/// The shape a crop quad leaves of a `width` by `height` picture. No crop,
/// or a quad with no area, leaves the whole picture.
fn cropped(crop: Crop, width: f64, height: f64) -> (f64, f64) {
    let edge = |x0: f64, y0: f64, x1: f64, y1: f64| {
        let (dx, dy) = ((x1 - x0) * width, (y1 - y0) * height);
        dx.hypot(dy)
    };
    let across = edge(
        crop.upper_left_x,
        crop.upper_left_y,
        crop.upper_right_x,
        crop.upper_right_y,
    );
    let down = edge(
        crop.upper_left_x,
        crop.upper_left_y,
        crop.lower_left_x,
        crop.lower_left_y,
    );
    if across.is_finite() && down.is_finite() && across >= 1.0 && down >= 1.0 {
        (across, down)
    } else {
        (width, height)
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{MaterialVideo, TimeRange, Track, TrackFlag};

    use super::*;

    const EPS: f64 = 1e-9;

    /// A 1920×1080 canvas with a 1080×1920 picture on main and a 1920×1080
    /// picture over it, both over 0..10.
    fn draft() -> Draft {
        let mut draft = Draft::empty("D");
        draft.canvas_config.width = 1920;
        draft.canvas_config.height = 1080;
        for (id, width, height) in [("tall", 1080, 1920), ("wide", 1920, 1080)] {
            let mut video = MaterialVideo::new(id);
            video.width = width;
            video.height = height;
            draft.materials.videos.push(video);
        }
        let clip = |id: &str, material: &str| {
            let mut segment = Segment::new(id, material, TimeRange::new(0, 10));
            segment.source_range = Some(TimeRange::new(0, 10));
            segment
        };
        draft.tracks[0].segments.push(clip("main", "tall"));
        let mut over = Track::main("over");
        over.flag = TrackFlag::SubVideo;
        over.segments.push(clip("over", "wide"));
        draft.tracks.push(over);
        draft
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn a_picture_fits_the_canvas_and_its_box_follows_the_placement() {
        let mut draft = draft();
        let main = SegmentId::new("main");
        let stage = stage_box(&draft, &main, &NoText).unwrap();
        assert!(near(stage.center_x, 960.0) && near(stage.center_y, 540.0));
        assert!(near(stage.width, 607.5) && near(stage.height, 1080.0));
        let segment = &mut draft.tracks[0].segments[0];
        segment.clip.transform.x = 0.5;
        segment.clip.transform.y = 0.5;
        segment.clip.rotation = 90.0;
        segment.uniform_scale.value = 0.5;
        let stage = stage_box(&draft, &main, &NoText).unwrap();
        assert!(near(stage.center_x, 1440.0) && near(stage.center_y, 270.0));
        assert!(near(stage.width, 303.75) && near(stage.turn, 270.0));
        // A quarter turn counter-clockwise lays the tall box on its side.
        let [top_left, ..] = stage.corners();
        assert!(near(top_left.0, 1440.0 - 270.0) && near(top_left.1, 270.0 + 151.875));
        assert!(stage.contains(1440.0 + 260.0, 270.0));
        assert!(!stage.contains(1440.0, 270.0 + 200.0));
    }

    struct Sized(f64, f64);

    impl TextSizes for Sized {
        fn text_size(&self, _segment: &Segment) -> Option<(f64, f64)> {
            Some((self.0, self.1))
        }
    }

    #[test]
    fn a_title_keeps_its_own_size_and_is_picked_above_the_picture() {
        let mut draft = draft();
        draft
            .materials
            .texts
            .push(cutlass_model::MaterialText::new("words"));
        let mut words = Track::main("words");
        words.kind = cutlass_model::TrackType::Text;
        words.flag = TrackFlag::SubVideo;
        words
            .segments
            .push(Segment::new("title", "words", TimeRange::new(0, 10)));
        draft.tracks.push(words);
        let title = SegmentId::new("title");
        assert_eq!(stage_box(&draft, &title, &NoText), None);
        let stage = stage_box(&draft, &title, &Sized(300.0, 80.0)).unwrap();
        assert!(near(stage.width, 300.0) && near(stage.height, 80.0));
        let huge = stage_box(&draft, &title, &Sized(3840.0, 80.0)).unwrap();
        assert!(near(huge.width, 1920.0) && near(huge.height, 40.0));
        let picked = stage_pick(&draft, Micros(5), (960.0, 540.0), &Sized(300.0, 80.0));
        assert_eq!(picked, Some(title.clone()));
        let beside = stage_pick(&draft, Micros(5), (960.0, 300.0), &Sized(300.0, 80.0));
        assert_eq!(beside, Some(SegmentId::new("over")));
        assert!(stage_gesture(&draft, &title, &Sized(300.0, 80.0)).is_some());
    }

    #[test]
    fn a_crop_gives_the_box_its_shape() {
        let mut draft = draft();
        draft.materials.videos[1].crop = Crop {
            upper_left_x: 0.0,
            upper_left_y: 0.0,
            upper_right_x: 0.5,
            upper_right_y: 0.0,
            lower_left_x: 0.0,
            lower_left_y: 1.0,
            lower_right_x: 0.5,
            lower_right_y: 1.0,
        };
        let stage = stage_box(&draft, &SegmentId::new("over"), &NoText).unwrap();
        assert!(near(stage.width, 960.0) && near(stage.height, 1080.0));
    }

    #[test]
    fn the_top_picture_under_a_finger_is_picked() {
        let mut draft = draft();
        let at = |draft: &Draft, x, y| stage_pick(draft, Micros(5), (x, y), &NoText);
        assert_eq!(at(&draft, 960.0, 540.0), Some(SegmentId::new("over")));
        draft.tracks[1].segments[0].uniform_scale.value = 0.25;
        assert_eq!(at(&draft, 960.0, 540.0), Some(SegmentId::new("over")));
        assert_eq!(at(&draft, 960.0, 100.0), Some(SegmentId::new("main")));
        assert_eq!(at(&draft, 100.0, 100.0), None);
        assert_eq!(
            stage_pick(&draft, Micros(10), (960.0, 540.0), &NoText),
            None
        );
        draft.tracks[1].attribute = TrackAttribute(TrackAttribute::LOCK);
        assert_eq!(at(&draft, 960.0, 540.0), Some(SegmentId::new("main")));
    }

    #[test]
    fn a_drag_moves_the_centre_and_snaps_to_the_middle() {
        let draft = draft();
        let gesture = stage_gesture(&draft, &SegmentId::new("main"), &NoText).unwrap();
        let step = gesture.step(480.0, -270.0, 1.0, 0.0);
        assert!(near(step.pose.x, 0.5) && near(step.pose.y, 0.5));
        assert_eq!(step.guides, StageGuides::default());
        let step = gesture.step(480.0, 10.0, 1.0, 0.0);
        assert!(near(step.pose.y, 0.0) && step.guides.horizontal && !step.guides.vertical);
        let step = gesture.step(5.0, 5.0, 1.0, 0.0);
        assert!(near(step.pose.x, 0.0) && near(step.pose.y, 0.0));
        assert!(step.guides.horizontal && step.guides.vertical);
    }

    #[test]
    fn a_square_edge_snaps_to_the_canvas_edge() {
        let draft = draft();
        let gesture = stage_gesture(&draft, &SegmentId::new("main"), &NoText).unwrap();
        // The tall box is 607.5 wide; its left edge lands near 0.
        let step = gesture.step(-960.0 + 303.75 + 10.0, 0.0, 1.0, 0.0);
        assert!(near(step.stage.center_x, 303.75));
        assert!(!step.guides.vertical);
        // Turned off square, the edges do not snap.
        let step = gesture.step(-960.0 + 303.75 + 10.0, 0.0, 1.0, 20.0);
        assert!(near(step.stage.center_x, 303.75 + 10.0));
    }

    #[test]
    fn a_pinch_scales_and_a_twist_turns_with_snaps() {
        let mut draft = draft();
        draft.tracks[0].segments[0].clip.rotation = 10.0;
        let gesture = stage_gesture(&draft, &SegmentId::new("main"), &NoText).unwrap();
        let step = gesture.step(0.0, 0.0, 2.0, 30.0);
        assert!(near(step.pose.scale, 2.0) && near(step.pose.rotation, 340.0));
        assert!(near(step.stage.turn, 20.0));
        let step = gesture.step(0.0, 0.0, 1.03, 7.0);
        assert!(near(step.pose.scale, 1.0) && near(step.pose.rotation, 0.0));
        let step = gesture.step(0.0, 0.0, 0.0001, -85.0);
        assert!(near(step.pose.scale, MIN_SCALE) && near(step.pose.rotation, 90.0));
        let step = gesture.step(f64::NAN, 0.0, f64::NAN, f64::NAN);
        assert!(near(step.pose.scale, 1.0) && near(step.pose.rotation, 10.0));
        assert!(step.pose.x.abs() < EPS);
    }

    #[test]
    fn a_step_becomes_the_three_placement_requests() {
        let draft = draft();
        let gesture = stage_gesture(&draft, &SegmentId::new("main"), &NoText).unwrap();
        let mut session = cutlass_session::Session::open(draft);
        let step = gesture.step(480.0, -270.0, 0.5, 45.0);
        session.apply(0, &gesture.requests(step.pose)).unwrap();
        let after = stage_box(session.draft(), gesture.segment_id(), &NoText).unwrap();
        assert_eq!(after, step.stage);
    }
}
