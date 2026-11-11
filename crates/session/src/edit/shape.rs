//! Shapes: placing one from the library, and its size, fill, stroke,
//! corners and shadow.
//!
//! A shape lives on a sticker row, as in the reference. Its size is the
//! outline's own size, so a resize rewrites the points; the clip's scale is
//! a separate thing that stays. Fill opacity is `global_alpha`, stroke
//! opacity `border_alpha`, and shadow opacity `shadow_alpha`.

use cutlass_model::{
    Draft, MaterialId, MaterialShape, SHAPE_SIZE, Segment, SegmentId, ShapeKind, ShapePoint,
    TimeRange, TrackId, TrackType,
};

use super::{EditError, ensure_unlocked, locate};

/// The widest or tallest a shape can be, in canvas pixels.
const MAX_SIZE: f64 = cutlass_model::MAX_SHAPE_SIZE;
/// The widest a stroke can be, in canvas pixels.
const MAX_STROKE: f64 = cutlass_model::MAX_SHAPE_STROKE;
/// The farthest a shadow can fall, in canvas pixels.
const MAX_SHADOW_DISTANCE: f64 = cutlass_model::MAX_SHAPE_SHADOW_DISTANCE;
/// Solid, dashed and dotted.
const STROKE_STYLES: i32 = 3;

/// The most points a drawn path can have.
const MAX_POINTS: usize = 1_024;

/// Put a new `kind` shape over `target` on `track_id`, a sticker row made
/// when it is missing.
pub fn add_shape(
    draft: &mut Draft,
    track_id: &TrackId,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    target: TimeRange,
    kind: ShapeKind,
) -> Result<(), EditError> {
    let shape = MaterialShape::preset(material_id.clone(), kind, SHAPE_SIZE, SHAPE_SIZE);
    put_shape(draft, track_id, segment_id, target, shape, [0.0, 0.0])
}

/// Put a shape drawn with the pen through `points` over `target` on
/// `track_id`, its clip where the drawing was. `ids` is the new segment's
/// and the new material's.
pub fn draw_shape(
    draft: &mut Draft,
    track_id: &TrackId,
    ids: [&str; 2],
    target: TimeRange,
    points: &[ShapePoint],
    closed: bool,
) -> Result<(), EditError> {
    let finite = |pair: [f64; 2]| pair.iter().all(|value| value.is_finite());
    let fits = |pair: [f64; 2]| pair.iter().all(|value| value.abs() <= MAX_SIZE);
    if points.len() < 2
        || points.len() > MAX_POINTS
        || !points.iter().all(|point| {
            [point.at, point.handle_in, point.handle_out]
                .into_iter()
                .all(|pair| finite(pair) && fits(pair))
        })
    {
        return Err(EditError::ShapeStyle);
    }
    let points: Vec<ShapePoint> = points
        .iter()
        .map(|point| ShapePoint {
            round: 0.0,
            ..*point
        })
        .collect();
    let (shape, middle) = MaterialShape::drawn(MaterialId::new(ids[1]), &points, closed);
    let (width, height) = shape.size();
    // A path that never leaves one spot draws nothing.
    if width.max(height) < 1.0 || width > MAX_SIZE || height > MAX_SIZE {
        return Err(EditError::ShapeStyle);
    }
    let half = |side: i32| {
        if side > 0 { f64::from(side) / 2.0 } else { 1.0 }
    };
    let at = [
        middle[0] / half(draft.canvas_config.width),
        middle[1] / half(draft.canvas_config.height),
    ];
    put_shape(draft, track_id, &SegmentId::new(ids[0]), target, shape, at)
}

/// Place `shape` on a sticker row, made when it is missing, its clip moved
/// to `at` (fractions of half the canvas, y up).
fn put_shape(
    draft: &mut Draft,
    track_id: &TrackId,
    segment_id: &SegmentId,
    target: TimeRange,
    shape: MaterialShape,
    at: [f64; 2],
) -> Result<(), EditError> {
    if shape.id.is_empty() || draft.materials.get(shape.id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    if let Some(track) = draft.tracks.iter().find(|track| &track.id == track_id) {
        if !matches!(track.kind, TrackType::Sticker) {
            return Err(EditError::TrackKind);
        }
        ensure_unlocked(track.attribute)?;
    } else {
        super::track::insert_track(draft, track_id, &TrackType::Sticker)?;
    }
    let mut segment = Segment::new(segment_id.clone(), shape.id.clone(), target);
    segment.render_index = TrackType::Sticker.render_floor().unwrap_or(0);
    segment.clip.transform.x = at[0];
    segment.clip.transform.y = at[1];
    super::place::place(draft, track_id, segment)?;
    draft.materials.shapes.push(shape);
    Ok(())
}

/// Make the shape `width` by `height` canvas pixels. A line keeps no
/// height.
pub fn set_shape_size(
    draft: &mut Draft,
    segment_id: &SegmentId,
    width: f64,
    height: f64,
) -> Result<(), EditError> {
    let index = own_shape(draft, segment_id)?;
    let shape = &mut draft.materials.shapes[index];
    let line = shape.is_straight();
    let height = if line { 0.0 } else { height };
    let fits = |value: f64| (1.0..=MAX_SIZE).contains(&value);
    if !fits(width) || !(fits(height) || line) {
        return Err(EditError::ShapeStyle);
    }
    shape.resize(width, height);
    Ok(())
}

/// Fill the shape with `color` at `opacity`; an empty color takes the fill
/// off.
pub fn set_shape_fill(
    draft: &mut Draft,
    segment_id: &SegmentId,
    color: &str,
    opacity: f64,
) -> Result<(), EditError> {
    let index = own_shape(draft, segment_id)?;
    let color = shape_color(color)?;
    if !(0.0..=1.0).contains(&opacity) {
        return Err(EditError::ShapeStyle);
    }
    let shape = &mut draft.materials.shapes[index];
    shape.color = color.to_owned();
    shape.global_alpha = opacity;
    Ok(())
}

/// Stroke the outline: `amount` is the width and the opacity. A width of 0
/// takes the stroke off and keeps the color.
pub fn set_shape_stroke(
    draft: &mut Draft,
    segment_id: &SegmentId,
    color: &str,
    amount: [f64; 2],
    style: i32,
) -> Result<(), EditError> {
    let index = own_shape(draft, segment_id)?;
    let color = shape_color(color)?;
    let [width, opacity] = amount;
    if !(0.0..=MAX_STROKE).contains(&width)
        || !(0.0..=1.0).contains(&opacity)
        || !(0..STROKE_STYLES).contains(&style)
    {
        return Err(EditError::ShapeStyle);
    }
    let shape = &mut draft.materials.shapes[index];
    if !color.is_empty() {
        shape.border_color = color.to_owned();
    }
    shape.border_width = width;
    shape.border_alpha = opacity;
    shape.border_line_style = style;
    Ok(())
}

/// Round the corners by `radius`: one value for all, or one per corner.
pub fn set_shape_corners(
    draft: &mut Draft,
    segment_id: &SegmentId,
    radius: &[f64],
) -> Result<(), EditError> {
    let index = own_shape(draft, segment_id)?;
    let shape = &mut draft.materials.shapes[index];
    let corners = shape.custom_points.len() / 2;
    let (width, height) = shape.size();
    // A corner can round no further than half the shorter side.
    let most = (width.min(height) / 2.0).max(0.0);
    let valid = |value: &f64| value.is_finite() && (0.0..=most.max(MAX_SIZE)).contains(value);
    let radius = match radius {
        [one] if valid(one) => vec![one.min(most); corners],
        many if many.len() == corners && many.iter().all(valid) => {
            many.iter().map(|value| value.min(most)).collect()
        }
        _ => return Err(EditError::ShapeStyle),
    };
    shape.roundness = radius;
    Ok(())
}

/// Cast a shadow: `amount` is the distance, the angle in degrees, and the
/// opacity. An opacity of 0 takes the shadow off.
pub fn set_shape_shadow(
    draft: &mut Draft,
    segment_id: &SegmentId,
    color: &str,
    amount: [f64; 3],
) -> Result<(), EditError> {
    let index = own_shape(draft, segment_id)?;
    let color = shape_color(color)?;
    let [distance, angle, opacity] = amount;
    if !(0.0..=MAX_SHADOW_DISTANCE).contains(&distance)
        || !angle.is_finite()
        || !(0.0..=1.0).contains(&opacity)
    {
        return Err(EditError::ShapeStyle);
    }
    let shape = &mut draft.materials.shapes[index];
    if !color.is_empty() {
        shape.shadow_color = color.to_owned();
    }
    shape.shadow_distance = distance;
    shape.shadow_angle = angle;
    shape.shadow_alpha = opacity;
    Ok(())
}

/// A `#rrggbb` or `#rrggbbaa` color, or nothing.
fn shape_color(color: &str) -> Result<&str, EditError> {
    let color = color.trim();
    let body = color.strip_prefix('#').unwrap_or(color);
    if color.is_empty()
        || ((body.len() == 6 || body.len() == 8) && body.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        Ok(color)
    } else {
        Err(EditError::ShapeStyle)
    }
}

/// The index of the segment's own shape material, copied first when
/// another segment shares it.
fn own_shape(draft: &mut Draft, segment_id: &SegmentId) -> Result<usize, EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let shared = draft.tracks[track_index].segments[segment_index]
        .material_id
        .clone();
    if shared.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if !draft
        .materials
        .shapes
        .iter()
        .any(|shape| shape.id == shared)
    {
        return Err(if draft.materials.get(shared.as_str()).is_some() {
            EditError::NotShape
        } else {
            EditError::MissingMaterial
        });
    }
    let id = super::own::own(draft, track_index, segment_index, &shared)?;
    draft
        .materials
        .shapes
        .iter()
        .position(|shape| shape.id == id)
        .ok_or(EditError::MissingMaterial)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialId, SHAPE_FILL, SegmentId, ShapeKind, ShapePoint, TimeRange, TrackId,
    };

    use super::*;

    fn placed(kind: ShapeKind) -> Draft {
        let mut draft = Draft::empty("d");
        add_shape(
            &mut draft,
            &TrackId::new("row"),
            &SegmentId::new("seg"),
            &MaterialId::new("mat"),
            TimeRange::new(0, 3_000_000),
            kind,
        )
        .unwrap();
        draft
    }

    #[test]
    fn a_drawn_path_sits_where_it_was_drawn() {
        let mut draft = Draft::empty("d");
        draft.canvas_config.width = 1920;
        draft.canvas_config.height = 1080;
        let corner = |x: f64, y: f64| ShapePoint {
            at: [x, y],
            round: 5.0,
            ..ShapePoint::default()
        };
        let points = [
            corner(480.0, 270.0),
            corner(960.0, 270.0),
            corner(960.0, 0.0),
        ];
        draw_shape(
            &mut draft,
            &TrackId::new("row"),
            ["seg", "mat"],
            TimeRange::new(0, 3_000_000),
            &points,
            true,
        )
        .unwrap();
        let shape = &draft.materials.shapes[0];
        assert_eq!(shape.size(), (480.0, 270.0));
        assert_eq!(shape.roundness, vec![0.0; 3]);
        let segment = &draft.tracks[0].segments[0];
        assert_eq!(draft.tracks[0].kind, TrackType::Sticker);
        assert_eq!(segment.clip.transform.x, 0.75);
        assert_eq!(segment.clip.transform.y, 0.25);

        let one = [corner(10.0, 10.0), corner(10.0, 10.0)];
        let refused = draw_shape(
            &mut draft,
            &TrackId::new("row"),
            ["seg2", "mat2"],
            TimeRange::new(3_000_000, 1_000_000),
            &one,
            false,
        );
        assert_eq!(refused, Err(EditError::ShapeStyle));
    }

    #[test]
    fn a_shape_goes_on_a_sticker_row_and_takes_its_style() {
        let mut draft = placed(ShapeKind::Rectangle);
        let row = draft
            .tracks
            .iter()
            .find(|track| track.id.as_str() == "row")
            .unwrap();
        assert_eq!(row.kind, TrackType::Sticker);
        assert_eq!(row.segments[0].render_index, 14_000);
        assert_eq!(draft.materials.shapes[0].color, SHAPE_FILL);
        let seg = SegmentId::new("seg");

        set_shape_size(&mut draft, &seg, 640.0, 200.0).unwrap();
        assert_eq!(draft.materials.shapes[0].size(), (640.0, 200.0));
        assert_eq!(draft.materials.shapes[0].points()[1].at, [320.0, 100.0]);
        assert_eq!(
            set_shape_size(&mut draft, &seg, 0.0, 200.0),
            Err(EditError::ShapeStyle)
        );

        set_shape_fill(&mut draft, &seg, "", 0.5).unwrap();
        assert!(draft.materials.shapes[0].color.is_empty());
        set_shape_stroke(&mut draft, &seg, "#ff0000", [6.0, 0.8], 1).unwrap();
        let shape = &draft.materials.shapes[0];
        assert_eq!((shape.border_width, shape.border_line_style), (6.0, 1));
        assert_eq!(
            set_shape_stroke(&mut draft, &seg, "#ff0000", [6.0, 0.8], 7),
            Err(EditError::ShapeStyle)
        );

        set_shape_corners(&mut draft, &seg, &[500.0]).unwrap();
        assert_eq!(draft.materials.shapes[0].roundness, vec![100.0; 4]);
        set_shape_corners(&mut draft, &seg, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(
            draft.materials.shapes[0].roundness,
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            set_shape_corners(&mut draft, &seg, &[1.0, 2.0]),
            Err(EditError::ShapeStyle)
        );

        set_shape_shadow(&mut draft, &seg, "#000000", [10.0, 45.0, 0.5]).unwrap();
        assert_eq!(draft.materials.shapes[0].shadow_alpha, 0.5);
        assert_eq!(
            set_shape_shadow(&mut draft, &seg, "blue", [10.0, 45.0, 0.5]),
            Err(EditError::ShapeStyle)
        );
    }

    #[test]
    fn a_line_keeps_no_height_and_a_shared_shape_is_copied_first() {
        let mut draft = placed(ShapeKind::Line);
        let seg = SegmentId::new("seg");
        set_shape_size(&mut draft, &seg, 500.0, 300.0).unwrap();
        assert_eq!(draft.materials.shapes[0].size(), (500.0, 0.0));

        // A second clip on the same material keeps its own look.
        let row = draft
            .tracks
            .iter()
            .position(|track| track.id.as_str() == "row")
            .unwrap();
        let mut twin = draft.tracks[row].segments[0].clone();
        twin.id = SegmentId::new("twin");
        twin.target = TimeRange::new(5_000_000, 1_000_000);
        draft.tracks[row].segments.push(twin);
        set_shape_stroke(&mut draft, &seg, "#00ff00", [12.0, 1.0], 0).unwrap();
        assert_eq!(draft.materials.shapes.len(), 2);
        assert_eq!(draft.materials.shapes[0].border_width, 8.0);
    }

    #[test]
    fn a_shape_goes_only_on_a_sticker_row() {
        let mut draft = Draft::empty("d");
        let main = draft.tracks[0].id.clone();
        assert_eq!(
            add_shape(
                &mut draft,
                &main,
                &SegmentId::new("seg"),
                &MaterialId::new("mat"),
                TimeRange::new(0, 1_000_000),
                ShapeKind::Circle,
            ),
            Err(EditError::TrackKind)
        );
    }
}
