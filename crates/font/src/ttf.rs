//! TrueType outlines from `loca` and `glyf`.
//!
//! Simple glyphs become lines and quadratic segments. A composite glyph is
//! drawn by placing each component, including its scale and translation.

use crate::gvar::{Gvar, PHANTOMS};
use crate::{
    Contour, Directory, FontError, Hmtx, Outline, Segment, bytes_at, i16_at, u16_at, u32_at,
};

const ON_CURVE: u8 = 0x01;
const X_SHORT: u8 = 0x02;
const Y_SHORT: u8 = 0x04;
const REPEAT: u8 = 0x08;
const X_SAME: u8 = 0x10;
const Y_SAME: u8 = 0x20;

const ARG_WORDS: u16 = 0x0001;
const ARG_XY: u16 = 0x0002;
const ROUND_XY: u16 = 0x0004;
const WE_HAVE_A_SCALE: u16 = 0x0008;
const MORE_COMPONENTS: u16 = 0x0020;
const WE_HAVE_AN_X_AND_Y_SCALE: u16 = 0x0040;
const WE_HAVE_A_TWO_BY_TWO: u16 = 0x0080;
const WE_HAVE_INSTRUCTIONS: u16 = 0x0100;
const USE_MY_METRICS: u16 = 0x0200;
const SCALED_COMPONENT_OFFSET: u16 = 0x0800;
const UNSCALED_COMPONENT_OFFSET: u16 = 0x1000;

// An ornate glyph of a real face has 158 contours and 4,103 points; a
// glyph's points are numbered in 16 bits.
const MAX_CONTOURS: usize = 1_024;
const MAX_POINTS: usize = 16_384;
const MAX_COMPONENTS: usize = 64;
const MAX_DEPTH: usize = 16;

#[derive(Clone)]
pub(crate) struct Source<'a> {
    glyf: &'a [u8],
    loca: Loca<'a>,
}

/// What moves the points of a face set away from its default: the table
/// of moves, the place along each axis, and how much each of the table's
/// shared peaks counts there.
#[derive(Clone, Copy)]
pub(crate) struct Moves<'v, 'a> {
    pub(crate) gvar: &'v Gvar<'a>,
    pub(crate) places: &'v [i16],
    pub(crate) weights: &'v [f32],
}

/// What a glyph is read with: the face's glyph count and metrics, and
/// what moves its points.
#[derive(Clone, Copy)]
pub(crate) struct Reading<'v, 'a> {
    pub(crate) num_glyphs: u16,
    pub(crate) hmtx: &'v Hmtx<'a>,
    pub(crate) moves: Option<Moves<'v, 'a>>,
}

/// A glyph's points by contour, and where its left and right edges are:
/// the left edge is where its origin lies among its points as stored,
/// and the right edge is its advance past that.
struct Loaded {
    contours: Vec<Vec<Point>>,
    left: f32,
    right: f32,
}

#[derive(Clone)]
enum Loca<'a> {
    Short(&'a [u8]),
    Long(&'a [u8]),
}

#[derive(Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
    on_curve: bool,
}

struct Component {
    glyph: u16,
    dx: f32,
    dy: f32,
    xx: f32,
    xy: f32,
    yx: f32,
    yy: f32,
    scaled_offset: bool,
    round_offset: bool,
    matched: Option<(u16, u16)>,
    /// The glyph has this part's advance and bearing.
    metrics: bool,
}

pub(crate) fn source<'a>(
    directory: &Directory<'a>,
    loca_format: i16,
    num_glyphs: u16,
) -> Result<Source<'a>, FontError> {
    let glyf = directory.table(b"glyf")?;
    let loca = directory.table(b"loca")?;
    let entries = usize::from(num_glyphs) + 1;
    let loca = if loca_format == 0 {
        let bytes = entries.checked_mul(2).ok_or(FontError::BadTable)?;
        Loca::Short(bytes_at(loca, 0, bytes)?)
    } else {
        let bytes = entries.checked_mul(4).ok_or(FontError::BadTable)?;
        Loca::Long(bytes_at(loca, 0, bytes)?)
    };
    Ok(Source { glyf, loca })
}

/// The outline of `glyph_id`. A TrueType glyph is drawn with the left edge
/// of its box at the left side bearing its face gives it, and not where
/// its points happen to be stored: an old face hangs a mark back over the
/// letter before it this way. Most glyphs are stored where they are
/// drawn, and are not moved. In a face set away from its default the
/// points move first, and the glyph's origin with them.
pub(crate) fn outline(
    source: &Source<'_>,
    glyph_id: u16,
    reading: &Reading<'_, '_>,
) -> Result<Outline, FontError> {
    let mut stack = Vec::new();
    let Loaded {
        mut contours, left, ..
    } = flatten(source, glyph_id, reading, &mut stack)?;
    if left != 0.0 {
        for point in contours.iter_mut().flatten() {
            point.x -= left;
        }
    }
    let mut outline = Outline {
        contours: Vec::with_capacity(contours.len()),
    };
    for contour in contours {
        if let Some(contour) = to_contour(&contour) {
            outline.contours.push(contour);
        }
    }
    Ok(outline)
}

/// How far `glyph_id` advances in a face set away from its default that
/// has no table of advances for that: the distance between its edges,
/// which move with its points.
pub(crate) fn advance(
    source: &Source<'_>,
    glyph_id: u16,
    reading: &Reading<'_, '_>,
) -> Result<f32, FontError> {
    let mut stack = Vec::new();
    let (left, right) = edges(source, glyph_id, reading, &mut stack)?;
    Ok(crate::vary::nearest(right - left).max(0.0))
}

/// Where a glyph's edges are before anything moves: its origin among its
/// stored points, which is the left edge of its box less its bearing,
/// and its advance past that. A glyph with no points has its origin at 0.
fn stored_edges(reading: &Reading<'_, '_>, glyph_id: u16, x_min: Option<i16>) -> (f32, f32) {
    let left = match (x_min, reading.hmtx.bearing(glyph_id)) {
        (Some(x_min), Some(bearing)) => f32::from(x_min) - f32::from(bearing),
        _ => 0.0,
    };
    let advance = reading.hmtx.advance(glyph_id).map_or(0.0, f32::from);
    (left, left + advance)
}

/// `edges` moved as the face's setting moves them: they are the points
/// after the `own` points of `glyph_id`, and their moves do not depend on
/// where those points are.
fn moved_edges(
    reading: &Reading<'_, '_>,
    glyph_id: u16,
    own: usize,
    edges: (f32, f32),
) -> Result<(f32, f32), FontError> {
    let Some(moves) = &reading.moves else {
        return Ok(edges);
    };
    let mut points = vec![(0.0, 0.0); own + PHANTOMS];
    points[own] = (edges.0, 0.0);
    points[own + 1] = (edges.1, 0.0);
    let moved = moves
        .gvar
        .moves(glyph_id, moves.places, moves.weights, &points, &[])?;
    Ok(match moved {
        Some(moved) => (moved[own].0, moved[own + 1].0),
        None => edges,
    })
}

/// The edges of a glyph alone, without its points.
fn edges(
    source: &Source<'_>,
    glyph_id: u16,
    reading: &Reading<'_, '_>,
    stack: &mut Vec<u16>,
) -> Result<(f32, f32), FontError> {
    if glyph_id >= reading.num_glyphs {
        return Err(FontError::BadTable);
    }
    if stack.len() >= MAX_DEPTH || stack.contains(&glyph_id) {
        return Err(FontError::BadTable);
    }
    let (start, end) = source.loca.span(glyph_id)?;
    let data = bytes_at(source.glyf, start, end - start)?;
    if data.is_empty() {
        return moved_edges(reading, glyph_id, 0, stored_edges(reading, glyph_id, None));
    }
    if data.len() < 10 {
        return Err(FontError::Truncated);
    }
    let contours = i16_at(data, 0)?;
    let stored = stored_edges(reading, glyph_id, Some(i16_at(data, 2)?));
    if contours == 0 {
        return moved_edges(reading, glyph_id, 0, stored);
    }
    if contours > 0 {
        let last = 10 + (usize::from(contours.unsigned_abs()) - 1) * 2;
        let own = usize::from(u16_at(data, last)?) + 1;
        return moved_edges(reading, glyph_id, own, stored);
    }
    let components = parse_composite(data)?;
    let mut found = moved_edges(reading, glyph_id, components.len(), stored)?;
    // A part may lend the glyph its edges.
    stack.push(glyph_id);
    for component in components.iter().filter(|component| component.metrics) {
        match edges(source, component.glyph, reading, stack) {
            Ok(lent) => found = lent,
            Err(error) => {
                stack.pop();
                return Err(error);
            }
        }
    }
    stack.pop();
    Ok(found)
}

fn flatten(
    source: &Source<'_>,
    glyph_id: u16,
    reading: &Reading<'_, '_>,
    stack: &mut Vec<u16>,
) -> Result<Loaded, FontError> {
    if glyph_id >= reading.num_glyphs {
        return Err(FontError::BadTable);
    }
    if stack.len() >= MAX_DEPTH || stack.contains(&glyph_id) {
        return Err(FontError::BadTable);
    }
    stack.push(glyph_id);
    let loaded = load_glyph(source, glyph_id, reading, stack);
    stack.pop();
    loaded
}

fn load_glyph(
    source: &Source<'_>,
    glyph_id: u16,
    reading: &Reading<'_, '_>,
    stack: &mut Vec<u16>,
) -> Result<Loaded, FontError> {
    let (start, end) = source.loca.span(glyph_id)?;
    // A glyph with no points still has edges, and they may move.
    let bare = |stored: (f32, f32)| {
        let (left, right) = moved_edges(reading, glyph_id, 0, stored)?;
        Ok(Loaded {
            contours: Vec::new(),
            left,
            right,
        })
    };
    if start == end {
        return bare(stored_edges(reading, glyph_id, None));
    }
    let data = bytes_at(source.glyf, start, end - start)?;
    if data.len() < 10 {
        return Err(FontError::Truncated);
    }
    let contours = i16_at(data, 0)?;
    let (mut left, mut right) = stored_edges(reading, glyph_id, Some(i16_at(data, 2)?));
    if contours == 0 {
        return bare((left, right));
    }
    if contours < 0 {
        let mut components = parse_composite(data)?;
        if let Some(moves) = &reading.moves {
            // Each part's place is one point of the glyph; then its edges.
            let own = components.len();
            let mut points: Vec<(f32, f32)> = components
                .iter()
                .map(|component| (component.dx, component.dy))
                .collect();
            points.extend([(left, 0.0), (right, 0.0), (0.0, 0.0), (0.0, 0.0)]);
            let moved = moves
                .gvar
                .moves(glyph_id, moves.places, moves.weights, &points, &[])?;
            if let Some(moved) = moved {
                for (component, to) in components.iter_mut().zip(&moved) {
                    if component.matched.is_none() {
                        (component.dx, component.dy) = *to;
                    }
                }
                (left, right) = (moved[own].0, moved[own + 1].0);
            }
        }
        let placed = place_components(source, &components, reading, stack)?;
        // A part may lend the glyph its edges, in a face that moves them.
        if let (Some(lent), true) = (placed.lent, reading.moves.is_some()) {
            (left, right) = lent;
        }
        return Ok(Loaded {
            contours: placed.contours,
            left,
            right,
        });
    }
    let mut contours = parse_simple(
        data,
        usize::from(u16::try_from(contours).map_err(|_| FontError::BadTable)?),
    )?;
    if let Some(moves) = &reading.moves {
        let mut points = Vec::new();
        let mut ends = Vec::with_capacity(contours.len());
        for contour in &contours {
            points.extend(contour.iter().map(|point| (point.x, point.y)));
            ends.push(points.len().saturating_sub(1));
        }
        let own = points.len();
        points.extend([(left, 0.0), (right, 0.0), (0.0, 0.0), (0.0, 0.0)]);
        let moved = moves
            .gvar
            .moves(glyph_id, moves.places, moves.weights, &points, &ends)?;
        if let Some(moved) = moved {
            for (point, to) in contours.iter_mut().flatten().zip(&moved) {
                (point.x, point.y) = *to;
            }
            (left, right) = (moved[own].0, moved[own + 1].0);
        }
    }
    Ok(Loaded {
        contours,
        left,
        right,
    })
}

/// The points of a composite's parts, each part placed, and the edges of
/// the last part that lends the glyph its own.
struct Placed {
    contours: Vec<Vec<Point>>,
    lent: Option<(f32, f32)>,
}

fn place_components(
    source: &Source<'_>,
    components: &[Component],
    reading: &Reading<'_, '_>,
    stack: &mut Vec<u16>,
) -> Result<Placed, FontError> {
    let mut placed = Vec::new();
    let mut total = 0usize;
    let mut lent = None;
    for component in components {
        let child = flatten(source, component.glyph, reading, stack)?;
        if component.metrics {
            lent = Some((child.left, child.right));
        }
        let child = child.contours;
        let (dx, dy) = translation(component, &placed, &child)?;
        for contour in child {
            let mut next = Vec::with_capacity(contour.len());
            for point in contour {
                next.push(transform(point, component, dx, dy));
            }
            total = total.checked_add(next.len()).ok_or(FontError::BadTable)?;
            if total > MAX_POINTS {
                return Err(FontError::BadTable);
            }
            placed.push(next);
        }
    }
    Ok(Placed {
        contours: placed,
        lent,
    })
}

fn translation(
    component: &Component,
    parent: &[Vec<Point>],
    child: &[Vec<Point>],
) -> Result<(f32, f32), FontError> {
    let Some((parent_index, child_index)) = component.matched else {
        return Ok((component.dx, component.dy));
    };
    let parent_point = point_at(parent, parent_index).ok_or(FontError::BadTable)?;
    let child_point = point_at(child, child_index).ok_or(FontError::BadTable)?;
    let mut dx = parent_point.x - child_point.x;
    let mut dy = parent_point.y - child_point.y;
    if component.round_offset {
        dx = dx.round();
        dy = dy.round();
    }
    Ok((dx, dy))
}

fn point_at(contours: &[Vec<Point>], index: u16) -> Option<Point> {
    let mut remaining = usize::from(index);
    for contour in contours {
        if remaining < contour.len() {
            return contour.get(remaining).copied();
        }
        remaining -= contour.len();
    }
    None
}

fn transform(point: Point, component: &Component, dx: f32, dy: f32) -> Point {
    // The four numbers of a part's matrix are stored a column at a time:
    // what x gives to x and to y, then what y gives to x and to y.
    let x = component.xx * point.x + component.yx * point.y;
    let y = component.xy * point.x + component.yy * point.y;
    let (x, y) = if component.scaled_offset {
        (
            x + component.xx * dx + component.yx * dy,
            y + component.xy * dx + component.yy * dy,
        )
    } else {
        (x + dx, y + dy)
    };
    Point {
        x,
        y,
        on_curve: point.on_curve,
    }
}

fn parse_simple(data: &[u8], contour_count: usize) -> Result<Vec<Vec<Point>>, FontError> {
    if contour_count > MAX_CONTOURS {
        return Err(FontError::BadTable);
    }
    let mut pos = 10;
    let mut ends = Vec::with_capacity(contour_count);
    let mut previous = -1i32;
    for _ in 0..contour_count {
        let end = i32::from(u16_at(data, pos)?);
        pos += 2;
        if end <= previous {
            return Err(FontError::BadTable);
        }
        previous = end;
        ends.push(end);
    }
    let point_count = usize::try_from(previous + 1).map_err(|_| FontError::BadTable)?;
    if point_count > MAX_POINTS {
        return Err(FontError::BadTable);
    }
    let instructions = usize::from(u16_at(data, pos)?);
    pos += 2;
    pos = pos.checked_add(instructions).ok_or(FontError::BadTable)?;
    if pos > data.len() {
        return Err(FontError::Truncated);
    }
    let flags = read_flags(data, &mut pos, point_count)?;
    let xs = read_coords(data, &mut pos, &flags, X_SHORT, X_SAME)?;
    let ys = read_coords(data, &mut pos, &flags, Y_SHORT, Y_SAME)?;
    let mut contours = Vec::with_capacity(contour_count);
    let mut start = 0usize;
    for end in ends {
        let end = usize::try_from(end).map_err(|_| FontError::BadTable)? + 1;
        let mut contour = Vec::with_capacity(end - start);
        for index in start..end {
            let (Some(x), Some(y), Some(flag)) = (xs.get(index), ys.get(index), flags.get(index))
            else {
                return Err(FontError::BadTable);
            };
            contour.push(Point {
                x: *x as f32,
                y: *y as f32,
                on_curve: flag & ON_CURVE != 0,
            });
        }
        contours.push(contour);
        start = end;
    }
    Ok(contours)
}

fn read_flags(data: &[u8], pos: &mut usize, count: usize) -> Result<Vec<u8>, FontError> {
    let mut flags = Vec::with_capacity(count);
    while flags.len() < count {
        let flag = *data.get(*pos).ok_or(FontError::Truncated)?;
        *pos += 1;
        let repeat = if flag & REPEAT == 0 {
            0
        } else {
            let extra = *data.get(*pos).ok_or(FontError::Truncated)?;
            *pos += 1;
            usize::from(extra)
        };
        let copies = repeat.checked_add(1).ok_or(FontError::BadTable)?;
        if flags.len() + copies > count {
            return Err(FontError::BadTable);
        }
        flags.extend(std::iter::repeat_n(flag, copies));
    }
    Ok(flags)
}

fn read_coords(
    data: &[u8],
    pos: &mut usize,
    flags: &[u8],
    short_bit: u8,
    same_bit: u8,
) -> Result<Vec<i32>, FontError> {
    let mut coords = Vec::with_capacity(flags.len());
    let mut cursor = 0i32;
    for flag in flags {
        let delta = if flag & short_bit != 0 {
            let magnitude = i32::from(*data.get(*pos).ok_or(FontError::Truncated)?);
            *pos += 1;
            if flag & same_bit == 0 {
                -magnitude
            } else {
                magnitude
            }
        } else if flag & same_bit != 0 {
            0
        } else {
            let delta = i32::from(i16_at(data, *pos)?);
            *pos += 2;
            delta
        };
        cursor = cursor.checked_add(delta).ok_or(FontError::BadTable)?;
        coords.push(cursor);
    }
    Ok(coords)
}

fn parse_composite(data: &[u8]) -> Result<Vec<Component>, FontError> {
    let mut pos = 10;
    let mut components = Vec::new();
    loop {
        if components.len() >= MAX_COMPONENTS {
            return Err(FontError::BadTable);
        }
        let flags = u16_at(data, pos)?;
        let glyph = u16_at(data, pos + 2)?;
        pos += 4;
        let args = read_args(data, pos, flags)?;
        pos += args.bytes;
        let (xx, xy, yx, yy, used) = read_scale(data, pos, flags)?;
        pos += used;
        let dx = if flags & ROUND_XY == 0 {
            args.dx
        } else {
            args.dx.round()
        };
        let dy = if flags & ROUND_XY == 0 {
            args.dy
        } else {
            args.dy.round()
        };
        let scaled_offset =
            flags & SCALED_COMPONENT_OFFSET != 0 && flags & UNSCALED_COMPONENT_OFFSET == 0;
        components.push(Component {
            glyph,
            dx,
            dy,
            xx,
            xy,
            yx,
            yy,
            scaled_offset,
            round_offset: flags & ROUND_XY != 0,
            matched: args.matched,
            metrics: flags & USE_MY_METRICS != 0,
        });
        if flags & MORE_COMPONENTS == 0 {
            if flags & WE_HAVE_INSTRUCTIONS != 0 {
                let length = usize::from(u16_at(data, pos)?);
                let end = pos.checked_add(2 + length).ok_or(FontError::BadTable)?;
                if end > data.len() {
                    return Err(FontError::Truncated);
                }
            }
            break;
        }
    }
    Ok(components)
}

struct Args {
    matched: Option<(u16, u16)>,
    dx: f32,
    dy: f32,
    bytes: usize,
}

fn read_args(data: &[u8], pos: usize, flags: u16) -> Result<Args, FontError> {
    if flags & ARG_WORDS != 0 {
        if flags & ARG_XY != 0 {
            Ok(Args {
                matched: None,
                dx: f32::from(i16_at(data, pos)?),
                dy: f32::from(i16_at(data, pos + 2)?),
                bytes: 4,
            })
        } else {
            Ok(Args {
                matched: Some((u16_at(data, pos)?, u16_at(data, pos + 2)?)),
                dx: 0.0,
                dy: 0.0,
                bytes: 4,
            })
        }
    } else if flags & ARG_XY != 0 {
        Ok(Args {
            matched: None,
            dx: f32::from(i8_at(data, pos)?),
            dy: f32::from(i8_at(data, pos + 1)?),
            bytes: 2,
        })
    } else {
        let first = u16::from(*data.get(pos).ok_or(FontError::Truncated)?);
        let second = u16::from(*data.get(pos + 1).ok_or(FontError::Truncated)?);
        Ok(Args {
            matched: Some((first, second)),
            dx: 0.0,
            dy: 0.0,
            bytes: 2,
        })
    }
}

fn read_scale(
    data: &[u8],
    pos: usize,
    flags: u16,
) -> Result<(f32, f32, f32, f32, usize), FontError> {
    if flags & WE_HAVE_A_SCALE != 0 {
        let scale = f2dot14(i16_at(data, pos)?);
        Ok((scale, 0.0, 0.0, scale, 2))
    } else if flags & WE_HAVE_AN_X_AND_Y_SCALE != 0 {
        let xx = f2dot14(i16_at(data, pos)?);
        let yy = f2dot14(i16_at(data, pos + 2)?);
        Ok((xx, 0.0, 0.0, yy, 4))
    } else if flags & WE_HAVE_A_TWO_BY_TWO != 0 {
        let xx = f2dot14(i16_at(data, pos)?);
        let xy = f2dot14(i16_at(data, pos + 2)?);
        let yx = f2dot14(i16_at(data, pos + 4)?);
        let yy = f2dot14(i16_at(data, pos + 6)?);
        Ok((xx, xy, yx, yy, 8))
    } else {
        Ok((1.0, 0.0, 0.0, 1.0, 0))
    }
}

fn f2dot14(value: i16) -> f32 {
    f32::from(value) / 16_384.0
}

fn i8_at(data: &[u8], offset: usize) -> Result<i8, FontError> {
    let byte = *data.get(offset).ok_or(FontError::Truncated)?;
    Ok(i8::from_be_bytes([byte]))
}

impl Loca<'_> {
    fn span(&self, glyph: u16) -> Result<(usize, usize), FontError> {
        let start = self.offset(glyph)?;
        let end = self.offset(glyph + 1)?;
        if end < start {
            return Err(FontError::BadTable);
        }
        Ok((start, end))
    }

    fn offset(&self, index: u16) -> Result<usize, FontError> {
        let at = usize::from(index);
        let offset = match self {
            Self::Short(data) => u32::from(u16_at(data, at * 2)?) * 2,
            Self::Long(data) => u32_at(data, at * 4)?,
        };
        usize::try_from(offset).map_err(|_| FontError::BadTable)
    }
}

fn to_contour(points: &[Point]) -> Option<Contour> {
    if points.is_empty() {
        return None;
    }
    if points.iter().all(|point| !point.on_curve) {
        return Some(off_curve_contour(points));
    }
    let start_at = points.iter().position(|point| point.on_curve).unwrap_or(0);
    let mut ordered = Vec::with_capacity(points.len());
    ordered.extend_from_slice(&points[start_at..]);
    ordered.extend_from_slice(&points[..start_at]);
    let start_x = ordered[0].x;
    let start_y = ordered[0].y;
    let mut segments = Vec::new();
    let mut index = 1;
    while index < ordered.len() {
        if ordered[index].on_curve {
            segments.push(Segment::Line {
                x: ordered[index].x,
                y: ordered[index].y,
            });
            index += 1;
            continue;
        }
        let control = ordered[index];
        let next = index + 1;
        if next < ordered.len() && ordered[next].on_curve {
            segments.push(Segment::Quad {
                cx: control.x,
                cy: control.y,
                x: ordered[next].x,
                y: ordered[next].y,
            });
            index += 2;
        } else if next < ordered.len() {
            let end = midpoint(control, ordered[next]);
            segments.push(Segment::Quad {
                cx: control.x,
                cy: control.y,
                x: end.0,
                y: end.1,
            });
            index += 1;
        } else {
            segments.push(Segment::Quad {
                cx: control.x,
                cy: control.y,
                x: start_x,
                y: start_y,
            });
            index += 1;
        }
    }
    close_to_start(&mut segments, start_x, start_y);
    Some(Contour {
        x: start_x,
        y: start_y,
        segments,
    })
}

fn off_curve_contour(points: &[Point]) -> Contour {
    let start = midpoint(points[points.len() - 1], points[0]);
    let mut segments = Vec::with_capacity(points.len());
    for index in 0..points.len() {
        let control = points[index];
        let next = if index + 1 == points.len() {
            points[0]
        } else {
            points[index + 1]
        };
        let end = midpoint(control, next);
        segments.push(Segment::Quad {
            cx: control.x,
            cy: control.y,
            x: end.0,
            y: end.1,
        });
    }
    Contour {
        x: start.0,
        y: start.1,
        segments,
    }
}

fn close_to_start(segments: &mut Vec<Segment>, x: f32, y: f32) {
    let closed = match segments.last() {
        Some(Segment::Line { x: end_x, y: end_y })
        | Some(Segment::Quad {
            x: end_x, y: end_y, ..
        })
        | Some(Segment::Cubic {
            x: end_x, y: end_y, ..
        }) => *end_x == x && *end_y == y,
        None => false,
    };
    if !closed {
        segments.push(Segment::Line { x, y });
    }
}

fn midpoint(left: Point, right: Point) -> (f32, f32) {
    ((left.x + right.x) / 2.0, (left.y + right.y) / 2.0)
}

#[cfg(test)]
mod tests {
    use crate::fixture::{be, boxed, font, signed};
    use crate::{FontError, read};

    /// A face whose glyph 1 is one contour of `points` points, each one
    /// unit right and one up from the last.
    fn many_points(points: u16) -> Vec<u8> {
        let mut glyph = vec![0, 1, 0, 0, 0, 0];
        glyph.extend(points.to_be_bytes());
        glyph.extend(points.to_be_bytes());
        glyph.extend((points - 1).to_be_bytes());
        glyph.extend([0, 0]);
        let count = usize::from(points);
        // On the curve, each delta one positive byte.
        glyph.extend(std::iter::repeat_n(0x37, count));
        glyph.extend(std::iter::repeat_n(1, count * 2));
        let mut loca = vec![0; 8];
        loca.extend(u32::try_from(glyph.len()).expect("length").to_be_bytes());
        let glyphs = [boxed(500), boxed(500)];
        font(&glyphs, &[('A', 1)], &[(b"glyf", glyph), (b"loca", loca)])
    }

    /// The left and right edges of a glyph's outline.
    fn edges(bytes: &[u8], glyph: u16) -> (f32, f32) {
        let face = read(bytes).expect("face");
        let outline = face.outline(glyph).expect("outline");
        let mut xs: Vec<f32> = Vec::new();
        for contour in &outline.contours {
            xs.push(contour.x);
            for segment in &contour.segments {
                if let crate::Segment::Line { x, .. } = *segment {
                    xs.push(x);
                }
            }
        }
        (
            xs.iter().copied().fold(f32::MAX, f32::min),
            xs.iter().copied().fold(f32::MIN, f32::max),
        )
    }

    #[test]
    fn a_glyph_is_drawn_with_its_left_edge_at_its_bearing() {
        // Glyph 1 is stored from 50 to 450. Its metrics say its left
        // edge is 300 before its origin, as an old face's accent is, to
        // hang over the letter before it; glyph 2's say what is stored.
        let glyphs = [boxed(500), boxed(500), boxed(500)];
        let metrics = |first: i16, second: i16| {
            let mut hmtx = Vec::new();
            for bearing in [0, first, second] {
                hmtx.extend(500_u16.to_be_bytes());
                hmtx.extend(bearing.to_be_bytes());
            }
            hmtx
        };
        let bytes = font(&glyphs, &[('A', 1)], &[(b"hmtx", metrics(-300, 50))]);
        assert_eq!(edges(&bytes, 1), (-300.0, 100.0));
        assert_eq!(edges(&bytes, 2), (50.0, 450.0));
        // Forward as well as back, and the advance is what it was.
        let bytes = font(&glyphs, &[('A', 1)], &[(b"hmtx", metrics(250, 50))]);
        assert_eq!(edges(&bytes, 1), (250.0, 650.0));
        assert_eq!(read(&bytes).expect("face").advance(1), Ok(500));
        // A face whose metrics stop before a glyph leaves it as stored.
        let short: Vec<u8> = metrics(-300, 50)[..8].to_vec();
        let mut with = font(&glyphs, &[('A', 1)], &[(b"hmtx", short)]);
        // Two glyphs have metrics of their own; the third has neither an
        // advance nor a bearing in the table.
        let hhea = (0..usize::from(u16::from_be_bytes([with[4], with[5]])))
            .map(|record| 12 + record * 16)
            .find(|at| &with[*at..*at + 4] == b"hhea")
            .expect("hhea");
        let at = usize::try_from(u32::from_be_bytes([
            with[hhea + 8],
            with[hhea + 9],
            with[hhea + 10],
            with[hhea + 11],
        ]))
        .expect("offset");
        with[at + 34..at + 36].copy_from_slice(&2_u16.to_be_bytes());
        if let Ok(face) = read(&with) {
            assert!(face.outline(2).is_ok());
        }
    }

    #[test]
    fn an_ornate_glyph_of_thousands_of_points_is_outlined() {
        // A real face has a glyph of 4,103 points.
        let bytes = many_points(5_000);
        let face = read(&bytes).expect("face");
        let outline = face.outline(1).expect("outline");
        assert_eq!(outline.contours.len(), 1);
        assert!(outline.contours[0].segments.len() >= 4_999);
        // Past the limit a glyph is refused, and the face still opens.
        let bytes = many_points(20_000);
        let face = read(&bytes).expect("face");
        assert_eq!(face.outline(1), Err(FontError::BadTable));
        assert!(face.outline(0).is_ok());
    }

    /// A face whose glyph 2 is glyph 1, a box from 50 to 450 across and 0
    /// to 700 up, through the matrix `matrix` and moved 1000 right and
    /// 2000 up.
    pub(crate) fn turned(matrix: [i16; 4]) -> Vec<u8> {
        let boxes = font(&[boxed(500), boxed(500)], &[('A', 1)], &[]);
        // The two boxes' records, as the builder writes them.
        let mut record = be(&[1, 50, 0, 450, 700, 3, 0]);
        record.extend([1, 1, 1, 1]);
        record.extend(be(&[50, 400, 0, signed(-400), 0, 0, 700, 0]));
        assert!(boxes.windows(record.len()).any(|window| window == record));
        let mut glyf = record.clone();
        glyf.extend(&record);
        let mut loca = Vec::new();
        for at in [0, record.len(), record.len() * 2] {
            loca.extend(u32::try_from(at).expect("loca").to_be_bytes());
        }
        // One part, with its place in words and a matrix of four numbers.
        glyf.extend(be(&[signed(-1), 0, 0, 0, 0, 0x0083, 1, 1000, 2000]));
        glyf.extend(be(&matrix.map(signed)));
        loca.extend(u32::try_from(glyf.len()).expect("loca").to_be_bytes());
        let third = crate::fixture::Glyph {
            advance: 500,
            outline: Some([0, 0, 0, 0]),
        };
        font(
            &[boxed(500), boxed(500), third],
            &[('A', 1), ('B', 2)],
            &[(b"glyf", glyf), (b"loca", loca)],
        )
    }

    /// The corners of a glyph's outline: left, bottom, right, top.
    fn corners(bytes: &[u8], glyph: u16) -> [f32; 4] {
        let face = read(bytes).expect("face");
        let outline = face.outline(glyph).expect("outline");
        let mut corners = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        let mut add = |x: f32, y: f32| {
            corners = [
                corners[0].min(x),
                corners[1].min(y),
                corners[2].max(x),
                corners[3].max(y),
            ];
        };
        for contour in &outline.contours {
            add(contour.x, contour.y);
            for segment in &contour.segments {
                if let crate::Segment::Line { x, y } = *segment {
                    add(x, y);
                }
            }
        }
        corners
    }

    #[test]
    fn a_part_is_turned_by_its_matrix_a_column_at_a_time() {
        // Each box is what `hb-vector` draws for the face.
        // A quarter turn: x goes to y, and y to x backwards.
        let one = f2dot14_one();
        assert_eq!(
            corners(&turned([0, one, -one, 0]), 2),
            [300.0, 2050.0, 1000.0, 2450.0]
        );
        // The other way round.
        assert_eq!(
            corners(&turned([0, -one, one, 0]), 2),
            [1000.0, 1550.0, 1700.0, 1950.0]
        );
        // Sheared: each unit up moves half a unit right.
        assert_eq!(
            corners(&turned([one, 0, one / 2, one]), 2),
            [1050.0, 2000.0, 1800.0, 2700.0]
        );
        // No turn at all.
        assert_eq!(
            corners(&turned([one, 0, 0, one]), 2),
            [1050.0, 2000.0, 1450.0, 2700.0]
        );
    }

    fn f2dot14_one() -> i16 {
        16_384
    }
}
