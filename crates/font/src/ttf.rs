//! TrueType outlines from `loca` and `glyf`.
//!
//! Simple glyphs become lines and quadratic segments. A composite glyph is
//! drawn by placing each component, including its scale and translation.

use crate::{Contour, Directory, FontError, Outline, Segment, bytes_at, i16_at, u16_at, u32_at};

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
const SCALED_COMPONENT_OFFSET: u16 = 0x0800;
const UNSCALED_COMPONENT_OFFSET: u16 = 0x1000;

const MAX_CONTOURS: usize = 256;
const MAX_POINTS: usize = 4_096;
const MAX_COMPONENTS: usize = 64;
const MAX_DEPTH: usize = 16;

pub(crate) struct Source<'a> {
    glyf: &'a [u8],
    loca: Loca<'a>,
}

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

pub(crate) fn outline(
    source: &Source<'_>,
    glyph_id: u16,
    num_glyphs: u16,
) -> Result<Outline, FontError> {
    let mut stack = Vec::new();
    let contours = flatten(source, glyph_id, num_glyphs, &mut stack)?;
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

fn flatten(
    source: &Source<'_>,
    glyph_id: u16,
    num_glyphs: u16,
    stack: &mut Vec<u16>,
) -> Result<Vec<Vec<Point>>, FontError> {
    if glyph_id >= num_glyphs {
        return Err(FontError::BadTable);
    }
    if stack.len() >= MAX_DEPTH || stack.contains(&glyph_id) {
        return Err(FontError::BadTable);
    }
    stack.push(glyph_id);
    let loaded = load_glyph(source, glyph_id, num_glyphs, stack);
    stack.pop();
    loaded
}

fn load_glyph(
    source: &Source<'_>,
    glyph_id: u16,
    num_glyphs: u16,
    stack: &mut Vec<u16>,
) -> Result<Vec<Vec<Point>>, FontError> {
    let (start, end) = source.loca.span(glyph_id)?;
    if start == end {
        return Ok(Vec::new());
    }
    let data = bytes_at(source.glyf, start, end - start)?;
    if data.len() < 10 {
        return Err(FontError::Truncated);
    }
    let contours = i16_at(data, 0)?;
    if contours < 0 {
        let components = parse_composite(data)?;
        place_components(source, &components, num_glyphs, stack)
    } else if contours == 0 {
        Ok(Vec::new())
    } else {
        parse_simple(
            data,
            usize::from(u16::try_from(contours).map_err(|_| FontError::BadTable)?),
        )
    }
}

fn place_components(
    source: &Source<'_>,
    components: &[Component],
    num_glyphs: u16,
    stack: &mut Vec<u16>,
) -> Result<Vec<Vec<Point>>, FontError> {
    let mut placed = Vec::new();
    let mut total = 0usize;
    for component in components {
        let child = flatten(source, component.glyph, num_glyphs, stack)?;
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
    Ok(placed)
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
    let x = component.xx * point.x + component.xy * point.y;
    let y = component.yx * point.x + component.yy * point.y;
    let (x, y) = if component.scaled_offset {
        (
            x + component.xx * dx + component.xy * dy,
            y + component.yx * dx + component.yy * dy,
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
