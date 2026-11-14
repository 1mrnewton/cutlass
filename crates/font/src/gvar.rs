//! `gvar`: how the points of each TrueType glyph move along a face's axes.
//!
//! A glyph has a list of tuples. Each is drawn for one region of the
//! design space and holds a move for some of the glyph's points, or for
//! all of them; a point of an outline that a tuple leaves out moves with
//! the points before and after it on its contour. The moves of every
//! tuple, each weighted by how near the face is set to the tuple's
//! region, add up. After a glyph's own points come four more that stand
//! for its left and right edges and its top and bottom: where they move,
//! the glyph's advance and bearing follow.

use crate::{FontError, bytes_at, u16_at, u32_at};

/// A glyph has this many points after its own: its edges.
pub(crate) const PHANTOMS: usize = 4;

/// The most points one glyph may be given moves for: what a glyph may
/// have, and its edges.
const MAX_MOVED: usize = 65_536 + PHANTOMS;

const SHARED_POINTS: u16 = 0x8000;
const COUNT_MASK: u16 = 0x0FFF;
const EMBEDDED_PEAK: u16 = 0x8000;
const INTERMEDIATE: u16 = 0x4000;
const PRIVATE_POINTS: u16 = 0x2000;
const TUPLE_INDEX: u16 = 0x0FFF;

/// The `gvar` table of a face.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gvar<'a> {
    /// Each glyph's variation data starts at an offset from here.
    array: &'a [u8],
    /// An offset for each glyph, and one more for the end of the last.
    offsets: &'a [u8],
    long: bool,
    /// Peaks many glyphs' tuples share: a place for each axis.
    shared: &'a [u8],
    axes: usize,
}

impl<'a> Gvar<'a> {
    /// Read the table of a face of `axes` axes.
    pub(crate) fn read(data: &'a [u8], axes: usize) -> Result<Self, FontError> {
        if u16_at(data, 0)? != 1 {
            return Err(FontError::Unsupported);
        }
        if usize::from(u16_at(data, 4)?) != axes {
            return Err(FontError::BadTable);
        }
        let offset =
            |at: usize| usize::try_from(u32_at(data, at)?).map_err(|_| FontError::BadTable);
        let shared_count = usize::from(u16_at(data, 6)?);
        let shared = bytes_at(data, offset(8)?, shared_count * axes * 2)?;
        let glyphs = usize::from(u16_at(data, 12)?);
        let long = u16_at(data, 14)? & 1 != 0;
        let entry = if long { 4 } else { 2 };
        Ok(Self {
            array: data.get(offset(16)?..).ok_or(FontError::Truncated)?,
            offsets: bytes_at(data, 20, (glyphs + 1) * entry)?,
            long,
            shared,
            axes,
        })
    }

    /// The variation data of `glyph`; none for a glyph the table lacks or
    /// one that does not vary.
    fn glyph(&self, glyph: u16) -> Result<&'a [u8], FontError> {
        let index = usize::from(glyph);
        let entry = if self.long { 4 } else { 2 };
        if (index + 2) * entry > self.offsets.len() {
            return Ok(&[]);
        }
        let read = |index: usize| -> Result<usize, FontError> {
            if self.long {
                usize::try_from(u32_at(self.offsets, index * 4)?).map_err(|_| FontError::BadTable)
            } else {
                Ok(usize::from(u16_at(self.offsets, index * 2)?) * 2)
            }
        };
        let (start, end) = (read(index)?, read(index + 1)?);
        if end < start {
            return Err(FontError::BadTable);
        }
        bytes_at(self.array, start, end - start)
    }

    /// How much each shared peak counts at `places`, to reuse for every
    /// glyph.
    pub(crate) fn weights(&self, places: &[i16]) -> Vec<f32> {
        if self.axes == 0 {
            return Vec::new();
        }
        self.shared
            .chunks_exact(self.axes * 2)
            .map(|peaks| weight(peaks, None, places))
            .collect()
    }

    /// Where each of `points` is at `places`: a glyph's own points, or a
    /// composite's components, and then its four edges. `ends` is the
    /// last point of each contour of an outline, and empty for a
    /// composite, whose parts each move alone. `weights` is from
    /// [`Self::weights`]. `None` when nothing moves. Each tuple's moves
    /// are added to the points in turn, as other readers add them, so a
    /// sum that lands on a half rounds the same way.
    pub(crate) fn moves(
        &self,
        glyph: u16,
        places: &[i16],
        weights: &[f32],
        points: &[(f32, f32)],
        ends: &[usize],
    ) -> Result<Option<Vec<(f32, f32)>>, FontError> {
        let data = self.glyph(glyph)?;
        if data.is_empty() {
            return Ok(None);
        }
        if points.len() > MAX_MOVED {
            return Err(FontError::BadTable);
        }
        let header = u16_at(data, 0)?;
        let count = usize::from(header & COUNT_MASK);
        let mut at = usize::from(u16_at(data, 2)?);
        let shared_points = if header & SHARED_POINTS == 0 {
            None
        } else {
            Some(point_numbers(data, &mut at)?)
        };
        let mut total: Option<Vec<(f32, f32)>> = None;
        let mut touched = Vec::new();
        let mut own = Vec::new();
        let mut head = 4;
        for _ in 0..count {
            let size = usize::from(u16_at(data, head)?);
            let index = u16_at(data, head + 2)?;
            head += 4;
            let peaks = if index & EMBEDDED_PEAK == 0 {
                None
            } else {
                let peaks = bytes_at(data, head, self.axes * 2)?;
                head += self.axes * 2;
                Some(peaks)
            };
            let between = if index & INTERMEDIATE == 0 {
                None
            } else {
                let between = bytes_at(data, head, self.axes * 4)?;
                head += self.axes * 4;
                Some(between)
            };
            let tuple = bytes_at(data, at, size)?;
            at += size;
            let weight = match (peaks, between) {
                (Some(peaks), _) => weight(peaks, between, places),
                // A shared peak the table lacks counts nothing.
                (None, None) => weights
                    .get(usize::from(index & TUPLE_INDEX))
                    .copied()
                    .unwrap_or(0.0),
                (None, Some(_)) => {
                    let shared = usize::from(index & TUPLE_INDEX) * self.axes * 2;
                    self.shared
                        .get(shared..shared + self.axes * 2)
                        .map_or(0.0, |peaks| weight(peaks, between, places))
                }
            };
            if weight == 0.0 {
                continue;
            }
            let mut cursor = 0;
            let private = if index & PRIVATE_POINTS == 0 {
                None
            } else {
                Some(point_numbers(tuple, &mut cursor)?)
            };
            // A tuple with no numbers of its own moves the points the
            // glyph's tuples share; and a list of no numbers is every
            // point.
            let numbers = match (&private, &shared_points) {
                (Some(numbers), _) | (None, Some(numbers)) => numbers.as_deref(),
                (None, None) => None,
            };
            let moved = numbers.map_or(points.len(), <[u16]>::len);
            let xs = deltas(tuple, &mut cursor, moved)?;
            let ys = deltas(tuple, &mut cursor, moved)?;
            let total = total.get_or_insert_with(|| points.to_vec());
            let Some(numbers) = numbers else {
                for ((sum, x), y) in total.iter_mut().zip(&xs).zip(&ys) {
                    sum.0 += scaled(*x, weight);
                    sum.1 += scaled(*y, weight);
                }
                continue;
            };
            own.clear();
            own.resize(points.len(), (0.0_f32, 0.0_f32));
            touched.clear();
            touched.resize(points.len(), false);
            for ((number, x), y) in numbers.iter().zip(&xs).zip(&ys) {
                let number = usize::from(*number);
                if let (Some(slot), Some(flag)) = (own.get_mut(number), touched.get_mut(number)) {
                    slot.0 += scaled(*x, weight);
                    slot.1 += scaled(*y, weight);
                    *flag = true;
                }
            }
            infer(points, ends, &touched, &mut own);
            for (sum, one) in total.iter_mut().zip(&own) {
                sum.0 += one.0;
                sum.1 += one.1;
            }
        }
        Ok(total)
    }
}

fn scaled(delta: i32, weight: f32) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let delta = delta as f32;
    delta * weight
}

/// How much a tuple counts at `places`. `peaks` holds a place for each
/// axis; `between`, when the tuple has it, a start for each axis and then
/// an end for each.
fn weight(peaks: &[u8], between: Option<&[u8]>, places: &[i16]) -> f32 {
    let read = |data: &[u8], index: usize| {
        data.get(index * 2..index * 2 + 2).map_or(0, |bytes| {
            i32::from(i16::from_be_bytes([bytes[0], bytes[1]]))
        })
    };
    let axes = peaks.len() / 2;
    let mut total = 1.0_f64;
    for axis in 0..axes {
        let peak = read(peaks, axis);
        if peak == 0 {
            continue;
        }
        let place = i32::from(places.get(axis).copied().unwrap_or(0));
        if place == 0 {
            return 0.0;
        }
        if place == peak {
            continue;
        }
        if let Some(between) = between {
            let (start, end) = (read(between, axis), read(between, axes + axis));
            if start > peak || peak > end || (start < 0 && end > 0) {
                continue;
            }
            if place < start || place > end {
                return 0.0;
            }
            if place < peak {
                if peak != start {
                    total *= f64::from(place - start) / f64::from(peak - start);
                }
            } else if peak != end {
                total *= f64::from(end - place) / f64::from(end - peak);
            }
        } else if place < peak.min(0) || place > peak.max(0) {
            return 0.0;
        } else {
            total *= f64::from(place) / f64::from(peak);
        }
    }
    #[allow(clippy::cast_possible_truncation)]
    let total = total as f32;
    total
}

/// The packed point numbers at `at`, which is moved past them: `None`
/// for a list that stands for every point.
fn point_numbers(data: &[u8], at: &mut usize) -> Result<Option<Vec<u16>>, FontError> {
    let mut byte = || {
        let value = *data.get(*at).ok_or(FontError::Truncated)?;
        *at += 1;
        Ok::<u8, FontError>(value)
    };
    let first = byte()?;
    if first == 0 {
        return Ok(None);
    }
    let count = if first & 0x80 == 0 {
        usize::from(first)
    } else {
        usize::from(first & 0x7F) << 8 | usize::from(byte()?)
    };
    let mut numbers = Vec::with_capacity(count);
    let mut number = 0_u16;
    while numbers.len() < count {
        let control = byte()?;
        let run = usize::from(control & 0x7F) + 1;
        if numbers.len() + run > count {
            return Err(FontError::BadTable);
        }
        for _ in 0..run {
            let step = if control & 0x80 == 0 {
                u16::from(byte()?)
            } else {
                u16::from(byte()?) << 8 | u16::from(byte()?)
            };
            // Each number is stored as its distance from the one before.
            number = number.wrapping_add(step);
            numbers.push(number);
        }
    }
    Ok(Some(numbers))
}

/// `count` packed deltas at `at`, which is moved past them.
fn deltas(data: &[u8], at: &mut usize, count: usize) -> Result<Vec<i32>, FontError> {
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        let control = *data.get(*at).ok_or(FontError::Truncated)?;
        *at += 1;
        let run = usize::from(control & 0x3F) + 1;
        if values.len() + run > count {
            return Err(FontError::BadTable);
        }
        // A run of zeros stores nothing; the others a byte, a word or
        // four bytes for each delta.
        let size = match control & 0xC0 {
            0x80 => {
                values.resize(values.len() + run, 0);
                continue;
            }
            0x00 => 1,
            0x40 => 2,
            _ => 4,
        };
        let bytes = bytes_at(data, *at, run * size)?;
        *at += run * size;
        for one in bytes.chunks_exact(size) {
            values.push(match size {
                1 => i32::from(i8::from_be_bytes([one[0]])),
                2 => i32::from(i16::from_be_bytes([one[0], one[1]])),
                _ => i32::from_be_bytes([one[0], one[1], one[2], one[3]]),
            });
        }
    }
    Ok(values)
}

/// Give each point of an outline that a tuple did not move the move its
/// neighbours on the contour imply. A contour none of whose points moved,
/// or all, is left as it is.
fn infer(points: &[(f32, f32)], ends: &[usize], touched: &[bool], moves: &mut [(f32, f32)]) {
    let mut start = 0;
    for end in ends {
        let end = *end;
        if end < start || end >= points.len() || end >= touched.len() {
            return;
        }
        let length = end - start + 1;
        let moved = touched[start..=end].iter().filter(|flag| **flag).count();
        if moved != 0 && moved != length {
            contour(points, touched, moves, start, end);
        }
        start = end + 1;
    }
}

/// One contour from `start` to `end`, some of whose points moved.
fn contour(
    points: &[(f32, f32)],
    touched: &[bool],
    moves: &mut [(f32, f32)],
    start: usize,
    end: usize,
) {
    let after = |index: usize| if index >= end { start } else { index + 1 };
    let Some(first) = (start..=end).find(|index| touched[*index]) else {
        return;
    };
    // Walk the contour from one moved point round to itself: each run of
    // points between two moved ones follows those two.
    let mut before = first;
    loop {
        let mut next = after(before);
        while !touched[next] {
            next = after(next);
        }
        let mut index = after(before);
        while index != next {
            moves[index] = (
                between(
                    points[index].0,
                    points[before].0,
                    points[next].0,
                    moves[before].0,
                    moves[next].0,
                ),
                between(
                    points[index].1,
                    points[before].1,
                    points[next].1,
                    moves[before].1,
                    moves[next].1,
                ),
            );
            index = after(index);
        }
        before = next;
        if before == first {
            return;
        }
    }
}

/// The move of a point at `at` between two moved points at `one` and
/// `other`: the nearer one's move when it lies outside them, and a mix of
/// the two by where it lies between them.
fn between(at: f32, one: f32, other: f32, one_move: f32, other_move: f32) -> f32 {
    if one - other == 0.0 {
        return if one_move - other_move == 0.0 {
            one_move
        } else {
            0.0
        };
    }
    if at <= one.min(other) {
        return if one < other { one_move } else { other_move };
    }
    if at >= one.max(other) {
        return if one > other { one_move } else { other_move };
    }
    let along = (at - one) / (other - one);
    one_move + along * (other_move - one_move)
}

#[cfg(test)]
mod tests {
    use super::{between, deltas, point_numbers};
    use crate::FontError;

    #[test]
    fn packed_point_numbers_are_runs_of_steps() {
        // Four numbers: a run of two bytes, then a run of two words.
        let data = [4, 0x01, 1, 2, 0x81, 0x01, 0x00, 0x00, 0x05];
        let mut at = 0;
        assert_eq!(
            point_numbers(&data, &mut at),
            Ok(Some(vec![1, 3, 259, 264]))
        );
        assert_eq!(at, data.len());
        // No numbers is every point.
        let mut at = 0;
        assert_eq!(point_numbers(&[0, 9], &mut at), Ok(None));
        assert_eq!(at, 1);
        // A count in two bytes.
        let mut data = vec![0x81, 0x00];
        for _ in 0..2 {
            data.push(0x7F);
            data.extend([1; 128]);
        }
        let mut at = 0;
        let numbers = point_numbers(&data, &mut at)
            .expect("numbers")
            .expect("list");
        assert_eq!(numbers.len(), 256);
        assert_eq!(numbers[255], 256);
    }

    #[test]
    fn packed_deltas_are_zeros_bytes_words_and_longs() {
        let data = [
            0x81, // two zeros
            0x01, 5, 0xFB, // two bytes
            0x40, 0x01, 0x00, // one word
            0xC0, 0xFF, 0xFF, 0xFF, 0xFE, // one long
        ];
        let mut at = 0;
        assert_eq!(deltas(&data, &mut at, 6), Ok(vec![0, 0, 5, -5, 256, -2]));
        assert_eq!(at, data.len());
    }

    #[test]
    fn torn_and_overlong_runs_are_refused() {
        let mut at = 0;
        assert_eq!(
            point_numbers(&[3, 0x01, 1], &mut at),
            Err(FontError::Truncated)
        );
        // A run of three where two numbers are left.
        let mut at = 0;
        assert_eq!(
            point_numbers(&[2, 0x02, 1, 1, 1], &mut at),
            Err(FontError::BadTable)
        );
        let mut at = 0;
        assert_eq!(deltas(&[0x02, 1, 2], &mut at, 3), Err(FontError::Truncated));
        let mut at = 0;
        assert_eq!(deltas(&[0x83], &mut at, 3), Err(FontError::BadTable));
        let mut at = 0;
        assert_eq!(deltas(&[], &mut at, 1), Err(FontError::Truncated));
    }

    #[test]
    fn a_point_between_two_moved_points_follows_them() {
        // Halfway between, a quarter of the way, and outside on each side.
        assert!((between(50.0, 0.0, 100.0, 10.0, 30.0) - 20.0).abs() < 1e-6);
        assert!((between(25.0, 100.0, 0.0, 30.0, 10.0) - 15.0).abs() < 1e-6);
        assert!((between(-5.0, 0.0, 100.0, 10.0, 30.0) - 10.0).abs() < 1e-6);
        assert!((between(120.0, 0.0, 100.0, 10.0, 30.0) - 30.0).abs() < 1e-6);
        // Two points at one place move it only when they move alike.
        assert!((between(7.0, 7.0, 7.0, 4.0, 4.0) - 4.0).abs() < 1e-6);
        assert!(between(7.0, 7.0, 7.0, 4.0, 5.0).abs() < 1e-6);
    }
}
