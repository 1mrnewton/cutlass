//! Floor packet decode and curve synthesis: floor 0 (section 6) and floor 1
//! (section 7).

use crate::bits::{BitReader, ilog};
use crate::codebook::Codebook;
use crate::header::malformed;
use crate::setup::{Floor0, Floor1};
use crate::tables::INVERSE_DB;
use cutlass_decode::CodecError;

/// Floor 1 Y range per multiplier.
pub(crate) const FLOOR1_RANGE: [i32; 4] = [256, 128, 86, 64];

/// What floor decode found for one channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    Used,
    Unused,
    /// The packet ended inside the floor.
    EndOfPacket,
}

/// One channel's decoded floor, kept until synthesis.
#[derive(Clone, Debug, Default)]
pub(crate) struct ChannelFloor {
    /// Floor 1: final Y per point and whether step 2 draws it.
    ys: Vec<i32>,
    drawn: Vec<bool>,
    /// Floor 0: amplitude and LSP coefficients.
    amplitude: u64,
    coefficients: Vec<f32>,
}

impl ChannelFloor {
    /// Decodes a floor 1 for one channel and runs amplitude synthesis
    /// (step 1 of section 7.2.4).
    pub(crate) fn decode_floor1(
        &mut self,
        floor: &Floor1,
        books: &[Codebook],
        reader: &mut BitReader<'_>,
    ) -> Status {
        let Some(nonzero) = reader.read_bit() else {
            return Status::EndOfPacket;
        };
        if !nonzero {
            return Status::Unused;
        }
        let range = FLOOR1_RANGE[usize::from(floor.multiplier - 1)];
        let width = ilog((range - 1) as u32);
        let values = floor.xs.len();
        self.ys.clear();
        self.ys.resize(values, 0);
        let (Some(first), Some(second)) = (reader.read(width), reader.read(width)) else {
            return Status::EndOfPacket;
        };
        self.ys[0] = first as i32;
        self.ys[1] = second as i32;
        let mut offset = 2;
        for &class in &floor.partition_classes {
            let class = &floor.classes[usize::from(class)];
            let bits = u32::from(class.subclass_bits);
            let mask = (1_u32 << bits) - 1;
            let mut cval = 0;
            if bits > 0 {
                let Some(value) = books[usize::from(class.masterbook)].decode(reader) else {
                    return Status::EndOfPacket;
                };
                cval = value;
            }
            for j in 0..usize::from(class.dimensions) {
                let book = class.subclass_books[(cval & mask) as usize];
                cval >>= bits;
                self.ys[offset + j] = match book {
                    Some(book) => match books[usize::from(book)].decode(reader) {
                        Some(value) => value as i32,
                        None => return Status::EndOfPacket,
                    },
                    None => 0,
                };
            }
            offset += usize::from(class.dimensions);
        }
        // Step 1: undo the prediction of each point from its neighbors.
        self.drawn.clear();
        self.drawn.resize(values, false);
        self.drawn[0] = true;
        self.drawn[1] = true;
        for i in 2..values {
            let low = floor.low[i];
            let high = floor.high[i];
            let predicted = render_point(
                i32::from(floor.xs[low]),
                self.ys[low],
                i32::from(floor.xs[high]),
                self.ys[high],
                i32::from(floor.xs[i]),
            );
            let value = self.ys[i];
            let high_room = range - predicted;
            let low_room = predicted;
            let room = if high_room < low_room {
                high_room * 2
            } else {
                low_room * 2
            };
            if value != 0 {
                self.drawn[low] = true;
                self.drawn[high] = true;
                self.drawn[i] = true;
                self.ys[i] = if value >= room {
                    if high_room > low_room {
                        value - low_room + predicted
                    } else {
                        predicted - value + high_room - 1
                    }
                } else if value & 1 == 1 {
                    predicted - (value + 1) / 2
                } else {
                    predicted + value / 2
                };
            } else {
                self.ys[i] = predicted;
            }
        }
        // The spec's suggested guard against hostile streams.
        for y in &mut self.ys {
            *y = (*y).clamp(0, range - 1);
        }
        Status::Used
    }

    /// Multiplies `spectrum` (`n` values) by the floor 1 curve (step 2).
    pub(crate) fn apply_floor1(&self, floor: &Floor1, spectrum: &mut [f32]) {
        floor1_curve(floor, &self.ys, &self.drawn, spectrum.len(), |x, y| {
            spectrum[x] *= gain(y)
        });
    }

    /// Decodes a floor 0 for one channel (section 6.2.2).
    pub(crate) fn decode_floor0(
        &mut self,
        floor: &Floor0,
        books: &[Codebook],
        reader: &mut BitReader<'_>,
    ) -> Result<Status, CodecError> {
        let at = reader.position();
        let Some(amplitude) = read_long(reader, floor.amplitude_bits) else {
            return Ok(Status::EndOfPacket);
        };
        if amplitude == 0 {
            return Ok(Status::Unused);
        }
        let Some(number) = reader.read(ilog(floor.books.len() as u32)) else {
            return Ok(Status::EndOfPacket);
        };
        let Some(&book) = floor.books.get(number as usize) else {
            return Err(malformed(at, "floor 0 book number"));
        };
        let book = &books[usize::from(book)];
        if book.values.is_none() {
            return Err(malformed(at, "floor 0 book without values"));
        }
        self.amplitude = amplitude;
        self.coefficients.clear();
        let mut last = 0.0_f32;
        while self.coefficients.len() < floor.order {
            let Some(vector) = book.decode_vector(reader) else {
                return Ok(Status::EndOfPacket);
            };
            for &value in vector {
                self.coefficients.push(value + last);
            }
            last = self.coefficients.last().copied().unwrap_or(0.0);
        }
        Ok(Status::Used)
    }

    /// Multiplies `spectrum` by the floor 0 curve (section 6.2.3), with the
    /// Bark map of this block size.
    pub(crate) fn apply_floor0(
        &mut self,
        floor: &Floor0,
        long: bool,
        spectrum: &mut [f32],
    ) -> Result<(), CodecError> {
        let map = &floor.maps[usize::from(long)];
        let order = floor.order;
        for value in &mut self.coefficients[..order] {
            *value = 2.0 * value.cos();
        }
        let step = std::f32::consts::PI / floor.bark_map_size as f32;
        let full = ((1_u64 << floor.amplitude_bits) - 1) as f32;
        let amplitude = self.amplitude as f32 * floor.amplitude_offset as f32;
        let offset = floor.amplitude_offset as f32;
        let mut i = 0;
        while i < spectrum.len() {
            let k = map[i];
            let w = 2.0 * (step * k as f32).cos();
            let mut p = 0.5_f32;
            let mut q = 0.5_f32;
            let mut j = 0;
            while j + 1 < order {
                q *= self.coefficients[j] - w;
                p *= self.coefficients[j + 1] - w;
                j += 2;
            }
            if j == order {
                p = p * p * (2.0 - w);
                q = q * q * (2.0 + w);
            } else {
                q *= self.coefficients[j] - w;
                p = p * p * (4.0 - w * w);
                q = q * q;
            }
            let sum = p + q;
            if sum <= 0.0 || !sum.is_finite() {
                return Err(malformed(0, "floor 0 curve"));
            }
            let gain = ((amplitude / (full * sum.sqrt()) - offset) * 0.115_129_25).exp();
            while i < spectrum.len() && map[i] == k {
                spectrum[i] *= gain;
                i += 1;
            }
        }
        Ok(())
    }
}

/// Reads up to 63 bits (floor 0 amplitude).
fn read_long(reader: &mut BitReader<'_>, width: u32) -> Option<u64> {
    if width <= 32 {
        return reader.read(width).map(u64::from);
    }
    let low = reader.read(32)?;
    let high = reader.read(width - 32)?;
    Some(u64::from(low) | (u64::from(high) << 32))
}

/// `render_point` (section 9.2.6).
pub(crate) fn render_point(x0: i32, y0: i32, x1: i32, y1: i32, x: i32) -> i32 {
    let dy = y1 - y0;
    let adx = x1 - x0;
    let err = dy.abs() * (x - x0);
    let off = err / adx;
    if dy < 0 { y0 - off } else { y0 + off }
}

/// `render_line` (section 9.2.7): `put(x, y)` for each `x` in `x0..x1`
/// below `n`.
fn render_line(x0: i32, y0: i32, x1: i32, y1: i32, n: usize, put: &mut impl FnMut(usize, i32)) {
    let n = n as i32;
    if x0 >= n || x1 <= x0 {
        return;
    }
    let dy = y1 - y0;
    let adx = x1 - x0;
    let base = dy / adx;
    let sy = if dy < 0 { base - 1 } else { base + 1 };
    let ady = dy.abs() - base.abs() * adx;
    let mut y = y0;
    let mut err = 0;
    put(x0 as usize, y);
    for x in (x0 + 1)..x1.min(n) {
        err += ady;
        if err >= adx {
            err -= adx;
            y += sy;
        } else {
            y += base;
        }
        put(x as usize, y);
    }
}

/// The floor 1 curve of `n` bins (step 2 of section 7.2.4): `put(x, y)`
/// with the scaled Y of every bin, for points `ys` of which `drawn` are
/// drawn.
pub(crate) fn floor1_curve(
    floor: &Floor1,
    ys: &[i32],
    drawn: &[bool],
    n: usize,
    mut put: impl FnMut(usize, i32),
) {
    let multiplier = i32::from(floor.multiplier);
    let mut lx = 0;
    let mut ly = ys[floor.sorted[0]] * multiplier;
    let mut hx = 0;
    let mut hy = ly;
    for &point in &floor.sorted[1..] {
        if !drawn[point] {
            continue;
        }
        hy = ys[point] * multiplier;
        hx = i32::from(floor.xs[point]);
        render_line(lx, ly, hx, hy, n, &mut put);
        lx = hx;
        ly = hy;
    }
    if (hx as usize) < n {
        render_line(hx, hy, n as i32, hy, n, &mut put);
    }
}

/// Floor 1 Y to linear gain.
pub(crate) fn gain(y: i32) -> f32 {
    INVERSE_DB[y.clamp(0, 255) as usize]
}

#[cfg(test)]
mod tests {
    use super::{ChannelFloor, INVERSE_DB, floor1_curve, render_line, render_point};
    use crate::setup::Floor1;

    #[test]
    fn render_point_rounds_toward_the_first_point() {
        assert_eq!(render_point(0, 10, 10, 20, 5), 15);
        assert_eq!(render_point(0, 20, 10, 10, 3), 17);
        assert_eq!(render_point(0, 0, 3, 1, 2), 0);
        assert_eq!(render_point(0, 1, 3, 0, 2), 1);
    }

    fn line(x0: i32, y0: i32, x1: i32, y1: i32, n: usize) -> Vec<Option<i32>> {
        let mut out = vec![None; n];
        render_line(x0, y0, x1, y1, n, &mut |x, y| out[x] = Some(y));
        out
    }

    #[test]
    fn render_line_draws_the_spec_line() {
        // A rise of 3 over 8: y steps up at err >= adx; x1 is not drawn.
        let ys = line(0, 100, 8, 103, 10);
        let want = [100, 100, 100, 101, 101, 101, 102, 102];
        assert_eq!(ys[..8], want.map(Some));
        assert_eq!(ys[8..], [None, None]);
        // A steep fall: base -2, ady 1, so the error never carries in four steps.
        assert_eq!(line(0, 200, 4, 191, 4), [200, 198, 196, 194].map(Some));
        // Past the spectrum: clipped.
        assert_eq!(line(0, 10, 100, 10, 2), [Some(10), Some(10)]);
        // An empty span draws nothing.
        assert_eq!(line(5, 10, 5, 20, 8), [None; 8]);
    }

    #[test]
    fn the_curve_joins_drawn_points_and_holds_the_last() {
        // Points at x 0, 16, 8 (sorted 0, 8, 16); the middle point undrawn.
        let floor = Floor1::new(Vec::new(), Vec::new(), 2, 4, [0, 16, 8].to_vec()).unwrap();
        let mut curve = [0; 24];
        floor1_curve(&floor, &[10, 20, 99], &[true, true, false], 24, |x, y| {
            curve[x] = y
        });
        // Multiplier 2: Y 20 to 40 over x 0..16, then held at 40.
        assert_eq!(curve[0], 20);
        assert_eq!(curve[8], 30);
        assert_eq!(curve[15], 38);
        assert!(curve[16..].iter().all(|&y| y == 40));
        let mut spectrum = vec![1.0_f32; 24];
        let channel = ChannelFloor {
            ys: vec![10, 20, 99],
            drawn: vec![true, true, false],
            ..ChannelFloor::default()
        };
        channel.apply_floor1(&floor, &mut spectrum);
        for (x, &value) in spectrum.iter().enumerate() {
            assert_eq!(value, INVERSE_DB[curve[x] as usize]);
        }
    }
}
