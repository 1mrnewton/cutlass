//! Byte conversions for one PCM packet.
//!
//! Integer samples wider than 16 bits become full-scale [`f32`]. Positive
//! full scale is the largest positive integer of that width, the same rule
//! the mix uses for [`i16`]. Values past that, including the extra negative
//! code, are kept. Finite floats outside ±1 are kept. A non-finite float is
//! an error, not a zero.

use cutlass_decode::CodecError;

use crate::layout::{Arrangement, Coding, Endian, Layout};
use crate::limit::{self, S24_FULL, S32_FULL, S64_SCALE};

pub(crate) enum Decoded {
    I16(Vec<i16>),
    F32(Vec<f32>),
}

pub(crate) fn decode(layout: Layout, channels: usize, data: &[u8]) -> Result<Decoded, CodecError> {
    limit::packet_len(data.len())?;
    let width = layout.coding().sample_bytes();
    let frame = limit::frame_bytes(channels, width)?;
    let frames = limit::whole_frames(data.len(), frame)?;
    let shape = Shape {
        layout,
        channels,
        frames,
    };
    match layout.coding() {
        Coding::S16 => Ok(Decoded::I16(read(
            shape,
            data,
            i16::from_le_bytes,
            i16::from_be_bytes,
        )?)),
        Coding::S24 => Ok(Decoded::F32(read(
            shape,
            data,
            |bytes: [u8; 3]| {
                unit(
                    i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]) >> 8,
                    S24_FULL,
                )
            },
            |bytes: [u8; 3]| {
                unit(
                    i32::from_be_bytes([bytes[0], bytes[1], bytes[2], 0]) >> 8,
                    S24_FULL,
                )
            },
        )?)),
        Coding::S32 => Ok(Decoded::F32(read(
            shape,
            data,
            |bytes| unit(i32::from_le_bytes(bytes), S32_FULL),
            |bytes| unit(i32::from_be_bytes(bytes), S32_FULL),
        )?)),
        Coding::F32 => finite(read(shape, data, f32::from_le_bytes, f32::from_be_bytes)?),
        Coding::F64 => finite(read(
            shape,
            data,
            |bytes| narrow_f64(f64::from_le_bytes(bytes)),
            |bytes| narrow_f64(f64::from_be_bytes(bytes)),
        )?),
        Coding::S64 => Ok(Decoded::F32(read(
            shape,
            data,
            |bytes| s64_unit(i64::from_le_bytes(bytes)),
            |bytes| s64_unit(i64::from_be_bytes(bytes)),
        )?)),
        Coding::S8 => Ok(Decoded::I16(read_bytes(shape, data, |byte| {
            i16::from(i8::from_ne_bytes([byte])) << 8
        })?)),
        Coding::U8 => Ok(Decoded::I16(read_bytes(shape, data, |byte| {
            (i16::from(byte) - 128) << 8
        })?)),
        Coding::ALaw => Ok(Decoded::I16(read_bytes(shape, data, |byte| {
            ALAW[usize::from(byte)]
        })?)),
        Coding::MuLaw => Ok(Decoded::I16(read_bytes(shape, data, |byte| {
            MULAW[usize::from(byte)]
        })?)),
    }
}

/// A non-finite sample is an error. A binary64 past the binary32 range
/// narrows to an infinity, so one pass over the narrowed samples finds a
/// non-finite source and an overflow alike. The pass has no early exit, so
/// it is a vector compare.
fn finite(samples: Vec<f32>) -> Result<Decoded, CodecError> {
    if all_finite(&samples) {
        Ok(Decoded::F32(samples))
    } else {
        Err(CodecError::NonFinite)
    }
}

fn all_finite(samples: &[f32]) -> bool {
    samples
        .iter()
        .fold(true, |every, sample| every & sample.is_finite())
}

/// G.711 A-law to 16-bit linear, every code. ITU-T G.711 A-law: the even
/// bits are inverted on the line, bit 7 set means positive, bits 6..4 are
/// the segment and bits 3..0 the step. A step decodes to the middle of its
/// interval, in 13-bit units scaled by 8 to 16 bits. All 256 values are
/// checked against FFmpeg and AudioToolbox in the tests.
pub(crate) const ALAW: [i16; 256] = alaw_table();

/// G.711 mu-law to 16-bit linear, every code. ITU-T G.711 mu-law: every
/// bit is inverted on the line, bit 7 set means negative, bits 6..4 are
/// the segment and bits 3..0 the step. Values are in 14-bit units scaled
/// by 4 to 16 bits. All 256 values are checked against FFmpeg and
/// AudioToolbox in the tests.
pub(crate) const MULAW: [i16; 256] = mulaw_table();

const fn alaw_table() -> [i16; 256] {
    let mut table = [0i16; 256];
    let mut code = 0usize;
    while code < 256 {
        #[allow(clippy::cast_possible_truncation)]
        let line = (code as u8) ^ 0x55;
        let segment = (line >> 4) & 0x07;
        let step = (line & 0x0F) as i32;
        // Interval midpoints: segment 0 covers 0..32 in steps of 2, and
        // segment s > 0 covers 2^(s+4)..2^(s+5) in steps of 2^s.
        let magnitude = if segment == 0 {
            2 * step + 1
        } else {
            (2 * step + 33) << (segment - 1)
        };
        let linear = magnitude << 3;
        #[allow(clippy::cast_possible_truncation)]
        {
            table[code] = if line & 0x80 == 0 { -linear } else { linear } as i16;
        }
        code += 1;
    }
    table
}

const fn mulaw_table() -> [i16; 256] {
    let mut table = [0i16; 256];
    let mut code = 0usize;
    while code < 256 {
        #[allow(clippy::cast_possible_truncation)]
        let line = !(code as u8);
        let segment = (line >> 4) & 0x07;
        let step = (line & 0x0F) as i32;
        // Biased by 33: segment s starts at 33 * 2^s - 33 and moves in
        // steps of 2^(s+1).
        let magnitude = ((2 * step + 33) << segment) - 33;
        let linear = magnitude << 2;
        #[allow(clippy::cast_possible_truncation)]
        {
            table[code] = if line & 0x80 == 0 { linear } else { -linear } as i16;
        }
        code += 1;
    }
    table
}

pub(crate) fn encode(
    layout: Layout,
    channels: usize,
    pcm_i16: Option<&[i16]>,
    pcm_f32: Option<&[f32]>,
) -> Result<Vec<u8>, CodecError> {
    layout.coding().encode()?;
    let wrong = CodecError::InvalidApi {
        what: "sample format",
    };
    match layout.coding() {
        Coding::S16 => write(
            layout,
            channels,
            pcm_i16.ok_or(wrong)?,
            i16::to_le_bytes,
            i16::to_be_bytes,
        ),
        Coding::F32 => write(
            layout,
            channels,
            every_finite(pcm_f32.ok_or(wrong)?)?,
            f32::to_le_bytes,
            f32::to_be_bytes,
        ),
        Coding::F64 => write(
            layout,
            channels,
            every_finite(pcm_f32.ok_or(wrong)?)?,
            |sample| f64::from(sample).to_le_bytes(),
            |sample| f64::from(sample).to_be_bytes(),
        ),
        Coding::S24 => write(
            layout,
            channels,
            every_finite(pcm_f32.ok_or(wrong)?)?,
            |sample| {
                let [low, mid, high, _] = code24(sample).to_le_bytes();
                [low, mid, high]
            },
            |sample| {
                let [_, high, mid, low] = code24(sample).to_be_bytes();
                [high, mid, low]
            },
        ),
        Coding::S32 => write(
            layout,
            channels,
            every_finite(pcm_f32.ok_or(wrong)?)?,
            |sample| code32(sample).to_le_bytes(),
            |sample| code32(sample).to_be_bytes(),
        ),
        Coding::S8 | Coding::U8 | Coding::ALaw | Coding::MuLaw | Coding::S64 => {
            Err(CodecError::Unsupported { what: "pcm encode" })
        }
    }
}

/// What one packet holds.
#[derive(Clone, Copy)]
struct Shape {
    layout: Layout,
    channels: usize,
    frames: usize,
}

/// Samples of `W` bytes, in frame order whatever the packet's arrangement.
/// The byte order is chosen once, outside the loop over the samples.
fn read<T, const W: usize>(
    shape: Shape,
    data: &[u8],
    little: impl Fn([u8; W]) -> T,
    big: impl Fn([u8; W]) -> T,
) -> Result<Vec<T>, CodecError>
where
    T: Copy + Default,
{
    match shape.layout.endian() {
        Endian::Little => read_with(shape, data, little),
        Endian::Big => read_with(shape, data, big),
    }
}

fn read_bytes<T>(shape: Shape, data: &[u8], sample: impl Fn(u8) -> T) -> Result<Vec<T>, CodecError>
where
    T: Copy + Default,
{
    read_with(shape, data, |[byte]: [u8; 1]| sample(byte))
}

fn read_with<T, const W: usize>(
    shape: Shape,
    data: &[u8],
    sample: impl Fn([u8; W]) -> T,
) -> Result<Vec<T>, CodecError>
where
    T: Copy + Default,
{
    let count = limit::sample_count(shape.frames, shape.channels)?;
    let (stored, tail) = data.as_chunks::<W>();
    if stored.len() != count || !tail.is_empty() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "pcm frame",
        });
    }
    let mut out = limit::reserve(count)?;
    match shape.layout.arrangement() {
        Arrangement::Interleaved => out.extend(stored.iter().map(|bytes| sample(*bytes))),
        Arrangement::Planar => {
            out.resize(count, T::default());
            // One pass per channel: the plane is read in order and written
            // every `channels` samples.
            for (channel, plane) in stored.chunks_exact(shape.frames.max(1)).enumerate() {
                let slots = out.iter_mut().skip(channel).step_by(shape.channels.max(1));
                for (slot, bytes) in slots.zip(plane) {
                    *slot = sample(*bytes);
                }
            }
        }
    }
    Ok(out)
}

/// Full scale is 2^63: `i64::MAX` is not a binary64 value and rounds to
/// 2^63, so this is the same rule as the narrower widths. Rounded once,
/// straight from the integer to `f32`, then scaled by a power of two,
/// which is exact.
fn s64_unit(value: i64) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let rounded = value as f32;
    rounded * S64_SCALE
}

fn unit(value: i32, full: f64) -> f32 {
    narrow_f64(f64::from(value) / full)
}

fn narrow_f64(value: f64) -> f32 {
    #[allow(clippy::cast_possible_truncation)]
    {
        value as f32
    }
}

/// The integer code of a full-scale float: `full` is positive full scale,
/// the inverse of [`unit`]. Rounded to nearest, ties to even, and kept
/// inside `low..=high`, so a sample past full scale is the end code and
/// not a wrapped one.
fn code(sample: f32, full: f64, low: f64, high: f64) -> i32 {
    // `max` then `min`, not `clamp`: the same value for a finite sample,
    // and two instructions the vector loop keeps.
    let scaled = nearest_even((f64::from(sample) * full).max(low).min(high));
    #[allow(clippy::cast_possible_truncation)]
    {
        scaled as i32
    }
}

/// `value` rounded to the nearest integer, ties to even, for a magnitude
/// below 2^51. Adding 1.5 * 2^52 leaves no fraction bits, so the sum is
/// rounded by the addition itself, in the default rounding mode, and the
/// subtraction is exact. It is two additions, where `round_ties_even` is a
/// library call on a baseline x86-64 target, so the loop over the samples
/// stays a vector loop.
fn nearest_even(value: f64) -> f64 {
    const SHIFT: f64 = 6_755_399_441_055_744.0;
    (value + SHIFT) - SHIFT
}

fn code24(sample: f32) -> i32 {
    code(sample, S24_FULL, -8_388_608.0, S24_FULL)
}

fn code32(sample: f32) -> i32 {
    code(sample, S32_FULL, -2_147_483_648.0, S32_FULL)
}

fn every_finite(samples: &[f32]) -> Result<&[f32], CodecError> {
    if all_finite(samples) {
        Ok(samples)
    } else {
        Err(CodecError::NonFinite)
    }
}

/// Bytes of `samples`, which are in frame order, in the packet's
/// arrangement and byte order.
fn write<T, const W: usize>(
    layout: Layout,
    channels: usize,
    samples: &[T],
    little: impl Fn(T) -> [u8; W],
    big: impl Fn(T) -> [u8; W],
) -> Result<Vec<u8>, CodecError>
where
    T: Copy,
{
    match layout.endian() {
        Endian::Little => write_with(layout, channels, samples, little),
        Endian::Big => write_with(layout, channels, samples, big),
    }
}

fn write_with<T, const W: usize>(
    layout: Layout,
    channels: usize,
    samples: &[T],
    bytes_of: impl Fn(T) -> [u8; W],
) -> Result<Vec<u8>, CodecError>
where
    T: Copy,
{
    if channels == 0 || !samples.len().is_multiple_of(channels) {
        return Err(CodecError::UnevenChannels);
    }
    let frames = samples.len() / channels;
    let bytes = samples
        .len()
        .checked_mul(W)
        .ok_or(CodecError::Limit { what: "allocation" })?;
    let mut out = limit::reserve_bytes(bytes)?;
    out.resize(bytes, 0);
    let (stored, _) = out.as_chunks_mut::<W>();
    match layout.arrangement() {
        Arrangement::Interleaved => {
            for (slot, sample) in stored.iter_mut().zip(samples) {
                *slot = bytes_of(*sample);
            }
        }
        Arrangement::Planar => {
            for (channel, plane) in stored.chunks_exact_mut(frames.max(1)).enumerate() {
                let source = samples.iter().skip(channel).step_by(channels);
                for (slot, sample) in plane.iter_mut().zip(source) {
                    *slot = bytes_of(*sample);
                }
            }
        }
    }
    Ok(out)
}
