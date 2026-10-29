//! Byte conversions for one PCM packet.
//!
//! Integer samples wider than 16 bits become full-scale [`f32`]. Positive
//! full scale is the largest positive integer of that width, the same rule
//! the mix uses for [`i16`]. Values past that, including the extra negative
//! code, are kept. Finite floats outside ±1 are kept. A non-finite float is
//! an error, not a zero.

use cutlass_decode::CodecError;

use crate::layout::{Arrangement, Coding, Endian, Layout};
use crate::limit::{self, S24_FULL, S32_FULL};

pub(crate) enum Decoded {
    I16(Vec<i16>),
    F32(Vec<f32>),
}

pub(crate) fn decode(layout: Layout, channels: usize, data: &[u8]) -> Result<Decoded, CodecError> {
    limit::packet_len(data.len())?;
    let width = layout.coding().sample_bytes();
    let frame = limit::frame_bytes(channels, width)?;
    let frames = limit::whole_frames(data.len(), frame)?;
    match layout.coding() {
        Coding::S16 => Ok(Decoded::I16(read(layout, channels, frames, data, i16_of)?)),
        Coding::S24 => Ok(Decoded::F32(read(
            layout,
            channels,
            frames,
            data,
            |bytes, endian| Ok(unit(i24_of(bytes, endian)?, S24_FULL)),
        )?)),
        Coding::S32 => Ok(Decoded::F32(read(
            layout,
            channels,
            frames,
            data,
            |bytes, endian| Ok(unit(i32_of(bytes, endian)?, S32_FULL)),
        )?)),
        Coding::F32 => Ok(Decoded::F32(read(layout, channels, frames, data, f32_of)?)),
        Coding::F64 => Ok(Decoded::F32(read(layout, channels, frames, data, f64_of)?)),
    }
}

pub(crate) fn encode(
    layout: Layout,
    channels: usize,
    pcm_i16: Option<&[i16]>,
    pcm_f32: Option<&[f32]>,
) -> Result<Vec<u8>, CodecError> {
    match layout.coding() {
        Coding::S16 => {
            let samples = pcm_i16.ok_or(CodecError::InvalidApi {
                what: "sample format",
            })?;
            write_i16(layout, channels, samples)
        }
        Coding::F32 => {
            let samples = pcm_f32.ok_or(CodecError::InvalidApi {
                what: "sample format",
            })?;
            write_f32(layout, channels, samples)
        }
        Coding::F64 => {
            let samples = pcm_f32.ok_or(CodecError::InvalidApi {
                what: "sample format",
            })?;
            write_f64(layout, channels, samples)
        }
        Coding::S24 => Err(CodecError::Unsupported { what: "s24 encode" }),
        Coding::S32 => Err(CodecError::Unsupported { what: "s32 encode" }),
    }
}

fn read<T, F>(
    layout: Layout,
    channels: usize,
    frames: usize,
    data: &[u8],
    mut sample: F,
) -> Result<Vec<T>, CodecError>
where
    F: FnMut(&[u8], Endian) -> Result<T, CodecError>,
{
    let width = layout.coding().sample_bytes();
    if width == 0 {
        return Err(CodecError::InvalidApi { what: "pcm sample" });
    }
    let count = limit::sample_count(frames, channels)?;
    let mut out = limit::reserve(count)?;
    let endian = layout.endian();
    match layout.arrangement() {
        Arrangement::Interleaved => {
            for chunk in data.chunks_exact(width) {
                out.push(sample(chunk, endian)?);
            }
        }
        Arrangement::Planar => {
            for frame in 0..frames {
                for channel in 0..channels {
                    let start = plane_at(channel, frames, frame, width)?;
                    let end = start
                        .checked_add(width)
                        .ok_or(CodecError::Limit { what: "offset" })?;
                    let chunk = data.get(start..end).ok_or(CodecError::Truncated {
                        at_bit: 0,
                        what: "pcm frame",
                    })?;
                    out.push(sample(chunk, endian)?);
                }
            }
        }
    }
    if out.len() != count {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "pcm frame",
        });
    }
    Ok(out)
}

fn plane_at(
    channel: usize,
    frames: usize,
    frame: usize,
    width: usize,
) -> Result<usize, CodecError> {
    channel
        .checked_mul(frames)
        .and_then(|plane| plane.checked_add(frame))
        .and_then(|index| index.checked_mul(width))
        .ok_or(CodecError::Limit { what: "offset" })
}

fn i16_of(bytes: &[u8], endian: Endian) -> Result<i16, CodecError> {
    let bytes: [u8; 2] = bytes.try_into().map_err(|_| CodecError::Truncated {
        at_bit: 0,
        what: "pcm sample",
    })?;
    Ok(match endian {
        Endian::Little => i16::from_le_bytes(bytes),
        Endian::Big => i16::from_be_bytes(bytes),
    })
}

fn i24_of(bytes: &[u8], endian: Endian) -> Result<i32, CodecError> {
    let bytes: [u8; 3] = bytes.try_into().map_err(|_| CodecError::Truncated {
        at_bit: 0,
        what: "pcm sample",
    })?;
    let le = match endian {
        Endian::Little => bytes,
        Endian::Big => [bytes[2], bytes[1], bytes[0]],
    };
    let sign = if le[2] & 0x80 == 0 { 0 } else { 0xFF };
    Ok(i32::from_le_bytes([le[0], le[1], le[2], sign]))
}

fn i32_of(bytes: &[u8], endian: Endian) -> Result<i32, CodecError> {
    let bytes: [u8; 4] = bytes.try_into().map_err(|_| CodecError::Truncated {
        at_bit: 0,
        what: "pcm sample",
    })?;
    Ok(match endian {
        Endian::Little => i32::from_le_bytes(bytes),
        Endian::Big => i32::from_be_bytes(bytes),
    })
}

fn f32_of(bytes: &[u8], endian: Endian) -> Result<f32, CodecError> {
    let bytes: [u8; 4] = bytes.try_into().map_err(|_| CodecError::Truncated {
        at_bit: 0,
        what: "pcm sample",
    })?;
    let sample = match endian {
        Endian::Little => f32::from_le_bytes(bytes),
        Endian::Big => f32::from_be_bytes(bytes),
    };
    if sample.is_finite() {
        Ok(sample)
    } else {
        Err(CodecError::NonFinite)
    }
}

fn f64_of(bytes: &[u8], endian: Endian) -> Result<f32, CodecError> {
    let bytes: [u8; 8] = bytes.try_into().map_err(|_| CodecError::Truncated {
        at_bit: 0,
        what: "pcm sample",
    })?;
    let wide = match endian {
        Endian::Little => f64::from_le_bytes(bytes),
        Endian::Big => f64::from_be_bytes(bytes),
    };
    if !wide.is_finite() {
        return Err(CodecError::NonFinite);
    }
    let narrow = narrow_f64(wide);
    if narrow.is_finite() {
        Ok(narrow)
    } else {
        Err(CodecError::NonFinite)
    }
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

fn write_i16(layout: Layout, channels: usize, samples: &[i16]) -> Result<Vec<u8>, CodecError> {
    write(layout, channels, samples, |sample, endian, slot| {
        let bytes = match endian {
            Endian::Little => sample.to_le_bytes(),
            Endian::Big => sample.to_be_bytes(),
        };
        copy_sample(slot, &bytes)
    })
}

fn write_f32(layout: Layout, channels: usize, samples: &[f32]) -> Result<Vec<u8>, CodecError> {
    for sample in samples {
        if !sample.is_finite() {
            return Err(CodecError::NonFinite);
        }
    }
    write(layout, channels, samples, |sample, endian, slot| {
        let bytes = match endian {
            Endian::Little => sample.to_le_bytes(),
            Endian::Big => sample.to_be_bytes(),
        };
        copy_sample(slot, &bytes)
    })
}

fn write_f64(layout: Layout, channels: usize, samples: &[f32]) -> Result<Vec<u8>, CodecError> {
    for sample in samples {
        if !sample.is_finite() {
            return Err(CodecError::NonFinite);
        }
    }
    write(layout, channels, samples, |sample, endian, slot| {
        let wide = f64::from(*sample);
        let bytes = match endian {
            Endian::Little => wide.to_le_bytes(),
            Endian::Big => wide.to_be_bytes(),
        };
        copy_sample(slot, &bytes)
    })
}

fn copy_sample(slot: &mut [u8], bytes: &[u8]) -> Result<(), CodecError> {
    if slot.len() != bytes.len() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "pcm sample",
        });
    }
    slot.copy_from_slice(bytes);
    Ok(())
}

fn write<T, F>(
    layout: Layout,
    channels: usize,
    samples: &[T],
    mut encode: F,
) -> Result<Vec<u8>, CodecError>
where
    F: FnMut(&T, Endian, &mut [u8]) -> Result<(), CodecError>,
{
    if channels == 0 || !samples.len().is_multiple_of(channels) {
        return Err(CodecError::UnevenChannels);
    }
    let frames = samples.len() / channels;
    let width = layout.coding().sample_bytes();
    let bytes = samples
        .len()
        .checked_mul(width)
        .ok_or(CodecError::Limit { what: "allocation" })?;
    let mut out = limit::reserve_bytes(bytes)?;
    out.resize(bytes, 0);
    let endian = layout.endian();
    for frame in 0..frames {
        for channel in 0..channels {
            let src = frame
                .checked_mul(channels)
                .and_then(|base| base.checked_add(channel))
                .ok_or(CodecError::Limit { what: "offset" })?;
            let sample = samples.get(src).ok_or(CodecError::Malformed {
                at_bit: 0,
                what: "pcm frame",
            })?;
            let start = match layout.arrangement() {
                Arrangement::Interleaved => src
                    .checked_mul(width)
                    .ok_or(CodecError::Limit { what: "offset" })?,
                Arrangement::Planar => plane_at(channel, frames, frame, width)?,
            };
            let end = start
                .checked_add(width)
                .ok_or(CodecError::Limit { what: "offset" })?;
            let slot = out.get_mut(start..end).ok_or(CodecError::Malformed {
                at_bit: 0,
                what: "pcm frame",
            })?;
            encode(sample, endian, slot)?;
        }
    }
    Ok(out)
}
