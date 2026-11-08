//! One frame to samples: subframes, residuals, prediction, and channel
//! decorrelation (RFC 9639, sections 9.2 and 9.3).
//!
//! Every reconstructed sample is checked against its subframe bit depth
//! before it feeds the next prediction. That bounds the 64-bit predictor
//! sums: a coefficient is under 2^15, a sample under 2^32 (a 33-bit side
//! channel), and an order at most 32, so a sum stays under 2^52.

use cutlass_decode::CodecError;

use crate::bits::BitReader;
use crate::crc::crc16;
use crate::header::{ChannelAssignment, FrameHeader};
use crate::streaminfo::StreamInfo;

/// One decoded frame.
///
/// `samples` is interleaved in the channel order of RFC 9639 section 9.1.3,
/// as signed integers at `bits` depth. Nothing is shifted or scaled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub header: FrameHeader,
    pub sample_rate: u32,
    pub bits: u8,
    pub channels: u8,
    pub samples: Vec<i32>,
}

impl Frame {
    /// Samples per channel.
    #[must_use]
    pub fn block_size(&self) -> u32 {
        self.header.block_size
    }
}

fn malformed(at_bit: u64, what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit, what }
}

/// Decodes one complete frame: header, subframes, padding, and CRC-16.
///
/// `info` supplies the sample rate and depth when the header defers to
/// STREAMINFO, and bounds the block size. A header that disagrees with
/// `info` is [`CodecError::Malformed`]. `planes` is scratch space.
pub(crate) fn decode_frame(
    bytes: &[u8],
    info: Option<&StreamInfo>,
    planes: &mut Vec<i64>,
) -> Result<Frame, CodecError> {
    let header = FrameHeader::parse(bytes)?;
    let total = bytes.len();
    if total < header.len + 2 {
        return Err(CodecError::Truncated {
            at_bit: u64::try_from(total).unwrap_or(u64::MAX / 8) * 8,
            what: "frame footer",
        });
    }
    let body_end = total - 2;
    let stored = u16::from_be_bytes([bytes[body_end], bytes[body_end + 1]]);
    if crc16(&bytes[..body_end]) != stored {
        return Err(malformed(
            u64::try_from(body_end).unwrap_or(0) * 8,
            "frame crc",
        ));
    }
    let bits = resolve_bits(&header, info)?;
    let sample_rate = resolve_rate(&header, info)?;
    let channels = header.assignment.channels();
    if let Some(info) = info {
        if info.channels != channels {
            return Err(malformed(24, "channel count differs from streaminfo"));
        }
        if header.block_size > u32::from(info.max_block) {
            return Err(malformed(16, "block larger than the streaminfo maximum"));
        }
    }
    let block =
        usize::try_from(header.block_size).map_err(|_| CodecError::Limit { what: "block size" })?;
    let count = usize::from(channels);
    planes.clear();
    planes.resize(block * count, 0);
    let mut reader = BitReader::at_byte(&bytes[..body_end], header.len);
    for (index, plane) in planes.chunks_exact_mut(block).enumerate() {
        let depth = u32::from(bits) + u32::from(header.assignment.is_side(index));
        decode_subframe(&mut reader, depth, plane)?;
    }
    let at = reader.position();
    if reader.align()? != 0 {
        return Err(malformed(at, "frame padding"));
    }
    if reader.remaining() != 0 {
        return Err(malformed(
            reader.position(),
            "bytes after the last subframe",
        ));
    }
    let samples = interleave(planes, block, header.assignment, bits)?;
    Ok(Frame {
        header,
        sample_rate,
        bits,
        channels,
        samples,
    })
}

fn resolve_bits(header: &FrameHeader, info: Option<&StreamInfo>) -> Result<u8, CodecError> {
    match (header.bits, info) {
        (Some(bits), Some(info)) if bits != info.bits => {
            Err(malformed(28, "bit depth differs from streaminfo"))
        }
        (Some(bits), _) => Ok(bits),
        (None, Some(info)) => Ok(info.bits),
        (None, None) => Err(malformed(28, "bit depth needs streaminfo")),
    }
}

fn resolve_rate(header: &FrameHeader, info: Option<&StreamInfo>) -> Result<u32, CodecError> {
    match (header.sample_rate, info) {
        (Some(rate), Some(info)) if rate != info.sample_rate => {
            Err(malformed(20, "sample rate differs from streaminfo"))
        }
        (Some(rate), _) => Ok(rate),
        (None, Some(info)) => Ok(info.sample_rate),
        (None, None) => Err(malformed(20, "sample rate needs streaminfo")),
    }
}

/// Decodes one subframe of `depth` bits (the frame depth, plus one for a
/// side channel) into `out`, one sample per slot.
fn decode_subframe(
    reader: &mut BitReader<'_>,
    depth: u32,
    out: &mut [i64],
) -> Result<(), CodecError> {
    let start = reader.position();
    if reader.read(1, "subframe header")? != 0 {
        return Err(malformed(start, "subframe padding bit"));
    }
    let kind = reader.read(6, "subframe header")?;
    let wasted = if reader.read(1, "subframe header")? == 1 {
        let zeros = reader.unary(u64::from(depth), "wasted bits")?;
        u32::try_from(zeros + 1).unwrap_or(u32::MAX)
    } else {
        0
    };
    if wasted >= depth {
        return Err(malformed(start, "wasted bits leave no sample bits"));
    }
    let bits = depth - wasted;
    match kind {
        0 => {
            let value = reader.read_signed(bits, "constant subframe")?;
            out.fill(value);
        }
        1 => {
            for slot in out.iter_mut() {
                *slot = reader.read_signed(bits, "verbatim subframe")?;
            }
        }
        8..=12 => {
            let order = usize::try_from(kind - 8).unwrap_or(0);
            fixed(reader, bits, order, out)?;
        }
        32..=63 => {
            let order = usize::try_from(kind - 31).unwrap_or(1);
            lpc(reader, bits, order, out)?;
        }
        _ => return Err(malformed(start + 1, "reserved subframe type")),
    }
    if wasted > 0 {
        for slot in out.iter_mut() {
            *slot <<= wasted;
        }
    }
    Ok(())
}

/// The representable range of a `bits`-wide signed sample.
fn range(bits: u32) -> (i64, i64) {
    let half = 1i64 << (bits - 1);
    (-half, half - 1)
}

fn warm_up(
    reader: &mut BitReader<'_>,
    bits: u32,
    order: usize,
    out: &mut [i64],
) -> Result<(), CodecError> {
    if order > out.len() {
        return Err(malformed(
            reader.position(),
            "predictor order above block size",
        ));
    }
    for slot in &mut out[..order] {
        *slot = reader.read_signed(bits, "warm-up sample")?;
    }
    Ok(())
}

fn fixed(
    reader: &mut BitReader<'_>,
    bits: u32,
    order: usize,
    out: &mut [i64],
) -> Result<(), CodecError> {
    warm_up(reader, bits, order, out)?;
    residual(reader, order, out)?;
    let (low, high) = range(bits);
    let at = reader.position();
    for i in order..out.len() {
        let prediction = match order {
            0 => 0,
            1 => out[i - 1],
            2 => 2 * out[i - 1] - out[i - 2],
            3 => 3 * out[i - 1] - 3 * out[i - 2] + out[i - 3],
            _ => 4 * out[i - 1] - 6 * out[i - 2] + 4 * out[i - 3] - out[i - 4],
        };
        let value = out[i] + prediction;
        if value < low || value > high {
            return Err(malformed(at, "fixed prediction leaves the bit depth"));
        }
        out[i] = value;
    }
    Ok(())
}

fn lpc(
    reader: &mut BitReader<'_>,
    bits: u32,
    order: usize,
    out: &mut [i64],
) -> Result<(), CodecError> {
    warm_up(reader, bits, order, out)?;
    let at = reader.position();
    let precision = u32::try_from(reader.read(4, "lpc precision")? + 1).unwrap_or(16);
    if precision == 16 {
        return Err(malformed(at, "forbidden lpc precision"));
    }
    let shift = reader.read_signed(5, "lpc shift")?;
    if shift < 0 {
        return Err(malformed(at + 4, "negative lpc shift"));
    }
    let mut coefficients = [0i64; 32];
    for coefficient in &mut coefficients[..order] {
        *coefficient = reader.read_signed(precision, "lpc coefficient")?;
    }
    residual(reader, order, out)?;
    let (low, high) = range(bits);
    let at = reader.position();
    let coefficients = &coefficients[..order];
    for i in order..out.len() {
        let history = &out[i - order..i];
        let sum: i64 = coefficients
            .iter()
            .zip(history.iter().rev())
            .map(|(c, x)| c * x)
            .sum();
        let value = out[i] + (sum >> shift);
        if value < low || value > high {
            return Err(malformed(at, "lpc prediction leaves the bit depth"));
        }
        out[i] = value;
    }
    Ok(())
}

/// Reads the coded residual (section 9.2.7) into `out[order..]`.
fn residual(reader: &mut BitReader<'_>, order: usize, out: &mut [i64]) -> Result<(), CodecError> {
    let at = reader.position();
    let method = reader.read(2, "residual coding method")?;
    if method > 1 {
        return Err(malformed(at, "reserved residual coding method"));
    }
    let partition_order = u32::try_from(reader.read(4, "partition order")?).unwrap_or(15);
    let block = out.len();
    let parts = 1usize << partition_order;
    if !block.is_multiple_of(parts) {
        return Err(malformed(
            at + 2,
            "partition order does not divide the block",
        ));
    }
    let size = block >> partition_order;
    if size < order {
        return Err(malformed(
            at + 2,
            "first partition shorter than the predictor",
        ));
    }
    let parameter_bits = 4 + u32::try_from(method).unwrap_or(0);
    let escape = (1u64 << parameter_bits) - 1;
    let mut index = order;
    for part in 0..parts {
        let end = (part + 1) * size;
        let parameter = reader.read(parameter_bits, "rice parameter")?;
        if parameter == escape {
            let width = u32::try_from(reader.read(5, "escape width")?).unwrap_or(0);
            for slot in &mut out[index..end] {
                *slot = reader.read_signed(width, "escaped residual")?;
            }
        } else {
            let k = u32::try_from(parameter).unwrap_or(0);
            let limit = u64::from(u32::MAX - 1) >> k;
            for slot in &mut out[index..end] {
                let quotient = reader.unary(limit, "rice residual")?;
                let low = reader.read(k, "rice residual")?;
                let folded = (quotient << k) | low;
                if folded > u64::from(u32::MAX - 1) {
                    return Err(malformed(reader.position(), "residual outside 32 bits"));
                }
                let half = (folded >> 1) as i64;
                *slot = if folded & 1 == 0 { half } else { -half - 1 };
            }
        }
        index = end;
    }
    Ok(())
}

/// Undoes the channel assignment and interleaves, checking each output
/// sample fits `bits`.
fn interleave(
    planes: &[i64],
    block: usize,
    assignment: ChannelAssignment,
    bits: u8,
) -> Result<Vec<i32>, CodecError> {
    let channels = usize::from(assignment.channels());
    let (low, high) = range(u32::from(bits));
    let mut samples = Vec::with_capacity(block * channels);
    let check = |value: i64| -> Result<i32, CodecError> {
        if value < low || value > high {
            return Err(malformed(0, "decorrelated sample leaves the bit depth"));
        }
        i32::try_from(value).map_err(|_| malformed(0, "sample wider than 32 bits"))
    };
    match assignment {
        ChannelAssignment::Independent(_) => {
            for i in 0..block {
                for channel in 0..channels {
                    samples.push(check(planes[channel * block + i])?);
                }
            }
        }
        ChannelAssignment::LeftSide => {
            let (left, side) = planes.split_at(block);
            for (l, s) in left.iter().zip(side) {
                samples.push(check(*l)?);
                samples.push(check(l - s)?);
            }
        }
        ChannelAssignment::SideRight => {
            let (side, right) = planes.split_at(block);
            for (s, r) in side.iter().zip(right) {
                samples.push(check(s + r)?);
                samples.push(check(*r)?);
            }
        }
        ChannelAssignment::MidSide => {
            let (mid, side) = planes.split_at(block);
            for (m, s) in mid.iter().zip(side) {
                let mid = (m << 1) | (s & 1);
                samples.push(check((mid + s) >> 1)?);
                samples.push(check((mid - s) >> 1)?);
            }
        }
    }
    Ok(samples)
}
