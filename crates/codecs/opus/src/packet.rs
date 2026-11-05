//! The Opus packet: the TOC byte and frame packing (RFC 6716 section 3).
//!
//! A packet is one TOC byte followed by one to 48 compressed frames that
//! share its mode, bandwidth, frame duration, and channel count. Codes 0 to
//! 3 pack the frames. The self-delimited form of RFC 6716 appendix B adds
//! one length for the last frame; multistream packets use it for every
//! stream but the last. Requirements R1 to R7 of section 3.4 are checked
//! here and a packet that breaks one is [`CodecError::Malformed`].

use cutlass_decode::CodecError;

/// Largest compressed frame (RFC 6716 section 3.2.1, R2).
pub(crate) const MAX_FRAME_BYTES: usize = 1275;
/// Most frames in one packet: 48 frames of 2.5 ms is 120 ms (R5).
pub(crate) const MAX_FRAMES: usize = 48;
/// Longest packet duration, in 48 kHz samples (120 ms, R5).
pub(crate) const MAX_PACKET_SAMPLES: u32 = 5760;

/// Which layer codes a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    SilkOnly,
    Hybrid,
    CeltOnly,
}

/// Audio bandwidth a frame codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Bandwidth {
    /// 4 kHz, 8 kHz effective sample rate.
    Narrow,
    /// 6 kHz, 12 kHz effective sample rate.
    Medium,
    /// 8 kHz, 16 kHz effective sample rate.
    Wide,
    /// 12 kHz, 24 kHz effective sample rate.
    SuperWide,
    /// 20 kHz, 48 kHz effective sample rate.
    Full,
}

impl Bandwidth {
    /// The CELT band after the last coded one: 13, 17, 17, 19, or 21.
    pub(crate) fn celt_end_band(self) -> usize {
        match self {
            Self::Narrow => 13,
            Self::Medium | Self::Wide => 17,
            Self::SuperWide => 19,
            Self::Full => 21,
        }
    }

    /// SILK internal rate for this bandwidth. Hybrid runs SILK at 16 kHz.
    pub(crate) fn silk_rate(self) -> u32 {
        match self {
            Self::Narrow => 8_000,
            Self::Medium => 12_000,
            Self::Wide | Self::SuperWide | Self::Full => 16_000,
        }
    }
}

/// The table-of-contents byte (RFC 6716 section 3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Toc {
    pub mode: Mode,
    pub bandwidth: Bandwidth,
    /// Duration of one frame, in 48 kHz samples: 120, 240, 480, 960, 1920, or 2880.
    pub frame_samples: u32,
    pub stereo: bool,
    /// Frame packing code, 0 to 3.
    pub code: u8,
}

impl Toc {
    #[must_use]
    pub fn parse(byte: u8) -> Self {
        let config = byte >> 3;
        let stereo = byte & 0x04 != 0;
        let code = byte & 0x03;
        let (mode, bandwidth, frame_samples) = match config {
            0..=11 => {
                let bandwidth = match config / 4 {
                    0 => Bandwidth::Narrow,
                    1 => Bandwidth::Medium,
                    _ => Bandwidth::Wide,
                };
                let samples = [480, 960, 1920, 2880][(config % 4) as usize];
                (Mode::SilkOnly, bandwidth, samples)
            }
            12..=15 => {
                let bandwidth = if config < 14 {
                    Bandwidth::SuperWide
                } else {
                    Bandwidth::Full
                };
                let samples = [480, 960][(config % 2) as usize];
                (Mode::Hybrid, bandwidth, samples)
            }
            _ => {
                let bandwidth = match (config - 16) / 4 {
                    0 => Bandwidth::Narrow,
                    1 => Bandwidth::Wide,
                    2 => Bandwidth::SuperWide,
                    _ => Bandwidth::Full,
                };
                let samples = [120, 240, 480, 960][(config % 4) as usize];
                (Mode::CeltOnly, bandwidth, samples)
            }
        };
        Self {
            mode,
            bandwidth,
            frame_samples,
            stereo,
            code,
        }
    }

    /// The configuration number 0 to 31 for this mode, bandwidth, and duration.
    pub(crate) fn config(mode: Mode, bandwidth: Bandwidth, frame_samples: u32) -> Option<u8> {
        let duration = |sizes: &[u32]| sizes.iter().position(|size| *size == frame_samples);
        match mode {
            Mode::SilkOnly => {
                let base = match bandwidth {
                    Bandwidth::Narrow => 0,
                    Bandwidth::Medium => 4,
                    Bandwidth::Wide => 8,
                    _ => return None,
                };
                Some(base + duration(&[480, 960, 1920, 2880])? as u8)
            }
            Mode::Hybrid => {
                let base = match bandwidth {
                    Bandwidth::SuperWide => 12,
                    Bandwidth::Full => 14,
                    _ => return None,
                };
                Some(base + duration(&[480, 960])? as u8)
            }
            Mode::CeltOnly => {
                let base = match bandwidth {
                    Bandwidth::Narrow => 16,
                    Bandwidth::Wide => 20,
                    Bandwidth::SuperWide => 24,
                    Bandwidth::Full => 28,
                    Bandwidth::Medium => return None,
                };
                Some(base + duration(&[120, 240, 480, 960])? as u8)
            }
        }
    }

    /// The TOC byte for one configuration, channel flag, and packing code.
    pub(crate) fn byte(config: u8, stereo: bool, code: u8) -> u8 {
        (config << 3) | (u8::from(stereo) << 2) | (code & 3)
    }
}

/// The frames of one packet, as byte ranges into that packet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frames {
    pub toc: Toc,
    /// Offset and length of each frame. A length of 0 is a frame the
    /// encoder dropped (DTX) or lost; the decoder conceals it.
    pub frames: Vec<(usize, usize)>,
    /// Bytes this packet took from the buffer, including padding. For a
    /// self-delimited packet this is where the next stream's packet starts.
    pub consumed: usize,
    /// Padding bytes after the last frame.
    pub padding: usize,
}

impl Frames {
    /// Samples per channel at 48 kHz this packet decodes to.
    #[must_use]
    pub fn samples(&self) -> u32 {
        self.toc.frame_samples * self.frames.len() as u32
    }
}

fn malformed(at: usize, what: &'static str) -> CodecError {
    CodecError::Malformed {
        at_bit: (at as u64) * 8,
        what,
    }
}

fn truncated(at: usize, what: &'static str) -> CodecError {
    CodecError::Truncated {
        at_bit: (at as u64) * 8,
        what,
    }
}

/// A one- or two-byte frame length (RFC 6716 section 3.2.1).
fn read_length(data: &[u8], at: usize) -> Result<(usize, usize), CodecError> {
    let Some(&first) = data.get(at) else {
        return Err(truncated(at, "frame length"));
    };
    if first < 252 {
        return Ok((usize::from(first), 1));
    }
    let Some(&second) = data.get(at + 1) else {
        return Err(truncated(at + 1, "frame length"));
    };
    Ok((usize::from(first) + 4 * usize::from(second), 2))
}

/// Splits one packet into frames.
///
/// `self_delimited` reads the extra length appendix B adds; that packet may
/// be followed by more bytes, and `consumed` says where it ends.
pub fn parse(data: &[u8], self_delimited: bool) -> Result<Frames, CodecError> {
    let Some(&toc_byte) = data.first() else {
        return Err(truncated(0, "toc"));
    };
    let toc = Toc::parse(toc_byte);
    let mut at = 1usize;
    let mut frames = Vec::new();
    let mut padding = 0usize;
    let end;
    match toc.code {
        0 => {
            let len = if self_delimited {
                let (len, used) = read_length(data, at)?;
                at += used;
                len
            } else {
                data.len() - at
            };
            frames.push((at, len));
            end = at + len;
        }
        1 => {
            if self_delimited {
                let (len, used) = read_length(data, at)?;
                at += used;
                frames.push((at, len));
                frames.push((at + len, len));
                end = at + 2 * len;
            } else {
                let rest = data.len() - at;
                if rest % 2 != 0 {
                    return Err(malformed(at, "odd code 1 payload"));
                }
                frames.push((at, rest / 2));
                frames.push((at + rest / 2, rest / 2));
                end = data.len();
            }
        }
        2 => {
            let (first, used) = read_length(data, at)?;
            at += used;
            let second = if self_delimited {
                let (len, used) = read_length(data, at)?;
                at += used;
                len
            } else {
                data.len()
                    .checked_sub(at + first)
                    .ok_or_else(|| truncated(at, "code 2 frame"))?
            };
            frames.push((at, first));
            frames.push((at + first, second));
            end = at + first + second;
        }
        _ => {
            let Some(&count_byte) = data.get(at) else {
                return Err(truncated(at, "frame count"));
            };
            at += 1;
            let vbr = count_byte & 0x80 != 0;
            let padded = count_byte & 0x40 != 0;
            let count = usize::from(count_byte & 0x3F);
            if count == 0 {
                return Err(malformed(at - 1, "zero frame count"));
            }
            if toc.frame_samples as usize * count > MAX_PACKET_SAMPLES as usize {
                return Err(malformed(at - 1, "packet longer than 120 ms"));
            }
            if padded {
                // RFC 8251 section 4: add 254 per 255 byte so a long run
                // cannot wrap the total.
                loop {
                    let Some(&byte) = data.get(at) else {
                        return Err(truncated(at, "padding length"));
                    };
                    at += 1;
                    if byte == 255 {
                        padding += 254;
                    } else {
                        padding += usize::from(byte);
                        break;
                    }
                    if padding > data.len() {
                        return Err(malformed(at, "padding past the end"));
                    }
                }
            }
            let mut lengths = Vec::with_capacity(count);
            if vbr {
                let known = if self_delimited { count } else { count - 1 };
                for _ in 0..known {
                    let (len, used) = read_length(data, at)?;
                    at += used;
                    lengths.push(len);
                }
                if !self_delimited {
                    let used: usize = lengths.iter().sum();
                    let last = data
                        .len()
                        .checked_sub(at)
                        .and_then(|rest| rest.checked_sub(padding))
                        .and_then(|rest| rest.checked_sub(used))
                        .ok_or_else(|| malformed(at, "padding past the end"))?;
                    lengths.push(last);
                }
            } else if self_delimited {
                let (len, used) = read_length(data, at)?;
                at += used;
                lengths.resize(count, len);
            } else {
                let rest = data
                    .len()
                    .checked_sub(at)
                    .and_then(|rest| rest.checked_sub(padding))
                    .ok_or_else(|| malformed(at, "padding past the end"))?;
                if rest % count != 0 {
                    return Err(malformed(at, "uneven constant-rate frames"));
                }
                lengths.resize(count, rest / count);
            }
            let mut offset = at;
            for len in lengths {
                frames.push((offset, len));
                offset += len;
            }
            end = offset + padding;
        }
    }
    for &(_, len) in &frames {
        if len > MAX_FRAME_BYTES {
            return Err(malformed(at, "frame longer than 1275 bytes"));
        }
    }
    let data_end = frames.last().map_or(at, |&(offset, len)| offset + len);
    if data_end > data.len() || end > data.len() {
        return Err(truncated(data.len(), "frame data"));
    }
    if toc.code == 2 && frames.first().is_some_and(|&(_, len)| len > data.len()) {
        return Err(truncated(at, "code 2 frame"));
    }
    Ok(Frames {
        toc,
        frames,
        consumed: end,
        padding,
    })
}

/// Writes a one- or two-byte frame length.
pub(crate) fn write_length(out: &mut Vec<u8>, len: usize) {
    if len < 252 {
        out.push(len as u8);
    } else {
        let low = 252 + (len & 3);
        out.push(low as u8);
        out.push(((len - low) >> 2) as u8);
    }
}

/// Splits `pad` added bytes into padding-length bytes and padding content.
///
/// `n` length bytes code between `254 * (n - 1)` and `254 * n` content bytes
/// (RFC 6716 section 3.2.5).
fn padding_split(pad: usize) -> (usize, usize) {
    let mut length_bytes = 1;
    loop {
        let content = pad.saturating_sub(length_bytes);
        if content <= 254 * length_bytes || length_bytes >= pad {
            return (length_bytes, content);
        }
        length_bytes += 1;
    }
}

/// Packs frames that share one TOC into a packet.
///
/// One frame is code 0. Two equal frames are code 1. Two unequal frames are
/// code 2. More are code 3, constant-rate when every frame has one size.
/// `pad_to` grows a code 3 packet to that many bytes with padding, for a
/// constant bitrate; a smaller target is ignored.
pub(crate) fn pack(
    config: u8,
    stereo: bool,
    frames: &[&[u8]],
    pad_to: Option<usize>,
    self_delimited: bool,
) -> Result<Vec<u8>, CodecError> {
    if frames.is_empty() || frames.len() > MAX_FRAMES {
        return Err(CodecError::InvalidApi {
            what: "frame count",
        });
    }
    if frames.iter().any(|frame| frame.len() > MAX_FRAME_BYTES) {
        return Err(CodecError::Limit {
            what: "frame longer than 1275 bytes",
        });
    }
    let total: usize = frames.iter().map(|frame| frame.len()).sum();
    let mut out = Vec::with_capacity(total + 8);
    let equal = frames.iter().all(|frame| frame.len() == frames[0].len());
    let wants_padding = pad_to.is_some_and(|target| target > 1 + total);
    if frames.len() == 1 && !wants_padding {
        out.push(Toc::byte(config, stereo, 0));
        if self_delimited {
            write_length(&mut out, frames[0].len());
        }
    } else if frames.len() == 2 && equal && !wants_padding {
        out.push(Toc::byte(config, stereo, 1));
        if self_delimited {
            write_length(&mut out, frames[0].len());
        }
    } else if frames.len() == 2 && !wants_padding {
        out.push(Toc::byte(config, stereo, 2));
        write_length(&mut out, frames[0].len());
        if self_delimited {
            write_length(&mut out, frames[1].len());
        }
    } else {
        out.push(Toc::byte(config, stereo, 3));
        let vbr = !equal;
        let mut header = Vec::new();
        if vbr {
            let known = if self_delimited {
                frames.len()
            } else {
                frames.len() - 1
            };
            for frame in &frames[..known] {
                write_length(&mut header, frame.len());
            }
        } else if self_delimited {
            write_length(&mut header, frames[0].len());
        }
        let base = 2 + header.len() + total;
        let padding = match pad_to {
            Some(target) if target > base => Some(padding_split(target - base)),
            _ => None,
        };
        let count = frames.len() as u8 | if vbr { 0x80 } else { 0 };
        out.push(count | if padding.is_some() { 0x40 } else { 0 });
        if let Some((length_bytes, content)) = padding {
            let mut left = content;
            for _ in 0..length_bytes.saturating_sub(1) {
                out.push(255);
                left = left.saturating_sub(254);
            }
            out.push(left as u8);
        }
        out.extend_from_slice(&header);
        for frame in frames {
            out.extend_from_slice(frame);
        }
        if let Some((_, content)) = padding {
            out.resize(out.len() + content, 0);
        }
        return Ok(out);
    }
    for frame in frames {
        out.extend_from_slice(frame);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{Bandwidth, MAX_FRAME_BYTES, Mode, Toc, pack, parse};
    use cutlass_decode::CodecError;

    #[test]
    fn every_toc_config_names_its_mode_bandwidth_and_duration() {
        let silk = Toc::parse(0 << 3);
        assert_eq!(silk.mode, Mode::SilkOnly);
        assert_eq!(silk.bandwidth, Bandwidth::Narrow);
        assert_eq!(silk.frame_samples, 480);
        let silk_mb_60 = Toc::parse(7 << 3);
        assert_eq!(silk_mb_60.bandwidth, Bandwidth::Medium);
        assert_eq!(silk_mb_60.frame_samples, 2880);
        let hybrid = Toc::parse(15 << 3 | 4);
        assert_eq!(hybrid.mode, Mode::Hybrid);
        assert_eq!(hybrid.bandwidth, Bandwidth::Full);
        assert_eq!(hybrid.frame_samples, 960);
        assert!(hybrid.stereo);
        let celt = Toc::parse(16 << 3 | 3);
        assert_eq!(celt.mode, Mode::CeltOnly);
        assert_eq!(celt.bandwidth, Bandwidth::Narrow);
        assert_eq!(celt.frame_samples, 120);
        assert_eq!(celt.code, 3);
        for config in 0..32u8 {
            let toc = Toc::parse(config << 3);
            assert_eq!(
                Toc::config(toc.mode, toc.bandwidth, toc.frame_samples),
                Some(config)
            );
        }
    }

    #[test]
    fn code_zero_is_one_frame_and_code_one_two_halves() {
        let packet = [0xF8, 1, 2, 3];
        let frames = parse(&packet, false).unwrap();
        assert_eq!(frames.frames, vec![(1, 3)]);
        let packet = [0xF9, 1, 2, 3, 4];
        let frames = parse(&packet, false).unwrap();
        assert_eq!(frames.frames, vec![(1, 2), (3, 2)]);
        assert!(matches!(
            parse(&[0xF9, 1, 2, 3], false),
            Err(CodecError::Malformed { .. })
        ));
    }

    #[test]
    fn code_two_reads_a_two_byte_length() {
        let mut packet = vec![0xFA, 252, 1];
        packet.extend(vec![7; 256 + 10]);
        let frames = parse(&packet, false).unwrap();
        assert_eq!(frames.frames, vec![(3, 256), (259, 10)]);
        assert!(parse(&[0xFA, 10, 1, 2], false).is_err());
    }

    #[test]
    fn code_three_constant_and_variable_rate_with_padding() {
        // Three 2-byte frames, constant rate, two padding bytes.
        let packet = [0xFB, 0x43, 2, 1, 1, 2, 2, 3, 3, 0, 0];
        let frames = parse(&packet, false).unwrap();
        assert_eq!(frames.frames, vec![(3, 2), (5, 2), (7, 2)]);
        assert_eq!(frames.padding, 2);
        // Two variable frames: first 1 byte, the rest is the second.
        let packet = [0xFB, 0x82, 1, 9, 8, 8];
        let frames = parse(&packet, false).unwrap();
        assert_eq!(frames.frames, vec![(3, 1), (4, 2)]);
    }

    #[test]
    fn a_frame_count_over_120_ms_is_refused() {
        // 20 ms CELT frames, 7 of them is 140 ms.
        let mut packet = vec![0xFB, 7];
        packet.extend(vec![0; 7]);
        assert!(matches!(
            parse(&packet, false),
            Err(CodecError::Malformed { .. })
        ));
        // 49 frames of 2.5 ms is past the 48-frame limit too.
        let mut packet = vec![(16 << 3) | 3, 49];
        packet.extend(vec![0; 49]);
        assert!(parse(&packet, false).is_err());
        assert!(matches!(
            parse(&[0xFB, 0], false),
            Err(CodecError::Malformed { .. })
        ));
    }

    #[test]
    fn padding_past_the_end_is_refused_without_wrapping() {
        let mut packet = vec![0xFB, 0x41];
        packet.extend(vec![255; 4000]);
        packet.push(10);
        assert!(parse(&packet, false).is_err());
        let packet = [0xFB, 0x41, 20, 1, 2];
        assert!(parse(&packet, false).is_err());
        assert!(parse(&[0xFB, 0x41], false).is_err());
    }

    #[test]
    fn an_empty_packet_and_a_long_frame_are_refused() {
        assert!(matches!(parse(&[], false), Err(CodecError::Truncated { .. })));
        let mut packet = vec![0xF8];
        packet.extend(vec![0; MAX_FRAME_BYTES + 1]);
        assert!(parse(&packet, false).is_err());
    }

    #[test]
    fn self_delimited_packets_end_where_their_lengths_say() {
        let first = pack(31, false, &[&[1, 2, 3]], None, true).unwrap();
        let mut both = first.clone();
        both.extend([0xF8, 9, 9]);
        let frames = parse(&both, true).unwrap();
        assert_eq!(frames.consumed, first.len());
        assert_eq!(frames.frames.len(), 1);
        let rest = parse(&both[frames.consumed..], false).unwrap();
        assert_eq!(rest.frames, vec![(1, 2)]);
    }

    #[test]
    fn packing_round_trips_every_code() {
        let a: Vec<u8> = (0..300).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..20).map(|i| i as u8).collect();
        let cases: Vec<Vec<&[u8]>> = vec![
            vec![&a],
            vec![&b, &b],
            vec![&a, &b],
            vec![&b, &b, &b],
            vec![&a, &b, &a],
        ];
        for delimited in [false, true] {
            for frames in &cases {
                let packet = pack(28, true, frames, None, delimited).unwrap();
                let parsed = parse(&packet, delimited).unwrap();
                assert_eq!(parsed.consumed, packet.len());
                assert_eq!(parsed.frames.len(), frames.len());
                for (&(offset, len), frame) in parsed.frames.iter().zip(frames) {
                    assert_eq!(&packet[offset..offset + len], *frame);
                }
            }
        }
    }

    #[test]
    fn padding_lands_on_the_requested_size() {
        let b: Vec<u8> = (0..20).map(|i| i as u8).collect();
        for target in [30usize, 100, 300, 600, 1000] {
            let packet = pack(28, false, &[&b], Some(target), false).unwrap();
            assert_eq!(packet.len(), target, "target {target}");
            let parsed = parse(&packet, false).unwrap();
            assert_eq!(parsed.frames.len(), 1);
            let (offset, len) = parsed.frames[0];
            assert_eq!(&packet[offset..offset + len], b.as_slice());
        }
    }
}
