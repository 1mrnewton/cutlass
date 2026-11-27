//! Sample descriptions. Configuration records stay bytes: `avcC`, `hvcC`,
//! `av1C`, and the decoder-specific info inside `esds`. Nothing here decodes them.

use crate::alac;
use crate::index::Framing;
use crate::legacy::{AudioParams, CodecId, ColorTags, Error, MediaKind, SampleFormat, VideoParams};
use crate::limits::MAX_STSD;
use crate::parse::{self, Reader};

pub struct SampleDesc {
    pub fourcc: [u8; 4],
    pub codec: CodecId,
    pub kind: MediaKind,
    pub extradata: Vec<u8>,
    pub color: Option<ColorTags>,
    pub data_reference: u16,
    /// For constant-size audio, the block the payload is cut on when the
    /// entry states it: one PCM frame, or a version-1 compressed block.
    pub framing: Option<Framing>,
}

pub fn parse_stsd(
    payload: &[u8],
    handler: [u8; 4],
    budget: &mut u32,
    depth: u32,
) -> Result<Vec<SampleDesc>, Error> {
    let depth = parse::enter(depth)?;
    let mut reader = Reader::new(payload, 0);
    let _version = reader.u8()?;
    reader.skip(3)?;
    let count = reader.u32()?;
    if u64::from(count) > u64::try_from(MAX_STSD).unwrap_or(0) {
        return Err(Error::Limit {
            what: "sample descriptions",
            value: u64::from(count),
        });
    }
    let entries = parse::atoms(reader.remaining(), 0, budget)?;
    if entries.len() != usize::try_from(count).unwrap_or(usize::MAX) && count != 0 {
        // A short table is a truncated description, not a silent drop.
        if entries.len() < usize::try_from(count).unwrap_or(0) {
            return Err(Error::Truncated {
                at: 0,
                needed: u64::from(count),
            });
        }
    }
    let mut out = Vec::new();
    for entry in entries {
        if out.len() >= MAX_STSD {
            return Err(Error::Limit {
                what: "sample descriptions",
                value: u64::try_from(out.len()).unwrap_or(u64::MAX),
            });
        }
        out.push(parse_entry(
            entry.kind,
            entry.payload,
            handler,
            budget,
            depth,
        )?);
    }
    Ok(out)
}

fn parse_entry(
    fourcc: [u8; 4],
    payload: &[u8],
    handler: [u8; 4],
    budget: &mut u32,
    depth: u32,
) -> Result<SampleDesc, Error> {
    let audio = known_audio(fourcc) || (handler == *b"soun" && !known_video(fourcc));
    let video = !audio && (known_video(fourcc) || handler == *b"vide");
    if audio {
        parse_audio(fourcc, payload, budget, depth)
    } else if video {
        parse_visual(fourcc, payload, budget, depth)
    } else {
        let data_reference = payload.get(6..8).map(read_u16).unwrap_or(1);
        Ok(SampleDesc {
            fourcc,
            codec: CodecId::Unknown(fourcc),
            kind: MediaKind::Data,
            extradata: payload.to_vec(),
            color: None,
            data_reference,
            framing: None,
        })
    }
}

fn parse_visual(
    fourcc: [u8; 4],
    payload: &[u8],
    budget: &mut u32,
    depth: u32,
) -> Result<SampleDesc, Error> {
    if payload.len() < 78 {
        return Err(Error::Truncated { at: 0, needed: 78 });
    }
    let data_reference = read_u16(&payload[6..8]);
    let width = u32::from(read_u16(&payload[24..26]));
    let height = u32::from(read_u16(&payload[26..28]));
    let children = payload.get(78..).unwrap_or(&[]);
    let configured = configuration(children, budget, depth)?;
    let codec = map_codec(fourcc, configured.object_type);
    // A ProRes entry has no configuration box. Its code names the profile,
    // which sets the decoder's output depth, so it rides as the extradata.
    // A QuickTime Motion JPEG code names its framing, and `fiel` the stored
    // field order, so both ride there too. ISO/IEC 14496-1 defines no
    // decoder-specific info for JPEG (0x6C), so an `mp4v` entry's is
    // dropped, as FFmpeg's decoder ignores it.
    let extradata = if codec == CodecId::ProRes && configured.extradata.is_empty() {
        fourcc.to_vec()
    } else if codec == CodecId::Mjpeg && fourcc == *b"mp4v" {
        Vec::new()
    } else if codec == CodecId::Mjpeg && configured.extradata.is_empty() {
        let mut extradata = fourcc.to_vec();
        if let Some(fiel) = configured.fiel {
            extradata.extend_from_slice(&fiel);
        }
        extradata
    } else {
        configured.extradata
    };
    Ok(SampleDesc {
        codec,
        fourcc,
        kind: MediaKind::Video(VideoParams { width, height }),
        extradata,
        color: configured.color,
        data_reference,
        framing: None,
    })
}

fn parse_audio(
    fourcc: [u8; 4],
    payload: &[u8],
    budget: &mut u32,
    depth: u32,
) -> Result<SampleDesc, Error> {
    if payload.len() < 28 {
        return Err(Error::Truncated { at: 0, needed: 28 });
    }
    let data_reference = read_u16(&payload[6..8]);
    let version = read_u16(&payload[8..10]);
    let mut channels = read_u16(&payload[16..18]);
    let mut bits = read_u16(&payload[18..20]);
    let mut sample_rate = u32::from(read_u16(&payload[24..26]));
    let mut format = pcm_format(fourcc, bits);
    let mut child_at = 28usize;
    // A compressed block: its audio frames and its bytes over all channels.
    let mut block = None;
    // Version 2 states the bytes of one PCM frame.
    let mut stated_frame = 0u32;
    if version == 1 {
        // Version 1 adds samples per packet, bytes per packet, bytes per
        // frame, and bytes per sample.
        child_at = 44;
        if let (Some(per_packet), Some(per_frame)) = (read_u32(payload, 28), read_u32(payload, 36))
            && per_packet > 1
            && per_frame > 0
        {
            block = Some(Framing {
                frames: per_packet,
                bytes: per_frame,
            });
        }
    } else if version == 2 {
        // QuickTime sound description v2 replaces the rate and channel fields.
        if payload.len() < 64 {
            return Err(Error::Truncated { at: 0, needed: 64 });
        }
        let rate_bits = u64::from_be_bytes([
            payload[32],
            payload[33],
            payload[34],
            payload[35],
            payload[36],
            payload[37],
            payload[38],
            payload[39],
        ]);
        let rounded = f64::from_bits(rate_bits).round();
        // The v2 rate is a float64, so 96 kHz and up are stated exactly.
        sample_rate = if rounded.is_finite() && rounded > 0.0 && rounded <= f64::from(u32::MAX) {
            u32::try_from(rounded as u64).unwrap_or(0)
        } else {
            0
        };
        channels = u16::try_from(u32::from_be_bytes([
            payload[40],
            payload[41],
            payload[42],
            payload[43],
        ]))
        .unwrap_or(u16::MAX);
        bits = u16::try_from(u32::from_be_bytes([
            payload[48],
            payload[49],
            payload[50],
            payload[51],
        ]))
        .unwrap_or(0);
        let flags = u32::from_be_bytes([payload[52], payload[53], payload[54], payload[55]]);
        format = lpcm_format(bits, flags);
        stated_frame = read_u32(payload, 56).unwrap_or(0);
        child_at = 64;
    }
    if child_at > payload.len() {
        return Err(Error::Truncated {
            at: 0,
            needed: u64::try_from(child_at).unwrap_or(u64::MAX),
        });
    }
    let mut configured = configuration(&payload[child_at..], budget, depth)?;
    let codec = map_codec(fourcc, configured.object_type);
    if codec == CodecId::Pcm && format == SampleFormat::Unknown {
        format = pcm_format(fourcc, bits);
    }
    // ISO/IEC 23003-5 PCM: `pcmC` gives the byte order and the sample size.
    if matches!(&fourcc, b"ipcm" | b"fpcm") {
        let (little, size) = configured.pcm.unwrap_or((false, 0));
        if size != 0 {
            bits = u16::from(size);
        }
        format = iso_pcm_format(fourcc == *b"fpcm", little, bits);
    }
    // QuickTime's `in24`, `in32`, `fl32` and `fl64` are big-endian unless a
    // `wave` atom's `enda` says little, as FFmpeg writes them.
    if configured.little_endian && matches!(&fourcc, b"in24" | b"in32" | b"fl32" | b"fl64") {
        format = match format {
            SampleFormat::S24Be => SampleFormat::S24Le,
            SampleFormat::S32Be => SampleFormat::S32Le,
            SampleFormat::F32Be => SampleFormat::F32Le,
            SampleFormat::F64Be => SampleFormat::F64Le,
            other => other,
        };
    }
    if !matches!(codec, CodecId::Pcm) {
        format = SampleFormat::Unknown;
    }
    let mut rate = sample_rate;
    if codec == CodecId::Alac
        && let Some(cookie) = alac::join(&configured.extradata, configured.chan.as_deref())
        && let Some(facts) = alac::facts(&cookie)
    {
        // The cookie is the stream's own description. Writers leave two
        // channels in the entry for more, and the entry's rate stops at 65535.
        channels = facts.channels;
        bits = facts.bits;
        if facts.rate != 0 {
            rate = facts.rate;
        }
        configured.extradata = cookie;
    }
    if codec == CodecId::Flac
        && let Some(stated) = flac_rate(&configured.extradata)
    {
        // STREAMINFO is the rate the decoder gives. The entry's field stops
        // at 65535, and FFmpeg halves a higher rate until it fits there, so
        // a 96 kHz track says 48000.
        rate = stated;
    }
    if rate == 0 {
        // The 16.16 rate field stops at 65535, and some writers store 0
        // above it. The codec configuration states the rate itself.
        rate = match codec {
            CodecId::Aac => aac_rate(&configured.extradata).unwrap_or(0),
            CodecId::Flac => flac_rate(&configured.extradata).unwrap_or(0),
            // Opus decodes at 48 kHz whatever the input rate in `dOps` was.
            CodecId::Opus => 48_000,
            _ => 0,
        };
    }
    // The bytes of one frame over all channels. FFmpeg's mov.c frames PCM
    // from the entry and not from `stsz`, where classic QuickTime writes 1.
    // An `ipcm` or `fpcm` track keeps its `stsz` size, as FFmpeg reads it:
    // FFmpeg 6.1 wrote 24-bit samples under a `pcmC` that says 32.
    let frame = if matches!(&fourcc, b"ipcm" | b"fpcm") {
        0
    } else if codec == CodecId::Pcm {
        let computed = u32::from(channels) * format_bytes(format);
        if stated_frame != 0 {
            stated_frame
        } else {
            computed
        }
    } else if matches!(&fourcc, b"alaw" | b"ulaw") {
        u32::from(channels)
    } else {
        0
    };
    let framing = if frame != 0 {
        Some(Framing {
            frames: 1,
            bytes: frame,
        })
    } else {
        block
    };
    Ok(SampleDesc {
        fourcc,
        codec,
        kind: MediaKind::Audio(AudioParams {
            sample_rate: rate,
            channels,
            bits_per_sample: bits,
            format,
        }),
        extradata: configured.extradata,
        color: configured.color,
        data_reference,
        framing,
    })
}

const fn format_bytes(format: SampleFormat) -> u32 {
    match format {
        SampleFormat::Unknown => 0,
        SampleFormat::U8 | SampleFormat::S8 => 1,
        SampleFormat::S16Le | SampleFormat::S16Be => 2,
        SampleFormat::S24Le | SampleFormat::S24Be => 3,
        SampleFormat::S32Le | SampleFormat::S32Be | SampleFormat::F32Le | SampleFormat::F32Be => 4,
        SampleFormat::F64Le | SampleFormat::F64Be => 8,
    }
}

fn iso_pcm_format(float: bool, little: bool, bits: u16) -> SampleFormat {
    match (float, little, bits) {
        (true, true, 64) => SampleFormat::F64Le,
        (true, false, 64) => SampleFormat::F64Be,
        (true, true, 32) => SampleFormat::F32Le,
        (true, false, 32) => SampleFormat::F32Be,
        (false, true, 16) => SampleFormat::S16Le,
        (false, false, 16) => SampleFormat::S16Be,
        (false, true, 24) => SampleFormat::S24Le,
        (false, false, 24) => SampleFormat::S24Be,
        (false, true, 32) => SampleFormat::S32Le,
        (false, false, 32) => SampleFormat::S32Be,
        _ => SampleFormat::Unknown,
    }
}

/// MPEG-4 sampling frequencies by index (ISO/IEC 14496-3 table 1.18).
const AAC_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// The output rate an `AudioSpecificConfig` states: the extension rate when
/// it signals SBR or PS explicitly, as FFmpeg reports it, or the core rate.
fn aac_rate(config: &[u8]) -> Option<u32> {
    let mut bits = Bits {
        data: config,
        at: 0,
    };
    let object = aac_object(&mut bits)?;
    let core = aac_frequency(&mut bits)?;
    let _channels = bits.take(4)?;
    if matches!(object, 5 | 29) {
        return aac_frequency(&mut bits).or(Some(core));
    }
    Some(core)
}

fn aac_object(bits: &mut Bits<'_>) -> Option<u32> {
    match bits.take(5)? {
        31 => Some(32 + bits.take(6)?),
        object => Some(object),
    }
}

fn aac_frequency(bits: &mut Bits<'_>) -> Option<u32> {
    let rate = match bits.take(4)? {
        15 => bits.take(24)?,
        index => *AAC_RATES.get(usize::try_from(index).ok()?)?,
    };
    (rate != 0).then_some(rate)
}

struct Bits<'a> {
    data: &'a [u8],
    at: usize,
}

impl Bits<'_> {
    /// The next `count` bits, most significant first. `count` is at most 32.
    fn take(&mut self, count: usize) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..count {
            let byte = *self.data.get(self.at / 8)?;
            let bit = (byte >> (7 - self.at % 8)) & 1;
            value = (value << 1) | u32::from(bit);
            self.at += 1;
        }
        Some(value)
    }
}

/// The rate in a `dfLa` box: four bytes of version and flags, a metadata
/// block header, then STREAMINFO, whose rate is 20 bits after ten bytes.
fn flac_rate(dfla: &[u8]) -> Option<u32> {
    let header = *dfla.get(4)?;
    if header & 0x7f != 0 {
        return None;
    }
    let [a, b, c] = *dfla.get(18..21)? else {
        return None;
    };
    let rate = (u32::from(a) << 12) | (u32::from(b) << 4) | (u32::from(c) >> 4);
    (rate != 0).then_some(rate)
}

struct Configured {
    extradata: Vec<u8>,
    /// A `chan` box beside the configuration: the ALAC channel layout.
    chan: Option<Vec<u8>>,
    color: Option<ColorTags>,
    object_type: Option<u8>,
    /// A `fiel` box: the field count, then the stored order.
    fiel: Option<[u8; 2]>,
    /// An `enda` box with a nonzero value: PCM samples are little-endian.
    little_endian: bool,
    /// A `pcmC` box: whether samples are little-endian, and their bits.
    pcm: Option<(bool, u8)>,
}

fn configuration(data: &[u8], budget: &mut u32, depth: u32) -> Result<Configured, Error> {
    let mut extradata = Vec::new();
    let mut color = None;
    let mut object_type = None;
    let mut chan = None;
    let mut fiel = None;
    let mut little_endian = false;
    let mut pcm = None;
    let boxes = parse::atoms(data, 0, budget)?;
    for atom in boxes {
        match &atom.kind {
            b"avcC" | b"hvcC" | b"av1C" | b"alac" | b"dOps" | b"dac3" | b"dec3" | b"dfLa" => {
                extradata = atom.payload.to_vec();
            }
            b"vpcC" => extradata = atom.payload.to_vec(),
            b"esds" => {
                if let Some(info) = esds(atom.payload)? {
                    object_type = info.oti.or(object_type);
                    if extradata.is_empty()
                        && let Some(specific) = info.specific
                    {
                        extradata = specific;
                    }
                }
            }
            b"wave" => {
                let nested = configuration(atom.payload, budget, parse::enter(depth)?)?;
                if extradata.is_empty() {
                    extradata = nested.extradata;
                }
                color = color.or(nested.color);
                object_type = object_type.or(nested.object_type);
                chan = chan.or(nested.chan);
                little_endian |= nested.little_endian;
                pcm = pcm.or(nested.pcm);
            }
            b"pcmC" => {
                if let [_, _, _, _, flags, size, ..] = *atom.payload {
                    pcm = Some((flags & 1 != 0, size));
                }
            }
            b"enda" => little_endian = matches!(*atom.payload, [_, flag, ..] if flag != 0),
            b"colr" => color = color.or(parse_colr(atom.payload)),
            b"chan" => chan = Some(atom.payload.to_vec()),
            b"fiel" => {
                if let [count, order, ..] = *atom.payload {
                    fiel = Some([count, order]);
                }
            }
            _ => {}
        }
    }
    Ok(Configured {
        extradata,
        chan,
        color,
        object_type,
        fiel,
        little_endian,
        pcm,
    })
}

struct EsInfo {
    oti: Option<u8>,
    specific: Option<Vec<u8>>,
}

fn esds(payload: &[u8]) -> Result<Option<EsInfo>, Error> {
    if payload.len() < 4 {
        return Err(Error::Truncated { at: 0, needed: 4 });
    }
    let mut found = None;
    for desc in descriptors(&payload[4..])? {
        if desc.tag != 0x03 {
            continue;
        }
        found = walk_es(desc.body)?;
    }
    Ok(found)
}

struct Desc<'a> {
    tag: u8,
    body: &'a [u8],
}

fn descriptors(data: &[u8]) -> Result<Vec<Desc<'_>>, Error> {
    let mut reader = Reader::new(data, 0);
    let mut out = Vec::new();
    while !reader.is_empty() {
        let tag = reader.u8()?;
        let mut size = 0usize;
        let mut saw_end = false;
        for _ in 0..4 {
            let byte = reader.u8()?;
            size = size
                .checked_shl(7)
                .and_then(|value| value.checked_add(usize::from(byte & 0x7f)))
                .ok_or(Error::HostileSize {
                    at: 0,
                    claimed: u64::MAX,
                })?;
            if byte & 0x80 == 0 {
                saw_end = true;
                break;
            }
        }
        if !saw_end {
            return Err(Error::HostileSize {
                at: 0,
                claimed: u64::try_from(size).unwrap_or(u64::MAX),
            });
        }
        out.push(Desc {
            tag,
            body: reader.take(size)?,
        });
    }
    Ok(out)
}

fn walk_es(body: &[u8]) -> Result<Option<EsInfo>, Error> {
    if body.len() < 3 {
        return Ok(None);
    }
    let flags = body[2];
    let mut index = 3usize;
    if flags & 0x80 != 0 {
        index = index.saturating_add(2);
    }
    if flags & 0x40 != 0 {
        let Some(len) = body.get(index).copied() else {
            return Err(Error::Truncated {
                at: u64::try_from(index).unwrap_or(u64::MAX),
                needed: u64::try_from(index).unwrap_or(0).saturating_add(1),
            });
        };
        index = index.saturating_add(1 + usize::from(len));
    }
    if flags & 0x20 != 0 {
        index = index.saturating_add(2);
    }
    let Some(rest) = body.get(index..) else {
        return Ok(None);
    };
    let mut oti = None;
    let mut specific = None;
    for desc in descriptors(rest)? {
        if desc.tag != 0x04 || desc.body.is_empty() {
            continue;
        }
        oti = Some(desc.body[0]);
        let Some(nested) = desc.body.get(13..) else {
            continue;
        };
        for inner in descriptors(nested)? {
            if inner.tag == 0x05 {
                specific = Some(inner.body.to_vec());
            }
        }
    }
    Ok(Some(EsInfo { oti, specific }))
}

fn parse_colr(payload: &[u8]) -> Option<ColorTags> {
    let kind = payload.get(..4)?;
    if kind != b"nclx" && kind != b"nclc" {
        return None;
    }
    if payload.len() < 10 {
        return None;
    }
    let full_range = kind == b"nclx" && payload.get(10).is_some_and(|byte| byte & 0x80 != 0);
    Some(ColorTags {
        primaries: read_u16(&payload[4..6]),
        transfer: read_u16(&payload[6..8]),
        matrix: read_u16(&payload[8..10]),
        full_range,
    })
}

fn known_audio(fourcc: [u8; 4]) -> bool {
    matches!(
        &fourcc,
        b"mp4a"
            | b"Opus"
            | b"fLaC"
            | b"alac"
            | b"ac-3"
            | b"ec-3"
            | b"sac3"
            | b"sowt"
            | b"twos"
            | b"raw "
            | b"in24"
            | b"in32"
            | b"fl32"
            | b"fl64"
            | b"lpcm"
            | b"ipcm"
            | b"fpcm"
            | b"alaw"
            | b"ulaw"
            | b"NONE"
            | b".mp3"
            | b"samr"
            | b"sawb"
    )
}

fn known_video(fourcc: [u8; 4]) -> bool {
    matches!(
        &fourcc,
        b"avc1"
            | b"avc3"
            | b"hvc1"
            | b"hev1"
            | b"dvh1"
            | b"dvhe"
            | b"av01"
            | b"vp08"
            | b"vp09"
            | b"mp4v"
            | b"mp2v"
            | b"jpeg"
            | b"mjpa"
            | b"mjpb"
            | b"apch"
            | b"apcn"
            | b"apcs"
            | b"apco"
            | b"ap4h"
            | b"ap4x"
            | b"encv"
    ) || mpeg_video(fourcc)
}

/// QuickTime sample entries for MPEG-1 and MPEG-2 video: camcorder, HDV,
/// IMX, and XDCAM. Both are one H.262 elementary stream.
fn mpeg_video(fourcc: [u8; 4]) -> bool {
    matches!(
        &fourcc,
        b"mp2v"
            | b"m2v1"
            | b"hdv1"
            | b"hdv2"
            | b"hdv3"
            | b"hdv4"
            | b"hdv5"
            | b"hdv6"
            | b"hdv7"
            | b"hdv8"
            | b"hdv9"
            | b"hdva"
            | b"mx3n"
            | b"mx3p"
            | b"mx4n"
            | b"mx4p"
            | b"mx5n"
            | b"mx5p"
            | b"xd51"
            | b"xd54"
            | b"xd55"
            | b"xd59"
            | b"xd5a"
            | b"xd5b"
            | b"xd5c"
            | b"xd5d"
            | b"xd5e"
            | b"xd5f"
            | b"xdv1"
            | b"xdv2"
            | b"xdv3"
            | b"xdv4"
            | b"xdv5"
            | b"xdv6"
            | b"xdv7"
            | b"xdv8"
            | b"xdv9"
            | b"xdva"
            | b"xdvb"
            | b"xdvc"
            | b"xdvd"
            | b"xdve"
            | b"xdvf"
            | b"xdhd"
            | b"xdh2"
            | b"AVmp"
            | b"m1v "
            | b"m1v1"
            | b"mpeg"
            | b"mp1v"
    )
}

fn map_codec(fourcc: [u8; 4], oti: Option<u8>) -> CodecId {
    match &fourcc {
        b"avc1" | b"avc3" => CodecId::H264,
        b"hvc1" | b"hev1" | b"dvh1" | b"dvhe" => CodecId::Hevc,
        b"av01" => CodecId::Av1,
        b"vp08" => CodecId::Vp8,
        b"vp09" => CodecId::Vp9,
        // ISO/IEC 14496-1 objectTypeIndication: MPEG-2 Simple, Main, SNR,
        // Spatial, High and 4:2:2, then MPEG-1; 0x6C is JPEG. An absent or
        // unlisted one stays MPEG-4 Visual (0x20).
        b"mp4v" => match oti {
            Some(0x60..=0x65 | 0x6A) => CodecId::Mpeg2,
            Some(0x6C) => CodecId::Mjpeg,
            _ => CodecId::Mpeg4,
        },
        _ if mpeg_video(fourcc) => CodecId::Mpeg2,
        b"jpeg" | b"mjpa" | b"mjpb" => CodecId::Mjpeg,
        b"apch" | b"apcn" | b"apcs" | b"apco" | b"ap4h" | b"ap4x" => CodecId::ProRes,
        b"Opus" => CodecId::Opus,
        b"fLaC" => CodecId::Flac,
        b"alac" => CodecId::Alac,
        b"ac-3" | b"sac3" => CodecId::Ac3,
        b"ec-3" => CodecId::Eac3,
        b"sowt" | b"twos" | b"raw " | b"in24" | b"in32" | b"fl32" | b"fl64" | b"lpcm" | b"ipcm"
        | b"fpcm" | b"NONE" => CodecId::Pcm,
        b".mp3" => CodecId::Mp3,
        b"mp4a" => match oti {
            Some(0x69 | 0x6B) => CodecId::Mp3,
            Some(0xA5) => CodecId::Ac3,
            Some(0xA6) => CodecId::Eac3,
            _ => CodecId::Aac,
        },
        _ => CodecId::Unknown(fourcc),
    }
}

#[must_use]
pub(crate) fn codec_for(fourcc: [u8; 4]) -> CodecId {
    map_codec(fourcc, None)
}

fn pcm_format(fourcc: [u8; 4], bits: u16) -> SampleFormat {
    match &fourcc {
        // 8-bit `sowt` and `twos` are signed, as FFmpeg's mov.c reads them;
        // unsigned 8-bit is `raw `.
        b"sowt" => match bits {
            8 => SampleFormat::S8,
            24 => SampleFormat::S24Le,
            32 => SampleFormat::S32Le,
            _ => SampleFormat::S16Le,
        },
        b"twos" | b"NONE" => match bits {
            8 => SampleFormat::S8,
            24 => SampleFormat::S24Be,
            32 => SampleFormat::S32Be,
            _ => SampleFormat::S16Be,
        },
        b"raw " => SampleFormat::U8,
        b"in24" => SampleFormat::S24Be,
        b"in32" => SampleFormat::S32Be,
        b"fl32" => SampleFormat::F32Be,
        b"fl64" => SampleFormat::F64Be,
        _ => SampleFormat::Unknown,
    }
}

fn lpcm_format(bits: u16, flags: u32) -> SampleFormat {
    let float = flags & 1 != 0;
    let big = flags & 2 != 0;
    match (float, big, bits) {
        (true, true, 64) => SampleFormat::F64Be,
        (true, false, 64) => SampleFormat::F64Le,
        (true, true, _) => SampleFormat::F32Be,
        (true, false, _) => SampleFormat::F32Le,
        // kAudioFormatFlagIsSignedInteger (4) marks signed 8-bit.
        (false, _, 8) if flags & 4 != 0 => SampleFormat::S8,
        (false, _, 8) => SampleFormat::U8,
        (false, true, 24) => SampleFormat::S24Be,
        (false, false, 24) => SampleFormat::S24Le,
        (false, true, 32) => SampleFormat::S32Be,
        (false, false, 32) => SampleFormat::S32Le,
        (false, true, _) => SampleFormat::S16Be,
        (false, false, _) => SampleFormat::S16Le,
    }
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    let end = at.checked_add(4)?;
    match bytes.get(at..end)? {
        [a, b, c, d] => Some(u32::from_be_bytes([*a, *b, *c, *d])),
        _ => None,
    }
}

fn read_u16(bytes: &[u8]) -> u16 {
    match bytes {
        [a, b, ..] => u16::from_be_bytes([*a, *b]),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = u32::try_from(payload.len() + 8)
            .unwrap()
            .to_be_bytes()
            .to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    fn visual(code: &[u8; 4], children: &[u8]) -> (CodecId, Vec<u8>) {
        let mut entry = vec![0u8; 78];
        entry[7] = 1;
        entry[24..26].copy_from_slice(&720u16.to_be_bytes());
        entry[26..28].copy_from_slice(&486u16.to_be_bytes());
        entry.extend_from_slice(children);
        let mut stsd = vec![0, 0, 0, 0, 0, 0, 0, 1];
        stsd.extend(boxed(code, &entry));
        let mut budget = 64;
        let mut entries = parse_stsd(&stsd, *b"vide", &mut budget, 0).unwrap();
        let desc = entries.remove(0);
        assert!(matches!(desc.kind, MediaKind::Video(video) if video.height == 486));
        (desc.codec, desc.extradata)
    }

    #[test]
    fn a_motion_jpeg_entry_carries_its_code_and_field_order() {
        assert_eq!(visual(b"jpeg", &[]), (CodecId::Mjpeg, b"jpeg".to_vec()));
        let fiel = boxed(b"fiel", &[2, 14]);
        assert_eq!(
            visual(b"mjpa", &fiel),
            (CodecId::Mjpeg, b"mjpa\x02\x0e".to_vec())
        );
        assert_eq!(
            visual(b"mjpb", &boxed(b"fiel", &[1, 0])),
            (CodecId::Mjpeg, b"mjpb\x01\x00".to_vec())
        );
        // A `fiel` box too short to name an order adds nothing.
        assert_eq!(
            visual(b"jpeg", &boxed(b"fiel", &[2])),
            (CodecId::Mjpeg, b"jpeg".to_vec())
        );
    }

    #[test]
    fn an_mp4v_entry_is_motion_jpeg_only_for_object_type_0x6c() {
        let esds = |oti: u8| {
            let mut payload = vec![0, 0, 0, 0, 3, 18, 0, 1, 0, 4, 13, oti, 0x11];
            payload.extend([0; 11]);
            boxed(b"esds", &payload)
        };
        assert_eq!(visual(b"mp4v", &esds(0x6C)), (CodecId::Mjpeg, Vec::new()));
        // A decoder-specific descriptor under object type 0x6C has no
        // meaning for JPEG and is not handed to the decoder.
        let mut specific = vec![0, 0, 0, 0, 3, 24, 0, 1, 0, 4, 19, 0x6C, 0x11];
        specific.extend([0; 11]);
        specific.extend([5, 4, 0x12, 0x34, 0x56, 0x78]);
        assert_eq!(
            visual(b"mp4v", &boxed(b"esds", &specific)),
            (CodecId::Mjpeg, Vec::new())
        );
        // The same descriptor under MPEG-4 Visual is its configuration.
        specific[11] = 0x20;
        assert_eq!(
            visual(b"mp4v", &boxed(b"esds", &specific)),
            (CodecId::Mpeg4, vec![0x12, 0x34, 0x56, 0x78])
        );
        assert_eq!(visual(b"mp4v", &esds(0x20)).0, CodecId::Mpeg4);
        assert_eq!(visual(b"mp4v", &esds(0x61)).0, CodecId::Mpeg2);
        assert_eq!(visual(b"mp4v", &esds(0x6A)).0, CodecId::Mpeg2);
        assert_eq!(visual(b"mp4v", &[]).0, CodecId::Mpeg4);
    }

    #[test]
    fn mp4v_is_mpeg2_only_for_an_mpeg2_or_mpeg1_object_type() {
        for oti in [0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x6A] {
            assert_eq!(map_codec(*b"mp4v", Some(oti)), CodecId::Mpeg2, "{oti:#x}");
        }
        for oti in [None, Some(0x20), Some(0x66), Some(0x6B)] {
            assert_eq!(map_codec(*b"mp4v", oti), CodecId::Mpeg4, "{oti:?}");
        }
        assert_eq!(map_codec(*b"mp4v", Some(0x6C)), CodecId::Mjpeg);
    }

    #[test]
    fn quicktime_mpeg_entries_are_mpeg2_pictures() {
        for code in [
            b"mp2v", b"m2v1", b"hdv2", b"hdva", b"mx5p", b"xd5b", b"xdvf", b"xdh2", b"AVmp",
            b"m1v ", b"mpeg", b"mp1v",
        ] {
            assert!(known_video(*code));
            assert_eq!(map_codec(*code, None), CodecId::Mpeg2);
        }
        assert!(!known_video(*b"xd52"));
        assert_eq!(map_codec(*b"xd52", None), CodecId::Unknown(*b"xd52"));
    }

    fn descriptor(tag: u8, body: &[u8]) -> Vec<u8> {
        let mut out = vec![tag, u8::try_from(body.len()).unwrap()];
        out.extend_from_slice(body);
        out
    }

    /// One `mp4v` entry whose `esds` names `oti` and carries `specific`.
    fn mp4v(oti: Option<u8>, specific: &[u8]) -> Vec<u8> {
        let mut visual = vec![0; 78];
        visual[7] = 1;
        visual[24..28].copy_from_slice(&[0, 48, 0, 32]);
        if let Some(oti) = oti {
            let mut config = vec![oti, 0x11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            config.extend(descriptor(0x05, specific));
            let mut es = vec![0, 1, 0];
            es.extend(descriptor(0x04, &config));
            let mut esds = vec![0; 4];
            esds.extend(descriptor(0x03, &es));
            visual.extend(boxed(b"esds", &esds));
        }
        let mut stsd = vec![0, 0, 0, 0, 0, 0, 0, 1];
        stsd.extend(boxed(b"mp4v", &visual));
        stsd
    }

    fn codec(oti: Option<u8>) -> SampleDesc {
        let mut budget = 64;
        let mut entries =
            parse_stsd(&mp4v(oti, &[0, 0, 1, 0xB0, 1]), *b"vide", &mut budget, 0).expect("parse");
        assert_eq!(entries.len(), 1);
        entries.remove(0)
    }

    #[test]
    fn an_mp4v_entry_is_named_by_its_object_type() {
        let visual = codec(Some(0x20));
        assert_eq!(visual.codec, CodecId::Mpeg4);
        assert!(matches!(
            visual.kind,
            MediaKind::Video(VideoParams {
                width: 48,
                height: 32
            })
        ));
        assert_eq!(visual.extradata, [0, 0, 1, 0xB0, 1]);
        assert_eq!(codec(None).codec, CodecId::Mpeg4);
        for mpeg2 in 0x60..=0x65 {
            assert_eq!(codec(Some(mpeg2)).codec, CodecId::Mpeg2, "{mpeg2:#x}");
        }
        assert_eq!(codec(Some(0x6A)).codec, CodecId::Mpeg2, "MPEG-1");
        assert_eq!(codec(Some(0x6C)).codec, CodecId::Mjpeg);
        assert_eq!(codec(Some(0x21)).codec, CodecId::Mpeg4);
    }

    /// One sound entry: `head` is the 28-byte version-0 body with the
    /// version, channels, bits and rate set, then `rest`.
    fn sound(
        code: &[u8; 4],
        version: u16,
        channels: u16,
        bits: u16,
        rate: u16,
        rest: &[u8],
    ) -> SampleDesc {
        let mut entry = vec![0u8; 28];
        entry[7] = 1;
        entry[8..10].copy_from_slice(&version.to_be_bytes());
        entry[16..18].copy_from_slice(&channels.to_be_bytes());
        entry[18..20].copy_from_slice(&bits.to_be_bytes());
        entry[24..26].copy_from_slice(&rate.to_be_bytes());
        entry.extend_from_slice(rest);
        let mut stsd = vec![0, 0, 0, 0, 0, 0, 0, 1];
        stsd.extend(boxed(code, &entry));
        let mut budget = 64;
        parse_stsd(&stsd, *b"soun", &mut budget, 0)
            .expect("parse")
            .remove(0)
    }

    fn audio(desc: &SampleDesc) -> AudioParams {
        match desc.kind {
            MediaKind::Audio(audio) => audio,
            _ => panic!("not audio"),
        }
    }

    #[test]
    fn a_pcm_entry_states_the_bytes_of_one_frame() {
        let one = |bytes| Some(Framing { frames: 1, bytes });
        assert_eq!(sound(b"sowt", 0, 2, 16, 48_000, &[]).framing, one(4));
        assert_eq!(sound(b"twos", 0, 1, 16, 48_000, &[]).framing, one(2));
        assert_eq!(sound(b"twos", 0, 2, 8, 8_000, &[]).framing, one(2));
        assert_eq!(sound(b"raw ", 0, 1, 8, 8_000, &[]).framing, one(1));
        assert_eq!(sound(b"in24", 0, 2, 24, 48_000, &[]).framing, one(6));
        assert_eq!(sound(b"in32", 0, 2, 32, 48_000, &[]).framing, one(8));
        assert_eq!(sound(b"fl32", 0, 6, 32, 48_000, &[]).framing, one(24));
        assert_eq!(sound(b"fl64", 0, 1, 64, 48_000, &[]).framing, one(8));
        // G.711 is a byte a channel whatever the bits field says.
        assert_eq!(sound(b"ulaw", 0, 2, 16, 8_000, &[]).framing, one(2));
        assert_eq!(sound(b"alaw", 0, 1, 16, 8_000, &[]).framing, one(1));
        // Version 1 of uncompressed sound: one sample per packet, and the
        // frame is still channels times the sample width.
        let mut v1 = Vec::new();
        for field in [1u32, 2, 4, 2] {
            v1.extend(field.to_be_bytes());
        }
        assert_eq!(sound(b"sowt", 1, 2, 16, 48_000, &v1).framing, one(4));
        // A compressed codec's version-1 block.
        let mut block = Vec::new();
        for field in [64u32, 34, 68, 2] {
            block.extend(field.to_be_bytes());
        }
        assert_eq!(
            sound(b"ima4", 1, 2, 16, 44_100, &block).framing,
            Some(Framing {
                frames: 64,
                bytes: 68
            })
        );
        // Compressed sound with no block, and a PCM entry with no channels,
        // leave the size to `stsz`.
        assert_eq!(sound(b"mp4a", 0, 2, 16, 48_000, &[]).framing, None);
        assert_eq!(sound(b"sowt", 0, 0, 16, 48_000, &[]).framing, None);
    }

    #[test]
    fn iso_pcm_entries_read_pcmc_and_keep_the_stsz_frame_size() {
        let pcmc = |flags: u8, size: u8| boxed(b"pcmC", &[0, 0, 0, 0, flags, size]);
        let cases = [
            (b"ipcm", 1, 16, SampleFormat::S16Le),
            (b"ipcm", 0, 16, SampleFormat::S16Be),
            (b"ipcm", 1, 24, SampleFormat::S24Le),
            (b"ipcm", 0, 32, SampleFormat::S32Be),
            (b"fpcm", 1, 32, SampleFormat::F32Le),
            (b"fpcm", 0, 64, SampleFormat::F64Be),
        ];
        for (code, flags, size, format) in cases {
            let desc = sound(code, 0, 2, 16, 48_000, &pcmc(flags, size));
            assert_eq!(desc.codec, CodecId::Pcm);
            assert_eq!(audio(&desc).format, format);
            assert_eq!(audio(&desc).bits_per_sample, u16::from(size));
            assert_eq!(desc.framing, None);
        }
        // No `pcmC`, a short one, and a size no PCM has: the layout is
        // unknown and nothing is guessed.
        assert_eq!(
            audio(&sound(b"fpcm", 0, 2, 16, 48_000, &[])).format,
            SampleFormat::Unknown
        );
        let short = boxed(b"pcmC", &[0, 0, 0, 0, 1]);
        assert_eq!(
            audio(&sound(b"ipcm", 0, 2, 20, 48_000, &short)).format,
            SampleFormat::Unknown
        );
        assert_eq!(
            audio(&sound(b"ipcm", 0, 2, 16, 48_000, &pcmc(1, 20))).format,
            SampleFormat::Unknown
        );
    }

    #[test]
    fn a_zero_rate_is_read_from_the_codec_configuration() {
        // AAC-LC, frequency index 0 (96 kHz), stereo.
        assert_eq!(aac_rate(&[0x10, 0x10]), Some(96_000));
        // Index 1 (88.2 kHz) and index 4 (44.1 kHz).
        assert_eq!(aac_rate(&[0x10, 0x90]), Some(88_200));
        assert_eq!(aac_rate(&[0x12, 0x10]), Some(44_100));
        // Index 15: the rate is written out in 24 bits (here 96000).
        assert_eq!(aac_rate(&[0x17, 0x80, 0xBB, 0x80, 0x10]), Some(96_000));
        // AOT 5 over a 48 kHz core with a 96 kHz extension.
        assert_eq!(aac_rate(&[0x29, 0x90, 0x08]), Some(96_000));
        // An escaped object type (32 + 10) still reads its frequency.
        assert_eq!(aac_rate(&[0xF9, 0x46, 0x20]), Some(48_000));
        // Empty, cut short, a reserved index, and a written rate of 0.
        assert_eq!(aac_rate(&[]), None);
        assert_eq!(aac_rate(&[0x10]), None);
        assert_eq!(aac_rate(&[0x16, 0x90]), None);
        assert_eq!(aac_rate(&[0x17, 0x80, 0x00, 0x00, 0x10]), None);

        let mut dfla = vec![0, 0, 0, 0, 0x80, 0, 0, 34];
        let mut info = [0u8; 34];
        info[10..13].copy_from_slice(&[0x17, 0x70, 0x00]);
        dfla.extend(info);
        assert_eq!(flac_rate(&dfla), Some(96_000));
        assert_eq!(flac_rate(&dfla[..20]), None);
        let mut other = dfla.clone();
        other[4] = 0x84;
        assert_eq!(flac_rate(&other), None);

        // The entry's own rate wins when it has one.
        let esds = |config: &[u8]| {
            let mut decoder = vec![0x40, 0x15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            decoder.extend(descriptor(0x05, config));
            let mut es = vec![0, 1, 0];
            es.extend(descriptor(0x04, &decoder));
            let mut body = vec![0; 4];
            body.extend(descriptor(0x03, &es));
            boxed(b"esds", &body)
        };
        let config = esds(&[0x10, 0x10]);
        assert_eq!(
            audio(&sound(b"mp4a", 0, 2, 16, 0, &config)).sample_rate,
            96_000
        );
        assert_eq!(
            audio(&sound(b"mp4a", 0, 2, 16, 44_100, &config)).sample_rate,
            44_100
        );
        assert_eq!(
            audio(&sound(b"fLaC", 0, 2, 16, 0, &boxed(b"dfLa", &dfla))).sample_rate,
            96_000
        );
        // Except FLAC's: STREAMINFO wins over the halved rate FFmpeg writes.
        assert_eq!(
            audio(&sound(b"fLaC", 0, 2, 16, 48_000, &boxed(b"dfLa", &dfla))).sample_rate,
            96_000
        );
        assert_eq!(audio(&sound(b"Opus", 0, 2, 16, 0, &[])).sample_rate, 48_000);
        // Nothing states a rate: it stays 0 for the media timescale.
        assert_eq!(audio(&sound(b"mp4a", 0, 2, 16, 0, &[])).sample_rate, 0);
        assert_eq!(
            audio(&sound(b"mp4a", 0, 2, 16, 0, &esds(&[0x10]))).sample_rate,
            0
        );
    }
}
