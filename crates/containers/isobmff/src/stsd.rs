//! Sample descriptions. Configuration records stay bytes: `avcC`, `hvcC`,
//! `av1C`, and the decoder-specific info inside `esds`. Nothing here decodes them.

use crate::alac;
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
    let extradata = if codec == CodecId::ProRes && configured.extradata.is_empty() {
        fourcc.to_vec()
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
    let mut sample_rate = read_u16(&payload[24..26]);
    let mut format = pcm_format(fourcc, bits);
    let mut child_at = 28usize;
    if version == 1 {
        child_at = 44;
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
        sample_rate = if rounded.is_finite() && rounded > 0.0 && rounded < f64::from(u16::MAX) {
            u16::try_from(rounded as u64).unwrap_or(0)
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
    if !matches!(codec, CodecId::Pcm) {
        format = SampleFormat::Unknown;
    }
    let mut rate = u32::from(sample_rate);
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
    })
}

struct Configured {
    extradata: Vec<u8>,
    /// A `chan` box beside the configuration: the ALAC channel layout.
    chan: Option<Vec<u8>>,
    color: Option<ColorTags>,
    object_type: Option<u8>,
}

fn configuration(data: &[u8], budget: &mut u32, depth: u32) -> Result<Configured, Error> {
    let mut extradata = Vec::new();
    let mut color = None;
    let mut object_type = None;
    let mut chan = None;
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
            }
            b"colr" => color = color.or(parse_colr(atom.payload)),
            b"chan" => chan = Some(atom.payload.to_vec()),
            _ => {}
        }
    }
    Ok(Configured {
        extradata,
        chan,
        color,
        object_type,
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
        b"mp2v" => CodecId::Mpeg2,
        b"jpeg" | b"mjpa" | b"mjpb" => CodecId::Mjpeg,
        b"apch" | b"apcn" | b"apcs" | b"apco" | b"ap4h" | b"ap4x" => CodecId::ProRes,
        b"Opus" => CodecId::Opus,
        b"fLaC" => CodecId::Flac,
        b"alac" => CodecId::Alac,
        b"ac-3" | b"sac3" | b"ec-3" => CodecId::Ac3,
        b"sowt" | b"twos" | b"raw " | b"in24" | b"in32" | b"fl32" | b"fl64" | b"lpcm" | b"NONE" => {
            CodecId::Pcm
        }
        b".mp3" => CodecId::Mp3,
        b"mp4a" => match oti {
            Some(0x69 | 0x6B) => CodecId::Mp3,
            Some(0xA5 | 0xA6) => CodecId::Ac3,
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
        b"sowt" => match bits {
            8 => SampleFormat::U8,
            24 => SampleFormat::S24Le,
            32 => SampleFormat::S32Le,
            _ => SampleFormat::S16Le,
        },
        b"twos" | b"NONE" => match bits {
            8 => SampleFormat::U8,
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
        (false, true, 8) => SampleFormat::U8,
        (false, false, 8) => SampleFormat::U8,
        (false, true, 24) => SampleFormat::S24Be,
        (false, false, 24) => SampleFormat::S24Le,
        (false, true, 32) => SampleFormat::S32Be,
        (false, false, 32) => SampleFormat::S32Le,
        (false, true, _) => SampleFormat::S16Be,
        (false, false, _) => SampleFormat::S16Le,
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

    fn atom(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = u32::try_from(body.len() + 8)
            .unwrap()
            .to_be_bytes()
            .to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
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
            visual.extend(atom(b"esds", &esds));
        }
        let mut stsd = vec![0, 0, 0, 0, 0, 0, 0, 1];
        stsd.extend(atom(b"mp4v", &visual));
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
}
