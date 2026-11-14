//! The magic cookie (`kuki`) and the extradata each codec crate reads.
//!
//! - AAC: the cookie is an MPEG-4 `ES_Descriptor` (the body of an `esds`
//!   box, ISO/IEC 14496-1). The extradata is the AudioSpecificConfig inside
//!   its `DecoderSpecificInfo`, which is what `cutlass-aac` parses.
//! - ALAC: the cookie is `ALACSpecificConfig` (24 bytes), optionally
//!   followed by a 24-byte `chan` atom, or the older form that wraps it in
//!   `frma` and `alac` atoms. The extradata is the bare config, plus the
//!   `chan` atom when present: the form `cutlass-alac` stores.
//! - FLAC: the cookie is a `dfLa` box whose first metadata block is
//!   STREAMINFO. The extradata is the 34-byte STREAMINFO body.
//! - Opus: AudioToolbox writes a 28-byte cookie whose layout is not
//!   published, and decodes a file without it. The extradata is an
//!   `OpusHead` (RFC 7845 §5.1) built from `desc` and the priming frames.

use cutlass_container::OpenError;

use crate::bytes::{be_u32, fourcc};
use crate::error::malformed;

/// `ALACSpecificConfig` bytes.
pub const ALAC_CONFIG_BYTES: usize = 24;
/// A `chan` atom inside an ALAC cookie.
pub const ALAC_CHAN_ATOM_BYTES: usize = 24;
/// FLAC STREAMINFO body.
pub const STREAMINFO_BYTES: usize = 34;
/// Longest AudioSpecificConfig accepted, the bound `cutlass-aac` uses.
pub const MAX_ASC_BYTES: usize = 64;

const ES_DESCRIPTOR: u8 = 0x03;
const DECODER_CONFIG: u8 = 0x04;
const DECODER_SPECIFIC: u8 = 0x05;
const SL_CONFIG: u8 = 0x06;

/// Reads one descriptor head: tag, then a length of up to four 7-bit groups.
/// Returns the tag, the body range, and checks it fits in `bytes`.
fn descriptor(bytes: &[u8], at: usize) -> Result<(u8, usize, usize), OpenError> {
    let tag = *bytes
        .get(at)
        .ok_or_else(|| malformed("esds descriptor is truncated"))?;
    let mut len = 0_usize;
    let mut pos = at + 1;
    for count in 0..4 {
        let byte = *bytes
            .get(pos)
            .ok_or_else(|| malformed("esds descriptor length is truncated"))?;
        pos += 1;
        len = (len << 7) | usize::from(byte & 0x7F);
        if byte & 0x80 == 0 {
            break;
        }
        if count == 3 {
            return Err(malformed(
                "esds descriptor length is longer than four bytes",
            ));
        }
    }
    let end = pos
        .checked_add(len)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| malformed("esds descriptor runs past its parent"))?;
    Ok((tag, pos, end))
}

/// The AudioSpecificConfig inside an AAC cookie.
///
/// The cookie may start at the `ES_Descriptor` tag (what AudioToolbox
/// writes), after a four-byte version and flags, or with a whole `esds`
/// box header.
pub fn aac_config(cookie: &[u8]) -> Result<Vec<u8>, OpenError> {
    let start = if cookie.get(4..8) == Some(b"esds") {
        12
    } else if cookie.first() == Some(&ES_DESCRIPTOR) {
        0
    } else if cookie.get(..4) == Some(&[0, 0, 0, 0]) && cookie.get(4) == Some(&ES_DESCRIPTOR) {
        4
    } else {
        return Err(malformed("AAC cookie is not an ES_Descriptor"));
    };
    let bytes = cookie
        .get(start..)
        .ok_or_else(|| malformed("AAC cookie is truncated"))?;
    let (tag, body, end) = descriptor(bytes, 0)?;
    if tag != ES_DESCRIPTOR {
        return Err(malformed("AAC cookie is not an ES_Descriptor"));
    }
    let es = &bytes[body..end];
    let flags = *es
        .get(2)
        .ok_or_else(|| malformed("ES_Descriptor is truncated"))?;
    let mut at = 3_usize;
    if flags & 0x80 != 0 {
        at += 2;
    }
    if flags & 0x40 != 0 {
        let url = *es
            .get(at)
            .ok_or_else(|| malformed("ES_Descriptor URL is truncated"))?;
        at += 1 + usize::from(url);
    }
    if flags & 0x20 != 0 {
        at += 2;
    }
    while at < es.len() {
        let (tag, body, end) = descriptor(es, at)?;
        if tag == DECODER_CONFIG {
            return decoder_config(&es[body..end]);
        }
        at = end;
    }
    Err(malformed("ES_Descriptor has no DecoderConfigDescriptor"))
}

fn decoder_config(config: &[u8]) -> Result<Vec<u8>, OpenError> {
    let object = *config
        .first()
        .ok_or_else(|| malformed("DecoderConfigDescriptor is truncated"))?;
    // 0x40 is MPEG-4 audio; 0x66 to 0x68 are the MPEG-2 AAC profiles.
    if !matches!(object, 0x40 | 0x66..=0x68) {
        return Err(malformed(format!(
            "AAC cookie names object type {object:#04x}, not AAC"
        )));
    }
    let mut at = 13_usize;
    while at < config.len() {
        let (tag, body, end) = descriptor(config, at)?;
        if tag == DECODER_SPECIFIC {
            let asc = &config[body..end];
            if asc.is_empty() || asc.len() > MAX_ASC_BYTES {
                return Err(malformed(format!(
                    "AudioSpecificConfig is {} bytes",
                    asc.len()
                )));
            }
            return Ok(asc.to_vec());
        }
        at = end;
    }
    Err(malformed(
        "DecoderConfigDescriptor has no AudioSpecificConfig",
    ))
}

fn put_descriptor(tag: u8, body: &[u8], out: &mut Vec<u8>) {
    // Four-byte lengths, as AudioToolbox writes them. libavformat's CAF
    // reader skips four bytes before the first tag and relies on this form.
    let len = u32::try_from(body.len()).unwrap_or(0) & 0x0FFF_FFFF;
    out.push(tag);
    for shift in [21_u32, 14, 7] {
        out.push(u8::try_from((len >> shift) & 0x7F).unwrap_or(0) | 0x80);
    }
    out.push(u8::try_from(len & 0x7F).unwrap_or(0));
    out.extend_from_slice(body);
}

/// Bitrates and buffer size a `DecoderConfigDescriptor` carries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AacRates {
    /// Largest packet, in bytes. Stored in 24 bits.
    pub buffer_bytes: u32,
    pub max_bitrate: u32,
    pub avg_bitrate: u32,
}

/// An AAC cookie for `asc`, in the form AudioToolbox writes.
pub fn aac_cookie(asc: &[u8], rates: AacRates) -> Vec<u8> {
    let mut specific = Vec::with_capacity(asc.len() + 5);
    put_descriptor(DECODER_SPECIFIC, asc, &mut specific);
    let mut config = Vec::with_capacity(13 + specific.len());
    config.push(0x40);
    config.push(0x15);
    let buffer = rates.buffer_bytes.min(0x00FF_FFFF).to_be_bytes();
    config.extend_from_slice(&buffer[1..]);
    config.extend_from_slice(&rates.max_bitrate.to_be_bytes());
    config.extend_from_slice(&rates.avg_bitrate.to_be_bytes());
    config.extend_from_slice(&specific);
    let mut es = Vec::with_capacity(3 + config.len() + 11);
    es.extend_from_slice(&[0, 0, 0]);
    put_descriptor(DECODER_CONFIG, &config, &mut es);
    put_descriptor(SL_CONFIG, &[0x02], &mut es);
    let mut out = Vec::with_capacity(es.len() + 5);
    put_descriptor(ES_DESCRIPTOR, &es, &mut out);
    out
}

/// The bare ALAC config, plus a `chan` atom when the cookie carries one.
pub fn alac_config(cookie: &[u8]) -> Result<Vec<u8>, OpenError> {
    let config_at = if fourcc(cookie, 4) == Some(*b"frma") {
        // frma (12 bytes), then a full `alac` atom: size, type, version and
        // flags, then the config.
        if fourcc(cookie, 16) != Some(*b"alac") {
            return Err(malformed(
                "ALAC cookie frma is not followed by an alac atom",
            ));
        }
        24
    } else if fourcc(cookie, 4) == Some(*b"alac") {
        12
    } else {
        0
    };
    let config_end = config_at + ALAC_CONFIG_BYTES;
    let config = cookie
        .get(config_at..config_end)
        .ok_or_else(|| malformed("ALAC cookie is shorter than ALACSpecificConfig"))?;
    let mut out = config.to_vec();
    let chan_end = config_end + ALAC_CHAN_ATOM_BYTES;
    if fourcc(cookie, config_end + 4) == Some(*b"chan")
        && be_u32(cookie, config_end) == u32::try_from(ALAC_CHAN_ATOM_BYTES).ok()
        && let Some(chan) = cookie.get(config_end..chan_end)
    {
        out.extend_from_slice(chan);
    }
    Ok(out)
}

/// The STREAMINFO body inside a FLAC cookie: a `dfLa` box, or a native
/// `fLaC` marker and its metadata blocks.
pub fn flac_streaminfo(cookie: &[u8]) -> Result<Vec<u8>, OpenError> {
    let blocks = if fourcc(cookie, 4) == Some(*b"dfLa") {
        if cookie.get(8) != Some(&0) {
            return Err(malformed("dfLa version is not 0"));
        }
        12
    } else if cookie.get(..4) == Some(b"fLaC") {
        4
    } else {
        return Err(malformed("FLAC cookie is neither a dfLa box nor fLaC"));
    };
    let head = be_u32(cookie, blocks).ok_or_else(|| malformed("FLAC cookie has no metadata"))?;
    let kind = (head >> 24) & 0x7F;
    let len = head & 0x00FF_FFFF;
    if kind != 0 || usize::try_from(len).ok() != Some(STREAMINFO_BYTES) {
        return Err(malformed("FLAC cookie does not start with STREAMINFO"));
    }
    let start = blocks + 4;
    cookie
        .get(start..start + STREAMINFO_BYTES)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| malformed("FLAC STREAMINFO is truncated"))
}

/// A `dfLa` box holding one STREAMINFO block, the form AudioToolbox writes.
pub fn flac_cookie(streaminfo: &[u8]) -> Vec<u8> {
    let total = 12 + 4 + streaminfo.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&u32::try_from(total).unwrap_or(0).to_be_bytes());
    out.extend_from_slice(b"dfLa");
    out.extend_from_slice(&[0, 0, 0, 0]);
    let len = u32::try_from(streaminfo.len()).unwrap_or(0) & 0x00FF_FFFF;
    out.extend_from_slice(&(0x8000_0000 | len).to_be_bytes());
    out.extend_from_slice(streaminfo);
    out
}

/// A STREAMINFO body from codec extradata: the 34-byte body itself, or a
/// `fLaC` marker and its first block.
pub fn streaminfo_from_extradata(extra: &[u8]) -> Option<Vec<u8>> {
    if extra.len() == STREAMINFO_BYTES {
        return Some(extra.to_vec());
    }
    flac_streaminfo(extra).ok()
}

/// An `OpusHead` for a mono or stereo stream, mapping family 0.
pub fn opus_head(channels: u8, pre_skip: u16, input_rate: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(19);
    out.extend_from_slice(b"OpusHead");
    out.push(1);
    out.push(channels);
    out.extend_from_slice(&pre_skip.to_le_bytes());
    out.extend_from_slice(&input_rate.to_le_bytes());
    out.extend_from_slice(&0_i16.to_le_bytes());
    out.push(0);
    out
}

/// The pre-skip of an `OpusHead`, when `extra` is one.
pub fn opus_pre_skip(extra: &[u8]) -> Option<u16> {
    if extra.get(..8) != Some(b"OpusHead") {
        return None;
    }
    Some(u16::from_le_bytes([*extra.get(10)?, *extra.get(11)?]))
}

#[cfg(test)]
mod tests {
    use super::{
        AacRates, aac_config, aac_cookie, alac_config, flac_cookie, flac_streaminfo, opus_head,
        opus_pre_skip, streaminfo_from_extradata,
    };

    /// The cookie afconvert wrote for 44.1 kHz stereo AAC-LC.
    const APPLE_AAC: &str =
        "038080802200000004808080144014001800000049600001f40005808080021210068080800102";

    fn hex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&text[at..at + 2], 16).expect("hex"))
            .collect()
    }

    #[test]
    fn the_audio_specific_config_comes_out_of_an_apple_cookie() {
        assert_eq!(aac_config(&hex(APPLE_AAC)).expect("asc"), vec![0x12, 0x10]);
        let mut versioned = vec![0, 0, 0, 0];
        versioned.extend(hex(APPLE_AAC));
        assert_eq!(aac_config(&versioned).expect("asc"), vec![0x12, 0x10]);
        let mut boxed = Vec::new();
        boxed.extend(
            u32::try_from(12 + APPLE_AAC.len() / 2)
                .expect("len")
                .to_be_bytes(),
        );
        boxed.extend(b"esds");
        boxed.extend([0, 0, 0, 0]);
        boxed.extend(hex(APPLE_AAC));
        assert_eq!(aac_config(&boxed).expect("asc"), vec![0x12, 0x10]);
    }

    #[test]
    fn a_written_aac_cookie_reads_back() {
        let rates = AacRates {
            buffer_bytes: 6_144,
            max_bitrate: 18_784,
            avg_bitrate: 128_000,
        };
        let cookie = aac_cookie(&[0x12, 0x10], rates);
        // afconvert writes stream type byte 0x14; ISO/IEC 14496-1 sets the
        // reserved low bit, 0x15, as Logic's own cookies do.
        let mut apple = hex(APPLE_AAC);
        apple[14] = 0x15;
        assert_eq!(cookie, apple);
        assert_eq!(aac_config(&cookie).expect("asc"), vec![0x12, 0x10]);
        // One-byte lengths, as other writers use.
        let short = hex("031900000004114015000000000000000000000005021210060102");
        assert_eq!(
            aac_config(&short).expect("one-byte lengths"),
            vec![0x12, 0x10]
        );
        // The specific info claims 0x12 bytes inside a 17-byte parent.
        let overrun = hex("031900000004114015000000000000000000000005121210060102");
        assert!(aac_config(&overrun).is_err());
    }

    #[test]
    fn hostile_aac_cookies_are_refused() {
        for cookie in [
            vec![],
            vec![0x03],
            vec![0x03, 0xFF, 0xFF, 0xFF, 0xFF, 0x01],
            vec![0x03, 0x80, 0x80, 0x80, 0x7F],
            vec![0x03, 0x03, 0, 0, 0],
            vec![0x03, 0x05, 0, 0, 0x40, 0x04, 0x00],
            // Object type 0x6B is MP3.
            vec![
                0x03, 0x14, 0, 0, 0, 0x04, 0x0F, 0x6B, 0x15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x05,
                0x00,
            ],
            // Empty AudioSpecificConfig.
            vec![
                0x03, 0x14, 0, 0, 0, 0x04, 0x0F, 0x40, 0x15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x05,
                0x00,
            ],
        ] {
            assert!(aac_config(&cookie).is_err(), "{cookie:02x?}");
        }
    }

    #[test]
    fn every_alac_cookie_form_gives_the_bare_config() {
        let config = hex("000010000010280a0e0200ff000006b300022fbd0000ac44");
        assert_eq!(alac_config(&config).expect("bare"), config);
        let mut frma = hex("0000000c66726d61616c616300000024616c616300000000");
        frma.extend(&config);
        assert_eq!(alac_config(&frma).expect("frma"), config);
        let mut chan = config.clone();
        chan.extend(hex("000000186368616e00000000006500020000000000000000"));
        assert_eq!(alac_config(&chan).expect("chan"), chan);
        assert!(alac_config(&config[..23]).is_err());
        assert!(alac_config(&hex("0000000c66726d61616c6163")).is_err());
    }

    #[test]
    fn flac_streaminfo_comes_out_of_dfla_and_goes_back_in() {
        let info: Vec<u8> = (0..34).collect();
        let cookie = flac_cookie(&info);
        assert_eq!(cookie.len(), 50);
        assert_eq!(&cookie[4..8], b"dfLa");
        assert_eq!(flac_streaminfo(&cookie).expect("info"), info);
        let mut native = b"fLaC".to_vec();
        native.extend(&cookie[12..]);
        assert_eq!(flac_streaminfo(&native).expect("native"), info);
        assert_eq!(streaminfo_from_extradata(&info), Some(info.clone()));
        assert_eq!(streaminfo_from_extradata(&native), Some(info.clone()));
        let mut wrong = cookie.clone();
        wrong[12] = 0x84;
        assert!(flac_streaminfo(&wrong).is_err());
        assert!(flac_streaminfo(&cookie[..40]).is_err());
        assert!(flac_streaminfo(b"nope").is_err());
    }

    #[test]
    fn an_opus_head_carries_the_pre_skip() {
        let head = opus_head(2, 312, 48_000);
        assert_eq!(head.len(), 19);
        assert_eq!(&head[..8], b"OpusHead");
        assert_eq!(opus_pre_skip(&head), Some(312));
        assert_eq!(opus_pre_skip(b"OpusTags"), None);
    }
}
