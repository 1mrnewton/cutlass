//! AAC-LC config record. The encoder's magic cookie is an `ES_Descriptor`.
//! An MP4 `esds` box stores only the audio specific config inside it.

use crate::error::Error;

/// The `DecoderSpecificInfo` payload inside AudioConverter's magic cookie.
pub(super) fn audio_specific_config(cookie: &[u8]) -> Result<Vec<u8>, Error> {
    let (tag, payload, rest) = descriptor(cookie)?;
    if tag != 0x03 || !rest.is_empty() || payload.len() < 3 {
        return Err(Error::InvalidConfig("AAC magic cookie"));
    }
    let mut body = &payload[3..];
    while !body.is_empty() {
        let (tag, payload, next) = descriptor(body)?;
        if tag == 0x04
            && let Some(config) = specific_config(payload)?
        {
            return Ok(config);
        }
        body = next;
    }
    Err(Error::InvalidConfig(
        "AAC magic cookie has no audio specific config",
    ))
}

fn specific_config(decoder_config: &[u8]) -> Result<Option<Vec<u8>>, Error> {
    // object type, stream type, buffer size, and the two bitrates.
    if decoder_config.len() < 13 {
        return Err(Error::InvalidConfig("AAC magic cookie"));
    }
    let mut body = &decoder_config[13..];
    while !body.is_empty() {
        let (tag, payload, next) = descriptor(body)?;
        if tag == 0x05 {
            if payload.is_empty() || payload.len() > 64 {
                return Err(Error::InvalidConfig("AAC audio specific config"));
            }
            return Ok(Some(payload.to_vec()));
        }
        body = next;
    }
    Ok(None)
}

fn descriptor(bytes: &[u8]) -> Result<(u8, &[u8], &[u8]), Error> {
    let (&tag, rest) = bytes
        .split_first()
        .ok_or(Error::InvalidConfig("AAC magic cookie"))?;
    let (size, rest) = expandable(rest)?;
    if rest.len() < size {
        return Err(Error::InvalidConfig("AAC magic cookie"));
    }
    let (payload, rest) = rest.split_at(size);
    Ok((tag, payload, rest))
}

fn expandable(bytes: &[u8]) -> Result<(usize, &[u8]), Error> {
    let mut size = 0usize;
    for index in 0..4 {
        let byte = *bytes
            .get(index)
            .ok_or(Error::InvalidConfig("AAC magic cookie"))?;
        size = size
            .checked_shl(7)
            .ok_or(Error::InvalidConfig("AAC magic cookie"))?
            | usize::from(byte & 0x7f);
        if byte & 0x80 == 0 {
            return Ok((size, &bytes[index + 1..]));
        }
    }
    Err(Error::InvalidConfig("AAC magic cookie"))
}
