//! The magic cookie AudioConverter wants for an MP4 audio specific config.

use crate::error::Error;
use crate::format::BITRATE;

/// AudioConverter wants an `ES_Descriptor`, not the raw audio specific config.
pub(super) fn cookie_for_specific_config(config: &[u8]) -> Result<Vec<u8>, Error> {
    if config.is_empty() || config.len() > 64 {
        return Err(Error::InvalidConfig("AAC audio specific config"));
    }
    let config_len = u8::try_from(config.len())
        .map_err(|_| Error::InvalidConfig("AAC audio specific config"))?;
    let decoder_len = 18 + config_len;
    let es_len = 14 + decoder_len;
    let mut cookie = Vec::with_capacity(5 + usize::from(es_len));
    cookie.push(0x03);
    cookie.extend_from_slice(&expand(es_len));
    cookie.extend_from_slice(&[0, 0, 0]);
    cookie.push(0x04);
    cookie.extend_from_slice(&expand(decoder_len));
    // MPEG-4 audio, audio stream type, buffer size 6144, then the two rates.
    cookie.extend_from_slice(&[0x40, 0x14, 0x00, 0x18, 0x00]);
    cookie.extend_from_slice(&BITRATE.to_be_bytes());
    cookie.extend_from_slice(&BITRATE.to_be_bytes());
    cookie.push(0x05);
    cookie.extend_from_slice(&expand(config_len));
    cookie.extend_from_slice(config);
    cookie.push(0x06);
    cookie.extend_from_slice(&expand(1));
    cookie.push(0x02);
    Ok(cookie)
}

fn expand(size: u8) -> [u8; 4] {
    [0x80, 0x80, 0x80, size]
}
