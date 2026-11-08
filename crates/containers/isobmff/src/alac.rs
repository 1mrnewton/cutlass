//! The ALAC magic cookie as an MP4 stores it.
//!
//! Apple's `ALACMagicCookieDescription.txt` defines the cookie as the
//! 24-byte `ALACSpecificConfig`, then an optional 24-byte `chan` atom. An MP4
//! `alac` sample entry holds an `alac` box whose payload is a four-byte
//! version and flags, then that config. Apple's writer puts the channel
//! layout in a sibling `chan` box with the same bytes as the cookie's atom.
//!
//! Extradata for an ALAC track is the bare cookie, the form a CAF `kuki`
//! chunk holds: demux joins the two boxes into it, and mux splits it again.
//! This module reads only the few fields the sample entry repeats. The
//! codec crate decodes the rest.

/// Bytes of `ALACSpecificConfig`.
const CONFIG: usize = 24;
/// Bytes of the `chan` box payload the cookie carries: version and flags,
/// tag, bitmap, and a description count of zero.
const CHAN_PAYLOAD: usize = 16;

/// Fields of the config that the sample entry also states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Facts {
    pub(crate) channels: u16,
    pub(crate) bits: u16,
    pub(crate) rate: u32,
}

/// The config inside `bytes`: a bare cookie, or the `alac` box payload with
/// its version first. A bare config cannot start with four zero bytes:
/// that is a frame length of zero.
fn config(bytes: &[u8]) -> Option<(&[u8], &[u8])> {
    let body = if bytes.len() >= CONFIG + 4 && bytes[..4] == [0, 0, 0, 0] {
        &bytes[4..]
    } else {
        bytes
    };
    if body.len() < CONFIG {
        return None;
    }
    Some(body.split_at(CONFIG))
}

/// Channel count, bit depth, and rate from a cookie in either form.
pub(crate) fn facts(cookie: &[u8]) -> Option<Facts> {
    let (config, _) = config(cookie)?;
    Some(Facts {
        channels: u16::from(config[9]),
        bits: u16::from(config[5]),
        rate: u32::from_be_bytes([config[20], config[21], config[22], config[23]]),
    })
}

/// The cookie from an `alac` box payload and, when the entry had one, the
/// sibling `chan` box payload. The layout is joined only when its tag names
/// the config's channel count, as the cookie's atom must.
pub(crate) fn join(alac: &[u8], chan: Option<&[u8]>) -> Option<Vec<u8>> {
    let (config, rest) = config(alac)?;
    let mut cookie = config.to_vec();
    if rest.len() >= 8 && rest.get(4..8) == Some(b"chan".as_slice()) {
        cookie.extend_from_slice(rest);
        return Some(cookie);
    }
    if let Some(chan) = chan
        && chan.len() >= CHAN_PAYLOAD
    {
        let tag = u32::from_be_bytes([chan[4], chan[5], chan[6], chan[7]]);
        if tag & 0xffff == u32::from(config[9]) {
            let size = u32::try_from(8 + CHAN_PAYLOAD).unwrap_or(24);
            cookie.extend_from_slice(&size.to_be_bytes());
            cookie.extend_from_slice(b"chan");
            cookie.extend_from_slice(&chan[..CHAN_PAYLOAD]);
        }
    }
    Some(cookie)
}

/// The `alac` box payload and, when the cookie has a layout, the `chan`
/// box payload.
pub(crate) fn split(cookie: &[u8]) -> Option<(Vec<u8>, Option<Vec<u8>>)> {
    let (config, rest) = config(cookie)?;
    let mut alac = vec![0, 0, 0, 0];
    alac.extend_from_slice(config);
    let chan = (rest.len() >= 8 + CHAN_PAYLOAD && rest.get(4..8) == Some(b"chan".as_slice()))
        .then(|| rest[8..8 + CHAN_PAYLOAD].to_vec());
    Some((alac, chan))
}

#[cfg(test)]
mod tests {
    use super::{Facts, facts, join, split};

    const CONFIG: [u8; 24] = [
        0x00, 0x00, 0x10, 0x00, 0x00, 0x18, 0x28, 0x0a, 0x0e, 0x06, 0x00, 0xff, 0x00, 0x00, 0x47,
        0x0d, 0x00, 0x10, 0xc1, 0x2d, 0x00, 0x01, 0x77, 0x00,
    ];
    const CHAN: [u8; 16] = [0, 0, 0, 0, 0x00, 0x7c, 0x00, 0x06, 0, 0, 0, 0, 0, 0, 0, 0];

    #[test]
    fn an_apple_entry_joins_into_the_caf_cookie_and_splits_back() {
        let mut alac = vec![0, 0, 0, 0];
        alac.extend_from_slice(&CONFIG);
        let cookie = join(&alac, Some(&CHAN)).unwrap();
        assert_eq!(cookie.len(), 48);
        assert_eq!(&cookie[..24], &CONFIG);
        assert_eq!(&cookie[24..32], &[0, 0, 0, 24, b'c', b'h', b'a', b'n']);
        assert_eq!(
            facts(&cookie),
            Some(Facts {
                channels: 6,
                bits: 24,
                rate: 96_000
            })
        );
        let (back, chan) = split(&cookie).unwrap();
        assert_eq!(back, alac);
        assert_eq!(chan.as_deref(), Some(CHAN.as_slice()));
    }

    #[test]
    fn a_layout_for_another_count_is_not_joined() {
        let mut chan = CHAN;
        chan[7] = 2;
        let cookie = join(&CONFIG, Some(&chan)).unwrap();
        assert_eq!(cookie, CONFIG);
        assert_eq!(split(&cookie).unwrap().1, None);
    }

    #[test]
    fn a_short_cookie_is_not_a_cookie() {
        assert_eq!(join(&CONFIG[..23], None), None);
        assert_eq!(split(&[0; 23]), None);
        assert_eq!(facts(&[]), None);
    }
}
