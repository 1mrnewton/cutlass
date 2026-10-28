//! Brands this crate will open. An unknown major brand with no known
//! compatible brand is a typed error.

use crate::legacy::Error;

const KNOWN: &[&[u8; 4]] = &[
    b"isom", b"iso2", b"iso3", b"iso4", b"iso5", b"iso6", b"iso7", b"iso8", b"iso9", b"mp41",
    b"mp42", b"mp71", b"avc1", b"dash", b"cmfc", b"msdh", b"msix", b"piff", b"M4A ", b"M4B ",
    b"M4P ", b"M4V ", b"qt  ", b"3gp4", b"3gp5", b"3gp6", b"3gp7", b"3gp8", b"3gp9", b"3g2a",
    b"3g2b", b"3g2c", b"3ge6", b"3ge7", b"3gg6", b"kddi", b"mif1", b"msf1", b"miaf", b"heic",
    b"heix", b"hevc", b"hevx", b"avif", b"avis",
];

#[must_use]
pub fn known(brand: &[u8; 4]) -> bool {
    KNOWN.contains(&brand)
}

pub struct FileType {
    pub major: [u8; 4],
    pub compatible: Vec<[u8; 4]>,
}

pub fn parse_ftyp(payload: &[u8]) -> Result<FileType, Error> {
    if payload.len() < 8 {
        return Err(Error::Truncated { at: 0, needed: 8 });
    }
    let major = [payload[0], payload[1], payload[2], payload[3]];
    let mut compatible = Vec::new();
    let mut index = 8usize;
    while index + 4 <= payload.len() {
        if compatible.len() >= crate::limits::MAX_BRANDS {
            return Err(Error::Limit {
                what: "brands",
                value: u64::try_from(compatible.len()).unwrap_or(u64::MAX),
            });
        }
        compatible.push([
            payload[index],
            payload[index + 1],
            payload[index + 2],
            payload[index + 3],
        ]);
        index += 4;
    }
    if !known(&major) && !compatible.iter().any(known) {
        return Err(Error::UnsupportedBrand { major });
    }
    Ok(FileType { major, compatible })
}

/// QuickTime files written before `ftyp` start with a movie or media box.
#[must_use]
pub fn headerless_quicktime(kind: &[u8; 4]) -> bool {
    matches!(
        kind,
        b"moov" | b"mdat" | b"wide" | b"free" | b"skip" | b"pnot"
    )
}

#[must_use]
pub fn probe(header: &[u8]) -> bool {
    if header.len() < 8 {
        return false;
    }
    let kind = [header[4], header[5], header[6], header[7]];
    if kind == *b"ftyp" {
        let major = if header.len() >= 16 {
            [header[8], header[9], header[10], header[11]]
        } else {
            return false;
        };
        if known(&major) {
            return true;
        }
        let mut index = 16usize;
        while index + 4 <= header.len() {
            let brand = [
                header[index],
                header[index + 1],
                header[index + 2],
                header[index + 3],
            ];
            if known(&brand) {
                return true;
            }
            index += 4;
        }
        // The compatible list may continue past the probe buffer.
        let size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        return u64::from(size) > u64::try_from(header.len()).unwrap_or(u64::MAX);
    }
    headerless_quicktime(&kind)
}
