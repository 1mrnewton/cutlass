//! PNG CRC. Section 5.5 uses the ISO-HDLC polynomial.

const TABLE: [u32; 256] = table();

const fn table() -> [u32; 256] {
    let mut out = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut crc = n as u32;
        let mut bit = 0;
        while bit < 8 {
            if crc & 1 == 1 {
                crc = 0xEDB8_8320 ^ (crc >> 1);
            } else {
                crc >>= 1;
            }
            bit += 1;
        }
        out[n] = crc;
        n += 1;
    }
    out
}

pub(crate) fn of(kind: &[u8; 4], data: &[u8]) -> u32 {
    finish(update(update(0xFFFF_FFFF, kind), data))
}

fn update(mut crc: u32, bytes: &[u8]) -> u32 {
    for &byte in bytes {
        let low =
            u8::try_from((crc ^ u32::from(byte)) & 0xff).expect("a masked byte fits in a byte");
        crc = TABLE[usize::from(low)] ^ (crc >> 8);
    }
    crc
}

fn finish(crc: u32) -> u32 {
    !crc
}

#[cfg(test)]
mod tests {
    use super::finish;
    use super::update;

    #[test]
    fn the_png_checksum_matches_iso_hdlc() {
        let crc = finish(update(0xFFFF_FFFF, b"123456789"));
        assert_eq!(crc, 0xCBF4_3926);
    }
}
