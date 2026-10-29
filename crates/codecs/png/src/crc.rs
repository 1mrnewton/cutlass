//! CRC-32 over a chunk type and its data.
//!
//! The polynomial is the reflected ISO-3309 value. The register starts at
//! all ones and is inverted at the end, which is the PNG chunk CRC.

const TABLE: [u32; 256] = crc_table();

const fn crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut crc = n;
        let mut bit = 0;
        while bit < 8 {
            if crc & 1 == 1 {
                crc = 0xedb8_8320 ^ (crc >> 1);
            } else {
                crc >>= 1;
            }
            bit += 1;
        }
        table[n as usize] = crc;
        n += 1;
    }
    table
}

pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in bytes {
        let index = ((crc ^ u32::from(byte)) & 0xff) as usize;
        crc = TABLE[index] ^ (crc >> 8);
    }
    !crc
}

pub(crate) fn crc32_update(crc: u32, bytes: &[u8]) -> u32 {
    let mut crc = !crc;
    for &byte in bytes {
        let index = ((crc ^ u32::from(byte)) & 0xff) as usize;
        crc = TABLE[index] ^ (crc >> 8);
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::crc32;

    #[test]
    fn the_check_value_for_123456789_is_the_iso_value() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }
}
