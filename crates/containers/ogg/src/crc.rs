//! The page checksum of RFC 3533 section 6.
//!
//! CRC-32 with generator polynomial `0x04C11DB7`, initial value 0, no bit
//! reflection on input or output, and no final XOR. The checksum is taken
//! over the whole page with its own four checksum bytes set to zero.

/// One table row per byte value, built at compile time.
const TABLE: [u32; 256] = build_table();

const fn build_table() -> [u32; 256] {
    let mut table = [0_u32; 256];
    let mut index = 0;
    while index < 256 {
        // `index` is below 256, so the cast is exact.
        let mut value = (index as u32) << 24;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 0x8000_0000 != 0 {
                (value << 1) ^ 0x04C1_1DB7
            } else {
                value << 1
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

/// Continues a checksum over `bytes`.
#[must_use]
pub fn update(mut crc: u32, bytes: &[u8]) -> u32 {
    for &byte in bytes {
        let at = ((crc >> 24) ^ u32::from(byte)) & 0xFF;
        crc = (crc << 8) ^ TABLE[at as usize];
    }
    crc
}

/// The checksum of one page whose checksum field is at bytes 22..26.
///
/// The field is treated as zero whatever it holds, so a page read from a
/// file can be checked without copying it.
#[must_use]
pub fn page_crc(header: &[u8], body: &[u8]) -> u32 {
    let mut crc = 0;
    if header.len() >= 26 {
        crc = update(crc, &header[..22]);
        crc = update(crc, &[0, 0, 0, 0]);
        crc = update(crc, &header[26..]);
    } else {
        crc = update(crc, header);
    }
    update(crc, body)
}

#[cfg(test)]
mod tests {
    use super::{page_crc, update};

    /// Bit at a time, straight from the polynomial, for comparison.
    fn bitwise(bytes: &[u8]) -> u32 {
        let mut crc = 0_u32;
        for &byte in bytes {
            crc ^= u32::from(byte) << 24;
            for _ in 0..8 {
                crc = if crc & 0x8000_0000 != 0 {
                    (crc << 1) ^ 0x04C1_1DB7
                } else {
                    crc << 1
                };
            }
        }
        crc
    }

    #[test]
    fn the_capture_pattern_has_the_documented_checksum() {
        // FFmpeg's oggdec.c names 0x4fa9b05f as the checksum of "OggS"
        // with this polynomial and a zero start. FFmpeg keeps its
        // big-endian CRC tables byte-swapped, so its value is this one with
        // the bytes reversed.
        assert_eq!(update(0, b"OggS"), 0x5FB0_A94F);
        assert_eq!(update(0, b"OggS").swap_bytes(), 0x4FA9_B05F);
    }

    #[test]
    fn the_table_matches_the_bitwise_definition() {
        assert_eq!(update(0, b""), 0);
        assert_eq!(update(0, b"123456789"), bitwise(b"123456789"));
        let ramp: Vec<u8> = (0..=255).collect();
        assert_eq!(update(0, &ramp), bitwise(&ramp));
        assert_eq!(update(0, &[0xFF; 64]), bitwise(&[0xFF; 64]));
    }

    #[test]
    fn the_checksum_field_counts_as_zero() {
        let mut header = [0_u8; 27];
        header[..4].copy_from_slice(b"OggS");
        let clean = page_crc(&header, b"body");
        header[22..26].copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(page_crc(&header, b"body"), clean);
        let mut zeroed = header;
        zeroed[22..26].copy_from_slice(&[0; 4]);
        let mut whole = zeroed.to_vec();
        whole.extend_from_slice(b"body");
        assert_eq!(clean, bitwise(&whole));
    }
}
