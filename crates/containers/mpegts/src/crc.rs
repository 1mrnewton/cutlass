//! CRC-32 of a PSI section, ITU-T H.222.0 Annex A.
//!
//! Polynomial `0x04C11DB7`, register preset to all ones, bits taken most
//! significant first, no final inversion. Run over a whole section,
//! `CRC_32` included, the register ends at zero.

const POLY: u32 = 0x04C1_1DB7;

const TABLE: [u32; 256] = table();

const fn table() -> [u32; 256] {
    let mut table = [0_u32; 256];
    let mut index = 0;
    while index < 256 {
        // `index` is below 256, so the shift keeps every bit.
        let mut crc = (index as u32) << 24;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ POLY
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[index] = crc;
        index += 1;
    }
    table
}

/// The register after `bytes`, from the all-ones preset.
#[must_use]
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        let at = usize::from(((crc >> 24) as u8) ^ byte);
        crc = (crc << 8) ^ TABLE[at];
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::{POLY, crc32};

    fn bitwise(bytes: &[u8]) -> u32 {
        let mut crc = u32::MAX;
        for &byte in bytes {
            crc ^= u32::from(byte) << 24;
            for _ in 0..8 {
                crc = if crc & 0x8000_0000 != 0 {
                    (crc << 1) ^ POLY
                } else {
                    crc << 1
                };
            }
        }
        crc
    }

    #[test]
    fn the_check_value_matches_crc32_mpeg2() {
        // The catalogued check value of CRC-32/MPEG-2 for "123456789".
        assert_eq!(crc32(b"123456789"), 0x0376_E6E7);
        assert_eq!(crc32(b""), u32::MAX);
    }

    #[test]
    fn the_table_matches_the_bitwise_register() {
        let bytes: Vec<u8> = (0..=255)
            .chain((0..100_u8).map(|i| i.wrapping_mul(7)))
            .collect();
        assert_eq!(crc32(&bytes), bitwise(&bytes));
    }

    #[test]
    fn a_section_with_its_crc_checks_to_zero() {
        // A PAT for program 1 on PID 0x1000, as FFmpeg writes it.
        let mut section = vec![
            0x00, 0xB0, 0x0D, 0x00, 0x01, 0xC1, 0x00, 0x00, 0x00, 0x01, 0xF0, 0x00,
        ];
        let crc = crc32(&section);
        section.extend_from_slice(&crc.to_be_bytes());
        assert_eq!(crc32(&section), 0);
        assert_eq!(crc, 0x2AB1_04B2);
    }
}
