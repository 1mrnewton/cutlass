//! The two frame checksums.
//!
//! RFC 9639, section 9.1.8: CRC-8, polynomial x^8 + x^2 + x + 1, initial
//! value 0, over the frame header including the sync code. Section 9.3:
//! CRC-16, polynomial x^16 + x^15 + x^2 + 1, initial value 0, over the
//! whole frame before the footer. Neither is reflected.

const fn crc8_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u8;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

const fn crc16_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = (i as u16) << 8;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static CRC8: [u8; 256] = crc8_table();
static CRC16: [u16; 256] = crc16_table();

#[must_use]
pub fn crc8(data: &[u8]) -> u8 {
    data.iter()
        .fold(0u8, |crc, byte| CRC8[usize::from(crc ^ byte)])
}

#[must_use]
pub fn crc16(data: &[u8]) -> u16 {
    crc16_update(0, data)
}

/// Continues a CRC-16 over more bytes.
#[must_use]
pub fn crc16_update(crc: u16, data: &[u8]) -> u16 {
    data.iter().fold(crc, |crc, byte| {
        (crc << 8) ^ CRC16[usize::from((crc >> 8) as u8 ^ byte)]
    })
}

#[cfg(test)]
mod tests {
    use super::{crc8, crc16, crc16_update};

    #[test]
    fn check_values_match_the_named_polynomials() {
        // CRC-8 (poly 0x07, init 0) and CRC-16/BUYPASS (poly 0x8005,
        // init 0) check values over the ASCII digits 1 through 9.
        assert_eq!(crc8(b"123456789"), 0xF4);
        assert_eq!(crc16(b"123456789"), 0xFEE8);
    }

    #[test]
    fn a_crc_over_its_own_footer_is_zero() {
        let body = b"frame bytes";
        let crc = crc16(body);
        let mut whole = body.to_vec();
        whole.extend_from_slice(&crc.to_be_bytes());
        assert_eq!(crc16(&whole), 0);
        assert_eq!(crc16_update(crc16(&body[..4]), &body[4..]), crc);
    }
}
