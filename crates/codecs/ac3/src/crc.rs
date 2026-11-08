//! The syncframe CRC (A/52 clause 7.10.1).
//!
//! Generator `x^16 + x^15 + x^2 + 1`, register cleared to zero, bits
//! shifted in most significant first. A word checks when the register is
//! zero after the bytes it covers. `crc1` is not the CRC of the first 5/8:
//! it is the 16-bit value that brings the register to zero at the 5/8
//! point, so the encoder solves for it.

const POLY: u16 = 0x8005;

const TABLE: [u16; 256] = table();

const fn table() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut index = 0;
    while index < 256 {
        let mut crc = (index as u16) << 8;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 0x8000 != 0 {
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

/// The register after `bytes`, starting from `crc`.
#[must_use]
pub(crate) fn update(crc: u16, bytes: &[u8]) -> u16 {
    bytes.iter().fold(crc, |crc, byte| {
        (crc << 8) ^ TABLE[usize::from((crc >> 8) as u8 ^ byte)]
    })
}

/// `a * b mod POLY`, as polynomials over GF(2).
fn mul_mod(a: u16, b: u16) -> u16 {
    let mut result = 0u16;
    let mut a = a;
    let mut b = b;
    while b != 0 {
        if b & 1 != 0 {
            result ^= a;
        }
        b >>= 1;
        a = if a & 0x8000 != 0 {
            (a << 1) ^ POLY
        } else {
            a << 1
        };
    }
    result
}

/// `x^power mod POLY`.
fn pow_x(power: u64) -> u16 {
    let mut result = 1u16;
    let mut base = 2u16;
    let mut power = power;
    while power > 0 {
        if power & 1 != 0 {
            result = mul_mod(result, base);
        }
        base = mul_mod(base, base);
        power >>= 1;
    }
    result
}

/// The `crc1` word for a syncframe whose first 5/8, after the sync word,
/// is `crc1` followed by `rest`.
///
/// The register after a message `m` is `m * x^16 mod g`, so after
/// `w || rest` it is `w * x^(8 len + 16) + crc(0 || rest)`. The generator's
/// constant term is 1, so `x` is a unit modulo it: with `k` the order of
/// `x`, the inverse of `x^n` is `x^(k - n mod k)`.
#[must_use]
pub(crate) fn crc1_for(rest: &[u8]) -> u16 {
    let remainder = update(update(0, &[0, 0]), rest);
    let shift = 8 * rest.len() as u64 + 16;
    let period = period();
    let inverse = pow_x((period - shift % period) % period);
    mul_mod(remainder, inverse)
}

/// The multiplicative order of `x` modulo the generator.
fn period() -> u64 {
    let mut value = 2u16;
    let mut order = 1u64;
    while value != 1 && order < 1 << 17 {
        value = mul_mod(value, 2);
        order += 1;
    }
    order
}

#[cfg(test)]
mod tests {
    use super::{crc1_for, update};

    #[test]
    fn the_register_reads_zero_after_its_own_crc_word() {
        let message = b"A/52 syncframe body";
        let crc = update(0, message);
        let mut sealed = message.to_vec();
        sealed.extend_from_slice(&crc.to_be_bytes());
        assert_eq!(update(0, &sealed), 0);
        assert_ne!(crc, 0);
    }

    #[test]
    fn a_solved_crc1_zeroes_the_register_at_the_five_eighths_point() {
        for len in [0usize, 1, 37, 158, 1278] {
            let rest: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
            let word = crc1_for(&rest);
            let mut covered = word.to_be_bytes().to_vec();
            covered.extend_from_slice(&rest);
            assert_eq!(update(0, &covered), 0, "length {len}");
        }
    }

    #[test]
    fn one_flipped_bit_breaks_the_check() {
        let rest = vec![0x5A; 64];
        let word = crc1_for(&rest);
        let mut covered = word.to_be_bytes().to_vec();
        covered.extend_from_slice(&rest);
        covered[20] ^= 0x04;
        assert_ne!(update(0, &covered), 0);
    }
}
