//! Exponent strategies and grouped differential exponents (A/52 section
//! 7.1.3).

use cutlass_decode::CodecError;

use crate::bits::BitReader;
use crate::tables::{D15, D25, D45};

/// Bins one differential exponent covers for a strategy.
pub(crate) fn group_size(strategy: u8) -> usize {
    match strategy {
        D25 => 2,
        D45 => 4,
        _ => 1,
    }
}

/// `nchgrps` for a full bandwidth channel ending at `endmant`.
pub(crate) fn fbw_groups(strategy: u8, endmant: usize) -> usize {
    match strategy {
        D15 => (endmant.saturating_sub(1)) / 3,
        D25 => (endmant.saturating_sub(1) + 3) / 6,
        _ => (endmant.saturating_sub(1) + 9) / 12,
    }
}

/// `ncplgrps` for the coupling channel.
pub(crate) fn cpl_groups(strategy: u8, start: usize, end: usize) -> usize {
    end.saturating_sub(start) / (3 * group_size(strategy))
}

/// Read `groups` 7-bit groups and expand them into `out` from bin `first`,
/// starting from the absolute exponent `reference`. The reference itself is
/// not written: callers that send it as bin 0 store it first.
pub(crate) fn read_groups(
    r: &mut BitReader<'_>,
    strategy: u8,
    reference: u8,
    groups: usize,
    first: usize,
    out: &mut [u8; 256],
) -> Result<(), CodecError> {
    let size = group_size(strategy);
    let mut previous = i32::from(reference);
    let mut bin = first;
    for _ in 0..groups {
        let at = r.pos() as u64;
        let code = r.read(7, "exponent group")?;
        if code > 124 {
            return Err(CodecError::Malformed {
                at_bit: at,
                what: "exponent group",
            });
        }
        for m in [code / 25, (code % 25) / 5, code % 5] {
            previous += m as i32 - 2;
            if !(0..=24).contains(&previous) {
                return Err(CodecError::Malformed {
                    at_bit: at,
                    what: "exponent out of range",
                });
            }
            for _ in 0..size {
                if let Some(slot) = out.get_mut(bin) {
                    *slot = previous as u8;
                }
                bin += 1;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    #[test]
    fn group_counts_cover_the_channel() {
        for endmant in [37, 49, 121, 229, 253] {
            for strategy in [D15, D25, D45] {
                let covered = 1 + fbw_groups(strategy, endmant) * 3 * group_size(strategy);
                assert!(covered >= endmant, "{strategy} {endmant}");
                assert!(covered < endmant + 3 * group_size(strategy));
            }
        }
        assert_eq!(cpl_groups(D15, 37, 253), 72);
        assert_eq!(cpl_groups(D45, 37, 253), 18);
    }

    #[test]
    fn groups_expand_into_repeated_exponents() {
        let mut w = BitWriter::default();
        w.put(7, 2 * 25 + 3 * 5 + 1);
        let bytes = w.finish();
        let mut out = [0u8; 256];
        let mut r = BitReader::new(&bytes);
        read_groups(&mut r, D25, 10, 1, 1, &mut out).unwrap();
        assert_eq!(&out[..7], &[0, 10, 10, 11, 11, 10, 10]);
    }

    #[test]
    fn exponents_that_leave_the_range_are_malformed() {
        let mut w = BitWriter::default();
        w.put(7, 0);
        let bytes = w.finish();
        let mut out = [0u8; 256];
        let mut r = BitReader::new(&bytes);
        assert!(matches!(
            read_groups(&mut r, D15, 1, 1, 1, &mut out),
            Err(CodecError::Malformed { .. })
        ));
        let mut w = BitWriter::default();
        w.put(7, 125);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert!(read_groups(&mut r, D15, 12, 1, 1, &mut out).is_err());
    }
}
