//! The AV1 sample a container stores.
//!
//! An AV1 packet is a temporal unit in the low-overhead format: OBUs one
//! after another, usually behind a temporal delimiter. The ISOBMFF binding
//! (AV1 Codec ISO Media File Format Binding, 2.4) and the Matroska mapping
//! store that unit without its temporal delimiters, padding, redundant
//! frame headers and tile lists, and with a size on every OBU. Both
//! families call [`av1_sample`] so a file from either holds the same bytes.

const OBU_TEMPORAL_DELIMITER: u8 = 2;
const OBU_REDUNDANT_FRAME_HEADER: u8 = 7;
const OBU_TILE_LIST: u8 = 8;
const OBU_PADDING: u8 = 15;

/// `unit` as a container sample: its OBUs in order, without the temporal
/// delimiter, padding, redundant frame header and tile list OBUs, each one
/// kept with `obu_has_size_field` set.
///
/// `None` when `unit` is not whole OBUs: the forbidden bit is set, a size
/// runs past the end, or a size does not fit eight `leb128` bytes.
#[must_use]
pub fn av1_sample(unit: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(unit.len());
    let mut rest = unit;
    while let Some(&header) = rest.first() {
        if header & 0x80 != 0 {
            return None;
        }
        let obu_type = (header >> 3) & 15;
        let extension = header & 4 != 0;
        let head = 1 + usize::from(extension);
        let body = rest.get(head..)?;
        // An OBU without a size runs to the end of the unit.
        let (size, size_bytes) = if header & 2 != 0 {
            leb128(body)?
        } else {
            (body.len(), 0)
        };
        let start = head + size_bytes;
        let end = start.checked_add(size)?;
        let payload = rest.get(start..end)?;
        if !matches!(
            obu_type,
            OBU_TEMPORAL_DELIMITER | OBU_REDUNDANT_FRAME_HEADER | OBU_TILE_LIST | OBU_PADDING
        ) {
            out.push(header | 2);
            if extension {
                out.push(rest[1]);
            }
            write_leb128(&mut out, payload.len());
            out.extend_from_slice(payload);
        }
        rest = &rest[end..];
    }
    Some(out)
}

/// An unsigned `leb128` value (AV1 4.10.5) and the bytes it took.
fn leb128(bytes: &[u8]) -> Option<(usize, usize)> {
    let mut value = 0u64;
    for (index, byte) in bytes.iter().enumerate().take(8) {
        value |= u64::from(byte & 0x7f) << (7 * index);
        if byte & 0x80 == 0 {
            // The value is at most 32 bits.
            let value = u32::try_from(value).ok()?;
            return Some((usize::try_from(value).ok()?, index + 1));
        }
    }
    None
}

fn write_leb128(out: &mut Vec<u8>, mut value: usize) {
    loop {
        let low = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(low);
            return;
        }
        out.push(low | 0x80);
    }
}

#[cfg(test)]
mod tests {
    use super::av1_sample;

    /// One OBU of `obu_type` with a size field.
    fn obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![(obu_type << 3) | 2, payload.len() as u8];
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn a_sample_is_the_unit_without_its_delimiter_and_filler() {
        // Delimiter, sequence header, padding, frame header, redundant
        // frame header, tile group, tile list.
        let mut unit = obu(2, &[]);
        unit.extend(obu(1, &[0xaa, 0xbb]));
        unit.extend(obu(15, &[0, 0, 0]));
        unit.extend(obu(3, &[0x10]));
        unit.extend(obu(7, &[0x10]));
        unit.extend(obu(4, &[1, 2, 3, 4]));
        unit.extend(obu(8, &[9]));
        let mut want = obu(1, &[0xaa, 0xbb]);
        want.extend(obu(3, &[0x10]));
        want.extend(obu(4, &[1, 2, 3, 4]));
        assert_eq!(av1_sample(&unit), Some(want.clone()));
        // A sample is already one: nothing changes.
        assert_eq!(av1_sample(&want), Some(want));
        assert_eq!(av1_sample(&[]), Some(Vec::new()));
        assert_eq!(av1_sample(&obu(2, &[])), Some(Vec::new()));
    }

    #[test]
    fn every_obu_kept_gets_a_size_and_keeps_its_extension() {
        // A frame OBU with an extension byte and no size field: it runs to
        // the end of the unit.
        let mut unit = obu(2, &[]);
        unit.extend([(6 << 3) | 4, 0x28, 7, 8, 9]);
        assert_eq!(
            av1_sample(&unit),
            Some(vec![(6 << 3) | 4 | 2, 0x28, 3, 7, 8, 9])
        );
        // A payload of 200 bytes takes a two-byte size.
        let mut long = vec![6 << 3];
        long.extend(vec![5u8; 200]);
        let sample = av1_sample(&long).expect("sample");
        assert_eq!(&sample[..3], [(6 << 3) | 2, 0xc8, 0x01]);
        assert_eq!(sample.len(), 203);
        // And is read back as itself.
        assert_eq!(av1_sample(&sample), Some(sample));
    }

    #[test]
    fn bytes_that_are_not_whole_obus_are_refused() {
        // The forbidden bit, a size past the end, a size that never ends,
        // a size past 32 bits, and an extension with no byte.
        assert_eq!(av1_sample(&[0x80 | (6 << 3) | 2, 0]), None);
        assert_eq!(av1_sample(&[(6 << 3) | 2, 5, 1, 2]), None);
        assert_eq!(av1_sample(&[(6 << 3) | 2, 0x80, 0x80]), None);
        assert_eq!(
            av1_sample(&[(6 << 3) | 2, 0xff, 0xff, 0xff, 0xff, 0x1f, 0]),
            None
        );
        assert_eq!(av1_sample(&[(6 << 3) | 4]), None);
        assert_eq!(av1_sample(&[(6 << 3) | 2]), None);
    }
}
