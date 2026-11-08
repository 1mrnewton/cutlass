//! What a Motion JPEG track says before its first frame.
//!
//! `CodecParams::extra` takes one of three forms:
//!
//! - empty;
//! - a QuickTime sample-entry type, `jpeg`, `mjpa`, or `mjpb`, then
//!   optionally the two payload bytes of its `fiel` atom (field count,
//!   detail);
//! - an abbreviated table-specification stream (ITU-T T.81 B.5), starting
//!   with SOI, whose tables every frame may use.

use cutlass_decode::CodecError;
use cutlass_jpeg::{Tables, read_tables};

/// Which field a two-field packet stores first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FieldOrder {
    TopFirst,
    BottomFirst,
}

/// The sample-entry flavour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Flavour {
    /// `jpeg`: plain JPEG images.
    Jpeg,
    /// `mjpa`: Motion JPEG format A, JPEG images with an APP1 `mjpg` header.
    FormatA,
    /// `mjpb`: Motion JPEG format B, marker-less. Refused.
    FormatB,
}

#[derive(Clone, Debug)]
pub(crate) struct Config {
    pub(crate) flavour: Option<Flavour>,
    /// From `fiel`, when it names two fields.
    pub(crate) field_order: Option<FieldOrder>,
    /// Tables in force before the first packet: the Annex K.3 Huffman
    /// tables, then whatever a table stream in the extradata defined.
    pub(crate) tables: Tables,
}

impl Config {
    pub(crate) fn parse(extra: &[u8]) -> Result<Self, CodecError> {
        let mut config = Self {
            flavour: None,
            field_order: None,
            tables: Tables::standard_huffman(),
        };
        if extra.is_empty() {
            return Ok(config);
        }
        if extra.starts_with(&[0xff, 0xd8]) {
            let used = read_tables(extra, &mut config.tables)?;
            if used != extra.len() {
                return Err(unsupported());
            }
            return Ok(config);
        }
        let (fourcc, fiel) = match extra.len() {
            4 => (&extra[..4], None),
            6 => (&extra[..4], Some([extra[4], extra[5]])),
            _ => return Err(unsupported()),
        };
        config.flavour = Some(match fourcc {
            b"jpeg" => Flavour::Jpeg,
            b"mjpa" => Flavour::FormatA,
            b"mjpb" => Flavour::FormatB,
            _ => return Err(unsupported()),
        });
        config.field_order = fiel.and_then(field_order);
        Ok(config)
    }
}

/// The `fiel` payload's stored order. Apple's detail values: 1 and 9 store
/// the top field first, 6 and 14 the bottom field.
fn field_order([fields, detail]: [u8; 2]) -> Option<FieldOrder> {
    if fields != 2 {
        return None;
    }
    match detail {
        1 | 9 => Some(FieldOrder::TopFirst),
        6 | 14 => Some(FieldOrder::BottomFirst),
        _ => None,
    }
}

fn unsupported() -> CodecError {
    CodecError::Unsupported {
        what: "motion jpeg extradata",
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, FieldOrder, Flavour};
    use cutlass_decode::CodecError;

    #[test]
    fn the_sample_entry_and_fiel_set_flavour_and_order() {
        let empty = Config::parse(&[]).expect("empty");
        assert_eq!(empty.flavour, None);
        assert_eq!(empty.field_order, None);
        assert!(empty.tables.has_huffman(0, 0) && empty.tables.has_huffman(1, 1));
        assert!(!empty.tables.has_quant(0));

        let a = Config::parse(b"mjpa").expect("mjpa");
        assert_eq!(a.flavour, Some(Flavour::FormatA));
        for (detail, order) in [
            (1, Some(FieldOrder::TopFirst)),
            (9, Some(FieldOrder::TopFirst)),
            (6, Some(FieldOrder::BottomFirst)),
            (14, Some(FieldOrder::BottomFirst)),
            (0, None),
            (7, None),
        ] {
            let config = Config::parse(&[b'j', b'p', b'e', b'g', 2, detail]).expect("fiel");
            assert_eq!(config.field_order, order, "detail {detail}");
        }
        let progressive = Config::parse(&[b'j', b'p', b'e', b'g', 1, 0]).expect("one field");
        assert_eq!(progressive.field_order, None);
        assert_eq!(
            Config::parse(b"mjpb").expect("mjpb").flavour,
            Some(Flavour::FormatB)
        );
    }

    #[test]
    fn other_extradata_is_declined() {
        for extra in [&b"avc1"[..], b"jpe", b"jpeg\x02", b"jpeg\x02\x01\x00"] {
            assert!(matches!(
                Config::parse(extra),
                Err(CodecError::Unsupported { .. })
            ));
        }
        assert!(Config::parse(&[0xff, 0xd8, 0xff]).is_err());
    }
}
