//! The `info` chunk: a count, then that many pairs of null-terminated UTF-8
//! strings (`CAFInfoStrings`).
//!
//! The chunk is parsed only up to [`MAX_INFO_BYTES`](crate::limits::MAX_INFO_BYTES);
//! a larger one is skipped. Text that is not UTF-8 is kept with replacement
//! characters. A count that claims more pairs than the bytes hold stops at
//! the last whole pair.

/// One key and its value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InfoEntry {
    pub key: String,
    pub value: String,
}

pub(crate) fn parse(body: &[u8]) -> Vec<InfoEntry> {
    let Some(head) = body.get(..4) else {
        return Vec::new();
    };
    let declared = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
    let mut rest = &body[4..];
    let mut entries = Vec::new();
    for _ in 0..declared {
        let Some((key, after_key)) = split_string(rest) else {
            break;
        };
        let Some((value, after_value)) = split_string(after_key) else {
            break;
        };
        entries.push(InfoEntry {
            key: String::from_utf8_lossy(key).into_owned(),
            value: String::from_utf8_lossy(value).into_owned(),
        });
        rest = after_value;
    }
    entries
}

fn split_string(bytes: &[u8]) -> Option<(&[u8], &[u8])> {
    let end = bytes.iter().position(|byte| *byte == 0)?;
    Some((&bytes[..end], &bytes[end + 1..]))
}

/// The chunk body for `entries`. A key or value with a null byte is cut there.
pub(crate) fn to_bytes(entries: &[InfoEntry]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(
        &u32::try_from(entries.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for entry in entries {
        for text in [&entry.key, &entry.value] {
            let bytes = text.as_bytes();
            let end = bytes
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(bytes.len());
            out.extend_from_slice(&bytes[..end]);
            out.push(0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{InfoEntry, parse, to_bytes};

    #[test]
    fn entries_round_trip_and_a_lying_count_stops_at_the_last_pair() {
        let entries = vec![
            InfoEntry {
                key: "title".to_owned(),
                value: "Song".to_owned(),
            },
            InfoEntry {
                key: "artist".to_owned(),
                value: "Ä".to_owned(),
            },
        ];
        let bytes = to_bytes(&entries);
        assert_eq!(parse(&bytes), entries);
        let mut lying = bytes.clone();
        lying[0..4].copy_from_slice(&u32::MAX.to_be_bytes());
        assert_eq!(parse(&lying), entries);
        assert_eq!(parse(&bytes[..bytes.len() - 1]), entries[..1]);
        assert!(parse(&[0, 0]).is_empty());
        let invalid = [0, 0, 0, 1, b'k', 0, 0xFF, 0];
        assert_eq!(parse(&invalid)[0].value, "\u{FFFD}");
    }
}
