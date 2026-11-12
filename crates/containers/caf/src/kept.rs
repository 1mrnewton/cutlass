//! Chunks carried through without being read: markers (`mark`), regions
//! (`regn`), the strings they name (`strg`), user data (`uuid`) and MIDI
//! (`midi`).
//!
//! Their contents do not depend on how the sound is coded, only on its
//! frames: marker and region positions are frame numbers. So a remux that
//! copies the packets can write them back as they were. `ovvw` and `peak`
//! describe the decoded sound and are left out, so a muxer fed re-encoded
//! sound does not carry a stale overview.

use crate::limits::MAX_KEPT_BYTES;

/// The chunk types that are kept, in no particular order.
pub const KEPT_KINDS: [[u8; 4]; 5] = [*b"mark", *b"regn", *b"strg", *b"uuid", *b"midi"];

/// One kept chunk: its type and its body as stored.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeptChunk {
    kind: [u8; 4],
    body: Vec<u8>,
}

impl KeptChunk {
    /// A chunk of a [`KEPT_KINDS`] type whose body holds at least the fixed
    /// head `CAFFile.h` gives it: 8 bytes for `mark` and `regn` (time type
    /// and count), 4 for `strg` (count), 16 for `uuid` (the UUID). `None`
    /// for anything else, or for a body over [`MAX_KEPT_BYTES`].
    #[must_use]
    pub fn new(kind: [u8; 4], body: Vec<u8>) -> Option<Self> {
        let head = match &kind {
            b"mark" | b"regn" => 8,
            b"strg" => 4,
            b"uuid" => 16,
            b"midi" => 0,
            _ => return None,
        };
        let len = u64::try_from(body.len()).ok()?;
        (body.len() >= head && len <= MAX_KEPT_BYTES).then_some(Self { kind, body })
    }

    #[must_use]
    pub const fn kind(&self) -> [u8; 4] {
        self.kind
    }

    #[must_use]
    pub fn body(&self) -> &[u8] {
        &self.body
    }

    pub(crate) fn len(&self) -> u64 {
        u64::try_from(self.body.len()).unwrap_or(u64::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::{KEPT_KINDS, KeptChunk};
    use crate::limits::MAX_KEPT_BYTES;

    #[test]
    fn only_kept_kinds_with_their_head_are_chunks() {
        for kind in KEPT_KINDS {
            assert!(KeptChunk::new(kind, vec![0; 16]).is_some());
        }
        assert!(KeptChunk::new(*b"midi", Vec::new()).is_some());
        assert!(KeptChunk::new(*b"mark", vec![0; 7]).is_none());
        assert!(KeptChunk::new(*b"regn", vec![0; 7]).is_none());
        assert!(KeptChunk::new(*b"strg", vec![0; 3]).is_none());
        assert!(KeptChunk::new(*b"uuid", vec![0; 15]).is_none());
        for kind in [
            *b"desc", *b"data", *b"pakt", *b"kuki", *b"chan", *b"info", *b"free",
        ] {
            assert!(KeptChunk::new(kind, vec![0; 16]).is_none());
        }
        for kind in [*b"ovvw", *b"peak"] {
            assert!(KeptChunk::new(kind, vec![0; 16]).is_none());
        }
        let big = usize::try_from(MAX_KEPT_BYTES).expect("fits") + 1;
        assert!(KeptChunk::new(*b"midi", vec![0; big]).is_none());
        let chunk = KeptChunk::new(*b"strg", vec![0, 0, 0, 0]).expect("strg");
        assert_eq!(chunk.kind(), *b"strg");
        assert_eq!(chunk.body(), &[0, 0, 0, 0]);
    }
}
