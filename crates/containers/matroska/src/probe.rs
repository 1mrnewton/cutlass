//! Recognition of a Matroska or WebM file from its first bytes.
//!
//! Four bytes decide: the EBML magic `1A 45 DF A3` (RFC 8794, 8.1). When the
//! window holds the whole EBML header, its DocType decides as well: another
//! EBML document is refused, and a Matroska version this reader does not
//! implement is accepted as an unsupported brand.

use cutlass_codec::ContainerId;
use cutlass_container::{HeaderVerdict, Probe, ProbeResult};

use crate::ebml::{self, Size};
use crate::error::Error;
use crate::ids;
use crate::limits::PROBE_BYTES;
use crate::meta;

/// The Matroska family, WebM included.
#[derive(Clone, Copy, Debug)]
pub struct MatroskaFamily;

impl Probe for MatroskaFamily {
    const HEADER_BYTES: usize = PROBE_BYTES;

    fn family(&self) -> ContainerId {
        ContainerId::Matroska
    }

    fn matches(&self, header: &[u8]) -> HeaderVerdict {
        if header.get(..PROBE_BYTES) != Some(&[0x1A, 0x45, 0xDF, 0xA3]) {
            return HeaderVerdict::Unrecognized;
        }
        let Some(data) = ebml_header_data(header) else {
            return HeaderVerdict::Match;
        };
        match meta::parse_ebml_header(data) {
            Ok(_) => HeaderVerdict::Match,
            Err(Error::NotMatroska) => HeaderVerdict::Unrecognized,
            Err(Error::UnsupportedVersion { .. }) => HeaderVerdict::UnsupportedBrand,
            // A header that does not parse is left for open to report.
            Err(_) => HeaderVerdict::Match,
        }
    }
}

/// The EBML header's data, when the window holds all of it.
fn ebml_header_data(header: &[u8]) -> Option<&[u8]> {
    let (id, id_len) = ebml::read_id(header).ok()?;
    if id != ids::EBML {
        return None;
    }
    let (Size::Known(size), size_len) = ebml::read_size(header.get(id_len..)?).ok()? else {
        return None;
    };
    let start = id_len + size_len;
    let end = start.checked_add(usize::try_from(size).ok()?)?;
    header.get(start..end)
}

/// Accepts a header that starts with the EBML magic and, when the window
/// is long enough to tell, names the `matroska` or `webm` DocType.
#[must_use]
pub fn probe(header: &[u8]) -> ProbeResult {
    MatroskaFamily.probe(header)
}

#[cfg(test)]
mod tests {
    use super::probe;
    use crate::ebml::{put_bytes, put_master, put_uint};
    use crate::ids;
    use cutlass_codec::ContainerId;
    use cutlass_container::ProbeResult;

    fn header(doc_type: &[u8], read_version: u64) -> Vec<u8> {
        let mut body = Vec::new();
        put_bytes(&mut body, ids::DOC_TYPE, doc_type);
        put_uint(&mut body, ids::DOC_TYPE_READ_VERSION, read_version);
        let mut out = Vec::new();
        put_master(&mut out, ids::EBML, &body);
        out
    }

    const ACCEPT: ProbeResult = ProbeResult::Accept {
        family: ContainerId::Matroska,
        examined: 4,
    };

    #[test]
    fn the_magic_and_the_doctype_decide() {
        assert_eq!(
            probe(&[0x1A, 0x45]),
            ProbeResult::NeedBytes { have: 2, need: 4 }
        );
        assert_eq!(probe(&[0x1A, 0x45, 0xDF, 0xA3]), ACCEPT);
        assert_eq!(probe(&header(b"matroska", 2)), ACCEPT);
        assert_eq!(probe(&header(b"webm", 2)), ACCEPT);
        // A newer read version is still this family.
        assert_eq!(probe(&header(b"matroska", 9)), ACCEPT);
        assert_eq!(
            probe(&header(b"other", 1)),
            ProbeResult::Refuse { examined: 4 }
        );
        assert_eq!(probe(b"RIFF"), ProbeResult::Refuse { examined: 4 });
        assert_eq!(probe(b"OggS\0"), ProbeResult::Refuse { examined: 4 });
    }
}
