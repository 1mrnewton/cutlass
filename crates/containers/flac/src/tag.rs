//! ID3v2 tags in front of the `fLaC` marker.
//!
//! RFC 9639 starts a file with `fLaC`, but taggers put an ID3v2 tag in
//! front of it, and libFLAC and FFmpeg both read such files. The tag is
//! skipped by its declared size and nothing in it is read.
//!
//! An ID3v2 header (id3.org ID3v2.4.0 structure, section 3.1) is `ID3`, a
//! major version and a revision that are not 0xFF, a flags byte, and a size
//! in four bytes whose top bits are clear. The size counts the bytes after
//! the header, and a footer (flag 0x10) adds ten more.

use std::io::{Read, Seek, SeekFrom};

/// Bytes in an ID3v2 header, and in its footer.
pub(crate) const ID3V2_HEADER: usize = 10;
/// ID3v2 tags skipped in front of the marker before a file is refused.
pub const MAX_ID3V2_TAGS: u32 = 16;

/// The bytes a tag takes, header and footer included, when `head` starts
/// with a valid ID3v2 header.
pub(crate) fn id3v2_len(head: &[u8]) -> Option<u64> {
    let head = head.get(..ID3V2_HEADER)?;
    if &head[..3] != b"ID3"
        || head[3] == 0xFF
        || head[4] == 0xFF
        || head[6..].iter().any(|byte| byte & 0x80 != 0)
    {
        return None;
    }
    let size = head[6..]
        .iter()
        .fold(0u64, |size, byte| size << 7 | u64::from(*byte));
    let footer = if head[5] & 0x10 != 0 {
        ID3V2_HEADER as u64
    } else {
        0
    };
    Some(ID3V2_HEADER as u64 + size + footer)
}

/// Where `fLaC` starts in `header`, after any whole ID3v2 tags.
pub(crate) fn marker_in(header: &[u8]) -> Option<usize> {
    let mut at = 0usize;
    for _ in 0..=MAX_ID3V2_TAGS {
        let rest = header.get(at..)?;
        if rest.starts_with(b"fLaC") {
            return Some(at);
        }
        at = at.checked_add(usize::try_from(id3v2_len(rest)?).ok()?)?;
    }
    None
}

/// `true` when the `length`-byte file in `reader` is `fLaC` after any
/// ID3v2 tags in front of it.
///
/// [`probe`](crate::probe) sees only the header it is given, so a tag longer
/// than that, as cover art makes, hides the marker. A reader that takes such
/// a file for MP3 asks this before opening it. It reads ten bytes for each
/// tag and none of a tag's body.
pub fn starts_with_flac<R: Read + Seek>(mut reader: R, length: u64) -> bool {
    let mut at = 0u64;
    for _ in 0..=MAX_ID3V2_TAGS {
        let mut head = Vec::with_capacity(ID3V2_HEADER);
        if reader.seek(SeekFrom::Start(at)).is_err()
            || (&mut reader)
                .take(ID3V2_HEADER as u64)
                .read_to_end(&mut head)
                .is_err()
        {
            return false;
        }
        if head.starts_with(b"fLaC") {
            return true;
        }
        match id3v2_len(&head).and_then(|len| at.checked_add(len)) {
            Some(next) if next < length => at = next,
            _ => return false,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{MAX_ID3V2_TAGS, id3v2_len, marker_in, starts_with_flac};

    fn tag(body: usize, flags: u8) -> Vec<u8> {
        let size = u32::try_from(body).expect("size");
        let synchsafe: Vec<u8> = (0..4)
            .rev()
            .map(|shift| u8::try_from((size >> (7 * shift)) & 0x7F).expect("seven bits"))
            .collect();
        let mut out = b"ID3\x04\x00".to_vec();
        out.push(flags);
        out.extend(&synchsafe);
        out.resize(out.len() + body, 0);
        if flags & 0x10 != 0 {
            out.extend(b"3DI\x04\x00\x10");
            out.extend(&synchsafe);
        }
        out
    }

    #[test]
    fn a_tag_is_measured_with_its_footer() {
        assert_eq!(id3v2_len(&tag(0, 0)), Some(10));
        assert_eq!(id3v2_len(&tag(300, 0)), Some(310));
        assert_eq!(id3v2_len(&tag(300, 0x10)), Some(320));
        assert_eq!(
            id3v2_len(b"ID3\x04\x00\x00\x7F\x7F\x7F\x7F"),
            Some(10 + (1 << 28) - 1)
        );
        assert_eq!(id3v2_len(b"ID3\x04\x00\x00\x00\x00\x00"), None, "short");
        assert_eq!(
            id3v2_len(b"ID3\x04\x00\x00\x00\x00\x00\x80"),
            None,
            "top bit"
        );
        assert_eq!(
            id3v2_len(b"ID3\xFF\x00\x00\x00\x00\x00\x00"),
            None,
            "version"
        );
        assert_eq!(
            id3v2_len(b"ID3\x04\xFF\x00\x00\x00\x00\x00"),
            None,
            "revision"
        );
        assert_eq!(id3v2_len(b"fLaC\x00\x00\x00\x22\x00\x00"), None);
    }

    #[test]
    fn the_marker_is_found_behind_tags() {
        let mut file = tag(20, 0);
        file.extend(tag(5, 0x10));
        file.extend(b"fLaC");
        assert_eq!(marker_in(&file), Some(30 + 25));
        assert!(starts_with_flac(Cursor::new(&file), file.len() as u64));
        assert_eq!(marker_in(b"fLaC"), Some(0));
        // The marker past the header, a tag that is not followed by it, and
        // a tag that runs past the end.
        assert_eq!(marker_in(&file[..40]), None);
        let mut mp3 = tag(20, 0);
        mp3.extend([0xFF, 0xFB, 0x90, 0x64]);
        assert_eq!(marker_in(&mp3), None);
        assert!(!starts_with_flac(Cursor::new(&mp3), mp3.len() as u64));
        assert!(!starts_with_flac(Cursor::new(&file[..40]), 40));
        assert!(!starts_with_flac(Cursor::new(b"fLa"), 3));
    }

    #[test]
    fn tags_past_the_limit_are_not_followed() {
        let tags = |count: u32| {
            let mut file = Vec::new();
            for _ in 0..count {
                file.extend(tag(0, 0));
            }
            file.extend(b"fLaC");
            file
        };
        let most = tags(MAX_ID3V2_TAGS);
        assert!(marker_in(&most).is_some());
        assert!(starts_with_flac(Cursor::new(&most), most.len() as u64));
        let more = tags(MAX_ID3V2_TAGS + 1);
        assert_eq!(marker_in(&more), None);
        assert!(!starts_with_flac(Cursor::new(&more), more.len() as u64));
    }
}
