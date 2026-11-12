//! Whether a video chunk starts a random-access point, read from its first
//! bytes. Only a rebuilt index uses this: `idx1` and OpenDML indexes carry
//! the keyframe flag themselves.
//!
//! The checks are the frame-type fields of each bitstream: H.264 `nal_unit_type`
//! 5 (IDR), HEVC IRAP types 16 to 23, MPEG-4 Part 2 `vop_coding_type` 0,
//! MPEG-1/2 `picture_coding_type` 1, VP8 and VP9 `frame_type` 0, and an AV1
//! sequence header OBU. A chunk the window does not decide is not a keyframe.

use cutlass_codec::CodecId;

pub(crate) fn keyframe(codec: Option<CodecId>, head: &[u8]) -> bool {
    match codec {
        Some(CodecId::H264) => nal_key(head, |byte| match byte & 0x1F {
            5 => Some(true),
            1..=4 => Some(false),
            _ => None,
        }),
        Some(CodecId::Hevc) => nal_key(head, |byte| match (byte >> 1) & 0x3F {
            16..=23 => Some(true),
            0..=9 => Some(false),
            _ => None,
        }),
        Some(CodecId::Mpeg4) => after_code(head, 0xB6).is_some_and(|byte| byte >> 6 == 0),
        Some(CodecId::Mpeg2) => after_code(head, 0x00)
            .zip(after_code_at(head, 0x00, 1))
            .is_some_and(|(_, second)| (second >> 3) & 0x07 == 1),
        Some(CodecId::Vp8) => head.first().is_some_and(|byte| byte & 1 == 0),
        Some(CodecId::Vp9) => vp9_key(head),
        Some(CodecId::Av1) => av1_key(head),
        Some(CodecId::Mjpeg) => true,
        _ => false,
    }
}

/// Start-code NAL units, or four-byte length-prefixed ones. `kind` returns
/// `Some` for the first picture NAL unit it can decide.
fn nal_key(head: &[u8], kind: impl Fn(u8) -> Option<bool>) -> bool {
    let mut at = 0;
    while at + 3 < head.len() {
        if head[at] == 0 && head[at + 1] == 0 && head[at + 2] == 1 {
            if let Some(decided) = kind(head[at + 3]) {
                return decided;
            }
            at += 3;
        } else {
            at += 1;
        }
    }
    let mut at = 0;
    while at + 4 < head.len() {
        let len = u32::from_be_bytes([head[at], head[at + 1], head[at + 2], head[at + 3]]);
        if len == 0 || head[at + 4] & 0x80 != 0 {
            return false;
        }
        if let Some(decided) = kind(head[at + 4]) {
            return decided;
        }
        let Some(next) = usize::try_from(len)
            .ok()
            .and_then(|len| at.checked_add(4)?.checked_add(len))
        else {
            return false;
        };
        at = next;
    }
    false
}

fn after_code(head: &[u8], code: u8) -> Option<u8> {
    after_code_at(head, code, 0)
}

fn after_code_at(head: &[u8], code: u8, offset: usize) -> Option<u8> {
    let at = head
        .windows(4)
        .position(|window| window == [0, 0, 1, code])?;
    head.get(at + 4 + offset).copied()
}

fn vp9_key(head: &[u8]) -> bool {
    let Some(&byte) = head.first() else {
        return false;
    };
    if byte >> 6 != 0b10 {
        return false;
    }
    let profile = ((byte >> 5) & 1) | (((byte >> 4) & 1) << 1);
    let mut bit = 4u8;
    if profile == 3 {
        bit -= 1;
    }
    let show_existing = (byte >> (bit - 1)) & 1;
    if show_existing == 1 {
        return false;
    }
    (byte >> (bit - 2)) & 1 == 0
}

fn av1_key(head: &[u8]) -> bool {
    let mut at = 0;
    while let Some(&header) = head.get(at) {
        let kind = (header >> 3) & 0x0F;
        if kind == 1 {
            return true;
        }
        let extension = usize::from((header >> 2) & 1);
        if (header >> 1) & 1 == 0 {
            return false;
        }
        let mut size = 0usize;
        let mut read = 0;
        at += 1 + extension;
        loop {
            let Some(&byte) = head.get(at) else {
                return false;
            };
            size |= usize::from(byte & 0x7F) << (7 * read);
            at += 1;
            read += 1;
            if byte & 0x80 == 0 || read == 8 {
                break;
            }
        }
        let Some(next) = at.checked_add(size) else {
            return false;
        };
        at = next;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::keyframe;
    use cutlass_codec::CodecId;

    #[test]
    fn each_bitstream_names_its_keyframes() {
        let idr = [0, 0, 0, 1, 0x67, 0x42, 0, 0, 0, 1, 0x65, 0x88];
        let slice = [0, 0, 0, 1, 0x41, 0x9A];
        assert!(keyframe(Some(CodecId::H264), &idr));
        assert!(!keyframe(Some(CodecId::H264), &slice));
        let prefixed = [0, 0, 0, 2, 0x09, 0x10, 0, 0, 0, 1, 0x65];
        assert!(keyframe(Some(CodecId::H264), &prefixed));
        assert!(keyframe(Some(CodecId::Hevc), &[0, 0, 1, 0x26, 0x01]));
        assert!(!keyframe(Some(CodecId::Hevc), &[0, 0, 1, 0x02, 0x01]));
        assert!(keyframe(Some(CodecId::Mpeg4), &[0, 0, 1, 0xB6, 0x10]));
        assert!(!keyframe(Some(CodecId::Mpeg4), &[0, 0, 1, 0xB6, 0x50]));
        assert!(keyframe(Some(CodecId::Mpeg2), &[0, 0, 1, 0xB3, 9, 0, 0, 1, 0x00, 0x00, 0x0F]));
        assert!(!keyframe(Some(CodecId::Mpeg2), &[0, 0, 1, 0x00, 0x00, 0x17]));
        assert!(keyframe(Some(CodecId::Vp8), &[0x10]));
        assert!(!keyframe(Some(CodecId::Vp8), &[0x11]));
        assert!(keyframe(Some(CodecId::Vp9), &[0x82]));
        assert!(!keyframe(Some(CodecId::Vp9), &[0x86]));
        assert!(keyframe(Some(CodecId::Av1), &[0x12, 0x00, 0x0A, 0x01, 0x00]));
        assert!(!keyframe(Some(CodecId::Av1), &[0x32, 0x01, 0x00]));
        assert!(keyframe(Some(CodecId::Mjpeg), &[]));
        assert!(!keyframe(None, &[0; 8]));
        assert!(!keyframe(Some(CodecId::H264), &[]));
        assert!(!keyframe(Some(CodecId::H264), &[0, 0, 0, 0, 0x65]));
        assert!(!keyframe(Some(CodecId::H264), &[0, 0, 0, 9, 0xE5]));
    }
}
