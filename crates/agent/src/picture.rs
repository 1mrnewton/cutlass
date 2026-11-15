//! A picture a tool hands the model, such as the frame at a time.
//!
//! The picture is one encoded image file kept in base64, the form every
//! model API takes it in, so a turn that sends it with each round encodes
//! it once. It lives for the turn it was made in: [`forget_pictures`]
//! takes it out of a conversation that is kept or sent again later.

use serde::{Deserialize, Serialize};

use crate::model::Message;

/// The most base64 characters a picture may have. Claude takes 5 MB an
/// image; a frame drawn for a look is a small fraction of that.
pub const PICTURE_LIMIT: usize = 4_000_000;

/// What a tool result says once its picture is gone.
const FORGOTTEN: &str = "\n(the picture is no longer kept; look again if it is needed)";

/// One encoded image.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Picture {
    /// The file's media type, such as `image/jpeg`.
    pub media_type: String,
    /// The file's bytes in base64, with padding and no line breaks.
    pub data: String,
}

impl Picture {
    /// A JPEG file's bytes.
    #[must_use]
    pub fn jpeg(bytes: &[u8]) -> Self {
        Self {
            media_type: "image/jpeg".to_owned(),
            data: base64(bytes),
        }
    }

    /// The picture as a `data:` address, the form a chat completions
    /// service takes.
    #[must_use]
    pub fn data_url(&self) -> String {
        format!("data:{};base64,{}", self.media_type, self.data)
    }
}

impl std::fmt::Debug for Picture {
    /// The type and size, not the bytes: a log or a failed assert would
    /// otherwise print the whole file.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Picture({}, {} base64 characters)",
            self.media_type,
            self.data.len()
        )
    }
}

/// Take the pictures out of `messages`, leaving a note on each result that
/// had one. A host does this to a turn once it is over: the model looked
/// then, and a kept conversation or a later round should not carry the
/// files.
pub fn forget_pictures(messages: &mut [Message]) {
    for message in messages {
        if let Message::Tool {
            content, picture, ..
        } = message
            && picture.take().is_some()
        {
            content.push_str(FORGOTTEN);
        }
    }
}

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// `bytes` in standard base64 (RFC 4648), padded.
fn base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let first = chunk[0];
        let second = chunk.get(1).copied();
        let third = chunk.get(2).copied();
        let group = u32::from(first) << 16
            | u32::from(second.unwrap_or(0)) << 8
            | u32::from(third.unwrap_or(0));
        let letter = |shift: u32| {
            // A six-bit value indexes the 64 letters.
            char::from(ALPHABET[((group >> shift) & 0x3f) as usize])
        };
        out.push(letter(18));
        out.push(letter(12));
        out.push(if second.is_some() { letter(6) } else { '=' });
        out.push(if third.is_some() { letter(0) } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Picture, base64, forget_pictures};
    use crate::model::Message;

    #[test]
    fn base64_matches_the_rfc_4648_vectors() {
        for (bytes, text) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64(bytes.as_bytes()), text);
        }
        // Every letter of the alphabet, the last two included.
        assert_eq!(base64(&[0xfb, 0xff, 0xfe]), "+//+");
        assert_eq!(base64(&[0x00, 0x10, 0x83]), "ABCD");
    }

    #[test]
    fn a_jpeg_is_kept_in_base64_and_written_as_a_data_address() {
        let picture = Picture::jpeg(&[0xff, 0xd8, 0xff]);
        assert_eq!(picture.media_type, "image/jpeg");
        assert_eq!(picture.data, "/9j/");
        assert_eq!(picture.data_url(), "data:image/jpeg;base64,/9j/");
        // A log names the picture without printing it.
        assert_eq!(
            format!("{picture:?}"),
            "Picture(image/jpeg, 4 base64 characters)"
        );
    }

    #[test]
    fn a_picture_is_saved_with_its_result_and_an_old_result_reads_without_one() {
        let seen = Message::Tool {
            call_id: "1".to_owned(),
            name: "player.look".to_owned(),
            content: "frame".to_owned(),
            is_error: false,
            picture: Some(Picture::jpeg(b"abc")),
        };
        let saved = serde_json::to_string(&seen).unwrap();
        assert!(saved.contains("\"picture\""), "{saved}");
        assert_eq!(serde_json::from_str::<Message>(&saved).unwrap(), seen);
        // A result with no picture is saved as it was before pictures.
        let plain = r#"{"role":"tool","call_id":"1","name":"draft.query","content":"revision=0","is_error":false}"#;
        let read: Message = serde_json::from_str(plain).unwrap();
        assert!(matches!(&read, Message::Tool { picture: None, .. }));
        assert_eq!(serde_json::to_string(&read).unwrap(), plain);
    }

    #[test]
    fn forgetting_takes_the_pictures_and_says_so_once() {
        let mut messages = vec![
            Message::User {
                text: "look".to_owned(),
            },
            Message::Tool {
                call_id: "1".to_owned(),
                name: "player.look".to_owned(),
                content: "frame".to_owned(),
                is_error: false,
                picture: Some(Picture::jpeg(b"abc")),
            },
            Message::Tool {
                call_id: "2".to_owned(),
                name: "draft.query".to_owned(),
                content: "revision=0".to_owned(),
                is_error: false,
                picture: None,
            },
        ];
        forget_pictures(&mut messages);
        let Message::Tool {
            content, picture, ..
        } = &messages[1]
        else {
            panic!("a tool result");
        };
        assert!(picture.is_none());
        assert_eq!(
            content,
            "frame\n(the picture is no longer kept; look again if it is needed)"
        );
        let once = messages.clone();
        forget_pictures(&mut messages);
        assert_eq!(messages, once);
        assert!(matches!(&messages[2], Message::Tool { content, .. } if content == "revision=0"));
    }
}
