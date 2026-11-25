//! What a clip is in a project template.
//!
//! A project template is a finished edit with some clips marked. A slot is
//! a clip the person using the template fills with their own photo or
//! video; editable text is a text clip whose words they change; music is
//! the one sound they may swap. Everything else stays as the author made
//! it. The mark is the app's own: the reference editor does not read it.

use serde::{Deserialize, Serialize};

/// What a slot takes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotTakes {
    /// A photo or a video.
    #[default]
    Either,
    /// A video only.
    Video,
    /// A photo only.
    Photo,
}

impl SlotTakes {
    /// Whether a medium takes this slot; `photo` is a still.
    #[must_use]
    pub const fn accepts(self, photo: bool) -> bool {
        match self {
            Self::Either => true,
            Self::Video => !photo,
            Self::Photo => photo,
        }
    }
}

/// A clip's mark in a project template, stored on the segment as
/// `cutlass_template`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum TemplateMark {
    /// A photo or video clip the user fills with their own media.
    Slot {
        #[serde(default)]
        takes: SlotTakes,
    },
    /// A text clip whose words the user may change.
    Text,
    /// The sound the user may swap for their own.
    Music,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{SlotTakes, TemplateMark};

    #[test]
    fn marks_store_as_a_role_and_read_back() {
        let marks = [
            (
                TemplateMark::Slot {
                    takes: SlotTakes::Video,
                },
                json!({"role": "slot", "takes": "video"}),
            ),
            (TemplateMark::Text, json!({"role": "text"})),
            (TemplateMark::Music, json!({"role": "music"})),
        ];
        for (mark, stored) in marks {
            assert_eq!(serde_json::to_value(mark).unwrap(), stored);
            assert_eq!(
                serde_json::from_value::<TemplateMark>(stored).unwrap(),
                mark
            );
        }
    }

    #[test]
    fn a_slot_without_takes_takes_either() {
        let mark: TemplateMark = serde_json::from_value(json!({"role": "slot"})).unwrap();
        assert_eq!(
            mark,
            TemplateMark::Slot {
                takes: SlotTakes::Either
            }
        );
    }

    #[test]
    fn an_unknown_role_is_refused() {
        assert!(serde_json::from_value::<TemplateMark>(json!({"role": "logo"})).is_err());
    }

    #[test]
    fn what_each_slot_accepts() {
        assert!(SlotTakes::Either.accepts(true) && SlotTakes::Either.accepts(false));
        assert!(SlotTakes::Video.accepts(false) && !SlotTakes::Video.accepts(true));
        assert!(SlotTakes::Photo.accepts(true) && !SlotTakes::Photo.accepts(false));
    }
}
