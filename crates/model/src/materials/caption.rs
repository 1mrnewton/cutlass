//! The speech fields a caption carries next to its words.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use super::words::WordColumns;

/// A field kept the way it was read: left out, `null`, or a value.
///
/// A draft that never had the key writes none, and one that wrote `null`
/// writes `null` again.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Kept<T> {
    #[default]
    Absent,
    Null,
    Set(T),
}

impl<T> Kept<T> {
    /// The value, when there is one.
    #[must_use]
    pub const fn get(&self) -> Option<&T> {
        match self {
            Self::Set(value) => Some(value),
            Self::Absent | Self::Null => None,
        }
    }

    /// Serde's skip test: an absent key stays absent.
    #[must_use]
    pub const fn is_absent(&self) -> bool {
        matches!(self, Self::Absent)
    }
}

impl<T: Serialize> Serialize for Kept<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Set(value) => value.serialize(serializer),
            Self::Absent | Self::Null => serializer.serialize_none(),
        }
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Kept<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Option::<T>::deserialize(deserializer)?.map_or(Self::Null, Self::Set))
    }
}

/// Marked spans of a caption's text.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct SubtitleKeywords {
    #[serde(default, skip_serializing_if = "Kept::is_absent")]
    pub range: Kept<Vec<TextRange>>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// The caption template a caption was styled with.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct CaptionTemplateInfo {
    #[serde(default)]
    pub resource_id: String,
    #[serde(default)]
    pub resource_name: String,
    /// Keys this model does not name: the package path, its category, and
    /// where it came from.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// A span of text: `location` and `length` count UTF-16 units of the
/// material's `content`.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct TextRange {
    #[serde(default)]
    pub location: i64,
    #[serde(default)]
    pub length: i64,
    /// Keys this model does not name, `source_type` among them.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// Words whose text overlaps a marked span, by index into `words`.
///
/// Each word is found in `content` after the word before it, so a word
/// spelled twice is matched where it stands. A span that covers part of a
/// word marks the whole word. A word not found in `content` is skipped, and
/// so is a line break marker.
#[must_use]
pub fn keywords_in_ranges(content: &str, words: &WordColumns, ranges: &[TextRange]) -> Vec<usize> {
    let mut marked = Vec::new();
    let mut from = 0;
    for (index, word) in words.text.iter().enumerate() {
        if word.trim_matches('\n').is_empty() {
            continue;
        }
        let Some(found) = content
            .get(from..)
            .and_then(|rest| rest.find(word.as_str()))
        else {
            continue;
        };
        let start_byte = from + found;
        from = start_byte + word.len();
        let start = utf16_len(&content[..start_byte]);
        let end = start + utf16_len(word);
        let hit = ranges.iter().any(|range| {
            let Ok(location) = usize::try_from(range.location) else {
                return false;
            };
            let Ok(length) = usize::try_from(range.length) else {
                return false;
            };
            length > 0 && location < end && start < location.saturating_add(length)
        });
        if hit {
            marked.push(index);
        }
    }
    marked
}

fn utf16_len(text: &str) -> usize {
    text.chars().map(char::len_utf16).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &[&str]) -> WordColumns {
        WordColumns {
            start_time: (0..text.len() as i64).map(|i| i * 100).collect(),
            end_time: (1..=text.len() as i64).map(|i| i * 100).collect(),
            text: text.iter().map(|word| (*word).to_owned()).collect(),
            rest: Map::new(),
        }
    }

    fn range(location: i64, length: i64) -> TextRange {
        TextRange {
            location,
            length,
            rest: Map::new(),
        }
    }

    #[test]
    fn a_span_marks_each_word_it_touches() {
        let columns = words(&["so", "very", "good"]);
        assert_eq!(
            keywords_in_ranges("so very good", &columns, &[range(3, 4)]),
            [1]
        );
        // Part of "very" and part of "good".
        assert_eq!(
            keywords_in_ranges("so very good", &columns, &[range(5, 4)]),
            [1, 2]
        );
        assert_eq!(
            keywords_in_ranges("so very good", &columns, &[range(0, 1), range(8, 4)]),
            [0, 2]
        );
    }

    #[test]
    fn locations_count_utf16_units_and_repeated_words_match_in_turn() {
        // The emoji is two UTF-16 units, so "go" starts at 3.
        let columns = words(&["😀", "go", "go"]);
        assert_eq!(
            keywords_in_ranges("😀 go go", &columns, &[range(6, 2)]),
            [2]
        );
        assert_eq!(
            keywords_in_ranges("😀 go go", &columns, &[range(3, 2)]),
            [1]
        );
    }

    #[test]
    fn empty_negative_and_far_spans_mark_nothing() {
        let columns = words(&["a", "\n", "b"]);
        let content = "a\nb";
        assert!(keywords_in_ranges(content, &columns, &[]).is_empty());
        assert!(keywords_in_ranges(content, &columns, &[range(0, 0)]).is_empty());
        assert!(keywords_in_ranges(content, &columns, &[range(-1, 5)]).is_empty());
        assert!(keywords_in_ranges(content, &columns, &[range(0, -1)]).is_empty());
        assert!(keywords_in_ranges(content, &columns, &[range(99, 1)]).is_empty());
        assert!(keywords_in_ranges(content, &columns, &[range(i64::MAX, i64::MAX)]).is_empty());
        assert_eq!(keywords_in_ranges(content, &columns, &[range(1, 2)]), [2]);
        assert!(keywords_in_ranges("", &columns, &[range(0, 9)]).is_empty());
    }
}
