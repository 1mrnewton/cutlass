//! Streams, edit lists, trim, and gaps.
//!
//! Edit lists, priming, and gaps are metadata. They are not applied to
//! packet timestamps and they are not converted to microseconds.

use crate::limits::Limits;
use cutlass_codec::{StreamParams, Timescale};
use std::fmt;

/// The track id a packet's `stream` field refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TrackId(u32);

impl TrackId {
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A language tag of up to 16 ASCII letters, digits, and hyphens.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Language {
    text: String,
}

/// A language tag this crate will not store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanguageError {
    Empty,
    TooLong { len: usize },
    NotAscii { byte: u8 },
}

impl fmt::Display for LanguageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("language tag is empty"),
            Self::TooLong { len } => write!(formatter, "language tag is {len} bytes"),
            Self::NotAscii { byte } => write!(formatter, "language tag contains byte {byte}"),
        }
    }
}

impl std::error::Error for LanguageError {}

impl Language {
    pub const MAX_LEN: usize = 16;

    pub fn new(text: impl Into<String>) -> Result<Self, LanguageError> {
        let text = text.into();
        if text.is_empty() {
            return Err(LanguageError::Empty);
        }
        if text.len() > Self::MAX_LEN {
            return Err(LanguageError::TooLong { len: text.len() });
        }
        for byte in text.bytes() {
            if !byte.is_ascii_alphanumeric() && byte != b'-' {
                return Err(LanguageError::NotAscii { byte });
            }
        }
        Ok(Self { text })
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.text
    }
}

/// One edit-list entry in the file timescale.
///
/// `media_time` of [`None`] is an empty edit. A present media time is the
/// file's value, including a negative one. Packet timestamps are not shifted
/// by this entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edit {
    media_time: Option<i64>,
    duration: i64,
    rate_numerator: u32,
    rate_denominator: u32,
}

/// An edit list entry that cannot be stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditError {
    NegativeDuration,
    ZeroRate,
}

impl fmt::Display for EditError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegativeDuration => formatter.write_str("edit duration is negative"),
            Self::ZeroRate => formatter.write_str("edit rate denominator is zero"),
        }
    }
}

impl std::error::Error for EditError {}

impl Edit {
    /// `rate_numerator` may be zero. That is a dwell, not a missing rate.
    pub const fn try_new(
        media_time: Option<i64>,
        duration: i64,
        rate_numerator: u32,
        rate_denominator: u32,
    ) -> Result<Self, EditError> {
        if duration < 0 {
            return Err(EditError::NegativeDuration);
        }
        if rate_denominator == 0 {
            return Err(EditError::ZeroRate);
        }
        Ok(Self {
            media_time,
            duration,
            rate_numerator,
            rate_denominator,
        })
    }

    #[must_use]
    pub const fn media_time(self) -> Option<i64> {
        self.media_time
    }

    #[must_use]
    pub const fn duration(self) -> i64 {
        self.duration
    }

    #[must_use]
    pub const fn rate_numerator(self) -> u32 {
        self.rate_numerator
    }

    #[must_use]
    pub const fn rate_denominator(self) -> u32 {
        self.rate_denominator
    }
}

/// Priming and trailing padding, in samples.
///
/// Absence is [`None`]. Zero is a stored zero. Neither value is subtracted
/// from packet timestamps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Trim {
    encoder_delay: Option<u64>,
    trailing_padding: Option<u64>,
}

impl Trim {
    #[must_use]
    pub const fn new(encoder_delay: Option<u64>, trailing_padding: Option<u64>) -> Self {
        Self {
            encoder_delay,
            trailing_padding,
        }
    }

    #[must_use]
    pub const fn encoder_delay(self) -> Option<u64> {
        self.encoder_delay
    }

    #[must_use]
    pub const fn trailing_padding(self) -> Option<u64> {
        self.trailing_padding
    }
}

/// A gap in the file timescale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gap {
    track: TrackId,
    start: i64,
    duration: i64,
}

/// A gap whose duration is not positive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GapError {
    NonPositiveDuration,
}

impl fmt::Display for GapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveDuration => formatter.write_str("gap duration is not positive"),
        }
    }
}

impl std::error::Error for GapError {}

impl Gap {
    pub const fn try_new(track: TrackId, start: i64, duration: i64) -> Result<Self, GapError> {
        if duration <= 0 {
            return Err(GapError::NonPositiveDuration);
        }
        Ok(Self {
            track,
            start,
            duration,
        })
    }

    #[must_use]
    pub const fn track(self) -> TrackId {
        self.track
    }

    #[must_use]
    pub const fn start(self) -> i64 {
        self.start
    }

    #[must_use]
    pub const fn duration(self) -> i64 {
        self.duration
    }
}

/// One stream in an opened file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stream {
    id: TrackId,
    params: StreamParams,
    timescale: Timescale,
    language: Option<Language>,
    edits: Vec<Edit>,
    trim: Trim,
}

impl Stream {
    #[must_use]
    pub fn new(
        id: TrackId,
        params: StreamParams,
        timescale: Timescale,
        language: Option<Language>,
        edits: Vec<Edit>,
        trim: Trim,
    ) -> Self {
        Self {
            id,
            params,
            timescale,
            language,
            edits,
            trim,
        }
    }

    #[must_use]
    pub const fn id(&self) -> TrackId {
        self.id
    }

    #[must_use]
    pub const fn params(&self) -> &StreamParams {
        &self.params
    }

    #[must_use]
    pub const fn timescale(&self) -> Timescale {
        self.timescale
    }

    #[must_use]
    pub const fn language(&self) -> Option<&Language> {
        self.language.as_ref()
    }

    #[must_use]
    pub fn edits(&self) -> &[Edit] {
        &self.edits
    }

    #[must_use]
    pub const fn trim(&self) -> Trim {
        self.trim
    }

    pub(crate) fn within_limits(&self, limits: &Limits) -> bool {
        let extra = u64::try_from(self.params.extradata().len()).unwrap_or(u64::MAX);
        let edits = u64::try_from(self.edits.len()).unwrap_or(u64::MAX);
        extra <= limits.max_extradata_bytes && edits <= u64::from(limits.max_edits)
    }
}

#[cfg(test)]
mod tests {
    use super::{Edit, EditError, Gap, GapError, Language, LanguageError, TrackId, Trim};
    use cutlass_codec::Timescale;

    #[test]
    fn an_edit_keeps_file_ticks_and_an_empty_media_time() {
        let edit = Edit::try_new(None, 90_000, 1, 1).expect("edit");
        assert_eq!(edit.media_time(), None);
        assert_eq!(edit.duration(), 90_000);
        assert_ne!(edit.duration(), 1_000_000);
        let _ = Timescale::new(90_000);
        assert_eq!(
            Edit::try_new(Some(-1), -1, 1, 1),
            Err(EditError::NegativeDuration)
        );
        assert_eq!(Edit::try_new(Some(0), 1, 1, 0), Err(EditError::ZeroRate));
        let negative = Edit::try_new(Some(-1), 10, 0, 1).expect("dwell");
        assert_eq!(negative.media_time(), Some(-1));
        assert_eq!(negative.rate_numerator(), 0);
    }

    #[test]
    fn trim_and_gaps_store_the_file_values() {
        let trim = Trim::new(Some(1_024), None);
        assert_eq!(trim.encoder_delay(), Some(1_024));
        assert_eq!(trim.trailing_padding(), None);
        let absent = Trim::default();
        assert_eq!(absent.encoder_delay(), None);
        let gap = Gap::try_new(TrackId::new(1), 0, 480).expect("gap");
        assert_eq!(gap.duration(), 480);
        assert_eq!(
            Gap::try_new(TrackId::new(1), 0, 0),
            Err(GapError::NonPositiveDuration)
        );
    }

    #[test]
    fn a_language_tag_is_a_short_ascii_token() {
        assert_eq!(Language::new("en").expect("tag").as_str(), "en");
        assert_eq!(Language::new(""), Err(LanguageError::Empty));
        assert_eq!(
            Language::new("has space"),
            Err(LanguageError::NotAscii { byte: b' ' })
        );
    }
}
