//! Domain models for Cutlass: timeline structure, project state, and structured edit
//! payloads. This crate stays UI- and I/O-free — plain data the engine and app share.
//!
//! These types are the **source of truth** for persisted project state. The Slint UI
//! layer mirrors them as flat DTOs (with `string` for IDs and 64-bit time numerators
//! that Slint's `int` can't hold) and converts via `From` / `TryFrom` impls that live
//! in the `app` crate next to the Slint-generated types.

use std::path::PathBuf;

use thiserror::Error;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// IDs — strongly-typed newtypes over Uuid. Prevents accidentally passing a
// TrackId where a ClipId is expected (timeline ops cross-reference these
// constantly, so the type guard pays for itself).
// ---------------------------------------------------------------------------

macro_rules! id_type {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub Uuid);

        impl $name {
            #[inline]
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }

            #[inline]
            pub fn from_uuid(u: Uuid) -> Self {
                Self(u)
            }

            #[inline]
            pub fn as_uuid(&self) -> Uuid {
                self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.0, f)
            }
        }

        impl std::str::FromStr for $name {
            type Err = uuid::Error;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Uuid::parse_str(s).map(Self)
            }
        }
    };
}

id_type!(ProjectId, "Identifier for a [`Project`].");
id_type!(SequenceId, "Identifier for a [`Sequence`].");
id_type!(TrackId, "Identifier for a [`Track`].");
id_type!(ClipId, "Identifier for a [`Clip`].");
id_type!(MediaId, "Identifier for a [`MediaSource`].");

// ---------------------------------------------------------------------------
// Rational time. Shape (`i64` num, `u32` den) mirrors `decoder::Rational` so
// we can unify them later. 64-bit numerator avoids the i32 overflow ceiling
// Slint's `int` would impose at typical project timebases (90_000 caps i32 at
// ~6.6h; with i64 we can hold the heat-death of the universe in microseconds).
// ---------------------------------------------------------------------------

/// Exact `num / den` seconds. `den` must be > 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RationalTime {
    pub num: i64,
    pub den: u32,
}

impl RationalTime {
    pub const ZERO: Self = Self { num: 0, den: 1 };

    /// Returns `None` if `den == 0`.
    pub const fn new(num: i64, den: u32) -> Option<Self> {
        if den == 0 {
            return None;
        }
        Some(Self { num, den })
    }

    /// # Panics
    /// if `den == 0`.
    pub const fn new_raw(num: i64, den: u32) -> Self {
        assert!(den != 0, "RationalTime denominator must be non-zero");
        Self { num, den }
    }

    /// Display-only conversion. **Do not** use for ordering/equality on long timelines.
    pub fn as_f64(self) -> f64 {
        self.num as f64 / f64::from(self.den)
    }
}

impl Default for RationalTime {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Small rational for fps and clip speed (denominators stay tiny, i32 is plenty).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    pub num: i32,
    pub den: u32,
}

impl Rational {
    pub const ONE: Self = Self { num: 1, den: 1 };

    pub const fn new(num: i32, den: u32) -> Option<Self> {
        if den == 0 {
            return None;
        }
        Some(Self { num, den })
    }

    pub const fn new_raw(num: i32, den: u32) -> Self {
        assert!(den != 0, "Rational denominator must be non-zero");
        Self { num, den }
    }

    pub fn as_f32(self) -> f32 {
        self.num as f32 / self.den as f32
    }

    pub fn as_f64(self) -> f64 {
        f64::from(self.num) / f64::from(self.den)
    }
}

impl Default for Rational {
    fn default() -> Self {
        Self::ONE
    }
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    Video,
    Audio,
    Image,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackKind {
    Video,
    Audio,
}

// ---------------------------------------------------------------------------
// Misc value types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchemaVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl SchemaVersion {
    pub const CURRENT: Self = Self {
        major: 0,
        minor: 1,
        patch: 0,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

// ---------------------------------------------------------------------------
// MediaSource — entry in the media bin (file on disk + cached probe data)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MediaSource {
    pub id: MediaId,
    /// Display name; defaults to filename when imported.
    pub name: String,
    /// Absolute path on disk.
    pub path: PathBuf,
    pub kind: MediaKind,
    pub has_video: bool,
    pub has_audio: bool,
    pub duration: RationalTime,
    /// `None` when the source has no video stream.
    pub video: Option<VideoStreamInfo>,
    /// `None` when the source has no audio stream.
    pub audio: Option<AudioStreamInfo>,
    /// Engine probed and reports it can decode this source.
    pub is_supported: bool,
    /// Probe still in flight.
    pub is_loading: bool,
    /// File moved/deleted since import — UI shows red warning.
    pub is_missing: bool,
    /// Human-readable error from the probe, if any.
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct VideoStreamInfo {
    pub width: u32,
    pub height: u32,
    pub fps: Rational,
    pub codec: String,
}

#[derive(Debug, Clone)]
pub struct AudioStreamInfo {
    pub sample_rate: u32,
    pub codec: String,
}

// ---------------------------------------------------------------------------
// Clip — an instance of media on a track.
//
// All `RationalTime` fields on a clip share `den == sequence.timebase`. The
// engine enforces this invariant on commit; ad-hoc construction should pass
// through helpers that quantize/snap to the active sequence timebase.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Clip {
    pub id: ClipId,
    /// `None` for generators / titles / colour mattes.
    pub media_id: Option<MediaId>,
    pub track_id: TrackId,
    /// Label drawn on the clip pill.
    pub name: String,
    /// Position on the timeline.
    pub start: RationalTime,
    /// Length on the timeline.
    pub duration: RationalTime,
    /// Trim into source.
    pub source_in: RationalTime,
    pub source_out: RationalTime,
    /// `1/1` = normal, `-1/1` = reverse, `1/2` = slowmo, etc.
    pub speed: Rational,
    /// 0..=1 (video clips).
    pub opacity: f32,
    /// 0..=2 (audio clips — allows +6 dB).
    pub volume: f32,
    /// Soft-disable without deleting.
    pub enabled: bool,
    pub color: Color,
}

// ---------------------------------------------------------------------------
// Track — a row in a Sequence. Order in `Sequence.tracks` is display order.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Track {
    pub id: TrackId,
    /// "V1", "A1", or user-renamed.
    pub name: String,
    pub kind: TrackKind,
    pub height_px: u32,
    pub muted: bool,
    pub solo: bool,
    pub locked: bool,
    /// Video only — eye toggle.
    pub visible: bool,
    pub clips: Vec<Clip>,
}

// ---------------------------------------------------------------------------
// Sequence — the project timeline.
//
// Pure persistence model: ephemeral UI state (playhead, zoom, scroll) lives
// in the Slint UI layer and is overlaid when constructing the DTO.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: SequenceId,
    pub name: String,
    /// Canvas dimensions (px).
    pub width: u32,
    pub height: u32,
    pub fps: Rational,
    pub sample_rate: u32,
    /// Canonical ticks-per-second for every `RationalTime` in this sequence.
    pub timebase: u32,
    /// Total length, frame-quantized on commit.
    pub duration: RationalTime,
    /// Export range start (None = no in-point set).
    pub in_point: Option<RationalTime>,
    /// Export range end (None = no out-point set).
    pub out_point: Option<RationalTime>,
    pub tracks: Vec<Track>,
}

// ---------------------------------------------------------------------------
// Project — one .cutlass file.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Project {
    pub id: ProjectId,
    pub name: String,
    /// `None` when the project is unsaved.
    pub file_path: Option<PathBuf>,
    pub schema: SchemaVersion,
    pub sequence: Sequence,
    pub media_bin: Vec<MediaSource>,
    pub is_dirty: bool,
}

// ---------------------------------------------------------------------------
// Errors used by DTO ⇄ domain conversion.
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ModelParseError {
    #[error("invalid {field} uuid: {source}")]
    BadUuid {
        field: &'static str,
        #[source]
        source: uuid::Error,
    },

    #[error("invalid {field} integer `{value}`: {source}")]
    BadInt {
        field: &'static str,
        value: String,
        #[source]
        source: std::num::ParseIntError,
    },

    #[error("invalid rational denominator in {field}: must be > 0")]
    BadDenominator { field: &'static str },

    #[error("unknown enum value `{value}` for {field}")]
    BadEnum {
        field: &'static str,
        value: String,
    },
}
