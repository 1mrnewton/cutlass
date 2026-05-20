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

    /// Pre-multiplied pixel offset at `px_per_sec`. Done in f64 so that
    /// long timelines (e.g. 36 000 s × 100 px/s = 3.6 M px) keep sub-pixel
    /// precision before the result is cast down to f32 for Slint — the
    /// alternative path through Slint's f32 `float` would lose ~½ frame
    /// at 60 fps once project length crosses ~2 hours.
    pub fn to_pixels(self, px_per_sec: f64) -> f64 {
        self.as_f64() * px_per_sec
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
    BadEnum { field: &'static str, value: String },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::RationalTime;

    // -----------------------------------------------------------------------
    // RationalTime::to_pixels
    //
    // The motivation for `to_pixels` living in Rust (rather than Slint doing
    // `seconds * zoom` in `float`/f32) is precision: f32 has ~7 decimal digits
    // of mantissa, which is enough for ~16 M before integer steps exceed 1 px
    // — i.e. the f32 path corrupts pixel positions on multi-hour timelines at
    // realistic zoom. These tests pin both the easy-case correctness and the
    // hard-case precision behaviour that motivates the API.
    // -----------------------------------------------------------------------

    #[test]
    fn to_pixels_unit_second() {
        assert_eq!(RationalTime::new_raw(1, 1).to_pixels(100.0), 100.0);
    }

    #[test]
    fn to_pixels_half_second() {
        assert_eq!(RationalTime::new_raw(1, 2).to_pixels(100.0), 50.0);
    }

    #[test]
    fn to_pixels_frame_timebase_one_second() {
        // 90_000 / 90_000 == exactly 1 s at the common 90 kHz timebase.
        assert_eq!(RationalTime::new_raw(90_000, 90_000).to_pixels(50.0), 50.0);
    }

    #[test]
    fn to_pixels_zero_is_zero() {
        assert_eq!(RationalTime::ZERO.to_pixels(50.0), 0.0);
        // Any zoom against ZERO is zero (incl. weird zooms).
        assert_eq!(RationalTime::ZERO.to_pixels(0.0), 0.0);
        assert_eq!(RationalTime::ZERO.to_pixels(1_000_000.0), 0.0);
    }

    #[test]
    fn to_pixels_negative_numerator_is_negative() {
        // RationalTime is signed (i64 numerator), so negative times are legal.
        assert_eq!(RationalTime::new_raw(-1, 1).to_pixels(100.0), -100.0);
        assert_eq!(RationalTime::new_raw(-3, 2).to_pixels(40.0), -60.0);
    }

    #[test]
    fn to_pixels_handles_numerator_beyond_i32() {
        // 10 h at 90 kHz timebase => num = 3_240_000_000, which overflows i32
        // (i32::MAX == 2_147_483_647). Slint's `int` is also i32, which is the
        // whole reason `to_pixels` is computed in Rust before crossing the FFI.
        let num: i64 = 90_000_i64 * 10 * 3600;
        assert!(num > i64::from(i32::MAX), "test premise: num must overflow i32");
        let t = RationalTime::new_raw(num, 90_000);
        let px = t.to_pixels(100.0);
        // 10 h * 3600 s * 100 px = 3_600_000 px, exact at this magnitude in f64.
        assert_eq!(px, 3_600_000.0);
    }

    #[test]
    fn to_pixels_precision_beats_f32_on_long_timelines() {
        // Regression guard for the f64 → f32 fix.
        //
        // 9 h at 90 kHz, minus 127 ticks: num = 2_915_999_873, den = 90_000.
        // The "−127 ticks" matters: 9 h exactly (num = 2_916_000_000) happens
        // to be a multiple of 256 and so falls *exactly* on the f32 grid for
        // values in [2^31, 2^32), making the f32 path accidentally exact. The
        // offset puts us at the worst-case half-step between two f32 values.
        //
        // At px_per_sec = 1000 (typical zoomed-in editing), the f32 path drifts
        // > 1 px; the f64 path is precise to a few ULPs.
        let num: i64 = 2_915_999_873;
        let den: u32 = 90_000;
        let px_per_sec: f64 = 1000.0;
        let t = RationalTime::new_raw(num, den);

        let got = t.to_pixels(px_per_sec);

        // Reference computed as an exact rational in i128, scaled to micropixels
        // — `num * px_int * SCALE / den`. With SCALE = 1_000_000 the unit of
        // comparison is micropixels (no floats, no rounding upstream of the f64
        // result being measured).
        const SCALE: i128 = 1_000_000;
        let px_int: i128 = 1000;
        let exact_micropx: i128 = (num as i128) * px_int * SCALE / (den as i128);
        let got_micropx: i128 = (got * SCALE as f64).round() as i128;
        let f64_err_micropx = (exact_micropx - got_micropx).abs();
        assert!(
            f64_err_micropx < SCALE / 1000, // < 0.001 px (well under any visible drift)
            "f64 to_pixels drifted by {} micropx (>{} = 0.001 px) — got {}, exact ≈ {}/{}",
            f64_err_micropx,
            SCALE / 1000,
            got,
            exact_micropx,
            SCALE
        );

        // Same computation through the old Slint path: every operand widens to
        // f32 first, the (num as f32) cast quantizes to the nearest 256-tick
        // multiple, and the error survives the / and * back into pixel space.
        let f32_path: f32 = ((num as f32) / (den as f32)) * (px_per_sec as f32);
        let f32_err_px = (f32_path as f64 - got).abs();
        // Empirically ~1.411 px on this case; assert strictly > 1 px to fail
        // loudly if anyone reverts to the f32 path, and < 3 px to fail loudly
        // if the magnitude of f32 error changes (e.g. someone swaps numbers
        // without rechecking the grid alignment).
        assert!(
            (1.0..3.0).contains(&f32_err_px),
            "expected f32 path to drift in (1, 3) px vs f64; got {} px",
            f32_err_px
        );
    }

    #[test]
    fn to_pixels_linear_in_px_per_sec() {
        // `to_pixels(z1 + z2)` should equal `to_pixels(z1) + to_pixels(z2)`
        // up to a couple of f64 ULPs. Catches refactors that accidentally
        // introduce non-linear scaling (e.g. clamping or rounding inside).
        let t = RationalTime::new_raw(7_654_321, 48_000);
        let (z1, z2): (f64, f64) = (37.5, 211.25);
        let combined = t.to_pixels(z1 + z2);
        let separate = t.to_pixels(z1) + t.to_pixels(z2);
        // Tolerance: a handful of ULPs at result magnitude (~40 000 px).
        let tol = combined.abs().max(separate.abs()) * 8.0 * f64::EPSILON;
        assert!(
            (combined - separate).abs() <= tol,
            "linearity violated: to_pixels({}+{}) = {} but sum = {} (tol={})",
            z1,
            z2,
            combined,
            separate,
            tol
        );
    }
}
