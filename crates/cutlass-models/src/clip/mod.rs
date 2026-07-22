use serde::{Deserialize, Serialize};

use crate::effects::EffectInstance;
use crate::error::ModelError;
use crate::ids::{ClipId, LinkId, MediaId};
use crate::param::Param;
use crate::time::{Rational, RationalTime, TimeRange, check_same_rate};
// Kept in scope for `clip/tests` (`use super::*`); not referenced by mod.rs itself.
#[cfg(test)]
#[allow(unused_imports)]
use crate::param::{Easing, Keyframe};

mod crop;
mod generator;
mod retiming;
mod template;
mod text;
mod transform;

#[cfg(test)]
mod tests;

pub use crop::{CropRect, MIN_CROP_FRACTION};
pub use generator::{
    ClipSource, Generator, MAX_SHAPE_DIM, MAX_STAR_POINTS, MAX_STROKE_WIDTH, SHAPE_DROP_HEIGHT,
    SHAPE_DROP_WIDTH, Shape, ShapePath, ShapePathPoint, ShapeStroke,
};
pub(crate) use retiming::split_speed_curve;
pub use retiming::{
    MAX_SPEED, MIN_SPEED, SPEED_CURVE_SCALE, SpeedPresetSpec, speed_curve_integral,
    speed_curve_source_fraction, speed_preset, speed_preset_catalog, speed_preset_id,
    validate_speed_curve,
};
pub use template::{Replaceable, SlotMedia};
pub use text::{
    TextAlignH, TextAlignV, TextBackground, TextCase, TextShadow, TextStroke, TextStyle,
};
pub use transform::{
    AnimatedTransform, ClipParam, ClipTransform, LookParam, ParamValue, Scale2, ShapeParam,
    StyleParam, TextParam,
};

/// A placement of some [`ClipSource`] on a track.
///
/// `timeline` is where the clip sits on the sequence, at the timeline rate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Clip {
    pub id: ClipId,
    pub content: ClipSource,
    pub timeline: TimeRange,
    /// A media-backed still made from one resolved video frame. The source
    /// window is exactly one native frame while `timeline` may have any
    /// positive duration. Frozen clips never own media audio and cannot be
    /// retimed. Absent from saves for ordinary clips.
    #[serde(default, skip_serializing_if = "is_false")]
    pub freeze_frame: bool,
    /// Link group (CapCut linkage): clips sharing a `LinkId` are selected,
    /// moved, and trimmed together — e.g. the video+audio pair created by
    /// dropping media with an audio stream. `None` ⇔ unlinked.
    #[serde(default)]
    pub link: Option<LinkId>,
    /// Spatial placement on the canvas, animatable per property. Identity
    /// (aspect-fit, centered) for clips created before transforms existed.
    /// Ignored on audio tracks. Sample at a clip-relative tick via
    /// [`AnimatedTransform::sample`]; never-animated transforms serialize
    /// exactly like the pre-M2 plain [`ClipTransform`].
    #[serde(default)]
    pub transform: AnimatedTransform,
    /// How this clip's pixels blend with the stack below (CapCut "Blend",
    /// visual clips only). `Normal` (and absent from saves) when never set,
    /// so old projects load unchanged.
    #[serde(default, skip_serializing_if = "is_normal_blend")]
    pub blend_mode: crate::look::BlendMode,
    /// Layer-quad styles (shadow/glow/outline/background) rendered from this
    /// clip's alpha by the compositor. Empty (and absent from saves) when
    /// never styled, so old projects load unchanged. Visual clips only.
    #[serde(default, skip_serializing_if = "crate::look::LayerStyles::is_empty")]
    pub styles: crate::look::LayerStyles,
    /// Playback rate (CapCut speed, M1): source time advances `speed`× per
    /// unit of timeline time — `2/1` plays double speed (the clip occupies
    /// half its source duration on the timeline), `1/2` is 50% slow motion.
    /// Always positive; direction is the separate `reversed` flag. Stored
    /// as an exact rational so source-tick math never drifts. Meaningful on
    /// media clips only; `1/1` (and absent from saves) when never retimed,
    /// so old files load unchanged and untouched projects keep their shape.
    #[serde(default = "unit_speed", skip_serializing_if = "is_unit_speed")]
    pub speed: Rational,
    /// Play the source window backwards (timeline forward ⇒ source
    /// backward). Media clips only; absent from saves while false.
    #[serde(default, skip_serializing_if = "is_false")]
    pub reversed: bool,
    /// Playback-rate ramp (CapCut speed curves, M2): the instantaneous speed
    /// *multiplier* over the clip's normalized span. Constant `1.0` (the
    /// default, omitted from saves) ⇔ a flat ramp, so `speed`/`reversed`
    /// alone govern retiming and old/never-rammed clips are byte-identical.
    ///
    /// Keyframe ticks are normalized to `0..=`[`SPEED_CURVE_SCALE`] (`0` =
    /// clip start, `SPEED_CURVE_SCALE` = clip end), so the ramp's *shape*
    /// rides along when the clip is trimmed or its base speed changes. Speed
    /// is a rate: the source position swept to a point in the clip is the
    /// integral of `speed × speed_curve`, and the clip's timeline duration
    /// re-derives from the curve's average (see [`Clip::source_time_at`] and
    /// [`crate::Project::set_clip_speed_curve`]). Meaningful on media clips
    /// only.
    #[serde(
        default = "default_speed_curve",
        skip_serializing_if = "is_unit_speed_curve"
    )]
    pub speed_curve: Param<f32>,
    /// Preserve pitch while retiming (CapCut's "pitch" toggle, M8 Phase 3).
    /// `true` (the default) time-stretches the audio so a sped-up clip keeps
    /// its original pitch; `false` is "chipmunk" mode where pitch rides the
    /// speed. Meaningful on retimed media clips only; `true` (and absent from
    /// saves) otherwise, so old files load pitch-locked.
    #[serde(default = "default_preserve_pitch", skip_serializing_if = "is_true")]
    pub preserve_pitch: bool,
    /// Audio gain envelope (CapCut volume, M1 → M8): `0.0` mutes, `1.0` is
    /// unchanged, up to [`MAX_CLIP_VOLUME`]× boost. Read by both audio mixers
    /// for clips on audio lanes; meaningless elsewhere. A constant for the
    /// common case (byte-identical to the pre-M8 bare-`f32` shape, so old
    /// files load unchanged), or a keyframed [`Param`] envelope (M8): the
    /// mixers sample it per sample-frame, and ducking writes ordinary volume
    /// keyframes. Keyframe ticks are clip-relative timeline ticks, like every
    /// other [`Param`]. `1.0` (and absent from saves) when never touched.
    #[serde(default = "default_volume", skip_serializing_if = "is_unit_volume")]
    pub volume: Param<f32>,
    /// Stereo pan envelope (CapCut left/right balance): `−1.0` full left,
    /// `0.0` center, `+1.0` full right. Read by both audio mixers for
    /// media-backed clips (video with sound and audio-lane clips); meaningless
    /// elsewhere. A constant for the common case (absent from saves while
    /// centered), or a keyframed [`Param`] envelope: the mixers sample it per
    /// sample-frame. Keyframe ticks are clip-relative timeline ticks, like
    /// every other [`Param`]. `0.0` (and absent from saves) when never touched.
    #[serde(default = "default_pan", skip_serializing_if = "is_zero_pan")]
    pub pan: Param<f32>,
    /// Fade-in duration in timeline ticks from the clip's start: a linear
    /// gain ramp 0 → `volume`. First-class field like CapCut, not keyframe
    /// sugar. Absent from saves while 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub fade_in: i64,
    /// Fade-out duration in timeline ticks ending at the clip's end: a
    /// linear gain ramp `volume` → 0. Absent from saves while 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub fade_out: i64,
    /// Noise reduction (CapCut "Reduce noise", M8 Phase 5): run this clip's
    /// audio through RNNoise to suppress steady background noise (hiss, hum,
    /// room tone) while keeping speech. Both audio mixers render the cleaned
    /// signal; meaningful for clips on audio lanes, ignored elsewhere. Pitch-
    /// preserving time-stretch for retimed clips is still deferred — varispeed
    /// resampling is used today. `false` (and absent from saves) when off, so
    /// old files load unchanged.
    #[serde(default, skip_serializing_if = "is_false")]
    pub denoise: bool,
    /// Normalized crop window into the content (CapCut crop, M1): only the
    /// kept region renders, aspect-fit and transformed like the full frame
    /// was. Meaningful on visual clips; full-frame constant (and absent from
    /// saves) when never cropped, so old files load unchanged. Constants
    /// serialize as the bare rect (`{"x", "y", "w", "h"}`); keyframed as
    /// `{"kf":[...]}`.
    #[serde(
        default = "default_crop",
        skip_serializing_if = "is_constant_full_crop"
    )]
    pub crop: Param<CropRect>,
    /// Mirror the content left-right (after crop). Absent from saves while
    /// false.
    #[serde(default, skip_serializing_if = "is_false")]
    pub flip_h: bool,
    /// Mirror the content top-bottom (after crop). Absent from saves while
    /// false.
    #[serde(default, skip_serializing_if = "is_false")]
    pub flip_v: bool,
    /// GPU effect chain (CapCut effects, M4): applied in order to the placed
    /// layer before it composites. Each entry is `{effect_id, params}` with
    /// parameters animatable per the catalog. Meaningful on visual clips;
    /// empty (and absent from saves) when never touched, so old files load
    /// unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<EffectInstance>,
    /// Detected beat positions (CapCut "Beat" markers, M8 Phase 6): source
    /// ticks at the media frame rate, sorted ascending. `DetectBeats` fills
    /// them from onset analysis; the timeline magnet snaps clip edges to them.
    /// Stored in *source* time so they ride the content — trims and splits
    /// keep exactly the beats inside each half's window visible (see
    /// [`Clip::beat_timeline_ticks`]). Meaningful on media clips; empty (and
    /// absent from saves) until detected, so old files load unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub beats: Vec<i64>,
    /// Shaped alpha mask (CapCut mask, Phase I): persisted + validated now,
    /// rendered later (documented render gap, like stickers). Meaningful on
    /// media-backed visual clips; `None` (and absent from saves) otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<crate::look::Mask>,
    /// Chroma keying (CapCut chroma key, Phase I): render-neutral this
    /// milestone. Media-backed visual clips only; absent while `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chroma_key: Option<crate::look::ChromaKey>,
    /// Stabilization strength (CapCut stabilize, Phase I): render-neutral
    /// this milestone. Media-backed *video* clips only (stills have no
    /// motion); absent while `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stabilize: Option<crate::look::StabilizeLevel>,
    /// Color-grade filter preset (CapCut filters, Phase I): render-neutral
    /// this milestone. Visual clips — including `Generator::Filter` lane
    /// bars, whose picked preset lives here; absent while `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<crate::look::Filter>,
    /// `.cube` 3D LUT (applied after filter + adjust): file-backed color
    /// lookup from the asset catalog or a user file. Visual clips — including
    /// `Generator::Filter` lane bars, where it grades everything beneath;
    /// absent while `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lut: Option<crate::look::Lut>,
    /// Manual color grade (CapCut adjust, Phase I): render-neutral this
    /// milestone. Visual clips — including `Generator::Adjustment` lane
    /// bars; neutral (and absent from saves) when never touched.
    #[serde(
        default,
        skip_serializing_if = "crate::look::ColorAdjustments::is_neutral"
    )]
    pub adjust: crate::look::ColorAdjustments,
    /// Entrance animation preset (CapCut animation In tab, Phase I):
    /// render-neutral this milestone. Visual clips; mutually exclusive with
    /// `animation_combo` (enforced by [`crate::Project::set_clip_animation`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation_in: Option<crate::look::AnimationRef>,
    /// Exit animation preset (CapCut animation Out tab, Phase I).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation_out: Option<crate::look::AnimationRef>,
    /// Looping presence animation (CapCut animation Combo tab, Phase I):
    /// replaces both entrance and exit while set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation_combo: Option<crate::look::AnimationRef>,
    /// What this audio-lane clip *is* (music / sound FX / voiceover /
    /// extracted, Phase I): badges and future mixing defaults. Audio-track
    /// clips only; absent while `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_role: Option<crate::look::AudioRole>,
    /// CapCut-style replaceable placeholder marker (templates). `None` for an
    /// ordinary clip; `Some` marks this clip as a user-fillable slot. Absent
    /// from saves while `None`, so non-template projects stay byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replaceable: Option<Replaceable>,
    /// Whether a viewer may re-word this clip's text when using the project as
    /// a template (the text keeps its style and animation). Meaningful on
    /// `Generator::Text` clips; `false` (and absent from saves) otherwise.
    #[serde(default, skip_serializing_if = "is_false")]
    pub text_editable: bool,
}

/// Upper bound for [`Clip::volume`] (CapCut's 1000% ceiling).
pub const MAX_CLIP_VOLUME: f32 = 10.0;

/// Bounds for [`Clip::pan`]: full left … full right.
pub const MIN_CLIP_PAN: f32 = -1.0;
pub const MAX_CLIP_PAN: f32 = 1.0;

/// Default entrance/exit look-animation window (~0.5 seconds), shortened to
/// half the clip for short placements. Kept in the model so structural edits
/// and the renderer use exactly the same timing rule.
pub fn look_animation_window_ticks(duration: i64, rate: Rational) -> i64 {
    const DEFAULT_ANIMATION_SECONDS: f64 = 0.5;
    let from_seconds = (DEFAULT_ANIMATION_SECONDS / rate.seconds_per_unit()).ceil() as i64;
    from_seconds.max(1).min((duration / 2).max(1))
}

/// Loop period for combo/presence look animations (~1 second).
pub fn look_animation_combo_period_ticks(rate: Rational) -> i64 {
    const COMBO_PERIOD_SECONDS: f64 = 1.0;
    (COMBO_PERIOD_SECONDS / rate.seconds_per_unit())
        .round()
        .max(1.0) as i64
}

fn unit_speed() -> Rational {
    Rational::new(1, 1)
}

pub(super) fn is_unit_speed(speed: &Rational) -> bool {
    speed.num == speed.den
}

fn default_speed_curve() -> Param<f32> {
    Param::Constant(1.0)
}

/// A flat unit ramp — no retiming contribution. `&` form for serde's
/// `skip_serializing_if`.
pub(super) fn is_unit_speed_curve(curve: &Param<f32>) -> bool {
    matches!(curve, Param::Constant(v) if *v == 1.0)
}

fn default_preserve_pitch() -> bool {
    true
}

// `&bool` is the signature `skip_serializing_if` requires.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_true(b: &bool) -> bool {
    *b
}

fn default_volume() -> Param<f32> {
    Param::Constant(1.0)
}

/// A flat unit-gain envelope — no audio edit. `&` form for serde's
/// `skip_serializing_if`.
fn is_unit_volume(volume: &Param<f32>) -> bool {
    matches!(volume, Param::Constant(v) if *v == 1.0)
}

fn default_pan() -> Param<f32> {
    Param::Constant(0.0)
}

/// A flat center pan — no stereo edit. `&` form for serde's
/// `skip_serializing_if`.
fn is_zero_pan(pan: &Param<f32>) -> bool {
    matches!(pan, Param::Constant(v) if *v == 0.0)
}

fn default_crop() -> Param<CropRect> {
    Param::Constant(CropRect::FULL)
}

/// Full-frame constant crop — absent from saves. Keyframed crops always
/// serialize (even when every keyframe is full).
fn is_constant_full_crop(crop: &Param<CropRect>) -> bool {
    matches!(crop, Param::Constant(c) if c.is_full())
}

/// Range check for one volume value: finite, within `0..=`[`MAX_CLIP_VOLUME`].
/// Shared by `set_clip_audio`, the envelope keyframe routing, and load-time
/// envelope validation.
pub fn validate_volume(v: f32) -> Result<(), ModelError> {
    if !v.is_finite() || !(0.0..=MAX_CLIP_VOLUME).contains(&v) {
        return Err(ModelError::InvalidParam(format!(
            "volume must be between 0 and {MAX_CLIP_VOLUME}"
        )));
    }
    Ok(())
}

/// Validate a volume envelope (M8) before it is stored: structurally sound
/// (sorted, non-empty when keyframed, valid easings) with every value in
/// gain range.
pub fn validate_volume_envelope(volume: &Param<f32>) -> Result<(), ModelError> {
    volume.validate_shape()?;
    volume.for_each_value(|v| validate_volume(*v))
}

/// Range check for one pan value: finite, within [`MIN_CLIP_PAN`]..=[`MAX_CLIP_PAN`].
/// Shared by the envelope keyframe routing and load-time envelope validation.
pub fn validate_pan(v: f32) -> Result<(), ModelError> {
    if !v.is_finite() || !(MIN_CLIP_PAN..=MAX_CLIP_PAN).contains(&v) {
        return Err(ModelError::InvalidParam(format!(
            "pan must be between {MIN_CLIP_PAN} and {MAX_CLIP_PAN}"
        )));
    }
    Ok(())
}

/// Validate a pan envelope before it is stored: structurally sound with every
/// value in pan range.
pub fn validate_pan_envelope(pan: &Param<f32>) -> Result<(), ModelError> {
    pan.validate_shape()?;
    pan.for_each_value(|v| validate_pan(*v))
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_zero(ticks: &i64) -> bool {
    *ticks == 0
}

/// Audio gain at `pos` within a span of `len` (clip-relative, any unit —
/// ticks or sample frames — as long as all arguments and the `volume`
/// envelope share it): the envelope sampled at `pos` shaped by the linear
/// fade ramps. Fades anchor at the span edges, so a fade longer than a
/// trimmed span just ramps part-way. Both audio mixers evaluate this per
/// sample frame; keep it branch-light. The mixers rebase the envelope into
/// the sample-frame domain once per span ([`Param::map_ticks`]) so this
/// stays an O(log k) lookup, not a tick conversion.
pub fn audio_gain_at(pos: i64, len: i64, volume: &Param<f32>, fade_in: i64, fade_out: i64) -> f32 {
    let mut gain = volume.sample(pos);
    if fade_in > 0 && pos < fade_in {
        gain *= pos.max(0) as f32 / fade_in as f32;
    }
    if fade_out > 0 {
        let remain = len - pos;
        if remain < fade_out {
            gain *= remain.max(0) as f32 / fade_out as f32;
        }
    }
    gain
}

// `&bool` is the signature `skip_serializing_if` requires.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_false(b: &bool) -> bool {
    !*b
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_normal_blend(b: &crate::look::BlendMode) -> bool {
    b.is_normal()
}

impl Clip {
    /// A clip backed by a trimmed range of imported media.
    pub fn from_media(media: MediaId, source: TimeRange, timeline: TimeRange) -> Self {
        Self {
            id: ClipId::next(),
            content: ClipSource::Media { media, source },
            timeline,
            freeze_frame: false,
            link: None,
            transform: AnimatedTransform::identity(),
            blend_mode: crate::look::BlendMode::default(),
            styles: crate::look::LayerStyles::default(),
            speed: unit_speed(),
            reversed: false,
            speed_curve: default_speed_curve(),
            preserve_pitch: default_preserve_pitch(),
            volume: default_volume(),
            pan: default_pan(),
            fade_in: 0,
            fade_out: 0,
            denoise: false,
            crop: default_crop(),
            flip_h: false,
            flip_v: false,
            effects: Vec::new(),
            beats: Vec::new(),
            mask: None,
            chroma_key: None,
            stabilize: None,
            filter: None,
            lut: None,
            adjust: crate::look::ColorAdjustments::default(),
            animation_in: None,
            animation_out: None,
            animation_combo: None,
            audio_role: None,
            replaceable: None,
            text_editable: false,
        }
    }

    /// Build the audio-lane companion for CapCut-style audio extraction.
    ///
    /// This is deliberately placement- and linkage-free: the engine owns lane
    /// selection, atomic insertion, and the shared [`LinkId`]. Only properties
    /// that affect audio playback ride across. Every visual, look, animation,
    /// and template field starts from the ordinary media-clip defaults.
    pub fn extracted_audio_companion(&self) -> Result<Self, ModelError> {
        let ClipSource::Media { media, source } = &self.content else {
            return Err(ModelError::InvalidParam(
                "audio extraction requires a media-backed clip".into(),
            ));
        };
        if self.freeze_frame {
            return Err(ModelError::InvalidParam(
                "freeze-frame clips are silent".into(),
            ));
        }

        let mut companion = Self::from_media(*media, *source, self.timeline);
        companion.speed = self.speed;
        // Reversed audio is not decoded yet by the forward-only mixers, but
        // preserving the flag keeps the detached half semantically exact for
        // future reverse-audio support and for undo/redo.
        companion.reversed = self.reversed;
        companion.speed_curve = self.speed_curve.clone();
        companion.preserve_pitch = self.preserve_pitch;
        companion.volume = self.volume.clone();
        companion.pan = self.pan.clone();
        companion.fade_in = self.fade_in;
        companion.fade_out = self.fade_out;
        companion.denoise = self.denoise;
        companion.beats = self.beats.clone();
        companion.audio_role = Some(crate::look::AudioRole::Extracted);
        Ok(companion)
    }

    /// Derive a silent still clip from this media clip at an already-resolved
    /// native source timestamp.
    ///
    /// `timeline.start` is also the absolute timeline position whose ordinary
    /// transform/effect animation state is baked into constants. The returned
    /// clip has a fresh unlinked identity, a one-frame source window, neutral
    /// retiming/audio/template state, and no edge/combo look animations.
    pub fn frozen_frame(
        &self,
        source_time: RationalTime,
        timeline: TimeRange,
    ) -> Result<Self, ModelError> {
        let ClipSource::Media { media, source } = &self.content else {
            return Err(ModelError::InvalidParam(
                "freeze frame requires a media-backed clip".into(),
            ));
        };
        if self.freeze_frame {
            return Err(ModelError::InvalidParam(
                "clip is already a freeze frame".into(),
            ));
        }
        check_same_rate(source_time.rate, source.start.rate)?;
        check_same_rate(timeline.start.rate, self.timeline.start.rate)?;
        check_same_rate(timeline.duration.rate, timeline.start.rate)?;
        let source_end = source.end()?;
        if !source_time.rate.is_valid()
            || !timeline.start.rate.is_valid()
            || timeline.is_empty()
            || source_time.value < source.start.value
            || source_time.value >= source_end.value
        {
            return Err(ModelError::InvalidRange);
        }
        // Validate the arbitrary hold's exclusive end without unchecked
        // `TimeRange::end_tick` arithmetic.
        timeline.end()?;

        let animation_tick = self.animation_tick(timeline.start.value);
        let held_transform = self.transform.sample(animation_tick);
        let mut frozen = self.clone();
        frozen.id = ClipId::next();
        frozen.content = ClipSource::Media {
            media: *media,
            source: TimeRange::at_rate(source_time.value, 1, source_time.rate),
        };
        frozen.timeline = timeline;
        frozen.freeze_frame = true;
        frozen.link = None;

        frozen.transform.set_constant(held_transform);
        let held_crop = self.crop.sample(animation_tick);
        frozen.crop.set_constant(held_crop);
        for effect in &mut frozen.effects {
            for param in effect.params.values_mut() {
                let held = param.sample(animation_tick);
                param.set_constant(held);
            }
        }

        frozen.speed = unit_speed();
        frozen.reversed = false;
        frozen.speed_curve = default_speed_curve();
        frozen.preserve_pitch = default_preserve_pitch();
        frozen.volume = default_volume();
        frozen.pan = default_pan();
        frozen.fade_in = 0;
        frozen.fade_out = 0;
        frozen.denoise = false;
        frozen.beats.clear();
        frozen.audio_role = None;

        frozen.animation_in = None;
        frozen.animation_out = None;
        frozen.animation_combo = None;
        frozen.replaceable = None;
        frozen.text_editable = false;
        Ok(frozen)
    }

    /// A generated clip (text, shape, solid, ...).
    pub fn generated(generator: Generator, timeline: TimeRange) -> Self {
        Self {
            id: ClipId::next(),
            content: ClipSource::Generated(generator),
            timeline,
            freeze_frame: false,
            link: None,
            transform: AnimatedTransform::identity(),
            blend_mode: crate::look::BlendMode::default(),
            styles: crate::look::LayerStyles::default(),
            speed: unit_speed(),
            reversed: false,
            speed_curve: default_speed_curve(),
            preserve_pitch: default_preserve_pitch(),
            volume: default_volume(),
            pan: default_pan(),
            fade_in: 0,
            fade_out: 0,
            denoise: false,
            crop: default_crop(),
            flip_h: false,
            flip_v: false,
            effects: Vec::new(),
            beats: Vec::new(),
            mask: None,
            chroma_key: None,
            stabilize: None,
            filter: None,
            lut: None,
            adjust: crate::look::ColorAdjustments::default(),
            animation_in: None,
            animation_out: None,
            animation_combo: None,
            audio_role: None,
            replaceable: None,
            text_editable: false,
        }
    }

    /// Rebase every ordinary clip-relative animation curve by `delta` ticks.
    ///
    /// This intentionally excludes [`Self::speed_curve`]: that curve lives on
    /// the normalized [`SPEED_CURVE_SCALE`] domain and must be segmented,
    /// rather than shifted, when a media clip is split.
    pub(crate) fn shift_timeline_params(&mut self, delta: i64) -> Result<(), ModelError> {
        self.transform.shift_ticks(delta)?;
        self.volume.shift_ticks(delta)?;
        self.pan.shift_ticks(delta)?;
        self.crop.shift_ticks(delta)?;
        for effect in &mut self.effects {
            effect.shift_param_ticks(delta)?;
        }
        if let ClipSource::Generated(generator) = &mut self.content {
            generator.shift_timeline_params(delta)?;
        }
        Ok(())
    }

    /// True iff the clip's framing differs from the default (full frame,
    /// no mirroring) — drives the inspector reset state.
    pub fn has_custom_crop(&self) -> bool {
        !is_constant_full_crop(&self.crop) || self.flip_h || self.flip_v
    }

    /// True iff the clip's audio mix differs from the default (full volume,
    /// centered pan, no fades) — drives the inspector reset state and timeline
    /// badges.
    pub fn has_custom_audio(&self) -> bool {
        !is_unit_volume(&self.volume)
            || !is_zero_pan(&self.pan)
            || self.fade_in > 0
            || self.fade_out > 0
    }

    /// True iff the clip carries a keyframed volume envelope (M8), versus a
    /// flat constant gain. Drives the inspector envelope UI and the badge.
    pub fn has_volume_envelope(&self) -> bool {
        self.volume.is_animated()
    }

    /// True iff the clip carries a keyframed pan envelope, versus a flat
    /// constant balance.
    pub fn has_pan_envelope(&self) -> bool {
        self.pan.is_animated()
    }

    /// True iff the clip is inaudible: a freeze frame or a constant gain of
    /// `0` (or below). A keyframed envelope on an ordinary clip is never
    /// treated as silent — it may be non-zero elsewhere — so the mixers keep
    /// it and sample per sample-frame.
    pub fn is_silent(&self) -> bool {
        self.freeze_frame || matches!(self.volume.constant(), Some(v) if v <= 0.0)
    }

    /// True iff the clip carries detected beat markers (M8 Phase 6).
    pub fn has_beats(&self) -> bool {
        !self.beats.is_empty()
    }

    /// Absolute timeline ticks (at the clip's timeline rate) for every detected
    /// beat that falls within the clip's visible source window. Beats are stored
    /// in source time, so this maps each through the clip's geometry: the
    /// fraction of the source window a beat sits at becomes the same fraction of
    /// the timeline span (exact for constant speed — the source window maps
    /// linearly onto the span — and a close approximation for speed ramps).
    /// `reversed` mirrors the window. Out-of-window beats (left behind by a
    /// trim/split) are skipped. Pure; `O(beats)`.
    pub fn beat_timeline_ticks(&self) -> Vec<i64> {
        let Some(source) = self.source_range() else {
            return Vec::new();
        };
        let dur = source.duration.value;
        if dur <= 0 || self.beats.is_empty() {
            return Vec::new();
        }
        let first = source.start.value;
        let last = first + dur; // exclusive
        let tl_start = self.timeline.start.value;
        let tl_dur = self.timeline.duration.value;
        let mut out = Vec::with_capacity(self.beats.len());
        for &b in &self.beats {
            if b < first || b >= last {
                continue;
            }
            let mut frac = (b - first) as f64 / dur as f64;
            if self.reversed {
                frac = 1.0 - frac;
            }
            out.push(tl_start + (frac * tl_dur as f64).round() as i64);
        }
        out.sort_unstable();
        out
    }
}
