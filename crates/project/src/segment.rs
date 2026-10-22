//! One use of an asset on the sequence.

use cutlass_core::Timestamp;

use crate::Error;
use crate::Speed;
use crate::asset::{Asset, AssetKind};
use crate::id::{AssetId, SegmentId};
use crate::time::{SEQUENCE_TIMESCALE, TimeRange, timeline_duration};

/// Picture, sound, or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channels {
    Picture,
    Sound,
    Both,
}

/// Placement of a segment. Identity is the start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    position_x: f64,
    position_y: f64,
    scale_x: f64,
    scale_y: f64,
    rotation: f64,
    opacity: f64,
    flip_horizontal: bool,
    flip_vertical: bool,
}

impl Transform {
    pub const IDENTITY: Self = Self {
        position_x: 0.0,
        position_y: 0.0,
        scale_x: 1.0,
        scale_y: 1.0,
        rotation: 0.0,
        opacity: 1.0,
        flip_horizontal: false,
        flip_vertical: false,
    };

    pub fn new(
        position: (f64, f64),
        scale: (f64, f64),
        rotation: f64,
        opacity: f64,
        flip_horizontal: bool,
        flip_vertical: bool,
    ) -> Result<Self, Error> {
        if !position.0.is_finite() || !position.1.is_finite() || !rotation.is_finite() {
            return Err(Error::NonFinite);
        }
        if !scale.0.is_finite() || !scale.1.is_finite() {
            return Err(Error::NonFinite);
        }
        if scale.0 <= 0.0 || scale.1 <= 0.0 {
            return Err(Error::NonPositiveScale);
        }
        check_opacity(opacity)?;
        Ok(Self {
            position_x: position.0,
            position_y: position.1,
            scale_x: scale.0,
            scale_y: scale.1,
            rotation,
            opacity,
            flip_horizontal,
            flip_vertical,
        })
    }

    pub const fn position(self) -> (f64, f64) {
        (self.position_x, self.position_y)
    }

    pub const fn scale(self) -> (f64, f64) {
        (self.scale_x, self.scale_y)
    }

    pub const fn rotation(self) -> f64 {
        self.rotation
    }

    pub const fn opacity(self) -> f64 {
        self.opacity
    }

    pub const fn flips(self) -> (bool, bool) {
        (self.flip_horizontal, self.flip_vertical)
    }
}

/// A sampled property. The offset is from the segment's timeline start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeyframeValue {
    Position { x: f64, y: f64 },
    Scale { x: f64, y: f64 },
    Rotation(f64),
    Opacity(f64),
    Volume(f64),
}

/// Which property a sample stores. Position and scale each hold both axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyframeProperty {
    Position,
    Scale,
    Rotation,
    Opacity,
    Volume,
}

/// How one sample blends into the next sample of the same property.
///
/// Linear is the start. Ease in starts slow and ends fast. Ease out starts
/// fast and ends slow. Ease in and out is slow, fast, then slow. The last
/// sample of a property has no next sample, so its easing is unused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Easing {
    #[default]
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
}

impl Easing {
    /// The portion of the way from this sample to the next.
    ///
    /// `t` is `0` at this sample and `1` at the next. Each curve is `0` at
    /// the start and `1` at the end.
    pub fn curve(self, t: f64) -> f64 {
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => t * (2.0 - t),
            Self::EaseInOut => t * t * (3.0 - 2.0 * t),
        }
    }
}

/// One sample of a segment property.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keyframe {
    offset: Timestamp,
    value: KeyframeValue,
    easing: Easing,
}

impl Keyframe {
    /// A sample at `offset_us` microseconds from the segment start.
    ///
    /// The easing into the next sample of this property is linear.
    pub fn new(offset_us: i64, value: KeyframeValue) -> Result<Self, Error> {
        if offset_us < 0 {
            return Err(Error::NegativeTime);
        }
        check_key_value(value)?;
        Ok(Self {
            offset: Timestamp::new(offset_us, SEQUENCE_TIMESCALE),
            value,
            easing: Easing::Linear,
        })
    }

    /// Sets the easing into the next sample of this property.
    pub const fn with_easing(mut self, easing: Easing) -> Self {
        self.easing = easing;
        self
    }

    pub const fn offset(self) -> Timestamp {
        self.offset
    }

    pub const fn value(self) -> KeyframeValue {
        self.value
    }

    pub const fn easing(self) -> Easing {
        self.easing
    }

    pub const fn property(self) -> KeyframeProperty {
        match self.value {
            KeyframeValue::Position { .. } => KeyframeProperty::Position,
            KeyframeValue::Scale { .. } => KeyframeProperty::Scale,
            KeyframeValue::Rotation(_) => KeyframeProperty::Rotation,
            KeyframeValue::Opacity(_) => KeyframeProperty::Opacity,
            KeyframeValue::Volume(_) => KeyframeProperty::Volume,
        }
    }
}

/// One use of an asset.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    id: SegmentId,
    asset: AssetId,
    source: Option<TimeRange>,
    timeline: TimeRange,
    channels: Channels,
    speed: Speed,
    visible: bool,
    volume: f64,
    fade_in: i64,
    fade_out: i64,
    denoise: bool,
    transform: Transform,
    keyframes: Vec<Keyframe>,
}

impl Segment {
    pub fn media(
        asset: &Asset,
        source: TimeRange,
        timeline_start_us: i64,
        channels: Channels,
        speed: Speed,
    ) -> Result<Self, Error> {
        check_media(asset, source, channels)?;
        let timeline_span = timeline_duration(source.duration(), speed)?;
        Ok(Self::new(
            asset.id(),
            Some(source),
            TimeRange::sequence(timeline_start_us, timeline_span.value)?,
            channels,
            speed,
        ))
    }

    pub fn image(
        asset: &Asset,
        timeline_start_us: i64,
        timeline_duration_us: i64,
    ) -> Result<Self, Error> {
        if !matches!(asset.kind(), AssetKind::Image { .. }) {
            return Err(Error::MediaSource);
        }
        Ok(Self::new(
            asset.id(),
            None,
            TimeRange::sequence(timeline_start_us, timeline_duration_us)?,
            Channels::Picture,
            Speed::ONE,
        ))
    }

    fn new(
        asset: AssetId,
        source: Option<TimeRange>,
        timeline: TimeRange,
        channels: Channels,
        speed: Speed,
    ) -> Self {
        Self {
            id: SegmentId::raw(0),
            asset,
            source,
            timeline,
            channels,
            speed,
            visible: true,
            volume: 1.0,
            fade_in: 0,
            fade_out: 0,
            denoise: false,
            transform: Transform::IDENTITY,
            keyframes: Vec::new(),
        }
    }

    pub(crate) fn recheck(&self, asset: &Asset) -> Result<(), Error> {
        if asset.id() != self.asset {
            return Err(Error::UnknownAsset(self.asset));
        }
        match self.source {
            Some(source) => check_media(asset, source, self.channels),
            None => {
                if matches!(asset.kind(), AssetKind::Image { .. }) {
                    Ok(())
                } else {
                    Err(Error::MediaSource)
                }
            }
        }
    }

    pub(crate) fn trim_source(&mut self, asset: &Asset, source: TimeRange) -> Result<(), Error> {
        check_media(asset, source, self.channels)?;
        let duration = timeline_duration(source.duration(), self.speed)?;
        self.finish_duration(duration)?;
        self.source = Some(source);
        Ok(())
    }

    pub(crate) fn change_speed(&mut self, speed: Speed) -> Result<(), Error> {
        let source = self.source.ok_or(Error::ImageSource)?;
        let duration = timeline_duration(source.duration(), speed)?;
        self.finish_duration(duration)?;
        self.speed = speed;
        Ok(())
    }

    fn finish_duration(&mut self, duration: Timestamp) -> Result<(), Error> {
        if self
            .keyframes
            .iter()
            .any(|key| key.offset.value > duration.value)
        {
            return Err(Error::KeyframeOutOfRange);
        }
        self.timeline = TimeRange::new(self.timeline.start(), duration)?;
        let span = duration.value;
        if self.fade_in > span {
            self.fade_in = span;
        }
        if self.fade_out > span {
            self.fade_out = span;
        }
        Ok(())
    }

    pub(crate) fn set_start(&mut self, start: i64) -> Result<(), Error> {
        self.timeline = TimeRange::sequence(start, self.timeline.duration().value)?;
        Ok(())
    }

    /// A copy with no id, starting at `start` microseconds on the sequence.
    ///
    /// The source, channels, speed, visible flag, volume, fades, denoise,
    /// transform, and keyframes stay. Place assigns the id. A negative start is refused, and
    /// so is a start that would overflow the sequence clock.
    pub fn duplicate(&self, start: i64) -> Result<Self, Error> {
        let mut copy = self.clone();
        copy.id = SegmentId::raw(0);
        copy.set_start(start)?;
        Ok(copy)
    }

    pub(crate) fn set_timeline_duration(&mut self, duration: i64) -> Result<(), Error> {
        self.finish_duration(Timestamp::new(duration, SEQUENCE_TIMESCALE))
    }

    /// Keyframes at or before `at_offset` stay. Later ones are returned, measured
    /// from `at_offset`.
    pub(crate) fn split_keyframes(&mut self, at_offset: i64) -> Result<Vec<Keyframe>, Error> {
        let mut right = Vec::new();
        let mut left = Vec::new();
        for key in self.keyframes.drain(..) {
            if key.offset.value <= at_offset {
                left.push(key);
            } else {
                right.push(
                    Keyframe::new(key.offset.value - at_offset, key.value)?.with_easing(key.easing),
                );
            }
        }
        self.keyframes = left;
        Ok(right)
    }

    pub(crate) fn replace_keyframes(&mut self, keyframes: Vec<Keyframe>) {
        self.keyframes = keyframes;
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn set_volume(&mut self, volume: f64) -> Result<(), Error> {
        check_volume(volume)?;
        self.volume = volume;
        Ok(())
    }

    /// Sets fade in and fade out, in microseconds from the segment edges.
    ///
    /// `0` is no fade. A negative duration, or one longer than the segment,
    /// changes nothing.
    pub(crate) fn set_fades(&mut self, fade_in: i64, fade_out: i64) -> Result<(), Error> {
        if fade_in < 0 || fade_out < 0 {
            return Err(Error::NegativeTime);
        }
        let duration = self.timeline.duration().value;
        if fade_in > duration || fade_out > duration {
            return Err(Error::FadeOutOfRange);
        }
        self.fade_in = fade_in;
        self.fade_out = fade_out;
        Ok(())
    }

    /// Turns noise reduction on or off.
    ///
    /// A segment with no sound cannot turn it on.
    pub(crate) fn set_denoise(&mut self, on: bool) -> Result<(), Error> {
        if on && self.channels == Channels::Picture {
            return Err(Error::NoSound);
        }
        self.denoise = on;
        Ok(())
    }

    pub fn set_transform(&mut self, transform: Transform) {
        self.transform = transform;
    }

    pub fn push_keyframe(&mut self, keyframe: Keyframe) -> Result<(), Error> {
        if keyframe.offset.value > self.timeline.duration().value {
            return Err(Error::KeyframeOutOfRange);
        }
        if self.keyframes.iter().any(|existing| {
            existing.property() == keyframe.property() && existing.offset == keyframe.offset
        }) {
            return Err(Error::DuplicateKeyframe);
        }
        self.keyframes.push(keyframe);
        Ok(())
    }

    /// Removes the sample of `property` at `offset_us` microseconds from the
    /// segment start. A missing sample changes nothing.
    pub fn remove_keyframe(
        &mut self,
        offset_us: i64,
        property: KeyframeProperty,
    ) -> Result<(), Error> {
        if offset_us < 0 {
            return Err(Error::NegativeTime);
        }
        let Some(index) = self
            .keyframes
            .iter()
            .position(|key| key.property() == property && key.offset.value == offset_us)
        else {
            return Err(Error::MissingKeyframe);
        };
        self.keyframes.remove(index);
        Ok(())
    }

    /// Sets the easing on the sample of `property` at `offset_us`.
    ///
    /// A missing sample changes nothing.
    pub(crate) fn set_easing(
        &mut self,
        offset_us: i64,
        property: KeyframeProperty,
        easing: Easing,
    ) -> Result<(), Error> {
        if offset_us < 0 {
            return Err(Error::NegativeTime);
        }
        let Some(key) = self
            .keyframes
            .iter_mut()
            .find(|key| key.property() == property && key.offset.value == offset_us)
        else {
            return Err(Error::MissingKeyframe);
        };
        key.easing = easing;
        Ok(())
    }

    pub(crate) fn assign(&mut self, id: SegmentId) {
        self.id = id;
    }

    pub const fn id(&self) -> SegmentId {
        self.id
    }

    pub const fn asset_id(&self) -> AssetId {
        self.asset
    }

    pub const fn source(&self) -> Option<TimeRange> {
        self.source
    }

    pub const fn timeline(&self) -> TimeRange {
        self.timeline
    }

    pub const fn channels(&self) -> Channels {
        self.channels
    }

    pub const fn speed(&self) -> Speed {
        self.speed
    }

    pub const fn visible(&self) -> bool {
        self.visible
    }

    pub const fn volume(&self) -> f64 {
        self.volume
    }

    pub const fn fade_in(&self) -> i64 {
        self.fade_in
    }

    pub const fn fade_out(&self) -> i64 {
        self.fade_out
    }

    pub const fn denoise(&self) -> bool {
        self.denoise
    }

    pub const fn transform(&self) -> Transform {
        self.transform
    }

    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }

    /// The value of `property` at `offset_us` microseconds from the segment start.
    ///
    /// With no sample of that property, the result is the segment's own value.
    /// The first sample holds back to the start and the last holds through the
    /// end. Between two samples, the earlier easing blends them. At a sample,
    /// the result is that sample. A negative offset, or one past the segment,
    /// returns no value.
    pub fn sample(
        &self,
        property: KeyframeProperty,
        offset_us: i64,
    ) -> Result<KeyframeValue, Error> {
        if offset_us < 0 {
            return Err(Error::NegativeTime);
        }
        if offset_us > self.timeline.duration().value {
            return Err(Error::KeyframeOutOfRange);
        }
        let mut keys: Vec<Keyframe> = self
            .keyframes
            .iter()
            .copied()
            .filter(|key| key.property() == property)
            .collect();
        if keys.is_empty() {
            return Ok(own_value(self, property));
        }
        keys.sort_by_key(|key| key.offset().value);
        let next = keys.partition_point(|key| key.offset().value <= offset_us);
        if next == 0 {
            return Ok(keys[0].value());
        }
        let earlier = keys[next - 1];
        if earlier.offset().value == offset_us || next == keys.len() {
            return Ok(earlier.value());
        }
        let later = keys[next];
        let span = later.offset().value - earlier.offset().value;
        if span <= 0 {
            return Ok(earlier.value());
        }
        let portion = (offset_us - earlier.offset().value) as f64 / span as f64;
        Ok(blend(
            earlier.value(),
            later.value(),
            earlier.easing().curve(portion),
        ))
    }

    /// The linear gain at `offset_us`, including the fades.
    ///
    /// The sampled volume is multiplied by the fade-in ramp and the fade-out
    /// ramp. A fade of `0` leaves that edge at full gain. A negative offset,
    /// or one past the segment, returns no value.
    pub fn gain(&self, offset_us: i64) -> Result<f64, Error> {
        let KeyframeValue::Volume(level) = self.sample(KeyframeProperty::Volume, offset_us)? else {
            return Ok(self.volume);
        };
        Ok(level
            * fade_ramps(
                offset_us,
                self.timeline.duration().value,
                self.fade_in,
                self.fade_out,
            ))
    }
}

fn check_media(asset: &Asset, source: TimeRange, channels: Channels) -> Result<(), Error> {
    match asset.kind() {
        AssetKind::Image { .. } => return Err(Error::ImageSource),
        AssetKind::Audio => {
            if channels != Channels::Sound {
                return Err(Error::Channels);
            }
        }
        AssetKind::Video { has_audio, .. } => {
            if !has_audio && channels != Channels::Picture {
                return Err(Error::Channels);
            }
        }
    }
    let Some(duration) = asset.duration() else {
        return Err(Error::ImageSource);
    };
    if source.start().timescale != duration.timescale {
        return Err(Error::SourceTimescale {
            asset: duration.timescale.get(),
            source_scale: source.start().timescale.get(),
        });
    }
    if source.start().value < 0 || source.end().value > duration.value {
        return Err(Error::SourceOutsideAsset);
    }
    Ok(())
}

fn check_opacity(opacity: f64) -> Result<(), Error> {
    if !opacity.is_finite() {
        return Err(Error::NonFinite);
    }
    if !(0.0..=1.0).contains(&opacity) {
        return Err(Error::Opacity);
    }
    Ok(())
}

fn check_volume(volume: f64) -> Result<(), Error> {
    if !volume.is_finite() {
        return Err(Error::NonFinite);
    }
    if volume < 0.0 {
        return Err(Error::NegativeVolume);
    }
    Ok(())
}

fn check_key_value(value: KeyframeValue) -> Result<(), Error> {
    match value {
        KeyframeValue::Position { x, y } => finite(x, y),
        KeyframeValue::Rotation(degrees) => {
            if degrees.is_finite() {
                Ok(())
            } else {
                Err(Error::NonFinite)
            }
        }
        KeyframeValue::Scale { x, y } => {
            finite(x, y)?;
            if x <= 0.0 || y <= 0.0 {
                return Err(Error::NonPositiveScale);
            }
            Ok(())
        }
        KeyframeValue::Opacity(opacity) => check_opacity(opacity),
        KeyframeValue::Volume(volume) => check_volume(volume),
    }
}

fn finite(x: f64, y: f64) -> Result<(), Error> {
    if x.is_finite() && y.is_finite() {
        Ok(())
    } else {
        Err(Error::NonFinite)
    }
}

fn own_value(segment: &Segment, property: KeyframeProperty) -> KeyframeValue {
    match property {
        KeyframeProperty::Position => {
            let (x, y) = segment.transform.position();
            KeyframeValue::Position { x, y }
        }
        KeyframeProperty::Scale => {
            let (x, y) = segment.transform.scale();
            KeyframeValue::Scale { x, y }
        }
        KeyframeProperty::Rotation => KeyframeValue::Rotation(segment.transform.rotation()),
        KeyframeProperty::Opacity => KeyframeValue::Opacity(segment.transform.opacity()),
        KeyframeProperty::Volume => KeyframeValue::Volume(segment.volume),
    }
}

fn blend(from: KeyframeValue, to: KeyframeValue, portion: f64) -> KeyframeValue {
    match (from, to) {
        (
            KeyframeValue::Position {
                x: from_x,
                y: from_y,
            },
            KeyframeValue::Position { x: to_x, y: to_y },
        ) => KeyframeValue::Position {
            x: lerp(from_x, to_x, portion),
            y: lerp(from_y, to_y, portion),
        },
        (
            KeyframeValue::Scale {
                x: from_x,
                y: from_y,
            },
            KeyframeValue::Scale { x: to_x, y: to_y },
        ) => KeyframeValue::Scale {
            x: lerp(from_x, to_x, portion),
            y: lerp(from_y, to_y, portion),
        },
        (KeyframeValue::Rotation(from), KeyframeValue::Rotation(to)) => {
            KeyframeValue::Rotation(lerp(from, to, portion))
        }
        (KeyframeValue::Opacity(from), KeyframeValue::Opacity(to)) => {
            KeyframeValue::Opacity(lerp(from, to, portion))
        }
        (KeyframeValue::Volume(from), KeyframeValue::Volume(to)) => {
            KeyframeValue::Volume(lerp(from, to, portion))
        }
        (from, _) => from,
    }
}

fn lerp(from: f64, to: f64, portion: f64) -> f64 {
    from + (to - from) * portion
}

fn fade_ramps(offset: i64, duration: i64, fade_in: i64, fade_out: i64) -> f64 {
    let into = if fade_in <= 0 || offset >= fade_in {
        1.0
    } else {
        offset as f64 / fade_in as f64
    };
    let out_start = duration.saturating_sub(fade_out);
    let out = if fade_out <= 0 || offset <= out_start {
        1.0
    } else {
        (duration - offset) as f64 / fade_out as f64
    };
    into * out
}

/// How a split divides the two fades. The left pair is first.
pub(crate) fn parted_fades(
    fade_in: i64,
    fade_out: i64,
    left: i64,
    right: i64,
) -> (i64, i64, i64, i64) {
    (
        fade_in.min(left),
        crossed(fade_out, right).min(left),
        crossed(fade_in, left).min(right),
        fade_out.min(right),
    )
}

/// The part of a fade that continues past `edge`. None, when the fade ends first.
fn crossed(fade: i64, edge: i64) -> i64 {
    if fade > edge { fade - edge } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::scale;
    use cutlass_core::Timestamp;

    fn video(has_audio: bool, ticks: i64) -> Asset {
        Asset::video(
            "file:///v.mp4",
            Timestamp::new(ticks, scale(15_360)),
            320,
            180,
            has_audio,
        )
        .unwrap()
    }

    fn source(start: i64, duration: i64) -> TimeRange {
        TimeRange::new(
            Timestamp::new(start, scale(15_360)),
            Timestamp::new(duration, scale(15_360)),
        )
        .unwrap()
    }

    #[test]
    fn the_source_range_keeps_the_asset_timescale() {
        let asset = video(true, 15_360);
        let segment =
            Segment::media(&asset, source(1_536, 7_680), 0, Channels::Both, Speed::ONE).unwrap();
        let source = segment.source().unwrap();
        assert_eq!(source.start().timescale.get(), 15_360);
        assert_eq!(source.start().value, 1_536);
        assert_eq!(source.duration().value, 7_680);
        assert_eq!(segment.timeline().start().timescale.get(), 1_000_000);
        assert_eq!(segment.timeline().duration().value, 500_000);
        assert_eq!(segment.channels(), Channels::Both);
        assert_eq!(segment.speed(), Speed::ONE);
        assert!(segment.visible());
        assert_eq!(segment.volume(), 1.0);
        assert_eq!(segment.transform(), Transform::IDENTITY);
    }

    #[test]
    fn double_speed_halves_the_timeline_span() {
        let asset = video(true, 15_360);
        let segment = Segment::media(
            &asset,
            source(0, 15_360),
            2_000,
            Channels::Picture,
            Speed::new(2, 1).unwrap(),
        )
        .unwrap();
        assert_eq!(segment.timeline().start().value, 2_000);
        assert_eq!(segment.timeline().duration().value, 500_000);
    }

    #[test]
    fn an_image_has_no_source_and_keeps_its_timeline_duration() {
        let asset = Asset::image("file:///s.png", 10, 10).unwrap();
        let segment = Segment::image(&asset, 0, 5_000_000).unwrap();
        assert!(segment.source().is_none());
        assert_eq!(segment.timeline().duration().value, 5_000_000);
        assert_eq!(segment.channels(), Channels::Picture);
        assert_eq!(segment.speed(), Speed::ONE);
    }

    #[test]
    fn media_and_image_constructors_reject_the_other_kind() {
        let video = video(true, 15_360);
        let image = Asset::image("file:///s.png", 10, 10).unwrap();
        assert_eq!(
            Segment::image(&video, 0, 1).unwrap_err(),
            Error::MediaSource
        );
        assert_eq!(
            Segment::media(&image, source(0, 1), 0, Channels::Picture, Speed::ONE).unwrap_err(),
            Error::ImageSource
        );
    }

    #[test]
    fn channels_follow_the_asset() {
        let with_audio = video(true, 15_360);
        let silent = video(false, 15_360);
        let audio = Asset::audio("file:///a.wav", Timestamp::new(15_360, scale(15_360))).unwrap();
        assert!(
            Segment::media(
                &with_audio,
                source(0, 15_360),
                0,
                Channels::Sound,
                Speed::ONE
            )
            .is_ok()
        );
        assert_eq!(
            Segment::media(&silent, source(0, 15_360), 0, Channels::Both, Speed::ONE).unwrap_err(),
            Error::Channels
        );
        assert_eq!(
            Segment::media(&audio, source(0, 15_360), 0, Channels::Picture, Speed::ONE)
                .unwrap_err(),
            Error::Channels
        );
        assert!(Segment::media(&audio, source(0, 15_360), 0, Channels::Sound, Speed::ONE).is_ok());
    }

    #[test]
    fn a_source_outside_the_asset_is_rejected() {
        let asset = video(true, 15_360);
        let error =
            Segment::media(&asset, source(15_360, 1), 0, Channels::Both, Speed::ONE).unwrap_err();
        assert_eq!(error, Error::SourceOutsideAsset);
        let negative = TimeRange::new(
            Timestamp::new(-1, scale(15_360)),
            Timestamp::new(2, scale(15_360)),
        )
        .unwrap();
        assert_eq!(
            Segment::media(&asset, negative, 0, Channels::Both, Speed::ONE).unwrap_err(),
            Error::SourceOutsideAsset
        );
    }

    #[test]
    fn a_source_on_another_timescale_is_rejected() {
        let asset = video(true, 15_360);
        let other = TimeRange::new(
            Timestamp::new(0, scale(1_000)),
            Timestamp::new(500, scale(1_000)),
        )
        .unwrap();
        assert_eq!(
            Segment::media(&asset, other, 0, Channels::Both, Speed::ONE).unwrap_err(),
            Error::SourceTimescale {
                asset: 15_360,
                source_scale: 1_000
            }
        );
    }

    #[test]
    fn a_source_that_rounds_to_zero_microseconds_is_rejected() {
        let asset = Asset::video(
            "file:///v.mp4",
            Timestamp::new(10, scale(3_000_000)),
            16,
            16,
            true,
        )
        .unwrap();
        let range = TimeRange::new(
            Timestamp::new(0, scale(3_000_000)),
            Timestamp::new(1, scale(3_000_000)),
        )
        .unwrap();
        assert_eq!(
            Segment::media(&asset, range, 0, Channels::Both, Speed::ONE).unwrap_err(),
            Error::RoundsToZero
        );
    }

    #[test]
    fn volume_transform_and_keyframes_keep_their_bounds() {
        let asset = video(true, 15_360);
        let mut segment =
            Segment::media(&asset, source(0, 15_360), 0, Channels::Both, Speed::ONE).unwrap();
        segment.set_visible(false);
        segment.set_volume(0.0).unwrap();
        assert!(!segment.visible());
        assert_eq!(segment.volume(), 0.0);
        assert_eq!(segment.set_volume(-0.1).unwrap_err(), Error::NegativeVolume);
        assert_eq!(segment.set_volume(f64::NAN).unwrap_err(), Error::NonFinite);
        assert_eq!(segment.volume(), 0.0);

        let moved = Transform::new((10.0, -4.0), (2.0, 2.0), 90.0, 0.5, true, false).unwrap();
        segment.set_transform(moved);
        assert_eq!(segment.transform().position(), (10.0, -4.0));
        assert_eq!(segment.transform().flips(), (true, false));
        assert_eq!(
            Transform::new((0.0, 0.0), (0.0, 1.0), 0.0, 1.0, false, false).unwrap_err(),
            Error::NonPositiveScale
        );
        assert_eq!(
            Transform::new((0.0, 0.0), (-1.0, 1.0), 0.0, 1.0, false, false).unwrap_err(),
            Error::NonPositiveScale
        );
        assert_eq!(
            Transform::new((f64::INFINITY, 0.0), (1.0, 1.0), 0.0, 1.0, false, false).unwrap_err(),
            Error::NonFinite
        );
        assert_eq!(
            Transform::new((0.0, 0.0), (1.0, 1.0), 0.0, 1.1, false, false).unwrap_err(),
            Error::Opacity
        );

        let end = segment.timeline().duration().value;
        segment
            .push_keyframe(Keyframe::new(end, KeyframeValue::Opacity(0.0)).unwrap())
            .unwrap();
        segment
            .push_keyframe(Keyframe::new(end, KeyframeValue::Volume(0.0)).unwrap())
            .unwrap();
        assert_eq!(
            segment
                .push_keyframe(Keyframe::new(end, KeyframeValue::Opacity(1.0)).unwrap())
                .unwrap_err(),
            Error::DuplicateKeyframe
        );
        assert_eq!(
            segment
                .push_keyframe(Keyframe::new(end + 1, KeyframeValue::Rotation(0.0)).unwrap())
                .unwrap_err(),
            Error::KeyframeOutOfRange
        );
        assert_eq!(segment.keyframes().len(), 2);
        assert_eq!(segment.keyframes()[0].offset().timescale.get(), 1_000_000);
    }

    #[test]
    fn a_keyframe_offset_cannot_be_negative() {
        assert_eq!(
            Keyframe::new(-1, KeyframeValue::Rotation(0.0)).unwrap_err(),
            Error::NegativeTime
        );
    }

    #[test]
    fn duplicate_keeps_the_clip_and_clears_its_id() {
        let asset = video(true, 15_360);
        let mut segment = Segment::media(
            &asset,
            source(0, 15_360),
            2_000,
            Channels::Both,
            Speed::new(2, 1).unwrap(),
        )
        .unwrap();
        segment.set_visible(false);
        segment.set_volume(0.25).unwrap();
        segment.set_fades(100_000, 50_000).unwrap();
        segment.set_denoise(true).unwrap();
        let transform = Transform::new((0.1, -0.2), (2.0, 3.0), 15.0, 0.4, true, false).unwrap();
        segment.set_transform(transform);
        segment
            .push_keyframe(
                Keyframe::new(100_000, KeyframeValue::Opacity(0.2))
                    .unwrap()
                    .with_easing(Easing::EaseIn),
            )
            .unwrap();
        segment.assign(SegmentId::raw(4));
        let copy = segment.duplicate(8_000).unwrap();
        assert_eq!(copy.id(), SegmentId::raw(0));
        assert_eq!(segment.id(), SegmentId::raw(4));
        assert_eq!(copy.timeline().start().value, 8_000);
        assert_eq!(segment.timeline().start().value, 2_000);
        assert_eq!(copy.timeline().duration(), segment.timeline().duration());
        assert_eq!(copy.source(), segment.source());
        assert_eq!(copy.channels(), Channels::Both);
        assert_eq!(copy.speed(), Speed::new(2, 1).unwrap());
        assert!(!copy.visible());
        assert_eq!(copy.volume(), 0.25);
        assert_eq!(copy.fade_in(), 100_000);
        assert_eq!(copy.fade_out(), 50_000);
        assert!(copy.denoise());
        assert_eq!(copy.transform(), transform);
        assert_eq!(copy.keyframes(), segment.keyframes());
        assert_eq!(copy.asset_id(), segment.asset_id());
    }

    #[test]
    fn duplicate_refuses_a_negative_start() {
        let asset = video(true, 15_360);
        let segment =
            Segment::media(&asset, source(0, 15_360), 0, Channels::Picture, Speed::ONE).unwrap();
        assert_eq!(segment.duplicate(-1).unwrap_err(), Error::NegativeTime);
        assert_eq!(segment.timeline().start().value, 0);
    }

    #[test]
    fn duplicate_refuses_a_start_past_the_end_of_the_clock() {
        let asset = video(true, 15_360);
        let segment =
            Segment::media(&asset, source(0, 15_360), 0, Channels::Picture, Speed::ONE).unwrap();
        assert_eq!(
            segment.duplicate(i64::MAX).unwrap_err(),
            Error::DurationOverflow
        );
    }

    fn one_second() -> Segment {
        let asset = video(true, 15_360);
        Segment::media(&asset, source(0, 15_360), 0, Channels::Both, Speed::ONE).unwrap()
    }

    #[test]
    fn each_easing_starts_at_zero_and_ends_at_one() {
        for easing in [
            Easing::Linear,
            Easing::EaseIn,
            Easing::EaseOut,
            Easing::EaseInOut,
        ] {
            assert_eq!(easing.curve(0.0), 0.0);
            assert_eq!(easing.curve(1.0), 1.0);
        }
    }

    #[test]
    fn no_samples_return_the_segments_own_value() {
        let mut segment = one_second();
        let transform = Transform::new((0.25, -0.5), (2.0, 3.0), 90.0, 0.4, false, false).unwrap();
        segment.set_transform(transform);
        segment.set_volume(0.25).unwrap();
        assert_eq!(
            segment.sample(KeyframeProperty::Position, 0).unwrap(),
            KeyframeValue::Position { x: 0.25, y: -0.5 }
        );
        assert_eq!(
            segment.sample(KeyframeProperty::Scale, 500_000).unwrap(),
            KeyframeValue::Scale { x: 2.0, y: 3.0 }
        );
        assert_eq!(
            segment
                .sample(KeyframeProperty::Rotation, 1_000_000)
                .unwrap(),
            KeyframeValue::Rotation(90.0)
        );
        assert_eq!(
            segment.sample(KeyframeProperty::Opacity, 0).unwrap(),
            KeyframeValue::Opacity(0.4)
        );
        assert_eq!(
            segment.sample(KeyframeProperty::Volume, 1_000_000).unwrap(),
            KeyframeValue::Volume(0.25)
        );
    }

    #[test]
    fn one_sample_holds_and_two_blend_by_the_earlier_easing() {
        let mut segment = one_second();
        segment
            .push_keyframe(Keyframe::new(400_000, KeyframeValue::Opacity(0.25)).unwrap())
            .unwrap();
        assert_eq!(
            segment.sample(KeyframeProperty::Opacity, 0).unwrap(),
            KeyframeValue::Opacity(0.25)
        );
        assert_eq!(
            segment
                .sample(KeyframeProperty::Opacity, 1_000_000)
                .unwrap(),
            KeyframeValue::Opacity(0.25)
        );
        assert_eq!(
            segment.sample(KeyframeProperty::Volume, 500_000).unwrap(),
            KeyframeValue::Volume(1.0)
        );

        let mut segment = one_second();
        segment
            .push_keyframe(
                Keyframe::new(1_000_000, KeyframeValue::Opacity(1.0))
                    .unwrap()
                    .with_easing(Easing::EaseOut),
            )
            .unwrap();
        segment
            .push_keyframe(
                Keyframe::new(0, KeyframeValue::Opacity(0.0))
                    .unwrap()
                    .with_easing(Easing::EaseIn),
            )
            .unwrap();
        assert_eq!(
            segment.sample(KeyframeProperty::Opacity, 0).unwrap(),
            KeyframeValue::Opacity(0.0)
        );
        assert_eq!(
            segment
                .sample(KeyframeProperty::Opacity, 1_000_000)
                .unwrap(),
            KeyframeValue::Opacity(1.0)
        );
        assert_eq!(
            segment.sample(KeyframeProperty::Opacity, 500_000).unwrap(),
            KeyframeValue::Opacity(0.25)
        );
        segment
            .set_easing(0, KeyframeProperty::Opacity, Easing::EaseOut)
            .unwrap();
        assert_eq!(
            segment.sample(KeyframeProperty::Opacity, 500_000).unwrap(),
            KeyframeValue::Opacity(0.75)
        );
        segment
            .set_easing(0, KeyframeProperty::Opacity, Easing::Linear)
            .unwrap();
        assert_eq!(
            segment.sample(KeyframeProperty::Opacity, 250_000).unwrap(),
            KeyframeValue::Opacity(0.25)
        );
        segment
            .set_easing(0, KeyframeProperty::Opacity, Easing::EaseIn)
            .unwrap();
        assert_eq!(
            segment.sample(KeyframeProperty::Opacity, 250_000).unwrap(),
            KeyframeValue::Opacity(0.0625)
        );
        segment
            .set_easing(0, KeyframeProperty::Opacity, Easing::EaseOut)
            .unwrap();
        assert_eq!(
            segment.sample(KeyframeProperty::Opacity, 250_000).unwrap(),
            KeyframeValue::Opacity(0.4375)
        );
        segment
            .set_easing(0, KeyframeProperty::Opacity, Easing::EaseInOut)
            .unwrap();
        assert_eq!(
            segment.sample(KeyframeProperty::Opacity, 250_000).unwrap(),
            KeyframeValue::Opacity(0.15625)
        );

        segment
            .push_keyframe(Keyframe::new(0, KeyframeValue::Position { x: 0.0, y: 0.0 }).unwrap())
            .unwrap();
        segment
            .push_keyframe(
                Keyframe::new(1_000_000, KeyframeValue::Position { x: 1.0, y: -2.0 }).unwrap(),
            )
            .unwrap();
        assert_eq!(
            segment.sample(KeyframeProperty::Position, 500_000).unwrap(),
            KeyframeValue::Position { x: 0.5, y: -1.0 }
        );
        segment
            .push_keyframe(Keyframe::new(0, KeyframeValue::Scale { x: 1.0, y: 1.0 }).unwrap())
            .unwrap();
        segment
            .push_keyframe(
                Keyframe::new(1_000_000, KeyframeValue::Scale { x: 3.0, y: 1.0 }).unwrap(),
            )
            .unwrap();
        assert_eq!(
            segment.sample(KeyframeProperty::Scale, 500_000).unwrap(),
            KeyframeValue::Scale { x: 2.0, y: 1.0 }
        );
        segment
            .push_keyframe(Keyframe::new(0, KeyframeValue::Rotation(0.0)).unwrap())
            .unwrap();
        segment
            .push_keyframe(Keyframe::new(1_000_000, KeyframeValue::Rotation(360.0)).unwrap())
            .unwrap();
        assert_eq!(
            segment.sample(KeyframeProperty::Rotation, 500_000).unwrap(),
            KeyframeValue::Rotation(180.0)
        );
        segment
            .push_keyframe(Keyframe::new(0, KeyframeValue::Volume(0.0)).unwrap())
            .unwrap();
        segment
            .push_keyframe(Keyframe::new(1_000_000, KeyframeValue::Volume(1.0)).unwrap())
            .unwrap();
        assert_eq!(
            segment.sample(KeyframeProperty::Volume, 500_000).unwrap(),
            KeyframeValue::Volume(0.5)
        );
    }

    #[test]
    fn a_sample_outside_the_segment_is_refused() {
        let segment = one_second();
        assert_eq!(
            segment.sample(KeyframeProperty::Opacity, -1).unwrap_err(),
            Error::NegativeTime
        );
        assert_eq!(
            segment
                .sample(KeyframeProperty::Opacity, 1_000_001)
                .unwrap_err(),
            Error::KeyframeOutOfRange
        );
        assert_eq!(
            segment
                .sample(KeyframeProperty::Opacity, 1_000_000)
                .unwrap(),
            KeyframeValue::Opacity(1.0)
        );
    }

    #[test]
    fn fades_ramp_the_gain_and_a_shorter_segment_clamps_them() {
        let mut segment = one_second();
        assert_eq!(segment.gain(0).unwrap(), 1.0);
        assert_eq!(segment.gain(1_000_000).unwrap(), 1.0);
        segment.set_fades(500_000, 500_000).unwrap();
        assert_eq!(segment.gain(0).unwrap(), 0.0);
        assert_eq!(segment.gain(250_000).unwrap(), 0.5);
        assert_eq!(segment.gain(500_000).unwrap(), 1.0);
        assert_eq!(segment.gain(750_000).unwrap(), 0.5);
        assert_eq!(segment.gain(1_000_000).unwrap(), 0.0);
        segment.set_fades(1_000_000, 1_000_000).unwrap();
        assert_eq!(segment.gain(500_000).unwrap(), 0.25);
        assert_eq!(
            segment.set_fades(1_000_001, 0).unwrap_err(),
            Error::FadeOutOfRange
        );
        assert_eq!(segment.set_fades(-1, 0).unwrap_err(), Error::NegativeTime);
        assert_eq!(segment.fade_in(), 1_000_000);
        segment.set_volume(0.5).unwrap();
        segment.set_fades(500_000, 0).unwrap();
        assert_eq!(segment.gain(250_000).unwrap(), 0.25);
        segment
            .push_keyframe(Keyframe::new(0, KeyframeValue::Volume(0.0)).unwrap())
            .unwrap();
        segment
            .push_keyframe(Keyframe::new(1_000_000, KeyframeValue::Volume(1.0)).unwrap())
            .unwrap();
        segment.set_fades(1_000_000, 0).unwrap();
        assert_eq!(segment.gain(500_000).unwrap(), 0.25);
        let mut shorter = one_second();
        shorter.set_fades(800_000, 700_000).unwrap();
        shorter.set_timeline_duration(400_000).unwrap();
        assert_eq!(shorter.fade_in(), 400_000);
        assert_eq!(shorter.fade_out(), 400_000);
        assert_eq!(
            parted_fades(600_000, 100_000, 400_000, 600_000),
            (400_000, 0, 200_000, 100_000)
        );
        assert_eq!(
            parted_fades(200_000, 300_000, 500_000, 500_000),
            (200_000, 0, 0, 300_000)
        );
    }

    #[test]
    fn denoise_needs_sound() {
        let mut sound = one_second();
        assert!(!sound.denoise());
        sound.set_denoise(true).unwrap();
        assert!(sound.denoise());
        sound.set_denoise(false).unwrap();
        assert!(!sound.denoise());

        let asset = video(false, 15_360);
        let mut picture =
            Segment::media(&asset, source(0, 15_360), 0, Channels::Picture, Speed::ONE).unwrap();
        assert_eq!(picture.set_denoise(true).unwrap_err(), Error::NoSound);
        assert!(!picture.denoise());
        picture.set_denoise(false).unwrap();
        assert!(!picture.denoise());
    }
}
