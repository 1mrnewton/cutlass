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

/// One sample of a segment property.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keyframe {
    offset: Timestamp,
    value: KeyframeValue,
}

impl Keyframe {
    pub fn new(offset_us: i64, value: KeyframeValue) -> Result<Self, Error> {
        if offset_us < 0 {
            return Err(Error::NegativeTime);
        }
        check_key_value(value)?;
        Ok(Self {
            offset: Timestamp::new(offset_us, SEQUENCE_TIMESCALE),
            value,
        })
    }

    pub const fn offset(self) -> Timestamp {
        self.offset
    }

    pub const fn value(self) -> KeyframeValue {
        self.value
    }

    fn property(self) -> u8 {
        match self.value {
            KeyframeValue::Position { .. } => 0,
            KeyframeValue::Scale { .. } => 1,
            KeyframeValue::Rotation(_) => 2,
            KeyframeValue::Opacity(_) => 3,
            KeyframeValue::Volume(_) => 4,
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
        Ok(())
    }

    pub(crate) fn set_start(&mut self, start: i64) -> Result<(), Error> {
        self.timeline = TimeRange::sequence(start, self.timeline.duration().value)?;
        Ok(())
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
                right.push(Keyframe::new(key.offset.value - at_offset, key.value)?);
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

    pub const fn transform(&self) -> Transform {
        self.transform
    }

    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
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
}
