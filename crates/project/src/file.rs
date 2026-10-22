//! Version 1 of the project file.
//!
//! The text is JSON. This module does not open a path. `docs/project.md` is
//! the format.

use std::num::NonZeroU32;

use cutlass_core::Timestamp;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Error;
use crate::asset::{Asset, AssetKind};
use crate::id::{AssetId, SegmentId, TrackId};
use crate::project::Project;
use crate::rate::{FrameRate, SequenceRate};
use crate::segment::{Channels, Crop, Easing, Keyframe, KeyframeValue, Segment, Transform};
use crate::settings::{AspectRatio, Canvas, Color, ColorSpace, Settings};
use crate::time::{Speed, TimeRange};
use crate::timeline::Timeline;
use crate::track::{Track, TrackKind};

/// The only project-file version this crate writes.
///
/// [`Project::from_json`] refuses any other version and loads nothing.
pub const FILE_VERSION: u32 = 1;

pub(crate) fn to_json(project: &Project) -> Result<String, Error> {
    let mut text = serde_json::to_string_pretty(&document_from(project)).map_err(unreadable)?;
    text.push('\n');
    Ok(text)
}

pub(crate) fn from_json(text: &str) -> Result<Project, Error> {
    let value: Value = serde_json::from_str(text).map_err(unreadable)?;
    // The version is decided before the body is read, so a newer file cannot
    // load part of itself.
    let found = version_of(&value)?;
    if found != u64::from(FILE_VERSION) {
        return Err(Error::FileVersion { found });
    }
    let document: Document = serde_json::from_value(value).map_err(unreadable)?;
    project_from(document)
}

fn unreadable(error: impl ToString) -> Error {
    Error::Unreadable(error.to_string())
}

fn version_of(value: &Value) -> Result<u64, Error> {
    let Some(Value::Number(number)) = value.get("version") else {
        return Err(Error::Unreadable(
            "version is missing or not a number".into(),
        ));
    };
    let Some(found) = number.as_u64() else {
        return Err(Error::Unreadable("version is not a whole number".into()));
    };
    if number.as_f64().is_some_and(|float| float.fract() != 0.0) {
        return Err(Error::Unreadable("version is not a whole number".into()));
    }
    Ok(found)
}

fn document_from(project: &Project) -> Document {
    Document {
        version: FILE_VERSION,
        id: project.id().to_string(),
        next_id: project.next_id(),
        name: project.name().to_string(),
        settings: settings_doc(project.settings()),
        assets: project.assets().iter().map(asset_doc).collect(),
        timeline: TimelineDoc {
            id: project.timeline().id().to_string(),
            tracks: project.timeline().tracks().iter().map(track_doc).collect(),
        },
    }
}

fn project_from(document: Document) -> Result<Project, Error> {
    if document.version != FILE_VERSION {
        return Err(Error::FileVersion {
            found: u64::from(document.version),
        });
    }
    let id = document.id.parse()?;
    let settings = settings_from(&document.settings)?;
    let assets = assets_from(document.assets)?;
    let timeline = timeline_from(&document.timeline, &assets)?;
    Project::restore(
        id,
        document.next_id,
        document.name,
        settings,
        assets,
        timeline,
    )
}

fn settings_doc(settings: &Settings) -> SettingsDoc {
    let (red, green, blue) = settings.background().channels();
    SettingsDoc {
        width: settings.width(),
        height: settings.height(),
        ratio: ratio_name(settings.ratio()),
        background: [red, green, blue],
        rate: RateDoc {
            numerator: settings.rate().rate().numerator(),
            denominator: settings.rate().rate().denominator(),
        },
        drop_frame: settings.rate().drop_frame(),
        color_space: space_name(settings.color_space()),
        magnet: settings.magnet(),
        linkage: settings.linkage(),
    }
}

fn settings_from(document: &SettingsDoc) -> Result<Settings, Error> {
    let canvas = Canvas::new(document.width, document.height, ratio_from(document.ratio))?;
    let [red, green, blue] = document.background;
    let rate = FrameRate::from_parts(document.rate.numerator, document.rate.denominator)?;
    Ok(Settings::new(
        canvas,
        Color::new(red, green, blue),
        SequenceRate::new(rate, document.drop_frame)?,
        space_from(document.color_space),
        document.magnet,
        document.linkage,
    ))
}

fn asset_doc(asset: &Asset) -> AssetDoc {
    let id = asset.id().to_string();
    let source = asset.source().to_string();
    match asset.kind() {
        AssetKind::Video {
            has_audio,
            width,
            height,
        } => AssetDoc {
            id,
            kind: AssetKindName::Video,
            source,
            duration: Some(media_stamp(asset)),
            width: Some(width),
            height: Some(height),
            has_audio: Some(has_audio),
        },
        AssetKind::Audio => AssetDoc {
            id,
            kind: AssetKindName::Audio,
            source,
            duration: Some(media_stamp(asset)),
            width: None,
            height: None,
            has_audio: None,
        },
        AssetKind::Image { width, height } => AssetDoc {
            id,
            kind: AssetKindName::Image,
            source,
            duration: None,
            width: Some(width),
            height: Some(height),
            has_audio: None,
        },
    }
}

fn media_stamp(asset: &Asset) -> Stamp {
    stamp_doc(asset.duration().expect("video and audio store a duration"))
}

fn assets_from(documents: Vec<AssetDoc>) -> Result<Vec<Asset>, Error> {
    documents.into_iter().map(asset_from).collect()
}

fn asset_from(document: AssetDoc) -> Result<Asset, Error> {
    let id: AssetId = document.id.parse()?;
    let mut asset = match document.kind {
        AssetKindName::Video => Asset::video(
            document.source,
            stamp(document.duration.ok_or_else(|| missing("video duration"))?)?,
            document.width.ok_or_else(|| missing("video width"))?,
            document.height.ok_or_else(|| missing("video height"))?,
            document
                .has_audio
                .ok_or_else(|| missing("video has_audio"))?,
        )?,
        AssetKindName::Audio => {
            if document.width.is_some() || document.height.is_some() || document.has_audio.is_some()
            {
                return Err(Error::Unreadable("audio has picture fields".into()));
            }
            Asset::audio(
                document.source,
                stamp(document.duration.ok_or_else(|| missing("audio duration"))?)?,
            )?
        }
        AssetKindName::Image => {
            if document.duration.is_some() || document.has_audio.is_some() {
                return Err(Error::Unreadable("an image has no duration".into()));
            }
            Asset::image(
                document.source,
                document.width.ok_or_else(|| missing("image width"))?,
                document.height.ok_or_else(|| missing("image height"))?,
            )?
        }
    };
    asset.assign(id);
    Ok(asset)
}

fn timeline_from(document: &TimelineDoc, assets: &[Asset]) -> Result<Timeline, Error> {
    let id = document.id.parse()?;
    let mut tracks = Vec::with_capacity(document.tracks.len());
    for track in &document.tracks {
        tracks.push(track_from(track, assets)?);
    }
    check_track_order(&tracks)?;
    let expected: Vec<_> = tracks.iter().map(Track::id).collect();
    let Some(main_index) = tracks.iter().position(Track::is_main) else {
        return Err(Error::MissingMain);
    };
    let main = tracks.remove(main_index);
    let mut timeline = Timeline::new(id, main.id());
    for segment in main.segments() {
        timeline.main_mut().insert(segment.clone())?;
    }
    for track in tracks {
        timeline.insert(track)?;
    }
    let stored: Vec<_> = timeline.tracks().iter().map(Track::id).collect();
    if stored != expected {
        return Err(Error::TrackOrder);
    }
    Ok(timeline)
}

fn check_track_order(tracks: &[Track]) -> Result<(), Error> {
    let mains = tracks.iter().filter(|track| track.is_main()).count();
    if mains == 0 {
        return Err(Error::MissingMain);
    }
    if mains > 1 {
        return Err(Error::SecondMainTrack);
    }
    for pair in tracks.windows(2) {
        if pair[0].band() > pair[1].band() {
            return Err(Error::TrackOrder);
        }
    }
    Ok(())
}

fn track_doc(track: &Track) -> TrackDoc {
    TrackDoc {
        id: track.id().to_string(),
        kind: kind_name(track.kind()),
        main: track.is_main(),
        segments: track.segments().iter().map(segment_doc).collect(),
    }
}

fn track_from(document: &TrackDoc, assets: &[Asset]) -> Result<Track, Error> {
    let id: TrackId = document.id.parse()?;
    let kind = kind_from(document.kind);
    if document.main && kind != TrackKind::Video {
        return Err(Error::WrongTrack);
    }
    let mut segments = Vec::with_capacity(document.segments.len());
    for segment in &document.segments {
        segments.push(segment_from(segment, assets)?);
    }
    if document.main {
        let mut track = Track::main(id);
        for segment in segments {
            track.insert(segment)?;
        }
        return Ok(track);
    }
    let mut rest = segments.into_iter();
    let Some(first) = rest.next() else {
        return Err(Error::EmptyTrack);
    };
    let mut track = Track::occupied(id, kind, first)?;
    for segment in rest {
        track.insert(segment)?;
    }
    Ok(track)
}

fn segment_doc(segment: &Segment) -> SegmentDoc {
    SegmentDoc {
        id: segment.id().to_string(),
        asset: segment.asset_id().to_string(),
        source: segment.source().map(range_doc),
        timeline: SpanDoc {
            start: segment.timeline().start().value,
            duration: segment.timeline().duration().value,
        },
        channels: channel_name(segment.channels()),
        speed: SpeedDoc {
            numerator: segment.speed().numerator(),
            denominator: segment.speed().denominator().get(),
        },
        visible: segment.visible(),
        volume: segment.volume(),
        fade_in: segment.fade_in(),
        fade_out: segment.fade_out(),
        denoise: segment.denoise(),
        keep_pitch: segment.keep_pitch(),
        crop: crop_doc(segment.crop()),
        transform: transform_doc(segment.transform()),
        keyframes: segment.keyframes().iter().copied().map(key_doc).collect(),
    }
}

fn segment_from(document: &SegmentDoc, assets: &[Asset]) -> Result<Segment, Error> {
    let id: SegmentId = document.id.parse()?;
    let asset_id: AssetId = document.asset.parse()?;
    let asset = assets
        .iter()
        .find(|asset| asset.id() == asset_id)
        .ok_or(Error::UnknownAsset(asset_id))?;
    let speed = Speed::new(document.speed.numerator, document.speed.denominator)?;
    let channels = channels_from(document.channels);
    let mut segment = match asset.kind() {
        AssetKind::Image { .. } => image_segment(asset, document, speed, channels)?,
        AssetKind::Video { .. } | AssetKind::Audio => {
            media_segment(asset, document, speed, channels)?
        }
    };
    segment.assign(id);
    segment.set_visible(document.visible);
    segment.set_volume(document.volume)?;
    segment.set_fades(document.fade_in, document.fade_out)?;
    segment.set_denoise(document.denoise)?;
    segment.set_keep_pitch(document.keep_pitch)?;
    segment.set_crop(crop_from(&document.crop)?)?;
    segment.set_transform(transform_from(&document.transform)?);
    for key in &document.keyframes {
        segment.push_keyframe(key_from(key)?)?;
    }
    Ok(segment)
}

fn image_segment(
    asset: &Asset,
    document: &SegmentDoc,
    speed: Speed,
    channels: Channels,
) -> Result<Segment, Error> {
    if document.source.is_some() {
        return Err(Error::ImageSource);
    }
    if speed != Speed::ONE {
        return Err(Error::ImageSpeed);
    }
    if channels != Channels::Picture {
        return Err(Error::Channels);
    }
    Segment::image(asset, document.timeline.start, document.timeline.duration)
}

fn media_segment(
    asset: &Asset,
    document: &SegmentDoc,
    speed: Speed,
    channels: Channels,
) -> Result<Segment, Error> {
    let source = document.source.as_ref().ok_or(Error::MediaSource)?;
    let range = TimeRange::new(
        Timestamp::new(source.start, scale(source.timescale)?),
        Timestamp::new(source.duration, scale(source.timescale)?),
    )?;
    let segment = Segment::media(asset, range, document.timeline.start, channels, speed)?;
    let expected = segment.timeline().duration().value;
    if document.timeline.duration != expected {
        return Err(Error::TimelineDuration {
            found: document.timeline.duration,
            expected,
        });
    }
    Ok(segment)
}

fn transform_doc(transform: Transform) -> TransformDoc {
    let (x, y) = transform.position();
    let (scale_x, scale_y) = transform.scale();
    let (flip_horizontal, flip_vertical) = transform.flips();
    TransformDoc {
        position: [x, y],
        scale: [scale_x, scale_y],
        rotation: transform.rotation(),
        opacity: transform.opacity(),
        flip_horizontal,
        flip_vertical,
    }
}

fn transform_from(document: &TransformDoc) -> Result<Transform, Error> {
    Transform::new(
        (document.position[0], document.position[1]),
        (document.scale[0], document.scale[1]),
        document.rotation,
        document.opacity,
        document.flip_horizontal,
        document.flip_vertical,
    )
}

fn key_doc(key: Keyframe) -> KeyDoc {
    let offset = key.offset().value;
    let easing = easing_name(key.easing());
    match key.value() {
        KeyframeValue::Position { x, y } => KeyDoc::Position {
            offset,
            x,
            y,
            easing,
        },
        KeyframeValue::Scale { x, y } => KeyDoc::Scale {
            offset,
            x,
            y,
            easing,
        },
        KeyframeValue::Rotation(degrees) => KeyDoc::Rotation {
            offset,
            degrees,
            easing,
        },
        KeyframeValue::Opacity(value) => KeyDoc::Opacity {
            offset,
            value,
            easing,
        },
        KeyframeValue::Volume(value) => KeyDoc::Volume {
            offset,
            value,
            easing,
        },
    }
}

fn key_from(document: &KeyDoc) -> Result<Keyframe, Error> {
    let (offset, value, easing) = match document {
        KeyDoc::Position {
            offset,
            x,
            y,
            easing,
        } => (*offset, KeyframeValue::Position { x: *x, y: *y }, *easing),
        KeyDoc::Scale {
            offset,
            x,
            y,
            easing,
        } => (*offset, KeyframeValue::Scale { x: *x, y: *y }, *easing),
        KeyDoc::Rotation {
            offset,
            degrees,
            easing,
        } => (*offset, KeyframeValue::Rotation(*degrees), *easing),
        KeyDoc::Opacity {
            offset,
            value,
            easing,
        } => (*offset, KeyframeValue::Opacity(*value), *easing),
        KeyDoc::Volume {
            offset,
            value,
            easing,
        } => (*offset, KeyframeValue::Volume(*value), *easing),
    };
    Ok(Keyframe::new(offset, value)?.with_easing(easing_from(easing)))
}

fn easing_name(easing: Easing) -> EasingName {
    match easing {
        Easing::Linear => EasingName::Linear,
        Easing::EaseIn => EasingName::EaseIn,
        Easing::EaseOut => EasingName::EaseOut,
        Easing::EaseInOut => EasingName::EaseInOut,
    }
}

fn easing_from(name: EasingName) -> Easing {
    match name {
        EasingName::Linear => Easing::Linear,
        EasingName::EaseIn => Easing::EaseIn,
        EasingName::EaseOut => Easing::EaseOut,
        EasingName::EaseInOut => Easing::EaseInOut,
    }
}

fn easing_is_linear(easing: &EasingName) -> bool {
    matches!(easing, EasingName::Linear)
}

fn is_zero(value: &i64) -> bool {
    *value == 0
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn is_true(value: &bool) -> bool {
    *value
}

fn keep_pitch_on() -> bool {
    true
}

fn crop_doc(crop: Crop) -> CropDoc {
    let (left, right, top, bottom) = crop.insets();
    CropDoc {
        left,
        right,
        top,
        bottom,
    }
}

fn crop_from(document: &CropDoc) -> Result<Crop, Error> {
    Crop::new(document.left, document.right, document.top, document.bottom)
}

fn crop_is_whole(crop: &CropDoc) -> bool {
    crop.left == 0.0 && crop.right == 0.0 && crop.top == 0.0 && crop.bottom == 0.0
}

fn stamp_doc(time: Timestamp) -> Stamp {
    Stamp {
        value: time.value,
        timescale: time.timescale.get(),
    }
}

fn stamp(document: Stamp) -> Result<Timestamp, Error> {
    Ok(Timestamp::new(document.value, scale(document.timescale)?))
}

fn range_doc(range: TimeRange) -> RangeDoc {
    RangeDoc {
        start: range.start().value,
        duration: range.duration().value,
        timescale: range.start().timescale.get(),
    }
}

fn scale(timescale: u32) -> Result<NonZeroU32, Error> {
    NonZeroU32::new(timescale).ok_or(Error::ZeroTimescale)
}

fn missing(what: &str) -> Error {
    Error::Unreadable(format!("missing {what}"))
}

fn ratio_name(ratio: AspectRatio) -> RatioName {
    match ratio {
        AspectRatio::Original => RatioName::Original,
        AspectRatio::R16x9 => RatioName::R16x9,
        AspectRatio::R9x16 => RatioName::R9x16,
        AspectRatio::Square => RatioName::Square,
        AspectRatio::R4x5 => RatioName::R4x5,
        AspectRatio::R4x3 => RatioName::R4x3,
    }
}

fn ratio_from(ratio: RatioName) -> AspectRatio {
    match ratio {
        RatioName::Original => AspectRatio::Original,
        RatioName::R16x9 => AspectRatio::R16x9,
        RatioName::R9x16 => AspectRatio::R9x16,
        RatioName::Square => AspectRatio::Square,
        RatioName::R4x5 => AspectRatio::R4x5,
        RatioName::R4x3 => AspectRatio::R4x3,
    }
}

fn space_name(space: ColorSpace) -> SpaceName {
    match space {
        ColorSpace::Rec709 => SpaceName::Rec709,
        ColorSpace::Rec2020Hlg => SpaceName::Rec2020Hlg,
        ColorSpace::Rec2100Pq => SpaceName::Rec2100Pq,
    }
}

fn space_from(space: SpaceName) -> ColorSpace {
    match space {
        SpaceName::Rec709 => ColorSpace::Rec709,
        SpaceName::Rec2020Hlg => ColorSpace::Rec2020Hlg,
        SpaceName::Rec2100Pq => ColorSpace::Rec2100Pq,
    }
}

fn kind_name(kind: TrackKind) -> KindName {
    match kind {
        TrackKind::Video => KindName::Video,
        TrackKind::Audio => KindName::Audio,
        TrackKind::Adjust => KindName::Adjust,
        TrackKind::Filter => KindName::Filter,
        TrackKind::Effect => KindName::Effect,
        TrackKind::Text => KindName::Text,
        TrackKind::Sticker => KindName::Sticker,
        TrackKind::PluginEffect => KindName::PluginEffect,
    }
}

fn kind_from(kind: KindName) -> TrackKind {
    match kind {
        KindName::Video => TrackKind::Video,
        KindName::Audio => TrackKind::Audio,
        KindName::Adjust => TrackKind::Adjust,
        KindName::Filter => TrackKind::Filter,
        KindName::Effect => TrackKind::Effect,
        KindName::Text => TrackKind::Text,
        KindName::Sticker => TrackKind::Sticker,
        KindName::PluginEffect => TrackKind::PluginEffect,
    }
}

fn channel_name(channels: Channels) -> ChannelName {
    match channels {
        Channels::Picture => ChannelName::Picture,
        Channels::Sound => ChannelName::Sound,
        Channels::Both => ChannelName::Both,
    }
}

fn channels_from(channels: ChannelName) -> Channels {
    match channels {
        ChannelName::Picture => Channels::Picture,
        ChannelName::Sound => Channels::Sound,
        ChannelName::Both => Channels::Both,
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    version: u32,
    id: String,
    next_id: u64,
    name: String,
    settings: SettingsDoc,
    assets: Vec<AssetDoc>,
    timeline: TimelineDoc,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SettingsDoc {
    width: u32,
    height: u32,
    ratio: RatioName,
    background: [u8; 3],
    rate: RateDoc,
    drop_frame: bool,
    color_space: SpaceName,
    magnet: bool,
    linkage: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RateDoc {
    numerator: u32,
    denominator: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AssetDoc {
    id: String,
    kind: AssetKindName,
    source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    duration: Option<Stamp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    has_audio: Option<bool>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stamp {
    value: i64,
    timescale: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TimelineDoc {
    id: String,
    tracks: Vec<TrackDoc>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrackDoc {
    id: String,
    kind: KindName,
    main: bool,
    segments: Vec<SegmentDoc>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SegmentDoc {
    id: String,
    asset: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<RangeDoc>,
    timeline: SpanDoc,
    channels: ChannelName,
    speed: SpeedDoc,
    visible: bool,
    volume: f64,
    #[serde(default, skip_serializing_if = "is_zero")]
    fade_in: i64,
    #[serde(default, skip_serializing_if = "is_zero")]
    fade_out: i64,
    #[serde(default, skip_serializing_if = "is_false")]
    denoise: bool,
    #[serde(default = "keep_pitch_on", skip_serializing_if = "is_true")]
    keep_pitch: bool,
    #[serde(default, skip_serializing_if = "crop_is_whole")]
    crop: CropDoc,
    transform: TransformDoc,
    keyframes: Vec<KeyDoc>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RangeDoc {
    start: i64,
    duration: i64,
    timescale: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpanDoc {
    start: i64,
    duration: i64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeedDoc {
    numerator: u32,
    denominator: u32,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CropDoc {
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransformDoc {
    position: [f64; 2],
    scale: [f64; 2],
    rotation: f64,
    opacity: f64,
    flip_horizontal: bool,
    flip_vertical: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "property", rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
enum KeyDoc {
    Position {
        offset: i64,
        x: f64,
        y: f64,
        #[serde(default, skip_serializing_if = "easing_is_linear")]
        easing: EasingName,
    },
    Scale {
        offset: i64,
        x: f64,
        y: f64,
        #[serde(default, skip_serializing_if = "easing_is_linear")]
        easing: EasingName,
    },
    Rotation {
        offset: i64,
        degrees: f64,
        #[serde(default, skip_serializing_if = "easing_is_linear")]
        easing: EasingName,
    },
    Opacity {
        offset: i64,
        value: f64,
        #[serde(default, skip_serializing_if = "easing_is_linear")]
        easing: EasingName,
    },
    Volume {
        offset: i64,
        value: f64,
        #[serde(default, skip_serializing_if = "easing_is_linear")]
        easing: EasingName,
    },
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum EasingName {
    #[default]
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum AssetKindName {
    Video,
    Audio,
    Image,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
enum RatioName {
    #[serde(rename = "original")]
    Original,
    #[serde(rename = "16:9")]
    R16x9,
    #[serde(rename = "9:16")]
    R9x16,
    #[serde(rename = "1:1")]
    Square,
    #[serde(rename = "4:5")]
    R4x5,
    #[serde(rename = "4:3")]
    R4x3,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
enum SpaceName {
    #[serde(rename = "rec709")]
    Rec709,
    #[serde(rename = "rec2020hlg")]
    Rec2020Hlg,
    #[serde(rename = "rec2100pq")]
    Rec2100Pq,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum KindName {
    Video,
    Audio,
    Adjust,
    Filter,
    Effect,
    Text,
    Sticker,
    PluginEffect,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ChannelName {
    Picture,
    Sound,
    Both,
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cutlass_core::Timestamp;
    use serde_json::{Value, json};

    use super::{ChannelName, EasingName, FILE_VERSION, KeyDoc, KindName, RatioName, SpaceName};
    use crate::Error;
    use crate::asset::{Asset, AssetKind};
    use crate::command::{Command, Outcome};
    use crate::history::History;
    use crate::id::ProjectId;
    use crate::project::Project;
    use crate::rate::{FrameRate, SequenceRate};
    use crate::segment::{Channels, Easing, Keyframe, KeyframeValue, Segment, Transform};
    use crate::settings::{AspectRatio, Color, ColorSpace};
    use crate::time::{Speed, TimeRange};
    use crate::track::TrackKind;

    const EMPTY: &str = r#"{
  "version": 1,
  "id": "00112233-4455-6677-8899-aabbccddeeff",
  "next_id": 3,
  "name": "Untitled",
  "settings": {
    "width": 1920,
    "height": 1080,
    "ratio": "original",
    "background": [0, 0, 0],
    "rate": { "numerator": 30, "denominator": 1 },
    "drop_frame": false,
    "color_space": "rec709",
    "magnet": true,
    "linkage": true
  },
  "assets": [],
  "timeline": {
    "id": "timeline-1",
    "tracks": [
      {
        "id": "track-2",
        "kind": "video",
        "main": true,
        "segments": []
      }
    ]
  }
}"#;

    fn sample_id() -> ProjectId {
        ProjectId::from_bytes([
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ])
    }

    fn assert_same(left: &Project, right: &Project) {
        assert_eq!(left.id(), right.id());
        assert_eq!(left.next_id(), right.next_id());
        assert_eq!(left.name(), right.name());
        assert_eq!(left.settings(), right.settings());
        assert_eq!(left.assets(), right.assets());
        assert_eq!(left.timeline(), right.timeline());
    }

    fn assert_unreadable(text: &str) {
        match Project::from_json(text).expect_err("unreadable") {
            Error::Unreadable(message) => assert!(!message.is_empty()),
            other => panic!("expected an unreadable file, got {other:?}"),
        }
    }

    fn edit(value: &Value, change: impl FnOnce(&mut Value)) -> String {
        let mut copy = value.clone();
        change(&mut copy);
        serde_json::to_string(&copy).expect("rewrite")
    }

    fn refuse(text: &str, expected: Error) {
        assert_eq!(Project::from_json(text).expect_err("refused"), expected);
    }

    #[test]
    fn an_empty_file_loads_the_starting_project() {
        let project = Project::new(sample_id(), "Untitled");
        let loaded = Project::from_json(EMPTY).expect("load");
        assert_same(&project, &loaded);
        let text = project.to_json().expect("save");
        assert!(text.ends_with('\n'));
        assert!(!text.contains("undo"));
        assert_same(&project, &Project::from_json(&text).expect("reload"));
        assert_eq!(
            Project::from_json(&text)
                .expect("reload")
                .to_json()
                .expect("again"),
            text
        );
        let upper = EMPTY.replace("aabbccddeeff", "AABBCCDDEEFF");
        assert_eq!(Project::from_json(&upper).expect_err("upper"), Error::BadId);
    }

    #[test]
    fn names_in_the_file_match_the_model_note() {
        let kinds = [
            (KindName::Video, "\"video\""),
            (KindName::Audio, "\"audio\""),
            (KindName::Adjust, "\"adjust\""),
            (KindName::Filter, "\"filter\""),
            (KindName::Effect, "\"effect\""),
            (KindName::Text, "\"text\""),
            (KindName::Sticker, "\"sticker\""),
            (KindName::PluginEffect, "\"plugin-effect\""),
        ];
        for (kind, text) in kinds {
            assert_eq!(serde_json::to_string(&kind).expect("kind"), text);
        }
        let ratios = [
            (RatioName::Original, "\"original\""),
            (RatioName::R16x9, "\"16:9\""),
            (RatioName::R9x16, "\"9:16\""),
            (RatioName::Square, "\"1:1\""),
            (RatioName::R4x5, "\"4:5\""),
            (RatioName::R4x3, "\"4:3\""),
        ];
        for (ratio, text) in ratios {
            assert_eq!(serde_json::to_string(&ratio).expect("ratio"), text);
        }
        assert_eq!(
            serde_json::to_string(&SpaceName::Rec709).expect("space"),
            "\"rec709\""
        );
        assert_eq!(
            serde_json::to_string(&SpaceName::Rec2020Hlg).expect("space"),
            "\"rec2020hlg\""
        );
        assert_eq!(
            serde_json::to_string(&SpaceName::Rec2100Pq).expect("space"),
            "\"rec2100pq\""
        );
        assert_eq!(
            serde_json::to_string(&ChannelName::Both).expect("channels"),
            "\"both\""
        );
        let key = KeyDoc::Rotation {
            offset: 0,
            degrees: 1.0,
            easing: EasingName::Linear,
        };
        assert_eq!(
            serde_json::to_string(&key).expect("key"),
            "{\"property\":\"rotation\",\"offset\":0,\"degrees\":1.0}"
        );
        let easings = [
            (EasingName::Linear, "\"linear\""),
            (EasingName::EaseIn, "\"ease-in\""),
            (EasingName::EaseOut, "\"ease-out\""),
            (EasingName::EaseInOut, "\"ease-in-out\""),
        ];
        for (easing, text) in easings {
            assert_eq!(serde_json::to_string(&easing).expect("easing"), text);
        }
    }

    #[test]
    fn a_cut_round_trips_with_clips_and_the_counter() {
        let project = cut();
        let text = project.to_json().expect("save");
        assert!(!text.contains("undo"));
        assert!(text.contains("file:///cut.mp4"));
        let loaded = Project::from_json(&text).expect("load");
        assert_same(&project, &loaded);
        assert_eq!(loaded.to_json().expect("again"), text);
    }

    #[test]
    fn a_missing_easing_is_linear_and_an_unknown_one_is_refused() {
        let (project, value) = cut_value();
        let text = project.to_json().expect("save");
        assert!(!text.contains("easing"));
        let loaded = Project::from_json(&text).expect("load");
        let key = loaded.timeline().tracks()[1].segments()[0].keyframes()[0];
        assert_eq!(key.easing(), Easing::Linear);

        let eased = edit(&value, |value| {
            value["timeline"]["tracks"][1]["segments"][0]["keyframes"][0]["easing"] =
                json!("ease-in");
        });
        let loaded = Project::from_json(&eased).expect("load");
        assert_eq!(
            loaded.timeline().tracks()[1].segments()[0].keyframes()[0].easing(),
            Easing::EaseIn
        );
        let saved = loaded.to_json().expect("save");
        assert!(saved.contains("\"easing\": \"ease-in\""));
        assert_eq!(saved.matches("easing").count(), 1);
        assert_same(&loaded, &Project::from_json(&saved).expect("reload"));

        let unknown = edit(&value, |value| {
            value["timeline"]["tracks"][1]["segments"][0]["keyframes"][0]["easing"] =
                json!("bounce");
        });
        assert_unreadable(&unknown);
    }

    #[test]
    fn a_missing_fade_is_zero_and_a_long_one_is_refused() {
        let (project, value) = cut_value();
        let text = project.to_json().expect("save");
        assert!(!text.contains("fade"));
        let loaded = Project::from_json(&text).expect("load");
        let clip = &loaded.timeline().tracks()[1].segments()[0];
        assert_eq!(clip.fade_in(), 0);
        assert_eq!(clip.fade_out(), 0);

        let faded = edit(&value, |value| {
            value["timeline"]["tracks"][1]["segments"][0]["fade_in"] = json!(100_000);
        });
        let loaded = Project::from_json(&faded).expect("load");
        assert_eq!(
            loaded.timeline().tracks()[1].segments()[0].fade_in(),
            100_000
        );
        let saved = loaded.to_json().expect("save");
        assert!(saved.contains("\"fade_in\": 100000"));
        assert!(!saved.contains("fade_out"));
        assert_same(&loaded, &Project::from_json(&saved).expect("reload"));

        let negative = edit(&value, |value| {
            value["timeline"]["tracks"][1]["segments"][0]["fade_out"] = json!(-1);
        });
        refuse(&negative, Error::NegativeTime);
        let longer = edit(&value, |value| {
            value["timeline"]["tracks"][1]["segments"][0]["fade_in"] = json!(500_001);
        });
        refuse(&longer, Error::FadeOutOfRange);
    }

    #[test]
    fn a_missing_denoise_is_off_and_a_silent_segment_cannot_turn_it_on() {
        let (project, value) = cut_value();
        let text = project.to_json().expect("save");
        assert!(!text.contains("denoise"));
        let loaded = Project::from_json(&text).expect("load");
        assert!(!loaded.timeline().tracks()[1].segments()[0].denoise());

        let marked = edit(&value, |value| {
            value["timeline"]["tracks"][1]["segments"][0]["denoise"] = json!(true);
        });
        let loaded = Project::from_json(&marked).expect("load");
        assert!(loaded.timeline().tracks()[1].segments()[0].denoise());
        let saved = loaded.to_json().expect("save");
        assert!(saved.contains("\"denoise\": true"));
        assert_same(&loaded, &Project::from_json(&saved).expect("reload"));

        let silent = edit(&value, |value| {
            value["timeline"]["tracks"][2]["segments"][0]["denoise"] = json!(true);
        });
        refuse(&silent, Error::NoSound);
    }

    #[test]
    fn a_missing_keep_pitch_holds_the_pitch() {
        let (project, value) = cut_value();
        let text = project.to_json().expect("save");
        assert!(!text.contains("keep_pitch"));
        let loaded = Project::from_json(&text).expect("load");
        assert!(loaded.timeline().tracks()[1].segments()[0].keep_pitch());

        let followed = edit(&value, |value| {
            value["timeline"]["tracks"][1]["segments"][0]["keep_pitch"] = json!(false);
        });
        let loaded = Project::from_json(&followed).expect("load");
        assert!(!loaded.timeline().tracks()[1].segments()[0].keep_pitch());
        let saved = loaded.to_json().expect("save");
        assert!(saved.contains("\"keep_pitch\": false"));
        assert_same(&loaded, &Project::from_json(&saved).expect("reload"));

        let explicit = edit(&value, |value| {
            value["timeline"]["tracks"][1]["segments"][0]["keep_pitch"] = json!(true);
        });
        let loaded = Project::from_json(&explicit).expect("load");
        assert!(loaded.timeline().tracks()[1].segments()[0].keep_pitch());
        assert!(!loaded.to_json().expect("save").contains("keep_pitch"));

        let silent = edit(&value, |value| {
            value["timeline"]["tracks"][2]["segments"][0]["keep_pitch"] = json!(false);
        });
        refuse(&silent, Error::NoSound);
    }

    #[test]
    fn a_missing_crop_is_the_whole_picture() {
        let (project, value) = cut_value();
        let text = project.to_json().expect("save");
        assert!(!text.contains("\"crop\""));
        let loaded = Project::from_json(&text).expect("load");
        assert!(
            loaded.timeline().tracks()[1].segments()[0]
                .crop()
                .is_whole()
        );

        let cropped = edit(&value, |value| {
            value["timeline"]["tracks"][1]["segments"][0]["crop"] = json!({
                "left": 0.1,
                "right": 0.2,
                "top": 0.0,
                "bottom": 0.25
            });
        });
        let loaded = Project::from_json(&cropped).expect("load");
        assert_eq!(
            loaded.timeline().tracks()[1].segments()[0].crop().insets(),
            (0.1, 0.2, 0.0, 0.25)
        );
        let saved = loaded.to_json().expect("save");
        assert!(saved.contains("\"left\": 0.1"));
        assert_same(&loaded, &Project::from_json(&saved).expect("reload"));

        let whole = edit(&value, |value| {
            value["timeline"]["tracks"][1]["segments"][0]["crop"] = json!({
                "left": 0.0,
                "right": 0.0,
                "top": 0.0,
                "bottom": 0.0
            });
        });
        let loaded = Project::from_json(&whole).expect("load");
        assert!(
            loaded.timeline().tracks()[1].segments()[0]
                .crop()
                .is_whole()
        );
        assert!(!loaded.to_json().expect("save").contains("\"crop\""));

        let empty = edit(&value, |value| {
            value["timeline"]["tracks"][1]["segments"][0]["crop"] = json!({
                "left": 0.5,
                "right": 0.5,
                "top": 0.0,
                "bottom": 0.0
            });
        });
        refuse(&empty, Error::Crop);
        let sound = edit(&value, |value| {
            value["timeline"]["tracks"][0]["segments"][0]["crop"] = json!({
                "left": 0.1,
                "right": 0.0,
                "top": 0.0,
                "bottom": 0.0
            });
        });
        refuse(&sound, Error::NoPicture);
    }

    #[test]
    fn undo_is_left_out_and_the_counter_stays_ahead() {
        let mut history = History::new();
        let mut project = Project::new(sample_id(), "Untitled");
        let asset = Asset::video(
            "file:///gone.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            16,
            16,
            false,
        )
        .expect("video");
        history
            .apply(&mut project, &[Command::AddAsset(asset)])
            .expect("add");
        history.undo(&mut project).expect("undo");
        assert!(project.assets().is_empty());
        assert_eq!(project.next_id(), 4);
        let text = project.to_json().expect("save");
        assert!(!text.contains("gone.mp4"));
        assert!(!text.contains("undo"));
        assert_same(&project, &Project::from_json(&text).expect("load"));
    }

    #[test]
    fn any_version_but_1_loads_nothing() {
        assert_eq!(FILE_VERSION, 1);
        assert_eq!(
            Error::FileVersion { found: 2 }.to_string(),
            "project file version 2 is not 1"
        );
        assert_eq!(
            Project::from_json(r#"{"version":2,"name":"later"}"#).expect_err("newer"),
            Error::FileVersion { found: 2 }
        );
        assert_eq!(
            Project::from_json(r#"{"version":0}"#).expect_err("older"),
            Error::FileVersion { found: 0 }
        );
        assert_eq!(
            Project::from_json(r#"{"version":99,"settings":{"width":1}}"#).expect_err("far"),
            Error::FileVersion { found: 99 }
        );
    }

    #[test]
    fn a_file_that_does_not_match_loads_nothing() {
        assert_unreadable("{");
        assert_unreadable("");
        assert_unreadable("[]");
        assert_unreadable(r#"{"version":"1"}"#);
        assert_unreadable(r#"{"version":1.5}"#);
        assert_unreadable(r#"{"version":1}"#);
        let text = Project::new(sample_id(), "Untitled")
            .to_json()
            .expect("save");
        let value: Value = serde_json::from_str(&text).expect("json");
        refuse(
            &edit(&value, |value| value["next_id"] = json!(2)),
            Error::IdCounter { next_id: 2 },
        );
        assert_unreadable(&edit(&value, |value| value["history"] = json!([])));
        refuse(
            &edit(&value, |value| {
                value["settings"]["drop_frame"] = json!(true)
            }),
            Error::DropFrameRate,
        );
        refuse(
            &edit(&value, |value| value["settings"]["width"] = json!(0)),
            Error::ZeroCanvas,
        );
    }

    #[test]
    fn a_cut_that_breaks_a_rule_loads_nothing() {
        let (_project, value) = cut_value();
        refuse(
            &edit(&value, |value| {
                let tracks = value["timeline"]["tracks"].as_array_mut().expect("tracks");
                let last = tracks.len() - 1;
                tracks.swap(0, last);
            }),
            Error::TrackOrder,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][1]["segments"][0]["timeline"]["duration"] = json!(1);
            }),
            Error::TimelineDuration {
                found: 1,
                expected: 500_000,
            },
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][1]["segments"][0]["channels"] = json!("sound");
            }),
            Error::WrongTrack,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][1]["segments"][0]["id"] = json!("segment-1");
            }),
            Error::RepeatedId(1),
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][2]["segments"][0]["speed"]["numerator"] = json!(2);
            }),
            Error::ImageSpeed,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][2]["main"] = json!(true)
            }),
            Error::SecondMainTrack,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][1]["main"] = json!(false)
            }),
            Error::MissingMain,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][0]["main"] = json!(true)
            }),
            Error::WrongTrack,
        );
        refuse(
            &edit(&value, |value| {
                value["assets"][0]["duration"]["timescale"] = json!(0);
            }),
            Error::ZeroTimescale,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][2]["segments"][0]["source"] = json!({
                    "start": 0,
                    "duration": 1,
                    "timescale": 1
                });
            }),
            Error::ImageSource,
        );
    }

    fn cut_value() -> (Project, Value) {
        let project = cut();
        let text = project.to_json().expect("save");
        let value: Value = serde_json::from_str(&text).expect("json");
        assert_eq!(value["assets"][0]["source"], json!("file:///cut.mp4"));
        assert_eq!(value["timeline"]["tracks"][0]["kind"], json!("audio"));
        assert_eq!(
            value["timeline"]["tracks"]
                .as_array()
                .expect("tracks")
                .len(),
            3
        );
        assert_eq!(value["timeline"]["tracks"][1]["main"], json!(true));
        assert_eq!(
            value["timeline"]["tracks"][1]["segments"][0]["timeline"]["duration"],
            json!(500_000)
        );
        assert_eq!(
            value["timeline"]["tracks"][2]["segments"][0]["asset"],
            json!(project.assets()[2].id().to_string())
        );
        assert!(matches!(
            project.assets()[2].kind(),
            AssetKind::Image { .. }
        ));
        (project, value)
    }

    fn cut() -> Project {
        let mut history = History::new();
        let mut project = Project::new(sample_id(), "Untitled");
        let scale = NonZeroU32::new(30_000).expect("scale");
        let added = history
            .apply(
                &mut project,
                &[
                    Command::AddAsset(
                        Asset::video(
                            "file:///cut.mp4",
                            Timestamp::new(60_000, scale),
                            1280,
                            720,
                            true,
                        )
                        .expect("video"),
                    ),
                    Command::AddAsset(
                        Asset::audio(
                            "file:///room.m4a",
                            Timestamp::new(48_000, NonZeroU32::new(48_000).expect("audio scale")),
                        )
                        .expect("audio"),
                    ),
                    Command::AddAsset(Asset::image("file:///still.png", 8, 8).expect("image")),
                ],
            )
            .expect("add");
        let (video_id, audio_id, image_id) = match added.as_slice() {
            [
                Outcome::Asset(video_id),
                Outcome::Asset(audio_id),
                Outcome::Asset(image_id),
            ] => (*video_id, *audio_id, *image_id),
            other => panic!("three assets, got {other:?}"),
        };
        history
            .apply(&mut project, &evening_settings())
            .expect("settings");
        let main = project.timeline().main_track().id();
        let audio = audio_clip(&project, audio_id);
        let video = video_clip(&project, video_id, scale);
        let image = image_clip(&project, image_id);
        let placed = history
            .apply(
                &mut project,
                &[
                    Command::Place {
                        track: None,
                        segment: audio,
                    },
                    Command::Place {
                        track: Some(main),
                        segment: video,
                    },
                    Command::Place {
                        track: Some(main),
                        segment: image,
                    },
                ],
            )
            .expect("place");
        let video_segment = match placed.as_slice() {
            [_, Outcome::Segment(id), _] => *id,
            other => panic!("placed audio, video, and image, got {other:?}"),
        };
        let duration = project
            .timeline()
            .segment(video_segment)
            .expect("video")
            .timeline()
            .duration()
            .value;
        assert_eq!(duration, 500_000);
        history
            .apply(&mut project, &keyframes(video_segment, duration))
            .expect("keyframes");
        assert_eq!(project.timeline().tracks().len(), 3);
        assert_eq!(project.timeline().tracks()[0].kind(), TrackKind::Audio);
        assert!(project.timeline().tracks()[1].is_main());
        assert!(!project.timeline().tracks()[1].segments()[0].visible());
        assert_eq!(
            project.timeline().tracks()[2].segments()[0].asset_id(),
            image_id
        );
        project
    }

    fn evening_settings() -> Vec<Command> {
        vec![
            Command::Rename("Evening".to_string()),
            Command::SetCanvas {
                width: 1080,
                height: 1920,
                ratio: AspectRatio::R9x16,
            },
            Command::SetRate(SequenceRate::new(FrameRate::Fps2997, true).expect("rate")),
            Command::SetColorSpace(ColorSpace::Rec2100Pq),
            Command::SetBackground(Color::new(12, 34, 56)),
            Command::SetMagnet(false),
            Command::SetLinkage(false),
        ]
    }

    fn video_clip(project: &Project, id: crate::id::AssetId, scale: NonZeroU32) -> Segment {
        let asset = project.asset(id).expect("video");
        let source = TimeRange::new(Timestamp::new(0, scale), Timestamp::new(30_000, scale))
            .expect("source");
        let mut clip = Segment::media(
            asset,
            source,
            0,
            Channels::Both,
            Speed::new(2, 1).expect("speed"),
        )
        .expect("clip");
        clip.set_visible(false);
        clip.set_volume(0.25).expect("volume");
        clip.set_transform(
            Transform::new((0.25, -0.25), (2.0, 0.5), 90.0, 0.5, true, false).expect("transform"),
        );
        clip
    }

    fn audio_clip(project: &Project, id: crate::id::AssetId) -> Segment {
        let asset = project.asset(id).expect("audio");
        let duration = asset.duration().expect("audio duration");
        let source =
            TimeRange::new(Timestamp::new(0, duration.timescale), duration).expect("range");
        Segment::media(asset, source, 0, Channels::Sound, Speed::ONE).expect("audio clip")
    }

    fn image_clip(project: &Project, id: crate::id::AssetId) -> Segment {
        let asset = project.asset(id).expect("image");
        Segment::image(asset, 0, 1_000_000).expect("image clip")
    }

    fn keyframes(segment: crate::id::SegmentId, duration: i64) -> Vec<Command> {
        vec![
            Command::AddKeyframe {
                segment,
                keyframe: Keyframe::new(0, KeyframeValue::Position { x: 0.25, y: -0.25 })
                    .expect("position"),
            },
            Command::AddKeyframe {
                segment,
                keyframe: Keyframe::new(duration, KeyframeValue::Volume(0.0)).expect("volume"),
            },
        ]
    }
}
