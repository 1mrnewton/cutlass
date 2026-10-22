//! Version 1 of the project file.
//!
//! The text is JSON. This module does not open a path. `docs/project.md` is
//! the format.

use std::num::NonZeroU32;

use cutlass_core::Timestamp;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Error;
use crate::animation::{Animation, AnimationKind};
use crate::asset::{Asset, AssetKind};
use crate::attachment::{Attachment, Parameter, ParameterValue};
use crate::curve::{SpeedCurve, SpeedPoint};
use crate::id::{AssetId, SegmentId, TimelineId, TrackId};
use crate::project::Project;
use crate::rate::{FrameRate, SequenceRate};
use crate::segment::{
    BezierEase, Blend, Channels, Crop, Easing, Keyframe, KeyframeValue, Segment, Transform,
};
use crate::settings::{AspectRatio, Canvas, Color, ColorSpace, Settings};
use crate::time::{Speed, TimeRange};
use crate::timeline::Timeline;
use crate::track::{Track, TrackKind};
use crate::transition::Transition;

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
        timeline: timeline_doc(project.timeline()),
        nests: project.nests().iter().map(timeline_doc).collect(),
    }
}

fn timeline_doc(timeline: &Timeline) -> TimelineDoc {
    TimelineDoc {
        id: timeline.id().to_string(),
        tracks: timeline.tracks().iter().map(track_doc).collect(),
        markers: timeline.markers().to_vec(),
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
    let mut nests = Vec::with_capacity(document.nests.len());
    for nest in &document.nests {
        nests.push(timeline_from(nest, &assets)?);
    }
    Project::restore(
        id,
        document.next_id,
        document.name,
        settings,
        assets,
        timeline,
        nests,
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
    let reverses = asset.reverses().map(|id| id.to_string());
    match asset.kind() {
        AssetKind::Video {
            has_audio,
            width,
            height,
        } => AssetDoc {
            id,
            kind: AssetKindName::Video,
            source,
            reverses,
            duration: Some(media_stamp(asset)),
            width: Some(width),
            height: Some(height),
            has_audio: Some(has_audio),
        },
        AssetKind::Audio => AssetDoc {
            id,
            kind: AssetKindName::Audio,
            source,
            reverses,
            duration: Some(media_stamp(asset)),
            width: None,
            height: None,
            has_audio: None,
        },
        AssetKind::Image { width, height } => AssetDoc {
            id,
            kind: AssetKindName::Image,
            source,
            reverses,
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
    let assets = documents
        .into_iter()
        .map(asset_from)
        .collect::<Result<Vec<_>, _>>()?;
    check_reverses(&assets)?;
    Ok(assets)
}

fn check_reverses(assets: &[Asset]) -> Result<(), Error> {
    for (index, asset) in assets.iter().enumerate() {
        let Some(target_id) = asset.reverses() else {
            continue;
        };
        if target_id == asset.id() {
            return Err(Error::Reverse);
        }
        let target = assets
            .iter()
            .find(|item| item.id() == target_id)
            .ok_or(Error::UnknownAsset(target_id))?;
        if asset.duration().is_none() || target.duration().is_none() {
            return Err(Error::ImageSource);
        }
        let repeated = assets
            .iter()
            .take(index)
            .any(|earlier| earlier.reverses() == Some(target_id));
        if target.reverses().is_some()
            || target.kind() != asset.kind()
            || target.duration() != asset.duration()
            || target.source() != asset.source()
            || repeated
        {
            return Err(Error::Reverse);
        }
    }
    Ok(())
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
    if let Some(target) = document.reverses {
        asset.set_reverses(target.parse()?);
    }
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
    timeline.main_mut().set_locked(main.locked());
    timeline.main_mut().set_hidden(main.hidden());
    timeline.main_mut().set_muted(main.muted());
    timeline
        .main_mut()
        .set_transitions(main.transitions().to_vec());
    for track in tracks {
        timeline.insert(track)?;
    }
    let stored: Vec<_> = timeline.tracks().iter().map(Track::id).collect();
    if stored != expected {
        return Err(Error::TrackOrder);
    }
    if !document.markers.is_empty() {
        let end = timeline.duration().value;
        timeline.set_markers(take_times(&document.markers, end, true)?);
    }
    Ok(timeline)
}

fn take_times(times: &[i64], span: i64, through_end: bool) -> Result<Vec<i64>, Error> {
    if times.is_empty() {
        return Ok(Vec::new());
    }
    if through_end && span <= 0 {
        return Err(Error::Marker);
    }
    let mut stored = Vec::with_capacity(times.len());
    for time in times {
        if *time < 0 {
            return Err(Error::NegativeTime);
        }
        let fits = if through_end {
            *time <= span
        } else {
            span > 0 && *time < span
        };
        if !fits || !crate::mark::insert(&mut stored, *time) {
            return Err(Error::Marker);
        }
    }
    Ok(stored)
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
        locked: track.locked(),
        hidden: track.hidden(),
        muted: track.muted(),
        segments: track.segments().iter().map(segment_doc).collect(),
        transitions: track.transitions().iter().map(transition_doc).collect(),
    }
}

fn transition_doc(transition: &Transition) -> TransitionDoc {
    TransitionDoc {
        left: transition.left().to_string(),
        right: transition.right().to_string(),
        duration: transition.duration_us(),
        overlap: transition.overlap(),
        attachment: attachment_doc(transition.attachment()),
    }
}

fn transitions_from(document: &TrackDoc, track: &Track) -> Result<Vec<Transition>, Error> {
    let mut transitions = Vec::with_capacity(document.transitions.len());
    for item in &document.transitions {
        let left: SegmentId = item.left.parse()?;
        let right: SegmentId = item.right.parse()?;
        let transition = Transition::new(
            left,
            right,
            item.duration,
            item.overlap,
            attachment_from(&item.attachment)?,
        )?;
        let segments = track.segments();
        let Some(index) = segments.iter().position(|segment| segment.id() == left) else {
            return Err(Error::Transition);
        };
        let Some(neighbor) = segments.get(index + 1) else {
            return Err(Error::Transition);
        };
        if neighbor.id() != right
            || segments[index].timeline().end().value != neighbor.timeline().start().value
            || !transition.fits(
                segments[index].timeline().duration().value,
                neighbor.timeline().duration().value,
            )
        {
            return Err(Error::Transition);
        }
        let cut = segments[index].timeline().end().value;
        let (before, after) = transition.sides();
        let start = cut.checked_sub(before).ok_or(Error::DurationOverflow)?;
        let end = cut.checked_add(after).ok_or(Error::DurationOverflow)?;
        let clashes = transitions.iter().any(|existing: &Transition| {
            let Some(other_index) = segments
                .iter()
                .position(|segment| segment.id() == existing.left())
            else {
                return true;
            };
            let other_cut = segments[other_index].timeline().end().value;
            let (other_before, other_after) = existing.sides();
            let Some(other_start) = other_cut.checked_sub(other_before) else {
                return true;
            };
            let Some(other_end) = other_cut.checked_add(other_after) else {
                return true;
            };
            start < other_end && other_start < end
        });
        if clashes || transitions.iter().any(|existing| existing.left() == left) {
            return Err(Error::Transition);
        }
        transitions.push(transition);
    }
    transitions.sort_by_key(|transition| {
        track
            .segments()
            .iter()
            .find(|segment| segment.id() == transition.left())
            .map(|segment| segment.timeline().start().value)
            .unwrap_or(0)
    });
    Ok(transitions)
}

fn apply_flags(track: &mut Track, document: &TrackDoc) {
    track.set_locked(document.locked);
    track.set_hidden(document.hidden);
    track.set_muted(document.muted);
}

fn track_from(document: &TrackDoc, assets: &[Asset]) -> Result<Track, Error> {
    let id: TrackId = document.id.parse()?;
    let kind = kind_from(document.kind);
    if document.main && kind != TrackKind::Video {
        return Err(Error::WrongTrack);
    }
    if !document.main && !document.transitions.is_empty() {
        return Err(Error::Transition);
    }
    let mut segments = Vec::with_capacity(document.segments.len());
    for segment in &document.segments {
        segments.push(segment_from(segment, assets)?);
    }
    if kind != TrackKind::Audio && segments.iter().any(|segment| !segment.beats().is_empty()) {
        return Err(Error::Marker);
    }
    if document.main {
        let mut track = Track::main(id);
        for segment in segments {
            track.insert(segment)?;
        }
        apply_flags(&mut track, document);
        track.set_transitions(transitions_from(document, &track)?);
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
    apply_flags(&mut track, document);
    Ok(track)
}

fn segment_doc(segment: &Segment) -> SegmentDoc {
    SegmentDoc {
        id: segment.id().to_string(),
        asset: segment.asset_id().map(|id| id.to_string()),
        text: segment.words().map(str::to_owned),
        sticker: segment.sticker_name().map(str::to_owned),
        effect: segment.is_effect(),
        filter: segment.is_filter(),
        adjust: segment.is_adjust(),
        attachment: segment.attachment().map(attachment_doc),
        source: segment.source().map(range_doc),
        timeline: SpanDoc {
            start: segment.timeline().start().value,
            duration: segment.timeline().duration().value,
        },
        channels: channel_name(segment.channels()),
        speed: speed_doc(segment),
        visible: segment.visible(),
        volume: segment.volume(),
        fade_in: segment.fade_in(),
        fade_out: segment.fade_out(),
        denoise: segment.denoise(),
        keep_pitch: segment.keep_pitch(),
        crop: crop_doc(segment.crop()),
        blend: blend_name(segment.blend()),
        frozen: segment.frozen(),
        transform: transform_doc(segment.transform()),
        keyframes: segment.keyframes().iter().copied().map(key_doc).collect(),
        animations: AnimationKind::ALL
            .into_iter()
            .filter_map(|kind| {
                segment.animation(kind).map(|animation| AnimationDoc {
                    kind: animation_kind_name(kind),
                    name: animation.name().to_owned(),
                    duration: animation.duration_us(),
                })
            })
            .collect(),
        markers: segment.markers().to_vec(),
        beats: segment.beats().to_vec(),
        nest: segment.nest().map(|id| id.to_string()),
    }
}

fn animation_kind_name(kind: AnimationKind) -> AnimationKindName {
    match kind {
        AnimationKind::In => AnimationKindName::In,
        AnimationKind::Out => AnimationKindName::Out,
        AnimationKind::Combo => AnimationKindName::Combo,
        AnimationKind::Loop => AnimationKindName::Loop,
    }
}

fn animation_kind_from(kind: AnimationKindName) -> AnimationKind {
    match kind {
        AnimationKindName::In => AnimationKind::In,
        AnimationKindName::Out => AnimationKind::Out,
        AnimationKindName::Combo => AnimationKind::Combo,
        AnimationKindName::Loop => AnimationKind::Loop,
    }
}

fn segment_from(document: &SegmentDoc, assets: &[Asset]) -> Result<Segment, Error> {
    let id: SegmentId = document.id.parse()?;
    if document.text.is_some() {
        if document.asset.is_some()
            || document.sticker.is_some()
            || graded(document)
            || document.attachment.is_some()
            || document.nest.is_some()
        {
            return Err(Error::Text);
        }
        let mut segment = text_segment(document)?;
        segment.assign(id);
        finish_segment(&mut segment, document)?;
        return Ok(segment);
    }
    if document.sticker.is_some() {
        if document.asset.is_some()
            || graded(document)
            || document.attachment.is_some()
            || document.nest.is_some()
        {
            return Err(Error::Sticker);
        }
        let mut segment = sticker_segment(document)?;
        segment.assign(id);
        finish_segment(&mut segment, document)?;
        return Ok(segment);
    }
    if let Some(grade) = grade_of(document)? {
        if document.asset.is_some() || document.nest.is_some() {
            return Err(grade.error());
        }
        let mut segment = grade_segment(document, grade)?;
        if let Some(attachment) = &document.attachment {
            segment.set_attachment(Some(attachment_from(attachment)?))?;
        }
        segment.assign(id);
        finish_segment(&mut segment, document)?;
        return Ok(segment);
    }
    if document.attachment.is_some() {
        return Err(Error::Attachment);
    }
    let Some(asset_name) = document.asset.as_deref() else {
        return Err(Error::Text);
    };
    let asset_id: AssetId = asset_name.parse()?;
    let asset = assets
        .iter()
        .find(|asset| asset.id() == asset_id)
        .ok_or(Error::UnknownAsset(asset_id))?;
    let pace = pace_from(&document.speed)?;
    let channels = channels_from(document.channels);
    let mut segment = match asset.kind() {
        AssetKind::Image { .. } => {
            if document.nest.is_some() || document.frozen {
                return Err(Error::ImageSource);
            }
            let Pace::Constant(speed) = pace else {
                return Err(Error::ImageSpeed);
            };
            image_segment(asset, document, speed, channels)?
        }
        AssetKind::Video { .. } | AssetKind::Audio => {
            media_segment(asset, document, pace, channels)?
        }
    };
    segment.assign(id);
    finish_segment(&mut segment, document)?;
    Ok(segment)
}

fn finish_segment(segment: &mut Segment, document: &SegmentDoc) -> Result<(), Error> {
    segment.set_visible(document.visible);
    segment.set_volume(document.volume)?;
    segment.set_fades(document.fade_in, document.fade_out)?;
    segment.set_denoise(document.denoise)?;
    segment.set_keep_pitch(document.keep_pitch)?;
    segment.set_crop(crop_from(&document.crop)?)?;
    segment.set_blend(blend_from(document.blend))?;
    segment.set_transform(transform_from(&document.transform)?);
    for key in &document.keyframes {
        segment.push_keyframe(key_from(key)?)?;
    }
    let mut seen = [false; 4];
    for animation in &document.animations {
        let kind = animation_kind_from(animation.kind);
        let index = match kind {
            AnimationKind::In => 0,
            AnimationKind::Out => 1,
            AnimationKind::Combo => 2,
            AnimationKind::Loop => 3,
        };
        if seen[index] {
            return Err(Error::Animation);
        }
        seen[index] = true;
        segment.set_animation(
            kind,
            Some(Animation::new(animation.name.clone(), animation.duration)?),
        )?;
    }
    let span = segment.timeline().duration().value;
    segment.set_marks(
        take_times(&document.markers, span, false)?,
        take_times(&document.beats, span, false)?,
    );
    Ok(())
}

fn text_segment(document: &SegmentDoc) -> Result<Segment, Error> {
    let Some(words) = document.text.clone() else {
        return Err(Error::Text);
    };
    if document.source.is_some() || document.frozen {
        return Err(Error::Text);
    }
    match pace_from(&document.speed)? {
        Pace::Curve(_) => return Err(Error::Text),
        Pace::Constant(speed) => {
            if speed != Speed::ONE {
                return Err(Error::Text);
            }
        }
    }
    if channels_from(document.channels) != Channels::Picture {
        return Err(Error::Channels);
    }
    Segment::text(words, document.timeline.start, document.timeline.duration)
}

fn sticker_segment(document: &SegmentDoc) -> Result<Segment, Error> {
    let Some(name) = document.sticker.clone() else {
        return Err(Error::Sticker);
    };
    if document.source.is_some() || document.frozen {
        return Err(Error::Sticker);
    }
    match pace_from(&document.speed)? {
        Pace::Curve(_) => return Err(Error::Sticker),
        Pace::Constant(speed) => {
            if speed != Speed::ONE {
                return Err(Error::Sticker);
            }
        }
    }
    if channels_from(document.channels) != Channels::Picture {
        return Err(Error::Channels);
    }
    Segment::sticker(name, document.timeline.start, document.timeline.duration)
}

fn graded(document: &SegmentDoc) -> bool {
    document.effect || document.filter || document.adjust
}

fn grade_of(document: &SegmentDoc) -> Result<Option<Grade>, Error> {
    match (document.effect, document.filter, document.adjust) {
        (false, false, false) => Ok(None),
        (true, false, false) => Ok(Some(Grade::Effect)),
        (false, true, false) => Ok(Some(Grade::Filter)),
        (false, false, true) => Ok(Some(Grade::Adjust)),
        _ => Err(Error::Effect),
    }
}

fn grade_segment(document: &SegmentDoc, grade: Grade) -> Result<Segment, Error> {
    if document.source.is_some()
        || document.frozen
        || document.text.is_some()
        || document.sticker.is_some()
    {
        return Err(grade.error());
    }
    match pace_from(&document.speed)? {
        Pace::Curve(_) => return Err(grade.error()),
        Pace::Constant(speed) => {
            if speed != Speed::ONE {
                return Err(grade.error());
            }
        }
    }
    if channels_from(document.channels) != Channels::Picture {
        return Err(Error::Channels);
    }
    grade.build(document.timeline.start, document.timeline.duration)
}

#[derive(Clone, Copy)]
enum Grade {
    Effect,
    Filter,
    Adjust,
}

impl Grade {
    fn error(self) -> Error {
        match self {
            Grade::Effect => Error::Effect,
            Grade::Filter => Error::Filter,
            Grade::Adjust => Error::Adjust,
        }
    }

    fn build(self, start: i64, duration: i64) -> Result<Segment, Error> {
        match self {
            Grade::Effect => Segment::effect(start, duration),
            Grade::Filter => Segment::filter(start, duration),
            Grade::Adjust => Segment::adjust(start, duration),
        }
    }
}

fn attachment_doc(attachment: &Attachment) -> AttachmentDoc {
    AttachmentDoc {
        plugin: attachment.plugin().to_owned(),
        version: attachment.version().to_owned(),
        parameters: attachment
            .parameters()
            .iter()
            .map(|parameter| ParameterDoc {
                name: parameter.name().to_owned(),
                value: match parameter.value() {
                    ParameterValue::Bool(value) => ValueDoc::Bool(*value),
                    ParameterValue::Number(value) => ValueDoc::Number(*value),
                    ParameterValue::Text(value) => ValueDoc::Text(value.clone()),
                },
            })
            .collect(),
    }
}

fn attachment_from(document: &AttachmentDoc) -> Result<Attachment, Error> {
    let parameters = document
        .parameters
        .iter()
        .map(|parameter| {
            let value = match &parameter.value {
                ValueDoc::Bool(value) => ParameterValue::Bool(*value),
                ValueDoc::Number(value) => ParameterValue::Number(*value),
                ValueDoc::Text(value) => ParameterValue::Text(value.clone()),
            };
            Parameter::new(parameter.name.clone(), value)
        })
        .collect();
    Attachment::new(
        document.plugin.clone(),
        document.version.clone(),
        parameters,
    )
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AttachmentDoc {
    plugin: String,
    version: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    parameters: Vec<ParameterDoc>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterDoc {
    name: String,
    value: ValueDoc,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum ValueDoc {
    Bool(bool),
    Number(f64),
    Text(String),
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
    pace: Pace,
    channels: Channels,
) -> Result<Segment, Error> {
    let source = document.source.as_ref().ok_or(Error::MediaSource)?;
    let range = TimeRange::new(
        Timestamp::new(source.start, scale(source.timescale)?),
        Timestamp::new(source.duration, scale(source.timescale)?),
    )?;
    let speed = match &pace {
        Pace::Constant(speed) => *speed,
        Pace::Curve(_) => Speed::ONE,
    };
    let mut segment = Segment::media(asset, range, document.timeline.start, channels, speed)?;
    if document.frozen {
        if document.nest.is_some() || matches!(pace, Pace::Curve(_)) {
            return Err(Error::Freeze);
        }
        segment.hold_frame(document.timeline.duration)?;
        return Ok(segment);
    }
    if let Pace::Curve(curve) = pace {
        segment.set_curve(Some(curve))?;
    }
    if let Some(name) = &document.nest {
        let nest: TimelineId = name.parse()?;
        segment.set_nest(Some(nest))?;
    }
    let expected = segment.timeline().duration().value;
    if document.timeline.duration != expected {
        return Err(Error::TimelineDuration {
            found: document.timeline.duration,
            expected,
        });
    }
    Ok(segment)
}

fn speed_doc(segment: &Segment) -> SpeedDoc {
    if let Some(curve) = segment.curve() {
        return SpeedDoc {
            numerator: None,
            denominator: None,
            points: curve
                .points()
                .iter()
                .map(|point| {
                    let (numerator, denominator) = point.at();
                    SpeedPointDoc {
                        at: [numerator, denominator],
                        speed: RationalDoc {
                            numerator: point.speed().numerator(),
                            denominator: point.speed().denominator().get(),
                        },
                    }
                })
                .collect(),
        };
    }
    SpeedDoc {
        numerator: Some(segment.speed().numerator()),
        denominator: Some(segment.speed().denominator().get()),
        points: Vec::new(),
    }
}

fn pace_from(speed: &SpeedDoc) -> Result<Pace, Error> {
    let curved = !speed.points.is_empty();
    let rated = speed.numerator.is_some() || speed.denominator.is_some();
    if curved && rated {
        return Err(Error::Curve);
    }
    if curved {
        let mut points = Vec::with_capacity(speed.points.len());
        for point in &speed.points {
            points.push(SpeedPoint::new(
                point.at[0],
                point.at[1],
                Speed::new(point.speed.numerator, point.speed.denominator)?,
            )?);
        }
        return Ok(Pace::Curve(SpeedCurve::new(points)?));
    }
    let numerator = speed.numerator.ok_or(Error::Curve)?;
    let denominator = speed.denominator.ok_or(Error::Curve)?;
    Ok(Pace::Constant(Speed::new(numerator, denominator)?))
}

enum Pace {
    Constant(Speed),
    Curve(SpeedCurve),
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
    let bezier = bezier_points(key.easing());
    match key.value() {
        KeyframeValue::Position { x, y } => KeyDoc::Position {
            offset,
            x,
            y,
            easing,
            bezier,
        },
        KeyframeValue::Scale { x, y } => KeyDoc::Scale {
            offset,
            x,
            y,
            easing,
            bezier,
        },
        KeyframeValue::Rotation(degrees) => KeyDoc::Rotation {
            offset,
            degrees,
            easing,
            bezier,
        },
        KeyframeValue::Opacity(value) => KeyDoc::Opacity {
            offset,
            value,
            easing,
            bezier,
        },
        KeyframeValue::Volume(value) => KeyDoc::Volume {
            offset,
            value,
            easing,
            bezier,
        },
    }
}

fn key_from(document: &KeyDoc) -> Result<Keyframe, Error> {
    let (offset, value, easing, bezier) = match document {
        KeyDoc::Position {
            offset,
            x,
            y,
            easing,
            bezier,
        } => (
            *offset,
            KeyframeValue::Position { x: *x, y: *y },
            *easing,
            *bezier,
        ),
        KeyDoc::Scale {
            offset,
            x,
            y,
            easing,
            bezier,
        } => (
            *offset,
            KeyframeValue::Scale { x: *x, y: *y },
            *easing,
            *bezier,
        ),
        KeyDoc::Rotation {
            offset,
            degrees,
            easing,
            bezier,
        } => (*offset, KeyframeValue::Rotation(*degrees), *easing, *bezier),
        KeyDoc::Opacity {
            offset,
            value,
            easing,
            bezier,
        } => (*offset, KeyframeValue::Opacity(*value), *easing, *bezier),
        KeyDoc::Volume {
            offset,
            value,
            easing,
            bezier,
        } => (*offset, KeyframeValue::Volume(*value), *easing, *bezier),
    };
    Ok(Keyframe::new(offset, value)?.with_easing(easing_from(easing, bezier)?))
}

fn easing_name(easing: Easing) -> EasingName {
    match easing {
        Easing::Linear => EasingName::Linear,
        Easing::EaseIn => EasingName::EaseIn,
        Easing::EaseOut => EasingName::EaseOut,
        Easing::EaseInOut => EasingName::EaseInOut,
        Easing::Bezier(_) => EasingName::Bezier,
    }
}

fn bezier_points(easing: Easing) -> Option<[f64; 4]> {
    match easing {
        Easing::Bezier(ease) => Some(ease.points()),
        Easing::Linear | Easing::EaseIn | Easing::EaseOut | Easing::EaseInOut => None,
    }
}

fn easing_from(name: EasingName, bezier: Option<[f64; 4]>) -> Result<Easing, Error> {
    if let Some(points) = bezier {
        if !matches!(name, EasingName::Bezier) {
            return Err(Error::Bezier);
        }
        return Ok(Easing::Bezier(BezierEase::new(
            points[0], points[1], points[2], points[3],
        )?));
    }
    match name {
        EasingName::Linear => Ok(Easing::Linear),
        EasingName::EaseIn => Ok(Easing::EaseIn),
        EasingName::EaseOut => Ok(Easing::EaseOut),
        EasingName::EaseInOut => Ok(Easing::EaseInOut),
        EasingName::Bezier => Err(Error::Bezier),
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

fn blend_name(blend: Blend) -> BlendName {
    match blend {
        Blend::Normal => BlendName::Normal,
        Blend::Multiply => BlendName::Multiply,
        Blend::Screen => BlendName::Screen,
        Blend::Overlay => BlendName::Overlay,
        Blend::Darken => BlendName::Darken,
        Blend::Lighten => BlendName::Lighten,
        Blend::Add => BlendName::Add,
        Blend::SoftLight => BlendName::SoftLight,
    }
}

fn blend_from(name: BlendName) -> Blend {
    match name {
        BlendName::Normal => Blend::Normal,
        BlendName::Multiply => Blend::Multiply,
        BlendName::Screen => Blend::Screen,
        BlendName::Overlay => Blend::Overlay,
        BlendName::Darken => Blend::Darken,
        BlendName::Lighten => Blend::Lighten,
        BlendName::Add => Blend::Add,
        BlendName::SoftLight => Blend::SoftLight,
    }
}

fn blend_is_normal(blend: &BlendName) -> bool {
    *blend == BlendName::Normal
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    nests: Vec<TimelineDoc>,
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
    reverses: Option<String>,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    markers: Vec<i64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrackDoc {
    id: String,
    kind: KindName,
    main: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    locked: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    hidden: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    muted: bool,
    segments: Vec<SegmentDoc>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    transitions: Vec<TransitionDoc>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransitionDoc {
    left: String,
    right: String,
    duration: i64,
    #[serde(default, skip_serializing_if = "is_false")]
    overlap: bool,
    attachment: AttachmentDoc,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SegmentDoc {
    id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    asset: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sticker: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    effect: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    filter: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    adjust: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    attachment: Option<AttachmentDoc>,
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
    #[serde(default, skip_serializing_if = "blend_is_normal")]
    blend: BlendName,
    #[serde(default, skip_serializing_if = "is_false")]
    frozen: bool,
    transform: TransformDoc,
    keyframes: Vec<KeyDoc>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    animations: Vec<AnimationDoc>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    markers: Vec<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    beats: Vec<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    nest: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnimationDoc {
    kind: AnimationKindName,
    name: String,
    duration: i64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum AnimationKindName {
    In,
    Out,
    Combo,
    Loop,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    numerator: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    denominator: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    points: Vec<SpeedPointDoc>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeedPointDoc {
    at: [u32; 2],
    speed: RationalDoc,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RationalDoc {
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bezier: Option<[f64; 4]>,
    },
    Scale {
        offset: i64,
        x: f64,
        y: f64,
        #[serde(default, skip_serializing_if = "easing_is_linear")]
        easing: EasingName,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bezier: Option<[f64; 4]>,
    },
    Rotation {
        offset: i64,
        degrees: f64,
        #[serde(default, skip_serializing_if = "easing_is_linear")]
        easing: EasingName,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bezier: Option<[f64; 4]>,
    },
    Opacity {
        offset: i64,
        value: f64,
        #[serde(default, skip_serializing_if = "easing_is_linear")]
        easing: EasingName,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bezier: Option<[f64; 4]>,
    },
    Volume {
        offset: i64,
        value: f64,
        #[serde(default, skip_serializing_if = "easing_is_linear")]
        easing: EasingName,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bezier: Option<[f64; 4]>,
    },
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum BlendName {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    Add,
    SoftLight,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum EasingName {
    #[default]
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bezier,
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

    use super::{
        AnimationKindName, BlendName, ChannelName, EasingName, FILE_VERSION, KeyDoc, KindName,
        RatioName, SpaceName,
    };
    use crate::Error;
    use crate::asset::{Asset, AssetKind};
    use crate::command::{Command, Outcome};
    use crate::curve::{SpeedCurve, SpeedPoint};
    use crate::history::History;
    use crate::id::{AssetId, ProjectId, TimelineId};
    use crate::project::Project;
    use crate::rate::{FrameRate, SequenceRate};
    use crate::segment::{
        BezierEase, Blend, Channels, Easing, Keyframe, KeyframeValue, Segment, Transform,
    };
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
        assert_eq!(left.nests(), right.nests());
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
        assert_eq!(
            serde_json::to_string(&AnimationKindName::In).expect("animation"),
            "\"in\""
        );
        assert_eq!(
            serde_json::to_string(&AnimationKindName::Out).expect("animation"),
            "\"out\""
        );
        assert_eq!(
            serde_json::to_string(&AnimationKindName::Combo).expect("animation"),
            "\"combo\""
        );
        assert_eq!(
            serde_json::to_string(&AnimationKindName::Loop).expect("animation"),
            "\"loop\""
        );
        let key = KeyDoc::Rotation {
            offset: 0,
            degrees: 1.0,
            easing: EasingName::Linear,
            bezier: None,
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
            (EasingName::Bezier, "\"bezier\""),
        ];
        for (easing, text) in easings {
            assert_eq!(serde_json::to_string(&easing).expect("easing"), text);
        }
        let blends = [
            (BlendName::Normal, "\"normal\""),
            (BlendName::Multiply, "\"multiply\""),
            (BlendName::Screen, "\"screen\""),
            (BlendName::Overlay, "\"overlay\""),
            (BlendName::Darken, "\"darken\""),
            (BlendName::Lighten, "\"lighten\""),
            (BlendName::Add, "\"add\""),
            (BlendName::SoftLight, "\"soft-light\""),
        ];
        for (blend, text) in blends {
            assert_eq!(serde_json::to_string(&blend).expect("blend"), text);
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
    fn a_bezier_easing_round_trips_and_a_bad_handle_refuses_the_file() {
        let (project, value) = cut_value();
        let text = project.to_json().expect("save");
        assert!(!text.contains("bezier"));

        let eased = edit(&value, |value| {
            let sample = &mut value["timeline"]["tracks"][1]["segments"][0]["keyframes"][0];
            put(sample, "easing", json!("bezier"));
            put(sample, "bezier", json!([0.0, 0.0, 1.0, 0.0]));
        });
        let loaded = Project::from_json(&eased).expect("load");
        assert_eq!(
            loaded.timeline().tracks()[1].segments()[0].keyframes()[0].easing(),
            Easing::Bezier(BezierEase::new(0.0, 0.0, 1.0, 0.0).expect("handles"))
        );
        let saved = loaded.to_json().expect("save");
        assert!(saved.contains("\"easing\": \"bezier\""));
        assert!(saved.contains("\"bezier\""));
        assert_same(&loaded, &Project::from_json(&saved).expect("reload"));

        let over = edit(&value, |value| {
            let sample = &mut value["timeline"]["tracks"][1]["segments"][0]["keyframes"][0];
            put(sample, "easing", json!("bezier"));
            put(sample, "bezier", json!([0.0, 2.0, 1.0, -1.0]));
        });
        let loaded = Project::from_json(&over).expect("load");
        assert_eq!(
            loaded.timeline().tracks()[1].segments()[0].keyframes()[0].easing(),
            Easing::Bezier(BezierEase::new(0.0, 2.0, 1.0, -1.0).expect("handles"))
        );
        assert_same(
            &loaded,
            &Project::from_json(&loaded.to_json().expect("save")).expect("reload"),
        );

        let missing = edit(&value, |value| {
            put(
                &mut value["timeline"]["tracks"][1]["segments"][0]["keyframes"][0],
                "easing",
                json!("bezier"),
            );
        });
        refuse(&missing, Error::Bezier);
        let extra = edit(&value, |value| {
            let sample = &mut value["timeline"]["tracks"][1]["segments"][0]["keyframes"][0];
            put(sample, "easing", json!("ease-in"));
            put(sample, "bezier", json!([0.0, 0.0, 1.0, 0.0]));
        });
        refuse(&extra, Error::Bezier);
        let outside = edit(&value, |value| {
            let sample = &mut value["timeline"]["tracks"][1]["segments"][0]["keyframes"][0];
            put(sample, "easing", json!("bezier"));
            put(sample, "bezier", json!([1.5, 0.0, 0.5, 1.0]));
        });
        refuse(&outside, Error::Bezier);
        let short = edit(&value, |value| {
            let sample = &mut value["timeline"]["tracks"][1]["segments"][0]["keyframes"][0];
            put(sample, "easing", json!("bezier"));
            put(sample, "bezier", json!([0.0, 0.0, 1.0]));
        });
        assert_unreadable(&short);
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
    fn a_missing_blend_is_normal() {
        let (project, value) = cut_value();
        let text = project.to_json().expect("save");
        assert!(!text.contains("\"blend\""));
        let loaded = Project::from_json(&text).expect("load");
        assert_eq!(
            loaded.timeline().tracks()[1].segments()[0].blend(),
            Blend::Normal
        );

        let multiplied = edit(&value, |value| {
            value["timeline"]["tracks"][1]["segments"][0]["blend"] = json!("multiply");
        });
        let loaded = Project::from_json(&multiplied).expect("load");
        assert_eq!(
            loaded.timeline().tracks()[1].segments()[0].blend(),
            Blend::Multiply
        );
        let saved = loaded.to_json().expect("save");
        assert!(saved.contains("\"blend\": \"multiply\""));
        assert_same(&loaded, &Project::from_json(&saved).expect("reload"));

        let normal = edit(&value, |value| {
            value["timeline"]["tracks"][1]["segments"][0]["blend"] = json!("normal");
        });
        let loaded = Project::from_json(&normal).expect("load");
        assert_eq!(
            loaded.timeline().tracks()[1].segments()[0].blend(),
            Blend::Normal
        );
        assert!(!loaded.to_json().expect("save").contains("\"blend\""));

        let unknown = edit(&value, |value| {
            value["timeline"]["tracks"][1]["segments"][0]["blend"] = json!("xor");
        });
        assert_unreadable(&unknown);
        let sound = edit(&value, |value| {
            value["timeline"]["tracks"][0]["segments"][0]["blend"] = json!("screen");
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

    #[test]
    fn a_freeze_is_one_picture_tick() {
        let (_project, value) = cut_value();
        refuse(
            &edit(&value, |value| {
                let segment = &mut value["timeline"]["tracks"][1]["segments"][0];
                segment["channels"] = json!("picture");
                segment["frozen"] = json!(true);
            }),
            Error::Freeze,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][2]["segments"][0]["frozen"] = json!(true);
            }),
            Error::ImageSource,
        );
        refuse(
            &edit(&value, |value| {
                let segment = &mut value["timeline"]["tracks"][1]["segments"][0];
                segment["channels"] = json!("picture");
                segment["frozen"] = json!(true);
                segment["source"]["duration"] = json!(1);
                segment["timeline"]["duration"] = json!(3_000_000);
                segment["speed"]["numerator"] = json!(2);
            }),
            Error::Freeze,
        );
        let text = edit(&value, |value| {
            let segment = &mut value["timeline"]["tracks"][1]["segments"][0];
            segment["channels"] = json!("picture");
            segment["frozen"] = json!(true);
            segment["source"]["duration"] = json!(1);
            segment["speed"]["numerator"] = json!(1);
            segment["timeline"]["duration"] = json!(3_000_000);
        });
        let loaded = Project::from_json(&text).expect("freeze");
        let segment = &loaded.timeline().main_track().segments()[0];
        assert!(segment.frozen());
        assert_eq!(segment.channels(), Channels::Picture);
        assert_eq!(segment.timeline().duration().value, 3_000_000);
        assert_eq!(segment.source().expect("source").duration().value, 1);
        assert_eq!(segment.speed(), Speed::ONE);
        let saved = loaded.to_json().expect("save");
        assert!(saved.contains("\"frozen\": true"));
        let again = Project::from_json(&saved).expect("reload");
        assert!(again.timeline().main_track().segments()[0].frozen());
        assert!(!again.to_json().expect("save").contains("\"frozen\": false"));
    }

    #[test]
    fn a_reverse_names_one_matching_asset() {
        let (_project, value) = cut_value();
        let next = value["next_id"].as_u64().expect("counter");
        let forward = value["assets"][0]["id"]
            .as_str()
            .expect("video")
            .to_string();
        let mut valid = value.clone();
        let mut reverse = valid["assets"][0].clone();
        reverse["id"] = json!(format!("asset-{next}"));
        reverse["reverses"] = json!(forward);
        valid["assets"]
            .as_array_mut()
            .expect("assets")
            .push(reverse);
        valid["next_id"] = json!(next + 1);
        let loaded =
            Project::from_json(&serde_json::to_string(&valid).expect("json")).expect("load");
        let (target, source) = {
            let stored = loaded
                .assets()
                .iter()
                .find(|asset| asset.reverses().is_some())
                .expect("reverse");
            (
                stored.reverses().expect("target").to_string(),
                stored.source().to_string(),
            )
        };
        assert_eq!(target, forward);
        assert_eq!(source, loaded.assets()[0].source());
        let saved = loaded.to_json().expect("save");
        assert_eq!(saved.matches("\"reverses\"").count(), 1);

        let mut missing = valid.clone();
        missing["assets"][3]["reverses"] = json!("asset-99");
        refuse(
            &serde_json::to_string(&missing).expect("json"),
            Error::UnknownAsset(AssetId::new(99).expect("id")),
        );
        let mut itself = valid.clone();
        itself["assets"][3]["reverses"] = json!(format!("asset-{next}"));
        refuse(
            &serde_json::to_string(&itself).expect("json"),
            Error::Reverse,
        );
        let mut drifted = valid.clone();
        drifted["assets"][3]["source"] = json!("file:///other.mp4");
        refuse(
            &serde_json::to_string(&drifted).expect("json"),
            Error::Reverse,
        );
        drifted = valid.clone();
        drifted["assets"][3]["duration"]["value"] = json!(1);
        refuse(
            &serde_json::to_string(&drifted).expect("json"),
            Error::Reverse,
        );

        let mut image = value.clone();
        image["assets"][2]["reverses"] = json!(forward);
        refuse(
            &serde_json::to_string(&image).expect("json"),
            Error::ImageSource,
        );

        let mut doubled = valid.clone();
        let mut second = doubled["assets"][0].clone();
        second["id"] = json!(format!("asset-{}", next + 1));
        second["reverses"] = json!(forward);
        doubled["assets"]
            .as_array_mut()
            .expect("assets")
            .push(second);
        doubled["next_id"] = json!(next + 2);
        refuse(
            &serde_json::to_string(&doubled).expect("json"),
            Error::Reverse,
        );

        let mut chained = valid;
        let mut further = chained["assets"][0].clone();
        further["id"] = json!(format!("asset-{}", next + 1));
        further["reverses"] = json!(format!("asset-{next}"));
        chained["assets"]
            .as_array_mut()
            .expect("assets")
            .push(further);
        chained["next_id"] = json!(next + 2);
        refuse(
            &serde_json::to_string(&chained).expect("json"),
            Error::Reverse,
        );
    }

    #[test]
    fn a_track_flag_is_left_out_when_it_is_off() {
        let project = Project::from_json(EMPTY).expect("load");
        assert!(!project.timeline().main_track().locked());
        assert!(!project.timeline().main_track().hidden());
        assert!(!project.timeline().main_track().muted());
        assert!(!project.to_json().expect("json").contains("\"locked\""));

        let marked = EMPTY.replace(
            "\"main\": true,",
            "\"main\": true, \"locked\": true, \"hidden\": true, \"muted\": true,",
        );
        let loaded = Project::from_json(&marked).expect("flags");
        assert!(loaded.timeline().main_track().locked());
        assert!(loaded.timeline().main_track().hidden());
        assert!(loaded.timeline().main_track().muted());
        let text = loaded.to_json().expect("json");
        assert!(text.contains("\"locked\": true"));
        assert!(text.contains("\"hidden\": true"));
        assert!(text.contains("\"muted\": true"));

        let missing = crate::id::TrackId::new(99).expect("id");
        let mut history = History::new();
        let mut project = loaded;
        let err = history
            .apply(
                &mut project,
                &[Command::SetLocked {
                    track: missing,
                    on: false,
                }],
            )
            .expect_err("missing track");
        assert_eq!(err, Error::UnknownTrack(missing));
        assert!(project.timeline().main_track().locked());
    }

    #[test]
    fn a_text_segment_has_words_and_no_file() {
        let mut project = Project::from_json(EMPTY).expect("load");
        let mut history = History::new();
        history
            .apply(
                &mut project,
                &[Command::Place {
                    track: None,
                    segment: Segment::text("Hi", 0, 1_000_000).expect("text"),
                }],
            )
            .expect("place");
        let saved = project.to_json().expect("json");
        assert!(saved.contains("\"text\": \"Hi\""));
        assert!(!saved.contains("\"asset\""));
        let mut value: Value = serde_json::from_str(&saved).expect("value");
        value["timeline"]["tracks"][1]["segments"][0]["asset"] = json!("asset-9");
        refuse(&value.to_string(), Error::Text);
        let mut bare = value;
        bare["timeline"]["tracks"][1]["segments"][0]
            .as_object_mut()
            .expect("segment")
            .remove("asset");
        bare["timeline"]["tracks"][1]["segments"][0]
            .as_object_mut()
            .expect("segment")
            .remove("text");
        refuse(&bare.to_string(), Error::Text);
    }

    #[test]
    fn a_sticker_segment_has_a_name_and_no_file() {
        let mut project = Project::from_json(EMPTY).expect("load");
        let mut history = History::new();
        history
            .apply(
                &mut project,
                &[Command::Place {
                    track: None,
                    segment: Segment::sticker("Star", 0, 1_000_000).expect("sticker"),
                }],
            )
            .expect("place");
        let saved = project.to_json().expect("json");
        assert!(saved.contains("\"sticker\": \"Star\""));
        assert!(!saved.contains("\"asset\""));
        assert!(!saved.contains("\"text\""));
        let loaded = Project::from_json(&saved).expect("load");
        assert_eq!(
            loaded.timeline().tracks()[1].segments()[0].sticker_name(),
            Some("Star")
        );
        let mut value: Value = serde_json::from_str(&saved).expect("value");
        value["timeline"]["tracks"][1]["segments"][0]["text"] = json!("Hi");
        refuse(&value.to_string(), Error::Text);
        value["timeline"]["tracks"][1]["segments"][0]
            .as_object_mut()
            .expect("segment")
            .remove("text");
        value["timeline"]["tracks"][1]["segments"][0]["asset"] = json!("asset-9");
        refuse(&value.to_string(), Error::Sticker);
        value["timeline"]["tracks"][1]["segments"][0]
            .as_object_mut()
            .expect("segment")
            .remove("asset");
        value["timeline"]["tracks"][1]["segments"][0]["sticker"] = json!("");
        refuse(&value.to_string(), Error::Sticker);
        value["timeline"]["tracks"][1]["segments"][0]["sticker"] = json!("Star");
        value["timeline"]["tracks"][1]["segments"][0]["source"] = json!({
            "start": 0,
            "duration": 1,
            "timescale": 30000
        });
        refuse(&value.to_string(), Error::Sticker);
    }

    #[test]
    fn an_effect_filter_or_adjustment_has_no_file() {
        let mut project = Project::from_json(EMPTY).expect("load");
        let mut history = History::new();
        history
            .apply(
                &mut project,
                &[Command::Place {
                    track: None,
                    segment: Segment::effect(0, 1_000_000).expect("effect"),
                }],
            )
            .expect("place");
        let saved = project.to_json().expect("json");
        assert!(saved.contains("\"effect\": true"));
        assert!(!saved.contains("\"filter\""));
        assert!(!saved.contains("\"adjust\""));
        assert!(!saved.contains("\"asset\""));
        let loaded = Project::from_json(&saved).expect("load");
        assert!(loaded.timeline().tracks()[1].segments()[0].is_effect());
        let mut value: Value = serde_json::from_str(&saved).expect("value");
        grade_field(&mut value, "filter", json!(true));
        refuse(&value.to_string(), Error::Effect);
        grade_field(&mut value, "filter", json!(false));
        grade_field(&mut value, "asset", json!("asset-9"));
        refuse(&value.to_string(), Error::Effect);
        value["timeline"]["tracks"][1]["segments"][0]
            .as_object_mut()
            .expect("segment")
            .remove("asset");
        grade_field(&mut value, "text", json!("Hi"));
        refuse(&value.to_string(), Error::Text);
        value["timeline"]["tracks"][1]["segments"][0]
            .as_object_mut()
            .expect("segment")
            .remove("text");
        grade_field(
            &mut value,
            "source",
            json!({
                "start": 0,
                "duration": 1,
                "timescale": 30000
            }),
        );
        refuse(&value.to_string(), Error::Effect);
        value["timeline"]["tracks"][1]["segments"][0]
            .as_object_mut()
            .expect("segment")
            .remove("source");
        value["timeline"]["tracks"][1]["segments"][0]["speed"]["numerator"] = json!(2);
        refuse(&value.to_string(), Error::Effect);
        value["timeline"]["tracks"][1]["segments"][0]["speed"]["numerator"] = json!(1);
        grade_field(&mut value, "channels", json!("sound"));
        refuse(&value.to_string(), Error::Channels);
        grade_field(&mut value, "channels", json!("picture"));
        grade_field(&mut value, "effect", json!(false));
        grade_field(&mut value, "filter", json!(true));
        grade_field(
            &mut value,
            "source",
            json!({
                "start": 0,
                "duration": 1,
                "timescale": 30000
            }),
        );
        refuse(&value.to_string(), Error::Filter);
        value["timeline"]["tracks"][1]["segments"][0]
            .as_object_mut()
            .expect("segment")
            .remove("source");
        grade_field(&mut value, "filter", json!(false));
        grade_field(&mut value, "adjust", json!(true));
        grade_field(&mut value, "frozen", json!(true));
        refuse(&value.to_string(), Error::Adjust);
    }

    #[test]
    fn an_attachment_round_trips_and_a_bad_one_refuses_the_file() {
        use crate::attachment::{Attachment, Parameter, ParameterValue};

        let mut project = Project::from_json(EMPTY).expect("load");
        let mut history = History::new();
        let placed = history
            .apply(
                &mut project,
                &[Command::Place {
                    track: None,
                    segment: Segment::effect(0, 1_000_000).expect("effect"),
                }],
            )
            .expect("place");
        let id = match placed.as_slice() {
            [crate::Outcome::Opened { segment, .. }] => *segment,
            other => panic!("expected a track, got {other:?}"),
        };
        history
            .apply(
                &mut project,
                &[Command::SetAttachment {
                    segment: id,
                    attachment: Some(
                        Attachment::new(
                            "blur",
                            "1",
                            vec![Parameter::new("amount", ParameterValue::Number(0.5))],
                        )
                        .expect("attachment"),
                    ),
                }],
            )
            .expect("attach");
        let saved = project.to_json().expect("json");
        assert!(saved.contains("\"plugin\": \"blur\""));
        let loaded = Project::from_json(&saved).expect("load");
        assert_eq!(
            loaded.timeline().tracks()[1].segments()[0]
                .attachment()
                .expect("attachment")
                .plugin(),
            "blur"
        );
        let mut value: Value = serde_json::from_str(&saved).expect("value");
        value["timeline"]["tracks"][1]["segments"][0]["attachment"]["version"] = json!("");
        refuse(&value.to_string(), Error::Plugin);
        value["timeline"]["tracks"][1]["segments"][0]["attachment"]["version"] = json!("1");
        value["timeline"]["tracks"][1]["segments"][0]["attachment"]["parameters"]
            .as_array_mut()
            .expect("parameters")
            .push(json!({
                "name": "amount",
                "value": true
            }));
        refuse(&value.to_string(), Error::Parameter);
        value["timeline"]["tracks"][1]["segments"][0]["effect"] = json!(false);
        value["timeline"]["tracks"][1]["segments"][0]["asset"] = json!("asset-9");
        refuse(&value.to_string(), Error::Attachment);
    }

    #[test]
    fn a_transition_round_trips_and_a_bad_cut_refuses_the_file() {
        use crate::attachment::Attachment;

        let mut project = Project::from_json(EMPTY).expect("load");
        let mut history = History::new();
        let mut ids = Vec::new();
        for (index, name) in ["file:///a.png", "file:///b.png"].into_iter().enumerate() {
            let asset = Asset::image(name, 8, 8).expect("image");
            history
                .apply(&mut project, &[Command::AddAsset(asset)])
                .expect("add");
            let stored = project.assets().last().expect("asset");
            let start = i64::try_from(index).expect("index") * 5_000_000;
            let segment = Segment::image(stored, start, 1_000_000).expect("still");
            let main = project.timeline().main_track().id();
            let placed = history
                .apply(
                    &mut project,
                    &[Command::Place {
                        track: Some(main),
                        segment,
                    }],
                )
                .expect("place");
            ids.push(match placed.as_slice() {
                [crate::Outcome::Segment(id)] => *id,
                other => panic!("expected the main track, got {other:?}"),
            });
        }
        history
            .apply(
                &mut project,
                &[Command::SetTransition {
                    segment: ids[0],
                    duration_us: 400_000,
                    overlap: false,
                    attachment: Attachment::new("crossfade", "1", Vec::new()).expect("plugin"),
                }],
            )
            .expect("transition");
        let saved = project.to_json().expect("json");
        assert!(saved.contains("\"duration\": 400000"));
        assert!(!saved.contains("\"overlap\""));
        let loaded = Project::from_json(&saved).expect("load");
        assert_eq!(
            loaded.timeline().main_track().transitions()[0].left(),
            ids[0]
        );
        let mut value: Value = serde_json::from_str(&saved).expect("value");
        value["timeline"]["tracks"][0]["transitions"][0]["duration"] = json!(9_000_000);
        refuse(&value.to_string(), Error::Transition);
        value["timeline"]["tracks"][0]["transitions"][0]["duration"] = json!(400_000);
        value["timeline"]["tracks"][0]["segments"][1]["timeline"]["start"] = json!(2_000_000);
        refuse(&value.to_string(), Error::Transition);
    }

    #[test]
    fn an_animation_round_trips_and_a_bad_one_refuses_the_file() {
        use crate::animation::{Animation, AnimationKind};

        let mut project = Project::from_json(EMPTY).expect("load");
        let mut history = History::new();
        let asset = Asset::image("file:///a.png", 8, 8).expect("image");
        history
            .apply(&mut project, &[Command::AddAsset(asset)])
            .expect("add");
        let stored = project.assets().last().expect("asset");
        let segment = Segment::image(stored, 0, 1_000_000).expect("still");
        let main = project.timeline().main_track().id();
        let placed = history
            .apply(
                &mut project,
                &[Command::Place {
                    track: Some(main),
                    segment,
                }],
            )
            .expect("place");
        let id = match placed.as_slice() {
            [crate::Outcome::Segment(id)] => *id,
            other => panic!("expected the main track, got {other:?}"),
        };
        history
            .apply(
                &mut project,
                &[Command::SetAnimation {
                    segment: id,
                    kind: AnimationKind::In,
                    animation: Some(Animation::new("rise", 400_000).expect("animation")),
                }],
            )
            .expect("animation");
        let saved = project.to_json().expect("json");
        assert!(saved.contains("\"kind\": \"in\""));
        let loaded = Project::from_json(&saved).expect("load");
        assert_eq!(
            loaded.timeline().main_track().segments()[0]
                .animation(AnimationKind::In)
                .expect("animation")
                .name(),
            "rise"
        );
        let mut value: Value = serde_json::from_str(&saved).expect("value");
        value["timeline"]["tracks"][0]["segments"][0]["animations"][0]["duration"] =
            json!(2_000_000);
        refuse(&value.to_string(), Error::Animation);
        value["timeline"]["tracks"][0]["segments"][0]["animations"][0]["duration"] = json!(400_000);
        let repeat = value["timeline"]["tracks"][0]["segments"][0]["animations"][0].clone();
        value["timeline"]["tracks"][0]["segments"][0]["animations"]
            .as_array_mut()
            .expect("animations")
            .push(repeat);
        refuse(&value.to_string(), Error::Animation);
    }

    #[test]
    fn a_mark_round_trips_and_a_bad_one_refuses_the_file() {
        let mut project = Project::from_json(EMPTY).expect("load");
        let mut history = History::new();
        let image = Asset::image("file:///a.png", 8, 8).expect("image");
        history
            .apply(&mut project, &[Command::AddAsset(image)])
            .expect("add");
        let stored = project.assets().last().expect("asset");
        let still = Segment::image(stored, 0, 1_000_000).expect("still");
        let main = project.timeline().main_track().id();
        let picture = match history
            .apply(
                &mut project,
                &[Command::Place {
                    track: Some(main),
                    segment: still,
                }],
            )
            .expect("place")
            .as_slice()
        {
            [Outcome::Segment(id)] => *id,
            other => panic!("expected the main track, got {other:?}"),
        };
        let scale = NonZeroU32::new(48_000).expect("scale");
        let audio = Asset::audio("file:///a.wav", Timestamp::new(48_000, scale)).expect("audio");
        history
            .apply(&mut project, &[Command::AddAsset(audio)])
            .expect("add");
        let stored = project.assets().last().expect("asset");
        let duration = stored.duration().expect("duration");
        let source =
            TimeRange::new(Timestamp::new(0, duration.timescale), duration).expect("source");
        let sound = Segment::media(stored, source, 0, Channels::Sound, Speed::ONE).expect("sound");
        let audio_id = match history
            .apply(
                &mut project,
                &[Command::Place {
                    track: None,
                    segment: sound,
                }],
            )
            .expect("audio")
            .as_slice()
        {
            [Outcome::Opened { segment, .. }] => *segment,
            other => panic!("expected an audio track, got {other:?}"),
        };
        history
            .apply(
                &mut project,
                &[
                    Command::AddMarker { time_us: 1_000_000 },
                    Command::AddClipMarker {
                        segment: picture,
                        offset_us: 100,
                    },
                    Command::AddBeat {
                        segment: audio_id,
                        offset_us: 100,
                    },
                ],
            )
            .expect("marks");
        let saved = project.to_json().expect("json");
        let loaded = Project::from_json(&saved).expect("load");
        assert_eq!(loaded.timeline().markers(), &[1_000_000]);
        assert_eq!(
            loaded.timeline().main_track().segments()[0].markers(),
            &[100]
        );
        assert_eq!(loaded.timeline().tracks()[0].segments()[0].beats(), &[100]);
        let value: Value = serde_json::from_str(&saved).expect("value");
        refuse(
            &edit(&value, |value| {
                value["timeline"]["markers"]
                    .as_array_mut()
                    .expect("markers")
                    .push(json!(1_000_000));
            }),
            Error::Marker,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["markers"][0] = json!(9_000_000);
            }),
            Error::Marker,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["markers"][0] = json!(-1);
            }),
            Error::NegativeTime,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][1]["segments"][0]["beats"] = json!([100]);
            }),
            Error::Marker,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][1]["segments"][0]["markers"] = json!([1_000_000]);
            }),
            Error::Marker,
        );
        let mut empty: Value = serde_json::from_str(EMPTY).expect("empty");
        empty["timeline"]["markers"] = json!([0]);
        refuse(&empty.to_string(), Error::Marker);
    }

    #[test]
    fn a_speed_curve_round_trips_and_a_bad_one_refuses_the_file() {
        let (_project, value) = cut_value();
        let speed = &value["timeline"]["tracks"][1]["segments"][0]["speed"];
        assert_eq!(speed["numerator"], json!(2));
        assert!(speed.get("points").is_none());
        let source = &value["timeline"]["tracks"][1]["segments"][0]["source"];
        let ticks = source["duration"].as_i64().expect("ticks");
        let scale_value = source["timescale"].as_u64().expect("scale");
        assert_eq!(ticks, 30_000);
        assert_eq!(scale_value, 30_000);
        let scale = NonZeroU32::new(u32::try_from(scale_value).expect("scale")).expect("scale");
        let curve = SpeedCurve::new(vec![
            SpeedPoint::new(0, 1, Speed::ONE).expect("start"),
            SpeedPoint::new(1, 1, Speed::new(2, 1).expect("fast")).expect("end"),
        ])
        .expect("curve");
        let span = curve
            .timeline_span(Timestamp::new(ticks, scale))
            .expect("span");
        assert_eq!(span.value, 693_147);

        let text = edit(&value, |value| {
            let segment = &mut value["timeline"]["tracks"][1]["segments"][0];
            segment["speed"] = ramp_points();
            segment["timeline"]["duration"] = json!(span.value);
        });
        let loaded = Project::from_json(&text).expect("curve");
        let segment = &loaded.timeline().main_track().segments()[0];
        assert_eq!(segment.curve(), Some(&curve));
        assert_eq!(segment.speed(), Speed::ONE);
        assert_eq!(segment.timeline().duration().value, 693_147);
        let saved = loaded.to_json().expect("save");
        let again: Value = serde_json::from_str(&saved).expect("json");
        let stored = &again["timeline"]["tracks"][1]["segments"][0]["speed"];
        assert!(stored.get("numerator").is_none());
        assert!(stored.get("denominator").is_none());
        assert_eq!(stored["points"].as_array().expect("points").len(), 2);
        assert_eq!(
            Project::from_json(&saved)
                .expect("reload")
                .to_json()
                .expect("again"),
            saved
        );

        refuse(
            &edit(&value, |value| {
                let segment = &mut value["timeline"]["tracks"][1]["segments"][0];
                segment["speed"] = ramp_points();
            }),
            Error::TimelineDuration {
                found: 500_000,
                expected: 693_147,
            },
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][1]["segments"][0]["speed"] = rated_ramp();
                value["timeline"]["tracks"][1]["segments"][0]["timeline"]["duration"] =
                    json!(span.value);
            }),
            Error::Curve,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][1]["segments"][0]["speed"] = json!({});
            }),
            Error::Curve,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][1]["segments"][0]["speed"] = json!({
                    "points": [{
                        "at": [0, 1],
                        "speed": { "numerator": 1, "denominator": 1 }
                    }]
                });
            }),
            Error::Curve,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][1]["segments"][0]["speed"] = json!({
                    "points": [
                        { "at": [0, 1], "speed": { "numerator": 1, "denominator": 1 } },
                        { "at": [2, 1], "speed": { "numerator": 2, "denominator": 1 } }
                    ]
                });
            }),
            Error::Curve,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][1]["segments"][0]["speed"] = json!({
                    "points": [
                        { "at": [0, 1], "speed": { "numerator": 1, "denominator": 1 } },
                        { "at": [1, 1], "speed": { "numerator": 0, "denominator": 1 } }
                    ]
                });
            }),
            Error::ZeroSpeed,
        );
        refuse(
            &edit(&value, |value| {
                let segment = &mut value["timeline"]["tracks"][1]["segments"][0];
                segment["frozen"] = json!(true);
                segment["speed"] = ramp_points();
                segment["timeline"]["duration"] = json!(span.value);
            }),
            Error::Freeze,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][2]["segments"][0]["speed"] = ramp_points();
            }),
            Error::ImageSpeed,
        );
        refuse(
            &edit(&value, |value| {
                value["timeline"]["tracks"][2]["segments"][0]["speed"] = rated_ramp();
            }),
            Error::Curve,
        );

        let mut history = History::new();
        let mut project = Project::new(sample_id(), "Untitled");
        let words = Segment::text("hi", 0, 1_000_000).expect("text");
        history
            .apply(
                &mut project,
                &[Command::Place {
                    track: None,
                    segment: words,
                }],
            )
            .expect("place");
        let caption: Value = serde_json::from_str(&project.to_json().expect("json")).expect("text");
        let text_track = caption["timeline"]["tracks"]
            .as_array()
            .expect("tracks")
            .iter()
            .position(|track| track["kind"] == json!("text"))
            .expect("text track");
        refuse(
            &edit(&caption, |value| {
                value["timeline"]["tracks"][text_track]["segments"][0]["speed"] = ramp_points();
            }),
            Error::Text,
        );
        refuse(
            &edit(&caption, |value| {
                value["timeline"]["tracks"][text_track]["segments"][0]["speed"] = rated_ramp();
            }),
            Error::Curve,
        );
    }

    fn ramp_points() -> Value {
        json!({
            "points": [
                { "at": [0, 1], "speed": { "numerator": 1, "denominator": 1 } },
                { "at": [1, 1], "speed": { "numerator": 2, "denominator": 1 } }
            ]
        })
    }

    fn rated_ramp() -> Value {
        json!({
            "numerator": 1,
            "denominator": 1,
            "points": [
                { "at": [0, 1], "speed": { "numerator": 1, "denominator": 1 } },
                { "at": [1, 1], "speed": { "numerator": 2, "denominator": 1 } }
            ]
        })
    }

    fn grade_field(value: &mut Value, key: &str, next: Value) {
        value["timeline"]["tracks"][1]["segments"][0][key] = next;
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
            Some(image_id)
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

    fn put(value: &mut Value, key: &str, next: Value) {
        value
            .as_object_mut()
            .expect("object")
            .insert(key.to_string(), next);
    }

    fn nested_project(timelines: usize) -> Project {
        let mut history = History::new();
        let mut project = Project::new(sample_id(), "Nest");
        let scale = NonZeroU32::new(30_000).expect("scale");
        history
            .apply(
                &mut project,
                &[Command::AddAsset(
                    Asset::video(
                        "file:///nest.mp4",
                        Timestamp::new(60_000, scale),
                        16,
                        16,
                        true,
                    )
                    .expect("video"),
                )],
            )
            .expect("asset");
        let asset = project.assets()[0].id();
        let source = TimeRange::new(Timestamp::new(0, scale), Timestamp::new(60_000, scale))
            .expect("source");
        let segment = Segment::media(
            project.asset(asset).expect("stored"),
            source,
            0,
            Channels::Both,
            Speed::ONE,
        )
        .expect("segment");
        let main = project.timeline().main_track().id();
        history
            .apply(
                &mut project,
                &[Command::Place {
                    track: Some(main),
                    segment,
                }],
            )
            .expect("place");
        for _ in 0..timelines {
            history
                .apply(&mut project, &[Command::AddTimeline])
                .expect("timeline");
        }
        project
    }

    fn value_of(project: &Project) -> Value {
        serde_json::from_str(&project.to_json().expect("json")).expect("value")
    }

    #[test]
    fn a_nested_timeline_round_trips_and_a_cycle_refuses_the_file() {
        let project = nested_project(2);
        let mut linked = value_of(&project);
        let clip = linked["timeline"]["tracks"][0]["segments"][0].clone();
        put(
            &mut linked["timeline"]["tracks"][0]["segments"][0],
            "nest",
            json!("timeline-5"),
        );
        let mut inside = clip.clone();
        put(&mut inside, "id", json!("segment-9"));
        put(&mut inside, "nest", json!("timeline-7"));
        linked["nests"][0]["tracks"][0]["segments"] = json!([inside]);
        linked["next_id"] = json!(10);
        let loaded =
            Project::from_json(&serde_json::to_string(&linked).expect("text")).expect("load");
        assert_eq!(
            loaded.timeline().main_track().segments()[0]
                .nest()
                .expect("name")
                .to_string(),
            "timeline-5"
        );
        assert_eq!(
            loaded.nests()[0].main_track().segments()[0]
                .nest()
                .expect("inner")
                .to_string(),
            "timeline-7"
        );
        assert!(loaded.nests()[1].main_track().segments().is_empty());

        let mut cycle = value_of(&project);
        let clip = cycle["timeline"]["tracks"][0]["segments"][0].clone();
        let mut left = clip.clone();
        put(&mut left, "id", json!("segment-9"));
        put(&mut left, "nest", json!("timeline-7"));
        let mut right = clip;
        put(&mut right, "id", json!("segment-10"));
        put(&mut right, "nest", json!("timeline-5"));
        cycle["nests"][0]["tracks"][0]["segments"] = json!([left]);
        cycle["nests"][1]["tracks"][0]["segments"] = json!([right]);
        cycle["next_id"] = json!(11);
        refuse(
            &serde_json::to_string(&cycle).expect("cycle"),
            Error::NestCycle,
        );

        let mut itself = value_of(&project);
        let mut clip = itself["timeline"]["tracks"][0]["segments"][0].clone();
        put(&mut clip, "id", json!("segment-9"));
        put(&mut clip, "nest", json!("timeline-5"));
        itself["nests"][0]["tracks"][0]["segments"] = json!([clip]);
        itself["next_id"] = json!(10);
        refuse(
            &serde_json::to_string(&itself).expect("itself"),
            Error::NestCycle,
        );

        let mut sequence = value_of(&project);
        let mut clip = sequence["timeline"]["tracks"][0]["segments"][0].clone();
        put(&mut clip, "id", json!("segment-9"));
        put(&mut clip, "nest", json!("timeline-1"));
        sequence["nests"][0]["tracks"][0]["segments"] = json!([clip]);
        sequence["next_id"] = json!(10);
        refuse(
            &serde_json::to_string(&sequence).expect("sequence"),
            Error::SequenceNest,
        );

        let missing = TimelineId::new(99).expect("id");
        let mut unknown = value_of(&project);
        put(
            &mut unknown["timeline"]["tracks"][0]["segments"][0],
            "nest",
            json!("timeline-99"),
        );
        refuse(
            &serde_json::to_string(&unknown).expect("unknown"),
            Error::UnknownTimeline(missing),
        );

        let mut frozen = value_of(&project);
        put(
            &mut frozen["timeline"]["tracks"][0]["segments"][0],
            "frozen",
            json!(true),
        );
        put(
            &mut frozen["timeline"]["tracks"][0]["segments"][0],
            "nest",
            json!("timeline-5"),
        );
        refuse(
            &serde_json::to_string(&frozen).expect("frozen"),
            Error::Freeze,
        );

        let mut only = value_of(&nested_project(1));
        let clip = only["timeline"]["tracks"][0]["segments"][0].clone();
        only["timeline"]["tracks"][0]["segments"] = json!([]);
        only["nests"][0]["tracks"][0]["segments"] = json!([clip]);
        let mut held =
            Project::from_json(&serde_json::to_string(&only).expect("held")).expect("held");
        let asset = held.assets()[0].id();
        assert_eq!(
            History::new()
                .apply(&mut held, &[Command::RemoveAsset(asset)])
                .expect_err("used"),
            Error::AssetInUse
        );
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
