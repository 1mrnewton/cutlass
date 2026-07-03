//! CapCut-style templates.
//!
//! A [`Template`] is a *finished* [`Project`] — every transition, effect,
//! animation, text layer, music track, and beat-synced cut baked in and
//! locked — in which the author has marked specific clips as user-replaceable
//! ([`Replaceable`](crate::Replaceable)). This mirrors CapCut's "set
//! replaceable material clips" + "use template" flow:
//!
//! - A replaceable clip keeps its **sample media**, so the template previews
//!   exactly like the author's video.
//! - [`apply`](Template::apply) fills the slots **in order** (the sequence the
//!   user/agent picks media in) by swapping each slot's media while its locked
//!   timeline duration, transform, effects, transitions, and animations are
//!   preserved.
//! - The customization surface ([`slots`](Template::slots),
//!   [`editable_texts`](Template::editable_texts), [`music`](Template::music))
//!   is *derived* by scanning the markers — there is no second manifest to keep
//!   in sync with the timeline.
//!
//! The result of `apply` is an ordinary [`Project`] that the existing
//! compositor and exporter render unchanged.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::clip::{Clip, Generator, SlotMedia};
use crate::error::ModelError;
use crate::ids::{ProjectId, TemplateId};
use crate::media::MediaSource;
use crate::project::Project;
use crate::schema::{PROJECT_SCHEMA_VERSION, ProjectSchema};
use crate::time::{Rational, RationalTime, TimeRange, resample};

/// Stable format family for a Cutlass template document.
pub const TEMPLATE_SCHEMA_KIND: &str = "cutlass.template";

/// Recommended extension for Cutlass template files.
pub const TEMPLATE_FILE_EXTENSION: &str = "cutlasst";

/// Coarse browse category for a template, mirroring the CapCut gallery tabs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemplateCategory {
    #[default]
    Trending,
    Travel,
    Lifestyle,
    Birthday,
    Wedding,
    Slideshow,
    Vlog,
    Gaming,
    Business,
    Education,
    Other,
}

/// User-facing metadata for a [`Template`] (everything the gallery shows that
/// is not the timeline itself).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateMeta {
    /// Display name.
    pub name: String,
    /// Optional creator label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// Browse category.
    #[serde(default)]
    pub category: TemplateCategory,
    /// Free-form search tags.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Timeline tick used as the cover/preview frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover: Option<RationalTime>,
    /// Optional longer description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl TemplateMeta {
    /// Metadata with just a name; everything else defaulted.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            author: None,
            category: TemplateCategory::default(),
            tags: Vec::new(),
            cover: None,
            description: None,
        }
    }

    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    pub fn with_category(mut self, category: TemplateCategory) -> Self {
        self.category = category;
        self
    }

    pub fn with_tags(mut self, tags: impl IntoIterator<Item = String>) -> Self {
        self.tags = tags.into_iter().collect();
        self
    }

    pub fn with_cover(mut self, cover: RationalTime) -> Self {
        self.cover = Some(cover);
        self
    }
}

/// One media choice supplied to [`Template::apply`], in slot order.
///
/// `source_in` is the in-point into the chosen media (at its native rate);
/// `None` starts from the beginning, matching CapCut ("takes from the start of
/// the clip"). The slot's locked timeline duration determines how much source
/// is drawn.
#[derive(Debug, Clone)]
pub struct Pick {
    pub media: MediaSource,
    pub source_in: Option<RationalTime>,
}

impl Pick {
    /// Fill a slot from the start of `media`.
    pub fn new(media: MediaSource) -> Self {
        Self {
            media,
            source_in: None,
        }
    }

    /// Fill a slot from a specific in-point in `media`.
    pub fn at(media: MediaSource, source_in: RationalTime) -> Self {
        Self {
            media,
            source_in: Some(source_in),
        }
    }
}

/// A CapCut-style template: a finished [`Project`] plus the markers that say
/// which clips a user may fill or edit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Template {
    /// Template document schema (kind [`TEMPLATE_SCHEMA_KIND`]).
    pub schema: ProjectSchema,
    pub id: TemplateId,
    pub meta: TemplateMeta,
    /// The finished timeline, with sample media in its pool and replaceable
    /// clips marked.
    pub project: Project,
}

impl Template {
    /// Author a template from a finished [`Project`]. Mark the slots first with
    /// [`Project::set_replaceable`] / [`Project::set_text_editable`].
    pub fn from_project(project: Project, meta: TemplateMeta) -> Self {
        Self {
            schema: Self::current_schema(),
            id: TemplateId::next(),
            meta,
            project,
        }
    }

    /// The finished, sample-filled project (what a preview renders).
    pub fn project(&self) -> &Project {
        &self.project
    }

    pub fn meta(&self) -> &TemplateMeta {
        &self.meta
    }

    /// Total timeline length.
    pub fn duration(&self) -> RationalTime {
        self.project.timeline().duration()
    }

    /// The visual fill slots in fill order (replaceable clips that take video
    /// or image media). Excludes the music slot, which is swapped separately.
    pub fn slots(&self) -> Vec<&Clip> {
        let mut slots: Vec<&Clip> = self
            .replaceable_clips()
            .filter(|clip| {
                clip.replaceable
                    .as_ref()
                    .is_some_and(|r| r.accepts != SlotMedia::AudioOnly)
            })
            .collect();
        slots.sort_by_key(|clip| {
            let order = clip.replaceable.as_ref().map_or(u32::MAX, |r| r.order);
            (order, clip.id.raw())
        });
        slots
    }

    /// How many visual slots a user must fill.
    pub fn slot_count(&self) -> usize {
        self.replaceable_clips()
            .filter(|clip| {
                clip.replaceable
                    .as_ref()
                    .is_some_and(|r| r.accepts != SlotMedia::AudioOnly)
            })
            .count()
    }

    /// The swappable music/soundtrack clip, if the author marked one
    /// (`accepts == AudioOnly`). Swap it post-apply with
    /// [`Project::set_clip_media`].
    pub fn music(&self) -> Option<&Clip> {
        self.replaceable_clips()
            .filter(|clip| {
                clip.replaceable
                    .as_ref()
                    .is_some_and(|r| r.accepts == SlotMedia::AudioOnly)
            })
            .min_by_key(|clip| clip.replaceable.as_ref().map_or(u32::MAX, |r| r.order))
    }

    /// Text clips a user may re-word (the text keeps its style and animation),
    /// ordered by timeline position.
    pub fn editable_texts(&self) -> Vec<&Clip> {
        let mut texts: Vec<&Clip> = self
            .project
            .timeline()
            .tracks_ordered()
            .flat_map(|track| track.clips())
            .filter(|clip| {
                clip.text_editable
                    && matches!(clip.content, crate::ClipSource::Generated(Generator::Text { .. }))
            })
            .collect();
        texts.sort_by_key(|clip| (clip.timeline.start.value, clip.id.raw()));
        texts
    }

    /// Fill the visual slots in order, returning a normal, render-ready
    /// [`Project`]. Each pick (in slot order) swaps its slot's media while the
    /// slot's locked duration, transform, effects, transitions, and animations
    /// are preserved.
    ///
    /// Supplying fewer picks than slots is allowed: the remaining slots keep
    /// their sample media, exactly like a CapCut template preview. Each pick is
    /// validated against the slot's media-type restriction and — for non-image
    /// media — that enough source exists to cover the slot's locked duration
    /// from the chosen in-point.
    pub fn apply(&self, picks: &[Pick]) -> Result<Project, ModelError> {
        let mut out = self.project.clone();
        out.id = ProjectId::next();
        let tl_rate = out.timeline().frame_rate;

        // Resolve every slot up front (immutable borrow of the template) before
        // mutating the cloned output, so the slot ids and geometry are stable.
        for (slot, pick) in self.slots().into_iter().zip(picks.iter()) {
            let accepts = slot
                .replaceable
                .as_ref()
                .map_or(SlotMedia::Any, |r| r.accepts);
            let kind = pick.media.kind();
            if !accepts.accepts(kind) {
                return Err(ModelError::SlotMediaMismatch {
                    slot: slot.id,
                    accepts,
                    found: kind,
                });
            }
            let source = slot_source(slot, tl_rate, pick.source_in, &pick.media)?;
            let media_id = out.add_media(pick.media.clone());
            out.set_clip_media(slot.id, media_id, source)?;
        }
        Ok(out)
    }

    fn replaceable_clips(&self) -> impl Iterator<Item = &Clip> {
        self.project
            .timeline()
            .tracks_ordered()
            .flat_map(|track| track.clips())
            .filter(|clip| clip.replaceable.is_some())
    }

    fn current_schema() -> ProjectSchema {
        ProjectSchema {
            version: PROJECT_SCHEMA_VERSION,
            kind: TEMPLATE_SCHEMA_KIND.into(),
            extensions: Vec::new(),
        }
    }

    /// Serialize this template to a `.cutlasst` JSON file.
    pub fn save_to_file(&self, path: &Path) -> io::Result<()> {
        let mut doc = self.clone();
        doc.schema = Self::current_schema();
        let json = serde_json::to_string_pretty(&doc).map_err(io::Error::other)?;
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(json.as_bytes())?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        Ok(())
    }

    /// Deserialize a template from a `.cutlasst` JSON file. Files that are not
    /// templates, or that are newer than this build, are refused.
    pub fn load_from_file(path: &Path) -> Result<Template, ModelError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| ModelError::InvalidProjectFile(e.to_string()))?;
        let template: Template = serde_json::from_str(&text)
            .map_err(|e| ModelError::InvalidProjectFile(e.to_string()))?;
        if template.schema.kind != TEMPLATE_SCHEMA_KIND
            || template.schema.version < 1
            || template.schema.version > PROJECT_SCHEMA_VERSION
        {
            return Err(ModelError::UnsupportedProjectSchema {
                found: template.schema.clone(),
                expected: Self::current_schema(),
            });
        }
        Ok(template)
    }
}

/// The source window a fill draws for `slot`: the slot's *locked* timeline
/// duration, scaled by the slot's base speed and resampled into the chosen
/// media's rate, starting at `source_in` (default 0). For non-image media the
/// window must lie within the source; images repeat one frame for any length.
fn slot_source(
    slot: &Clip,
    tl_rate: Rational,
    source_in: Option<RationalTime>,
    media: &MediaSource,
) -> Result<TimeRange, ModelError> {
    let tl_dur = slot.timeline.duration.value;
    let speed = slot.speed;
    let den = i128::from(speed.den).max(1);
    let src_ticks_tl = (i128::from(tl_dur) * i128::from(speed.num) / den) as i64;
    let src_dur = resample(RationalTime::new(src_ticks_tl, tl_rate), media.frame_rate)
        .value
        .max(1);
    let start = match source_in {
        Some(rt) => resample(rt, media.frame_rate).value,
        None => 0,
    };
    if start < 0 || (!media.is_image && start + src_dur > media.duration.value) {
        return Err(ModelError::SlotDurationUnmet { slot: slot.id });
    }
    Ok(TimeRange::at_rate(start, src_dur, media.frame_rate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::Replaceable;
    use crate::ids::ClipId;
    use crate::time::Rational;
    use crate::track::TrackKind;
    use crate::{Generator, MediaSource, Project};

    const R24: Rational = Rational::FPS_24;

    /// Build a tiny 2-slot template: two 1s video slots on a video track plus a
    /// locked title on a text track.
    fn sample_template() -> (Template, ClipId, ClipId) {
        let mut project = Project::new("Trendy Intro", R24);
        let sample = project.add_media(MediaSource::new("sample.mp4", 1920, 1080, R24, 240, true));
        let video = project.add_track(TrackKind::Video, "V1");
        let text = project.add_track(TrackKind::Text, "T1");

        let slot_a = project
            .add_clip(video, sample, TimeRange::at_rate(0, 24, R24), RationalTime::new(0, R24))
            .unwrap();
        let slot_b = project
            .add_clip(video, sample, TimeRange::at_rate(0, 24, R24), RationalTime::new(24, R24))
            .unwrap();
        let title = project
            .add_generated(
                text,
                Generator::text("Your Name"),
                TimeRange::at_rate(0, 48, R24),
            )
            .unwrap();

        project
            .set_replaceable(slot_b, Some(Replaceable::new(1)))
            .unwrap();
        project
            .set_replaceable(slot_a, Some(Replaceable::new(0)))
            .unwrap();
        project.set_text_editable(title, true).unwrap();

        (
            Template::from_project(project, TemplateMeta::new("Trendy Intro")),
            slot_a,
            slot_b,
        )
    }

    #[test]
    fn slots_are_listed_in_fill_order() {
        let (template, slot_a, slot_b) = sample_template();
        let slots: Vec<ClipId> = template.slots().iter().map(|c| c.id).collect();
        assert_eq!(template.slot_count(), 2);
        assert_eq!(slots, vec![slot_a, slot_b]);
        assert_eq!(template.editable_texts().len(), 1);
    }

    #[test]
    fn apply_fills_slots_at_locked_durations() {
        let (template, slot_a, slot_b) = sample_template();
        let picks = vec![
            Pick::new(MediaSource::new("a.mp4", 1920, 1080, R24, 120, true)),
            Pick::new(MediaSource::new("b.mp4", 1280, 720, R24, 90, true)),
        ];
        let out = template.apply(&picks).unwrap();

        // The slots now point at the picked media, windowed to the locked 1s.
        for (slot, name) in [(slot_a, "a.mp4"), (slot_b, "b.mp4")] {
            let clip = out.clip(slot).unwrap();
            match &clip.content {
                crate::ClipSource::Media { media, source } => {
                    assert_eq!(source.duration.value, 24, "locked 1s slot");
                    assert_eq!(out.media(*media).unwrap().path().to_str().unwrap(), name);
                }
                other => panic!("slot not filled: {other:?}"),
            }
        }
    }

    #[test]
    fn fewer_picks_keep_sample_media() {
        let (template, slot_a, slot_b) = sample_template();
        let sample_media = match &template.project().clip(slot_a).unwrap().content {
            crate::ClipSource::Media { media, .. } => *media,
            _ => unreachable!(),
        };
        let out = template
            .apply(&[Pick::new(MediaSource::new("a.mp4", 1920, 1080, R24, 120, true))])
            .unwrap();
        // slot_b was not picked: it keeps the author's sample media.
        match &out.clip(slot_b).unwrap().content {
            crate::ClipSource::Media { media, .. } => assert_eq!(*media, sample_media),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn apply_rejects_wrong_media_kind() {
        let mut project = Project::new("Photo Slot", R24);
        let sample = project.add_media(MediaSource::image("sample.png", 800, 600));
        let track = project.add_track(TrackKind::Video, "V1");
        let slot = project
            .add_clip(
                track,
                sample,
                TimeRange::at_rate(0, STILL_TICKS, crate::media::STILL_TICK_RATE),
                RationalTime::new(0, R24),
            )
            .unwrap();
        project
            .set_replaceable(slot, Some(Replaceable::new(0).with_accepts(SlotMedia::ImageOnly)))
            .unwrap();
        let template = Template::from_project(project, TemplateMeta::new("Photo Slot"));

        let video = Pick::new(MediaSource::new("clip.mp4", 1920, 1080, R24, 120, true));
        assert!(matches!(
            template.apply(&[video]),
            Err(ModelError::SlotMediaMismatch { .. })
        ));
    }

    #[test]
    fn apply_rejects_media_too_short_for_slot() {
        let (template, _, _) = sample_template();
        // A 1s slot needs 24 frames; this clip only has 10.
        let tiny = Pick::new(MediaSource::new("tiny.mp4", 1920, 1080, R24, 10, true));
        assert!(matches!(
            template.apply(&[tiny]),
            Err(ModelError::SlotDurationUnmet { .. })
        ));
    }

    #[test]
    fn template_file_roundtrips() {
        let (template, _, _) = sample_template();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("intro.cutlasst");
        template.save_to_file(&path).unwrap();
        let loaded = Template::load_from_file(&path).unwrap();
        assert_eq!(loaded.schema.kind, TEMPLATE_SCHEMA_KIND);
        assert_eq!(loaded.slot_count(), template.slot_count());
        assert_eq!(loaded.meta.name, "Trendy Intro");
    }

    const STILL_TICKS: i64 = crate::media::STILL_DEFAULT_DURATION_TICKS;

    #[test]
    fn load_rejects_non_template_document() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not.cutlasst");
        let project = Project::new("plain", R24);
        project.save_to_file(&path).unwrap();
        assert!(matches!(
            Template::load_from_file(&path),
            Err(ModelError::InvalidProjectFile(_)) | Err(ModelError::UnsupportedProjectSchema { .. })
        ));
    }
}
