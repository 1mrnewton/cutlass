//! Project templates on the desktop: marking clips, saving the open
//! project as a template, and using one.
//!
//! A template in use is one closed clip. It is never opened like a
//! compound clip: its slots, texts and music are reached through the
//! clip, and each change is stored back on it as one undo step.

use std::path::{Path, PathBuf};

use cutlass_editor::{
    Append, AppendLane, DropGesture, DropOutcome, MediaPlace, Pointer, TEMPLATE_CARD_PREFIX,
    TemplateFill, append_placement, drop_gesture, template_clip_fill_requests,
    template_clip_requests, template_inside_requests,
};
use cutlass_model::{
    Draft, MaterialAudio, MaterialRef, MaterialVideo, Micros, SegmentId, TemplateMark,
    TemplateSlot, TrackId, TrackType, template_clip, template_music, template_slots,
    template_texts, template_unfilled,
};
use cutlass_session::{ImportedMedium, Request};
use cutlass_store::ListedTemplate;

use super::{Shell, ShellError, locate, place_in_project, probe};

/// One editable text of a template, with its words now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemplateText {
    pub segment_id: SegmentId,
    pub words: String,
}

/// One slot of a template clip, with the file it shows now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotView {
    pub slot: TemplateSlot,
    /// The media in the slot: the sample, or what the person put there.
    pub file: Option<PathBuf>,
    /// The material in the slot, for its thumbnail.
    pub material_id: String,
}

/// A template clip on the timeline, as the window shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct TemplateClip {
    pub segment_id: SegmentId,
    pub name: String,
    pub slots: Vec<SlotView>,
    /// How many slots still hold their sample.
    pub unfilled: usize,
    pub texts: Vec<TemplateText>,
    /// The music the person may swap, with its volume, if one is marked.
    pub music: Option<(SegmentId, f64)>,
}

/// A file the window should ask the person for: media for the slot
/// `slot` of the template clip `clip`, or a sound for its music.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemplateAsk {
    pub clip: SegmentId,
    pub slot: Option<SegmentId>,
}

/// The template folder a shelf card names.
fn card_folder(card_id: &str) -> Result<PathBuf, ShellError> {
    card_id
        .strip_prefix(TEMPLATE_CARD_PREFIX)
        .filter(|folder| !folder.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| ShellError::Unavailable("that is not a template".to_owned()))
}

/// The shelf card for the template at `folder`.
#[must_use]
pub fn template_card_id(folder: &Path) -> String {
    format!("{TEMPLATE_CARD_PREFIX}{}", folder.to_string_lossy())
}

/// How long the template a shelf card names plays.
pub(super) fn card_length(card_id: &str) -> Option<Micros> {
    let folder = card_folder(card_id).ok()?;
    cutlass_store::listed_template(&folder).map(|listed| listed.duration)
}

fn words_of(draft: &Draft, segment_id: &SegmentId) -> String {
    locate(draft, segment_id)
        .and_then(
            |(segment, _)| match draft.materials.get(segment.material_id.as_str()) {
                Some(MaterialRef::Text(text)) => {
                    Some(text.words().unwrap_or_default().into_owned())
                }
                _ => None,
            },
        )
        .unwrap_or_default()
}

impl Shell {
    /// The selected clip's template mark.
    #[must_use]
    pub fn template_mark(&self) -> Option<TemplateMark> {
        self.selected_segment()?.template_mark
    }

    /// Mark the selected clip as a template slot, editable text or the
    /// music, or take its mark off. One undo step.
    pub fn set_template_mark(&mut self, mark: Option<TemplateMark>) -> Result<(), ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetTemplateMark { segment_id, mark })
    }

    /// The open project's own slots: what saving it as a template offers.
    #[must_use]
    pub fn template_slots(&self) -> Vec<TemplateSlot> {
        self.draft().map(template_slots).unwrap_or_default()
    }

    /// The templates the user saved, newest first.
    #[must_use]
    pub fn list_templates(root: &Path) -> Vec<ListedTemplate> {
        cutlass_store::list_templates(root)
    }

    /// Save the open project, then save it as a template named `name`
    /// under `category` in `root`, and return the template's folder.
    pub fn save_as_template(
        &mut self,
        root: &Path,
        name: &str,
        category: &str,
    ) -> Result<PathBuf, ShellError> {
        let project = self.path.clone().ok_or(ShellError::NoProject)?;
        self.save()?;
        let template = cutlass_store::save_as_template(&project, root, name, category)
            .map_err(ShellError::Store)?;
        cutlass_logger::info!("saved a template at {}", template.display());
        Ok(template)
    }

    /// Start a new project under `projects` that holds only the template
    /// at `template`, as one clip, and open it with that clip picked.
    pub fn use_template(&mut self, template: &Path, projects: &Path) -> Result<(), ShellError> {
        let project = cutlass_store::use_template(template, projects).map_err(ShellError::Store)?;
        self.open_folder(&project)?;
        let clip = self.draft().and_then(|draft| {
            draft
                .tracks
                .iter()
                .flat_map(|track| &track.segments)
                .map(|segment| segment.id.clone())
                .find(|id| template_clip(draft, id).is_some())
        });
        self.select(clip);
        Ok(())
    }

    /// Delete the template at `template`.
    pub fn delete_template(template: &Path) -> Result<(), ShellError> {
        cutlass_store::delete_template(template).map_err(ShellError::Store)
    }

    /// Add the template a shelf card names at the end of the main row, as
    /// one clip, and pick it.
    pub fn add_template(&mut self, card_id: &str) -> Result<(), ShellError> {
        let folder = card_folder(card_id)?;
        let (track_id, start, lane) = {
            let Append { lane, start } = append_placement(self.session_draft()?, TrackType::Video);
            match lane {
                AppendLane::Track(track_id) => (track_id, start, None),
                AppendLane::New(kind) => (TrackId::new(self.fresh("track")), start, Some(kind)),
            }
        };
        self.place_template(&folder, track_id, start, lane)?;
        self.seek(start);
        Ok(())
    }

    /// Place the template a shelf card names where it was dropped.
    pub fn drop_template(&mut self, card_id: &str, x: f64, y: f64) -> Result<(), ShellError> {
        let folder = card_folder(card_id)?;
        let duration = card_length(card_id)
            .filter(|length| length.0 > 0)
            .ok_or_else(|| ShellError::Unavailable("the template is gone".to_owned()))?;
        let x = self.time_zero_x(x);
        let outcome = {
            let gesture = DropGesture {
                kind: TrackType::Video,
                duration,
                pointer: Pointer { x, y },
                scale: self.scale(),
                playhead: self.playhead,
                snap_threshold: self.snap_threshold(),
            };
            drop_gesture(self.session_draft()?, &gesture)
                .map_err(|_| ShellError::Unavailable("the drag could not be read".to_owned()))?
        };
        let (track_id, start, lane) = match outcome {
            DropOutcome::Miss => {
                return Err(ShellError::Unavailable(
                    "the drag missed the lanes".to_owned(),
                ));
            }
            DropOutcome::Place { track_id, start } => (track_id, start, None),
            DropOutcome::NewLane { kind, start } => {
                (TrackId::new(self.fresh("track")), start, Some(kind))
            }
        };
        self.place_template(&folder, track_id, start, lane)
    }

    /// Put the template at `folder` on `track_id` at `start` as one closed
    /// clip, opening the lane first when `lane` names its kind. Its files
    /// are copied into the project first, and taken back out if the clip
    /// does not land.
    pub(super) fn place_template(
        &mut self,
        folder: &Path,
        track_id: TrackId,
        start: Micros,
        lane: Option<TrackType>,
    ) -> Result<(), ShellError> {
        let project = self.path.clone().ok_or(ShellError::NoProject)?;
        let first_clip = self
            .session_draft()?
            .tracks
            .iter()
            .all(|track| track.segments.is_empty());
        let inside = cutlass_store::template_inside(folder, &project).map_err(ShellError::Store)?;
        let place = MediaPlace {
            track_id,
            segment_id: SegmentId::new(self.fresh("seg")),
            material_id: self.fresh("template"),
            start,
            duration: Micros(0),
            new_lane: lane,
        };
        let placed = (|| {
            let requests = template_clip_requests(
                self.session_draft()?,
                &inside.name,
                inside.draft,
                &place,
                self.linkage_enabled(),
            );
            if requests.is_empty() {
                return Err(ShellError::Unavailable(
                    "the template plays nothing".to_owned(),
                ));
            }
            self.apply_requests(&requests)
        })();
        if let Err(error) = placed {
            for file in inside.copied {
                let _ = std::fs::remove_file(file);
            }
            return Err(error);
        }
        self.reveal = Some(place.segment_id.clone());
        self.select(Some(place.segment_id));
        if first_clip {
            let _ = self.zoom_fit();
        }
        Ok(())
    }

    /// The clip `segment_id` as a template, or `None` for any other clip.
    #[must_use]
    pub fn template_clip(&self, segment_id: &SegmentId) -> Option<TemplateClip> {
        let draft = self.draft()?;
        let (material, inside) = template_clip(draft, segment_id)?;
        let slots = template_slots(inside)
            .into_iter()
            .map(|slot| {
                let material_id = locate(inside, &slot.segment_id)
                    .map(|(segment, _)| segment.material_id.as_str().to_owned())
                    .unwrap_or_default();
                let file = match inside.materials.get(&material_id) {
                    Some(MaterialRef::Video(video)) => self.media_file(Some(video.path.as_str())),
                    _ => None,
                };
                SlotView {
                    slot,
                    file,
                    material_id,
                }
            })
            .collect();
        let texts = template_texts(inside)
            .into_iter()
            .map(|segment_id| TemplateText {
                words: words_of(inside, &segment_id),
                segment_id,
            })
            .collect();
        let music = template_music(inside).map(|segment_id| {
            let volume = locate(inside, &segment_id).map_or(1.0, |(segment, _)| segment.volume);
            (segment_id, volume)
        });
        Some(TemplateClip {
            segment_id: segment_id.clone(),
            name: material.name.clone(),
            slots,
            unfilled: template_unfilled(inside),
            texts,
            music,
        })
    }

    /// What the clip `segment_id` says on the timeline when it is a
    /// template: how many of its clips are still to be replaced.
    #[must_use]
    pub fn template_badge(&self, segment_id: &SegmentId) -> Option<String> {
        let (_, inside) = template_clip(self.draft()?, segment_id)?;
        let slots = template_slots(inside).len();
        Some(match template_unfilled(inside) {
            0 if slots == 0 => "Template".to_owned(),
            0 => "All clips replaced".to_owned(),
            1 => "1 clip to be replaced".to_owned(),
            left => format!("{left} clips to be replaced"),
        })
    }

    /// Whether the slot strip shows over the template clip `segment_id`:
    /// while it alone is picked, until the person hides it. Picking the
    /// clip again brings a hidden strip back.
    pub fn template_strip_shown(&mut self, segment_id: &SegmentId) -> bool {
        let alone = self.selection.len() == 1 && self.selected() == Some(segment_id);
        if !alone {
            if self.template_strip_hidden.as_ref() == Some(segment_id) {
                self.template_strip_hidden = None;
            }
            return false;
        }
        self.template_strip_hidden.as_ref() != Some(segment_id)
    }

    /// Hide the slot strip of the template clip `segment_id`.
    pub fn hide_template_strip(&mut self, segment_id: &SegmentId) {
        self.template_strip_hidden = Some(segment_id.clone());
    }

    /// Show the slot strip of the picked template clip again.
    pub fn show_template_strip(&mut self) {
        self.template_strip_hidden = None;
    }

    /// Have the window ask for a file for a template clip.
    pub fn ask_template_file(&mut self, ask: TemplateAsk) {
        self.template_ask = Some(ask);
    }

    /// The file the window should ask for, once.
    pub fn take_template_ask(&mut self) -> Option<TemplateAsk> {
        self.template_ask.take()
    }

    /// The selected clip as a template.
    #[must_use]
    pub fn selected_template(&self) -> Option<TemplateClip> {
        self.template_clip(self.selected()?)
    }

    /// Put the file at `path` in the slot `slot` of the template clip
    /// `clip`. The file is read and copied into the project first; the
    /// fill is one undo step. A file that does not fit the slot is refused
    /// before anything changes, and its copy is taken back out.
    pub fn fill_template_slot(
        &mut self,
        clip: &SegmentId,
        slot: &SegmentId,
        path: &Path,
    ) -> Result<(), ShellError> {
        self.fill_template(clip, |shell, project, copied| {
            match shell.read_into(project, path, copied)? {
                Read::Picture(material) => Ok(TemplateFill {
                    media: vec![(slot.clone(), *material)],
                    ..TemplateFill::default()
                }),
                Read::Sound(_) => Err(ShellError::Unavailable(
                    "a slot takes a photo or a video".to_owned(),
                )),
            }
        })
    }

    /// Play the sound at `path` in place of the music of the template
    /// clip `clip`. One undo step.
    pub fn set_template_music(&mut self, clip: &SegmentId, path: &Path) -> Result<(), ShellError> {
        self.fill_template(clip, |shell, project, copied| {
            match shell.read_into(project, path, copied)? {
                Read::Sound(material) => Ok(TemplateFill {
                    music: Some(*material),
                    ..TemplateFill::default()
                }),
                Read::Picture(_) => Err(ShellError::Unavailable(
                    "the music takes a sound file".to_owned(),
                )),
            }
        })
    }

    /// Give the editable text `text` of the template clip `clip` new
    /// words. One undo step; the same words change nothing.
    pub fn set_template_text(
        &mut self,
        clip: &SegmentId,
        text: &SegmentId,
        words: &str,
    ) -> Result<(), ShellError> {
        let same = self.template_clip(clip).is_some_and(|template| {
            template
                .texts
                .iter()
                .any(|item| &item.segment_id == text && item.words == words)
        });
        if same {
            return Ok(());
        }
        self.fill_template(clip, |_, _, _| {
            Ok(TemplateFill {
                texts: vec![(text.clone(), words.to_owned())],
                ..TemplateFill::default()
            })
        })
    }

    /// Set how loud the music of the template clip `clip` plays: 1 is as
    /// the author left the file. One undo step.
    pub fn set_template_music_volume(
        &mut self,
        clip: &SegmentId,
        volume: f64,
    ) -> Result<(), ShellError> {
        self.refuse_locked(clip)?;
        let music = self
            .template_clip(clip)
            .and_then(|template| template.music)
            .ok_or_else(|| ShellError::Unavailable("the template has no music".to_owned()))?;
        let requests = template_inside_requests(
            self.session_draft()?,
            clip,
            &[Request::SetVolume {
                segment_id: music.0,
                volume,
            }],
        )
        .map_err(|error| ShellError::Unavailable(error.to_string()))?;
        self.apply_requests(&requests)
    }

    /// Build a fill with `build`, which may copy files into the project,
    /// and store it on the template clip `clip` as one undo step. On any
    /// failure the copied files are taken back out.
    fn fill_template(
        &mut self,
        clip: &SegmentId,
        build: impl FnOnce(&mut Self, &Path, &mut Vec<PathBuf>) -> Result<TemplateFill, ShellError>,
    ) -> Result<(), ShellError> {
        let project = self.path.clone().ok_or(ShellError::NoProject)?;
        self.refuse_locked(clip)?;
        let mut copied: Vec<PathBuf> = Vec::new();
        let filled = (|| {
            let fill = build(self, &project, &mut copied)?;
            let requests = template_clip_fill_requests(self.session_draft()?, clip, &fill)
                .map_err(|error| ShellError::Unavailable(error.to_string()))?;
            if requests.is_empty() {
                return Ok(());
            }
            self.apply_requests(&requests)
        })();
        if filled.is_err() {
            for file in copied {
                let _ = std::fs::remove_file(file);
            }
        }
        filled
    }

    /// Read the file at `path`, place it in `project`, and build its
    /// library record without adding it yet.
    fn read_into(
        &mut self,
        project: &Path,
        path: &Path,
        copied: &mut Vec<PathBuf>,
    ) -> Result<Read, ShellError> {
        let link = self.settings.import.link;
        let probed = probe::probe(path).map_err(|error| ShellError::Probe(error.to_string()))?;
        let (stored, copy) = place_in_project(project, path, None, link)?;
        copied.extend(copy);
        Ok(match self.medium_request(probed, stored) {
            Request::Import {
                medium: ImportedMedium::Video(material),
            } => Read::Picture(material),
            Request::Import {
                medium: ImportedMedium::Audio(material),
            } => Read::Sound(material),
            _ => {
                return Err(ShellError::Unavailable(
                    "the file is not a picture or a sound".to_owned(),
                ));
            }
        })
    }
}

enum Read {
    Picture(Box<MaterialVideo>),
    Sound(Box<MaterialAudio>),
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use cutlass_model::{
        Draft, MaterialText, MaterialVideo, MetaType, Micros, Segment, SegmentId, SlotTakes,
        TemplateMark, TimeRange, Track, TrackFlag, TrackType,
    };
    use cutlass_store::Folder;

    use super::template_card_id;
    use crate::shell::Shell;
    use crate::shell::tests::{Temp, shell};

    fn pattern() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../crates/engine/fixtures/pattern.png")
    }

    /// A project at `path` with a photo slot `a`, a video-only slot `b`
    /// and an editable title `t`.
    fn template_project(path: &Path) {
        let mut draft = Draft::empty("draft");
        draft.name = "Trip".to_owned();
        for (id, kind) in [("p", MetaType::Photo), ("v", MetaType::Video)] {
            let mut material = MaterialVideo::new(id);
            material.kind = kind.as_str().to_owned();
            material.width = 16;
            material.height = 16;
            material.duration = Micros(10_000_000);
            draft.materials.videos.push(material);
        }
        draft.materials.texts.push(MaterialText::new("words"));
        let mut photo = Segment::new("a", "p", TimeRange::new(0, 1_000_000));
        photo.source_range = Some(TimeRange::new(0, 1_000_000));
        photo.template_mark = Some(TemplateMark::slot(SlotTakes::Photo));
        let mut video = Segment::new("b", "v", TimeRange::new(1_000_000, 1_000_000));
        video.source_range = Some(TimeRange::new(0, 1_000_000));
        video.template_mark = Some(TemplateMark::slot(SlotTakes::Video));
        draft.tracks[0].segments.extend([photo, video]);
        let mut title = Segment::new("t", "words", TimeRange::new(0, 2_000_000));
        title.template_mark = Some(TemplateMark::Text);
        let mut row = Track::main("text");
        row.kind = TrackType::Text;
        row.flag = TrackFlag::None;
        row.segments.push(title);
        draft.tracks.push(row);
        Folder::new(draft).save(path).unwrap();
    }

    /// A shell on a new project made from the template of
    /// [`template_project`], and the template clip's id.
    fn used(temp: &Temp) -> (Shell, SegmentId, PathBuf) {
        let project = temp.path.join("Projects").join("Trip");
        template_project(&project);
        let mut desktop = shell();
        desktop.open_folder(&project).unwrap();
        let templates = temp.path.join("Templates");
        let template = desktop
            .save_as_template(&templates, "Beach day", "Travel")
            .unwrap();
        desktop
            .use_template(&template, &temp.path.join("Projects"))
            .unwrap();
        let clip = desktop.selected().cloned().unwrap();
        (desktop, clip, template)
    }

    #[test]
    fn a_clip_is_marked_and_unmarked_from_the_inspector() {
        let temp = Temp::new();
        template_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("a")));
        assert!(matches!(
            desktop.template_mark(),
            Some(TemplateMark::Slot { .. })
        ));
        desktop.set_template_mark(None).unwrap();
        assert_eq!(desktop.template_mark(), None);
        assert_eq!(desktop.template_slots().len(), 1);
        desktop.select(None);
        assert!(desktop.set_template_mark(None).is_err());
    }

    #[test]
    fn a_used_template_opens_as_one_closed_clip_with_its_strip_up() {
        let temp = Temp::new();
        let (mut desktop, clip, template) = used(&temp);
        let draft = desktop.draft().unwrap();
        assert_eq!(draft.name, "Beach day");
        assert_eq!(draft.tracks.iter().flat_map(|t| &t.segments).count(), 1);
        // The project's own timeline has no slots: they are in the clip.
        assert!(desktop.template_slots().is_empty());
        let shown = desktop.template_clip(&clip).unwrap();
        assert_eq!(shown.name, "Beach day");
        assert_eq!(shown.slots.len(), 2);
        assert_eq!(shown.unfilled, 2);
        assert_eq!(shown.texts.len(), 1);
        assert_eq!(
            desktop.template_badge(&clip).as_deref(),
            Some("2 clips to be replaced")
        );
        // It is not a compound clip: it is never opened or ungrouped.
        assert!(!desktop.compound_selected());
        assert!(desktop.open_compound().is_err());
        assert!(desktop.ungroup_compound().is_err());
        assert_eq!(desktop.compound_depth(), 0);
        // The strip is up while the clip is picked, until it is hidden;
        // picking the clip again brings it back.
        assert!(desktop.template_strip_shown(&clip));
        desktop.hide_template_strip(&clip);
        assert!(!desktop.template_strip_shown(&clip));
        desktop.select(None);
        assert!(!desktop.template_strip_shown(&clip));
        desktop.select(Some(clip.clone()));
        assert!(desktop.template_strip_shown(&clip));
        Shell::delete_template(&template).unwrap();
        assert!(Shell::list_templates(&temp.path.join("Templates")).is_empty());
        // The project keeps playing its own copy.
        assert!(desktop.template_clip(&clip).is_some());
    }

    #[test]
    fn a_picked_photo_fills_its_slot_inside_the_clip_in_one_undo_step() {
        let temp = Temp::new();
        let (mut desktop, clip, _) = used(&temp);
        let project = desktop.path.clone().unwrap();
        let slots = desktop.template_clip(&clip).unwrap().slots;
        let photo_slot = slots[0].slot.segment_id.clone();
        desktop
            .fill_template_slot(&clip, &photo_slot, &pattern())
            .unwrap();
        let shown = desktop.template_clip(&clip).unwrap();
        assert!(shown.slots[0].slot.filled);
        assert_eq!(
            shown.slots[0].file.as_deref().and_then(Path::file_name),
            Some(std::ffi::OsStr::new("pattern.png"))
        );
        assert_eq!(shown.unfilled, 1);
        assert_eq!(
            desktop.template_badge(&clip).as_deref(),
            Some("1 clip to be replaced")
        );
        assert!(project.join("Resources/pattern.png").is_file());
        // The project's library and timeline hold nothing new.
        let draft = desktop.draft().unwrap();
        assert!(draft.materials.videos.is_empty());
        assert_eq!(draft.tracks.iter().flat_map(|t| &t.segments).count(), 1);
        desktop.undo().unwrap();
        assert_eq!(desktop.template_clip(&clip).unwrap().unfilled, 2);
    }

    #[test]
    fn a_photo_in_a_video_slot_is_refused_and_its_copy_taken_back_out() {
        let temp = Temp::new();
        let (mut desktop, clip, _) = used(&temp);
        let project = desktop.path.clone().unwrap();
        let before = desktop.draft().unwrap().clone();
        let video_slot = desktop.template_clip(&clip).unwrap().slots[1]
            .slot
            .segment_id
            .clone();
        assert!(
            desktop
                .fill_template_slot(&clip, &video_slot, &pattern())
                .is_err()
        );
        assert_eq!(desktop.draft().unwrap(), &before);
        let left =
            std::fs::read_dir(project.join("Resources")).map_or(0, |entries| entries.count());
        assert_eq!(left, 0, "no copied file is left behind");
        // A clip that is not a template takes no fill.
        assert!(
            desktop
                .fill_template_slot(&SegmentId::new("nope"), &video_slot, &pattern())
                .is_err()
        );
    }

    #[test]
    fn an_editable_text_takes_new_words_inside_the_clip() {
        let temp = Temp::new();
        let (mut desktop, clip, _) = used(&temp);
        let text = desktop.template_clip(&clip).unwrap().texts[0]
            .segment_id
            .clone();
        desktop.set_template_text(&clip, &text, "Our trip").unwrap();
        assert_eq!(
            desktop.template_clip(&clip).unwrap().texts[0].words,
            "Our trip"
        );
        // The same words change nothing, so there is nothing more to undo.
        let revision = desktop.draft().unwrap().clone();
        desktop.set_template_text(&clip, &text, "Our trip").unwrap();
        assert_eq!(desktop.draft().unwrap(), &revision);
        desktop.undo().unwrap();
        assert_ne!(
            desktop.template_clip(&clip).unwrap().texts[0].words,
            "Our trip"
        );
        // No music is marked, so there is none to turn down.
        assert!(desktop.set_template_music_volume(&clip, 0.5).is_err());
    }

    #[test]
    fn a_template_card_adds_one_clip_to_an_open_project() {
        let temp = Temp::new();
        let (mut desktop, first, template) = used(&temp);
        let card = template_card_id(&template);
        desktop.add_template(&card).unwrap();
        let draft = desktop.draft().unwrap();
        let clips: Vec<_> = draft.tracks.iter().flat_map(|t| &t.segments).collect();
        assert_eq!(clips.len(), 2);
        let second = desktop.selected().cloned().unwrap();
        assert_ne!(second, first);
        assert_eq!(clips[1].id, second);
        assert_eq!(clips[1].target, TimeRange::new(2_000_000, 2_000_000));
        // Each clip keeps its own slots: their ids never meet.
        let (a, b) = (
            desktop.template_clip(&first).unwrap(),
            desktop.template_clip(&second).unwrap(),
        );
        assert_ne!(a.slots[0].slot.segment_id, b.slots[0].slot.segment_id);
        desktop.undo().unwrap();
        assert!(desktop.template_clip(&second).is_none());
        assert!(desktop.add_template("template:").is_err());
        assert!(desktop.add_template("template:/nowhere/at/all").is_err());
    }
}
