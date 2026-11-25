//! Project templates on the desktop: marking clips, saving the open
//! project as a template, starting a project from one, and filling its
//! slots with the person's own files.

use std::path::{Path, PathBuf};

use cutlass_editor::{TemplateFill, template_fill_requests};
use cutlass_model::{
    MaterialAudio, MaterialRef, MaterialVideo, SegmentId, TemplateMark, TemplateSlot,
    template_music, template_slots, template_texts,
};
use cutlass_session::{ImportedMedium, Request};
use cutlass_store::ListedTemplate;

use super::{Shell, ShellError, place_in_project, probe};

/// One editable text of the open template, with its words now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemplateText {
    pub segment_id: SegmentId,
    pub words: String,
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

    /// The open project's slots in the order they fill.
    #[must_use]
    pub fn template_slots(&self) -> Vec<TemplateSlot> {
        self.draft().map(template_slots).unwrap_or_default()
    }

    /// The open project's editable texts, with their words now.
    #[must_use]
    pub fn template_texts(&self) -> Vec<TemplateText> {
        let Some(draft) = self.draft() else {
            return Vec::new();
        };
        template_texts(draft)
            .into_iter()
            .map(|segment_id| {
                let words = draft
                    .tracks
                    .iter()
                    .flat_map(|track| &track.segments)
                    .find(|segment| segment.id == segment_id)
                    .and_then(
                        |segment| match draft.materials.get(segment.material_id.as_str()) {
                            Some(MaterialRef::Text(text)) => {
                                Some(text.words().unwrap_or_default().into_owned())
                            }
                            _ => None,
                        },
                    )
                    .unwrap_or_default();
                TemplateText { segment_id, words }
            })
            .collect()
    }

    /// Whether the open project has a music clip a template fill may swap.
    #[must_use]
    pub fn template_has_music(&self) -> bool {
        self.draft().and_then(template_music).is_some()
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

    /// Start a new project under `projects` from the template at
    /// `template`, and open it.
    pub fn use_template(&mut self, template: &Path, projects: &Path) -> Result<(), ShellError> {
        let project = cutlass_store::use_template(template, projects).map_err(ShellError::Store)?;
        self.open_folder(&project)
    }

    /// Delete the template at `template`.
    pub fn delete_template(template: &Path) -> Result<(), ShellError> {
        cutlass_store::delete_template(template).map_err(ShellError::Store)
    }

    /// Fill the open template: each picked file goes in its slot, each
    /// text takes its words, and `music` swaps the marked music. The files
    /// are read and copied into the project first; the fill itself is one
    /// undo step. A file that does not fit its slot is refused before
    /// anything changes, and the files already copied are taken back out.
    pub fn fill_template(
        &mut self,
        picks: &[(SegmentId, PathBuf)],
        texts: &[(SegmentId, String)],
        music: Option<&Path>,
    ) -> Result<(), ShellError> {
        let project = self.path.clone().ok_or(ShellError::NoProject)?;
        let link = self.settings.import.link;
        let mut copied: Vec<PathBuf> = Vec::new();
        let filled = (|| {
            let mut fill = TemplateFill {
                texts: texts.to_vec(),
                ..TemplateFill::default()
            };
            for (segment_id, path) in picks {
                match self.read_into(&project, path, link, &mut copied)? {
                    Read::Picture(material) => fill.media.push((segment_id.clone(), *material)),
                    Read::Sound(_) => {
                        return Err(ShellError::Unavailable(
                            "a slot takes a photo or a video".to_owned(),
                        ));
                    }
                }
            }
            if let Some(path) = music {
                match self.read_into(&project, path, link, &mut copied)? {
                    Read::Sound(material) => fill.music = Some(*material),
                    Read::Picture(_) => {
                        return Err(ShellError::Unavailable(
                            "the music takes a sound file".to_owned(),
                        ));
                    }
                }
            }
            let draft = self.session_draft()?;
            let requests = template_fill_requests(draft, &fill)
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
        link: bool,
        copied: &mut Vec<PathBuf>,
    ) -> Result<Read, ShellError> {
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
    use std::path::PathBuf;

    use cutlass_model::{
        Draft, MaterialVideo, MetaType, Micros, Segment, SegmentId, SlotTakes, TemplateMark,
        TimeRange,
    };
    use cutlass_store::Folder;

    use crate::shell::tests::{Temp, shell};

    fn pattern() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../crates/engine/fixtures/pattern.png")
    }

    /// A project at `path` with a photo slot `a` and a video-only slot `b`.
    fn template_project(path: &std::path::Path) {
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
        let mut photo = Segment::new("a", "p", TimeRange::new(0, 1_000_000));
        photo.source_range = Some(TimeRange::new(0, 1_000_000));
        photo.template_mark = Some(TemplateMark::Slot {
            takes: SlotTakes::Photo,
        });
        let mut video = Segment::new("b", "v", TimeRange::new(1_000_000, 1_000_000));
        video.source_range = Some(TimeRange::new(0, 1_000_000));
        video.template_mark = Some(TemplateMark::Slot {
            takes: SlotTakes::Video,
        });
        draft.tracks[0].segments.extend([photo, video]);
        Folder::new(draft).save(path).unwrap();
    }

    fn material_of(desktop: &crate::shell::Shell, id: &str) -> String {
        desktop
            .draft()
            .unwrap()
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id.as_str() == id)
            .unwrap()
            .material_id
            .as_str()
            .to_owned()
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
    fn a_picked_photo_fills_its_slot_in_one_undo_step() {
        let temp = Temp::new();
        template_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        assert_eq!(desktop.template_slots().len(), 2);
        desktop
            .fill_template(&[(SegmentId::new("a"), pattern())], &[], None)
            .unwrap();
        assert_ne!(material_of(&desktop, "a"), "p");
        assert!(temp.path.join("Resources/pattern.png").is_file());
        desktop.undo().unwrap();
        assert_eq!(material_of(&desktop, "a"), "p");
    }

    #[test]
    fn a_photo_in_a_video_slot_is_refused_and_its_copy_taken_back_out() {
        let temp = Temp::new();
        template_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let refused = desktop.fill_template(
            &[
                (SegmentId::new("a"), pattern()),
                (SegmentId::new("b"), pattern()),
            ],
            &[],
            None,
        );
        assert!(refused.is_err());
        assert_eq!(material_of(&desktop, "a"), "p");
        assert_eq!(material_of(&desktop, "b"), "v");
        let left =
            std::fs::read_dir(temp.path.join("Resources")).map_or(0, |entries| entries.count());
        assert_eq!(left, 0, "no copied file is left behind");
    }

    #[test]
    fn the_open_project_saves_as_a_template_and_starts_a_new_project() {
        let temp = Temp::new();
        let project = temp.path.join("Projects").join("Trip");
        template_project(&project);
        let mut desktop = shell();
        desktop.open_folder(&project).unwrap();
        let templates = temp.path.join("Templates");
        let template = desktop
            .save_as_template(&templates, "Beach day", "Travel")
            .unwrap();
        let listed = crate::shell::Shell::list_templates(&templates);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].slots, 2);
        desktop
            .use_template(&template, &temp.path.join("Projects"))
            .unwrap();
        assert_eq!(desktop.draft().unwrap().name, "Beach day");
        assert_eq!(desktop.template_slots().len(), 2);
        crate::shell::Shell::delete_template(&template).unwrap();
        assert!(crate::shell::Shell::list_templates(&templates).is_empty());
    }
}
