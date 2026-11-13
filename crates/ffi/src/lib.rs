//! The calls SwiftUI and Compose make.
//!
//! Desktop does not use this crate. A phone shell opens one project folder,
//! applies the same session requests the desktop applies, and looks up a
//! bundled or cached file. Playback here is the playhead and the audio
//! output. Seeking a picture needs a device the shell constructs; this crate
//! does not construct one.

#![forbid(unsafe_code)]

mod animate;
mod cache;
mod canvas;
mod clip;
mod codecs;
mod effect;
mod error;
mod export;
mod host;
mod keys;
mod layer;
mod look;
mod media;
mod player;
mod preview;
mod project;
mod stage;
mod strip;
mod text;
mod timeline;
mod transition;

pub use animate::{PhoneAnimation, PhoneAnimationPreset, PhoneAnimations, animation_presets};
pub use cache::ResourceCache;
pub use canvas::{
    BACKGROUND_BLURS, BACKGROUND_COLORS, PhoneBackground, PhoneRatio, phone_background,
    phone_ratios,
};
pub use clip::{CROP_RATIOS, PhoneClipState, clip_state, clockwise_quarter, crop_for};
pub use codecs::{decoders, readers};
pub use cutlass_editor::{
    Bar, Tool, ToolId, clock, fling_start, fling_step, speed_at, speed_position, tools, volume_at,
    volume_position,
};
pub use cutlass_effects::{bundled_look, look_poster};
pub use cutlass_export::video_bitrate;
pub use cutlass_logger::InitError as LogError;
pub use cutlass_model::TrackType;
pub use cutlass_playback::AudioHost;
pub use cutlass_probe::{ProbeError, Probed};
pub use cutlass_session::{Micros, SegmentId, TrackId};
pub use cutlass_storage::Home;
pub use cutlass_store::{ListedProject, create_project, list_projects};
pub use effect::{NEW_EFFECT_LENGTH, PhoneEffect, phone_effects};
pub use error::FfiError;
pub use export::{PHONE_EXPORT_SIDES, PhoneExport, encoders};
pub use host::Commands;
pub use keys::{PhoneGraph, PhoneKeys, graphs};
pub use layer::{PhoneShape, PhoneSticker, phone_shapes, phone_stickers, sticker_poster};
pub use look::{
    PhoneAdjust, PhoneClipFilter, PhoneFilter, adjust_sliders, clip_filter, phone_filters,
};
pub use player::PhonePlayer;
pub use preview::{Picture, Preview};
pub use project::Project;
pub use stage::PhoneStage;
pub use strip::{PhoneFilm, PhoneTick, PhoneTile, PhoneWave, Strip};
pub use text::{
    NEW_TEXT_LENGTH, PhoneFont, PhoneText, PhoneTextStyle, TEXT_COLORS, TEXT_STYLE_RANGES,
    TextStyleChange, TextStyleRanges,
};
pub use timeline::{Drag, PhoneClip, PhoneLane, phone_lanes};
pub use transition::{
    PhoneCut, PhoneTransition, phone_cuts, phone_transition_categories, phone_transitions,
};

use std::path::{Path, PathBuf};

use cutlass_playback::{FaceTable, Program};
use cutlass_session::{Draft, LinkageSettings, Request, Session};
use cutlass_store::RepairNote;

/// Open the process log in `directory` and record panics in it.
///
/// A phone shell calls this once at start. The public build writes the file
/// and does not write stderr.
pub fn start_log(directory: &Path) -> Result<PathBuf, LogError> {
    let opened = cutlass_logger::init(directory);
    cutlass_logger::install_panic_hook();
    opened
}

/// Give the calling thread the interface's priority.
///
/// A shell runs each [`PhoneHost`] on one thread and its interface waits for
/// the answers, so that thread must not rank below the interface. Apple
/// platforms raise it to the main thread's class; elsewhere nothing changes.
pub fn serve_the_interface() {
    #[cfg(target_vendor = "apple")]
    if !cutlass_platform_apple::run_at_interface_priority() {
        cutlass_logger::warn!("the host thread kept its priority");
    }
}

/// The open project, its session, the playhead, and the resource cache.
///
/// Save writes the session draft into the project folder. The undo stack stays
/// in the session. A new open starts the revision at zero again.
pub struct PhoneHost {
    project: Project,
    commands: Commands,
    player: PhonePlayer,
    cache: ResourceCache,
    preview: Preview,
    strip: Strip,
    faces: FaceTable,
    /// The preview gesture in flight.
    live: Option<stage::Live>,
    /// The title being typed into, and the revision of its last change.
    typing: Option<(SegmentId, u64)>,
    /// The title a slider is restyling, and the revision of its last change.
    styling: Option<(SegmentId, u64)>,
}

impl PhoneHost {
    /// Create an empty project and open the bundle and the cache beside it.
    ///
    /// `bundle` must already be a directory. The cache directory is created.
    /// Linkage comes from the project file: a new folder has no biz config and
    /// no pairs, so the master switch is off.
    pub fn create(
        project: &Path,
        draft_id: &str,
        bundle: &Path,
        cache: &Path,
        limit: u64,
    ) -> Result<Self, FfiError> {
        Self::from_project(Project::create(project, draft_id)?, bundle, cache, limit)
    }

    /// Open a project folder that is already on disk.
    pub fn open(project: &Path, bundle: &Path, cache: &Path, limit: u64) -> Result<Self, FfiError> {
        Self::from_project(Project::open(project)?, bundle, cache, limit)
    }

    fn from_project(
        project: Project,
        bundle: &Path,
        cache: &Path,
        limit: u64,
    ) -> Result<Self, FfiError> {
        let commands = Commands::open(project.draft().clone(), project.stored_linkage());
        let cache = ResourceCache::open(bundle, cache, limit)?;
        Ok(Self {
            project,
            commands,
            player: PhonePlayer::new(),
            cache,
            preview: Preview::new(),
            strip: Strip::default(),
            faces: FaceTable::default(),
            live: None,
            typing: None,
            styling: None,
        })
    }

    /// The session draft. This is the cut `apply` changes.
    #[must_use]
    pub fn draft(&self) -> &Draft {
        self.commands.draft()
    }

    /// The revision the next apply must name.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.commands.revision()
    }

    /// Linkage loaded from the project when this host was opened.
    #[must_use]
    pub fn linkage(&self) -> LinkageSettings {
        self.commands.linkage()
    }

    /// Canonical project root.
    #[must_use]
    pub fn root(&self) -> &Path {
        self.project.root()
    }

    /// Repair notes from open. The draft was not changed to clear them.
    #[must_use]
    pub fn report(&self) -> &[RepairNote] {
        self.project.report()
    }

    /// Write the session draft into the project folder.
    pub fn save(&mut self) -> Result<(), FfiError> {
        self.project.set_draft(self.commands.draft().clone());
        self.project.save()
    }

    /// A path inside the project folder. A path that leaves it is refused.
    pub fn resolve(&self, relative: &Path) -> Result<PathBuf, FfiError> {
        self.project.resolve(relative)
    }

    /// One combo. A failed request leaves the draft and the revision unchanged.
    pub fn apply(&mut self, expected: u64, requests: &[Request]) -> Result<u64, FfiError> {
        self.commands.apply(expected, requests)
    }

    /// Restore the draft from before the last apply. The revision advances.
    pub fn undo(&mut self) -> Result<u64, FfiError> {
        self.commands.undo()
    }

    /// Reapply the draft that undo removed. The revision advances again.
    pub fn redo(&mut self) -> Result<u64, FfiError> {
        self.commands.redo()
    }

    /// Snapshot the open draft.
    pub fn begin(&mut self) -> Result<(), FfiError> {
        self.commands.begin()
    }

    /// Open a recording, or mark the transaction `begin` already opened.
    pub fn record(&mut self) -> Result<(), FfiError> {
        self.commands.record()
    }

    /// Close the transaction.
    pub fn commit(&mut self) -> Result<(), FfiError> {
        self.commands.commit()
    }

    /// Fold the open draft into the last recorded step.
    pub fn amend(&mut self) -> Result<(), FfiError> {
        self.commands.amend()
    }

    /// Put the open draft back where it was at begin or record.
    pub fn rollback(&mut self) -> Result<(), FfiError> {
        self.commands.rollback()
    }

    /// Whether record has marked the open transaction.
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.commands.is_recording()
    }

    /// The session playback borrows. Applying a request does not move the playhead.
    #[must_use]
    pub fn session(&self) -> &Session {
        self.commands.session()
    }

    /// Open this device's audio output. Until then sound goes to the
    /// software buffer, as in tests.
    pub fn open_audio(&mut self, host: AudioHost) {
        self.player.open_output(host);
    }

    /// Play from the playhead with sound. An empty cut does not play.
    pub fn play(&mut self) -> Result<(), FfiError> {
        let end = self.draft().duration;
        let program = program(&self.commands, &self.project, &self.preview);
        self.player.play(program, end)?;
        self.strip.set_paused(self.player.is_playing());
        Ok(())
    }

    pub fn pause(&mut self) {
        self.player.pause();
        self.strip.set_paused(false);
    }

    /// Move the playhead, held inside the cut. Playing goes on from there.
    pub fn seek(&mut self, time: Micros) {
        let end = self.draft().duration;
        let program = program(&self.commands, &self.project, &self.preview);
        self.player.seek(program, time, end);
    }

    /// Move a playing playhead by `elapsed` microseconds with the heard
    /// sound, and return it. Play stops at the end of the cut. An edit made
    /// while playing is heard from the playhead on.
    pub fn tick(&mut self, elapsed: i64) -> Micros {
        let end = self.draft().duration;
        let program = program(&self.commands, &self.project, &self.preview);
        self.player.tick(program, elapsed, end)
    }

    /// The playhead, in microseconds.
    #[must_use]
    pub fn playhead(&self) -> Micros {
        self.player.playhead()
    }

    #[must_use]
    pub fn is_playing(&self) -> bool {
        self.player.is_playing()
    }

    /// Probe a file the phone picked, copy it into the project, and add it to
    /// the library.
    ///
    /// Returns the new material id. A refused probe or a failed request
    /// leaves the draft unchanged and removes the copy.
    pub fn import(&mut self, source: &Path) -> Result<String, FfiError> {
        let probed = cutlass_probe::probe(source).map_err(|error| FfiError::Refused {
            reason: error.to_string(),
        })?;
        self.import_probed(source, probed)
    }

    fn import_probed(&mut self, source: &Path, probed: Probed) -> Result<String, FfiError> {
        media::check(&probed)?;
        let id = media::fresh(&media::ids(self.draft()), "mat");
        let (stored, copy) = media::copy_in(self.root(), source)?;
        let request = Request::Import {
            medium: media::imported(&id, stored, probed),
        };
        if let Err(error) = self.apply(self.revision(), &[request]) {
            let _ = std::fs::remove_file(copy);
            return Err(error);
        }
        Ok(id)
    }

    /// Play the file at `source` in `segment_id` in place of its media,
    /// keeping the clip's place and edits, as one undo step. A picture
    /// takes a photo or video and a sound a sound; the file must be as
    /// long as the part the clip plays. A refusal leaves no copy behind.
    pub fn replace_file(&mut self, segment_id: &SegmentId, source: &Path) -> Result<(), FfiError> {
        let (_, track) = clip::unlocked(self.draft(), segment_id)?;
        let wants_picture = matches!(track.kind, TrackType::Video);
        if !wants_picture && track.kind != TrackType::Audio {
            return Err(FfiError::Refused {
                reason: "only a photo, video or sound clip can be replaced".to_owned(),
            });
        }
        let probed = cutlass_probe::probe(source).map_err(|error| FfiError::Refused {
            reason: error.to_string(),
        })?;
        media::check(&probed)?;
        if matches!(probed, Probed::Video { .. }) != wants_picture {
            return Err(FfiError::Refused {
                reason: if wants_picture {
                    "a picture clip takes a photo or a video".to_owned()
                } else {
                    "a sound clip takes a sound".to_owned()
                },
            });
        }
        let id = media::fresh(&media::ids(self.draft()), "mat");
        let (stored, copy) = media::copy_in(self.root(), source)?;
        let requests = [
            Request::Import {
                medium: media::imported(&id, stored, probed),
            },
            Request::ReplaceMedia {
                segment_id: segment_id.clone(),
                material_id: cutlass_model::MaterialId::new(id),
            },
        ];
        if let Err(error) = self.apply(self.revision(), &requests) {
            let _ = std::fs::remove_file(copy);
            return Err(error);
        }
        Ok(())
    }

    /// Bring in the sound of the video at `source` as a sound in the
    /// library, without its picture. Returns the new material id, for
    /// [`Self::append`]. A photo or a video with no sound is refused.
    pub fn import_sound_of(&mut self, source: &Path) -> Result<String, FfiError> {
        let probed = cutlass_probe::probe(source).map_err(|error| FfiError::Refused {
            reason: error.to_string(),
        })?;
        let duration = match probed {
            Probed::Video { meta, duration, .. } if meta != cutlass_model::MetaType::Photo => {
                duration
            }
            _ => {
                return Err(FfiError::Refused {
                    reason: "pick a video to take its sound".to_owned(),
                });
            }
        };
        if !self
            .preview
            .readers()
            .carries_sound(source)
            .unwrap_or(false)
        {
            return Err(FfiError::Refused {
                reason: "the video has no sound".to_owned(),
            });
        }
        let id = media::fresh(&media::ids(self.draft()), "mat");
        let (stored, copy) = media::copy_in(self.root(), source)?;
        let mut medium = media::imported(&id, stored, Probed::Audio { duration });
        if let cutlass_session::ImportedMedium::Audio(audio) = &mut medium {
            cutlass_model::MetaType::ExtractMusic
                .as_str()
                .clone_into(&mut audio.kind);
        }
        if let Err(error) = self.apply(self.revision(), &[Request::Import { medium }]) {
            let _ = std::fs::remove_file(copy);
            return Err(error);
        }
        Ok(id)
    }

    /// Add the whole of `material_id` at the end of its lane, as one undo step.
    /// A picture goes on main; a sound opens an audio lane when there is none.
    pub fn append(&mut self, material_id: &str) -> Result<SegmentId, FfiError> {
        let (requests, segment_id) = media::append_requests(self.draft(), material_id)?;
        self.apply(self.revision(), &requests)?;
        Ok(segment_id)
    }

    /// The canvas ratios a picker offers, Original first, each with the
    /// frame it gives this project.
    pub fn ratios(&self) -> Vec<PhoneRatio> {
        phone_ratios(self.draft())
    }

    /// Give the canvas `ratio`, as one undo step. A canvas that already
    /// has that ratio and frame is not a step.
    pub fn set_ratio(&mut self, ratio: &str) -> Result<(), FfiError> {
        let request = cutlass_session::ratio_request(self.draft(), ratio)?;
        let canvas = &self.draft().canvas_config;
        if let Request::SetCanvasSize {
            width,
            height,
            ratio,
        } = &request
            && (canvas.width, canvas.height, canvas.ratio.as_str()) == (*width, *height, ratio)
        {
            return Ok(());
        }
        self.apply(self.revision(), &[request])?;
        Ok(())
    }

    /// What fills the canvas around the main pictures.
    pub fn background(&self) -> PhoneBackground {
        phone_background(self.draft())
    }

    /// Fill the canvas around the main pictures, as one undo step. A blur
    /// needs a picture on the main lane.
    pub fn set_background(&mut self, background: &PhoneBackground) -> Result<(), FfiError> {
        if *background == self.background() {
            return Ok(());
        }
        let taken = media::ids(self.draft());
        let requests =
            canvas::background_requests(self.draft(), background, &taken).ok_or_else(|| {
                FfiError::Refused {
                    reason: "add a clip to blur behind".to_owned(),
                }
            })?;
        self.apply(self.revision(), &requests)?;
        Ok(())
    }

    /// The box of the picture clip `segment_id` on the canvas at `time`,
    /// where its keyframes put it. A clip that is not a picture has none.
    #[must_use]
    pub fn stage(&self, segment_id: &SegmentId, time: Micros) -> Option<PhoneStage> {
        let stage = cutlass_editor::stage_box(self.draft(), segment_id, time, &self.sizes())?;
        let canvas = cutlass_editor::canvas_size(self.draft());
        Some(PhoneStage::new(
            self.draft(),
            canvas,
            &stage,
            Default::default(),
        ))
    }

    /// The topmost picture clip under the canvas point `(x, y)` at `time`.
    #[must_use]
    pub fn stage_pick(&self, time: Micros, x: f64, y: f64) -> Option<SegmentId> {
        cutlass_editor::stage_pick(self.draft(), time, (x, y), &self.sizes())
    }

    /// Start moving the picture clip `segment_id` on the preview at `time`.
    /// A clip with keyframes keys its new pose there. A gesture still open
    /// is ended first.
    pub fn stage_begin(&mut self, segment_id: &SegmentId, time: Micros) -> Result<(), FfiError> {
        let gesture = cutlass_editor::stage_gesture(self.draft(), segment_id, time, &self.sizes())
            .ok_or_else(|| FfiError::Refused {
                reason: "only an unlocked picture or title moves on the preview".to_owned(),
            })?;
        self.live = Some(stage::Live {
            gesture,
            applied: None,
        });
        Ok(())
    }

    /// Place the clip `(shift_x, shift_y)` canvas pixels from where the
    /// gesture started, `pinch` times its size, and turned `turn` degrees
    /// clockwise. The first change is an undo step and later ones join it.
    pub fn stage_move(
        &mut self,
        shift_x: f64,
        shift_y: f64,
        pinch: f64,
        turn: f64,
    ) -> Result<PhoneStage, FfiError> {
        let live = self.live.as_ref().ok_or_else(|| FfiError::Refused {
            reason: "no preview gesture is open".to_owned(),
        })?;
        let step = live.gesture.step(shift_x, shift_y, pinch, turn);
        let canvas = cutlass_editor::canvas_size(self.draft());
        let stage = PhoneStage::new(self.draft(), canvas, &step.stage, step.guides);
        let revision = self.revision();
        let joins = match live.applied {
            Some((_, pose)) if pose == step.pose => return Ok(stage),
            Some((applied, _)) => applied == revision,
            None if live.gesture.start() == step.pose => return Ok(stage),
            None => false,
        };
        let requests = live.gesture.requests(step.pose);
        let revision = if joins {
            self.commands.fold(revision, &requests)?
        } else {
            self.apply(revision, &requests)?
        };
        if let Some(live) = self.live.as_mut() {
            live.applied = Some((revision, step.pose));
        }
        Ok(stage)
    }

    /// End the preview gesture. A gesture that changed nothing leaves no step.
    pub fn stage_end(&mut self) {
        self.live = None;
    }

    /// The keyframes of the picture or title clip `segment_id`, and the one
    /// within a frame of `time`. Other clips have none to show.
    #[must_use]
    pub fn keyframes(&self, segment_id: &SegmentId, time: Micros) -> Option<PhoneKeys> {
        cutlass_editor::clip_keys(self.draft(), segment_id, time, &self.sizes())
    }

    /// Ease the span of the clip's keyframes that `time` is in along the
    /// Graphs panel curve `graph`, or make it straight with `None`. One undo
    /// step.
    pub fn ease_keyframes(
        &mut self,
        segment_id: &SegmentId,
        time: Micros,
        graph: Option<String>,
    ) -> Result<(), FfiError> {
        let keys = self
            .keyframes(segment_id, time)
            .ok_or_else(|| FfiError::Refused {
                reason: "only a picture or title takes keyframes".to_owned(),
            })?;
        let request = keys
            .ease(segment_id, graph)
            .ok_or_else(|| FfiError::Refused {
                reason: "move the playhead between two keyframes to ease them".to_owned(),
            })?;
        self.apply(self.revision(), &[request])?;
        Ok(())
    }

    /// Remove the keyframe within a frame of `time` on the clip, or key the
    /// clip's pose and opacity at `time`. One undo step.
    pub fn toggle_keyframe(
        &mut self,
        segment_id: &SegmentId,
        time: Micros,
    ) -> Result<(), FfiError> {
        let keys = self
            .keyframes(segment_id, time)
            .ok_or_else(|| FfiError::Refused {
                reason: "only a picture or title takes keyframes".to_owned(),
            })?;
        let request = keys
            .toggle(segment_id, time)
            .ok_or_else(|| FfiError::Refused {
                reason: "move the playhead onto the clip to key it".to_owned(),
            })?;
        self.apply(self.revision(), &[request])?;
        Ok(())
    }

    /// The picture or title clip's in and out animations.
    #[must_use]
    pub fn animations(&self, segment_id: &SegmentId) -> Option<PhoneAnimations> {
        cutlass_editor::clip_animations(self.draft(), segment_id, &self.sizes())
    }

    /// Give the clip the in (`enter`) or out animation `resource_id`,
    /// `duration` long, or clear that side with `None`. One undo step.
    pub fn set_animation(
        &mut self,
        segment_id: &SegmentId,
        enter: bool,
        resource_id: Option<String>,
        duration: Micros,
    ) -> Result<(), FfiError> {
        if self.animations(segment_id).is_none() {
            return Err(FfiError::Refused {
                reason: "only a picture or title takes animations".to_owned(),
            });
        }
        let kind = if enter {
            cutlass_model::AnimType::In
        } else {
            cutlass_model::AnimType::Out
        };
        self.apply(
            self.revision(),
            &[Request::SetAnimation {
                segment_id: segment_id.clone(),
                kind,
                animation: resource_id,
                duration,
            }],
        )?;
        Ok(())
    }

    fn sizes(&self) -> stage::Sizes<'_> {
        stage::Sizes {
            draft: self.draft(),
            fonts: &self.faces,
        }
    }

    /// The composited picture at `time`, reading media from the project folder.
    pub fn frame(&mut self, time: Micros) -> Result<Picture, FfiError> {
        self.preview.frame(
            self.commands.session(),
            self.project.root(),
            &self.faces,
            time,
        )
    }

    /// Write the whole cut to `destination` as an MP4 at `short_side`, one
    /// of [`PHONE_EXPORT_SIDES`]. Play stops first. `progress` hears the
    /// frames done and the frames in all, and stops the export by returning
    /// false; nothing is left at `destination` then.
    pub fn export(
        &mut self,
        destination: &Path,
        short_side: u32,
        progress: &mut dyn FnMut(u64, u64) -> bool,
    ) -> Result<PhoneExport, FfiError> {
        self.pause();
        let draft = self.commands.draft();
        let readers = self.preview.readers().clone();
        let root = self.project.root().to_path_buf();
        let renderer = self.preview.renderer()?;
        export::export_movie(
            &export::Sources {
                draft,
                root: &root,
                readers: &readers,
                fonts: &self.faces,
                renderer: Some(renderer),
            },
            destination,
            short_side,
            progress,
        )
    }

    /// Look for faces in `folders` and the folders below them, replacing any
    /// found before. Only each file's first bytes are read now.
    pub fn set_font_folders(&mut self, folders: &[PathBuf]) {
        self.faces = FaceTable::scan(folders);
    }

    /// The faces text can be set in, by file name.
    #[must_use]
    pub fn fonts(&self) -> Vec<PhoneFont> {
        text::fonts(self.faces.labelled())
    }

    /// Add `content` at `at` in `face`, or in the first face (a common one
    /// when found) when `face` is empty, as one undo step. Returns the new
    /// clip's id.
    pub fn add_text(
        &mut self,
        content: &str,
        face: &str,
        at: Micros,
    ) -> Result<SegmentId, FfiError> {
        let face = if face.is_empty() {
            self.faces.name()
        } else {
            face
        };
        if face.is_empty() {
            return Err(FfiError::Refused {
                reason: "no fonts were found".to_owned(),
            });
        }
        let (request, segment_id) = text::add_request(self.draft(), content, face, at);
        self.apply(self.revision(), &[request])?;
        Ok(segment_id)
    }

    /// Type into a title as the words change. With no `segment_id`, the
    /// first words that are not blank add one at `at` in `face`, as
    /// [`Self::add_text`] does. The first change is an undo step and the
    /// rest of the typing joins it. Blank words change nothing. Returns the
    /// clip, or `None` when nothing has been added yet.
    pub fn type_text(
        &mut self,
        segment_id: Option<&SegmentId>,
        content: &str,
        face: &str,
        at: Micros,
    ) -> Result<Option<SegmentId>, FfiError> {
        if content.trim().is_empty() {
            return Ok(segment_id.cloned());
        }
        let Some(segment_id) = segment_id else {
            let added = self.add_text(content, face, at)?;
            self.typing = Some((added.clone(), self.revision()));
            return Ok(Some(added));
        };
        if self
            .text(segment_id)
            .is_some_and(|text| text.content == content)
        {
            return Ok(Some(segment_id.clone()));
        }
        let requests = [Request::SetTextContent {
            segment_id: segment_id.clone(),
            content: content.to_owned(),
        }];
        let revision = self.revision();
        let joins = self
            .typing
            .as_ref()
            .is_some_and(|(typed, at)| typed == segment_id && *at == revision);
        let revision = if joins {
            self.commands.fold(revision, &requests)?
        } else {
            self.apply(revision, &requests)?
        };
        self.typing = Some((segment_id.clone(), revision));
        Ok(Some(segment_id.clone()))
    }

    /// Add the bundled effect `effect_id` at `at` over everything below it,
    /// as one undo step. Returns the new clip's id.
    pub fn add_effect(&mut self, effect_id: &str, at: Micros) -> Result<SegmentId, FfiError> {
        let (request, segment_id) = effect::add_request(self.draft(), effect_id, at);
        self.apply(self.revision(), &[request])?;
        Ok(segment_id)
    }

    /// Run effect `effect_id` over the picture clip `clip` alone, for its
    /// span and linked to it, as one undo step. Returns the effect clip.
    pub fn add_clip_effect(
        &mut self,
        clip: &SegmentId,
        effect_id: &str,
    ) -> Result<SegmentId, FfiError> {
        let (requests, segment_id) = look::clip_effect_requests(self.draft(), clip, effect_id)?;
        self.apply(self.revision(), &requests)?;
        Ok(segment_id)
    }

    /// Give the picture clip `clip` the filter `filter_id` over its span,
    /// swapping out the one it kept, or take its filter off with an empty
    /// id; a filter layer swaps in place. One undo step. Returns the new
    /// filter layer, if one was placed.
    pub fn set_clip_filter(
        &mut self,
        clip: &SegmentId,
        filter_id: &str,
    ) -> Result<Option<SegmentId>, FfiError> {
        let (requests, placed) = look::clip_filter_requests(self.draft(), clip, filter_id)?;
        if requests.is_empty() {
            return Ok(None);
        }
        self.apply(self.revision(), &requests)?;
        Ok(placed)
    }

    /// Put the filter layer `filter_id` at `at`, over every picture below
    /// it, as one undo step. Returns the layer.
    pub fn add_filter(&mut self, filter_id: &str, at: Micros) -> Result<SegmentId, FfiError> {
        let (request, segment_id) = look::filter_at_request(self.draft(), filter_id, at)?;
        self.apply(self.revision(), &[request])?;
        Ok(segment_id)
    }

    /// Set how strongly the filter, effect or adjustment `segment_id`
    /// works, 0 to 1, as one undo step.
    pub fn set_strength(&mut self, segment_id: &SegmentId, value: f64) -> Result<(), FfiError> {
        clip::unlocked(self.draft(), segment_id)?;
        if !(0.0..=1.0).contains(&value) {
            return Err(FfiError::Refused {
                reason: "strength runs from 0 to 1".to_owned(),
            });
        }
        let material_id = cutlass_model::MaterialId::new(
            cutlass_editor::FreshIds::new(self.draft(), 0).make("filter"),
        );
        self.apply(
            self.revision(),
            &[Request::SetEffectStrength {
                segment_id: segment_id.clone(),
                material_id,
                value,
            }],
        )?;
        Ok(())
    }

    /// Put an adjustment layer at `at`, its sliders at zero, as one undo
    /// step. Returns the layer.
    pub fn add_adjustment(&mut self, at: Micros) -> Result<SegmentId, FfiError> {
        let (request, segment_id) = look::adjustment_request(self.draft(), at);
        self.apply(self.revision(), &[request])?;
        Ok(segment_id)
    }

    /// Put sticker `id` at `at` on the first sticker row with room, else
    /// on a new one, as one undo step: an animation lasts whole rounds.
    /// Its file is written into the project the first time.
    pub fn add_sticker(&mut self, id: &str, at: Micros) -> Result<SegmentId, FfiError> {
        let (requests, segment_id) =
            layer::sticker_at_requests(self.draft(), self.root(), id, at, self.linkage().enabled)?;
        self.apply(self.revision(), &requests)?;
        Ok(segment_id)
    }

    /// Put shape `card_id` at `at` on the first sticker row with room,
    /// else on a new one, as one undo step.
    pub fn add_shape(&mut self, card_id: &str, at: Micros) -> Result<SegmentId, FfiError> {
        let (requests, segment_id) = layer::shape_at_request(self.draft(), card_id, at)?;
        self.apply(self.revision(), &requests)?;
        Ok(segment_id)
    }

    /// Put a freehand stroke through `trace`, canvas pixels from the top
    /// left, y down, at `at` as a smooth line on a sticker row: one undo
    /// step. A stroke that goes nowhere is refused.
    pub fn draw_stroke(&mut self, trace: &[[f64; 2]], at: Micros) -> Result<SegmentId, FfiError> {
        let (requests, segment_id) = layer::stroke_request(self.draft(), trace, at)?;
        self.apply(self.revision(), &requests)?;
        Ok(segment_id)
    }

    /// The cuts between touching clips on the main row, with their blends.
    #[must_use]
    pub fn cuts(&self) -> Vec<PhoneCut> {
        phone_cuts(self.draft())
    }

    /// Put transition `card_id` on the cut after `outgoing_id`, fitted to
    /// the clips; a blend already there is swapped. One undo step.
    pub fn set_transition(
        &mut self,
        outgoing_id: &SegmentId,
        card_id: &str,
    ) -> Result<(), FfiError> {
        let request = transition::set_request(self.draft(), outgoing_id, card_id)?;
        self.apply(self.revision(), &[request])?;
        Ok(())
    }

    /// Make the blend after `outgoing_id` last `duration`, held to its
    /// range. One undo step; nothing when it lasts that long already.
    pub fn set_transition_length(
        &mut self,
        outgoing_id: &SegmentId,
        duration: Micros,
    ) -> Result<(), FfiError> {
        if let Some(request) = transition::length_request(self.draft(), outgoing_id, duration)? {
            self.apply(self.revision(), &[request])?;
        }
        Ok(())
    }

    /// Copy the blend after `outgoing_id` to every other cut of the main
    /// row, each fitted to its clips, as one undo step.
    pub fn transition_everywhere(&mut self, outgoing_id: &SegmentId) -> Result<(), FfiError> {
        let requests = transition::everywhere_requests(self.draft(), outgoing_id)?;
        if !requests.is_empty() {
            self.apply(self.revision(), &requests)?;
        }
        Ok(())
    }

    /// Take the blend off the cut after `outgoing_id`. One undo step.
    pub fn remove_transition(&mut self, outgoing_id: &SegmentId) -> Result<(), FfiError> {
        let request = transition::remove_request(self.draft(), outgoing_id)?;
        self.apply(self.revision(), &[request])?;
        Ok(())
    }

    /// Set the adjust slider `key` of the picture clip or adjustment layer
    /// `segment_id` to `value`, as one undo step.
    pub fn set_adjust(
        &mut self,
        segment_id: &SegmentId,
        key: &str,
        value: f64,
    ) -> Result<(), FfiError> {
        let request = look::adjust_request(self.draft(), segment_id, key, value)?;
        self.apply(self.revision(), &[request])?;
        Ok(())
    }

    /// Every adjust slider of `segment_id`, or `None` for a clip that has
    /// none.
    #[must_use]
    pub fn adjust_sliders(&self, segment_id: &SegmentId) -> Option<Vec<PhoneAdjust>> {
        adjust_sliders(self.draft(), segment_id)
    }

    /// The filter `clip` keeps, or a filter layer's own.
    #[must_use]
    pub fn clip_filter(&self, clip: &SegmentId) -> Option<PhoneClipFilter> {
        clip_filter(self.draft(), clip)
    }

    /// Bring in the subtitle file `file_name` (SRT or LRC, by its
    /// extension), whose text the phone read, as captions on a new subtitle
    /// lane: one undo step. A file that does not read is refused with why.
    pub fn import_subtitles(&mut self, file_name: &str, text: &str) -> Result<(), FfiError> {
        let refused = |reason: String| FfiError::Refused { reason };
        let format = cutlass_editor::SubtitleFormat::of_file(file_name)
            .ok_or_else(|| refused(format!("{file_name} is not a subtitle file")))?;
        let mut taken = media::ids(self.draft());
        let mut next = std::collections::HashMap::<String, u64>::new();
        let mut fresh = |prefix: &str| {
            let n = next.entry(prefix.to_owned()).or_insert(1);
            loop {
                let id = format!("{prefix}-{n}");
                *n += 1;
                if taken.insert(id.clone()) {
                    return id;
                }
            }
        };
        // Captions are set in the first face, as a new title is.
        let face = self.faces.name().to_owned();
        let request = cutlass_editor::subtitle_request(text, format, &face, &mut fresh)
            .map_err(|miss| refused(miss.to_string()))?;
        self.apply(self.revision(), &[request])?;
        Ok(())
    }

    /// Change what `segment_id` says, as one undo step. Blank words are
    /// refused; delete the clip instead.
    pub fn set_text(&mut self, segment_id: &SegmentId, content: &str) -> Result<(), FfiError> {
        if content.trim().is_empty() {
            return Err(cutlass_session::EditError::TextEmpty.into());
        }
        self.apply(
            self.revision(),
            &[Request::SetTextContent {
                segment_id: segment_id.clone(),
                content: content.to_owned(),
            }],
        )?;
        Ok(())
    }

    /// Set `segment_id` in `face`, as one undo step.
    pub fn set_text_font(&mut self, segment_id: &SegmentId, face: &str) -> Result<(), FfiError> {
        self.apply(
            self.revision(),
            &[Request::SetTextFont {
                segment_id: segment_id.clone(),
                face: face.to_owned(),
            }],
        )?;
        Ok(())
    }

    /// Change how `segment_id` looks. Changes made while `live` join one
    /// undo step, so a slider drag is one step; its last change passes
    /// `live` false, and so does a single tap.
    pub fn style_text(
        &mut self,
        segment_id: &SegmentId,
        change: TextStyleChange,
        live: bool,
    ) -> Result<(), FfiError> {
        let requests = [text::style_request(self.draft(), segment_id, change)];
        let revision = self.revision();
        let joins = self
            .styling
            .as_ref()
            .is_some_and(|(styled, at)| styled == segment_id && *at == revision);
        let revision = if joins {
            self.commands.fold(revision, &requests)?
        } else {
            self.apply(revision, &requests)?
        };
        self.styling = live.then(|| (segment_id.clone(), revision));
        Ok(())
    }

    /// What `segment_id` says, or `None` when it is not a text clip.
    #[must_use]
    pub fn text(&self, segment_id: &SegmentId) -> Option<PhoneText> {
        text::text_of(self.draft(), segment_id)
    }

    /// The rows to draw, top first.
    #[must_use]
    pub fn lanes(&self) -> Vec<PhoneLane> {
        phone_lanes(self.draft())
    }

    /// The tool bar for `segment_id` while it is selected, or `None` for an
    /// id no row holds.
    #[must_use]
    pub fn clip_bar(&self, segment_id: &SegmentId) -> Option<Bar> {
        self.lanes().into_iter().find_map(|lane| {
            lane.clips
                .iter()
                .any(|clip| &clip.id == segment_id)
                .then(|| cutlass_editor::clip_bar(lane.kind, lane.main))
        })
    }

    /// Cut `segment_id` at `at`. Returns the right piece's id.
    pub fn split(&mut self, segment_id: &SegmentId, at: Micros) -> Result<SegmentId, FfiError> {
        let (request, right_id) = timeline::split_request(self.draft(), segment_id, at)?;
        self.apply(self.revision(), &[request])?;
        Ok(right_id)
    }

    /// Move `segment_id` to the row and start in `drag`, as one undo step.
    /// Returns where it starts once the session placed it: snapped, and
    /// packed when the row is magnetic.
    pub fn move_clip(&mut self, segment_id: &SegmentId, drag: Drag) -> Result<Micros, FfiError> {
        let (request, asked) = timeline::move_request(self.draft(), segment_id, drag)?;
        self.apply(self.revision(), &[request])?;
        Ok(self
            .draft()
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| &segment.id == segment_id)
            .map_or(asked, |segment| segment.target.start))
    }

    /// Drag the start or end edge of `segment_id` to `drag.time`, as one undo
    /// step. Returns where the edge landed.
    pub fn trim(
        &mut self,
        segment_id: &SegmentId,
        start: bool,
        drag: Drag,
    ) -> Result<Micros, FfiError> {
        let (request, time) = timeline::trim_request(self.draft(), segment_id, start, drag)?;
        self.apply(self.revision(), &[request])?;
        Ok(time)
    }

    /// Set the linear gain of `segment_id`, as one undo step. A negative or
    /// unfinite gain is refused.
    pub fn set_volume(&mut self, segment_id: &SegmentId, volume: f64) -> Result<(), FfiError> {
        self.apply(
            self.revision(),
            &[Request::SetVolume {
                segment_id: segment_id.clone(),
                volume,
            }],
        )?;
        Ok(())
    }

    /// Play `segment_id` at a constant `speed`, as one undo step. Its length
    /// follows the speed, and a magnetic row packs behind it.
    pub fn set_speed(&mut self, segment_id: &SegmentId, speed: f64) -> Result<(), FfiError> {
        self.apply(
            self.revision(),
            &[Request::SetConstantSpeed {
                segment_id: segment_id.clone(),
                speed,
            }],
        )?;
        Ok(())
    }

    /// Put `segment_id` at `place` in the main row's order, counted from
    /// zero without it, as one undo step. Linkage follows as for any move. A
    /// clip that is not on main is refused.
    pub fn reorder(&mut self, segment_id: &SegmentId, place: u32) -> Result<(), FfiError> {
        let refused = || FfiError::Refused {
            reason: "only a main clip changes places".to_owned(),
        };
        let track = self
            .draft()
            .tracks
            .iter()
            .find(|track| track.is_main())
            .ok_or_else(refused)?;
        let start =
            cutlass_editor::reorder_start(track, segment_id, place as usize).ok_or_else(refused)?;
        let request = Request::Move {
            segment_id: segment_id.clone(),
            track_id: track.id.clone(),
            start,
        };
        self.apply(self.revision(), &[request])?;
        Ok(())
    }

    /// Ruler ticks from `from` through `to` at `pixel_size` points per
    /// microsecond.
    #[must_use]
    pub fn ruler(&self, pixel_size: f64, from: Micros, to: Micros) -> Vec<PhoneTick> {
        strip::ruler(self.draft(), pixel_size, from, to)
    }

    /// `pixel_size` times `factor`, held inside what the cut and a row
    /// `lane_width` points wide allow.
    #[must_use]
    pub fn zoom(&self, pixel_size: f64, factor: f64, lane_width: f64) -> f64 {
        strip::zoom(self.draft(), pixel_size, factor, lane_width)
    }

    /// Filmstrip tiles for the picture clips that meet `from` to `to`.
    pub fn film(
        &mut self,
        from: Micros,
        to: Micros,
        pixel_size: f64,
        height: f64,
    ) -> Vec<PhoneFilm> {
        let draft = self.commands.draft();
        self.strip.film(
            draft,
            self.project.root(),
            self.preview.readers(),
            from,
            to,
            pixel_size,
            height,
        )
    }

    /// The waveforms of the clips with sound that meet `from` to `to`.
    pub fn waves(&mut self, from: Micros, to: Micros, pixel_size: f64) -> Vec<PhoneWave> {
        let draft = self.commands.draft();
        self.strip.waves(
            draft,
            self.project.root(),
            self.preview.readers(),
            from,
            to,
            pixel_size,
        )
    }

    /// The picture of a tile [`Self::film`] named, once stored.
    #[must_use]
    pub fn thumb(&self, key: &str) -> Option<Picture> {
        self.strip.thumb(key)
    }

    /// Remove one clip. Its dependents go with it when linkage follow is on.
    pub fn remove(&mut self, segment_id: &SegmentId) -> Result<(), FfiError> {
        self.apply(
            self.revision(),
            &[Request::Remove {
                segment_id: segment_id.clone(),
            }],
        )?;
        Ok(())
    }

    /// How `segment_id` stands at `time`, for its clip tools. `None` when
    /// no row holds it.
    #[must_use]
    pub fn clip_state(&self, segment_id: &SegmentId, time: Micros) -> Option<PhoneClipState> {
        clip_state(self.draft(), segment_id, time)
    }

    fn picture_clip(&self, segment_id: &SegmentId, what: &str) -> Result<(), FfiError> {
        let (segment, track) = clip::unlocked(self.draft(), segment_id)?;
        if cutlass_editor::can_turn(self.draft(), segment, &track.kind) {
            Ok(())
        } else {
            Err(FfiError::Refused {
                reason: format!("{what} belongs on a picture"),
            })
        }
    }

    /// Flip the picture left to right, or back, as one undo step.
    pub fn toggle_mirror(&mut self, segment_id: &SegmentId) -> Result<(), FfiError> {
        self.picture_clip(segment_id, "a mirror")?;
        let horizontal = !clip::locate(self.draft(), segment_id)
            .is_some_and(|(segment, _)| segment.clip.flip.horizontal);
        self.apply(
            self.revision(),
            &[Request::SetMirror {
                segment_id: segment_id.clone(),
                horizontal,
            }],
        )?;
        Ok(())
    }

    /// Turn the picture a quarter turn clockwise, as one undo step. Four
    /// turns bring it upright.
    pub fn turn(&mut self, segment_id: &SegmentId) -> Result<(), FfiError> {
        self.picture_clip(segment_id, "a turn")?;
        let degrees = clip::locate(self.draft(), segment_id)
            .map_or(0.0, |(segment, _)| clockwise_quarter(segment.clip.rotation));
        self.apply(
            self.revision(),
            &[Request::SetRotation {
                segment_id: segment_id.clone(),
                degrees,
            }],
        )?;
        Ok(())
    }

    /// Play a picture or a sound backward through its own file, or forward
    /// again, as one undo step.
    pub fn toggle_reverse(&mut self, segment_id: &SegmentId) -> Result<(), FfiError> {
        let (segment, _) = clip::unlocked(self.draft(), segment_id)?;
        let plays = matches!(
            self.draft().materials.get(segment.material_id.as_str()),
            Some(cutlass_model::MaterialRef::Video(_) | cutlass_model::MaterialRef::Audio(_))
        );
        if !plays {
            return Err(FfiError::Refused {
                reason: "reverse belongs on a picture or a sound".to_owned(),
            });
        }
        let reverse = !segment.reverse;
        self.apply(
            self.revision(),
            &[Request::SetReverse {
                segment_id: segment_id.clone(),
                reverse,
                path: String::new(),
            }],
        )?;
        Ok(())
    }

    /// Crop the video to its middle at `ratio`, one of [`CROP_RATIOS`], or
    /// show its whole frame for an empty ratio, as one undo step.
    pub fn set_crop(&mut self, segment_id: &SegmentId, ratio: &str) -> Result<(), FfiError> {
        let request = clip::crop_request(self.draft(), segment_id, ratio)?;
        self.apply(self.revision(), &[request])?;
        Ok(())
    }

    /// Make the picture `opacity` solid, from 0 clear to 1, as one undo step.
    pub fn set_opacity(&mut self, segment_id: &SegmentId, opacity: f64) -> Result<(), FfiError> {
        self.picture_clip(segment_id, "an opacity")?;
        if !(0.0..=1.0).contains(&opacity) {
            return Err(FfiError::Refused {
                reason: "opacity runs from 0 to 1".to_owned(),
            });
        }
        self.apply(
            self.revision(),
            &[Request::SetOpacity {
                segment_id: segment_id.clone(),
                opacity,
            }],
        )?;
        Ok(())
    }

    /// Fade a picture or a sound in over `fade_in` and out over `fade_out`,
    /// as one undo step.
    pub fn set_fade(
        &mut self,
        segment_id: &SegmentId,
        fade_in: Micros,
        fade_out: Micros,
    ) -> Result<(), FfiError> {
        let (segment, _) = clip::unlocked(self.draft(), segment_id)?;
        if fade_in.0 < 0 || fade_out.0 < 0 {
            return Err(FfiError::Refused {
                reason: "a fade cannot be shorter than nothing".to_owned(),
            });
        }
        if fade_in.0.saturating_add(fade_out.0) > segment.target.duration.0 {
            return Err(FfiError::Refused {
                reason: "the fades are longer than the clip".to_owned(),
            });
        }
        if clip::fades(self.draft(), segment) == (fade_in.0, fade_out.0) {
            return Ok(());
        }
        let fade_id = cutlass_model::MaterialId::new(
            cutlass_editor::FreshIds::new(self.draft(), 0).make("fade"),
        );
        self.apply(
            self.revision(),
            &[Request::SetFade {
                segment_id: segment_id.clone(),
                fade_id,
                fade_in,
                fade_out,
            }],
        )?;
        Ok(())
    }

    /// Put a copy of the clip right after it, as one undo step, and return
    /// the copy. On main the clips after it move along; elsewhere the copy
    /// goes on the first clear row of its kind, as a paste does.
    pub fn duplicate(&mut self, segment_id: &SegmentId) -> Result<SegmentId, FfiError> {
        let draft = self.draft();
        let (segment, track) = clip::unlocked(draft, segment_id)?;
        let at = segment.target.end();
        let copies =
            cutlass_editor::copy_clips(draft, std::slice::from_ref(segment_id)).map_err(|_| {
                FfiError::Refused {
                    reason: "the clip cannot be copied".to_owned(),
                }
            })?;
        let mut ids = cutlass_editor::FreshIds::new(draft, 0);
        let (requests, copy) = if track.is_main() {
            let mut placed = copies
                .first()
                .map(|copy| copy.segment.clone())
                .ok_or_else(|| FfiError::Refused {
                    reason: "the clip cannot be copied".to_owned(),
                })?;
            let id = SegmentId::new(ids.make("seg"));
            placed.id = id.clone();
            placed.target.start = at;
            let request = Request::Place {
                track_id: track.id.clone(),
                segment: Box::new(placed),
            };
            (vec![request], id)
        } else {
            let mut fresh = |prefix: &str| ids.make(prefix);
            let (requests, pasted) = cutlass_editor::paste_requests(draft, &copies, at, &mut fresh)
                .map_err(|_| FfiError::Refused {
                    reason: "the clip cannot be copied".to_owned(),
                })?;
            let copy = pasted.into_iter().next().ok_or_else(|| FfiError::Refused {
                reason: "the clip cannot be copied".to_owned(),
            })?;
            (requests, copy)
        };
        self.apply(self.revision(), &requests)?;
        Ok(copy)
    }

    /// Hold the frame of the video under `at` as a still for three seconds
    /// there, splitting the clip around it, as one undo step. The still is
    /// saved as a picture in the project and its clip returned.
    pub fn freeze(&mut self, segment_id: &SegmentId, at: Micros) -> Result<SegmentId, FfiError> {
        let refused = |reason: &str| FfiError::Refused {
            reason: reason.to_owned(),
        };
        clip::unlocked(self.draft(), segment_id)?;
        let point = cutlass_eval::freeze_point(self.draft(), segment_id, at).map_err(|miss| {
            refused(match miss {
                cutlass_eval::FreezeMiss::UnknownSegment => "the clip is not on the timeline",
                cutlass_eval::FreezeMiss::NotVideo => "a freeze belongs on video",
                cutlass_eval::FreezeMiss::NoFrame => "the clip has no frame to hold",
                cutlass_eval::FreezeMiss::Outside => "move the playhead onto the clip",
                cutlass_eval::FreezeMiss::AlreadyHeld => "the clip is already a still",
            })
        })?;
        let source = self.media_file(&point.freeze.source_material_path)?;
        let image = cutlass_eval::freeze_image(
            self.preview.readers(),
            &encoders(),
            &source,
            point.freeze.timestamp,
        )
        .ok_or_else(|| refused("the clip's frame could not be read"))?;
        let name = format!(
            "freeze-{}-{}.png",
            segment_id.as_str(),
            point.freeze.timestamp.0 / 1000
        );
        let (stored, written) = media::write_in(self.root(), &name, &image.png)?;
        let size = |side: u32| i32::try_from(side).unwrap_or(i32::MAX);
        let mut ids = cutlass_editor::FreshIds::new(self.draft(), 0);
        let mut fresh = |prefix: &str| ids.make(prefix);
        let requests = cutlass_editor::freeze_requests(
            self.draft(),
            segment_id,
            point.at,
            point.freeze,
            cutlass_editor::FreezeImage {
                path: stored,
                width: size(image.width),
                height: size(image.height),
            },
            cutlass_eval::FREEZE_DURATION,
            &mut fresh,
        );
        let Some(requests) = requests else {
            let _ = std::fs::remove_file(&written);
            return Err(refused("the clip is not on the timeline"));
        };
        let still = requests.iter().rev().find_map(|request| match request {
            Request::Place { segment, .. } => Some(segment.id.clone()),
            _ => None,
        });
        if let Err(error) = self.apply(self.revision(), &requests) {
            let _ = std::fs::remove_file(&written);
            return Err(error);
        }
        still.ok_or_else(|| refused("the still was not placed"))
    }

    /// The file a stored media path names: inside the project when it is
    /// relative, as it is when absolute.
    fn media_file(&self, stored: &str) -> Result<PathBuf, FfiError> {
        if stored.is_empty() {
            return Err(FfiError::Refused {
                reason: "the clip has no file".to_owned(),
            });
        }
        let path = Path::new(stored);
        if path.is_absolute() {
            Ok(path.to_path_buf())
        } else {
            self.resolve(path)
        }
    }

    /// Put the video's sound on a clip of its own on a sound row, lined up
    /// with it, as one undo step. A video with no sound is refused.
    pub fn extract_audio(&mut self, segment_id: &SegmentId) -> Result<(), FfiError> {
        let (segment, _) = clip::unlocked(self.draft(), segment_id)?;
        let Some(cutlass_model::MaterialRef::Video(video)) =
            self.draft().materials.get(segment.material_id.as_str())
        else {
            return Err(FfiError::Refused {
                reason: "an extract belongs on video".to_owned(),
            });
        };
        let path = if segment.reverse && !video.reverse_path.is_empty() {
            video.reverse_path.clone()
        } else {
            video.path.clone()
        };
        let file = self.media_file(&path)?;
        let heard = self.preview.readers().carries_sound(&file).unwrap_or(false);
        if !heard {
            return Err(FfiError::Refused {
                reason: "the clip has no sound".to_owned(),
            });
        }
        self.place_extracted(segment_id)
    }

    fn place_extracted(&mut self, segment_id: &SegmentId) -> Result<(), FfiError> {
        let mut ids = cutlass_editor::FreshIds::new(self.draft(), 0);
        let mut fresh = |prefix: &str| ids.make(prefix);
        let request = cutlass_editor::extract_request(self.draft(), segment_id, &mut fresh)
            .ok_or_else(|| FfiError::Refused {
                reason: "the clip is not on the timeline".to_owned(),
            })?;
        self.apply(self.revision(), &[request])?;
        Ok(())
    }

    /// Give a video whose sound was extracted its own sound back, as one
    /// undo step. The extracted clip goes when it is still linked to it.
    pub fn recover_audio(&mut self, segment_id: &SegmentId) -> Result<(), FfiError> {
        clip::unlocked(self.draft(), segment_id)?;
        self.apply(
            self.revision(),
            &[Request::RecoverAudio {
                segment_id: segment_id.clone(),
            }],
        )?;
        Ok(())
    }

    /// Mark a beat at `time` on a sound clip, or take off the beat already
    /// there, as one undo step.
    pub fn toggle_beat(&mut self, segment_id: &SegmentId, time: Micros) -> Result<(), FfiError> {
        let mut ids = cutlass_editor::FreshIds::new(self.draft(), 0);
        let mut fresh = |prefix: &str| ids.make(prefix);
        let request = clip::beat_request(self.draft(), segment_id, time, &mut fresh)?;
        self.apply(self.revision(), &[request])?;
        Ok(())
    }

    /// Move a picture from another row onto main, where its start puts it
    /// in the order, as one undo step.
    pub fn to_main(&mut self, segment_id: &SegmentId) -> Result<(), FfiError> {
        let state = self
            .clip_state(segment_id, Micros(0))
            .ok_or_else(|| FfiError::Refused {
                reason: "the clip is not on the timeline".to_owned(),
            })?;
        if !state.can_splice {
            return Err(FfiError::Refused {
                reason: "only a picture on another row moves to main".to_owned(),
            });
        }
        clip::unlocked(self.draft(), segment_id)?;
        let start = clip::locate(self.draft(), segment_id)
            .map_or(Micros(0), |(segment, _)| segment.target.start);
        let track_id = clip::main_row(self.draft())?;
        self.apply(
            self.revision(),
            &[Request::Move {
                segment_id: segment_id.clone(),
                track_id,
                start,
            }],
        )?;
        Ok(())
    }

    /// Put the whole of the picture `material_id` at `at` on a row above
    /// main, a new one when none is clear, as one undo step. Returns the
    /// new clip.
    pub fn add_overlay(&mut self, material_id: &str, at: Micros) -> Result<SegmentId, FfiError> {
        let (kind, duration) = cutlass_editor::material_span(self.draft(), material_id)
            .ok_or_else(|| FfiError::Refused {
                reason: "the media is not in the library".to_owned(),
            })?;
        if kind != TrackType::Video {
            return Err(FfiError::Refused {
                reason: "an overlay is a photo or a video".to_owned(),
            });
        }
        let at = Micros(at.0.max(0));
        let mut ids = cutlass_editor::FreshIds::new(self.draft(), 0);
        let (track_id, new_lane) = match clip::overlay_row(self.draft(), at, duration) {
            Some(track_id) => (track_id, None),
            None => (TrackId::new(ids.make("track")), Some(TrackType::Video)),
        };
        let place = cutlass_editor::MediaPlace {
            track_id,
            segment_id: SegmentId::new(ids.make("seg")),
            material_id: material_id.to_owned(),
            start: at,
            duration,
            new_lane,
        };
        let requests = cutlass_editor::media_requests(self.draft(), &place, self.linkage().enabled);
        self.apply(self.revision(), &requests)?;
        Ok(place.segment_id)
    }

    /// Whether clips on other rows move and go with the main clip they sit on.
    #[must_use]
    pub fn linkage_enabled(&self) -> bool {
        self.commands.linkage().enabled
    }

    /// Turn linkage on or off and keep the switch with the project. This
    /// is not an undo step.
    pub fn set_linkage_enabled(&mut self, on: bool) -> Result<(), FfiError> {
        let mut settings = self.commands.linkage();
        settings.enabled = on;
        self.commands.set_linkage(settings);
        self.project.set_linkage(settings);
        self.save()
    }

    /// Mute or unmute every clip on a row, as one undo step.
    pub fn set_row_muted(&mut self, track_id: &TrackId, on: bool) -> Result<(), FfiError> {
        self.apply(
            self.revision(),
            &[Request::SetTrackAttribute {
                track_id: track_id.clone(),
                bit: cutlass_model::TrackAttribute::MUTE,
                on,
            }],
        )?;
        Ok(())
    }

    /// Published path for a bundled or cached id. A missing id is an error.
    /// This does not start a transfer.
    pub fn resource(&self, id: &str) -> Result<PathBuf, FfiError> {
        self.cache.resolve(id)
    }
}

/// What playback hears: the session draft, read from the project folder.
fn program<'a>(commands: &'a Commands, project: &'a Project, preview: &'a Preview) -> Program<'a> {
    Program {
        draft: commands.session().draft(),
        revision: commands.revision(),
        root: Some(project.root()),
        readers: preview.readers(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use cutlass_session::{EditError, Micros, Request, TrackAttribute, TrackId};
    use cutlass_storage::StorageError;

    use super::PhoneHost;
    use crate::{FfiError, PhoneAnimation, animation_presets, start_log};

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("cutlass-ffi-host-{}-{n}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn start_log_writes_the_file() {
        let temp = Temp::new();
        let log = start_log(&temp.path).unwrap();
        let text = fs::read_to_string(&log).unwrap();
        assert!(text.contains("log file"), "{text}");
        assert!(text.contains("INFO"), "{text}");
        assert_eq!(log, temp.path.join("cutlass.log"));
    }

    struct Roots<'a> {
        temp: &'a Temp,
        project: PathBuf,
        bundle: PathBuf,
        cache: PathBuf,
    }

    impl Roots<'_> {
        fn new(temp: &Temp) -> Roots<'_> {
            let project = temp.path.join("project");
            let bundle = temp.path.join("bundle");
            let cache = temp.path.join("cache");
            fs::create_dir(&bundle).unwrap();
            Roots {
                temp,
                project,
                bundle,
                cache,
            }
        }

        fn create(&self) -> PhoneHost {
            PhoneHost::create(&self.project, "D", &self.bundle, &self.cache, 0).unwrap()
        }

        fn open(&self) -> PhoneHost {
            PhoneHost::open(&self.project, &self.bundle, &self.cache, 0).unwrap()
        }
    }

    fn mute(host: &PhoneHost, on: bool) -> Request {
        Request::SetTrackAttribute {
            track_id: host.draft().tracks[0].id.clone(),
            bit: TrackAttribute::MUTE,
            on,
        }
    }

    #[test]
    fn an_empty_project_opens() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let host = roots.create();
        assert_eq!(host.draft().id.as_str(), "D");
        assert_eq!(host.draft().tracks.len(), 1);
        assert!(host.draft().tracks[0].is_main());
        assert_eq!(host.draft().duration.0, 0);
        assert_eq!(host.revision(), 0);
        assert!(!host.linkage().enabled);
        assert!(host.report().is_empty());
        assert!(host.root().join("draft_info.json").is_file());
        assert!(!host.is_playing());
        assert_eq!(host.playhead().0, 0);
    }

    #[test]
    fn apply_one_request_and_save_keeps_the_mute() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let revision = host.apply(0, &[mute(&host, true)]).unwrap();
        assert_eq!(revision, 1);
        assert!(
            host.draft().tracks[0]
                .attribute
                .contains(TrackAttribute::MUTE)
        );
        host.save().unwrap();
        let opened = roots.open();
        assert!(
            opened.draft().tracks[0]
                .attribute
                .contains(TrackAttribute::MUTE)
        );
        assert_eq!(opened.revision(), 0);
    }

    #[test]
    fn a_failed_request_leaves_the_draft_and_the_file() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let before = host.draft().clone();
        let on_disk = fs::read(host.root().join("draft_info.json")).unwrap();
        let error = host
            .apply(
                0,
                &[
                    mute(&host, true),
                    Request::SetTrackAttribute {
                        track_id: TrackId::new("missing"),
                        bit: TrackAttribute::HIDE,
                        on: true,
                    },
                ],
            )
            .unwrap_err();
        assert!(matches!(error, FfiError::Edit(EditError::MissingTrack)));
        assert_eq!(host.draft(), &before);
        assert_eq!(host.revision(), 0);
        host.save().unwrap();
        assert_eq!(
            fs::read(host.root().join("draft_info.json")).unwrap(),
            on_disk
        );
    }

    #[test]
    fn undo_restores_the_draft() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        host.apply(0, &[mute(&host, true)]).unwrap();
        let revision = host.undo().unwrap();
        assert_eq!(revision, 2);
        assert_eq!(host.draft().tracks[0].attribute.0, 0);
        assert!(matches!(
            host.undo().unwrap_err(),
            FfiError::Edit(EditError::EmptyUndo)
        ));
    }

    #[test]
    fn a_path_that_leaves_the_project_is_refused() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let host = roots.create();
        let secret = roots.temp.path.join("secret");
        fs::write(&secret, b"known-secret").unwrap();
        let error = host.resolve(Path::new("../secret")).unwrap_err();
        assert!(matches!(
            &error,
            FfiError::PathEscapes { name } if name == "../secret"
        ));
        assert_eq!(fs::read(&secret).unwrap(), b"known-secret");
        let inside = host.resolve(Path::new("draft_info.json")).unwrap();
        assert!(inside.starts_with(host.root()));
        assert!(inside.is_file());
    }

    #[test]
    fn a_missing_storage_id_is_an_error() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let host = roots.create();
        let error = host.resource("glow").unwrap_err();
        assert!(matches!(
            &error,
            FfiError::Storage(StorageError::Missing { .. })
        ));
        assert!(error.to_string().contains("glow"), "{error}");
        assert!(
            fs::read_dir(roots.cache.join("incoming"))
                .unwrap()
                .next()
                .is_none()
        );
    }

    #[test]
    fn play_does_not_rewrite_the_draft() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let before = host.draft().clone();
        host.play().unwrap();
        assert!(!host.is_playing());
        assert_eq!(host.playhead().0, 0);
        assert_eq!(host.draft(), &before);
        assert_eq!(host.revision(), 0);
    }

    #[test]
    fn the_phone_can_name_the_draft_and_the_clock() {
        let draft = cutlass_store::Draft::empty("D");
        assert_eq!(draft.id.as_str(), "D");
        let linkage = cutlass_session::linkage_from_biz(None, draft.id.as_str(), false);
        assert!(!linkage.enabled);
        assert_eq!(Micros(5).0, 5);
    }

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../engine/fixtures")
            .join(name)
    }

    fn still(duration: i64) -> crate::Probed {
        crate::Probed::Video {
            meta: cutlass_model::MetaType::Photo,
            width: 16,
            height: 16,
            duration,
        }
    }

    #[test]
    fn an_imported_still_is_copied_in_and_appends_to_the_end_of_main() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let source = fixture("pattern.png");
        let first = host.import_probed(&source, still(2_000_000)).unwrap();
        let second = host.import_probed(&source, still(1_000_000)).unwrap();
        assert_ne!(first, second);
        let videos = &host.draft().materials.videos;
        assert_eq!(videos[0].path, "Resources/pattern.png");
        assert_eq!(videos[1].path, "Resources/pattern 2.png");
        assert_eq!(videos[0].kind, "photo");
        assert!(host.root().join("Resources/pattern 2.png").is_file());

        host.append(&first).unwrap();
        let segment = host.append(&second).unwrap();
        let main = &host.draft().tracks[0];
        assert_eq!(main.segments.len(), 2);
        assert_eq!(main.segments[1].id, segment);
        assert_eq!(
            main.segments[1].target,
            cutlass_model::TimeRange::new(2_000_000, 1_000_000)
        );
        let revision = host.revision();
        host.undo().unwrap();
        assert_eq!(host.draft().tracks[0].segments.len(), 1);
        assert!(host.revision() > revision);
    }

    #[test]
    fn the_first_picture_sizes_an_original_canvas_and_a_ratio_is_one_step() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let tall = crate::Probed::Video {
            meta: cutlass_model::MetaType::Photo,
            width: 1_080,
            height: 1_920,
            duration: 1_000_000,
        };
        let first = host.import_probed(&fixture("pattern.png"), tall).unwrap();
        host.append(&first).unwrap();
        let canvas = |host: &PhoneHost| {
            let canvas = &host.draft().canvas_config;
            (canvas.width, canvas.height, canvas.ratio.clone())
        };
        assert_eq!(canvas(&host), (1_080, 1_920, "original".to_owned()));
        let second = host
            .import_probed(&fixture("pattern.png"), still(1_000_000))
            .unwrap();
        host.append(&second).unwrap();
        assert_eq!(canvas(&host), (1_080, 1_920, "original".to_owned()));

        host.set_ratio("16:9").unwrap();
        assert_eq!(canvas(&host), (1_920, 1_080, "16:9".to_owned()));
        let revision = host.revision();
        host.set_ratio("16:9").unwrap();
        assert_eq!(host.revision(), revision);
        assert!(
            host.ratios()
                .iter()
                .any(|ratio| ratio.chosen && ratio.ratio == "16:9")
        );
        assert!(host.set_ratio("custom").is_err());

        host.set_ratio("original").unwrap();
        assert_eq!(canvas(&host), (1_080, 1_920, "original".to_owned()));
        host.undo().unwrap();
        assert_eq!(canvas(&host), (1_920, 1_080, "16:9".to_owned()));
    }

    #[test]
    fn a_preview_gesture_moves_a_picture_as_one_undo_step() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let id = host
            .import_probed(&fixture("pattern.png"), still(1_000_000))
            .unwrap();
        let clip = host.append(&id).unwrap();
        let stage = host.stage(&clip, Micros(0)).unwrap();
        let middle = (stage.canvas_width / 2.0, stage.canvas_height / 2.0);
        assert_eq!(
            host.stage_pick(Micros(0), middle.0, middle.1),
            Some(clip.clone())
        );
        assert_eq!(host.stage_pick(Micros(0), -1.0, -1.0), None);
        assert!(host.stage_move(1.0, 1.0, 1.0, 0.0).is_err());

        let steps = host.revision();
        host.stage_begin(&clip, Micros(0)).unwrap();
        host.stage_move(0.0, 0.0, 1.0, 0.0).unwrap();
        assert_eq!(host.revision(), steps);
        host.stage_move(100.0, 0.0, 1.0, 0.0).unwrap();
        let moved = host.stage_move(200.0, 0.0, 0.5, 30.0).unwrap();
        host.stage_end();
        assert!(host.revision() > steps);
        assert_eq!(
            host.stage(&clip, Micros(0)).unwrap().center_x,
            moved.center_x
        );
        assert!((moved.center_x - middle.0 - 200.0).abs() < 1e-6);
        assert!((moved.turn - 30.0).abs() < 1e-6);
        let segment = &host.draft().tracks[0].segments[0];
        assert!((segment.uniform_scale.value - 0.5).abs() < 1e-9);

        host.undo().unwrap();
        let back = host.stage(&clip, Micros(0)).unwrap();
        assert_eq!((back.center_x, back.turn), (middle.0, 0.0));
    }

    #[test]
    fn the_keyframe_button_keys_a_clip_and_a_drag_keys_it_again_later() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let id = host
            .import_probed(&fixture("pattern.png"), still(1_000_000))
            .unwrap();
        let clip = host.append(&id).unwrap();
        let none = host.keyframes(&clip, Micros(0)).unwrap();
        assert!(none.times.is_empty() && none.here.is_none());

        host.toggle_keyframe(&clip, Micros(100_000)).unwrap();
        let keyed = host.keyframes(&clip, Micros(110_000)).unwrap();
        assert_eq!(keyed.times, [Micros(100_000)]);
        assert_eq!(keyed.here, Some(Micros(100_000)));
        assert_eq!(host.keyframes(&clip, Micros(500_000)).unwrap().here, None);

        let middle = host.stage(&clip, Micros(500_000)).unwrap().center_x;
        host.stage_begin(&clip, Micros(500_000)).unwrap();
        host.stage_move(200.0, 0.0, 1.0, 0.0).unwrap();
        host.stage_end();
        let keyed = host.keyframes(&clip, Micros(500_000)).unwrap();
        assert_eq!(keyed.times, [Micros(100_000), Micros(500_000)]);
        let halfway = host.stage(&clip, Micros(300_000)).unwrap().center_x;
        assert!((halfway - middle - 100.0).abs() < 1e-6, "{halfway}");

        host.toggle_keyframe(&clip, Micros(495_000)).unwrap();
        assert_eq!(
            host.keyframes(&clip, Micros(0)).unwrap().times,
            [Micros(100_000)]
        );
        assert!(host.toggle_keyframe(&clip, Micros(2_000_000)).is_err());
        host.undo().unwrap();
        assert_eq!(host.keyframes(&clip, Micros(0)).unwrap().times.len(), 2);
    }

    #[test]
    fn a_picture_takes_an_in_and_an_out_animation_from_the_presets() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let id = host
            .import_probed(&fixture("pattern.png"), still(1_000_000))
            .unwrap();
        let clip = host.append(&id).unwrap();
        let none = host.animations(&clip).unwrap();
        assert_eq!(
            (none.enter, none.leave, none.length),
            (None, None, Micros(1_000_000))
        );

        let presets = animation_presets();
        let zoom = presets
            .iter()
            .find(|preset| !preset.enter && preset.name == "Zoom Out")
            .unwrap();
        host.set_animation(
            &clip,
            false,
            Some(zoom.resource_id.clone()),
            Micros(5_000_000),
        )
        .unwrap();
        let set = host.animations(&clip).unwrap();
        assert_eq!(
            set.leave,
            Some(PhoneAnimation {
                resource_id: zoom.resource_id.clone(),
                duration: Micros(1_000_000),
            })
        );
        assert!(
            host.set_animation(&clip, true, Some(zoom.resource_id.clone()), Micros(1))
                .is_err()
        );
        host.set_animation(&clip, false, None, Micros(0)).unwrap();
        assert_eq!(host.animations(&clip).unwrap().leave, None);
        host.undo().unwrap();
        assert!(host.animations(&clip).unwrap().leave.is_some());
    }

    #[test]
    fn a_graph_eases_the_span_the_playhead_is_in() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let id = host
            .import_probed(&fixture("pattern.png"), still(1_000_000))
            .unwrap();
        let clip = host.append(&id).unwrap();
        host.toggle_keyframe(&clip, Micros(0)).unwrap();
        let start = host.stage(&clip, Micros(0)).unwrap().center_x;
        host.stage_begin(&clip, Micros(800_000)).unwrap();
        host.stage_move(400.0, 0.0, 1.0, 0.0).unwrap();
        host.stage_end();

        let keys = host.keyframes(&clip, Micros(0)).unwrap();
        assert_eq!(
            (keys.span, keys.graph.as_deref()),
            (Some(Micros(800_000)), None)
        );
        assert_eq!(host.keyframes(&clip, Micros(800_000)).unwrap().span, None);
        assert!(host.ease_keyframes(&clip, Micros(900_000), None).is_err());

        let straight = host.stage(&clip, Micros(400_000)).unwrap().center_x;
        assert!((straight - start - 200.0).abs() < 1e-6, "{straight}");
        host.ease_keyframes(&clip, Micros(400_000), Some("Ease In".to_owned()))
            .unwrap();
        let keys = host.keyframes(&clip, Micros(400_000)).unwrap();
        assert_eq!(keys.graph.as_deref(), Some("Ease In"));
        let eased = host.stage(&clip, Micros(400_000)).unwrap().center_x;
        assert!(eased < straight - 50.0, "{eased} {straight}");
        host.undo().unwrap();
        assert_eq!(host.keyframes(&clip, Micros(400_000)).unwrap().graph, None);

        let graphs = crate::graphs();
        assert_eq!(graphs.len(), 26);
        assert_eq!(graphs[0].name, "Linear");
        for graph in &graphs {
            let (first, last) = (graph.points[0], graph.points[graph.points.len() - 1]);
            assert_eq!(first, (0.0, 0.0), "{}", graph.name);
            assert!((last.0 - 1.0).abs() < 1e-9 && (last.1 - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn a_blur_goes_behind_every_main_picture_and_a_color_takes_its_place() {
        use crate::PhoneBackground;

        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        assert!(host.set_background(&PhoneBackground::Blur(0.05)).is_err());
        for _ in 0..2 {
            let id = host
                .import_probed(&fixture("pattern.png"), still(1_000_000))
                .unwrap();
            host.append(&id).unwrap();
        }
        assert_eq!(host.background(), PhoneBackground::None);

        host.set_background(&PhoneBackground::Blur(0.05)).unwrap();
        assert_eq!(host.background(), PhoneBackground::Blur(0.05));
        let canvases = &host.draft().materials.canvases;
        assert_eq!(canvases.len(), 2);
        assert_ne!(canvases[0].id, canvases[1].id);
        let revision = host.revision();
        host.set_background(&PhoneBackground::Blur(0.05)).unwrap();
        assert_eq!(host.revision(), revision);

        host.set_background(&PhoneBackground::Color("#E5484D".to_owned()))
            .unwrap();
        assert_eq!(
            host.background(),
            PhoneBackground::Color("#E5484D".to_owned())
        );
        assert!(host.draft().materials.canvases.is_empty());
        host.undo().unwrap();
        assert_eq!(host.background(), PhoneBackground::Blur(0.05));

        host.set_background(&PhoneBackground::None).unwrap();
        assert_eq!(host.background(), PhoneBackground::None);
        assert_eq!(host.draft().canvas_config.background, None);
    }

    #[test]
    fn play_ticks_through_the_cut_and_stops_at_its_end_without_writing_the_draft() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let id = host
            .import_probed(&fixture("pattern.png"), still(100_000))
            .unwrap();
        host.append(&id).unwrap();
        let before = host.draft().clone();
        let revision = host.revision();

        host.play().unwrap();
        assert!(host.is_playing());
        assert_eq!(host.tick(40_000).0, 40_000);
        host.pause();
        assert_eq!(host.tick(40_000).0, 40_000, "a paused playhead holds");
        host.seek(Micros(90_000));
        host.play().unwrap();
        assert_eq!(host.tick(40_000).0, 100_000);
        assert!(!host.is_playing(), "play stops at the end of the cut");
        host.play().unwrap();
        assert_eq!(host.playhead().0, 0, "play from the end starts over");
        host.seek(Micros(5_000_000));
        assert_eq!(host.playhead().0, 100_000, "a seek stays inside the cut");

        assert_eq!(host.draft(), &before);
        assert_eq!(host.revision(), revision);
    }

    fn drag(row: u32, time: i64) -> crate::Drag {
        crate::Drag {
            row,
            time: Micros(time),
            snap: Micros(0),
            playhead: Micros(0),
        }
    }

    fn two_stills(
        host: &mut PhoneHost,
    ) -> (cutlass_session::SegmentId, cutlass_session::SegmentId) {
        let source = fixture("pattern.png");
        let first = host.import_probed(&source, still(2_000_000)).unwrap();
        let second = host.import_probed(&source, still(1_000_000)).unwrap();
        (host.append(&first).unwrap(), host.append(&second).unwrap())
    }

    #[test]
    fn a_transition_goes_on_a_cut_swaps_grows_spreads_and_comes_off() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (first, second) = two_stills(&mut host);
        let source = fixture("pattern.png");
        let third = host.import_probed(&source, still(1_000_000)).unwrap();
        let third = host.append(&third).unwrap();

        let shelf = crate::phone_transitions();
        assert!(shelf.len() > 20);
        assert!(
            shelf
                .iter()
                .all(|item| item.preview.starts_with("transition-"))
        );
        assert!(!crate::phone_transition_categories().is_empty());

        let cuts = host.cuts();
        assert_eq!(cuts.len(), 2);
        assert_eq!(cuts[0].outgoing_id, first);
        assert_eq!(cuts[0].at, Micros(2_000_000));
        assert_eq!(cuts[0].transition, None);
        assert_eq!(cuts[0].duration, Micros(0));

        let before = host.draft().clone();
        host.set_transition(&first, &shelf[0].id).unwrap();
        let cut = &host.cuts()[0];
        assert_eq!(cut.transition.as_deref(), Some(shelf[0].id.as_str()));
        assert_eq!(cut.name, shelf[0].name);
        assert!(cut.shortest.0 > 0 && cut.longest >= cut.duration);
        // Swapping keeps one blend on the cut.
        host.set_transition(&first, &shelf[3].id).unwrap();
        assert_eq!(host.draft().materials.transitions.len(), 1);
        assert_eq!(host.cuts()[0].name, shelf[3].name);

        host.set_transition_length(&first, Micros(800_000)).unwrap();
        assert_eq!(host.cuts()[0].duration, Micros(800_000));
        host.transition_everywhere(&first).unwrap();
        assert_eq!(host.cuts()[1].name, shelf[3].name);
        assert_eq!(host.draft().materials.transitions.len(), 2);

        host.remove_transition(&second).unwrap();
        assert_eq!(host.cuts()[1].transition, None);
        assert!(host.remove_transition(&second).is_err());
        assert!(host.set_transition(&third, &shelf[0].id).is_err());
        assert!(host.set_transition(&first, "transition:nope").is_err());
        assert!(host.set_transition_length(&second, Micros(1)).is_err());

        for _ in 0..5 {
            host.undo().unwrap();
        }
        assert_eq!(host.draft(), &before);
    }

    #[test]
    fn a_dragged_edge_trims_the_clip_as_one_undo_step() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (_, second) = two_stills(&mut host);
        let landed = host.trim(&second, false, drag(0, 2_500_000)).unwrap();
        assert_eq!(landed.0, 2_500_000);
        let main = &host.draft().tracks[0];
        assert_eq!(
            main.segments[1].target,
            cutlass_model::TimeRange::new(2_000_000, 500_000)
        );
        host.undo().unwrap();
        assert_eq!(
            host.draft().tracks[0].segments[1].target,
            cutlass_model::TimeRange::new(2_000_000, 1_000_000)
        );
        let past = host.trim(&second, true, drag(0, 9_000_000)).unwrap();
        assert!(past.0 < 3_000_000, "the start edge stays before the end");
    }

    #[test]
    fn volume_and_speed_reach_the_rows_and_a_bad_value_changes_nothing() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (first, second) = two_stills(&mut host);
        host.set_volume(&first, 0.5).unwrap();
        let clip = &host.lanes()[0].clips[0];
        assert_eq!(clip.volume, 0.5);
        host.undo().unwrap();
        assert_eq!(host.lanes()[0].clips[0].volume, 1.0);

        let before = host.draft().clone();
        let revision = host.revision();
        for volume in [-1.0, f64::NAN] {
            assert!(host.set_volume(&first, volume).is_err());
        }
        for speed in [0.0, -2.0, f64::NAN] {
            assert!(host.set_speed(&first, speed).is_err());
        }
        assert!(
            host.set_volume(&cutlass_session::SegmentId::new("nope"), 0.5)
                .is_err()
        );
        assert_eq!(host.draft(), &before);
        assert_eq!(host.revision(), revision);

        host.set_speed(&first, 2.0).unwrap();
        let clips = &host.lanes()[0].clips;
        assert_eq!(clips[0].speed, 2.0);
        assert_eq!(clips[0].duration.0, 1_000_000);
        // The magnet packs the next clip against the shorter one.
        assert_eq!(clips[1].id, second);
        assert_eq!(clips[1].start.0, 1_000_000);
    }

    #[test]
    fn a_main_clip_changes_places_as_one_undo_step() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (first, second) = two_stills(&mut host);
        host.reorder(&first, 1).unwrap();
        let order: Vec<_> = host.lanes()[0]
            .clips
            .iter()
            .map(|clip| clip.id.clone())
            .collect();
        assert_eq!(order, [second.clone(), first.clone()]);
        assert_eq!(host.lanes()[0].clips[0].start.0, 0);
        host.undo().unwrap();
        let order: Vec<_> = host.lanes()[0]
            .clips
            .iter()
            .map(|clip| clip.id.clone())
            .collect();
        assert_eq!(order, [first, second]);

        let before = host.draft().clone();
        let error = host
            .reorder(&cutlass_session::SegmentId::new("nope"), 0)
            .unwrap_err();
        assert!(matches!(error, FfiError::Refused { .. }), "{error:?}");
        assert_eq!(host.draft(), &before);
    }

    /// A folder holding the repo's sample face as `Sample.ttf`.
    fn font_folder(temp: &Temp) -> PathBuf {
        let folder = temp.path.join("fonts");
        fs::create_dir_all(&folder).unwrap();
        fs::write(
            folder.join("Sample.ttf"),
            include_bytes!("../../font/fixtures/sample.ttf"),
        )
        .unwrap();
        folder
    }

    #[test]
    fn text_goes_on_a_free_text_row_in_a_found_face_and_draws() {
        use crate::{Bar, NEW_TEXT_LENGTH, PhoneFont, PhoneText, SegmentId};

        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        two_stills(&mut host);
        let plain = host.frame(Micros(500_000)).unwrap();

        host.set_font_folders(&[font_folder(&temp)]);
        assert_eq!(
            host.fonts(),
            [PhoneFont {
                name: "Sample.ttf".to_owned(),
                label: "Sample".to_owned(),
            }]
        );
        let first = host.add_text("A", "", Micros(0)).unwrap();
        let PhoneText { content, face, .. } = host.text(&first).unwrap();
        assert_eq!((content.as_str(), face.as_str()), ("A", "Sample.ttf"));
        // The sample face has one letter, A.
        assert!(host.frame(Micros(500_000)).unwrap() != plain);

        // Room after the first clip: same row. Overlapping it: a new row.
        let after = host.add_text("Later", "", NEW_TEXT_LENGTH).unwrap();
        let over = host
            .add_text("Over", "Sample.ttf", Micros(1_000_000))
            .unwrap();
        let row_of = |host: &PhoneHost, id: &SegmentId| {
            host.lanes()
                .into_iter()
                .position(|lane| lane.clips.iter().any(|clip| &clip.id == id))
                .unwrap()
        };
        assert_eq!(row_of(&host, &first), row_of(&host, &after));
        assert_ne!(row_of(&host, &first), row_of(&host, &over));
        assert_eq!(host.clip_bar(&first), Some(Bar::TextClip));

        host.set_text(&first, "Hello").unwrap();
        assert_eq!(host.text(&first).unwrap().content, "Hello");
        host.undo().unwrap();
        assert_eq!(host.text(&first).unwrap().content, "A");
    }

    #[test]
    fn a_slider_drag_restyles_a_title_in_one_step() {
        use crate::TextStyleChange;

        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        two_stills(&mut host);
        host.set_font_folders(&[font_folder(&temp)]);
        let title = host.add_text("A", "", Micros(0)).unwrap();
        let outline = |width: f64| TextStyleChange::Outline {
            color: "#000000".to_owned(),
            width,
        };
        for (width, live) in [(0.02, true), (0.05, true), (0.08, false)] {
            host.style_text(&title, outline(width), live).unwrap();
        }
        let style = host.text(&title).unwrap().style;
        assert_eq!(
            (style.outline_color.as_str(), style.outline_width),
            ("#000000", 0.08)
        );
        // A tap after the drag is its own step.
        host.style_text(&title, TextStyleChange::Color("#FF0000".to_owned()), false)
            .unwrap();
        let shadow = TextStyleChange::Shadow {
            color: "#000000".to_owned(),
            distance: 10.0,
            blur: 0.3,
        };
        host.style_text(&title, shadow, false).unwrap();
        assert_eq!(host.text(&title).unwrap().style.shadow_color, "#000000");
        assert_eq!(host.draft().materials.texts[0].shadow_angle, -45.0);
        host.undo().unwrap();
        host.undo().unwrap();
        assert_eq!(host.text(&title).unwrap().style.outline_width, 0.08);
        host.undo().unwrap();
        assert!(host.text(&title).unwrap().style.outline_color.is_empty());
    }

    #[test]
    fn a_glow_goes_on_a_title_and_a_drag_of_it_is_one_step() {
        use crate::TextStyleChange;

        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        two_stills(&mut host);
        host.set_font_folders(&[font_folder(&temp)]);
        let title = host.add_text("A", "", Micros(0)).unwrap();
        let glow = |color: &str, strength: f64| TextStyleChange::Glow {
            color: color.to_owned(),
            strength,
            range: 0.5,
        };
        host.style_text(&title, glow("#FFCC00", 0.6), false)
            .unwrap();
        for (strength, live) in [(0.7, true), (0.8, true), (0.9, false)] {
            host.style_text(&title, glow("#FFCC00", strength), live)
                .unwrap();
        }
        let style = host.text(&title).unwrap().style;
        assert_eq!(
            (
                style.glow_color.as_str(),
                style.glow_strength,
                style.glow_range
            ),
            ("#FFCC00", 0.9, 0.5)
        );
        assert_eq!(host.draft().materials.effects.len(), 1);
        host.undo().unwrap();
        assert_eq!(host.text(&title).unwrap().style.glow_strength, 0.6);
        host.style_text(&title, glow("", 0.6), false).unwrap();
        assert!(host.text(&title).unwrap().style.glow_color.is_empty());
        assert!(host.draft().materials.effects.is_empty());
    }

    #[test]
    fn a_title_aligns_takes_bold_italic_underline_and_spaces_its_letters_and_lines() {
        use crate::TextStyleChange;

        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        two_stills(&mut host);
        host.set_font_folders(&[font_folder(&temp)]);
        let title = host.add_text("A\nB", "", Micros(0)).unwrap();
        let before = host.text(&title).unwrap().style;
        assert_eq!(before.alignment, 0);
        assert!(!before.bold && !before.italic && !before.underline);

        host.style_text(&title, TextStyleChange::Align(2), false)
            .unwrap();
        let pattern = TextStyleChange::Pattern {
            bold: true,
            italic: false,
            underline: true,
        };
        host.style_text(&title, pattern, false).unwrap();
        for (spacing, live) in [(0.1, true), (0.3, false)] {
            host.style_text(&title, TextStyleChange::LetterSpacing(spacing), live)
                .unwrap();
        }
        host.style_text(&title, TextStyleChange::LineSpacing(0.5), false)
            .unwrap();
        let style = host.text(&title).unwrap().style;
        assert_eq!(style.alignment, 2);
        assert!(style.bold && !style.italic && style.underline);
        assert_eq!((style.letter_spacing, style.line_spacing), (0.3, 0.5));
        // A gap that closes the lines is refused.
        assert!(
            host.style_text(&title, TextStyleChange::LineSpacing(-1.0), false)
                .is_err()
        );
        assert!(
            host.style_text(&title, TextStyleChange::Align(9), false)
                .is_err()
        );
        for _ in 0..4 {
            host.undo().unwrap();
        }
        assert_eq!(host.text(&title).unwrap().style, before);
    }

    #[test]
    fn typing_adds_a_title_as_one_step_and_it_moves_on_the_preview() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        two_stills(&mut host);
        host.set_font_folders(&[font_folder(&temp)]);
        let steps = host.revision();
        assert_eq!(host.type_text(None, " ", "", Micros(0)).unwrap(), None);
        assert_eq!(host.revision(), steps);

        let title = host.type_text(None, "A", "", Micros(0)).unwrap().unwrap();
        for words in ["AA", "AAA", "AAA"] {
            let typed = host.type_text(Some(&title), words, "", Micros(0)).unwrap();
            assert_eq!(typed.as_ref(), Some(&title));
        }
        host.type_text(Some(&title), "", "", Micros(0)).unwrap();
        assert_eq!(host.text(&title).unwrap().content, "AAA");

        let stage = host.stage(&title, Micros(0)).unwrap();
        assert!(stage.text);
        // The test canvas is 16 pixels, so the title fits it.
        assert!((stage.width - stage.canvas_width).abs() < 1e-6 && stage.height > 0.0);
        let middle = (stage.canvas_width / 2.0, stage.canvas_height / 2.0);
        assert_eq!(
            host.stage_pick(Micros(0), middle.0, middle.1),
            Some(title.clone())
        );
        let picture = host.draft().main_track().unwrap().segments[0].id.clone();
        assert!(!host.stage(&picture, Micros(0)).unwrap().text);

        host.stage_begin(&title, Micros(0)).unwrap();
        host.stage_move(100.0, 0.0, 2.0, 0.0).unwrap();
        host.stage_end();
        let moved = host.stage(&title, Micros(0)).unwrap();
        assert!((moved.width - stage.width * 2.0).abs() < 1e-6, "{moved:?}");
        assert_eq!(moved.turn, 0.0);
        host.undo().unwrap();
        host.undo().unwrap();
        assert_eq!(host.text(&title), None);
    }

    #[test]
    fn every_effect_has_its_preview_video_in_the_phone_app() {
        let folder =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../apps/ios/ios/EffectPreviews");
        let effects = crate::phone_effects();
        assert_eq!(effects.len(), cutlass_effects::bundled().len());
        for effect in effects {
            assert!(effect.id.starts_with("recipe/"), "{}", effect.id);
            assert!(!effect.name.is_empty());
            assert!(folder.join(&effect.preview).is_file(), "{}", effect.preview);
        }
    }

    #[test]
    fn every_transition_has_its_preview_video_in_the_phone_app() {
        let folder =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../apps/ios/ios/TransitionPreviews");
        let transitions = crate::phone_transitions();
        assert_eq!(transitions.len(), cutlass_editor::transition_offers().len());
        for transition in transitions {
            assert!(
                folder.join(&transition.preview).is_file(),
                "{}",
                transition.preview
            );
        }
    }

    #[test]
    fn a_subtitle_file_lands_as_captions_and_a_bad_one_is_refused() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let before = host.draft().clone();
        let error = host.import_subtitles("talk.txt", "").unwrap_err();
        assert!(matches!(error, FfiError::Refused { .. }), "{error:?}");
        let error = host.import_subtitles("talk.srt", "1\nhello\n").unwrap_err();
        let FfiError::Refused { reason } = error else {
            panic!("not refused");
        };
        assert!(reason.contains("line 2"), "{reason}");
        assert_eq!(host.draft(), &before);

        let srt =
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n2\n00:00:01,000 --> 00:00:02,000\nagain\n";
        host.import_subtitles("talk.SRT", srt).unwrap();
        host.import_subtitles("song.lrc", "[00:00.50]la\n").unwrap();
        let lanes: Vec<_> = host
            .draft()
            .tracks
            .iter()
            .filter(|track| track.flag == cutlass_model::TrackFlag::TextSubtitle)
            .map(|track| track.segments.len())
            .collect();
        assert_eq!(lanes.len(), 2);
        assert!(lanes.contains(&2) && lanes.contains(&1));
        host.undo().unwrap();
        host.undo().unwrap();
        assert_eq!(host.draft(), &before);
    }

    #[test]
    fn an_effect_goes_on_an_effect_row_changes_the_picture_and_undoes() {
        use crate::NEW_EFFECT_LENGTH;

        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        two_stills(&mut host);
        let plain = host.frame(Micros(500_000)).unwrap();
        let flash_time = host.frame(Micros(80_000)).unwrap();
        let before = host.draft().clone();

        assert!(host.add_effect("recipe/none", Micros(0)).is_err());
        assert!(host.add_effect("shake", Micros(0)).is_err());
        assert_eq!(host.draft(), &before);

        let first = host.add_effect("recipe/white-flash", Micros(0)).unwrap();
        assert_ne!(host.frame(Micros(80_000)).unwrap(), flash_time);
        let after = host
            .add_effect("recipe/vignette", NEW_EFFECT_LENGTH)
            .unwrap();
        let over = host.add_effect("recipe/vhs", Micros(1_000_000)).unwrap();
        let row_of = |host: &PhoneHost, id: &crate::SegmentId| {
            host.lanes()
                .into_iter()
                .position(|lane| lane.clips.iter().any(|clip| &clip.id == id))
                .unwrap()
        };
        assert_eq!(row_of(&host, &first), row_of(&host, &after));
        assert_ne!(row_of(&host, &first), row_of(&host, &over));
        let names: Vec<String> = host
            .lanes()
            .into_iter()
            .flat_map(|lane| lane.clips)
            .filter(|clip| clip.id == first)
            .map(|clip| clip.name)
            .collect();
        assert_eq!(names, ["White Flash"]);

        host.undo().unwrap();
        host.undo().unwrap();
        host.undo().unwrap();
        assert_eq!(host.draft(), &before);
        assert_eq!(host.frame(Micros(500_000)).unwrap(), plain);
    }

    #[test]
    fn text_without_words_a_face_or_a_text_clip_changes_nothing() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (still, _) = two_stills(&mut host);
        let before = host.draft().clone();
        assert!(matches!(
            host.add_text("Hi", "", Micros(0)),
            Err(FfiError::Refused { .. })
        ));
        host.set_font_folders(&[font_folder(&temp)]);
        assert!(host.add_text("   ", "", Micros(0)).is_err());
        assert!(host.add_text("Hi", "../Sample.ttf", Micros(0)).is_err());
        assert_eq!(host.draft(), &before);
        assert_eq!(host.text(&still), None);

        let id = host.add_text("Hi", "", Micros(0)).unwrap();
        let added = host.draft().clone();
        assert!(host.set_text(&id, " ").is_err());
        assert!(host.set_text_font(&id, "a/b.ttf").is_err());
        assert_eq!(host.draft(), &added);
        host.set_text_font(&id, "Other.ttf").unwrap();
        assert_eq!(host.text(&id).unwrap().face, "Other.ttf");
    }

    #[test]
    fn a_selected_clip_gets_the_bar_of_its_row() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (first, _) = two_stills(&mut host);
        use crate::{Bar, SegmentId, ToolId, tools};

        assert_eq!(host.clip_bar(&first), Some(Bar::MainClip));
        assert_eq!(host.clip_bar(&SegmentId::new("nope")), None);
        assert!(
            tools(Bar::MainClip)
                .iter()
                .any(|tool| tool.id == ToolId::Split)
        );
    }

    #[test]
    fn the_ruler_labels_seconds_and_zoom_stays_inside_the_cut() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        two_stills(&mut host);
        // 50 points a second.
        let size = 0.000_05;
        let ticks = host.ruler(size, Micros(0), Micros(2_000_000));
        assert_eq!(ticks[0].time, Micros(0));
        assert_eq!(ticks[0].label.as_deref(), Some("00:00"));
        assert!(ticks.iter().any(|tick| !tick.major));
        // At 50 points a second a label sits on each second.
        let labels: Vec<_> = ticks.iter().filter_map(|tick| tick.label.clone()).collect();
        assert_eq!(labels, ["00:00", "00:01", "00:02"]);
        // So do 40 and 75, the ends of the usual zoom.
        for points in [40.0, 75.0] {
            let ticks = host.ruler(points / 1e6, Micros(0), Micros(1_000_000));
            let labels: Vec<_> = ticks.iter().filter_map(|tick| tick.label.clone()).collect();
            assert_eq!(labels, ["00:00", "00:01"], "{points}");
        }
        assert!(host.ruler(-1.0, Micros(0), Micros(1)).is_empty());

        let wider = host.zoom(size, 2.0, 400.0);
        assert_eq!(wider, size * 2.0);
        // Far out stops at a quarter of the row for the whole cut.
        let out = host.zoom(size, 1e-9, 400.0);
        assert!(out > 0.0 && out < size);
        assert_eq!(host.zoom(size, f64::NAN, 400.0), size);
        assert_eq!(host.zoom(size, -2.0, 400.0), size);
    }

    /// A mono 16-bit wave file: `seconds` of a loud tone.
    fn tone(path: &Path, seconds: u32) {
        let frames = 48_000 * seconds;
        let mut bytes = Vec::new();
        let data = frames * 2;
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&48_000_u32.to_le_bytes());
        bytes.extend_from_slice(&96_000_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data.to_le_bytes());
        for frame in 0..frames {
            let sample = if frame / 50 % 2 == 0 {
                24_000_i16
            } else {
                -24_000
            };
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn stickers_shapes_and_strokes_go_on_a_sticker_row_at_the_playhead() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        two_stills(&mut host);
        let stickers = crate::phone_stickers();
        assert!(stickers.iter().any(|sticker| sticker.animated));
        let still = stickers.iter().find(|sticker| !sticker.animated).unwrap();
        let poster = crate::sticker_poster(&still.id, 64).unwrap();
        assert_eq!(poster.rgba.len(), 64 * 64 * 4);
        assert!(crate::sticker_poster("nope", 64).is_none());

        let row_of = |host: &PhoneHost, id: &cutlass_session::SegmentId| {
            crate::clip::locate(host.draft(), id)
                .unwrap()
                .1
                .kind
                .clone()
        };
        let placed = host.add_sticker(&still.id, Micros(500_000)).unwrap();
        assert_eq!(row_of(&host, &placed), crate::TrackType::Sticker);
        let (segment, _) = crate::clip::locate(host.draft(), &placed).unwrap();
        assert_eq!(segment.target.start, Micros(500_000));
        assert!(
            host.root()
                .join("Resources")
                .join(format!("sticker-{}.png", still.id))
                .is_file()
        );
        let again = host.add_sticker(&still.id, Micros(500_000)).unwrap();
        assert_ne!(
            crate::clip::locate(host.draft(), &again).unwrap().1.id,
            crate::clip::locate(host.draft(), &placed).unwrap().1.id
        );
        assert!(host.add_sticker("nope", Micros(0)).is_err());

        let shapes = crate::phone_shapes();
        assert_eq!(shapes.len(), 7);
        let shape = host.add_shape(&shapes[1].id, Micros(0)).unwrap();
        assert_eq!(row_of(&host, &shape), crate::TrackType::Sticker);
        assert!(host.add_shape("shape:Hexagon", Micros(0)).is_err());

        let trace: Vec<[f64; 2]> = (0..40)
            .map(|i| [100.0 + f64::from(i) * 10.0, 200.0 + f64::from(i % 10) * 8.0])
            .collect();
        let stroke = host.draw_stroke(&trace, Micros(0)).unwrap();
        assert_eq!(row_of(&host, &stroke), crate::TrackType::Sticker);
        // A sticker shows its own name and a shape its kind, not a file
        // or an id.
        let name_of = |host: &PhoneHost, id: &cutlass_session::SegmentId| {
            host.lanes()
                .into_iter()
                .flat_map(|lane| lane.clips)
                .find(|clip| &clip.id == id)
                .unwrap()
                .name
        };
        assert_eq!(name_of(&host, &placed), still.name);
        assert_eq!(name_of(&host, &shape), shapes[1].name);
        let revision = host.revision();
        assert!(host.draw_stroke(&[[5.0, 5.0]], Micros(0)).is_err());
        assert_eq!(host.revision(), revision);
        host.undo().unwrap();
        assert!(crate::clip::locate(host.draft(), &stroke).is_none());
    }

    #[test]
    fn the_sound_of_a_video_comes_in_alone_and_a_sound_or_photo_is_refused() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let sound = temp.path.join("tone.wav");
        tone(&sound, 1);
        assert!(host.import_sound_of(&sound).is_err());
        assert!(host.import_sound_of(&fixture("pattern.png")).is_err());
        let local = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        // A movie with only a picture, and one with sound (AAC).
        let silent = local.join("16929689_1920_1080_30fps.mp4");
        let heard = local.join("7688387224072670472.mp4");
        if !silent.is_file() || !heard.is_file() {
            return;
        }
        assert!(matches!(
            host.import_sound_of(&silent),
            Err(FfiError::Refused { reason }) if reason == "the video has no sound"
        ));
        let material = host.import_sound_of(&heard).unwrap();
        let clip = host.append(&material).unwrap();
        assert_eq!(
            crate::clip::locate(host.draft(), &clip).unwrap().1.kind,
            crate::TrackType::Audio
        );
    }

    #[test]
    fn replace_swaps_a_clips_file_in_one_step_and_refuses_the_wrong_kind() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (first, _) = two_stills(&mut host);
        let material = |host: &PhoneHost| {
            crate::clip::locate(host.draft(), &first)
                .unwrap()
                .0
                .material_id
                .clone()
        };
        let stored = |host: &PhoneHost| {
            fs::read_dir(host.root().join("Resources")).map_or(0, |entries| entries.count())
        };
        let before = material(&host);
        let span = crate::clip::locate(host.draft(), &first)
            .unwrap()
            .0
            .target
            .clone();
        host.replace_file(&first, &fixture("pattern.png")).unwrap();
        assert_ne!(material(&host), before);
        assert_eq!(
            crate::clip::locate(host.draft(), &first).unwrap().0.target,
            span
        );
        host.undo().unwrap();
        assert_eq!(material(&host), before);

        let sound = temp.path.join("tone.wav");
        tone(&sound, 1);
        let files = stored(&host);
        let revision = host.revision();
        assert!(host.replace_file(&first, &sound).is_err());
        assert_eq!(host.revision(), revision);
        assert_eq!(stored(&host), files);
    }

    #[test]
    fn a_sound_clip_draws_its_waveform_and_a_still_has_none() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (still_clip, _) = two_stills(&mut host);
        let file = temp.path.join("tone.wav");
        tone(&file, 1);
        let material = host.import(&file).unwrap();
        let sound = host.append(&material).unwrap();
        let size = 100.0 / 1e6;
        let ready = |waves: &[crate::PhoneWave]| {
            waves
                .iter()
                .find(|wave| wave.id == sound)
                .is_some_and(|wave| wave.ready)
        };
        let mut waves = host.waves(Micros(0), Micros(3_000_000), size);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !ready(&waves) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
            waves = host.waves(Micros(0), Micros(3_000_000), size);
        }
        let wave = waves.iter().find(|wave| wave.id == sound).expect("a wave");
        assert!(wave.ready && !wave.under_picture && !wave.muted);
        assert_eq!(wave.first, 0);
        // A second of sound at 100 points a second.
        assert_eq!(wave.levels.len(), 100);
        assert!(wave.levels.iter().all(|level| (0.6..=1.0).contains(level)));
        assert!(waves.iter().all(|wave| wave.id != still_clip));
        // Half volume draws half as tall.
        host.set_volume(&sound, 0.5).unwrap();
        let half = host.waves(Micros(0), Micros(3_000_000), size);
        let half = half.iter().find(|wave| wave.id == sound).unwrap();
        assert!((half.levels[10] - wave.levels[10] / 2.0).abs() < 0.01);
        // Nothing past the cut, and a bad size draws nothing.
        assert!(
            host.waves(Micros(9_000_000), Micros(10_000_000), size)
                .is_empty()
        );
        assert!(host.waves(Micros(0), Micros(1), f64::NAN).is_empty());
    }

    /// A 16-bit 5.1 wave file (WAVEFORMATEXTENSIBLE, mask 0x3F): `seconds`
    /// of a loud tone on the center channel and silence on the other five.
    fn surround_tone(path: &Path, seconds: u32) {
        let frames = 48_000 * seconds;
        let data = frames * 12;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(60 + data).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&40_u32.to_le_bytes());
        bytes.extend_from_slice(&0xFFFE_u16.to_le_bytes());
        bytes.extend_from_slice(&6_u16.to_le_bytes());
        bytes.extend_from_slice(&48_000_u32.to_le_bytes());
        bytes.extend_from_slice(&(48_000_u32 * 12).to_le_bytes());
        bytes.extend_from_slice(&12_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(&22_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(&0x3F_u32.to_le_bytes());
        // KSDATAFORMAT_SUBTYPE_PCM.
        bytes.extend_from_slice(&[
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38,
            0x9B, 0x71,
        ]);
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data.to_le_bytes());
        for frame in 0..frames {
            let center = if frame / 50 % 2 == 0 {
                24_000_i16
            } else {
                -24_000
            };
            // FL, FR, FC, LFE, BL, BR.
            for sample in [0, 0, center, 0, 0, 0] {
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
        }
        fs::write(path, bytes).unwrap();
    }

    /// A 5.1 file used to open as nothing past stereo. It now imports, its
    /// waveform draws from every channel, and the mix folds its center onto
    /// both sides of the exported stereo sound.
    #[test]
    fn a_surround_wave_imports_draws_and_exports_folded_to_stereo() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        two_stills(&mut host);
        let file = temp.path.join("surround.wav");
        surround_tone(&file, 1);
        let material = host.import(&file).unwrap();
        let sound = host.append(&material).unwrap();
        let size = 100.0 / 1e6;
        let ready = |waves: &[crate::PhoneWave]| {
            waves
                .iter()
                .find(|wave| wave.id == sound)
                .is_some_and(|wave| wave.ready)
        };
        let mut waves = host.waves(Micros(0), Micros(3_000_000), size);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !ready(&waves) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
            waves = host.waves(Micros(0), Micros(3_000_000), size);
        }
        let wave = waves.iter().find(|wave| wave.id == sound).expect("a wave");
        assert!(wave.ready);
        assert_eq!(wave.levels.len(), 100);
        assert!(wave.levels.iter().all(|level| (0.6..=1.0).contains(level)));

        let out = temp.path.join("cut.mp4");
        host.export(&out, 720, &mut |_, _| true).unwrap();
        let mut media = crate::readers().open(&out).unwrap();
        let heard = media.audio_from(Micros(250_000), 4_800).unwrap();
        assert_eq!(heard.layout.len(), 2);
        let samples = heard.as_f32().unwrap();
        let peak = |side: usize| {
            samples
                .iter()
                .skip(side)
                .step_by(2)
                .fold(0.0_f32, |peak, sample| peak.max(sample.abs()))
        };
        // The center at -3 dB: 0.73 x 0.707, give or take the AAC encode.
        for side in [0, 1] {
            assert!(
                (0.4..0.7).contains(&peak(side)),
                "side {side}: {}",
                peak(side)
            );
        }
    }

    #[test]
    fn the_cut_exports_as_an_mp4_with_picture_and_sound() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (_, _) = two_stills(&mut host);
        let file = temp.path.join("tone.wav");
        tone(&file, 1);
        let material = host.import(&file).unwrap();
        host.append(&material).unwrap();
        let out = temp.path.join("cut.mp4");
        let mut seen = Vec::new();
        let made = host
            .export(&out, 720, &mut |done, total| {
                seen.push((done, total));
                true
            })
            .unwrap();
        assert_eq!(made.path, out);
        // Three seconds of pictures at the draft's 30 a second.
        assert_eq!(made.frames, 90);
        assert_eq!(seen.last(), Some(&(90, 90)));
        assert_eq!(made.width.min(made.height), 720);
        assert!(!temp.path.join(".cut.mp4.part").exists());
        // The phone's own readers open what it wrote.
        let cutlass_probe::Probed::Video {
            width,
            height,
            duration,
            ..
        } = cutlass_probe::probe(&out).unwrap()
        else {
            panic!("a movie");
        };
        assert_eq!(
            (width, height),
            (
                i32::try_from(made.width).unwrap(),
                i32::try_from(made.height).unwrap()
            )
        );
        assert!((2_900_000..=3_100_000).contains(&duration), "{duration}");
        let imported = host.import(&out).unwrap();
        assert!(
            host.draft()
                .materials
                .videos
                .iter()
                .any(|video| video.id.as_str() == imported)
        );
    }

    #[test]
    fn a_stopped_or_refused_export_leaves_no_file() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let out = temp.path.join("cut.mp4");
        assert!(
            host.export(&out, 720, &mut |_, _| true).is_err(),
            "nothing to export"
        );
        two_stills(&mut host);
        assert!(
            host.export(&out, 999, &mut |_, _| true).is_err(),
            "not a size"
        );
        let stopped = host.export(&out, 720, &mut |done, _| done < 5).unwrap_err();
        assert_eq!(stopped.to_string(), "export was stopped");
        assert!(!out.exists());
        assert!(!temp.path.join(".cut.mp4.part").exists());
    }

    #[test]
    fn a_still_filmstrip_fills_from_the_worker() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (first, _) = two_stills(&mut host);
        let size = 0.000_05;
        let mut films = host.film(Micros(0), Micros(3_000_000), size, 40.0);
        assert_eq!(films.len(), 2);
        assert_eq!(films[0].id, first);
        assert!(!films[0].tiles.is_empty());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !films[0].tiles[0].ready && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
            films = host.film(Micros(0), Micros(3_000_000), size, 40.0);
        }
        let picture = host.thumb(&films[0].tiles[0].key).expect("stored");
        assert!(picture.width > 0 && picture.height > 0);
        assert_eq!(
            picture.rgba.len(),
            (picture.width * picture.height * 4) as usize
        );
        assert!(host.thumb("not a tile").is_none());
        // Nothing meets a window past the cut, and a bad size draws nothing.
        assert!(
            host.film(Micros(9_000_000), Micros(10_000_000), size, 40.0)
                .is_empty()
        );
        assert!(host.film(Micros(0), Micros(1), 0.0, 40.0).is_empty());
    }

    #[test]
    fn every_ready_movie_tile_hands_back_its_own_picture() {
        let movie = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/16929689_1920_1080_30fps.mp4");
        if !movie.is_file() {
            eprintln!("skipped: the movie is not in ../local-files");
            return;
        }
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let id = host.import(&movie).unwrap();
        host.append(&id).unwrap();
        let size = 0.000_1;
        // A few tiles first, so later tiles have a nearby picture to wait on.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        let mut films = host.film(Micros(0), Micros(1_000_000), size, 40.0);
        while !films[0].tiles.iter().all(|tile| tile.ready) && std::time::Instant::now() < deadline
        {
            std::thread::sleep(std::time::Duration::from_millis(20));
            films = host.film(Micros(0), Micros(1_000_000), size, 40.0);
        }
        films = host.film(Micros(0), Micros(10_000_000), size, 40.0);
        while !films[0].tiles.iter().all(|tile| tile.ready) && std::time::Instant::now() < deadline
        {
            for tile in films[0].tiles.iter().filter(|tile| tile.ready) {
                assert!(host.thumb(&tile.key).is_some(), "ready but empty");
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
            films = host.film(Micros(0), Micros(10_000_000), size, 40.0);
        }
        assert!(films[0].tiles.iter().all(|tile| tile.ready));
        for tile in &films[0].tiles {
            assert!(host.thumb(&tile.key).is_some());
        }
    }

    #[test]
    fn a_dragged_clip_moves_and_a_drop_off_the_rows_is_refused() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (first, second) = two_stills(&mut host);
        let before = host.draft().clone();
        let revision = host.revision();
        let error = host.move_clip(&second, drag(7, 0)).unwrap_err();
        assert!(matches!(error, FfiError::Refused { .. }), "{error:?}");
        let error = host
            .move_clip(&cutlass_session::SegmentId::new("nope"), drag(0, 0))
            .unwrap_err();
        assert!(matches!(error, FfiError::Refused { .. }), "{error:?}");
        assert_eq!(host.draft(), &before);
        assert_eq!(host.revision(), revision);

        let start = host.move_clip(&first, drag(0, 4_000_000)).unwrap();
        let main = &host.draft().tracks[0];
        let moved = main
            .segments
            .iter()
            .find(|segment| segment.id == first)
            .unwrap();
        assert_eq!(moved.target.start, start);
        assert!(host.revision() > revision);
    }

    #[test]
    fn a_picked_file_is_probed_here_and_a_file_that_is_not_media_is_refused() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let id = host.import(&fixture("pattern.jpg")).unwrap();
        let material = &host.draft().materials.videos[0];
        assert_eq!(material.id.as_str(), id);
        assert_eq!(material.kind, "photo");
        assert_eq!((material.width, material.height), (16, 16));
        assert_eq!(
            material.duration,
            cutlass_model::MaterialVideo::MAPPING_DURATION
        );

        let before = host.draft().clone();
        let text = temp.path.join("notes.mp4");
        fs::write(&text, b"not media").unwrap();
        let error = host.import(&text).unwrap_err();
        assert!(
            error.to_string().contains("not a media container"),
            "{error}"
        );
        let empty = temp.path.join("empty.jpg");
        fs::write(&empty, b"").unwrap();
        assert!(host.import(&empty).is_err());
        assert!(host.import(&temp.path).is_err());
        assert_eq!(host.draft(), &before);
        assert!(!host.root().join("Resources/notes.mp4").exists());
    }

    #[test]
    fn a_sound_appends_to_a_new_audio_lane() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let id = host
            .import_probed(
                &fixture("pattern.png"),
                crate::Probed::Audio { duration: 500_000 },
            )
            .unwrap();
        host.append(&id).unwrap();
        let draft = host.draft();
        assert_eq!(draft.tracks.len(), 2);
        let sound = draft
            .tracks
            .iter()
            .find(|track| track.kind == cutlass_model::TrackType::Audio)
            .unwrap();
        assert_eq!(sound.segments[0].target.start.0, 0);
    }

    /// A libvorbis file in Ogg imports through the probe as half a second
    /// of sound, as ffprobe reads it.
    #[test]
    fn an_ogg_vorbis_file_imports_as_a_sound() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../containers/ogg/tests/fixtures/stereo_44100_q4.ogg");
        let material = host.import(&file).unwrap();
        let audio = host
            .draft()
            .materials
            .audios
            .iter()
            .find(|audio| audio.id.as_str() == material)
            .expect("the sound is in the library");
        assert_eq!(audio.duration, Micros(500_000));
        assert!(
            audio.path.ends_with("stereo_44100_q4.ogg"),
            "{}",
            audio.path
        );
    }

    #[test]
    fn a_refused_import_leaves_the_draft_and_writes_no_copy() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let before = host.draft().clone();
        let error = host
            .import_probed(&fixture("pattern.png"), still(0))
            .unwrap_err();
        assert_eq!(error.to_string(), "the media has no duration");
        let flat = crate::Probed::Video {
            meta: cutlass_model::MetaType::Video,
            width: 0,
            height: 9,
            duration: 1,
        };
        let error = host
            .import_probed(&fixture("pattern.png"), flat)
            .unwrap_err();
        assert_eq!(error.to_string(), "the picture has no size");
        let missing = temp.path.join("missing.mov");
        let error = host.import_probed(&missing, still(1_000)).unwrap_err();
        assert!(matches!(error, FfiError::Refused { .. }));
        let resources = host.root().join("Resources");
        assert!(!resources.exists() || fs::read_dir(&resources).unwrap().next().is_none());
        let error = host.append("nothing").unwrap_err();
        assert_eq!(error.to_string(), "the media is not in the library");
        assert_eq!(host.draft(), &before);
        assert_eq!(host.revision(), 0);
    }

    #[test]
    fn the_frame_shows_the_imported_picture_and_a_negative_time_is_refused() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        host.apply(
            0,
            &[Request::SetCanvasSize {
                width: 16,
                height: 16,
                ratio: "1:1".to_owned(),
            }],
        )
        .unwrap();
        let empty = host.frame(Micros(0)).unwrap();
        assert_eq!((empty.width, empty.height), (16, 16));
        let id = host
            .import_probed(&fixture("pattern.png"), still(1_000_000))
            .unwrap();
        host.append(&id).unwrap();
        let shown = host.frame(Micros(500_000)).unwrap();
        assert_eq!((shown.width, shown.height), (16, 16));
        assert_eq!(shown.rgba.len(), 16 * 16 * 4);
        assert_ne!(shown.rgba, empty.rgba);
        assert!(matches!(
            host.frame(Micros(-1)),
            Err(FfiError::Frame { .. })
        ));
    }

    #[test]
    fn a_truncated_picture_draws_the_canvas_without_a_panic() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        host.apply(
            0,
            &[Request::SetCanvasSize {
                width: 16,
                height: 16,
                ratio: "1:1".to_owned(),
            }],
        )
        .unwrap();
        let empty = host.frame(Micros(0)).unwrap();
        let id = host
            .import_probed(&fixture("truncated.jpg"), still(1_000_000))
            .unwrap();
        host.append(&id).unwrap();
        let shown = host.frame(Micros(0)).unwrap();
        assert_eq!(shown, empty);
    }

    #[test]
    fn rows_list_the_clips_and_split_keeps_a_frame_each_side() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let id = host
            .import_probed(&fixture("pattern.png"), still(1_000_000))
            .unwrap();
        let left = host.append(&id).unwrap();
        let lanes = host.lanes();
        assert_eq!(lanes.len(), 1);
        assert!(lanes[0].main);
        assert_eq!(lanes[0].clips[0].name, "pattern.png");
        assert_eq!(lanes[0].clips[0].duration.0, 1_000_000);

        let before = host.draft().clone();
        let edge = host.split(&left, Micros(10)).unwrap_err();
        assert_eq!(edge.to_string(), "split is not available there");
        let missing = host
            .split(&cutlass_session::SegmentId::new("nope"), Micros(500_000))
            .unwrap_err();
        assert_eq!(missing.to_string(), "the clip is not on the timeline");
        assert_eq!(host.draft(), &before);

        let right = host.split(&left, Micros(400_000)).unwrap();
        let clips = &host.lanes()[0].clips;
        assert_eq!(clips.len(), 2);
        assert_eq!(clips[0].duration.0, 400_000);
        assert_eq!(clips[1].id, right);
        assert_eq!(clips[1].start.0, 400_000);

        host.remove(&right).unwrap();
        assert_eq!(host.lanes()[0].clips.len(), 1);
        assert!(host.remove(&right).is_err());
        assert_eq!(host.lanes()[0].clips.len(), 1);
    }

    #[test]
    fn an_edit_redraws_the_picture_at_the_same_time() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        host.apply(
            0,
            &[Request::SetCanvasSize {
                width: 16,
                height: 16,
                ratio: "1:1".to_owned(),
            }],
        )
        .unwrap();
        let png = host
            .import_probed(&fixture("pattern.png"), still(1_000_000))
            .unwrap();
        let jpeg = host
            .import_probed(&fixture("pattern.jpg"), still(1_000_000))
            .unwrap();
        let first = host.append(&png).unwrap();
        host.append(&jpeg).unwrap();
        let at = Micros(500_000);
        let before = host.frame(at).unwrap();
        let second = host.frame(Micros(1_500_000)).unwrap();
        assert_ne!(before, second);

        host.remove(&first).unwrap();
        assert_eq!(host.frame(at).unwrap(), second);
        host.undo().unwrap();
        assert_eq!(host.frame(at).unwrap(), before);
    }

    #[test]
    fn a_split_and_a_ripple_delete_redraw_a_movie_at_the_cut() {
        let movie = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/16929689_1920_1080_30fps.mp4");
        if !movie.is_file() {
            eprintln!("skipped: the movie is not in ../local-files");
            return;
        }
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let id = host.import(&movie).unwrap();
        let material = &host.draft().materials.videos[0];
        assert_eq!((material.width, material.height), (1920, 1080));
        assert!((10_000_000..10_500_000).contains(&material.duration.0));
        let left = host.append(&id).unwrap();
        let at = Micros(3_000_000);
        let whole = host.frame(at).unwrap();
        host.split(&left, at).unwrap();
        let cut = host.frame(at).unwrap();
        assert_eq!(cut, whole);
        let later = host.frame(Micros(6_000_000)).unwrap();
        assert_ne!(cut, later);

        host.remove(&left).unwrap();
        assert_eq!(host.frame(at).unwrap(), later);
        host.undo().unwrap();
        assert_eq!(host.frame(at).unwrap(), cut);
        host.redo().unwrap();
        assert_eq!(host.frame(at).unwrap(), later);
    }

    #[test]
    fn a_clip_keeps_one_filter_that_swaps_takes_a_strength_and_comes_off() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (first, _) = two_stills(&mut host);
        let filters = crate::phone_filters();
        assert!(filters.len() > 1);
        assert!(host.clip_filter(&first).is_none());

        let layer = host
            .set_clip_filter(&first, &filters[0].id)
            .unwrap()
            .expect("placed");
        let kept = host.clip_filter(&first).unwrap();
        assert_eq!(kept.segment_id, layer);
        let name = |host: &PhoneHost, id: &cutlass_session::SegmentId| {
            host.lanes()
                .into_iter()
                .flat_map(|lane| lane.clips)
                .find(|clip| &clip.id == id)
                .unwrap()
                .name
        };
        assert_eq!(name(&host, &layer), filters[0].name);
        assert_eq!(kept.filter_id, filters[0].id);
        assert_eq!(kept.strength, 1.0);
        let span = |host: &PhoneHost, id: &cutlass_session::SegmentId| {
            let (segment, _) = crate::clip::locate(host.draft(), id).unwrap();
            (segment.target.start, segment.target.duration)
        };
        assert_eq!(span(&host, &layer), span(&host, &first));

        let swapped = host
            .set_clip_filter(&first, &filters[1].id)
            .unwrap()
            .expect("placed");
        let layers = |host: &PhoneHost| {
            host.draft()
                .tracks
                .iter()
                .filter(|track| track.kind == cutlass_model::TrackType::Filter)
                .map(|track| track.segments.len())
                .sum::<usize>()
        };
        assert_eq!(layers(&host), 1);
        assert_eq!(host.clip_filter(&first).unwrap().filter_id, filters[1].id);

        host.set_strength(&swapped, 0.4).unwrap();
        assert_eq!(host.clip_filter(&first).unwrap().strength, 0.4);
        assert!(host.set_strength(&swapped, 1.5).is_err());

        assert_eq!(host.set_clip_filter(&first, "").unwrap(), None);
        assert!(host.clip_filter(&first).is_none());
        assert_eq!(layers(&host), 0);
        host.undo().unwrap();
        assert_eq!(host.clip_filter(&first).unwrap().strength, 0.4);

        assert!(host.set_clip_filter(&first, "look/nope").is_err());
        let at = host.add_filter(&filters[0].id, Micros(0)).unwrap();
        assert_eq!(host.clip_filter(&at).unwrap().filter_id, filters[0].id);
    }

    #[test]
    fn adjust_sliders_sit_on_a_picture_or_an_adjustment_layer() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (first, _) = two_stills(&mut host);
        let sliders = host.adjust_sliders(&first).unwrap();
        assert_eq!(sliders.len(), 17);
        assert!(sliders.iter().all(|slider| slider.value == 0.0));

        host.set_adjust(&first, "contrast", -0.25).unwrap();
        let value = |host: &PhoneHost, id: &cutlass_session::SegmentId, key: &str| {
            host.adjust_sliders(id)
                .unwrap()
                .into_iter()
                .find(|slider| slider.key == key)
                .unwrap()
                .value
        };
        assert!((value(&host, &first, "contrast") + 0.25).abs() < 1e-6);
        let revision = host.revision();
        assert!(host.set_adjust(&first, "contrast", -2.0).is_err());
        assert!(host.set_adjust(&first, "grain", -0.5).is_err());
        assert!(host.set_adjust(&first, "nope", 0.5).is_err());
        assert_eq!(host.revision(), revision);

        let layer = host.add_adjustment(Micros(0)).unwrap();
        let named = host
            .lanes()
            .into_iter()
            .flat_map(|lane| lane.clips)
            .find(|clip| clip.id == layer)
            .unwrap()
            .name;
        assert_eq!(named, "Adjustment");
        host.set_adjust(&layer, "temperature", 0.5).unwrap();
        assert!((value(&host, &layer, "temperature") - 0.5).abs() < 1e-6);
        host.undo().unwrap();
        assert_eq!(value(&host, &layer, "temperature"), 0.0);
    }

    #[test]
    fn an_effect_on_a_clip_runs_over_it_alone_for_its_span() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (first, second) = two_stills(&mut host);
        let effect = &crate::phone_effects()[0];
        let placed = host.add_clip_effect(&first, &effect.id).unwrap();
        let on = cutlass_editor::effects_on(host.draft(), &first);
        assert_eq!(on.len(), 1);
        assert_eq!(on[0].segment_id, placed);
        assert!(cutlass_editor::effects_on(host.draft(), &second).is_empty());
        assert!(host.add_clip_effect(&first, "recipe/nope").is_err());
    }

    #[test]
    fn a_reversed_movie_draws_from_its_first_frame() {
        let movie = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/16929689_1920_1080_30fps.mp4");
        if !movie.is_file() {
            eprintln!("skipped: the movie is not in ../local-files");
            return;
        }
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let id = host.import(&movie).unwrap();
        let clip = host.append(&id).unwrap();
        // The last pictures display after the last one decodes; a reversed
        // clip starts on them.
        let lit = |picture: &crate::Picture| picture.rgba.chunks(4).filter(|p| p[0] > 16).count();
        let last = host.frame(Micros(10_200_000)).unwrap();
        assert!(lit(&last) > 0);
        host.toggle_reverse(&clip).unwrap();
        assert!(lit(&host.frame(Micros(0)).unwrap()) > 0);
    }

    fn movie(duration: i64) -> crate::Probed {
        crate::Probed::Video {
            meta: cutlass_model::MetaType::Video,
            width: 16,
            height: 16,
            duration,
        }
    }

    #[test]
    fn mirror_turn_crop_and_opacity_change_a_picture_one_step_each() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (first, _) = two_stills(&mut host);
        let state = host.clip_state(&first, Micros(0)).unwrap();
        assert!(state.picture && state.can_crop && state.on_main);
        assert!(!state.mirrored && state.crop.is_empty() && state.opacity == 1.0);

        host.toggle_mirror(&first).unwrap();
        host.turn(&first).unwrap();
        host.set_crop(&first, "9:16").unwrap();
        host.set_opacity(&first, 0.25).unwrap();
        let state = host.clip_state(&first, Micros(0)).unwrap();
        assert!(state.mirrored);
        assert_eq!(state.rotation, 270.0);
        assert_eq!(state.crop, "9:16");
        assert_eq!(state.opacity, 0.25);

        // Four steps, four undos.
        for _ in 0..4 {
            host.undo().unwrap();
        }
        let state = host.clip_state(&first, Micros(0)).unwrap();
        assert!(!state.mirrored && state.rotation == 0.0 && state.crop.is_empty());
        assert_eq!(state.opacity, 1.0);

        let revision = host.revision();
        assert!(host.set_opacity(&first, 1.5).is_err());
        assert!(host.set_crop(&first, "wide").is_err());
        assert_eq!(host.revision(), revision);
    }

    #[test]
    fn a_mirrored_or_turned_picture_still_draws() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        host.apply(
            0,
            &[Request::SetCanvasSize {
                width: 16,
                height: 16,
                ratio: "1:1".to_owned(),
            }],
        )
        .unwrap();
        let empty = host.frame(Micros(0)).unwrap();
        let id = host
            .import_probed(&fixture("pattern.png"), still(1_000_000))
            .unwrap();
        let clip = host.append(&id).unwrap();
        let plain = host.frame(Micros(500_000)).unwrap();
        host.toggle_mirror(&clip).unwrap();
        let mirrored = host.frame(Micros(500_000)).unwrap();
        assert_ne!(mirrored, empty);
        assert_ne!(mirrored, plain);
        host.undo().unwrap();
        host.turn(&clip).unwrap();
        let turned = host.frame(Micros(500_000)).unwrap();
        assert_ne!(turned, empty);

        let movie = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/16929689_1920_1080_30fps.mp4");
        if !movie.is_file() {
            return;
        }
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let id = host.import(&movie).unwrap();
        let clip = host.append(&id).unwrap();
        let lit = |picture: &crate::Picture| picture.rgba.chunks(4).filter(|p| p[0] > 16).count();
        let plain = host.frame(Micros(0)).unwrap();
        host.toggle_mirror(&clip).unwrap();
        let mirrored = host.frame(Micros(0)).unwrap();
        eprintln!("plain lit {} mirrored lit {}", lit(&plain), lit(&mirrored));
        assert!(lit(&mirrored) > lit(&plain) / 2);
    }

    #[test]
    fn a_fade_and_a_reverse_land_on_the_clip_and_too_long_a_fade_is_refused() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (first, _) = two_stills(&mut host);
        host.set_fade(&first, Micros(500_000), Micros(250_000))
            .unwrap();
        let state = host.clip_state(&first, Micros(0)).unwrap();
        assert_eq!((state.fade_in.0, state.fade_out.0), (500_000, 250_000));
        // The same fades again are not a step.
        let revision = host.revision();
        host.set_fade(&first, Micros(500_000), Micros(250_000))
            .unwrap();
        assert_eq!(host.revision(), revision);
        assert!(
            host.set_fade(&first, Micros(1_500_000), Micros(1_000_000))
                .is_err()
        );
        assert!(host.set_fade(&first, Micros(-1), Micros(0)).is_err());

        host.toggle_reverse(&first).unwrap();
        assert!(host.clip_state(&first, Micros(0)).unwrap().reversed);
        host.toggle_reverse(&first).unwrap();
        assert!(!host.clip_state(&first, Micros(0)).unwrap().reversed);
    }

    #[test]
    fn a_duplicate_on_main_goes_right_after_and_the_rest_move_along() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (first, second) = two_stills(&mut host);
        let copy = host.duplicate(&first).unwrap();
        let main = &host.draft().tracks[0];
        let order: Vec<_> = main.segments.iter().map(|segment| &segment.id).collect();
        assert_eq!(order, [&first, &copy, &second]);
        assert_eq!(main.segments[1].target.start.0, 2_000_000);
        assert_eq!(main.segments[2].target.start.0, 4_000_000);
        assert_eq!(main.segments[1].material_id, main.segments[0].material_id);
        host.undo().unwrap();
        assert_eq!(host.draft().tracks[0].segments.len(), 2);
    }

    #[test]
    fn an_overlay_goes_above_main_and_moves_to_main_as_one_step_each() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let (first, _) = two_stills(&mut host);
        let source = fixture("pattern.png");
        let picture = host.import_probed(&source, still(1_000_000)).unwrap();
        let over = host.add_overlay(&picture, Micros(500_000)).unwrap();
        let lanes = host.lanes();
        assert_eq!(lanes.len(), 2);
        assert!(!lanes[0].main && lanes[1].main);
        assert_eq!(lanes[0].clips[0].id, over);
        assert_eq!(lanes[0].clips[0].start.0, 500_000);
        assert!(host.clip_state(&over, Micros(0)).unwrap().can_splice);
        assert!(!host.clip_state(&first, Micros(0)).unwrap().can_splice);

        // A second overlay at the same time needs a row of its own.
        let again = host.add_overlay(&picture, Micros(500_000)).unwrap();
        assert_eq!(host.lanes().len(), 3);

        host.to_main(&over).unwrap();
        let main = host.lanes().into_iter().find(|lane| lane.main).unwrap();
        assert!(main.clips.iter().any(|clip| clip.id == over));
        assert!(host.to_main(&first).is_err());

        let sound = host
            .import_probed(&source, crate::Probed::Audio { duration: 500_000 })
            .unwrap();
        assert!(host.add_overlay(&sound, Micros(0)).is_err());
        host.remove(&again).unwrap();
    }

    #[test]
    fn linkage_and_a_muted_row_are_kept_with_the_project() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        assert!(!host.linkage_enabled());
        host.set_linkage_enabled(true).unwrap();
        assert!(host.linkage_enabled());
        let main = host.lanes()[0].track_id.clone();
        host.set_row_muted(&main, true).unwrap();
        assert!(host.lanes()[0].muted);
        host.save().unwrap();
        let opened = roots.open();
        assert!(opened.linkage_enabled());
        assert!(opened.lanes()[0].muted);
    }

    #[test]
    fn a_beat_goes_on_a_sound_at_the_playhead_and_a_second_tap_takes_it_off() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let wave = temp.path.join("tone.wav");
        tone(&wave, 2);
        let sound = host.import(&wave).unwrap();
        let clip = host.append(&sound).unwrap();
        let state = host.clip_state(&clip, Micros(500_000)).unwrap();
        assert!(state.can_beat && !state.beat_here && !state.picture);

        host.toggle_beat(&clip, Micros(500_000)).unwrap();
        assert!(host.clip_state(&clip, Micros(500_000)).unwrap().beat_here);
        let lane = host.lanes().into_iter().find(|lane| !lane.main).unwrap();
        assert_eq!(lane.clips[0].beats, [Micros(500_000)]);

        host.toggle_beat(&clip, Micros(500_000)).unwrap();
        assert!(!host.clip_state(&clip, Micros(500_000)).unwrap().beat_here);
        assert!(host.toggle_beat(&clip, Micros(9_000_000)).is_err());

        let (picture, _) = two_stills(&mut host);
        assert!(host.toggle_beat(&picture, Micros(0)).is_err());
    }

    #[test]
    fn a_video_freezes_a_frame_into_a_still_and_a_still_does_not() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let source = fixture("pattern.png");
        let id = host.import_probed(&source, movie(4_000_000)).unwrap();
        let clip = host.append(&id).unwrap();
        assert!(host.clip_state(&clip, Micros(0)).unwrap().can_freeze);
        let still = host.freeze(&clip, Micros(2_000_000)).unwrap();
        let main = &host.draft().tracks[0];
        assert_eq!(main.segments.len(), 3);
        assert_eq!(main.segments[1].id, still);
        assert_eq!(main.segments[1].target.start.0, 2_000_000);
        assert_eq!(
            main.segments[1].target.duration,
            cutlass_eval::FREEZE_DURATION
        );
        let stored = host
            .draft()
            .materials
            .videos
            .iter()
            .find(|video| video.id == main.segments[1].material_id)
            .unwrap()
            .path
            .clone();
        assert!(stored.starts_with("Resources/freeze-"), "{stored}");
        assert!(host.root().join(&stored).is_file());
        assert!(
            !host
                .clip_state(&still, Micros(2_000_000))
                .unwrap()
                .can_freeze
        );
        assert!(host.freeze(&still, Micros(2_500_000)).is_err());

        let (photo, _) = two_stills(&mut host);
        assert!(host.freeze(&photo, Micros(0)).is_err());
    }

    #[test]
    fn an_extract_needs_sound_and_places_it_on_a_sound_row_and_recovers() {
        let temp = Temp::new();
        let roots = Roots::new(&temp);
        let mut host = roots.create();
        let source = fixture("pattern.png");
        let id = host.import_probed(&source, movie(2_000_000)).unwrap();
        let clip = host.append(&id).unwrap();
        let revision = host.revision();
        assert!(host.extract_audio(&clip).is_err());
        assert_eq!(host.revision(), revision);

        host.place_extracted(&clip).unwrap();
        let state = host.clip_state(&clip, Micros(0)).unwrap();
        assert!(state.extracted && !state.can_extract);
        let sound = host
            .lanes()
            .into_iter()
            .find(|lane| lane.kind == cutlass_model::TrackType::Audio)
            .unwrap();
        assert_eq!(sound.clips[0].start.0, 0);
        assert_eq!(sound.clips[0].duration.0, 2_000_000);

        host.recover_audio(&clip).unwrap();
        assert!(!host.clip_state(&clip, Micros(0)).unwrap().extracted);
    }
}
