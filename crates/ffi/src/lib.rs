//! The calls SwiftUI and Compose make.
//!
//! Desktop does not use this crate. A phone shell opens one project folder,
//! applies the same session requests the desktop applies, and looks up a
//! bundled or cached file. Playback here is the playhead and the audio
//! output. Seeking a picture needs a device the shell constructs; this crate
//! does not construct one.

#![forbid(unsafe_code)]

mod cache;
mod codecs;
mod error;
mod host;
mod media;
mod player;
mod preview;
mod project;
mod timeline;

pub use cache::ResourceCache;
pub use codecs::{decoders, readers};
pub use cutlass_logger::InitError as LogError;
pub use cutlass_model::TrackType;
pub use cutlass_probe::{ProbeError, Probed};
pub use cutlass_session::{Micros, SegmentId};
pub use cutlass_storage::Home;
pub use cutlass_store::{ListedProject, create_project, list_projects};
pub use error::FfiError;
pub use host::Commands;
pub use player::PhonePlayer;
pub use preview::{Picture, Preview};
pub use project::Project;
pub use timeline::{PhoneClip, PhoneLane, phone_lanes};

use std::path::{Path, PathBuf};

use cutlass_session::{Draft, LinkageSettings, Request, Session};
use cutlass_store::RepairNote;

/// Open the process log in `directory`.
///
/// A phone shell calls this once at start. The public build writes the file
/// and does not write stderr.
pub fn start_log(directory: &Path) -> Result<PathBuf, LogError> {
    cutlass_logger::init(directory)
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

    /// Start play when the playhead is inside the stored duration.
    pub fn play(&mut self) -> Result<(), FfiError> {
        self.player.play(self.commands.session())
    }

    /// The playhead, in microseconds.
    #[must_use]
    pub fn playhead(&self) -> Micros {
        self.player.playhead()
    }

    /// Play is running inside the stored duration.
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

    /// Add the whole of `material_id` at the end of its lane, as one undo step.
    /// A picture goes on main; a sound opens an audio lane when there is none.
    pub fn append(&mut self, material_id: &str) -> Result<SegmentId, FfiError> {
        let (requests, segment_id) = media::append_requests(self.draft(), material_id)?;
        self.apply(self.revision(), &requests)?;
        Ok(segment_id)
    }

    /// The composited picture at `time`, reading media from the project folder.
    pub fn frame(&mut self, time: Micros) -> Result<Picture, FfiError> {
        self.preview
            .frame(self.commands.session(), self.project.root(), time)
    }

    /// The rows to draw, top first.
    #[must_use]
    pub fn lanes(&self) -> Vec<PhoneLane> {
        phone_lanes(self.draft())
    }

    /// Cut `segment_id` at `at`. Returns the right piece's id.
    pub fn split(&mut self, segment_id: &SegmentId, at: Micros) -> Result<SegmentId, FfiError> {
        let (request, right_id) = timeline::split_request(self.draft(), segment_id, at)?;
        self.apply(self.revision(), &[request])?;
        Ok(right_id)
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

    /// Published path for a bundled or cached id. A missing id is an error.
    /// This does not start a transfer.
    pub fn resource(&self, id: &str) -> Result<PathBuf, FfiError> {
        self.cache.resolve(id)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use cutlass_session::{EditError, Request, TrackAttribute, TrackId};
    use cutlass_storage::StorageError;

    use super::PhoneHost;
    use crate::{FfiError, start_log};

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
        assert_eq!(cutlass_session::Micros(5).0, 5);
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
        let empty = host.frame(cutlass_session::Micros(0)).unwrap();
        assert_eq!((empty.width, empty.height), (16, 16));
        let id = host
            .import_probed(&fixture("pattern.png"), still(1_000_000))
            .unwrap();
        host.append(&id).unwrap();
        let shown = host.frame(cutlass_session::Micros(500_000)).unwrap();
        assert_eq!((shown.width, shown.height), (16, 16));
        assert_eq!(shown.rgba.len(), 16 * 16 * 4);
        assert_ne!(shown.rgba, empty.rgba);
        assert!(matches!(
            host.frame(cutlass_session::Micros(-1)),
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
        let empty = host.frame(cutlass_session::Micros(0)).unwrap();
        let id = host
            .import_probed(&fixture("truncated.jpg"), still(1_000_000))
            .unwrap();
        host.append(&id).unwrap();
        let shown = host.frame(cutlass_session::Micros(0)).unwrap();
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
        let edge = host.split(&left, cutlass_session::Micros(10)).unwrap_err();
        assert_eq!(edge.to_string(), "split is not available there");
        let missing = host
            .split(
                &cutlass_session::SegmentId::new("nope"),
                cutlass_session::Micros(500_000),
            )
            .unwrap_err();
        assert_eq!(missing.to_string(), "the clip is not on the timeline");
        assert_eq!(host.draft(), &before);

        let right = host.split(&left, cutlass_session::Micros(400_000)).unwrap();
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
        let at = cutlass_session::Micros(500_000);
        let before = host.frame(at).unwrap();
        let second = host.frame(cutlass_session::Micros(1_500_000)).unwrap();
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
        let at = cutlass_session::Micros(3_000_000);
        let whole = host.frame(at).unwrap();
        host.split(&left, at).unwrap();
        let cut = host.frame(at).unwrap();
        assert_eq!(cut, whole);
        let later = host.frame(cutlass_session::Micros(6_000_000)).unwrap();
        assert_ne!(cut, later);

        host.remove(&left).unwrap();
        assert_eq!(host.frame(at).unwrap(), later);
        host.undo().unwrap();
        assert_eq!(host.frame(at).unwrap(), cut);
        host.redo().unwrap();
        assert_eq!(host.frame(at).unwrap(), later);
    }
}
