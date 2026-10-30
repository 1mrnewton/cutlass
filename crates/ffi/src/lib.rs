//! The calls SwiftUI and Compose make.
//!
//! Desktop does not use this crate. A phone shell opens one project folder,
//! applies the same session requests the desktop applies, and looks up a
//! bundled or cached file. Playback here is the playhead and the audio
//! output. Seeking a picture needs a device the shell constructs; this crate
//! does not construct one.

#![forbid(unsafe_code)]

mod cache;
mod error;
mod host;
mod player;
mod project;

pub use cache::ResourceCache;
pub use error::FfiError;
pub use host::Commands;
pub use player::PhonePlayer;
pub use project::Project;

use std::path::{Path, PathBuf};

use cutlass_session::{Draft, LinkageSettings, Micros, Request, Session};
use cutlass_store::RepairNote;

/// The open project, its session, the playhead, and the resource cache.
///
/// Save writes the session draft into the project folder. The undo stack stays
/// in the session. A new open starts the revision at zero again.
pub struct PhoneHost {
    project: Project,
    commands: Commands,
    player: PhonePlayer,
    cache: ResourceCache,
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
    use crate::FfiError;

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
}
