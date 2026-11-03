//! The Swift and Kotlin surface over `ffi`.
//!
//! One thread owns each open [`cutlass_ffi::PhoneHost`], because its audio
//! output cannot move between threads. That thread runs at the interface's
//! priority. A call sends it a job and waits for the answer, then turns it into plain records. Times are
//! microseconds. A failure is a message for the phone to show; the draft is
//! unchanged.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Sender};
use std::time::UNIX_EPOCH;

use cutlass_ffi::{Drag, FfiError, Home, PhoneHost};
use cutlass_ffi::{Micros, SegmentId};

uniffi::setup_scaffolding!();

/// Why a call left the project as it was.
#[derive(Debug, uniffi::Error)]
#[uniffi(flat_error)]
pub enum EditorError {
    Failed { message: String },
}

impl std::fmt::Display for EditorError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed { message } => formatter.write_str(message),
        }
    }
}

impl std::error::Error for EditorError {}

impl From<FfiError> for EditorError {
    fn from(error: FfiError) -> Self {
        Self::Failed {
            message: error.to_string(),
        }
    }
}

/// A composited picture, eight-bit RGBA, rows top to bottom.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// One clip on a row.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Clip {
    pub id: String,
    pub name: String,
    pub start_us: i64,
    pub duration_us: i64,
}

/// What a row holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum LaneKind {
    Video,
    Audio,
    Text,
    Sticker,
    Effect,
    Filter,
    Adjust,
    Other,
}

/// One timeline row, top first.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Lane {
    pub kind: LaneKind,
    pub main: bool,
    pub locked: bool,
    pub clips: Vec<Clip>,
}

/// One saved project in the launch list. `modified_ms` is milliseconds
/// since 1970.
#[derive(Clone, Debug, uniffi::Record)]
pub struct ProjectSummary {
    pub path: String,
    pub name: String,
    pub duration_us: i64,
    pub modified_ms: i64,
}

/// Lay out the Cutlass folder under `root` and open the process log in it.
///
/// `root` is the one folder everything Cutlass writes lives under: the
/// projects, the cache, the downloads, the proxies, the models, and the
/// logs. A phone passes a folder in its app container, once at launch.
/// Calling it again with the same root changes nothing.
#[uniffi::export]
pub fn start(root: String) -> Result<(), EditorError> {
    let home = open_home(&root)?;
    cutlass_ffi::start_log(&home.logs()).map_err(|error| EditorError::Failed {
        message: error.to_string(),
    })?;
    Ok(())
}

/// The projects in the Cutlass folder at `root`, newest first. A missing
/// root lists nothing.
#[uniffi::export]
#[must_use]
pub fn list_projects(root: String) -> Vec<ProjectSummary> {
    let Ok(home) = Home::at(root) else {
        return Vec::new();
    };
    cutlass_ffi::list_projects(&home.projects())
        .into_iter()
        .map(|project| ProjectSummary {
            path: project.path.display().to_string(),
            name: project.name,
            duration_us: project.duration.0,
            modified_ms: project
                .modified
                .duration_since(UNIX_EPOCH)
                .map_or(0, |since| {
                    i64::try_from(since.as_millis()).unwrap_or(i64::MAX)
                }),
        })
        .collect()
}

/// Write a new empty project in the Cutlass folder at `root`, named like the
/// desktop's, and return its folder.
#[uniffi::export]
pub fn create_project(root: String) -> Result<String, EditorError> {
    let home = open_home(&root)?;
    cutlass_ffi::create_project(&home.projects())
        .map(|path| path.display().to_string())
        .map_err(|error| EditorError::from(FfiError::from(error)))
}

/// The app hears this device. Tests keep the software buffer, so they need
/// no speaker.
fn device_audio() -> cutlass_ffi::AudioHost {
    if cfg!(test) {
        cutlass_ffi::AudioHost::Software
    } else {
        cutlass_ffi::AudioHost::current()
    }
}

fn open_home(root: &str) -> Result<Home, EditorError> {
    Home::open(root).map_err(|error| EditorError::from(FfiError::from(error)))
}

/// A project folder must sit directly in the root's projects folder.
fn project_in(home: &Home, project: &Path) -> Result<(), EditorError> {
    if project.parent() == Some(home.projects().as_path()) {
        Ok(())
    } else {
        Err(EditorError::Failed {
            message: format!(
                "{} is not a project in the Cutlass folder",
                project.display()
            ),
        })
    }
}

/// One open project.
#[derive(uniffi::Object)]
pub struct Editor {
    jobs: Sender<Job>,
}

type Job = Box<dyn FnOnce(&mut PhoneHost) + Send>;

#[uniffi::export]
impl Editor {
    /// Open a project folder in the Cutlass folder at `root`, or create an
    /// empty one when it has no draft. `bundle` is the app's read-only
    /// resources and must be a directory. The cache is the root's.
    #[uniffi::constructor]
    pub fn open(root: String, bundle: String, project: String) -> Result<Arc<Self>, EditorError> {
        let home = open_home(&root)?;
        let project = PathBuf::from(project);
        project_in(&home, &project)?;
        let cache = home.cache();
        let (jobs, inbox) = mpsc::channel::<Job>();
        let (opened, answer) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("cutlass-editor".to_owned())
            .spawn(move || {
                cutlass_ffi::serve_the_interface();
                let bundle = PathBuf::from(bundle);
                let host = if project.join("draft_info.json").is_file() {
                    PhoneHost::open(&project, &bundle, &cache, 0)
                } else {
                    PhoneHost::create(&project, "draft", &bundle, &cache, 0)
                };
                let mut host = match host {
                    Ok(mut host) => {
                        host.open_audio(device_audio());
                        let _ = opened.send(Ok(()));
                        host
                    }
                    Err(error) => {
                        let _ = opened.send(Err(EditorError::from(error)));
                        return;
                    }
                };
                // The loop ends when the editor is dropped.
                while let Ok(job) = inbox.recv() {
                    job(&mut host);
                }
            })
            .map_err(|error| EditorError::Failed {
                message: error.to_string(),
            })?;
        answer.recv().map_err(|_| closed())??;
        Ok(Arc::new(Self { jobs }))
    }

    /// Read a picked file, copy it into the project, and add it to the
    /// library. Rust probes it; the phone only hands over the path.
    /// Returns the material id.
    pub fn import(&self, path: String) -> Result<String, EditorError> {
        self.run(move |host| host.import(&PathBuf::from(path)))?
            .map_err(EditorError::from)
    }

    /// Add a library item at the end of its lane. Returns the clip id.
    pub fn append(&self, material_id: String) -> Result<String, EditorError> {
        self.run(move |host| host.append(&material_id))?
            .map(|id| id.as_str().to_owned())
            .map_err(EditorError::from)
    }

    /// The rows to draw, top first.
    pub fn lanes(&self) -> Result<Vec<Lane>, EditorError> {
        let lanes = self.run(|host| host.lanes())?;
        Ok(lanes
            .into_iter()
            .map(|lane| Lane {
                kind: lane_kind(lane.kind),
                main: lane.main,
                locked: lane.locked,
                clips: lane
                    .clips
                    .into_iter()
                    .map(|clip| Clip {
                        id: clip.id.as_str().to_owned(),
                        name: clip.name,
                        start_us: clip.start.0,
                        duration_us: clip.duration.0,
                    })
                    .collect(),
            })
            .collect())
    }

    /// The end of the last clip, in microseconds.
    pub fn duration_us(&self) -> Result<i64, EditorError> {
        self.run(|host| host.draft().duration.0)
    }

    /// Play from the playhead with sound. An empty project does not play.
    pub fn play(&self) -> Result<(), EditorError> {
        self.run(PhoneHost::play)?.map_err(EditorError::from)
    }

    pub fn pause(&self) -> Result<(), EditorError> {
        self.run(PhoneHost::pause)
    }

    /// Move the playhead to `time_us`, held inside the project. Playing
    /// goes on from there.
    pub fn seek(&self, time_us: i64) -> Result<(), EditorError> {
        self.run(move |host| host.seek(Micros(time_us)))
    }

    /// Move a playing playhead by `elapsed_us` with the heard sound and
    /// return it. Call once per screen frame while playing.
    pub fn tick(&self, elapsed_us: i64) -> Result<i64, EditorError> {
        self.run(move |host| host.tick(elapsed_us).0)
    }

    /// Play stopped at the end, or was never started.
    pub fn is_playing(&self) -> Result<bool, EditorError> {
        self.run(|host| host.is_playing())
    }

    /// The picture at `time_us`.
    pub fn frame(&self, time_us: i64) -> Result<Frame, EditorError> {
        let picture = self
            .run(move |host| host.frame(Micros(time_us)))?
            .map_err(EditorError::from)?;
        Ok(Frame {
            width: picture.width,
            height: picture.height,
            rgba: picture.rgba,
        })
    }

    /// Cut a clip at `at_us`. Returns the right piece's id.
    pub fn split(&self, clip_id: String, at_us: i64) -> Result<String, EditorError> {
        self.run(move |host| host.split(&SegmentId::new(clip_id), Micros(at_us)))?
            .map(|id| id.as_str().to_owned())
            .map_err(EditorError::from)
    }

    /// Move a clip to `row` (top is 0) starting near `start_us`, as one undo
    /// step. It snaps to a cut or the playhead within `snap_us`. Returns the
    /// start it landed on.
    pub fn move_clip(
        &self,
        clip_id: String,
        row: u32,
        start_us: i64,
        snap_us: i64,
        playhead_us: i64,
    ) -> Result<i64, EditorError> {
        let drag = Drag {
            row,
            time: Micros(start_us),
            snap: Micros(snap_us),
            playhead: Micros(playhead_us),
        };
        self.run(move |host| host.move_clip(&SegmentId::new(clip_id), drag))?
            .map(|start| start.0)
            .map_err(EditorError::from)
    }

    /// Drag a clip's start edge, or its end edge, to `time_us`, as one undo
    /// step. Returns where the edge landed.
    pub fn trim(
        &self,
        clip_id: String,
        start_edge: bool,
        time_us: i64,
        snap_us: i64,
        playhead_us: i64,
    ) -> Result<i64, EditorError> {
        let drag = Drag {
            row: 0,
            time: Micros(time_us),
            snap: Micros(snap_us),
            playhead: Micros(playhead_us),
        };
        self.run(move |host| host.trim(&SegmentId::new(clip_id), start_edge, drag))?
            .map(|time| time.0)
            .map_err(EditorError::from)
    }

    /// Remove one clip.
    pub fn remove(&self, clip_id: String) -> Result<(), EditorError> {
        self.run(move |host| host.remove(&SegmentId::new(clip_id)))?
            .map_err(EditorError::from)
    }

    pub fn undo(&self) -> Result<(), EditorError> {
        self.run(|host| host.undo().map(|_| ()))?
            .map_err(EditorError::from)
    }

    pub fn redo(&self) -> Result<(), EditorError> {
        self.run(|host| host.redo().map(|_| ()))?
            .map_err(EditorError::from)
    }

    /// Write the draft into the project folder.
    pub fn save(&self) -> Result<(), EditorError> {
        self.run(PhoneHost::save)?.map_err(EditorError::from)
    }
}

impl Editor {
    /// Run `job` on the host's thread and wait for its answer.
    fn run<T: Send + 'static>(
        &self,
        job: impl FnOnce(&mut PhoneHost) -> T + Send + 'static,
    ) -> Result<T, EditorError> {
        let (reply, answer) = mpsc::sync_channel(1);
        self.jobs
            .send(Box::new(move |host| {
                let _ = reply.send(job(host));
            }))
            .map_err(|_| closed())?;
        answer.recv().map_err(|_| closed())
    }
}

fn closed() -> EditorError {
    EditorError::Failed {
        message: "the project is closed".to_owned(),
    }
}

fn lane_kind(kind: cutlass_ffi::TrackType) -> LaneKind {
    use cutlass_ffi::TrackType;
    match kind {
        TrackType::Video => LaneKind::Video,
        TrackType::Audio => LaneKind::Audio,
        TrackType::Text => LaneKind::Text,
        TrackType::Sticker => LaneKind::Sticker,
        TrackType::Effect => LaneKind::Effect,
        TrackType::Filter => LaneKind::Filter,
        TrackType::Adjust => LaneKind::Adjust,
        _ => LaneKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::{Editor, LaneKind, create_project, list_projects, start};

    struct Temp(PathBuf);

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("cutlass-bindings-{}-{n}", std::process::id()));
            std::fs::create_dir_all(path.join("bundle")).unwrap();
            Self(path)
        }

        fn root(&self) -> String {
            self.0.join("Cutlass").display().to_string()
        }

        fn bundle(&self) -> String {
            self.0.join("bundle").display().to_string()
        }

        fn project(&self) -> String {
            self.0
                .join("Cutlass/Projects/project")
                .display()
                .to_string()
        }

        fn open(&self) -> std::sync::Arc<Editor> {
            Editor::open(self.root(), self.bundle(), self.project()).unwrap()
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn pattern() -> String {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../engine/fixtures/pattern.png")
            .display()
            .to_string()
    }

    #[test]
    fn a_clip_trims_and_moves_and_a_bad_drop_says_why() {
        let temp = Temp::new();
        let editor = temp.open();
        let first = editor.append(editor.import(pattern()).unwrap()).unwrap();
        let second = editor.append(editor.import(pattern()).unwrap()).unwrap();
        let length = editor.lanes().unwrap()[0].clips[0].duration_us;
        let edge = editor
            .trim(second.clone(), false, length + length / 2, 0, 0)
            .unwrap();
        assert_eq!(edge, length + length / 2);
        assert_eq!(editor.duration_us().unwrap(), length + length / 2);
        let start = editor
            .move_clip(first.clone(), 0, 10 * length, 0, 0)
            .unwrap();
        let clips = &editor.lanes().unwrap()[0].clips;
        assert_eq!(
            clips.iter().find(|clip| clip.id == first).unwrap().start_us,
            start
        );
        let super::EditorError::Failed { message } =
            editor.move_clip(first, 9, 0, 0, 0).unwrap_err();
        assert!(!message.is_empty());
    }

    #[test]
    fn play_ticks_the_playhead_and_pause_holds_it() {
        let temp = Temp::new();
        let editor = temp.open();
        editor.play().unwrap();
        assert!(
            !editor.is_playing().unwrap(),
            "an empty project does not play"
        );
        let material = editor.import(pattern()).unwrap();
        editor.append(material).unwrap();
        let end = editor.duration_us().unwrap();
        editor.play().unwrap();
        assert!(editor.is_playing().unwrap());
        assert_eq!(editor.tick(40_000).unwrap(), 40_000);
        editor.pause().unwrap();
        assert_eq!(editor.tick(40_000).unwrap(), 40_000);
        editor.seek(end + 1).unwrap();
        editor.play().unwrap();
        assert_eq!(editor.tick(0).unwrap(), 0, "play from the end starts over");
        editor.seek(end - 10).unwrap();
        assert_eq!(editor.tick(20).unwrap(), end);
        assert!(!editor.is_playing().unwrap());
    }

    #[test]
    fn a_picked_still_lands_on_main_draws_and_saves() {
        let temp = Temp::new();
        let editor = temp.open();
        let lanes = editor.lanes().unwrap();
        assert_eq!(lanes.len(), 1);
        assert!(lanes[0].main && lanes[0].clips.is_empty());

        let material = editor.import(pattern()).unwrap();
        let clip = editor.append(material).unwrap();
        let lanes = editor.lanes().unwrap();
        assert_eq!(lanes[0].kind, LaneKind::Video);
        assert_eq!(lanes[0].clips[0].id, clip);
        assert_eq!(lanes[0].clips[0].name, "pattern.png");
        assert_eq!(editor.duration_us().unwrap(), 3_000_000);
        let frame = editor.frame(500_000).unwrap();
        assert_eq!(
            frame.rgba.len(),
            frame.width as usize * frame.height as usize * 4
        );

        let right = editor.split(clip.clone(), 300_000).unwrap();
        assert_eq!(editor.lanes().unwrap()[0].clips.len(), 2);
        editor.undo().unwrap();
        assert_eq!(editor.lanes().unwrap()[0].clips.len(), 1);
        editor.redo().unwrap();
        editor.remove(right).unwrap();
        editor.save().unwrap();
        drop(editor);

        let opened = temp.open();
        let clips = &opened.lanes().unwrap()[0].clips;
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].duration_us, 300_000);
    }

    #[test]
    fn a_refusal_comes_back_as_a_message_and_changes_nothing() {
        let temp = Temp::new();
        let editor = temp.open();
        let error = editor.append("missing".to_owned()).unwrap_err();
        assert_eq!(error.to_string(), "the media is not in the library");
        let notes = temp.0.join("notes.txt");
        std::fs::write(&notes, b"not media").unwrap();
        let error = editor.import(notes.display().to_string()).unwrap_err();
        assert!(
            error.to_string().contains("not a media container"),
            "{error}"
        );
        assert!(editor.frame(-1).is_err());
        assert!(editor.undo().is_err());
        assert_eq!(editor.lanes().unwrap()[0].clips.len(), 0);
    }

    #[test]
    fn a_bundle_that_is_not_a_folder_refuses_open() {
        let temp = Temp::new();
        let error = Editor::open(
            temp.root(),
            temp.0.join("no-bundle").display().to_string(),
            temp.project(),
        );
        assert!(error.is_err());
    }

    #[test]
    fn a_created_project_is_listed_and_opens() {
        let temp = Temp::new();
        let root = temp.root();
        assert!(list_projects(root.clone()).is_empty());
        let made = create_project(root.clone()).unwrap();
        assert!(made.ends_with("Project"));
        let listed = list_projects(root.clone());
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "Project");
        assert_eq!(listed[0].path, made);
        assert_eq!(listed[0].duration_us, 0);
        assert!(listed[0].modified_ms > 0);
        assert!(made.starts_with(&format!("{root}/Projects/")));
        let editor = Editor::open(root, temp.bundle(), made).unwrap();
        assert_eq!(editor.duration_us().unwrap(), 0);
        assert!(temp.0.join("Cutlass/Cache/index.json").is_file());
    }

    #[test]
    fn start_lays_out_the_root_and_writes_the_log_there() {
        let temp = Temp::new();
        start(temp.root()).unwrap();
        for folder in [
            "Projects",
            "Cache",
            "Downloads",
            "Proxies",
            "Models",
            "Logs",
        ] {
            assert!(temp.0.join("Cutlass").join(folder).is_dir(), "{folder}");
        }
        assert!(temp.0.join("Cutlass/Logs/cutlass.log").is_file());
        start(temp.root()).unwrap();
    }

    #[test]
    fn a_relative_root_is_refused_and_lists_nothing() {
        assert!(start("Cutlass".to_owned()).is_err());
        assert!(create_project("Cutlass".to_owned()).is_err());
        assert!(list_projects("Cutlass".to_owned()).is_empty());
        assert!(!std::path::Path::new("Cutlass").exists());
    }

    #[test]
    fn a_project_outside_the_root_is_refused_and_not_created() {
        let temp = Temp::new();
        for outside in [
            temp.0.join("elsewhere"),
            temp.0.join("Cutlass/Projects/a/nested"),
            temp.0.join("Cutlass/Projects/../escape"),
        ] {
            let path = outside.display().to_string();
            let Err(error) = Editor::open(temp.root(), temp.bundle(), path) else {
                panic!("{} opened", outside.display());
            };
            assert!(error.to_string().contains("not a project"), "{error}");
            assert!(!outside.exists());
        }
    }

    #[test]
    fn a_root_that_is_a_file_cannot_hold_a_project() {
        let temp = Temp::new();
        let file = temp.0.join("not-a-folder");
        std::fs::write(&file, b"x").unwrap();
        let root = file.display().to_string();
        assert!(list_projects(root.clone()).is_empty());
        assert!(create_project(root).is_err());
    }
}
