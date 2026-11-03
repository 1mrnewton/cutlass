//! The Swift and Kotlin surface over `ffi`.
//!
//! One thread owns each open [`cutlass_ffi::PhoneHost`], because its audio
//! output cannot move between threads. A call sends that thread a job and
//! waits for the answer, then turns it into plain records. Times are
//! microseconds. A failure is a message for the phone to show; the draft is
//! unchanged.

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{self, Sender};

use cutlass_ffi::{FfiError, PhoneHost, Probed, STILL_DURATION};
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

/// What the phone read from a picked file. A still lasts as long as it
/// does on the desktop.
#[derive(Clone, Copy, Debug, uniffi::Enum)]
pub enum Media {
    Still {
        width: i32,
        height: i32,
    },
    Movie {
        width: i32,
        height: i32,
        duration_us: i64,
    },
    Sound {
        duration_us: i64,
    },
}

impl From<Media> for Probed {
    fn from(media: Media) -> Self {
        match media {
            Media::Still { width, height } => Self::Picture {
                still: true,
                width,
                height,
                duration: STILL_DURATION,
            },
            Media::Movie {
                width,
                height,
                duration_us,
            } => Self::Picture {
                still: false,
                width,
                height,
                duration: Micros(duration_us),
            },
            Media::Sound { duration_us } => Self::Sound {
                duration: Micros(duration_us),
            },
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

/// One open project.
#[derive(uniffi::Object)]
pub struct Editor {
    jobs: Sender<Job>,
}

type Job = Box<dyn FnOnce(&mut PhoneHost) + Send>;

#[uniffi::export]
impl Editor {
    /// Open the project folder, or create an empty one when it has no draft.
    /// `bundle` must be a directory. The cache directory is created.
    #[uniffi::constructor]
    pub fn open(project: String, bundle: String, cache: String) -> Result<Arc<Self>, EditorError> {
        let (jobs, inbox) = mpsc::channel::<Job>();
        let (opened, answer) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("cutlass-editor".to_owned())
            .spawn(move || {
                let (project, bundle, cache) = (
                    PathBuf::from(project),
                    PathBuf::from(bundle),
                    PathBuf::from(cache),
                );
                let host = if project.join("draft_info.json").is_file() {
                    PhoneHost::open(&project, &bundle, &cache, 0)
                } else {
                    PhoneHost::create(&project, "draft", &bundle, &cache, 0)
                };
                let mut host = match host {
                    Ok(host) => {
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

    /// Copy a picked file into the project and add it to the library.
    /// Returns the material id.
    pub fn import(&self, path: String, media: Media) -> Result<String, EditorError> {
        self.run(move |host| host.import(&PathBuf::from(path), media.into()))?
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

    use super::{Editor, LaneKind, Media};

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

        fn open(&self) -> std::sync::Arc<Editor> {
            Editor::open(
                self.0.join("project").display().to_string(),
                self.0.join("bundle").display().to_string(),
                self.0.join("cache").display().to_string(),
            )
            .unwrap()
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

    const STILL: Media = Media::Still {
        width: 16,
        height: 16,
    };

    #[test]
    fn a_picked_still_lands_on_main_draws_and_saves() {
        let temp = Temp::new();
        let editor = temp.open();
        let lanes = editor.lanes().unwrap();
        assert_eq!(lanes.len(), 1);
        assert!(lanes[0].main && lanes[0].clips.is_empty());

        let material = editor.import(pattern(), STILL).unwrap();
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
        let error = editor
            .import(pattern(), Media::Sound { duration_us: 0 })
            .unwrap_err();
        assert_eq!(error.to_string(), "the media has no duration");
        assert!(editor.frame(-1).is_err());
        assert!(editor.undo().is_err());
        assert_eq!(editor.lanes().unwrap()[0].clips.len(), 0);
    }

    #[test]
    fn a_bundle_that_is_not_a_folder_refuses_open() {
        let temp = Temp::new();
        let error = Editor::open(
            temp.0.join("project").display().to_string(),
            temp.0.join("no-bundle").display().to_string(),
            temp.0.join("cache").display().to_string(),
        );
        assert!(error.is_err());
    }
}
