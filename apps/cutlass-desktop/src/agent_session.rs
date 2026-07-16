//! Per-draft AI session persistence.
//!
//! Sessions live beside app-owned drafts so project switches preserve context.
//! Private DTOs keep the disk schema versioned and ensure image bytes never
//! enter project storage.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use cutlass_ai::provider::{ImagePart, Message, ToolCall};
use serde::{Deserialize, Serialize};

const SESSION_FILE: &str = "agent-session.json";
const TEMP_FILE: &str = ".agent-session.json.tmp";
const FORMAT_VERSION: u32 = 1;
const MAX_FILE_SIZE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_HISTORY_MESSAGES: usize = 1_000;
const MAX_TRANSCRIPT_ENTRIES: usize = 2_000;

/// One rendered transcript row. `kind` is persisted verbatim; the UI decides
/// how known, unknown, or empty kinds should look.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TranscriptEntry {
    pub kind: String,
    pub text: String,
}

/// Provider history and visible transcript for one draft.
///
/// Loaded user and tool-result messages always have empty image vectors. Each
/// saved image is represented instead by a label placeholder in its content.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct AgentSession {
    pub history: Vec<Message>,
    pub transcript: Vec<TranscriptEntry>,
}

/// Return the session sidecar beside `project`, or `None` when it has no
/// parent directory.
pub(crate) fn path_for_project(project: &Path) -> Option<PathBuf> {
    project.parent().map(|parent| parent.join(SESSION_FILE))
}

/// Load one draft's session.
///
/// A missing sidecar is an empty session. Files larger than 4 MiB, malformed
/// JSON, and unsupported versions are rejected. Valid oversized collections
/// retain only their newest bounded entries.
pub(crate) fn load(project: &Path) -> Result<AgentSession, String> {
    let path = session_path(project)?;
    let file = match fs::File::open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(AgentSession::default());
        }
        Err(error) => {
            return Err(format!(
                "failed to open agent session '{}': {error}",
                path.display()
            ));
        }
    };

    let file_size = file
        .metadata()
        .map_err(|error| {
            format!(
                "failed to inspect agent session '{}': {error}",
                path.display()
            )
        })?
        .len();
    if file_size > MAX_FILE_SIZE_BYTES {
        return Err(oversize_error(&path, file_size));
    }

    // The metadata check prevents preallocation from an already-large file;
    // `take` also bounds a file that grows between metadata and reading.
    let mut bytes = Vec::with_capacity(file_size as usize);
    file.take(MAX_FILE_SIZE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read agent session '{}': {error}", path.display()))?;
    if bytes.len() as u64 > MAX_FILE_SIZE_BYTES {
        return Err(oversize_error(&path, bytes.len() as u64));
    }

    let header: PersistedSessionHeader = serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "failed to parse agent session '{}': {error}",
            path.display()
        )
    })?;
    if header.version != FORMAT_VERSION {
        return Err(format!(
            "unsupported agent session version {} in '{}'; expected {}",
            header.version,
            path.display(),
            FORMAT_VERSION
        ));
    }
    let mut persisted: PersistedSession = serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "failed to parse agent session '{}': {error}",
            path.display()
        )
    })?;

    retain_newest(&mut persisted.history, MAX_HISTORY_MESSAGES);
    retain_newest(&mut persisted.transcript, MAX_TRANSCRIPT_ENTRIES);
    Ok(persisted.into_runtime())
}

/// Persist one draft's session without storing image MIME types or bytes.
///
/// The draft directory is created when needed. A complete pretty-printed
/// document is written to a fixed same-directory temporary file and then
/// renamed over the prior sidecar.
pub(crate) fn save(project: &Path, session: &AgentSession) -> Result<(), String> {
    let path = session_path(project)?;
    let parent = path.parent().ok_or_else(|| {
        format!(
            "agent session path '{}' has no parent directory",
            path.display()
        )
    })?;
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "failed to create agent session directory '{}': {error}",
            parent.display()
        )
    })?;

    let bytes = serde_json::to_vec_pretty(&PersistedSession::from_runtime(session))
        .map_err(|error| format!("failed to serialize agent session: {error}"))?;
    if bytes.len() as u64 > MAX_FILE_SIZE_BYTES {
        return Err(format!(
            "agent session is too large to save ({} bytes; maximum is {} bytes)",
            bytes.len(),
            MAX_FILE_SIZE_BYTES
        ));
    }

    let temp = parent.join(TEMP_FILE);
    if let Err(error) = fs::write(&temp, bytes) {
        let _ = fs::remove_file(&temp);
        return Err(format!(
            "failed to write temporary agent session '{}': {error}",
            temp.display()
        ));
    }
    if let Err(error) = replace_file(&temp, &path) {
        let _ = fs::remove_file(&temp);
        return Err(format!(
            "failed to replace agent session '{}': {error}",
            path.display()
        ));
    }
    Ok(())
}

fn session_path(project: &Path) -> Result<PathBuf, String> {
    path_for_project(project).ok_or_else(|| {
        format!(
            "cannot locate agent session for project path '{}': no parent directory",
            project.display()
        )
    })
}

fn oversize_error(path: &Path, size: u64) -> String {
    format!(
        "agent session '{}' is too large ({size} bytes; maximum is {MAX_FILE_SIZE_BYTES} bytes)",
        path.display()
    )
}

fn retain_newest<T>(entries: &mut Vec<T>, maximum: usize) {
    let excess = entries.len().saturating_sub(maximum);
    if excess > 0 {
        entries.drain(..excess);
    }
}

fn replace_file(temp: &Path, destination: &Path) -> std::io::Result<()> {
    match fs::rename(temp, destination) {
        Ok(()) => Ok(()),
        Err(error) => {
            // Rust's Windows implementation normally replaces files via
            // MoveFileExW. The fallback also covers filesystems that reject
            // replacement while a destination entry exists.
            #[cfg(windows)]
            {
                if destination.is_file() {
                    fs::remove_file(destination)?;
                    fs::rename(temp, destination)
                } else {
                    Err(error)
                }
            }
            #[cfg(not(windows))]
            {
                Err(error)
            }
        }
    }
}

fn content_with_image_labels(content: &str, images: &[ImagePart]) -> String {
    let mut persisted = content.to_owned();
    for image in images {
        persisted.push_str("\n[image: ");
        persisted.push_str(&image.label);
        persisted.push(']');
    }
    persisted
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedSessionHeader {
    version: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedSession {
    version: u32,
    history: Vec<PersistedMessage>,
    transcript: Vec<PersistedTranscriptEntry>,
}

impl PersistedSession {
    fn from_runtime(session: &AgentSession) -> Self {
        let history_start = session.history.len().saturating_sub(MAX_HISTORY_MESSAGES);
        let transcript_start = session
            .transcript
            .len()
            .saturating_sub(MAX_TRANSCRIPT_ENTRIES);
        Self {
            version: FORMAT_VERSION,
            history: session
                .history
                .iter()
                .skip(history_start)
                .map(PersistedMessage::from_runtime)
                .collect(),
            transcript: session
                .transcript
                .iter()
                .skip(transcript_start)
                .map(PersistedTranscriptEntry::from_runtime)
                .collect(),
        }
    }

    fn into_runtime(self) -> AgentSession {
        AgentSession {
            history: self
                .history
                .into_iter()
                .map(PersistedMessage::into_runtime)
                .collect(),
            transcript: self
                .transcript
                .into_iter()
                .map(PersistedTranscriptEntry::into_runtime)
                .collect(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PersistedMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: String,
        tool_calls: Vec<PersistedToolCall>,
    },
    ToolResult {
        call_id: String,
        content: String,
    },
}

impl PersistedMessage {
    fn from_runtime(message: &Message) -> Self {
        match message {
            Message::System { content } => Self::System {
                content: content.clone(),
            },
            Message::User { content, images } => Self::User {
                content: content_with_image_labels(content, images),
            },
            Message::Assistant {
                content,
                tool_calls,
            } => Self::Assistant {
                content: content.clone(),
                tool_calls: tool_calls
                    .iter()
                    .map(PersistedToolCall::from_runtime)
                    .collect(),
            },
            Message::ToolResult {
                call_id,
                content,
                images,
            } => Self::ToolResult {
                call_id: call_id.clone(),
                content: content_with_image_labels(content, images),
            },
        }
    }

    fn into_runtime(self) -> Message {
        match self {
            Self::System { content } => Message::System { content },
            Self::User { content } => Message::User {
                content,
                images: Vec::new(),
            },
            Self::Assistant {
                content,
                tool_calls,
            } => Message::Assistant {
                content,
                tool_calls: tool_calls
                    .into_iter()
                    .map(PersistedToolCall::into_runtime)
                    .collect(),
            },
            Self::ToolResult { call_id, content } => Message::ToolResult {
                call_id,
                content,
                images: Vec::new(),
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedToolCall {
    id: String,
    name: String,
    arguments: serde_json::Value,
}

impl PersistedToolCall {
    fn from_runtime(call: &ToolCall) -> Self {
        Self {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        }
    }

    fn into_runtime(self) -> ToolCall {
        ToolCall {
            id: self.id,
            name: self.name,
            arguments: self.arguments,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedTranscriptEntry {
    kind: String,
    text: String,
}

impl PersistedTranscriptEntry {
    fn from_runtime(entry: &TranscriptEntry) -> Self {
        Self {
            kind: entry.kind.clone(),
            text: entry.text.clone(),
        }
    }

    fn into_runtime(self) -> TranscriptEntry {
        TranscriptEntry {
            kind: self.kind,
            text: self.text,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn project_path(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("draft").join("project.cutlass")
    }

    fn write_sidecar(project: &Path, contents: impl AsRef<[u8]>) {
        let path = path_for_project(project).expect("session path");
        fs::create_dir_all(path.parent().expect("draft directory")).expect("create draft");
        fs::write(path, contents).expect("write sidecar");
    }

    #[test]
    fn missing_file_loads_default_session() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project = project_path(&dir);

        assert_eq!(load(&project), Ok(AgentSession::default()));
    }

    #[test]
    fn all_message_variants_round_trip_without_image_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project = project_path(&dir);
        let tool_calls = vec![ToolCall {
            id: "call-17".to_owned(),
            name: "timeline.apply".to_owned(),
            arguments: json!({
                "commands": [
                    {"type": "split", "clip_id": 7, "at": 2.0},
                    {"type": "trim", "range": [1, 4], "enabled": true}
                ],
                "note": null
            }),
        }];
        let session = AgentSession {
            history: vec![
                Message::System {
                    content: "You edit timelines.".to_owned(),
                },
                Message::User {
                    content: "Cut at the frame shown.".to_owned(),
                    images: vec![
                        ImagePart::png(vec![0, 1, 2, 3], "timeline at 2.00s"),
                        ImagePart::jpeg(vec![4, 5, 6], "selected clip"),
                    ],
                },
                Message::Assistant {
                    content: "I'll make that cut.".to_owned(),
                    tool_calls: tool_calls.clone(),
                },
                Message::ToolResult {
                    call_id: "call-17".to_owned(),
                    content: "Applied.".to_owned(),
                    images: vec![ImagePart::png(vec![7, 8, 9], "result at 2.00s")],
                },
            ],
            transcript: vec![
                TranscriptEntry {
                    kind: "user".to_owned(),
                    text: "Cut at the frame shown.".to_owned(),
                },
                TranscriptEntry {
                    kind: "reasoning".to_owned(),
                    text: "The requested frame is a safe split point.".to_owned(),
                },
                TranscriptEntry {
                    kind: "assistant".to_owned(),
                    text: "Done.".to_owned(),
                },
            ],
        };

        save(&project, &session).expect("save");
        let raw =
            fs::read_to_string(path_for_project(&project).expect("session path")).expect("read");
        assert!(!raw.contains("image/png"));
        assert!(!raw.contains("image/jpeg"));
        assert!(raw.contains("\"kind\": \"reasoning\""));

        let loaded = load(&project).expect("load");
        assert_eq!(
            loaded,
            AgentSession {
                history: vec![
                    Message::System {
                        content: "You edit timelines.".to_owned(),
                    },
                    Message::User {
                        content: concat!(
                            "Cut at the frame shown.",
                            "\n[image: timeline at 2.00s]",
                            "\n[image: selected clip]"
                        )
                        .to_owned(),
                        images: Vec::new(),
                    },
                    Message::Assistant {
                        content: "I'll make that cut.".to_owned(),
                        tool_calls,
                    },
                    Message::ToolResult {
                        call_id: "call-17".to_owned(),
                        content: "Applied.\n[image: result at 2.00s]".to_owned(),
                        images: Vec::new(),
                    },
                ],
                transcript: session.transcript,
            }
        );
    }

    #[test]
    fn version_mismatch_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project = project_path(&dir);
        write_sidecar(&project, br#"{"version":2,"history":[],"transcript":[]}"#);

        let error = load(&project).expect_err("version mismatch should fail");
        assert!(error.contains("unsupported agent session version 2"));
        assert!(error.contains("expected 1"));
    }

    #[test]
    fn malformed_json_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project = project_path(&dir);
        write_sidecar(&project, b"{not valid json");

        let error = load(&project).expect_err("malformed JSON should fail");
        assert!(error.contains("failed to parse agent session"));
    }

    #[test]
    fn oversized_file_is_rejected_before_parsing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project = project_path(&dir);
        write_sidecar(&project, vec![b'x'; MAX_FILE_SIZE_BYTES as usize + 1]);

        let error = load(&project).expect_err("oversized session should fail");
        assert!(error.contains("is too large"));
        assert!(!error.contains("failed to parse"));
    }

    #[test]
    fn collection_caps_retain_newest_entries() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project = project_path(&dir);
        let extra_history = 7;
        let extra_transcript = 11;
        let session = AgentSession {
            history: (0..MAX_HISTORY_MESSAGES + extra_history)
                .map(|index| Message::user(format!("history-{index}")))
                .collect(),
            transcript: (0..MAX_TRANSCRIPT_ENTRIES + extra_transcript)
                .map(|index| TranscriptEntry {
                    kind: String::new(),
                    text: format!("transcript-{index}"),
                })
                .collect(),
        };
        save(&project, &session).expect("save");

        let raw =
            fs::read(path_for_project(&project).expect("session path")).expect("read sidecar");
        let persisted: PersistedSession = serde_json::from_slice(&raw).expect("persisted session");
        assert_eq!(persisted.history.len(), MAX_HISTORY_MESSAGES);
        assert_eq!(persisted.transcript.len(), MAX_TRANSCRIPT_ENTRIES);

        let loaded = load(&project).expect("load");
        assert_eq!(loaded.history.len(), MAX_HISTORY_MESSAGES);
        assert_eq!(
            loaded.history.first(),
            Some(&Message::user(format!("history-{extra_history}")))
        );
        assert_eq!(
            loaded.history.last(),
            Some(&Message::user(format!(
                "history-{}",
                MAX_HISTORY_MESSAGES + extra_history - 1
            )))
        );
        assert_eq!(loaded.transcript.len(), MAX_TRANSCRIPT_ENTRIES);
        assert_eq!(
            loaded.transcript.first(),
            Some(&TranscriptEntry {
                kind: String::new(),
                text: format!("transcript-{extra_transcript}"),
            })
        );
        assert_eq!(
            loaded.transcript.last(),
            Some(&TranscriptEntry {
                kind: String::new(),
                text: format!(
                    "transcript-{}",
                    MAX_TRANSCRIPT_ENTRIES + extra_transcript - 1
                ),
            })
        );
    }

    #[test]
    fn save_replaces_prior_sidecar_and_removes_temp_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project = project_path(&dir);
        fs::create_dir_all(project.parent().expect("draft directory")).expect("create draft");
        fs::write(&project, b"project sentinel").expect("write project");

        let first = AgentSession {
            history: vec![Message::user("first")],
            transcript: vec![],
        };
        save(&project, &first).expect("first save");
        let second = AgentSession {
            history: vec![Message::assistant_text("second")],
            transcript: vec![TranscriptEntry {
                kind: "custom".to_owned(),
                text: "replacement".to_owned(),
            }],
        };
        save(&project, &second).expect("replacement save");

        assert_eq!(load(&project), Ok(second));
        assert_eq!(
            fs::read(&project).expect("read project"),
            b"project sentinel"
        );
        assert!(
            !project
                .parent()
                .expect("draft directory")
                .join(TEMP_FILE)
                .exists()
        );
    }
}
