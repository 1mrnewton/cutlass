//! Per-draft AI session persistence.
//!
//! Sessions live beside app-owned drafts so project switches preserve context.
//! Private DTOs keep the disk schema versioned and ensure image bytes never
//! enter project storage.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_ai::provider::{ImagePart, Message, ToolCall};
use serde::{Deserialize, Serialize};

const SESSION_FILE: &str = "agent-session.json";
#[cfg(test)]
const TEMP_FILE: &str = ".agent-session.json.tmp";
const CHAT_DIRECTORY: &str = "agent-chats";
const CHAT_ID_PREFIX: &str = "chat-";
const CHAT_TITLE_CHARS: usize = 40;
const FORMAT_VERSION: u32 = 1;
const MAX_FILE_SIZE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_HISTORY_MESSAGES: usize = 1_000;
const MAX_TRANSCRIPT_ENTRIES: usize = 2_000;

/// One rendered transcript row. `kind` is persisted verbatim; the UI decides
/// how known, unknown, or empty kinds should look. `usage` is the optional
/// per-prompt token line attached to assistant rows.
#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct TranscriptEntry {
    pub kind: String,
    pub text: String,
    pub usage: String,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ChatMeta {
    pub id: String,
    pub title: String,
    pub updated_millis: u64,
}

/// Return the session sidecar beside `project`, or `None` when it has no
/// parent directory.
pub(crate) fn path_for_project(project: &Path) -> Option<PathBuf> {
    project.parent().map(|parent| parent.join(SESSION_FILE))
}

pub(crate) fn allocate_chat_id(project: &Path) -> Result<String, String> {
    let directory = chat_directory(project)?;
    let now = system_time_millis(SystemTime::now());
    for offset in 0..10_000 {
        let Some(timestamp) = now.checked_add(offset) else {
            break;
        };
        let id = format!("{CHAT_ID_PREFIX}{timestamp}");
        if !directory.join(format!("{id}.json")).exists() {
            return Ok(id);
        }
    }
    Err("could not allocate a unique agent chat id".to_string())
}

pub(crate) fn list_chats(project: &Path) -> Result<Vec<ChatMeta>, String> {
    migrate_legacy_session(project)?;
    let directory = chat_directory(project)?;
    let entries = match fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(format!(
                "failed to list agent chats '{}': {error}",
                directory.display()
            ));
        }
    };
    let mut chats = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "failed to read an agent chat entry in '{}': {error}",
                directory.display()
            )
        })?;
        let path = entry.path();
        if !entry
            .file_type()
            .map_err(|error| format!("failed to inspect '{}': {error}", path.display()))?
            .is_file()
        {
            continue;
        }
        let Some(id) = chat_id_from_path(&path) else {
            continue;
        };
        let metadata = entry
            .metadata()
            .map_err(|error| format!("failed to inspect '{}': {error}", path.display()))?;
        let updated_millis = metadata
            .modified()
            .map(system_time_millis)
            .unwrap_or_default();
        let title = load_file(&path)
            .map(|session| chat_title(&session))
            .unwrap_or_else(|_| "Unreadable chat".to_string());
        chats.push(ChatMeta {
            id,
            title,
            updated_millis,
        });
    }
    chats.sort_by(|left, right| {
        right
            .updated_millis
            .cmp(&left.updated_millis)
            .then_with(|| right.id.cmp(&left.id))
    });
    Ok(chats)
}

pub(crate) fn load_chat(project: &Path, id: &str) -> Result<AgentSession, String> {
    migrate_legacy_session(project)?;
    let path = chat_path(project, id)?;
    load_file(&path)
}

pub(crate) fn save_chat(project: &Path, id: &str, session: &AgentSession) -> Result<(), String> {
    let path = chat_path(project, id)?;
    let temp_name = format!(".{id}.tmp");
    save_file(&path, &temp_name, session)
}

/// Load one draft's session.
///
/// A missing sidecar is an empty session. Files larger than 4 MiB, malformed
/// JSON, and unsupported versions are rejected. Valid oversized collections
/// retain only their newest bounded entries.
#[cfg(test)]
pub(crate) fn load(project: &Path) -> Result<AgentSession, String> {
    let path = session_path(project)?;
    load_file(&path)
}

fn load_file(path: &Path) -> Result<AgentSession, String> {
    let file = match fs::File::open(path) {
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
        return Err(oversize_error(path, file_size));
    }

    // The metadata check prevents preallocation from an already-large file;
    // `take` also bounds a file that grows between metadata and reading.
    let mut bytes = Vec::with_capacity(file_size as usize);
    file.take(MAX_FILE_SIZE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read agent session '{}': {error}", path.display()))?;
    if bytes.len() as u64 > MAX_FILE_SIZE_BYTES {
        return Err(oversize_error(path, bytes.len() as u64));
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
#[cfg(test)]
pub(crate) fn save(project: &Path, session: &AgentSession) -> Result<(), String> {
    let path = session_path(project)?;
    save_file(&path, TEMP_FILE, session)
}

fn save_file(path: &Path, temp_name: &str, session: &AgentSession) -> Result<(), String> {
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

    let temp = parent.join(temp_name);
    if let Err(error) = fs::write(&temp, bytes) {
        let _ = fs::remove_file(&temp);
        return Err(format!(
            "failed to write temporary agent session '{}': {error}",
            temp.display()
        ));
    }
    if let Err(error) = replace_file(&temp, path) {
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

fn chat_directory(project: &Path) -> Result<PathBuf, String> {
    project
        .parent()
        .map(|parent| parent.join(CHAT_DIRECTORY))
        .ok_or_else(|| {
            format!(
                "cannot locate agent chats for project path '{}': no parent directory",
                project.display()
            )
        })
}

fn chat_path(project: &Path, id: &str) -> Result<PathBuf, String> {
    if !valid_chat_id(id) {
        return Err(format!("invalid agent chat id '{id}'"));
    }
    Ok(chat_directory(project)?.join(format!("{id}.json")))
}

fn valid_chat_id(id: &str) -> bool {
    id.strip_prefix(CHAT_ID_PREFIX).is_some_and(|suffix| {
        !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
    })
}

fn chat_id_from_path(path: &Path) -> Option<String> {
    (path.extension().and_then(|extension| extension.to_str()) == Some("json"))
        .then(|| path.file_stem().and_then(|stem| stem.to_str()))
        .flatten()
        .filter(|id| valid_chat_id(id))
        .map(str::to_owned)
}

fn migrate_legacy_session(project: &Path) -> Result<(), String> {
    let legacy = session_path(project)?;
    let directory = chat_directory(project)?;
    if directory.exists() || !legacy.exists() {
        return Ok(());
    }
    fs::create_dir_all(&directory).map_err(|error| {
        format!(
            "failed to create agent chat directory '{}': {error}",
            directory.display()
        )
    })?;
    let id = allocate_chat_id(project)?;
    let destination = chat_path(project, &id)?;
    if let Err(error) = fs::rename(&legacy, &destination) {
        let _ = fs::remove_dir(&directory);
        return Err(format!(
            "failed to migrate agent session '{}' to '{}': {error}",
            legacy.display(),
            destination.display()
        ));
    }
    Ok(())
}

fn chat_title(session: &AgentSession) -> String {
    let title = session
        .transcript
        .iter()
        .find(|entry| entry.kind == "user")
        .map(|entry| entry.text.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|title| !title.is_empty())
        .unwrap_or_else(|| "New chat".to_string());
    let mut chars = title.chars();
    let abbreviated = chars.by_ref().take(CHAT_TITLE_CHARS).collect::<String>();
    if chars.next().is_some() {
        format!("{abbreviated}…")
    } else {
        abbreviated
    }
}

fn system_time_millis(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
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
    /// Absent in sessions written before usage lines existed.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    usage: String,
}

impl PersistedTranscriptEntry {
    fn from_runtime(entry: &TranscriptEntry) -> Self {
        Self {
            kind: entry.kind.clone(),
            text: entry.text.clone(),
            usage: entry.usage.clone(),
        }
    }

    fn into_runtime(self) -> TranscriptEntry {
        TranscriptEntry {
            kind: self.kind,
            text: self.text,
            usage: self.usage,
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

    fn chat_session(prompt: &str) -> AgentSession {
        AgentSession {
            history: vec![Message::user(prompt)],
            transcript: vec![TranscriptEntry {
                kind: "user".to_owned(),
                text: prompt.to_owned(),
                usage: String::new(),
            }],
        }
    }

    fn entry(kind: &str, text: &str) -> TranscriptEntry {
        TranscriptEntry {
            kind: kind.to_owned(),
            text: text.to_owned(),
            usage: String::new(),
        }
    }

    #[test]
    fn missing_file_loads_default_session() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project = project_path(&dir);

        assert_eq!(load(&project), Ok(AgentSession::default()));
    }

    #[test]
    fn chat_files_round_trip_and_list_newest_first_with_titles() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project = project_path(&dir);
        let first = chat_session("First editing request");
        let second = chat_session("Second editing request");

        save_chat(&project, "chat-100", &first).expect("save first chat");
        save_chat(&project, "chat-200", &second).expect("save second chat");

        assert_eq!(load_chat(&project, "chat-100"), Ok(first));
        assert_eq!(load_chat(&project, "chat-200"), Ok(second));
        let chats = list_chats(&project).expect("list chats");
        assert_eq!(
            chats
                .iter()
                .map(|chat| (chat.id.as_str(), chat.title.as_str()))
                .collect::<Vec<_>>(),
            [
                ("chat-200", "Second editing request"),
                ("chat-100", "First editing request"),
            ]
        );
    }

    #[test]
    fn chat_titles_normalize_whitespace_truncate_and_fall_back() {
        let long = "  make\n something   visually interesting with all of these clips please  ";
        let title = chat_title(&chat_session(long));
        assert!(title.ends_with('…'), "{title}");
        assert_eq!(title.trim(), title);
        assert!(!title.contains('\n'));
        assert_eq!(title.chars().count(), CHAT_TITLE_CHARS + 1);
        assert_eq!(chat_title(&AgentSession::default()), "New chat");
    }

    #[test]
    fn legacy_sidecar_migrates_once_into_the_chat_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project = project_path(&dir);
        let legacy = chat_session("Continue my original conversation");
        save(&project, &legacy).expect("save legacy sidecar");
        let legacy_path = path_for_project(&project).expect("legacy path");
        assert!(legacy_path.exists());

        let chats = list_chats(&project).expect("migration and listing");
        assert_eq!(chats.len(), 1);
        assert_eq!(chats[0].title, "Continue my original conversation");
        assert!(!legacy_path.exists(), "legacy file is removed by rename");
        assert_eq!(load_chat(&project, &chats[0].id), Ok(legacy));

        let again = list_chats(&project).expect("second listing");
        assert_eq!(again, chats, "migration is idempotent");
    }

    #[test]
    fn chat_ids_reject_path_traversal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project = project_path(&dir);
        let error = load_chat(&project, "../agent-session").expect_err("invalid id");
        assert!(error.contains("invalid agent chat id"), "{error}");
    }

    #[test]
    fn collection_caps_apply_to_each_chat_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project = project_path(&dir);
        let session = AgentSession {
            history: (0..MAX_HISTORY_MESSAGES + 3)
                .map(|index| Message::user(format!("history-{index}")))
                .collect(),
            transcript: (0..MAX_TRANSCRIPT_ENTRIES + 5)
                .map(|index| entry("user", &format!("transcript-{index}")))
                .collect(),
        };

        save_chat(&project, "chat-42", &session).expect("save bounded chat");
        let loaded = load_chat(&project, "chat-42").expect("load bounded chat");
        assert_eq!(loaded.history.len(), MAX_HISTORY_MESSAGES);
        assert_eq!(loaded.transcript.len(), MAX_TRANSCRIPT_ENTRIES);
        assert_eq!(
            loaded.history.first(),
            Some(&Message::user("history-3".to_string()))
        );
        assert_eq!(loaded.transcript[0].text, "transcript-5");
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
                entry("user", "Cut at the frame shown."),
                entry("reasoning", "The requested frame is a safe split point."),
                TranscriptEntry {
                    kind: "assistant".to_owned(),
                    text: "Done.".to_owned(),
                    usage: "12.3k tokens in (0% cached) · 456 out · $0.68".to_owned(),
                },
            ],
        };

        save(&project, &session).expect("save");
        let raw =
            fs::read_to_string(path_for_project(&project).expect("session path")).expect("read");
        assert!(!raw.contains("image/png"));
        assert!(!raw.contains("image/jpeg"));
        assert!(raw.contains("\"kind\": \"reasoning\""));
        assert!(raw.contains("12.3k tokens in (0% cached)"));

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
                .map(|index| entry("", &format!("transcript-{index}")))
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
            Some(&entry("", &format!("transcript-{extra_transcript}")))
        );
        assert_eq!(
            loaded.transcript.last(),
            Some(&entry(
                "",
                &format!(
                    "transcript-{}",
                    MAX_TRANSCRIPT_ENTRIES + extra_transcript - 1
                )
            ))
        );
    }

    #[test]
    fn transcript_usage_defaults_when_absent_from_older_sessions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project = project_path(&dir);
        write_sidecar(
            &project,
            br#"{"version":1,"history":[],"transcript":[{"kind":"assistant","text":"Hello"}]}"#,
        );
        let loaded = load(&project).expect("load legacy transcript");
        assert_eq!(
            loaded.transcript,
            vec![TranscriptEntry {
                kind: "assistant".to_owned(),
                text: "Hello".to_owned(),
                usage: String::new(),
            }]
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
            transcript: vec![entry("custom", "replacement")],
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
