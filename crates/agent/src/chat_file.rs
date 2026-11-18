//! A conversation kept on disk between sessions: the model's messages as
//! JSON, written whole each turn.

use std::path::Path;

use crate::model::Message;

/// Why a kept conversation was not handed back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatMiss {
    /// The file does not read as a saved conversation.
    Unread(String),
    /// The file is the conversation of another project: one that was at
    /// the same place before the project asking now.
    Another,
}

impl std::fmt::Display for ChatMiss {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unread(why) => formatter.write_str(why),
            Self::Another => formatter.write_str("the conversation is another project's"),
        }
    }
}

/// A conversation [`save_chat`] wrote; empty when there is none. A file
/// that does not read is an error, for the host to log.
pub fn load_chat(file: &Path) -> Result<Vec<Message>, String> {
    load_chat_of(file, "").map_err(|miss| miss.to_string())
}

/// [`load_chat`], for the project that calls itself `project`. A host
/// names a conversation's file by where the project is, and a new project
/// made where a deleted one was is at the same place: a conversation
/// [`save_chat_of`] kept under another name is not this project's, and is
/// not handed to it, or to its model. One kept with no name is taken, as
/// every conversation was before they were named. An empty `project` asks
/// for whatever is there.
pub fn load_chat_of(file: &Path, project: &str) -> Result<Vec<Message>, ChatMiss> {
    let text = match std::fs::read_to_string(file) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(ChatMiss::Unread(format!(
                "{} does not read: {error}",
                file.display()
            )));
        }
    };
    let unread = || ChatMiss::Unread(format!("{} is not a saved conversation", file.display()));
    let saved: serde_json::Value = serde_json::from_str(&text).map_err(|_| unread())?;
    let kept_for = saved
        .get("project")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if !project.is_empty() && !kept_for.is_empty() && kept_for != project {
        return Err(ChatMiss::Another);
    }
    saved
        .get("messages")
        .and_then(|messages| serde_json::from_value(messages.clone()).ok())
        .ok_or_else(unread)
}

/// Write `messages` to `file`, or remove it when there are none.
pub fn save_chat(file: &Path, messages: &[Message]) -> std::io::Result<()> {
    save_chat_of(file, "", messages)
}

/// [`save_chat`], kept as the conversation of the project that calls
/// itself `project`, for [`load_chat_of`] to tell from another's.
pub fn save_chat_of(file: &Path, project: &str, messages: &[Message]) -> std::io::Result<()> {
    if messages.is_empty() {
        return match std::fs::remove_file(file) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
            _ => Ok(()),
        };
    }
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut saved = serde_json::json!({ "version": 1, "messages": messages });
    if !project.is_empty() {
        saved["project"] = serde_json::json!(project);
    }
    let temporary = file.with_extension("json.part");
    std::fs::write(&temporary, saved.to_string())?;
    std::fs::rename(&temporary, file)
}

#[cfg(test)]
mod tests {
    use super::{load_chat, save_chat};
    use crate::model::{Message, ToolCall};

    #[test]
    fn a_conversation_comes_back_as_it_was_and_an_empty_one_is_removed() {
        let dir = std::env::temp_dir().join(format!("cutlass-chat-{}", std::process::id()));
        let file = dir.join("chats").join("a.json");
        assert_eq!(load_chat(&file).unwrap(), Vec::new());
        let messages = vec![
            Message::User {
                text: "hi".to_owned(),
            },
            Message::Assistant {
                text: String::new(),
                calls: vec![ToolCall {
                    id: "1".to_owned(),
                    name: "draft.query".to_owned(),
                    arguments: "project".to_owned(),
                }],
            },
            Message::Tool {
                call_id: "1".to_owned(),
                name: "draft.query".to_owned(),
                content: "revision=0".to_owned(),
                is_error: false,
                picture: None,
            },
        ];
        save_chat(&file, &messages).unwrap();
        assert_eq!(load_chat(&file).unwrap(), messages);
        std::fs::write(&file, "not json").unwrap();
        assert!(load_chat(&file).is_err());
        save_chat(&file, &[]).unwrap();
        assert!(!file.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_conversation_is_handed_only_to_the_project_it_was_kept_for() {
        use super::{ChatMiss, load_chat_of, save_chat_of};
        let dir = std::env::temp_dir().join(format!("cutlass-chat-of-{}", std::process::id()));
        let file = dir.join("chats").join("a.json");
        let messages = vec![Message::User {
            text: "about the first project".to_owned(),
        }];
        save_chat_of(&file, "project-2@1700", &messages).unwrap();
        // Its own project reads it, and so does a caller that names none.
        assert_eq!(load_chat_of(&file, "project-2@1700").unwrap(), messages);
        assert_eq!(load_chat(&file).unwrap(), messages);
        // A project made later at the same place does not.
        assert_eq!(
            load_chat_of(&file, "project-2@1900"),
            Err(ChatMiss::Another)
        );
        // One kept before conversations were named is taken by whoever is
        // there, and is the taker's once it is kept again.
        save_chat(&file, &messages).unwrap();
        assert_eq!(load_chat_of(&file, "project-2@1900").unwrap(), messages);
        save_chat_of(&file, "project-2@1900", &messages).unwrap();
        assert_eq!(
            load_chat_of(&file, "project-2@1700"),
            Err(ChatMiss::Another)
        );
        // A file that does not read is told apart from another's.
        std::fs::write(&file, "not json").unwrap();
        assert!(matches!(
            load_chat_of(&file, "project-2@1900"),
            Err(ChatMiss::Unread(_))
        ));
        // None there is an empty conversation for anyone.
        std::fs::remove_file(&file).unwrap();
        assert_eq!(load_chat_of(&file, "project-2@1900").unwrap(), Vec::new());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
