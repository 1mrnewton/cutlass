//! A conversation kept on disk between sessions: the model's messages as
//! JSON, written whole each turn.

use std::path::Path;

use crate::model::Message;

/// A conversation [`save_chat`] wrote; empty when there is none. A file
/// that does not read is an error, for the host to log.
pub fn load_chat(file: &Path) -> Result<Vec<Message>, String> {
    let text = match std::fs::read_to_string(file) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("{} does not read: {error}", file.display())),
    };
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|saved| serde_json::from_value(saved.get("messages")?.clone()).ok())
        .ok_or_else(|| format!("{} is not a saved conversation", file.display()))
}

/// Write `messages` to `file`, or remove it when there are none.
pub fn save_chat(file: &Path, messages: &[Message]) -> std::io::Result<()> {
    if messages.is_empty() {
        return match std::fs::remove_file(file) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
            _ => Ok(()),
        };
    }
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let saved = serde_json::json!({ "version": 1, "messages": messages });
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
}
