//! The assistant's cloud model with the user's own key: Claude through Anthropic's
//! Messages API, or any chat completions service such as OpenAI's.
//!
//! The key is not a setting. Settings are shown to the assistant and copied
//! between machines; the key stays in its own file in the Cutlass folder,
//! readable by this user only, and is sent only to the service it is for.
//! Credits for a Cutlass-run service will come with accounts.

#![forbid(unsafe_code)]

use std::path::Path;
use std::time::Duration;

use cutlass_agent::{
    ChatModel, ChatTransport, ClaudeModel, Completion, CompletionRequest, Model, ModelError,
};
use cutlass_settings::{ModelChoice, Settings};

/// How long one answer may take.
const ANSWER_WITHIN: Duration = Duration::from_secs(600);

/// The key kept for `provider` in `file`; empty when there is none.
#[must_use]
pub fn key(file: &Path, provider: &str) -> String {
    std::fs::read_to_string(file)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|keys| {
            keys.get(provider)
                .and_then(|key| key.as_str())
                .map(str::to_owned)
        })
        .unwrap_or_default()
}

/// Keep `key` for `provider` in `file`, or forget it when it is empty. The
/// file is readable by this user only.
pub fn set_key(file: &Path, provider: &str, key: &str) -> Result<(), String> {
    let mut keys = std::fs::read_to_string(file)
        .ok()
        .and_then(|text| {
            serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&text).ok()
        })
        .unwrap_or_default();
    if key.trim().is_empty() {
        keys.remove(provider);
    } else {
        keys.insert(provider.to_owned(), key.trim().into());
    }
    let failed = |error: std::io::Error| format!("the key was not saved: {error}");
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent).map_err(failed)?;
    }
    let text = serde_json::Value::Object(keys).to_string();
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(file)
            .map_err(failed)?;
        // A file made before keeps its mode; narrow it too.
        out.set_permissions(std::fs::Permissions::from_mode(0o600))
            .map_err(failed)?;
        out.write_all(text.as_bytes()).map_err(failed)
    }
    #[cfg(not(unix))]
    {
        std::fs::write(file, text).map_err(failed)
    }
}

/// HTTPS posts to one service with its headers, the key among them.
pub struct CloudHttp {
    pub(crate) http: cutlass_download::Http,
    url: String,
    headers: Vec<(String, String)>,
}

impl ChatTransport for CloudHttp {
    fn post(&mut self, body: &str) -> Result<String, ModelError> {
        let headers: Vec<(&str, &str)> = self
            .headers
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        let posted = self.http.post(&self.url, &headers, body).map_err(|error| {
            ModelError::Inner(format!("the cloud model did not answer: {error}"))
        })?;
        if (200..300).contains(&posted.status) {
            return Ok(posted.body);
        }
        // An error body in JSON is the service's own explanation.
        if serde_json::from_str::<serde_json::Value>(&posted.body)
            .is_ok_and(|value| value.get("error").is_some())
        {
            return Ok(posted.body);
        }
        Err(ModelError::Inner(format!(
            "the cloud model answered {}: {}",
            posted.status,
            posted.body.chars().take(300).collect::<String>()
        )))
    }
}

/// The cloud model, by whose API it speaks.
pub enum Cloud {
    /// Anthropic's Messages API.
    Claude(ClaudeModel<CloudHttp>),
    /// A chat completions service.
    Chat(ChatModel<CloudHttp>),
}

impl Model for Cloud {
    fn complete(&mut self, request: &CompletionRequest) -> Result<Completion, ModelError> {
        match self {
            Self::Claude(model) => model.complete(request),
            Self::Chat(model) => model.complete(request),
        }
    }
}

/// The cloud model `settings` choose, with `key`, told `context` after the
/// system prompt.
pub fn model(settings: &Settings, key: &str, context: &str) -> Result<Cloud, String> {
    let choice = &settings.model;
    if key.trim().is_empty() {
        return Err(
            "The cloud model has no key yet. Paste yours in the assistant's settings.".to_owned(),
        );
    }
    let name = choice.cloud_name.trim();
    if name.is_empty() {
        return Err("Name the cloud model in the assistant's settings.".to_owned());
    }
    let http = cutlass_download::Http::patient(ANSWER_WITHIN);
    let json = ("Content-Type".to_owned(), "application/json".to_owned());
    if choice.cloud_provider == ModelChoice::OPENAI {
        let url = if choice.cloud_url.trim().is_empty() {
            cutlass_settings::OPENAI_CHAT_URL.to_owned()
        } else {
            choice.cloud_url.trim().to_owned()
        };
        if !url.starts_with("https://") {
            return Err(format!(
                "The cloud address must start with https://, not {url}"
            ));
        }
        let transport = CloudHttp {
            http,
            url,
            headers: vec![
                json,
                ("Authorization".to_owned(), format!("Bearer {}", key.trim())),
            ],
        };
        return Ok(Cloud::Chat(
            ChatModel::new(transport, name).with_context(context),
        ));
    }
    let mut headers = vec![
        json,
        ("x-api-key".to_owned(), key.trim().to_owned()),
        (
            "anthropic-version".to_owned(),
            cutlass_agent::ANTHROPIC_VERSION.to_owned(),
        ),
    ];
    if cutlass_agent::wants_fallback(name) {
        headers.push((
            "anthropic-beta".to_owned(),
            cutlass_agent::FALLBACK_BETA.to_owned(),
        ));
    }
    let transport = CloudHttp {
        http,
        url: cutlass_agent::ANTHROPIC_URL.to_owned(),
        headers,
    };
    Ok(Cloud::Claude(
        ClaudeModel::new(transport, name).with_context(context),
    ))
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;

    use cutlass_agent::{ChatTransport, CompletionRequest, Message, Model};

    use super::{CloudHttp, key, set_key};

    #[test]
    fn a_key_is_kept_apart_readable_by_this_user_only_and_forgotten() {
        let dir = std::env::temp_dir().join(format!("cutlass-keys-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let file = dir.join("Assistant").join("keys.json");
        assert_eq!(key(&file, "claude"), "");
        set_key(&file, "claude", " sk-test ").unwrap();
        set_key(&file, "openai", "sk-other").unwrap();
        assert_eq!(key(&file, "claude"), "sk-test");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&file).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        set_key(&file, "claude", "").unwrap();
        assert_eq!(key(&file, "claude"), "");
        assert_eq!(key(&file, "openai"), "sk-other");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_key_or_no_name_is_said_plainly() {
        let settings = cutlass_settings::Settings::default();
        assert!(
            super::model(&settings, "", "")
                .err()
                .unwrap()
                .contains("no key")
        );
        let mut unnamed = settings.clone();
        unnamed.model.cloud_name.clear();
        assert!(
            super::model(&unnamed, "k", "")
                .err()
                .unwrap()
                .contains("Name")
        );
        let mut plain = settings;
        plain.model.cloud_provider = cutlass_settings::ModelChoice::OPENAI.to_owned();
        plain.model.cloud_url = "http://example.com/v1/chat/completions".to_owned();
        assert!(
            super::model(&plain, "k", "")
                .err()
                .unwrap()
                .contains("https://")
        );
    }

    /// A Messages API stand-in on this machine: it checks the key and the
    /// version, then answers.
    #[test]
    fn claude_is_asked_with_the_key_and_its_answer_read() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut head = Vec::new();
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line.trim().is_empty() {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
                head.push(line.trim().to_owned());
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let answer = r#"{"type":"message","stop_reason":"end_turn","content":[{"type":"text","text":"hi from claude"}]}"#;
            let mut stream = stream;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{answer}",
                answer.len()
            )
            .unwrap();
            (head, String::from_utf8(body).unwrap())
        });
        let transport = CloudHttp {
            http: cutlass_download::Http::new(),
            url: format!("http://127.0.0.1:{port}/v1/messages"),
            headers: vec![
                ("x-api-key".to_owned(), "sk-test".to_owned()),
                (
                    "anthropic-version".to_owned(),
                    cutlass_agent::ANTHROPIC_VERSION.to_owned(),
                ),
                ("Content-Type".to_owned(), "application/json".to_owned()),
            ],
        };
        let mut model = cutlass_agent::ClaudeModel::new(transport, "claude-opus-5-5");
        let completion = model
            .complete(&CompletionRequest {
                messages: vec![Message::User {
                    text: "hello".to_owned(),
                }],
                tools: cutlass_agent::tool_definitions(),
            })
            .unwrap();
        assert_eq!(completion.text, "hi from claude");
        let (head, body) = server.join().unwrap();
        assert!(head[0].starts_with("POST /v1/messages"), "{head:?}");
        let has = |name: &str, value: &str| {
            head.iter().any(|line| {
                line.to_ascii_lowercase()
                    .starts_with(&format!("{name}: {}", value.to_ascii_lowercase()))
            })
        };
        assert!(has("x-api-key", "sk-test"), "{head:?}");
        assert!(
            has("anthropic-version", cutlass_agent::ANTHROPIC_VERSION),
            "{head:?}"
        );
        let sent: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(sent["model"], "claude-opus-5-5");
        assert_eq!(sent["messages"][0]["content"], "hello");
        // An error status with the service's JSON reads as its explanation.
        let refused = format!("{}", {
            let mut http = CloudHttp {
                http: cutlass_download::Http::new(),
                url: "http://127.0.0.1:1/v1/messages".to_owned(),
                headers: Vec::new(),
            };
            http.post("{}").unwrap_err()
        });
        assert!(refused.contains("did not answer"), "{refused}");
    }

    /// The real service, with a key it refuses: the request reaches it over
    /// HTTPS and its refusal reads plainly.
    /// `cargo test -p cutlass-agent-cloud real_service -- --ignored`.
    #[test]
    #[ignore = "reaches api.anthropic.com"]
    fn the_real_service_names_a_wrong_key() {
        let settings = cutlass_settings::Settings::default();
        let mut model = super::model(&settings, "sk-ant-not-a-key", "").unwrap();
        let error = model
            .complete(&CompletionRequest {
                messages: vec![Message::User {
                    text: "hi".to_owned(),
                }],
                tools: cutlass_agent::tool_definitions(),
            })
            .unwrap_err()
            .to_string();
        eprintln!("{error}");
        assert!(error.contains("invalid x-api-key"), "{error}");
    }

    /// A real answer from Claude, with the key in `ANTHROPIC_API_KEY`:
    /// `ANTHROPIC_API_KEY=... cargo test -p cutlass-agent-cloud real_answer -- --ignored`.
    /// The model is asked to read the project, so the answer comes back
    /// as a tool call; this checks the request is accepted end to end.
    #[test]
    #[ignore = "needs ANTHROPIC_API_KEY and spends a little"]
    fn a_real_answer_comes_back_as_a_tool_call() {
        let key = std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY");
        let settings = cutlass_settings::Settings::default();
        let mut model = super::model(&settings, &key, "").unwrap();
        let completion = model
            .complete(&CompletionRequest {
                messages: vec![Message::User {
                    text: "How many clips are in the project? Read it first.".to_owned(),
                }],
                tools: cutlass_agent::tool_definitions(),
            })
            .unwrap();
        eprintln!("{completion:?}");
        assert!(
            completion.calls.iter().any(|call| call.name == "draft.query"),
            "{completion:?}"
        );
    }
}
