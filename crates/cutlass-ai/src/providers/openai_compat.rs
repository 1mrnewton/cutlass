//! The generic OpenAI-compatible chat provider.
//!
//! One implementation, many backends: Ollama (`http://localhost:11434/v1`),
//! llama.cpp-server, LM Studio, OpenAI itself, and OpenAI-compatible
//! gateways — "cloud providers later" is config, not code. Speaks
//! `POST {base_url}/chat/completions` with `stream: true` and parses the
//! SSE chunk stream; tool-call argument fragments are accumulated per
//! index and assembled into whole [`ToolCall`]s.

use std::io::{BufRead, BufReader, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use base64::Engine as _;

use crate::provider::{
    ChatProvider, ChatRequest, ChatTurn, FinishReason, ImagePart, Message, ProviderError, ToolCall,
};

pub struct OpenAiCompatProvider {
    base_url: String,
    model: String,
    api_key: Option<String>,
    agent: ureq::Agent,
}

impl OpenAiCompatProvider {
    pub fn new(base_url: &str, model: &str, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(10))
                .build(),
        }
    }

    /// Liveness probe for the Settings dialog: `GET {base_url}/models`, the
    /// OpenAI-compatible health endpoint (Ollama/LM Studio/OpenAI all serve
    /// it). Returns a short human summary on success; spends no tokens.
    pub fn test_connection(&self) -> Result<String, ProviderError> {
        let url = format!("{}/models", self.base_url);
        let mut http = self.agent.get(&url);
        if let Some(key) = &self.api_key {
            http = http.set("Authorization", &format!("Bearer {key}"));
        }
        match http.call() {
            Ok(response) => {
                let body = response
                    .into_string()
                    .map_err(|e| ProviderError::Network(format!("reading /models: {e}")))?;
                let parsed: serde_json::Value = serde_json::from_str(&body)
                    .map_err(|e| ProviderError::Protocol(format!("bad /models response: {e}")))?;
                let count = parsed["data"].as_array().map(|a| a.len()).unwrap_or(0);
                Ok(match count {
                    0 => "Connected.".to_string(),
                    1 => "Connected · 1 model available.".to_string(),
                    n => format!("Connected · {n} models available."),
                })
            }
            Err(ureq::Error::Status(status, response)) => {
                let message = response
                    .into_string()
                    .unwrap_or_else(|_| "<unreadable error body>".to_string());
                Err(ProviderError::Provider {
                    status,
                    message: truncate(&message, 200),
                })
            }
            Err(ureq::Error::Transport(t)) => Err(ProviderError::Network(format!("{url}: {t}"))),
        }
    }

    fn request_body(&self, request: &ChatRequest<'_>) -> serde_json::Value {
        let messages = to_openai_messages(request.messages);
        let mut body = serde_json::json!({
            "model": self.model,
            "stream": true,
            "messages": messages,
        });
        if !request.tools.is_empty() {
            body["tools"] = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": &t.name,
                            "description": &t.description,
                            "parameters": &t.parameters,
                        },
                    })
                })
                .collect();
        }
        body
    }
}

impl ChatProvider for OpenAiCompatProvider {
    fn chat(
        &self,
        request: &ChatRequest<'_>,
        cancel: &AtomicBool,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<ChatTurn, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = self.request_body(request).to_string();

        // Only the initial send retries: no stream bytes have been
        // consumed yet, so a retry can't duplicate text in the UI.
        // Mid-stream failures (in consume_sse) always surface.
        let mut attempt = 0usize;
        let response = loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(ProviderError::Cancelled);
            }
            let mut http = self
                .agent
                .post(&url)
                .set("Content-Type", "application/json");
            if let Some(key) = &self.api_key {
                http = http.set("Authorization", &format!("Bearer {key}"));
            }
            match http.send_string(&body) {
                Ok(response) => break response,
                Err(ureq::Error::Status(status, response)) => {
                    if retryable_status(status) {
                        match retry_delay(attempt) {
                            Some(delay) if sleep_unless_cancelled(delay, cancel) => {
                                attempt += 1;
                                continue;
                            }
                            Some(_) => return Err(ProviderError::Cancelled),
                            None => {}
                        }
                    }
                    let message = response
                        .into_string()
                        .unwrap_or_else(|_| "<unreadable error body>".to_string());
                    return Err(ProviderError::Provider {
                        status,
                        message: truncate(&message, 500),
                    });
                }
                Err(ureq::Error::Transport(t)) => match retry_delay(attempt) {
                    Some(delay) if sleep_unless_cancelled(delay, cancel) => attempt += 1,
                    Some(_) => return Err(ProviderError::Cancelled),
                    None => return Err(ProviderError::Network(format!("{url}: {t}"))),
                },
            }
        };

        consume_sse(response.into_reader(), cancel, on_text)
    }
}

/// Convert a complete history while preserving OpenAI's parallel-tool-call
/// ordering rule: every `role=tool` response for an assistant turn must
/// precede the next user message. Image attachments are therefore hoisted
/// only after the entire contiguous run of tool results, not immediately
/// after whichever tool happened to return the first image.
fn to_openai_messages(messages: &[Message]) -> Vec<serde_json::Value> {
    let mut wire = Vec::new();
    let mut index = 0usize;
    while index < messages.len() {
        if !matches!(messages[index], Message::ToolResult { .. }) {
            wire.extend(to_openai(&messages[index]));
            index += 1;
            continue;
        }

        let run_start = index;
        while index < messages.len() && matches!(messages[index], Message::ToolResult { .. }) {
            let Message::ToolResult {
                call_id, content, ..
            } = &messages[index]
            else {
                unreachable!("tool-result run checked above");
            };
            wire.push(serde_json::json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            }));
            index += 1;
        }

        let image_results: Vec<(&str, &[ImagePart])> = messages[run_start..index]
            .iter()
            .filter_map(|message| match message {
                Message::ToolResult {
                    call_id, images, ..
                } if !images.is_empty() => Some((call_id.as_str(), images.as_slice())),
                _ => None,
            })
            .collect();
        if !image_results.is_empty() {
            let labels = image_results
                .iter()
                .map(|(call_id, images)| {
                    format!(
                        "{call_id}: {}",
                        images
                            .iter()
                            .map(|image| image.label.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            let mut parts = vec![serde_json::json!({
                "type": "text",
                "text": format!("[tool attachments: {labels}]"),
            })];
            parts.extend(
                image_results
                    .iter()
                    .flat_map(|(_, images)| images.iter())
                    .map(image_url_part),
            );
            wire.push(serde_json::json!({ "role": "user", "content": parts }));
        }
    }
    wire
}

/// Statuses whose response is explicitly temporary. Authentication,
/// malformed requests, and payment failures stay single-shot; retrying
/// them only delays the actionable error.
fn retryable_status(status: u16) -> bool {
    status == 408 || status == 429 || (500..=599).contains(&status)
}

/// Backoff before re-sending the initial request after a transport
/// failure: two retries, spaced so a briefly-napping local server
/// (Ollama model load, sleep wake) gets a second chance without turning
/// a dead endpoint into a long hang.
fn retry_delay(attempt: usize) -> Option<Duration> {
    match attempt {
        0 => Some(Duration::from_millis(300)),
        1 => Some(Duration::from_millis(900)),
        _ => None,
    }
}

/// Sleep in ~50ms slices, polling `cancel`. Returns false — retry
/// abandoned — the moment cancellation shows up.
fn sleep_unless_cancelled(total: Duration, cancel: &AtomicBool) -> bool {
    let slice = Duration::from_millis(50);
    let mut remaining = total;
    while !remaining.is_zero() {
        if cancel.load(Ordering::Relaxed) {
            return false;
        }
        let step = remaining.min(slice);
        std::thread::sleep(step);
        remaining = remaining.saturating_sub(step);
    }
    !cancel.load(Ordering::Relaxed)
}

/// The `image_url` content part: raw bytes become a base64 data URL here,
/// at the wire boundary, and nowhere earlier.
fn image_url_part(image: &ImagePart) -> serde_json::Value {
    let b64 = base64::engine::general_purpose::STANDARD.encode(image.data.as_slice());
    serde_json::json!({
        "type": "image_url",
        "image_url": { "url": format!("data:{};base64,{b64}", image.media_type) },
    })
}

/// One [`Message`] can map to multiple wire messages (a tool result with
/// images), so this returns a `Vec`. Image-free messages keep plain string
/// content — not a one-element parts array — so local models/servers that
/// don't understand arrays keep working.
fn to_openai(message: &Message) -> Vec<serde_json::Value> {
    match message {
        Message::System { content } => {
            vec![serde_json::json!({ "role": "system", "content": content })]
        }
        Message::User { content, images } => {
            if images.is_empty() {
                return vec![serde_json::json!({ "role": "user", "content": content })];
            }
            let mut parts = vec![serde_json::json!({ "type": "text", "text": content })];
            parts.extend(images.iter().map(image_url_part));
            vec![serde_json::json!({ "role": "user", "content": parts })]
        }
        Message::Assistant {
            content,
            tool_calls,
        } => {
            let mut m = serde_json::json!({ "role": "assistant", "content": content });
            if !tool_calls.is_empty() {
                m["tool_calls"] = tool_calls
                    .iter()
                    .map(|c| {
                        serde_json::json!({
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": c.arguments.to_string(),
                            },
                        })
                    })
                    .collect();
            }
            vec![m]
        }
        Message::ToolResult {
            call_id,
            content,
            images,
        } => {
            let mut wire = vec![serde_json::json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            })];
            // The OpenAI tool role only carries strings; hoisting images
            // into an adjacent user message is the interoperable pattern.
            if !images.is_empty() {
                let labels: Vec<&str> = images.iter().map(|i| i.label.as_str()).collect();
                let mut parts = vec![serde_json::json!({
                    "type": "text",
                    "text": format!("[attached: {}]", labels.join(", ")),
                })];
                parts.extend(images.iter().map(image_url_part));
                wire.push(serde_json::json!({ "role": "user", "content": parts }));
            }
            wire
        }
    }
}

/// A tool call being assembled from streamed fragments.
#[derive(Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

/// Parse an OpenAI-style SSE stream into one completed turn, forwarding
/// text deltas as they arrive. Factored over `Read` so fixtures can drive
/// it in tests.
pub(crate) fn consume_sse(
    reader: impl Read,
    cancel: &AtomicBool,
    on_text: &mut dyn FnMut(&str),
) -> Result<ChatTurn, ProviderError> {
    let mut text = String::new();
    let mut calls: Vec<PartialCall> = Vec::new();
    let mut finish = FinishReason::Other;

    for line in BufReader::new(reader).lines() {
        if cancel.load(Ordering::Relaxed) {
            return Err(ProviderError::Cancelled);
        }
        let line = line.map_err(|e| ProviderError::Network(format!("stream read failed: {e}")))?;
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue; // comments, event names, keep-alive blank lines
        };
        if data == "[DONE]" {
            break;
        }
        let chunk: serde_json::Value = serde_json::from_str(data)
            .map_err(|e| ProviderError::Protocol(format!("bad SSE chunk: {e}: {data}")))?;
        let Some(choice) = chunk["choices"].get(0) else {
            continue; // e.g. usage-only chunks
        };

        if let Some(reason) = choice["finish_reason"].as_str() {
            finish = match reason {
                "stop" => FinishReason::Stop,
                "tool_calls" => FinishReason::ToolCalls,
                "length" => FinishReason::Length,
                _ => FinishReason::Other,
            };
        }

        let delta = &choice["delta"];
        if let Some(piece) = delta["content"].as_str() {
            if !piece.is_empty() {
                text.push_str(piece);
                on_text(piece);
            }
        }
        if let Some(fragments) = delta["tool_calls"].as_array() {
            for fragment in fragments {
                let index = fragment["index"].as_u64().unwrap_or(0) as usize;
                if calls.len() <= index {
                    calls.resize_with(index + 1, PartialCall::default);
                }
                let call = &mut calls[index];
                if let Some(id) = fragment["id"].as_str() {
                    call.id.push_str(id);
                }
                if let Some(name) = fragment["function"]["name"].as_str() {
                    call.name.push_str(name);
                }
                if let Some(args) = fragment["function"]["arguments"].as_str() {
                    call.arguments.push_str(args);
                }
            }
        }
    }

    let tool_calls = calls
        .into_iter()
        .map(|c| {
            let arguments = if c.arguments.trim().is_empty() {
                serde_json::json!({})
            } else {
                serde_json::from_str(&c.arguments).map_err(|e| {
                    ProviderError::Protocol(format!(
                        "tool call '{}' has unparseable arguments: {e}: {}",
                        c.name, c.arguments
                    ))
                })?
            };
            Ok(ToolCall {
                id: c.id,
                name: c.name,
                arguments,
            })
        })
        .collect::<Result<Vec<_>, ProviderError>>()?;

    if finish == FinishReason::Other && !tool_calls.is_empty() {
        finish = FinishReason::ToolCalls;
    }
    Ok(ChatTurn {
        text,
        tool_calls,
        finish,
    })
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(fixture: &str) -> (ChatTurn, String) {
        let cancel = AtomicBool::new(false);
        let mut streamed = String::new();
        let turn = consume_sse(fixture.as_bytes(), &cancel, &mut |t| streamed.push_str(t))
            .expect("fixture parses");
        (turn, streamed)
    }

    #[test]
    fn text_only_stream() {
        let fixture = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"The timeline \"}}]}\n",
            "\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"is 12s long.\"}}]}\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n",
            "data: [DONE]\n",
        );
        let (turn, streamed) = run(fixture);
        assert_eq!(turn.text, "The timeline is 12s long.");
        assert_eq!(streamed, turn.text);
        assert_eq!(turn.finish, FinishReason::Stop);
        assert!(turn.tool_calls.is_empty());
    }

    #[test]
    fn tool_call_assembles_from_fragments() {
        let fixture = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"trim_clip\",\"arguments\":\"\"}}]}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"clip\\\": 12,\"}}]}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\" \\\"start\\\": 14.0, \\\"duration\\\": 4.0}\"}}]}}]}\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n",
            "data: [DONE]\n",
        );
        let (turn, _) = run(fixture);
        assert_eq!(turn.finish, FinishReason::ToolCalls);
        assert_eq!(turn.tool_calls.len(), 1);
        let call = &turn.tool_calls[0];
        assert_eq!(call.id, "call_1");
        assert_eq!(call.name, "trim_clip");
        assert_eq!(
            call.arguments,
            serde_json::json!({ "clip": 12, "start": 14.0, "duration": 4.0 })
        );
    }

    #[test]
    fn parallel_tool_calls_keep_their_indices() {
        let fixture = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"remove_clip\",\"arguments\":\"{\\\"clip\\\":1}\"}}]}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"remove_clip\",\"arguments\":\"{\\\"clip\\\":2}\"}}]}}]}\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n",
            "data: [DONE]\n",
        );
        let (turn, _) = run(fixture);
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.tool_calls[0].arguments["clip"], 1);
        assert_eq!(turn.tool_calls[1].arguments["clip"], 2);
    }

    #[test]
    fn cancellation_stops_the_stream() {
        let cancel = AtomicBool::new(true);
        let err = consume_sse(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n".as_bytes(),
            &cancel,
            &mut |_| {},
        )
        .unwrap_err();
        assert!(matches!(err, ProviderError::Cancelled));
    }

    #[test]
    fn malformed_chunks_and_arguments_are_protocol_errors() {
        let cancel = AtomicBool::new(false);
        let err = consume_sse("data: {not json}\n".as_bytes(), &cancel, &mut |_| {}).unwrap_err();
        assert!(matches!(err, ProviderError::Protocol(_)));

        let fixture = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"x\",\"function\":{\"name\":\"trim_clip\",\"arguments\":\"{oops\"}}]}}]}\n",
            "data: [DONE]\n",
        );
        let err = consume_sse(fixture.as_bytes(), &cancel, &mut |_| {}).unwrap_err();
        match err {
            ProviderError::Protocol(msg) => assert!(msg.contains("trim_clip"), "{msg}"),
            other => panic!("expected protocol error, got {other:?}"),
        }
    }

    #[test]
    fn request_body_includes_tools_and_messages() {
        let provider = OpenAiCompatProvider::new("http://localhost:11434/v1/", "qwen3", None);
        assert_eq!(provider.base_url, "http://localhost:11434/v1");

        let messages = vec![
            Message::system("You edit video timelines."),
            Message::Assistant {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "remove_clip".into(),
                    arguments: serde_json::json!({"clip": 3}),
                }],
            },
            Message::tool_result("call_1", "removed clip 3"),
        ];
        let tools = crate::wire::tool_specs();
        let body = provider.request_body(&ChatRequest {
            messages: &messages,
            tools: &tools,
        });

        assert_eq!(body["model"], "qwen3");
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["arguments"],
            "{\"clip\":3}"
        );
        assert_eq!(body["messages"][2]["role"], "tool");
        assert_eq!(body["messages"][2]["tool_call_id"], "call_1");
        assert_eq!(body["tools"].as_array().unwrap().len(), 45);
        assert_eq!(body["tools"][0]["function"]["name"], "add_track");
    }

    #[test]
    fn user_message_with_image_becomes_content_parts() {
        let bytes = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a];
        let message = Message::User {
            content: "what's on the timeline?".into(),
            images: vec![ImagePart::png(bytes.clone(), "timeline at 12.40s")],
        };
        let wire = to_openai(&message);
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0]["role"], "user");
        let parts = wire[0]["content"].as_array().expect("content parts array");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "what's on the timeline?");
        assert_eq!(parts[1]["type"], "image_url");
        let url = parts[1]["image_url"]["url"].as_str().unwrap();
        let b64 = url
            .strip_prefix("data:image/png;base64,")
            .expect("png data URL prefix");
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .expect("valid base64");
        assert_eq!(decoded, bytes, "round-trips the original bytes");
    }

    #[test]
    fn tool_result_images_hoist_into_a_synthetic_user_message() {
        let with_images = Message::ToolResult {
            call_id: "call_1".into(),
            content: "screenshot taken".into(),
            images: vec![ImagePart::jpeg(vec![1, 2, 3], "preview at 3.00s")],
        };
        let wire = to_openai(&with_images);
        assert_eq!(wire.len(), 2);
        assert_eq!(wire[0]["role"], "tool");
        assert_eq!(wire[0]["tool_call_id"], "call_1");
        assert!(
            wire[0]["content"].is_string(),
            "the tool role carries only the text content"
        );
        assert_eq!(wire[0]["content"], "screenshot taken");
        assert_eq!(wire[1]["role"], "user");
        let parts = wire[1]["content"].as_array().expect("content parts array");
        assert_eq!(parts[0]["type"], "text");
        assert!(
            parts[0]["text"]
                .as_str()
                .unwrap()
                .contains("preview at 3.00s"),
            "{}",
            parts[0]["text"]
        );
        assert_eq!(parts[1]["type"], "image_url");
        assert!(
            parts[1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/jpeg;base64,")
        );

        let without = Message::tool_result("call_2", "removed clip 3");
        assert_eq!(to_openai(&without).len(), 1);
    }

    #[test]
    fn parallel_tool_results_all_precede_hoisted_images() {
        let messages = vec![
            Message::Assistant {
                content: String::new(),
                tool_calls: vec![
                    ToolCall {
                        id: "call_a".into(),
                        name: "media_preview_frame".into(),
                        arguments: serde_json::json!({}),
                    },
                    ToolCall {
                        id: "call_b".into(),
                        name: "describe_project".into(),
                        arguments: serde_json::json!({}),
                    },
                ],
            },
            Message::ToolResult {
                call_id: "call_a".into(),
                content: "frame ready".into(),
                images: vec![ImagePart::png(vec![1, 2], "preview frame")],
            },
            Message::tool_result("call_b", "project summary"),
        ];

        let wire = to_openai_messages(&messages);
        let roles: Vec<_> = wire
            .iter()
            .map(|message| message["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["assistant", "tool", "tool", "user"]);
        assert_eq!(wire[1]["tool_call_id"], "call_a");
        assert_eq!(wire[2]["tool_call_id"], "call_b");
        assert!(
            wire[3]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("call_a: preview frame")
        );
        assert_eq!(wire[3]["content"][1]["type"], "image_url");
    }

    #[test]
    fn user_message_without_images_keeps_plain_string_content() {
        let wire = to_openai(&Message::user("split the clip"));
        assert_eq!(wire.len(), 1);
        assert!(
            wire[0]["content"].is_string(),
            "plain string, not a parts array, so array-less servers keep working"
        );
        assert_eq!(wire[0]["content"], "split the clip");
    }

    #[test]
    fn retry_backoff_is_two_attempts_then_give_up() {
        assert_eq!(retry_delay(0), Some(Duration::from_millis(300)));
        assert_eq!(retry_delay(1), Some(Duration::from_millis(900)));
        assert_eq!(retry_delay(2), None);
        assert_eq!(retry_delay(99), None);
    }

    #[test]
    fn only_transient_http_statuses_retry() {
        for status in [408, 429, 500, 502, 599] {
            assert!(retryable_status(status), "{status}");
        }
        for status in [400, 401, 402, 403, 404, 422, 600] {
            assert!(!retryable_status(status), "{status}");
        }
    }

    #[test]
    fn cancellation_shortcuts_the_retry_sleep() {
        let cancel = AtomicBool::new(true);
        let start = std::time::Instant::now();
        assert!(!sleep_unless_cancelled(Duration::from_millis(900), &cancel));
        assert!(
            start.elapsed() < Duration::from_millis(300),
            "a raised cancel flag must not wait out the backoff"
        );

        let live = AtomicBool::new(false);
        assert!(sleep_unless_cancelled(Duration::from_millis(10), &live));
    }
}
