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

use crate::config::AiSection;
use crate::provider::{
    ChatProvider, ChatRequest, ChatTurn, FinishReason, Message, ProviderError, ToolCall,
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

    /// Build from a parsed `[ai]` config section (resolves `api_key_env`).
    pub fn from_config(config: &AiSection) -> Result<Self, ProviderError> {
        let api_key = config.resolve_api_key().map_err(ProviderError::NotConfigured)?;
        Ok(Self::new(&config.base_url, &config.model, api_key))
    }

    fn request_body(&self, request: &ChatRequest<'_>) -> serde_json::Value {
        let messages: Vec<serde_json::Value> = request.messages.iter().map(to_openai).collect();
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
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
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

        let mut http = self
            .agent
            .post(&url)
            .set("Content-Type", "application/json");
        if let Some(key) = &self.api_key {
            http = http.set("Authorization", &format!("Bearer {key}"));
        }

        let response = match http.send_string(&body) {
            Ok(response) => response,
            Err(ureq::Error::Status(status, response)) => {
                let message = response
                    .into_string()
                    .unwrap_or_else(|_| "<unreadable error body>".to_string());
                return Err(ProviderError::Provider {
                    status,
                    message: truncate(&message, 500),
                });
            }
            Err(ureq::Error::Transport(t)) => {
                return Err(ProviderError::Network(format!("{url}: {t}")));
            }
        };

        consume_sse(response.into_reader(), cancel, on_text)
    }
}

fn to_openai(message: &Message) -> serde_json::Value {
    match message {
        Message::System { content } => serde_json::json!({ "role": "system", "content": content }),
        Message::User { content } => serde_json::json!({ "role": "user", "content": content }),
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
            m
        }
        Message::ToolResult { call_id, content } => serde_json::json!({
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }),
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
            Message::System {
                content: "You edit video timelines.".into(),
            },
            Message::Assistant {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "remove_clip".into(),
                    arguments: serde_json::json!({"clip": 3}),
                }],
            },
            Message::ToolResult {
                call_id: "call_1".into(),
                content: "removed clip 3".into(),
            },
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
        assert_eq!(body["tools"].as_array().unwrap().len(), 20);
        assert_eq!(body["tools"][0]["function"]["name"], "add_track");
    }
}
