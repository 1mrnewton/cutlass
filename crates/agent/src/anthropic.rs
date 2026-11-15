//! Claude through the Messages API (`POST /v1/messages`), for a cloud
//! model the user brings their own key to.
//!
//! The host's transport adds the address and the key; this module builds
//! the body and reads the answer. Tool names go out with `_` for `.`, as
//! for MCP, since the API takes only letters, digits, `_` and `-`.
//!
//! Thinking is always on for the current models. The API binds each
//! thinking block to the conversation before it, so this runner keeps the
//! content blocks of every reply it got exactly as they came and sends
//! them back unchanged. A reply from an earlier runner (an earlier turn,
//! or another model) is sent as its text and tool calls only, which drops
//! the oldest thinking blocks: a leading run, which the API allows.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::chat::ChatTransport;
use crate::mcp_server::{input_schema, registry_name, takes_object};
use crate::model::{Completion, CompletionRequest, Message, Model, ModelError, ToolCall};

/// The API version this runner speaks.
pub const ANTHROPIC_VERSION: &str = "2026-10-21";

/// The Messages API address.
pub const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";

/// The model a new cloud setting starts with.
pub const DEFAULT_CLAUDE_MODEL: &str = "claude-opus-5-5";

/// The beta that lets the API answer a declined request with another
/// model inside the same call.
pub const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

/// Models that take `fallbacks: "default"`.
const FALLBACK_MODELS: &[&str] = &[
    "claude-fable-5-2",
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
];

/// Whether requests for `model` should carry [`FALLBACK_BETA`].
#[must_use]
pub fn wants_fallback(model: &str) -> bool {
    FALLBACK_MODELS.contains(&model)
}

/// Room for one reply. A larger reply would need streaming.
const MAX_TOKENS: u32 = 16_000;

/// A Claude model behind `transport`.
#[derive(Debug)]
pub struct ClaudeModel<T> {
    transport: T,
    model: String,
    system: String,
    /// Each reply's content blocks as the API sent them, by the reply's
    /// text and tool call ids.
    replies: BTreeMap<String, Value>,
}

impl<T: ChatTransport> ClaudeModel<T> {
    /// Talks to `model` through `transport`, with the harness's system
    /// prompt.
    #[must_use]
    pub fn new(transport: T, model: &str) -> Self {
        Self {
            transport,
            model: model.to_owned(),
            system: crate::chat::SYSTEM_PROMPT.to_owned(),
            replies: BTreeMap::new(),
        }
    }

    /// Adds `context` after the system prompt.
    #[must_use]
    pub fn with_context(mut self, context: &str) -> Self {
        if !context.trim().is_empty() {
            self.system.push_str("\n\n");
            self.system.push_str(context.trim());
        }
        self
    }

    /// The transport, to look at in tests.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// The request body for `request`.
    #[must_use]
    pub fn body(&self, request: &CompletionRequest) -> Value {
        let mut messages: Vec<Value> = Vec::new();
        let mut results: Vec<Value> = Vec::new();
        let flush = |messages: &mut Vec<Value>, results: &mut Vec<Value>| {
            if !results.is_empty() {
                messages.push(json!({ "role": "user", "content": std::mem::take(results) }));
            }
        };
        for message in &request.messages {
            match message {
                Message::User { text } => {
                    flush(&mut messages, &mut results);
                    messages.push(json!({ "role": "user", "content": text }));
                }
                Message::Assistant { text, calls } => {
                    flush(&mut messages, &mut results);
                    let content = self
                        .replies
                        .get(&reply_key(text, calls))
                        .cloned()
                        .unwrap_or_else(|| plain_content(text, calls));
                    messages.push(json!({ "role": "assistant", "content": content }));
                }
                Message::Tool {
                    call_id,
                    content,
                    is_error,
                    ..
                } => results.push(json!({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": content,
                    "is_error": is_error,
                })),
            }
        }
        flush(&mut messages, &mut results);
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.name.replace('.', "_"),
                    "description": tool.description,
                    "input_schema": input_schema(&tool.name),
                })
            })
            .collect();
        let mut body = json!({
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": self.system,
            "messages": messages,
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        if wants_fallback(&self.model) {
            body["fallbacks"] = json!("default");
        }
        body
    }
}

impl<T: ChatTransport> Model for ClaudeModel<T> {
    fn complete(&mut self, request: &CompletionRequest) -> Result<Completion, ModelError> {
        let body = self.body(request).to_string();
        let reply = self.transport.post(&body)?;
        let (completion, content) = parse_claude_reply(&reply)?;
        self.replies
            .insert(reply_key(&completion.text, &completion.calls), content);
        Ok(completion)
    }
}

/// A reply's key: its text and its tool call ids.
fn reply_key(text: &str, calls: &[ToolCall]) -> String {
    let ids: Vec<&str> = calls.iter().map(|call| call.id.as_str()).collect();
    format!("{text}\u{0}{}", ids.join("\u{0}"))
}

/// The text and tool calls of a reply this runner did not receive.
fn plain_content(text: &str, calls: &[ToolCall]) -> Value {
    let mut content = Vec::new();
    if !text.is_empty() {
        content.push(json!({ "type": "text", "text": text }));
    }
    for call in calls {
        content.push(json!({
            "type": "tool_use",
            "id": call.id,
            "name": call.name.replace('.', "_"),
            "input": tool_input(&call.name, &call.arguments),
        }));
    }
    if content.is_empty() {
        content.push(json!({ "type": "text", "text": "(no answer)" }));
    }
    Value::Array(content)
}

/// A call's arguments as the tool's input object.
fn tool_input(tool: &str, arguments: &str) -> Value {
    if takes_object(tool) {
        return serde_json::from_str::<Value>(arguments)
            .ok()
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({}));
    }
    if arguments.is_empty() {
        json!({})
    } else {
        json!({ "input": arguments })
    }
}

/// The completion in a Messages API reply, and the reply's content blocks
/// to send back unchanged.
pub fn parse_claude_reply(reply: &str) -> Result<(Completion, Value), ModelError> {
    let value: Value = serde_json::from_str(reply)
        .map_err(|error| ModelError::Inner(format!("Claude's reply is not JSON: {error}")))?;
    if value.get("type").and_then(Value::as_str) == Some("error") || value.get("error").is_some() {
        let error = value.get("error").cloned().unwrap_or(Value::Null);
        let kind = error.get("type").and_then(Value::as_str).unwrap_or("error");
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .map_or_else(|| error.to_string(), str::to_owned);
        let hint = match kind {
            "authentication_error" => " Check the key in Settings → Assistant.",
            "permission_error" => " The key may not have access to that model.",
            "not_found_error" => " Check the model name in Settings → Assistant.",
            "rate_limit_error" | "overloaded_error" => " Wait a moment and ask again.",
            _ => "",
        };
        return Err(ModelError::Inner(format!(
            "Claude answered {kind}: {message}.{hint}"
        )));
    }
    let stop = value
        .get("stop_reason")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if stop == "refusal" {
        let why = value
            .pointer("/stop_details/explanation")
            .and_then(Value::as_str)
            .unwrap_or("no reason given");
        return Err(ModelError::Inner(format!(
            "Claude declined this request: {why}"
        )));
    }
    let content = value.get("content").cloned().unwrap_or_else(|| json!([]));
    let mut text = Vec::new();
    let mut calls = Vec::new();
    for block in content.as_array().into_iter().flatten() {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(part) = block.get("text").and_then(Value::as_str) {
                    text.push(part.to_owned());
                }
            }
            Some("tool_use") => {
                let wire = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let name = registry_name(wire).map_or_else(|| wire.to_owned(), str::to_owned);
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                let arguments = if takes_object(&name) {
                    input.to_string()
                } else {
                    match input.get("input") {
                        Some(Value::String(text)) => text.clone(),
                        Some(other) => other.to_string(),
                        None => String::new(),
                    }
                };
                calls.push(ToolCall {
                    id: block
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    name,
                    arguments,
                });
            }
            // Thinking and anything newer stays in `content` for replay.
            _ => {}
        }
    }
    let mut text = text.join("\n").trim().to_owned();
    if stop == "max_tokens" && calls.is_empty() {
        text.push_str("\n\n(The answer was cut off at its length limit.)");
    }
    Ok((
        Completion {
            text,
            calls,
            proposed: Vec::new(),
        },
        content,
    ))
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{ClaudeModel, parse_claude_reply, wants_fallback};
    use crate::chat::ChatTransport;
    use crate::model::{CompletionRequest, Message, Model, ModelError, ToolDefinition};

    struct Replies {
        sent: Vec<Value>,
        replies: Vec<String>,
    }

    impl ChatTransport for Replies {
        fn post(&mut self, body: &str) -> Result<String, ModelError> {
            self.sent.push(serde_json::from_str(body).unwrap());
            Ok(self.replies.remove(0))
        }
    }

    fn tools() -> Vec<ToolDefinition> {
        crate::tools::tool_definitions()
    }

    #[test]
    fn a_tool_round_sends_back_the_reply_unchanged_and_the_results_together() {
        let first = json!({
            "type": "message",
            "stop_reason": "tool_use",
            "content": [
                { "type": "thinking", "thinking": "", "signature": "sig-1" },
                { "type": "text", "text": "Reading." },
                { "type": "tool_use", "id": "t1", "name": "draft_query", "input": { "input": "project" } },
                { "type": "tool_use", "id": "t2", "name": "draft_apply", "input": { "expected_revision": 0, "requests": [] } },
            ],
        });
        let second = json!({
            "type": "message",
            "stop_reason": "end_turn",
            "content": [{ "type": "text", "text": "Done." }],
        });
        let mut model = ClaudeModel::new(
            Replies {
                sent: Vec::new(),
                replies: vec![first.to_string(), second.to_string()],
            },
            "claude-opus-5-5",
        )
        .with_context("Notes.");
        let mut messages = vec![
            // An earlier turn this runner did not see: plain text only.
            Message::User {
                text: "before".to_owned(),
            },
            Message::Assistant {
                text: "earlier".to_owned(),
                calls: Vec::new(),
            },
            Message::User {
                text: "split it".to_owned(),
            },
        ];
        let completion = model
            .complete(&CompletionRequest {
                messages: messages.clone(),
                tools: tools(),
            })
            .unwrap();
        assert_eq!(completion.text, "Reading.");
        assert_eq!(completion.calls[0].name, "draft.query");
        assert_eq!(completion.calls[0].arguments, "project");
        assert_eq!(completion.calls[1].name, "draft.apply");
        let apply: Value = serde_json::from_str(&completion.calls[1].arguments).unwrap();
        assert_eq!(apply["expected_revision"], 0);
        messages.push(Message::Assistant {
            text: completion.text.clone(),
            calls: completion.calls.clone(),
        });
        for (id, name) in [("t1", "draft.query"), ("t2", "draft.apply")] {
            messages.push(Message::Tool {
                call_id: id.to_owned(),
                name: name.to_owned(),
                content: "ok".to_owned(),
                is_error: id == "t2",
                picture: None,
            });
        }
        let done = model
            .complete(&CompletionRequest {
                messages,
                tools: tools(),
            })
            .unwrap();
        assert_eq!(done.text, "Done.");
        let sent = &model.transport().sent;
        let body = &sent[1];
        assert_eq!(body["model"], "claude-opus-5-5");
        assert_eq!(body["fallbacks"], "default");
        assert!(body["system"].as_str().unwrap().ends_with("\n\nNotes."));
        assert!(body.get("thinking").is_none());
        assert!(
            body["tools"]
                .as_array()
                .unwrap()
                .iter()
                .all(|tool| !tool["name"].as_str().unwrap().contains('.'))
        );
        let sent_messages = body["messages"].as_array().unwrap();
        assert_eq!(sent_messages.len(), 5);
        assert_eq!(
            sent_messages[1]["content"],
            json!([{ "type": "text", "text": "earlier" }])
        );
        // The reply goes back exactly as it came, thinking block included.
        assert_eq!(sent_messages[3]["content"], first["content"]);
        // Both results in one user message, in order.
        let results = sent_messages[4]["content"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["tool_use_id"], "t1");
        assert_eq!(results[1]["is_error"], true);
        // The first request ended on the user's words.
        assert_eq!(sent[0]["messages"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn errors_and_refusals_are_explained() {
        let error = parse_claude_reply(
            r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("invalid x-api-key"), "{error}");
        assert!(error.contains("Settings → Assistant"), "{error}");
        let refused = parse_claude_reply(
            r#"{"type":"message","stop_reason":"refusal","content":[],"stop_details":{"type":"refusal","category":null,"explanation":"not this"}}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(
            refused.contains("declined") && refused.contains("not this"),
            "{refused}"
        );
        assert!(parse_claude_reply("nope").is_err());
        let cut = parse_claude_reply(
            r#"{"type":"message","stop_reason":"max_tokens","content":[{"type":"text","text":"half"}]}"#,
        )
        .unwrap()
        .0;
        assert!(cut.text.contains("cut off"), "{}", cut.text);
        assert!(wants_fallback("claude-opus-5-5"));
        assert!(!wants_fallback("claude-haiku-4-5"));
    }
}
