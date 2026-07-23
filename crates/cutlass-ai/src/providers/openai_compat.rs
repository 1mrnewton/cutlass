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
    ChatProvider, ChatRequest, ChatTurn, FinishReason, ImagePart, Message, ProviderError,
    ProviderStreamEvent, TokenUsage, ToolCall, json_u64,
};

/// Optional OpenRouter-only request extras (provider pinning + app headers).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAiCompatExtras {
    /// `provider.order` upstream preference list.
    pub provider_order: Option<Vec<String>>,
    /// When false with a non-empty order, OR will not fall back to other hosts.
    pub allow_fallbacks: bool,
    /// Send `HTTP-Referer` / `X-Title` attribution headers.
    pub openrouter_headers: bool,
    /// OpenRouter usage-accounting extension (`usage.include`), which adds
    /// `cost` to the final usage chunk. Leave false for non-OpenRouter hosts.
    pub usage_accounting: bool,
    /// OpenRouter automatic prompt caching (`cache_control: ephemeral`).
    /// Harmless on non-Anthropic OpenRouter models; leave false elsewhere.
    pub prompt_caching: bool,
}

pub struct OpenAiCompatProvider {
    base_url: String,
    model: String,
    api_key: Option<String>,
    extras: OpenAiCompatExtras,
    agent: ureq::Agent,
}

impl OpenAiCompatProvider {
    pub fn new(base_url: &str, model: &str, api_key: Option<String>) -> Self {
        Self::with_extras(base_url, model, api_key, OpenAiCompatExtras::default())
    }

    pub fn with_extras(
        base_url: &str,
        model: &str,
        api_key: Option<String>,
        extras: OpenAiCompatExtras,
    ) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            extras,
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(10))
                .build(),
        }
    }

    fn apply_auth_headers(&self, mut http: ureq::Request) -> ureq::Request {
        if let Some(key) = &self.api_key {
            http = http.set("Authorization", &format!("Bearer {key}"));
        }
        if self.extras.openrouter_headers {
            http = http
                .set("HTTP-Referer", crate::catalog::OPENROUTER_HTTP_REFERER)
                .set("X-Title", crate::catalog::OPENROUTER_APP_TITLE);
        }
        http
    }

    /// Liveness probe for the Settings dialog: `GET {base_url}/models`, the
    /// OpenAI-compatible health endpoint (Ollama/LM Studio/OpenAI all serve
    /// it). Returns a short human summary on success; spends no tokens.
    pub fn test_connection(&self) -> Result<String, ProviderError> {
        let ids = self.list_models()?;
        Ok(match ids.len() {
            0 => "Connected.".to_string(),
            1 => "Connected · 1 model available.".to_string(),
            n => format!("Connected · {n} models available."),
        })
    }

    /// `GET {base_url}/models` → installed model ids (OpenAI `data[].id`).
    pub fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        let url = format!("{}/models", self.base_url);
        let http = self.apply_auth_headers(self.agent.get(&url));
        match http.call() {
            Ok(response) => {
                let body = response
                    .into_string()
                    .map_err(|e| ProviderError::Network(format!("reading /models: {e}")))?;
                let parsed: serde_json::Value = serde_json::from_str(&body)
                    .map_err(|e| ProviderError::Protocol(format!("bad /models response: {e}")))?;
                let ids = parsed["data"]
                    .as_array()
                    .map(|rows| {
                        rows.iter()
                            .filter_map(|row| row["id"].as_str().map(str::to_owned))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                Ok(ids)
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

    /// Build the Chat Completions wire body for `request`.
    ///
    /// `pub(crate)` so offline request-size harnesses can measure the exact
    /// payload without opening a network connection.
    pub(crate) fn request_body(&self, request: &ChatRequest<'_>) -> serde_json::Value {
        self.request_body_with_stream_options(request, true)
    }

    /// Like [`Self::request_body`], optionally omitting `stream_options` for
    /// older OpenAI-compatible servers that reject the field with HTTP 400.
    pub(crate) fn request_body_with_stream_options(
        &self,
        request: &ChatRequest<'_>,
        include_stream_options: bool,
    ) -> serde_json::Value {
        let messages = to_openai_messages(request.messages);
        let mut body = serde_json::json!({
            "model": self.model,
            "stream": true,
            "messages": messages,
        });
        if include_stream_options {
            body["stream_options"] = serde_json::json!({ "include_usage": true });
        }
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
        if self.extras.usage_accounting {
            body["usage"] = serde_json::json!({ "include": true });
        }
        if self.extras.prompt_caching {
            body["cache_control"] = serde_json::json!({ "type": "ephemeral" });
            if let Some(session_id) = request.session_id {
                body["session_id"] = serde_json::Value::String(session_id.to_string());
            }
        }
        if let Some(order) = &self.extras.provider_order
            && !order.is_empty()
        {
            body["provider"] = serde_json::json!({
                "order": order,
                "allow_fallbacks": self.extras.allow_fallbacks,
            });
        }
        body
    }
}

impl ChatProvider for OpenAiCompatProvider {
    fn chat(
        &self,
        request: &ChatRequest<'_>,
        cancel: &AtomicBool,
        on_event: &mut dyn FnMut(ProviderStreamEvent<'_>),
    ) -> Result<ChatTurn, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);

        // Only the initial send retries: no stream bytes have been
        // consumed yet, so a retry can't duplicate text in the UI.
        // Mid-stream failures (in consume_sse) always surface.
        // A one-shot stream_options fallback is separate from the transport
        // retry budget — strict servers that reject the field get one retry
        // without it, without burning a backoff attempt.
        let mut attempt = 0usize;
        let mut include_stream_options = true;
        let mut stream_options_retried = false;
        let response = loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(ProviderError::Cancelled);
            }
            let body = if include_stream_options {
                self.request_body(request)
            } else {
                self.request_body_with_stream_options(request, false)
            }
            .to_string();
            let http = self.apply_auth_headers(
                self.agent
                    .post(&url)
                    .set("Content-Type", "application/json"),
            );
            match http.send_string(&body) {
                Ok(response) => break response,
                Err(ureq::Error::Status(status, response)) => {
                    let message = response
                        .into_string()
                        .unwrap_or_else(|_| "<unreadable error body>".to_string());
                    if !stream_options_retried && is_stream_options_rejection(status, &message) {
                        stream_options_retried = true;
                        include_stream_options = false;
                        continue;
                    }
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
                    return Err(ProviderError::Provider {
                        status,
                        message: chat_error_message(status, &message),
                    });
                }
                Err(ureq::Error::Transport(t)) => match retry_delay(attempt) {
                    Some(delay) if sleep_unless_cancelled(delay, cancel) => attempt += 1,
                    Some(_) => return Err(ProviderError::Cancelled),
                    None => return Err(ProviderError::Network(format!("{url}: {t}"))),
                },
            }
        };

        consume_sse(response.into_reader(), cancel, &mut |delta| {
            on_event(ProviderStreamEvent::TextDelta(delta));
        })
    }
}

/// True when a Chat Completions server rejected `stream_options` (strict /
/// older OpenAI-compatible endpoints that treat unknown fields as errors).
pub(crate) fn is_stream_options_rejection(status: u16, body: &str) -> bool {
    status == 400 && body.to_ascii_lowercase().contains("stream_options")
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
pub(super) fn retryable_status(status: u16) -> bool {
    status == 408 || status == 429 || (500..=599).contains(&status)
}

/// Backoff before re-sending the initial request after a transport
/// failure: two retries, spaced so a briefly-napping local server
/// (Ollama model load, sleep wake) gets a second chance without turning
/// a dead endpoint into a long hang.
pub(super) fn retry_delay(attempt: usize) -> Option<Duration> {
    match attempt {
        0 => Some(Duration::from_millis(300)),
        1 => Some(Duration::from_millis(900)),
        _ => None,
    }
}

/// Sleep in ~50ms slices, polling `cancel`. Returns false — retry
/// abandoned — the moment cancellation shows up.
pub(super) fn sleep_unless_cancelled(total: Duration, cancel: &AtomicBool) -> bool {
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
    let mut usage: Option<TokenUsage> = None;

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
        // Usage can arrive on a usage-only chunk or beside choices — keep the
        // last non-empty value so a trailing `"usage": {}` cannot clobber it.
        if let Some(parsed) = parse_compat_usage(&chunk["usage"])
            && !parsed.is_empty()
        {
            usage = Some(parsed);
        }
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
        if let Some(piece) = delta["content"].as_str()
            && !piece.is_empty()
        {
            text.push_str(piece);
            on_text(piece);
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
        reasoning_summary: String::new(),
        tool_calls,
        finish,
        usage,
    })
}

fn parse_compat_usage(usage: &serde_json::Value) -> Option<TokenUsage> {
    if !usage.is_object() {
        return None;
    }
    Some(TokenUsage {
        input_tokens: json_u64(&usage["prompt_tokens"]),
        cached_input_tokens: json_u64(&usage["prompt_tokens_details"]["cached_tokens"]),
        output_tokens: json_u64(&usage["completion_tokens"]),
        cost: usage["cost"].as_f64(),
    })
}

pub(super) fn truncate(s: &str, max: usize) -> String {
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

fn chat_error_message(status: u16, message: &str) -> String {
    let mut message = truncate(message, 500);
    let lower = message.to_ascii_lowercase();
    if status == 400
        && (lower.contains("reasoning_effort") || lower.contains("reasoning effort"))
        && (lower.contains("tool") || lower.contains("function"))
    {
        message.push_str(
            "\nThis reasoning model cannot call tools through Chat Completions. \
             In Settings → AI provider, choose the Responses API protocol.",
        );
    }
    message
}

#[cfg(test)]
mod tests;
