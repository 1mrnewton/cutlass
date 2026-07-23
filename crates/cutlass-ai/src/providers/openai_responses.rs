//! OpenAI Responses API provider for reasoning models with function tools.
//!
//! Requests are stateless (`store: false`). During one agent prompt, the
//! provider retains only the previous response's output items so encrypted
//! reasoning and function calls can be replayed with the corresponding tool
//! outputs. The desktop creates a fresh provider for every user prompt, and
//! terminal responses clear this in-memory state.

use std::collections::{BTreeMap, HashSet};
use std::io::{BufRead, BufReader, Read};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use base64::Engine as _;
use serde::Deserialize;

use crate::provider::{
    ChatProvider, ChatRequest, ChatTurn, FinishReason, ImagePart, Message, ProviderError,
    ProviderStreamEvent, ToolCall,
};

use super::openai_compat::{retry_delay, retryable_status, sleep_unless_cancelled, truncate};

/// A Responses API transport. One instance is scoped to one agent prompt so
/// replay state cannot leak into persisted conversation history.
pub struct OpenAiResponsesProvider {
    base_url: String,
    model: String,
    api_key: Option<String>,
    reasoning_summary: bool,
    agent: ureq::Agent,
    replay: Mutex<Option<ReplayState>>,
}

#[derive(Clone)]
struct ReplayState {
    /// Messages through the user prompt and persisted text history. Items after
    /// this boundary belong to the current tool loop and are replayed exactly.
    base_message_count: usize,
    /// Number of provider-agnostic messages present before the response that
    /// produced the latest output. The agent appends its assistant turn here.
    request_message_count: usize,
    /// Exact Responses output and function-call-output items since the latest
    /// user message, including encrypted reasoning from every tool round.
    continuation: Vec<serde_json::Value>,
    call_ids: HashSet<String>,
}

#[derive(Debug)]
struct ParsedResponse {
    turn: ChatTurn,
    output: Vec<serde_json::Value>,
}

impl OpenAiResponsesProvider {
    pub fn new(
        base_url: &str,
        model: &str,
        api_key: Option<String>,
        reasoning_summary: bool,
    ) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            reasoning_summary,
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(10))
                .build(),
            replay: Mutex::new(None),
        }
    }

    /// The Responses protocol shares the provider's token-free `/models`
    /// liveness endpoint with Chat Completions.
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
                let count = parsed["data"].as_array().map_or(0, Vec::len);
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
            Err(ureq::Error::Transport(error)) => {
                Err(ProviderError::Network(format!("{url}: {error}")))
            }
        }
    }

    fn request_body(&self, request: &ChatRequest<'_>) -> serde_json::Value {
        let instructions = request
            .messages
            .iter()
            .filter_map(|message| match message {
                Message::System { content } => Some(content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let replay = self.replay.lock().unwrap().clone();
        let input = match replay.filter(|state| replay_matches(state, request.messages)) {
            Some(state) => replay_input(request.messages, &state),
            None => to_responses_input(request.messages),
        };
        let tools = request
            .tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "name": &tool.name,
                    "description": &tool.description,
                    "parameters": &tool.parameters,
                    "strict": false,
                })
            })
            .collect::<Vec<_>>();

        let mut body = serde_json::json!({
            "model": self.model,
            "stream": true,
            "store": false,
            "instructions": instructions,
            "input": input,
            "include": ["reasoning.encrypted_content"],
        });
        if !tools.is_empty() {
            body["tools"] = tools.into();
        }
        if self.reasoning_summary {
            body["reasoning"] = serde_json::json!({ "summary": "auto" });
        }
        body
    }

    fn update_replay(&self, messages: &[Message], turn: &ChatTurn, output: Vec<serde_json::Value>) {
        let mut replay = self.replay.lock().unwrap();
        if turn.finish == FinishReason::ToolCalls && !turn.tool_calls.is_empty() {
            let (base_message_count, mut continuation) = match replay.take() {
                Some(state) if replay_matches(&state, messages) => {
                    let tool_outputs = replay_tool_outputs(messages, &state);
                    let base_message_count = state.base_message_count;
                    let mut continuation = state.continuation;
                    continuation.extend(tool_outputs);
                    (base_message_count, continuation)
                }
                _ => (messages.len(), Vec::new()),
            };
            continuation.extend(output);
            *replay = Some(ReplayState {
                base_message_count,
                request_message_count: messages.len(),
                continuation,
                call_ids: turn.tool_calls.iter().map(|call| call.id.clone()).collect(),
            });
        } else {
            *replay = None;
        }
    }

    fn clear_replay(&self) {
        *self.replay.lock().unwrap() = None;
    }
}

impl ChatProvider for OpenAiResponsesProvider {
    fn chat(
        &self,
        request: &ChatRequest<'_>,
        cancel: &AtomicBool,
        on_event: &mut dyn FnMut(ProviderStreamEvent<'_>),
    ) -> Result<ChatTurn, ProviderError> {
        let url = format!("{}/responses", self.base_url);
        let body = self.request_body(request).to_string();

        // Match Chat Completions: retry only while sending the initial
        // request. Once an SSE reader exists, emitted deltas are never replayed.
        let mut attempt = 0usize;
        let response = loop {
            if cancel.load(Ordering::Relaxed) {
                self.clear_replay();
                return Err(ProviderError::Cancelled);
            }
            let mut http = self
                .agent
                .post(&url)
                .set("Content-Type", "application/json")
                .set("Accept", "text/event-stream");
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
                            Some(_) => {
                                self.clear_replay();
                                return Err(ProviderError::Cancelled);
                            }
                            None => {}
                        }
                    }
                    let message = response
                        .into_string()
                        .unwrap_or_else(|_| "<unreadable error body>".to_string());
                    self.clear_replay();
                    return Err(ProviderError::Provider {
                        status,
                        message: truncate(&message, 500),
                    });
                }
                Err(ureq::Error::Transport(error)) => match retry_delay(attempt) {
                    Some(delay) if sleep_unless_cancelled(delay, cancel) => attempt += 1,
                    Some(_) => {
                        self.clear_replay();
                        return Err(ProviderError::Cancelled);
                    }
                    None => {
                        self.clear_replay();
                        return Err(ProviderError::Network(format!("{url}: {error}")));
                    }
                },
            }
        };

        match consume_responses_sse(response.into_reader(), cancel, on_event) {
            Ok(parsed) => {
                self.update_replay(request.messages, &parsed.turn, parsed.output);
                Ok(parsed.turn)
            }
            Err(error) => {
                self.clear_replay();
                Err(error)
            }
        }
    }
}

fn replay_matches(state: &ReplayState, messages: &[Message]) -> bool {
    let Some(Message::Assistant { tool_calls, .. }) = messages.get(state.request_message_count)
    else {
        return false;
    };
    let assistant_ids = tool_calls
        .iter()
        .map(|call| call.id.as_str())
        .collect::<HashSet<_>>();
    let result_ids = messages[state.request_message_count + 1..]
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect::<HashSet<_>>();
    assistant_ids.len() == state.call_ids.len()
        && state
            .call_ids
            .iter()
            .all(|id| assistant_ids.contains(id.as_str()))
        && state
            .call_ids
            .iter()
            .all(|id| result_ids.contains(id.as_str()))
}

fn replay_input(messages: &[Message], state: &ReplayState) -> Vec<serde_json::Value> {
    let mut input = to_responses_input(&messages[..state.base_message_count]);
    input.extend(state.continuation.iter().cloned());
    input.extend(replay_tool_outputs(messages, state));
    input
}

fn replay_tool_outputs(messages: &[Message], state: &ReplayState) -> Vec<serde_json::Value> {
    messages[state.request_message_count + 1..]
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult { call_id, .. } if state.call_ids.contains(call_id) => {
                Some(tool_result_input(message))
            }
            _ => None,
        })
        .collect()
}

fn to_responses_input(messages: &[Message]) -> Vec<serde_json::Value> {
    let mut input = Vec::new();
    for message in messages {
        match message {
            Message::System { .. } => {}
            Message::User { content, images } => {
                input.push(user_input_message(content, images));
            }
            Message::Assistant {
                content,
                tool_calls,
            } => {
                if !content.is_empty() {
                    input.push(assistant_output_message(content));
                }
                input.extend(tool_calls.iter().map(|call| {
                    serde_json::json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments.to_string(),
                    })
                }));
            }
            Message::ToolResult { .. } => input.push(tool_result_input(message)),
        }
    }
    input
}

fn user_input_message(content: &str, images: &[ImagePart]) -> serde_json::Value {
    let mut parts = vec![serde_json::json!({
        "type": "input_text",
        "text": content,
    })];
    parts.extend(images.iter().map(input_image_part));
    serde_json::json!({
        "role": "user",
        "content": parts,
    })
}

fn assistant_output_message(content: &str) -> serde_json::Value {
    serde_json::json!({
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": content,
        }],
    })
}

fn tool_result_input(message: &Message) -> serde_json::Value {
    let Message::ToolResult {
        call_id,
        content,
        images,
    } = message
    else {
        unreachable!("tool_result_input requires a tool result");
    };
    if images.is_empty() {
        return serde_json::json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": content,
        });
    }

    let labels = images
        .iter()
        .map(|image| image.label.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let mut output = vec![serde_json::json!({
        "type": "input_text",
        "text": format!("{content}\n[attachments: {labels}]"),
    })];
    output.extend(images.iter().map(input_image_part));
    serde_json::json!({
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    })
}

fn input_image_part(image: &ImagePart) -> serde_json::Value {
    let encoded = base64::engine::general_purpose::STANDARD.encode(image.data.as_slice());
    serde_json::json!({
        "type": "input_image",
        "image_url": format!("data:{};base64,{encoded}", image.media_type),
        "detail": "auto",
    })
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ResponsesEvent {
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta { delta: String },
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta { delta: String },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: usize,
        item: serde_json::Value,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta { output_index: usize, delta: String },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        output_index: usize,
        arguments: String,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        output_index: usize,
        item: serde_json::Value,
    },
    #[serde(rename = "response.completed")]
    Completed { response: ResponseEnvelope },
    #[serde(rename = "response.incomplete")]
    Incomplete { response: ResponseEnvelope },
    #[serde(rename = "response.failed")]
    Failed { response: ResponseEnvelope },
    #[serde(rename = "error")]
    Error { message: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Default, Deserialize)]
struct ResponseEnvelope {
    #[serde(default)]
    output: Vec<serde_json::Value>,
    #[serde(default)]
    error: Option<ResponseApiError>,
    #[serde(default)]
    incomplete_details: Option<IncompleteDetails>,
}

#[derive(Debug, Default, Deserialize)]
struct ResponseApiError {
    #[serde(default)]
    message: String,
}

#[derive(Debug, Default, Deserialize)]
struct IncompleteDetails {
    #[serde(default)]
    reason: String,
}

enum Terminal {
    Completed(ResponseEnvelope),
    Incomplete(ResponseEnvelope),
}

/// Parse typed Responses events while preserving separate answer and provider
/// reasoning-summary channels.
fn consume_responses_sse(
    reader: impl Read,
    cancel: &AtomicBool,
    on_event: &mut dyn FnMut(ProviderStreamEvent<'_>),
) -> Result<ParsedResponse, ProviderError> {
    let mut streamed_text = String::new();
    let mut streamed_reasoning = String::new();
    let mut output_items = BTreeMap::<usize, serde_json::Value>::new();
    let mut terminal = None;

    for line in BufReader::new(reader).lines() {
        if cancel.load(Ordering::Relaxed) {
            return Err(ProviderError::Cancelled);
        }
        let line = line.map_err(|error| {
            ProviderError::Network(format!("Responses stream read failed: {error}"))
        })?;
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }
        let event: ResponsesEvent = serde_json::from_str(data).map_err(|error| {
            ProviderError::Protocol(format!("bad Responses SSE event: {error}: {data}"))
        })?;
        match event {
            ResponsesEvent::OutputTextDelta { delta } => {
                if !delta.is_empty() {
                    streamed_text.push_str(&delta);
                    on_event(ProviderStreamEvent::TextDelta(&delta));
                }
            }
            ResponsesEvent::ReasoningSummaryTextDelta { delta } => {
                if !delta.is_empty() {
                    streamed_reasoning.push_str(&delta);
                    on_event(ProviderStreamEvent::ReasoningSummaryDelta(&delta));
                }
            }
            ResponsesEvent::OutputItemAdded { output_index, item }
            | ResponsesEvent::OutputItemDone { output_index, item } => {
                output_items.insert(output_index, item);
            }
            ResponsesEvent::FunctionCallArgumentsDelta {
                output_index,
                delta,
            } => append_function_arguments(&mut output_items, output_index, &delta),
            ResponsesEvent::FunctionCallArgumentsDone {
                output_index,
                arguments,
            } => set_function_arguments(&mut output_items, output_index, arguments),
            ResponsesEvent::Completed { response } => {
                terminal = Some(Terminal::Completed(response));
                break;
            }
            ResponsesEvent::Incomplete { response } => {
                terminal = Some(Terminal::Incomplete(response));
                break;
            }
            ResponsesEvent::Failed { response } => {
                return Err(ProviderError::Protocol(format!(
                    "Responses request failed: {}",
                    response_error_message(&response)
                )));
            }
            ResponsesEvent::Error { message } => {
                return Err(ProviderError::Protocol(format!(
                    "Responses stream error: {message}"
                )));
            }
            ResponsesEvent::Other => {}
        }
    }

    if cancel.load(Ordering::Relaxed) {
        return Err(ProviderError::Cancelled);
    }
    let Some(terminal) = terminal else {
        return Err(ProviderError::Protocol(
            "Responses stream ended without a terminal event".to_string(),
        ));
    };
    let (envelope, finish) = match terminal {
        Terminal::Completed(response) => (response, FinishReason::Stop),
        Terminal::Incomplete(response) => {
            let reason = response
                .incomplete_details
                .as_ref()
                .map(|details| details.reason.as_str())
                .unwrap_or("unknown");
            if !matches!(reason, "max_output_tokens" | "max_tokens") {
                return Err(ProviderError::Protocol(format!(
                    "Responses request was incomplete: {reason}"
                )));
            }
            (response, FinishReason::Length)
        }
    };

    let output = if envelope.output.is_empty() {
        output_items.into_values().collect()
    } else {
        envelope.output
    };
    let final_text = output_text(&output);
    let text = if streamed_text.is_empty() {
        if !final_text.is_empty() {
            on_event(ProviderStreamEvent::TextDelta(&final_text));
        }
        final_text
    } else {
        streamed_text
    };
    let final_reasoning = reasoning_summary(&output);
    let reasoning_summary = if streamed_reasoning.is_empty() {
        if !final_reasoning.is_empty() {
            on_event(ProviderStreamEvent::ReasoningSummaryDelta(&final_reasoning));
        }
        final_reasoning
    } else {
        streamed_reasoning
    };
    let tool_calls = if finish == FinishReason::Stop {
        parse_function_calls(&output)?
    } else {
        Vec::new()
    };
    let finish = if !tool_calls.is_empty() {
        FinishReason::ToolCalls
    } else {
        finish
    };

    Ok(ParsedResponse {
        turn: ChatTurn {
            text,
            reasoning_summary,
            tool_calls,
            finish,
        },
        output,
    })
}

fn append_function_arguments(
    items: &mut BTreeMap<usize, serde_json::Value>,
    output_index: usize,
    delta: &str,
) {
    let item = items.entry(output_index).or_insert_with(|| {
        serde_json::json!({
            "type": "function_call",
            "arguments": "",
        })
    });
    let arguments = item
        .as_object_mut()
        .expect("internally-created output item is an object")
        .entry("arguments")
        .or_insert_with(|| serde_json::Value::String(String::new()));
    let combined = arguments
        .as_str()
        .map(|current| format!("{current}{delta}"))
        .unwrap_or_else(|| delta.to_string());
    *arguments = serde_json::Value::String(combined);
}

fn set_function_arguments(
    items: &mut BTreeMap<usize, serde_json::Value>,
    output_index: usize,
    arguments: String,
) {
    let item = items.entry(output_index).or_insert_with(|| {
        serde_json::json!({
            "type": "function_call",
        })
    });
    item["arguments"] = arguments.into();
}

fn output_text(output: &[serde_json::Value]) -> String {
    output
        .iter()
        .filter(|item| item["type"] == "message")
        .filter_map(|item| item["content"].as_array())
        .flatten()
        .filter(|part| part["type"] == "output_text")
        .filter_map(|part| part["text"].as_str())
        .collect()
}

fn reasoning_summary(output: &[serde_json::Value]) -> String {
    output
        .iter()
        .filter(|item| item["type"] == "reasoning")
        .filter_map(|item| item["summary"].as_array())
        .flatten()
        .filter_map(|part| part["text"].as_str())
        .collect()
}

fn parse_function_calls(output: &[serde_json::Value]) -> Result<Vec<ToolCall>, ProviderError> {
    output
        .iter()
        .filter(|item| item["type"] == "function_call")
        .map(|item| {
            let id = item["call_id"]
                .as_str()
                .or_else(|| item["id"].as_str())
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    ProviderError::Protocol(
                        "Responses function call is missing call_id".to_string(),
                    )
                })?;
            let name = item["name"]
                .as_str()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    ProviderError::Protocol(format!(
                        "Responses function call {id} is missing a name"
                    ))
                })?;
            let raw_arguments = item["arguments"].as_str().unwrap_or("");
            let arguments = if raw_arguments.trim().is_empty() {
                serde_json::json!({})
            } else {
                serde_json::from_str(raw_arguments).map_err(|error| {
                    ProviderError::Protocol(format!(
                        "Responses function call '{name}' has unparseable arguments: \
                         {error}: {raw_arguments}"
                    ))
                })?
            };
            Ok(ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments,
            })
        })
        .collect()
}

fn response_error_message(response: &ResponseEnvelope) -> String {
    response
        .error
        .as_ref()
        .map(|error| error.message.as_str())
        .filter(|message| !message.is_empty())
        .unwrap_or("unknown provider error")
        .to_string()
}

#[cfg(test)]
mod tests;
