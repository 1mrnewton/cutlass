//! The provider seam: chat completion with tool calling, behind one trait.
//!
//! Blocking by design — the agent runs on its own thread (never the UI
//! thread), and a synchronous trait keeps tokio out of the app. Streaming
//! is a text callback (for the chat panel) plus a completed [`ChatTurn`]
//! return; tool calls arrive whole, in the turn.

use std::sync::atomic::AtomicBool;

use crate::wire::ToolSpec;

/// One entry in the conversation, provider-agnostic.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: String,
    },
    /// A prior model turn (text and/or the tool calls it made).
    Assistant {
        content: String,
        tool_calls: Vec<ToolCall>,
    },
    /// The outcome of one tool call, fed back to the model.
    ToolResult {
        call_id: String,
        content: String,
    },
}

/// A tool invocation the model requested.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// Provider-assigned id; echoed back in the matching [`Message::ToolResult`].
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Why the model stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Natural end of a text answer.
    Stop,
    /// The model wants its tool calls executed.
    ToolCalls,
    /// Token limit hit; the turn is truncated.
    Length,
    Other,
}

/// One completed model turn.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatTurn {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish: FinishReason,
}

/// Everything a provider needs for one completion.
pub struct ChatRequest<'a> {
    pub messages: &'a [Message],
    pub tools: &'a [ToolSpec],
}

/// Provider failures, kept distinct so the UI can say "Ollama isn't
/// running at localhost:11434" instead of "something failed".
#[derive(Debug)]
pub enum ProviderError {
    /// No `[ai]` config, or it is unusable (missing key, bad env var).
    NotConfigured(String),
    /// Could not reach the endpoint at all.
    Network(String),
    /// The endpoint answered with an error (HTTP status, rate limit, …).
    Provider { status: u16, message: String },
    /// The endpoint answered with something we could not parse.
    Protocol(String),
    /// The cancel flag was raised mid-stream.
    Cancelled,
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotConfigured(msg) => write!(f, "AI is not configured: {msg}"),
            Self::Network(msg) => write!(f, "could not reach the AI provider: {msg}"),
            Self::Provider { status, message } => {
                write!(f, "the AI provider returned HTTP {status}: {message}")
            }
            Self::Protocol(msg) => write!(f, "unexpected response from the AI provider: {msg}"),
            Self::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for ProviderError {}

/// Chat completion with tool calling and streamed text.
///
/// Implementations must check `cancel` between chunks and return
/// [`ProviderError::Cancelled`] promptly when it goes true. `on_text`
/// receives assistant text deltas as they stream.
pub trait ChatProvider {
    fn chat(
        &self,
        request: &ChatRequest<'_>,
        cancel: &AtomicBool,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<ChatTurn, ProviderError>;
}
