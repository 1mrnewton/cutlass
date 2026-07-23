//! The provider seam: chat completion with tool calling, behind one trait.
//!
//! Blocking by design — the agent runs on its own thread (never the UI
//! thread), and a synchronous trait keeps tokio out of the app. Streaming
//! is a text callback (for the chat panel) plus a completed [`ChatTurn`]
//! return; tool calls arrive whole, in the turn.

use std::sync::atomic::AtomicBool;

use crate::wire::ToolSpec;

/// An image attached to a message. Raw encoded bytes (PNG or JPEG) —
/// base64 encoding happens at the provider boundary, never earlier.
/// Images are per-turn working memory: the runtime budgets them per
/// request and strips them from session history (see agent.rs).
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePart {
    /// MIME type: "image/png" or "image/jpeg".
    pub media_type: String,
    /// Raw encoded bytes, shared so message clones stay cheap.
    pub data: std::sync::Arc<Vec<u8>>,
    /// Short human label for transcripts and placeholders, e.g. "timeline at 12.40s".
    pub label: String,
}

impl ImagePart {
    pub fn png(data: Vec<u8>, label: impl Into<String>) -> Self {
        Self {
            media_type: "image/png".to_string(),
            data: std::sync::Arc::new(data),
            label: label.into(),
        }
    }

    pub fn jpeg(data: Vec<u8>, label: impl Into<String>) -> Self {
        Self {
            media_type: "image/jpeg".to_string(),
            data: std::sync::Arc::new(data),
            label: label.into(),
        }
    }
}

/// One entry in the conversation, provider-agnostic.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: String,
        images: Vec<ImagePart>,
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
        images: Vec<ImagePart>,
    },
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
            images: Vec::new(),
        }
    }

    pub fn assistant_text(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult {
            call_id: call_id.into(),
            content: content.into(),
            images: Vec::new(),
        }
    }
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

/// Token usage for one completed provider turn, as reported by the API.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TokenUsage {
    pub input_tokens: u64,
    /// Portion of input_tokens served from the provider's prompt cache.
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    /// Provider-reported cost in USD (OpenRouter usage accounting), when available.
    pub cost: Option<f64>,
}

impl TokenUsage {
    /// Saturating sum of token counts.
    ///
    /// Cost is `Some` only when every accumulated non-empty turn reported a
    /// cost. The accumulator starts empty, so the first added turn's cost
    /// carries through; a later usage-bearing turn without cost poisons the
    /// total to `None` permanently.
    pub fn add(&mut self, other: &TokenUsage) {
        let self_was_empty = self.is_empty();
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.cached_input_tokens = self
            .cached_input_tokens
            .saturating_add(other.cached_input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cost = if self_was_empty {
            other.cost
        } else {
            match (self.cost, other.cost) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            }
        };
    }

    pub fn is_empty(&self) -> bool {
        self.input_tokens == 0
            && self.cached_input_tokens == 0
            && self.output_tokens == 0
            && self.cost.is_none()
    }
}

/// Parse a JSON number as a token count. Accepts integers and floats
/// (rounded to nearest); anything else yields 0.
pub(crate) fn json_u64(value: &serde_json::Value) -> u64 {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                u
            } else if let Some(i) = n.as_i64() {
                u64::try_from(i).unwrap_or(0)
            } else if let Some(f) = n.as_f64() {
                if f.is_finite() && f >= 0.0 {
                    f.round() as u64
                } else {
                    0
                }
            } else {
                0
            }
        }
        _ => 0,
    }
}

/// One completed model turn.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatTurn {
    pub text: String,
    /// Provider-generated explanation of the model's reasoning. This is
    /// display-only and must never be copied into [`Message`] history.
    pub reasoning_summary: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish: FinishReason,
    /// Token usage for this turn, when the provider reported it.
    pub usage: Option<TokenUsage>,
}

impl ChatTurn {
    /// Attach provider-reported usage (for scripted tests and fixtures).
    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.usage = Some(usage);
        self
    }
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

/// One provider stream delta. Reasoning summaries stay a distinct channel so
/// callers cannot accidentally append them to assistant text or model history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderStreamEvent<'a> {
    TextDelta(&'a str),
    ReasoningSummaryDelta(&'a str),
}

/// Chat completion with tool calling and streamed display events.
///
/// Implementations must check `cancel` between chunks and return
/// [`ProviderError::Cancelled`] promptly when it goes true.
pub trait ChatProvider {
    fn chat(
        &self,
        request: &ChatRequest<'_>,
        cancel: &AtomicBool,
        on_event: &mut dyn FnMut(ProviderStreamEvent<'_>),
    ) -> Result<ChatTurn, ProviderError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_usage_add_and_is_empty() {
        let mut total = TokenUsage::default();
        assert!(total.is_empty());

        total.add(&TokenUsage {
            input_tokens: 10,
            cached_input_tokens: 4,
            output_tokens: 2,
            cost: None,
        });
        assert!(!total.is_empty());
        assert_eq!(total.input_tokens, 10);
        assert_eq!(total.cached_input_tokens, 4);
        assert_eq!(total.output_tokens, 2);
        assert_eq!(total.cost, None);

        // A later turn with cost cannot resurrect a poisoned total.
        total.add(&TokenUsage {
            input_tokens: 5,
            cached_input_tokens: 1,
            output_tokens: 3,
            cost: Some(0.01),
        });
        assert_eq!(total.input_tokens, 15);
        assert_eq!(total.cached_input_tokens, 5);
        assert_eq!(total.output_tokens, 5);
        assert_eq!(total.cost, None);
    }

    #[test]
    fn token_usage_cost_empty_plus_some_carries_through() {
        let mut total = TokenUsage::default();
        total.add(&TokenUsage {
            input_tokens: 10,
            cached_input_tokens: 0,
            output_tokens: 2,
            cost: Some(0.01),
        });
        assert_eq!(total.cost, Some(0.01));
    }

    #[test]
    fn token_usage_cost_some_plus_some_sums() {
        let mut total = TokenUsage {
            input_tokens: 10,
            cached_input_tokens: 0,
            output_tokens: 2,
            cost: Some(0.01),
        };
        total.add(&TokenUsage {
            input_tokens: 5,
            cached_input_tokens: 0,
            output_tokens: 1,
            cost: Some(0.02),
        });
        assert_eq!(total.cost, Some(0.03));
    }

    #[test]
    fn token_usage_cost_some_plus_none_poisons_total() {
        let mut total = TokenUsage {
            input_tokens: 10,
            cached_input_tokens: 0,
            output_tokens: 2,
            cost: Some(0.01),
        };
        total.add(&TokenUsage {
            input_tokens: 5,
            cached_input_tokens: 0,
            output_tokens: 1,
            cost: None,
        });
        assert_eq!(total.cost, None);
    }

    #[test]
    fn token_usage_cost_none_stays_none_when_later_turns_have_cost() {
        let mut total = TokenUsage {
            input_tokens: 10,
            cached_input_tokens: 0,
            output_tokens: 2,
            cost: None,
        };
        total.add(&TokenUsage {
            input_tokens: 5,
            cached_input_tokens: 0,
            output_tokens: 1,
            cost: Some(0.02),
        });
        assert_eq!(total.cost, None);
    }

    #[test]
    fn json_u64_accepts_integers_and_rounded_floats() {
        assert_eq!(json_u64(&serde_json::json!(123)), 123);
        assert_eq!(json_u64(&serde_json::json!(123.0)), 123);
        assert_eq!(json_u64(&serde_json::json!(123.4)), 123);
        assert_eq!(json_u64(&serde_json::json!(123.6)), 124);
        assert_eq!(json_u64(&serde_json::json!("nope")), 0);
        assert_eq!(json_u64(&serde_json::Value::Null), 0);
    }
}
