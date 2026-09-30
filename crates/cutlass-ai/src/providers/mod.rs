//! Provider implementations behind the [`crate::provider::ChatProvider`] seam.

use std::sync::atomic::AtomicBool;

use crate::provider::{ChatProvider, ChatRequest, ChatTurn, ProviderError, ProviderStreamEvent};

pub mod openai_compat;
pub mod openai_responses;
pub mod scripted;

#[cfg(test)]
mod request_size_tests;

pub use openai_compat::{OpenAiCompatExtras, OpenAiCompatProvider};
pub use openai_responses::OpenAiResponsesProvider;
pub use scripted::ScriptedProvider;

/// Explicit wire protocol for an OpenAI-style endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiProtocol {
    ChatCompletions,
    Responses,
}

/// Provider-safe reasoning visibility. Raw chain-of-thought is never exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningSummary {
    Auto,
    Off,
}

/// Runtime protocol dispatcher used by desktop settings. Keeping this wrapper
/// outside either transport prevents Responses-only state from changing the
/// broadly-compatible Chat Completions implementation.
pub struct OpenAiProvider {
    inner: OpenAiProviderInner,
}

enum OpenAiProviderInner {
    Chat(OpenAiCompatProvider),
    Responses(OpenAiResponsesProvider),
}

/// OpenRouter Chat Completions extras for a curated model slug.
pub fn openrouter_compat_extras(model_id: &str) -> OpenAiCompatExtras {
    let mut extras = OpenAiCompatExtras {
        openrouter_headers: true,
        usage_accounting: true,
        prompt_caching: true,
        ..OpenAiCompatExtras::default()
    };
    if let Some(pin) = crate::catalog::openrouter_model(model_id).and_then(|m| m.pin) {
        extras.provider_order = Some(pin.order.iter().map(|s| (*s).to_string()).collect());
        extras.allow_fallbacks = pin.allow_fallbacks;
    }
    extras
}

impl OpenAiProvider {
    pub fn new(
        base_url: &str,
        model: &str,
        api_key: Option<String>,
        protocol: OpenAiProtocol,
        reasoning_summary: ReasoningSummary,
    ) -> Self {
        Self::with_extras(
            base_url,
            model,
            api_key,
            protocol,
            reasoning_summary,
            OpenAiCompatExtras::default(),
        )
    }

    pub fn with_extras(
        base_url: &str,
        model: &str,
        api_key: Option<String>,
        protocol: OpenAiProtocol,
        reasoning_summary: ReasoningSummary,
        extras: OpenAiCompatExtras,
    ) -> Self {
        let inner = match protocol {
            OpenAiProtocol::ChatCompletions => OpenAiProviderInner::Chat(
                OpenAiCompatProvider::with_extras(base_url, model, api_key, extras),
            ),
            OpenAiProtocol::Responses => {
                // Responses is Advanced-only; OpenRouter extras do not apply.
                OpenAiProviderInner::Responses(OpenAiResponsesProvider::new(
                    base_url,
                    model,
                    api_key,
                    reasoning_summary == ReasoningSummary::Auto,
                ))
            }
        };
        Self { inner }
    }

    pub fn test_connection(&self) -> Result<String, ProviderError> {
        match &self.inner {
            OpenAiProviderInner::Chat(provider) => provider.test_connection(),
            OpenAiProviderInner::Responses(provider) => provider.test_connection(),
        }
    }

    /// Installed model ids from `GET /v1/models` (Chat Completions path).
    pub fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        match &self.inner {
            OpenAiProviderInner::Chat(provider) => provider.list_models(),
            OpenAiProviderInner::Responses(_) => Err(ProviderError::Protocol(
                "list_models is only available for Chat Completions endpoints".into(),
            )),
        }
    }
}

impl ChatProvider for OpenAiProvider {
    fn chat(
        &self,
        request: &ChatRequest<'_>,
        cancel: &AtomicBool,
        on_event: &mut dyn FnMut(ProviderStreamEvent<'_>),
    ) -> Result<ChatTurn, ProviderError> {
        match &self.inner {
            OpenAiProviderInner::Chat(provider) => provider.chat(request, cancel, on_event),
            OpenAiProviderInner::Responses(provider) => provider.chat(request, cancel, on_event),
        }
    }
}
