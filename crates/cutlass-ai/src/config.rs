//! API-key resolution and provider construction from [`AiSettings`].
//!
//! The config *file* (`~/.cutlass/config.toml`) is owned by the
//! `cutlass-settings` crate — its `[ai]` table parses into
//! `cutlass_settings::AiSettings`. This module resolves secrets and builds
//! the runtime [`OpenAiProvider`] for Local / OpenRouter / Custom.

use cutlass_settings::{AiApiProtocol, AiSettings, AiSource, ReasoningSummary as SettingsSummary};

use crate::catalog::{
    OPENROUTER_BASE_URL, is_known_local_model, is_known_openrouter_model, openrouter_model,
};
use crate::providers::{
    OpenAiCompatExtras, OpenAiProtocol, OpenAiProvider, ReasoningSummary, openrouter_compat_extras,
};

/// Resolve the API key to send, honoring `api_key_env` over a literal
/// `api_key`. `Ok(None)` means no key (fine for local servers); `Err` names
/// what is missing (an `api_key_env` pointing at an unset variable).
pub fn resolve_api_key(
    api_key: Option<&str>,
    api_key_env: Option<&str>,
) -> Result<Option<String>, String> {
    if let Some(var) = api_key_env {
        return match std::env::var(var) {
            Ok(key) if !key.is_empty() => Ok(Some(key)),
            _ => Err(format!(
                "api_key_env points at '{var}' but that environment variable is unset"
            )),
        };
    }
    Ok(api_key.map(str::to_owned))
}

/// Structural + allowlist checks beyond [`AiSettings::is_configured`].
pub fn validate_ai_settings(ai: &AiSettings) -> Result<(), String> {
    if !ai.is_configured() {
        return Err(match ai.source {
            AiSource::Local => "Local AI needs a base URL and a supported model.".into(),
            AiSource::OpenRouter => "OpenRouter needs an API key and a supported model.".into(),
            AiSource::Custom => "Advanced AI needs an endpoint URL and a model name.".into(),
        });
    }
    match ai.source {
        AiSource::Local => {
            if !is_known_local_model(&ai.model) {
                return Err(format!(
                    "Model '{}' is not in the supported local list.",
                    ai.model.trim()
                ));
            }
        }
        AiSource::OpenRouter => {
            if !is_known_openrouter_model(&ai.model) {
                return Err(format!(
                    "Model '{}' is not in the supported OpenRouter list.",
                    ai.model.trim()
                ));
            }
        }
        AiSource::Custom => {}
    }
    Ok(())
}

/// Build a ready-to-chat provider from Settings.
pub fn provider_from_ai(ai: &AiSettings) -> Result<OpenAiProvider, String> {
    validate_ai_settings(ai)?;
    let api_key = resolve_api_key(ai.api_key.as_deref(), ai.api_key_env.as_deref())?;
    match ai.source {
        AiSource::Local => Ok(OpenAiProvider::new(
            ai.base_url.trim(),
            ai.model.trim(),
            api_key,
            OpenAiProtocol::ChatCompletions,
            ReasoningSummary::Off,
        )),
        AiSource::OpenRouter => {
            let key = api_key.ok_or_else(|| {
                "OpenRouter requires an API key (paste it in Settings or set api_key_env)."
                    .to_string()
            })?;
            let model = ai.model.trim();
            // Ensure the catalog entry exists (validate already checked).
            let _ = openrouter_model(model);
            Ok(OpenAiProvider::with_extras(
                OPENROUTER_BASE_URL,
                model,
                Some(key),
                OpenAiProtocol::ChatCompletions,
                ReasoningSummary::Off,
                openrouter_compat_extras(model),
            ))
        }
        AiSource::Custom => {
            let protocol = match ai.api_protocol {
                AiApiProtocol::ChatCompletions => OpenAiProtocol::ChatCompletions,
                AiApiProtocol::Responses => OpenAiProtocol::Responses,
            };
            let reasoning_summary = match ai.reasoning_summary {
                SettingsSummary::Auto => ReasoningSummary::Auto,
                SettingsSummary::Off => ReasoningSummary::Off,
            };
            Ok(OpenAiProvider::with_extras(
                ai.base_url.trim(),
                ai.model.trim(),
                api_key,
                protocol,
                reasoning_summary,
                OpenAiCompatExtras::default(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_settings::AiSettings;

    #[test]
    fn literal_key_passes_through() {
        assert_eq!(
            resolve_api_key(Some("sk-literal"), None),
            Ok(Some("sk-literal".into()))
        );
        assert_eq!(resolve_api_key(None, None), Ok(None));
    }

    #[test]
    fn env_indirection_wins_over_literal_and_errors_when_unset() {
        // An env var that is (almost certainly) unset.
        let err =
            resolve_api_key(Some("ignored"), Some("CUTLASS_TEST_KEY_THAT_IS_UNSET")).unwrap_err();
        assert!(err.contains("unset"), "{err}");

        // SAFETY: single-threaded test; restored immediately after.
        unsafe { std::env::set_var("CUTLASS_TEST_KEY_PRESENT", "sk-from-env") };
        assert_eq!(
            resolve_api_key(Some("ignored"), Some("CUTLASS_TEST_KEY_PRESENT")),
            Ok(Some("sk-from-env".into()))
        );
        unsafe { std::env::remove_var("CUTLASS_TEST_KEY_PRESENT") };
    }

    #[test]
    fn validate_rejects_unknown_local_model() {
        let ai = AiSettings {
            source: AiSource::Local,
            base_url: "http://localhost:11434/v1".into(),
            model: "not-a-real-model".into(),
            ..AiSettings::default()
        };
        assert!(
            validate_ai_settings(&ai)
                .unwrap_err()
                .contains("supported local")
        );
    }

    #[test]
    fn openrouter_requires_known_slug_and_key() {
        let ai = AiSettings {
            source: AiSource::OpenRouter,
            model: "openai/gpt-5.6-sol".into(),
            api_key: Some("sk-or".into()),
            ..AiSettings::default()
        };
        assert!(provider_from_ai(&ai).is_ok());

        let bad = AiSettings {
            source: AiSource::OpenRouter,
            model: "openai/gpt-5.6-sol".into(),
            ..AiSettings::default()
        };
        assert!(validate_ai_settings(&bad).is_err());
    }
}
