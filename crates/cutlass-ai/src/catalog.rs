//! Curated model allowlists for Local and OpenRouter.
//!
//! The Settings picker and agent routing both read from these tables.
//! Adding or removing a model is one change here — never the full
//! OpenRouter catalog.

/// Fixed OpenRouter Chat Completions root.
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Attribution headers OpenRouter asks apps to send.
pub const OPENROUTER_HTTP_REFERER: &str = "https://cutlass.sh";
pub const OPENROUTER_APP_TITLE: &str = "Cutlass";

/// Default local model (Ollama-style id).
pub const DEFAULT_LOCAL_MODEL: &str = "qwen3:14b";

/// Default OpenRouter cloud model slug.
pub const DEFAULT_OPENROUTER_MODEL: &str = "openai/gpt-5.6-sol";

/// One curated local model. `aliases` cover Ollama vs LM Studio naming;
/// `id` is the preferred persist form (usually the Ollama tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalModel {
    pub id: &'static str,
    pub display: &'static str,
    pub role: &'static str,
    pub aliases: &'static [&'static str],
}

/// OpenRouter upstream pin for open-weight models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderPin {
    pub order: &'static [&'static str],
    pub allow_fallbacks: bool,
}

/// One curated OpenRouter model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenRouterModel {
    pub id: &'static str,
    pub display: &'static str,
    pub vendor: &'static str,
    pub role: &'static str,
    pub pin: Option<ProviderPin>,
}

/// Local allowlist — the entire Local picker.
pub const LOCAL_MODELS: &[LocalModel] = &[
    LocalModel {
        id: "qwen3:14b",
        display: "Qwen3 14B",
        role: "Default local agent",
        aliases: &["qwen3:14b", "qwen3-14b"],
    },
    LocalModel {
        id: "qwen3:32b",
        display: "Qwen3 32B",
        role: "Stronger local quality",
        aliases: &["qwen3:32b", "qwen3-32b"],
    },
    LocalModel {
        id: "llama3.3:70b",
        display: "Llama 3.3 70B",
        role: "High-end local (needs RAM/GPU)",
        aliases: &["llama3.3:70b", "llama-3.3-70b"],
    },
    LocalModel {
        id: "gpt-oss:20b",
        display: "GPT-OSS 20B",
        role: "Compact open reasoning",
        aliases: &["gpt-oss:20b", "gpt-oss-20b"],
    },
    LocalModel {
        id: "gpt-oss:120b",
        display: "GPT-OSS 120B",
        role: "Best open local (heavy)",
        aliases: &["gpt-oss:120b", "gpt-oss-120b"],
    },
];

const PIN_GROQ: ProviderPin = ProviderPin {
    order: &["groq"],
    allow_fallbacks: false,
};

const PIN_MOONSHOT: ProviderPin = ProviderPin {
    order: &["moonshotai"],
    allow_fallbacks: false,
};

const PIN_ZAI: ProviderPin = ProviderPin {
    order: &["z-ai"],
    allow_fallbacks: false,
};

/// OpenRouter allowlist — the entire cloud picker, grouped by `vendor`.
pub const OPENROUTER_MODELS: &[OpenRouterModel] = &[
    OpenRouterModel {
        id: "openai/gpt-5.5",
        display: "GPT-5.5",
        vendor: "OpenAI",
        role: "Frontier (prior gen)",
        pin: None,
    },
    OpenRouterModel {
        id: "openai/gpt-5.6-sol",
        display: "GPT-5.6",
        vendor: "OpenAI",
        role: "Cloud default",
        pin: None,
    },
    OpenRouterModel {
        id: "google/gemini-3.5-flash",
        display: "Gemini 3.5 Flash",
        vendor: "Google",
        role: "Fast agent / coding",
        pin: None,
    },
    OpenRouterModel {
        id: "google/gemini-3.1-pro-preview",
        display: "Gemini 3.1 Pro",
        vendor: "Google",
        role: "Strongest reasoning + vision",
        pin: None,
    },
    OpenRouterModel {
        id: "google/gemini-3-flash-preview",
        display: "Gemini 3 Flash",
        vendor: "Google",
        role: "Speed + grounding",
        pin: None,
    },
    OpenRouterModel {
        id: "anthropic/claude-sonnet-4",
        display: "Claude Sonnet 4",
        vendor: "Anthropic",
        role: "Workhorse (4.x line)",
        pin: None,
    },
    OpenRouterModel {
        id: "anthropic/claude-sonnet-4.5",
        display: "Claude Sonnet 4.5",
        vendor: "Anthropic",
        role: "Mid Sonnet",
        pin: None,
    },
    OpenRouterModel {
        id: "anthropic/claude-sonnet-4.6",
        display: "Claude Sonnet 4.6",
        vendor: "Anthropic",
        role: "Latest Sonnet 4.x",
        pin: None,
    },
    OpenRouterModel {
        id: "anthropic/claude-opus-4.8",
        display: "Claude Opus 4.8",
        vendor: "Anthropic",
        role: "Deep agentic / coding",
        pin: None,
    },
    OpenRouterModel {
        id: "anthropic/claude-fable-5",
        display: "Claude Fable 5",
        vendor: "Anthropic",
        role: "Best agent quality",
        pin: None,
    },
    OpenRouterModel {
        id: "x-ai/grok-4.5",
        display: "Grok 4.5",
        vendor: "xAI",
        role: "Flagship coding / agents",
        pin: None,
    },
    OpenRouterModel {
        id: "x-ai/grok-4.3",
        display: "Grok 4.3",
        vendor: "xAI",
        role: "General, large context",
        pin: None,
    },
    OpenRouterModel {
        id: "x-ai/grok-4.20",
        display: "Grok 4.20 Reasoning",
        vendor: "xAI",
        role: "Fast agentic tool calling",
        pin: None,
    },
    OpenRouterModel {
        id: "moonshotai/kimi-k3",
        display: "Kimi K3",
        vendor: "Kimi",
        role: "Vision + 1M context",
        pin: Some(PIN_MOONSHOT),
    },
    OpenRouterModel {
        id: "moonshotai/kimi-k2.6",
        display: "Kimi K2.6",
        vendor: "Kimi",
        role: "General agent / multimodal",
        pin: Some(PIN_MOONSHOT),
    },
    OpenRouterModel {
        id: "moonshotai/kimi-k2.7-code",
        display: "Kimi K2.7 Code",
        vendor: "Kimi",
        role: "Coding specialist",
        pin: Some(PIN_MOONSHOT),
    },
    OpenRouterModel {
        id: "moonshotai/kimi-k2.7-code-highspeed",
        display: "Kimi K2.7 Code Highspeed",
        vendor: "Kimi",
        role: "Fast coding variant",
        pin: Some(PIN_MOONSHOT),
    },
    OpenRouterModel {
        id: "z-ai/glm-5",
        display: "GLM-5",
        vendor: "GLM",
        role: "Flagship agentic",
        pin: Some(PIN_ZAI),
    },
    OpenRouterModel {
        id: "z-ai/glm-5.1",
        display: "GLM-5.1",
        vendor: "GLM",
        role: "Refined flagship",
        pin: Some(PIN_ZAI),
    },
    OpenRouterModel {
        id: "z-ai/glm-5.2",
        display: "GLM-5.2",
        vendor: "GLM",
        role: "Latest coding / 1M ctx",
        pin: Some(PIN_ZAI),
    },
    OpenRouterModel {
        id: "z-ai/glm-4.5",
        display: "GLM-4.5",
        vendor: "GLM",
        role: "Prior open workhorse",
        pin: Some(PIN_ZAI),
    },
    OpenRouterModel {
        id: "openai/gpt-oss-120b",
        display: "GPT-OSS 120B",
        vendor: "Open weight",
        role: "Fast/cheap iteration (Groq)",
        pin: Some(PIN_GROQ),
    },
    OpenRouterModel {
        id: "meta-llama/llama-3.3-70b-instruct",
        display: "Llama 3.3 70B",
        vendor: "Open weight",
        role: "Reliable general + tools",
        pin: Some(PIN_GROQ),
    },
];

/// Look up a local catalog entry by persisted id or any alias.
pub fn local_model(id: &str) -> Option<&'static LocalModel> {
    let id = id.trim();
    LOCAL_MODELS
        .iter()
        .find(|m| m.id == id || m.aliases.contains(&id))
}

/// True when `id` is in the local allowlist (canonical or alias).
pub fn is_known_local_model(id: &str) -> bool {
    local_model(id).is_some()
}

/// Look up an OpenRouter catalog entry by slug.
pub fn openrouter_model(id: &str) -> Option<&'static OpenRouterModel> {
    let id = id.trim();
    OPENROUTER_MODELS.iter().find(|m| m.id == id)
}

/// True when `id` is in the OpenRouter allowlist.
pub fn is_known_openrouter_model(id: &str) -> bool {
    openrouter_model(id).is_some()
}

/// Match a curated local entry against ids returned by `GET /v1/models`.
/// Returns the installed id to send on chat requests (prefer exact alias hit).
pub fn resolve_local_installed_id(catalog_id: &str, installed: &[String]) -> Option<String> {
    let entry = local_model(catalog_id)?;
    for alias in entry.aliases {
        if let Some(hit) = installed.iter().find(|id| id_matches(id, alias)) {
            return Some(hit.clone());
        }
    }
    // Some servers prefix org paths; also try suffix / contains on bare names.
    installed.iter().find_map(|id| {
        entry
            .aliases
            .iter()
            .find(|alias| id_matches(id, alias))
            .map(|_| id.clone())
    })
}

fn id_matches(installed: &str, alias: &str) -> bool {
    let installed = installed.trim();
    if installed == alias {
        return true;
    }
    // LM Studio sometimes returns `vendor/model` or `:tag` suffixes.
    installed.ends_with(&format!("/{alias}"))
        || installed.ends_with(&format!(":{alias}"))
        || installed
            .strip_prefix(alias)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with(':') || rest.starts_with('-'))
}

/// Which curated local models are present among `installed` ids.
pub fn local_models_availability(
    installed: &[String],
) -> Vec<(&'static LocalModel, bool, Option<String>)> {
    LOCAL_MODELS
        .iter()
        .map(|entry| {
            let resolved = resolve_local_installed_id(entry.id, installed);
            let available = resolved.is_some();
            (entry, available, resolved)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_aliases_resolve_ollama_and_lmstudio() {
        let installed = vec!["qwen3-14b".into(), "other".into()];
        assert_eq!(
            resolve_local_installed_id("qwen3:14b", &installed).as_deref(),
            Some("qwen3-14b")
        );
        assert!(is_known_local_model("qwen3-14b"));
        assert_eq!(local_model("qwen3-14b").unwrap().id, "qwen3:14b");
    }

    #[test]
    fn openrouter_pin_present_for_open_weight() {
        let m = openrouter_model("openai/gpt-oss-120b").unwrap();
        let pin = m.pin.expect("pin");
        assert_eq!(pin.order, &["groq"]);
        assert!(!pin.allow_fallbacks);
    }

    #[test]
    fn defaults_are_in_allowlists() {
        assert!(is_known_local_model(DEFAULT_LOCAL_MODEL));
        assert!(is_known_openrouter_model(DEFAULT_OPENROUTER_MODEL));
    }
}
