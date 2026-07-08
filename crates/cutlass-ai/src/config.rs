//! API-key resolution for the AI provider.
//!
//! The config *file* (`~/.cutlass/config.toml`) is owned by the
//! `cutlass-settings` crate — its `[ai]` table parses into
//! `cutlass_settings::AiSettings`. The one piece that stays here is
//! key *resolution*: the `api_key_env` indirection that lets cloud keys live
//! in the environment instead of on disk. That's an AI-domain concern (the
//! provider is the only thing that needs the actual secret), so it does not
//! belong in the settings model.

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
