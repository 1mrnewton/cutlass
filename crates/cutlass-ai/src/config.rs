//! Provider configuration: `~/.cutlass/config.toml`, `[ai]` table.
//!
//! Keys never live in project files. An absent file or absent `[ai]`
//! table means "not configured" — a state the UI surfaces with setup
//! instructions, never an error dialog.
//!
//! ```toml
//! [ai]
//! base_url = "http://localhost:11434/v1"   # Ollama
//! model = "qwen3:14b"
//! # api_key = "sk-..."          # literal key, or:
//! # api_key_env = "OPENAI_API_KEY"  # read from the environment
//! ```

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// The `[ai]` table of `config.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct AiSection {
    /// OpenAI-compatible endpoint root, e.g. `http://localhost:11434/v1`.
    pub base_url: String,
    /// Model name as the endpoint knows it, e.g. `qwen3:14b` or `gpt-4o`.
    pub model: String,
    /// Literal API key. Local servers usually need none.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Name of an environment variable holding the key (preferred over a
    /// literal for cloud providers).
    #[serde(default)]
    pub api_key_env: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ConfigFile {
    #[serde(default)]
    ai: Option<AiSection>,
}

impl AiSection {
    /// The key to send, resolving `api_key_env` if set. `Ok(None)` means
    /// no key (fine for local servers); `Err` names what is missing.
    pub fn resolve_api_key(&self) -> Result<Option<String>, String> {
        if let Some(var) = &self.api_key_env {
            return match std::env::var(var) {
                Ok(key) if !key.is_empty() => Ok(Some(key)),
                _ => Err(format!(
                    "api_key_env points at '{var}' but that environment variable is unset"
                )),
            };
        }
        Ok(self.api_key.clone())
    }
}

/// `~/.cutlass/config.toml` — the user's home dir on every platform
/// (`C:\Users\<name>\.cutlass\config.toml` on Windows, where `HOME` is unset).
/// Falls back to the temp dir only if the home dir can't be resolved; never
/// the working directory, which is the read-only install folder on Windows.
pub fn default_config_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join(".cutlass")
        .join("config.toml")
}

/// Load the `[ai]` section from `path`. `Ok(None)` = not configured
/// (missing file or missing table); `Err` = the file exists but is broken,
/// with a message naming the problem.
pub fn load_ai_config(path: &Path) -> Result<Option<AiSection>, String> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("could not read {}: {e}", path.display())),
    };
    let parsed: ConfigFile =
        toml::from_str(&raw).map_err(|e| format!("could not parse {}: {e}", path.display()))?;
    Ok(parsed.ai)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_is_not_configured() {
        assert_eq!(
            load_ai_config(Path::new("/nonexistent/config.toml")),
            Ok(None)
        );
    }

    #[test]
    fn parses_ai_section_and_tolerates_unknown_tables() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[editor]
something_else = true

[ai]
base_url = "http://localhost:11434/v1"
model = "qwen3:14b"
"#,
        )
        .unwrap();

        let section = load_ai_config(&path).unwrap().unwrap();
        assert_eq!(section.base_url, "http://localhost:11434/v1");
        assert_eq!(section.model, "qwen3:14b");
        assert_eq!(section.api_key, None);
        assert_eq!(section.resolve_api_key(), Ok(None));
    }

    #[test]
    fn missing_ai_table_is_not_configured_and_broken_toml_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        std::fs::write(&path, "[editor]\nx = 1\n").unwrap();
        assert_eq!(load_ai_config(&path), Ok(None));

        std::fs::write(&path, "[ai]\nbase_url = \n").unwrap();
        assert!(
            load_ai_config(&path)
                .unwrap_err()
                .contains("could not parse")
        );
    }

    #[test]
    fn api_key_env_resolution() {
        let section = AiSection {
            base_url: "https://api.openai.com/v1".into(),
            model: "gpt-4o".into(),
            api_key: Some("ignored-when-env-set".into()),
            api_key_env: Some("CUTLASS_TEST_KEY_THAT_IS_UNSET".into()),
        };
        assert!(section.resolve_api_key().unwrap_err().contains("unset"));

        let literal = AiSection {
            api_key_env: None,
            ..section
        };
        assert_eq!(
            literal.resolve_api_key(),
            Ok(Some("ignored-when-env-set".into()))
        );
    }
}
