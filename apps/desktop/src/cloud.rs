//! Where the desktop keeps the cloud model's keys; the model itself is in
//! `cutlass-agent-cloud`.

use std::path::PathBuf;

pub use cutlass_agent_cloud::{Cloud, key, model, set_key};

/// Where the user's keys are kept, by provider.
pub fn keys_file() -> PathBuf {
    crate::shell::Shell::home().assistant_keys_file()
}
