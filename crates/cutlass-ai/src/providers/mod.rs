//! Provider implementations behind the [`crate::provider::ChatProvider`] seam.

pub mod openai_compat;
pub mod scripted;

pub use openai_compat::OpenAiCompatProvider;
pub use scripted::ScriptedProvider;
