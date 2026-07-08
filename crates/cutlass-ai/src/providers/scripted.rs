//! Deterministic provider double: canned turns, no network.
//!
//! The substrate for agent-loop tests and the eval harness — scripted
//! prompts run against a real engine in CI without a live model.

use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

use crate::provider::{ChatProvider, ChatRequest, ChatTurn, Message, ProviderError};

pub struct ScriptedProvider {
    turns: Mutex<std::vec::IntoIter<ChatTurn>>,
    /// Every request's messages, recorded for assertions.
    requests: Mutex<Vec<Vec<Message>>>,
}

impl ScriptedProvider {
    pub fn new(turns: Vec<ChatTurn>) -> Self {
        Self {
            turns: Mutex::new(turns.into_iter()),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// The message histories this provider was called with, in order.
    pub fn requests(&self) -> Vec<Vec<Message>> {
        self.requests.lock().unwrap().clone()
    }
}

impl ChatProvider for ScriptedProvider {
    fn chat(
        &self,
        request: &ChatRequest<'_>,
        _cancel: &AtomicBool,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<ChatTurn, ProviderError> {
        self.requests
            .lock()
            .unwrap()
            .push(request.messages.to_vec());
        let turn = self.turns.lock().unwrap().next().ok_or_else(|| {
            ProviderError::Protocol("scripted provider ran out of turns".to_string())
        })?;
        if !turn.text.is_empty() {
            on_text(&turn.text);
        }
        Ok(turn)
    }
}
