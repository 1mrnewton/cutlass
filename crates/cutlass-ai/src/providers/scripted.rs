//! Deterministic provider double: canned turns, no network.
//!
//! The substrate for agent-loop tests and the eval harness — scripted
//! prompts run against a real engine in CI without a live model.

use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

use crate::provider::{
    ChatProvider, ChatRequest, ChatTurn, Message, ProviderError, ProviderStreamEvent,
};

pub struct ScriptedProvider {
    turns: Mutex<std::vec::IntoIter<ChatTurn>>,
    /// Every request's messages, recorded for assertions.
    requests: Mutex<Vec<Vec<Message>>>,
    /// Tool names offered on each request, recorded for assertions.
    tools: Mutex<Vec<Vec<String>>>,
    /// `session_id` offered on each request, recorded for assertions.
    session_ids: Mutex<Vec<Option<String>>>,
}

impl ScriptedProvider {
    pub fn new(turns: Vec<ChatTurn>) -> Self {
        Self {
            turns: Mutex::new(turns.into_iter()),
            requests: Mutex::new(Vec::new()),
            tools: Mutex::new(Vec::new()),
            session_ids: Mutex::new(Vec::new()),
        }
    }

    /// The message histories this provider was called with, in order.
    pub fn requests(&self) -> Vec<Vec<Message>> {
        self.requests.lock().unwrap().clone()
    }

    /// The tool names offered on each request, in order.
    pub fn tool_names(&self) -> Vec<Vec<String>> {
        self.tools.lock().unwrap().clone()
    }

    /// The `session_id` values offered on each request, in order.
    pub fn session_ids(&self) -> Vec<Option<String>> {
        self.session_ids.lock().unwrap().clone()
    }
}

impl ChatProvider for ScriptedProvider {
    fn chat(
        &self,
        request: &ChatRequest<'_>,
        _cancel: &AtomicBool,
        on_event: &mut dyn FnMut(ProviderStreamEvent<'_>),
    ) -> Result<ChatTurn, ProviderError> {
        self.requests
            .lock()
            .unwrap()
            .push(request.messages.to_vec());
        self.tools
            .lock()
            .unwrap()
            .push(request.tools.iter().map(|t| t.name.to_string()).collect());
        self.session_ids
            .lock()
            .unwrap()
            .push(request.session_id.map(str::to_owned));
        let turn = self.turns.lock().unwrap().next().ok_or_else(|| {
            ProviderError::Protocol("scripted provider ran out of turns".to_string())
        })?;
        if !turn.reasoning_summary.is_empty() {
            on_event(ProviderStreamEvent::ReasoningSummaryDelta(
                &turn.reasoning_summary,
            ));
        }
        if !turn.text.is_empty() {
            on_event(ProviderStreamEvent::TextDelta(&turn.text));
        }
        Ok(turn)
    }
}
