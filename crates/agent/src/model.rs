//! The model trait, the local runner that ships, and the cloud runner.
//!
//! The local runner is a scripted double: no weights and no HTTP. The cloud
//! runner stays off until settings enable it. A flag from the model does not
//! turn it on.

use std::fmt;

use cutlass_session::Request;
use cutlass_settings::Settings;
use serde::{Deserialize, Serialize};

use crate::picture::Picture;

/// A tool the model is allowed to call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolDefinition {
    /// The tool name.
    pub name: String,
    /// What the tool does.
    pub description: String,
}

/// One call the model asked the harness to run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Caller-chosen id for this call.
    pub id: String,
    /// The tool to run.
    pub name: String,
    /// Opaque arguments. There is no approved flag in this type.
    pub arguments: String,
}

/// One item in a model transcript. It is saved as JSON with a `role`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    /// Text from the user.
    User {
        /// The user's text.
        text: String,
    },
    /// Text and tool calls from the model.
    Assistant {
        /// The model's text.
        text: String,
        /// Tool calls to run, in order.
        calls: Vec<ToolCall>,
    },
    /// The result of one tool call.
    Tool {
        /// The id from the tool call.
        call_id: String,
        /// The tool name from the tool call.
        name: String,
        /// The tool's text.
        content: String,
        /// Whether the tool failed.
        is_error: bool,
        /// A picture the tool handed the model with its text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        picture: Option<Picture>,
    },
}

/// Shorten the tool results in `messages` that are longer than `limit`
/// to their start and a note. A host does this to the turns before the one
/// it is about to run: what a tool read then has been used, and sending
/// every old listing and transcript again with each round fills the
/// model's context.
pub fn shorten_results(messages: &mut [Message], limit: usize) {
    for message in messages {
        let Message::Tool { content, .. } = message else {
            continue;
        };
        if content.len() <= limit {
            continue;
        }
        let mut end = limit;
        while !content.is_char_boundary(end) {
            end -= 1;
        }
        let whole = content.len();
        content.truncate(end);
        content.push_str(&format!(
            "\n(shortened: {whole} characters when it was read; read it again if it is needed)"
        ));
    }
}

/// Where what a model is sent of `messages` begins, for the earlier turns
/// of a conversation to take no more than `budget` characters: the start
/// of the oldest turn from which the rest fits. A turn is a user's ask and
/// everything that answered it, and is sent whole or not at all, so no
/// tool call is sent without its result. 0 sends every message, and
/// `messages.len()` none, when the newest turn alone is past the budget.
///
/// A host does this to the turns before the one it is about to run. A
/// conversation that is only ever added to outgrows the model's context,
/// and from then on every ask in it fails.
#[must_use]
pub fn first_sent(messages: &[Message], budget: usize) -> usize {
    let mut from = messages.len();
    let mut used = 0usize;
    let mut turn = 0usize;
    for (index, message) in messages.iter().enumerate().rev() {
        turn = turn.saturating_add(size_of(message));
        // A turn starts at a user's words, and the first message starts
        // one whoever said it.
        if index > 0 && !matches!(message, Message::User { .. }) {
            continue;
        }
        used = used.saturating_add(turn);
        if used > budget {
            break;
        }
        from = index;
        turn = 0;
    }
    from
}

/// The characters `message` adds to a request.
fn size_of(message: &Message) -> usize {
    match message {
        Message::User { text } => text.len(),
        Message::Assistant { text, calls } => calls
            .iter()
            .map(|call| call.name.len() + call.arguments.len())
            .fold(text.len(), usize::saturating_add),
        Message::Tool {
            content, picture, ..
        } => content.len() + picture.as_ref().map_or(0, |picture| picture.data.len()),
    }
}

/// Messages and tool definitions sent to a model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionRequest {
    /// The transcript so far.
    pub messages: Vec<Message>,
    /// The tools the model may call.
    pub tools: Vec<ToolDefinition>,
}

/// The model's text, tool calls, and proposed session requests.
#[derive(Clone, Debug, PartialEq)]
pub struct Completion {
    /// The model's text.
    pub text: String,
    /// Tool calls to run, in order.
    pub calls: Vec<ToolCall>,
    /// A child loop may return session requests here. The parent turn does not apply them.
    pub proposed: Vec<Request>,
}

/// Why a completion did not return.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelError {
    /// The cloud runner is off.
    CloudDisabled,
    /// The scripted runner has no completion left.
    NoMoreSteps,
    /// The runner failed with this message.
    Inner(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CloudDisabled => write!(formatter, "the cloud model is off"),
            Self::NoMoreSteps => write!(formatter, "the scripted model has no more steps"),
            Self::Inner(message) => write!(formatter, "{message}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// Turns a completion request into text and tool calls.
pub trait Model {
    /// Returns the next completion for this request.
    fn complete(&mut self, request: &CompletionRequest) -> Result<Completion, ModelError>;
}

/// Scripted local runner with no weights and no HTTP.
#[derive(Debug)]
pub struct LocalModel {
    steps: Vec<Completion>,
    cursor: usize,
    seen: Vec<CompletionRequest>,
}

impl LocalModel {
    /// Returns a runner that yields `steps` in order and then stops.
    #[must_use]
    pub fn scripted(steps: Vec<Completion>) -> Self {
        Self {
            steps,
            cursor: 0,
            seen: Vec::new(),
        }
    }

    /// Returns the requests that produced a scripted completion.
    #[must_use]
    pub fn seen(&self) -> &[CompletionRequest] {
        &self.seen
    }
}

impl Model for LocalModel {
    fn complete(&mut self, request: &CompletionRequest) -> Result<Completion, ModelError> {
        let Some(step) = self.steps.get(self.cursor) else {
            return Err(ModelError::NoMoreSteps);
        };
        let completion = step.clone();
        self.cursor += 1;
        self.seen.push(request.clone());
        Ok(completion)
    }
}

/// Cloud runner that stays off until settings enable it.
#[derive(Debug)]
pub struct CloudModel<M> {
    enabled: bool,
    inner: M,
    ignored_model_flag: bool,
}

impl<M> CloudModel<M> {
    /// Enables the runner from `settings.model.cloud`.
    #[must_use]
    pub fn from_settings(settings: &Settings, inner: M) -> Self {
        Self {
            enabled: settings.model.cloud,
            inner,
            ignored_model_flag: false,
        }
    }

    /// Records a true model-emitted enable flag without changing whether the runner is enabled.
    pub fn consider_model_flag(&mut self, model_says_enabled: bool) {
        if model_says_enabled {
            self.ignored_model_flag = true;
        }
    }

    /// Reports whether `settings.model.cloud` turned this runner on.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Reports whether a true enable flag from the model was ignored.
    #[must_use]
    pub fn ignored_model_flag(&self) -> bool {
        self.ignored_model_flag
    }
}

impl<M: Model> Model for CloudModel<M> {
    fn complete(&mut self, request: &CompletionRequest) -> Result<Completion, ModelError> {
        if !self.enabled {
            return Err(ModelError::CloudDisabled);
        }
        self.inner.complete(request)
    }
}

#[cfg(test)]
mod history_tests {
    use super::{Message, ToolCall, first_sent};

    fn user(text: &str) -> Message {
        Message::User {
            text: text.to_owned(),
        }
    }

    fn said(text: &str, calls: &[&str]) -> Message {
        Message::Assistant {
            text: text.to_owned(),
            calls: calls
                .iter()
                .map(|id| ToolCall {
                    id: (*id).to_owned(),
                    name: "t".to_owned(),
                    arguments: "ab".to_owned(),
                })
                .collect(),
        }
    }

    fn result(id: &str, content: &str) -> Message {
        Message::Tool {
            call_id: id.to_owned(),
            name: "t".to_owned(),
            content: content.to_owned(),
            is_error: false,
            picture: None,
        }
    }

    #[test]
    fn the_oldest_turns_are_left_out_whole_once_a_conversation_is_past_its_budget() {
        // Three turns of 10, 16 and 8 characters.
        let messages = vec![
            user("12345"),
            said("12345", &[]),
            user("1234"),
            said("123", &["c1"]),
            result("c1", "1234"),
            said("12", &[]),
            user("1234"),
            said("1234", &[]),
        ];
        assert!(first_sent(&[], 0) == 0 && first_sent(&[], 100) == 0);
        // All of it fits.
        assert_eq!(first_sent(&messages, 34), 0);
        assert_eq!(first_sent(&messages, usize::MAX), 0);
        // One character short: the first turn goes, whole.
        assert_eq!(first_sent(&messages, 33), 2);
        assert_eq!(first_sent(&messages, 24), 2);
        // The middle turn is never cut between a call and its result.
        assert_eq!(first_sent(&messages, 23), 6);
        assert_eq!(first_sent(&messages, 8), 6);
        // Not even the newest fits: none of it is sent.
        assert_eq!(first_sent(&messages, 7), messages.len());
        assert_eq!(first_sent(&messages, 0), messages.len());
        // What is sent always begins at a user's words.
        for budget in 0..40 {
            let from = first_sent(&messages, budget);
            assert!(
                from == messages.len() || matches!(messages[from], Message::User { .. }),
                "{budget}"
            );
        }
        // A conversation that begins with a reply is one turn from there.
        let headless = vec![said("12345", &[]), user("12"), said("12", &[])];
        assert_eq!(first_sent(&headless, 9), 0);
        assert_eq!(first_sent(&headless, 8), 1);
    }
}

#[cfg(test)]
mod tests {
    use cutlass_settings::Settings;

    use super::{
        CloudModel, Completion, CompletionRequest, LocalModel, Message, Model, ModelError,
        ToolCall, ToolDefinition,
    };

    fn user_request(text: &str) -> CompletionRequest {
        CompletionRequest {
            messages: vec![Message::User {
                text: text.to_owned(),
            }],
            tools: vec![ToolDefinition {
                name: "draft.query".to_owned(),
                description: "Read the open draft.".to_owned(),
            }],
        }
    }

    fn text_completion(text: &str) -> Completion {
        Completion {
            text: text.to_owned(),
            calls: Vec::new(),
            proposed: Vec::new(),
        }
    }

    #[test]
    fn old_tool_results_are_shortened_and_the_rest_is_left() {
        let tool = |content: &str| Message::Tool {
            call_id: "1".to_owned(),
            name: "draft.query".to_owned(),
            content: content.to_owned(),
            is_error: false,
            picture: None,
        };
        let mut messages = vec![
            Message::User {
                text: "y".repeat(500),
            },
            tool(&"é".repeat(300)),
            tool("short"),
        ];
        super::shorten_results(&mut messages, 101);
        assert!(matches!(&messages[0], Message::User { text } if text.len() == 500));
        let Message::Tool { content, .. } = &messages[1] else {
            panic!("not a tool result");
        };
        assert!(content.starts_with(&"é".repeat(50)), "{content}");
        assert!(content.contains("(shortened: 600 characters"), "{content}");
        assert!(content.len() < 250);
        assert_eq!(messages[2], tool("short"));
    }

    #[test]
    fn local_model_returns_scripted_tool_calls() {
        let call = ToolCall {
            id: "1".to_owned(),
            name: "draft.query".to_owned(),
            arguments: "project".to_owned(),
        };
        let completion = Completion {
            text: "looking".to_owned(),
            calls: vec![call.clone()],
            proposed: Vec::new(),
        };
        let mut model = LocalModel::scripted(vec![completion.clone()]);
        let got = model.complete(&user_request("hello")).unwrap();
        assert_eq!(got, completion);
        assert_eq!(got.calls, vec![call]);
    }

    #[test]
    fn local_model_records_the_request() {
        let request = user_request("hello");
        let mut model = LocalModel::scripted(vec![text_completion("ok")]);
        model.complete(&request).unwrap();
        assert_eq!(model.seen(), &[request]);
    }

    #[test]
    fn local_model_ends_when_the_script_ends() {
        let request = user_request("hello");
        let mut empty = LocalModel::scripted(Vec::new());
        let error = empty.complete(&request).unwrap_err();
        assert_eq!(error, ModelError::NoMoreSteps);
        assert!(empty.seen().is_empty());

        let mut model = LocalModel::scripted(vec![text_completion("only")]);
        assert_eq!(model.complete(&request).unwrap().text, "only");
        let again = user_request("again");
        let error = model.complete(&again).unwrap_err();
        assert_eq!(error, ModelError::NoMoreSteps);
        assert_eq!(model.seen(), &[request]);
    }

    #[test]
    fn cloud_model_stays_off_by_default() {
        let settings = Settings::default();
        assert!(!settings.model.cloud);
        let mut model =
            CloudModel::from_settings(&settings, LocalModel::scripted(vec![text_completion("ok")]));
        assert!(!model.is_enabled());
        let error = model.complete(&user_request("hello")).unwrap_err();
        assert_eq!(error, ModelError::CloudDisabled);
        assert!(error.to_string().contains("cloud"));
        assert!(model.inner.seen().is_empty());
    }

    #[test]
    fn cloud_model_runs_when_settings_enable_it() {
        let mut settings = Settings::default();
        settings.model.cloud = true;
        let completion = text_completion("ok");
        let mut model =
            CloudModel::from_settings(&settings, LocalModel::scripted(vec![completion.clone()]));
        assert!(model.is_enabled());
        let request = user_request("hello");
        let got = model.complete(&request).unwrap();
        assert_eq!(got.text, "ok");
        assert_eq!(got, completion);
        assert_eq!(model.inner.seen(), &[request]);
    }

    #[test]
    fn model_flag_does_not_enable_the_cloud_runner() {
        let mut model = CloudModel::from_settings(
            &Settings::default(),
            LocalModel::scripted(vec![text_completion("ok")]),
        );
        assert!(!model.ignored_model_flag());
        model.consider_model_flag(true);
        assert!(!model.is_enabled());
        assert!(model.ignored_model_flag());
        model.consider_model_flag(false);
        assert!(model.ignored_model_flag());
        assert!(!model.is_enabled());
        let error = model.complete(&user_request("hello")).unwrap_err();
        assert_eq!(error, ModelError::CloudDisabled);
        assert!(model.inner.seen().is_empty());
    }

    #[test]
    fn cloud_switch_does_not_read_the_sandbox_switch() {
        let mut settings = Settings::default();
        settings.model.cloud = false;
        settings.sandbox.online = true;
        assert!(!settings.model.cloud);
        assert!(settings.sandbox.online);
        let mut model =
            CloudModel::from_settings(&settings, LocalModel::scripted(vec![text_completion("ok")]));
        assert!(!model.is_enabled());
        let error = model.complete(&user_request("hello")).unwrap_err();
        assert_eq!(error, ModelError::CloudDisabled);
        assert!(model.inner.seen().is_empty());
    }
}
