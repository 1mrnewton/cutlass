//! One ask, then model rounds, with tool results fed back to the model.
//!
//! Each model completion is a round. The next round is refused once the step
//! cap is reached. A tool error stays in the transcript so the model can try
//! another tool. Proposed session requests stay on the completion and are
//! not applied here.

use std::fmt;

use cutlass_settings::Settings;

use crate::model::{
    Completion, CompletionRequest, Message, Model, ModelError, ToolCall, ToolDefinition,
};

/// The text a tool returned to the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolOutcome {
    /// The tool's text.
    pub content: String,
    /// Whether the tool failed.
    pub is_error: bool,
}

/// Runs one tool call and returns the text the model should see.
pub trait ToolSink {
    /// Runs `call` and returns its outcome.
    fn run_tool(&mut self, call: &ToolCall) -> ToolOutcome;
}

/// The rounds, transcript, and final completion of one ask.
#[derive(Clone, Debug, PartialEq)]
pub struct Turn {
    /// How many model completions ran.
    pub rounds: u32,
    /// The completion that returned no tool calls.
    pub completion: Completion,
    /// The user ask, each assistant reply, and each tool result.
    pub messages: Vec<Message>,
}

/// Why the turn stopped before a final answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopError {
    /// The next round would pass the step cap.
    StepCap {
        /// The cap that blocked the round.
        cap: u32,
    },
    /// The model returned an error.
    Model(ModelError),
}

impl fmt::Display for LoopError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StepCap { cap } => write!(formatter, "the step cap {cap} stopped the turn"),
            Self::Model(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for LoopError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StepCap { .. } => None,
            Self::Model(error) => Some(error),
        }
    }
}

/// Returns the step cap stored on `settings`.
#[must_use]
pub fn cap_from_settings(settings: &Settings) -> u32 {
    settings.step_cap
}

/// Loops completions and tool results until the model returns no tool calls or the next round would pass `cap`.
pub fn run<M: Model, S: ToolSink>(
    model: &mut M,
    tools: &mut S,
    cap: u32,
    ask: &str,
    definitions: &[ToolDefinition],
) -> Result<Turn, LoopError> {
    let mut messages = vec![Message::User {
        text: ask.to_owned(),
    }];
    let mut rounds: u32 = 0;
    loop {
        if rounds >= cap {
            return Err(LoopError::StepCap { cap });
        }
        let request = CompletionRequest {
            messages: messages.clone(),
            tools: definitions.to_vec(),
        };
        let completion = model.complete(&request).map_err(LoopError::Model)?;
        rounds += 1;
        messages.push(Message::Assistant {
            text: completion.text.clone(),
            calls: completion.calls.clone(),
        });
        if completion.calls.is_empty() {
            return Ok(Turn {
                rounds,
                completion,
                messages,
            });
        }
        for call in &completion.calls {
            let outcome = tools.run_tool(call);
            messages.push(Message::Tool {
                call_id: call.id.clone(),
                name: call.name.clone(),
                content: outcome.content,
                is_error: outcome.is_error,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::SegmentId;
    use cutlass_session::Request;
    use cutlass_settings::{DEFAULT_STEP_CAP, Settings};

    use super::{LoopError, ToolOutcome, ToolSink, cap_from_settings, run};
    use crate::model::{Completion, LocalModel, Message, ToolCall, ToolDefinition};

    struct RecordingSink {
        outcomes: Vec<ToolOutcome>,
        calls: Vec<ToolCall>,
    }

    impl RecordingSink {
        fn scripted(outcomes: Vec<ToolOutcome>) -> Self {
            Self {
                outcomes,
                calls: Vec::new(),
            }
        }
    }

    impl ToolSink for RecordingSink {
        fn run_tool(&mut self, call: &ToolCall) -> ToolOutcome {
            let index = self.calls.len();
            self.calls.push(call.clone());
            self.outcomes
                .get(index)
                .cloned()
                .unwrap_or_else(|| ToolOutcome {
                    content: "ok".to_owned(),
                    is_error: false,
                })
        }
    }

    fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_owned(),
            name: name.to_owned(),
            arguments: arguments.to_owned(),
        }
    }

    fn with_call(text: &str, call: ToolCall) -> Completion {
        Completion {
            text: text.to_owned(),
            calls: vec![call],
            proposed: Vec::new(),
        }
    }

    fn done(text: &str) -> Completion {
        Completion {
            text: text.to_owned(),
            calls: Vec::new(),
            proposed: Vec::new(),
        }
    }

    #[test]
    fn one_ask_then_tool_results_go_back_to_the_model() {
        let ask = "what is the revision?";
        let call = tool_call("1", "draft.query", "project");
        let definitions = vec![ToolDefinition {
            name: "draft.query".to_owned(),
            description: "Read the open draft.".to_owned(),
        }];
        let mut model = LocalModel::scripted(vec![with_call("", call.clone()), done("done")]);
        let mut sink = RecordingSink::scripted(vec![ToolOutcome {
            content: "revision=0".to_owned(),
            is_error: false,
        }]);
        let turn = run(&mut model, &mut sink, 8, ask, &definitions).unwrap();
        assert_eq!(turn.rounds, 2);
        assert_eq!(turn.completion.text, "done");
        assert!(turn.completion.calls.is_empty());
        assert_eq!(
            turn.messages[0],
            Message::User {
                text: ask.to_owned(),
            }
        );
        let seen = model.seen();
        assert_eq!(seen.len(), 2);
        assert_eq!(
            seen[0].messages,
            vec![Message::User {
                text: ask.to_owned(),
            }]
        );
        assert_eq!(
            seen[1].messages,
            vec![
                Message::User {
                    text: ask.to_owned(),
                },
                Message::Assistant {
                    text: String::new(),
                    calls: vec![call.clone()],
                },
                Message::Tool {
                    call_id: "1".to_owned(),
                    name: "draft.query".to_owned(),
                    content: "revision=0".to_owned(),
                    is_error: false,
                },
            ]
        );
        assert_eq!(seen[1].tools, definitions);
        assert_eq!(sink.calls, vec![call]);
        assert_eq!(
            turn.messages,
            vec![
                Message::User {
                    text: ask.to_owned(),
                },
                Message::Assistant {
                    text: String::new(),
                    calls: sink.calls.clone(),
                },
                Message::Tool {
                    call_id: "1".to_owned(),
                    name: "draft.query".to_owned(),
                    content: "revision=0".to_owned(),
                    is_error: false,
                },
                Message::Assistant {
                    text: "done".to_owned(),
                    calls: Vec::new(),
                },
            ]
        );
    }

    #[test]
    fn default_step_cap_is_32() {
        let settings = Settings::default();
        let cap = cap_from_settings(&settings);
        assert_eq!(settings.step_cap, DEFAULT_STEP_CAP);
        assert_eq!(cap, 32);
        let steps: Vec<Completion> = (0..cap)
            .map(|index| with_call("", tool_call(&index.to_string(), "draft.query", "")))
            .collect();
        let mut model = LocalModel::scripted(steps);
        let mut sink = RecordingSink::scripted(Vec::new());
        let error = run(&mut model, &mut sink, cap, "keep going", &[]).unwrap_err();
        assert_eq!(error, LoopError::StepCap { cap });
        assert_eq!(model.seen().len(), 32);
        assert_eq!(sink.calls.len(), 32);
        let text = error.to_string();
        assert!(text.contains("step cap"), "{text}");
        assert!(text.contains("32"), "{text}");
    }

    #[test]
    fn raised_step_cap_allows_another_round() {
        let settings = Settings {
            step_cap: 2,
            ..Settings::default()
        };
        let cap = cap_from_settings(&settings);
        assert_eq!(cap, 2);
        let mut model = LocalModel::scripted(vec![
            with_call("", tool_call("1", "draft.query", "project")),
            done("done"),
        ]);
        let mut sink = RecordingSink::scripted(vec![ToolOutcome {
            content: "ok".to_owned(),
            is_error: false,
        }]);
        let turn = run(&mut model, &mut sink, cap, "go", &[]).unwrap();
        assert_eq!(turn.rounds, 2);
        assert_eq!(turn.completion.text, "done");
        assert!(turn.completion.calls.is_empty());
    }

    #[test]
    fn stop_at_the_step_cap_names_the_error() {
        let mut model = LocalModel::scripted(vec![
            with_call("", tool_call("1", "draft.query", "project")),
            with_call("", tool_call("2", "draft.query", "project")),
        ]);
        let mut sink = RecordingSink::scripted(Vec::new());
        let error = run(&mut model, &mut sink, 1, "go", &[]).unwrap_err();
        assert_eq!(error, LoopError::StepCap { cap: 1 });
        assert_eq!(sink.calls.len(), 1);
        assert_eq!(model.seen().len(), 1);
        let text = error.to_string();
        assert!(text.contains("step cap"), "{text}");
        assert!(text.contains('1'), "{text}");
    }

    #[test]
    fn a_tool_error_lets_the_model_try_another_tool() {
        let apply = tool_call("1", "draft.apply", "edit");
        let query = tool_call("2", "draft.query", "project");
        let mut model = LocalModel::scripted(vec![
            with_call("", apply.clone()),
            with_call("", query.clone()),
            done("done"),
        ]);
        let mut sink = RecordingSink::scripted(vec![
            ToolOutcome {
                content: "refused".to_owned(),
                is_error: true,
            },
            ToolOutcome {
                content: "ok".to_owned(),
                is_error: false,
            },
        ]);
        let turn = run(&mut model, &mut sink, 3, "edit the draft", &[]).unwrap();
        assert_eq!(turn.rounds, 3);
        assert_eq!(turn.completion.text, "done");
        assert_eq!(sink.calls, vec![apply.clone(), query.clone()]);
        assert_eq!(
            turn.messages,
            vec![
                Message::User {
                    text: "edit the draft".to_owned(),
                },
                Message::Assistant {
                    text: String::new(),
                    calls: vec![apply],
                },
                Message::Tool {
                    call_id: "1".to_owned(),
                    name: "draft.apply".to_owned(),
                    content: "refused".to_owned(),
                    is_error: true,
                },
                Message::Assistant {
                    text: String::new(),
                    calls: vec![query],
                },
                Message::Tool {
                    call_id: "2".to_owned(),
                    name: "draft.query".to_owned(),
                    content: "ok".to_owned(),
                    is_error: false,
                },
                Message::Assistant {
                    text: "done".to_owned(),
                    calls: Vec::new(),
                },
            ]
        );
    }

    #[test]
    fn proposed_requests_are_not_applied_by_the_turn() {
        let proposed = vec![Request::Remove {
            segment_id: SegmentId::new("s"),
        }];
        let completion = Completion {
            text: "ready".to_owned(),
            calls: Vec::new(),
            proposed: proposed.clone(),
        };
        let mut model = LocalModel::scripted(vec![completion.clone()]);
        let mut sink = RecordingSink::scripted(Vec::new());
        let turn = run(&mut model, &mut sink, 4, "ready the cut", &[]).unwrap();
        assert_eq!(turn.rounds, 1);
        assert_eq!(turn.completion, completion);
        assert_eq!(turn.completion.proposed, proposed);
        assert!(sink.calls.is_empty());
        assert_eq!(model.seen().len(), 1);
        assert!(
            turn.messages
                .iter()
                .all(|message| !matches!(message, Message::Tool { .. }))
        );
    }

    #[test]
    fn cap_zero_does_not_call_the_model() {
        let mut model = LocalModel::scripted(vec![with_call(
            "nope",
            tool_call("1", "draft.query", "project"),
        )]);
        let mut sink = RecordingSink::scripted(Vec::new());
        let error = run(&mut model, &mut sink, 0, "go", &[]).unwrap_err();
        assert_eq!(error, LoopError::StepCap { cap: 0 });
        assert!(model.seen().is_empty());
        assert!(sink.calls.is_empty());
        let text = error.to_string();
        assert!(text.contains("step cap"), "{text}");
        assert!(text.contains('0'), "{text}");
    }

    #[test]
    fn model_error_display_includes_the_inner_text() {
        let error = LoopError::Model(crate::model::ModelError::Inner(
            "host unreachable".to_owned(),
        ));
        assert!(error.to_string().contains("host unreachable"));
        let wrapped = LoopError::Model(crate::model::ModelError::CloudDisabled);
        assert!(wrapped.to_string().contains("cloud"));
    }
}
