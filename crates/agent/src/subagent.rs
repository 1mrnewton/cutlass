//! A child loop that may query the draft and run the sandbox.

use std::fmt;

use cutlass_session::Request;

use crate::draft::{ApplyInput, ApplyMode, ApplyResult, DraftError, DraftQuery, DraftTools};
use crate::r#loop::{LoopError, ToolOutcome, ToolSink};
use crate::model::{Model, ToolCall, ToolDefinition};
use crate::sandbox::{Sandbox, SandboxError};

/// Session requests the child proposed. The child does not apply them.
#[derive(Clone, Debug, PartialEq)]
pub struct SubagentResult {
    /// Requests for the parent to commit.
    pub requests: Vec<Request>,
}

/// Why a child loop did not return requests.
pub enum SubagentError {
    /// The model loop stopped.
    Loop(LoopError),
    /// The model called a tool outside the child list.
    ToolNotAllowed {
        /// The tool name that was refused.
        name: String,
    },
    /// A draft call failed.
    Draft(DraftError),
    /// A sandbox call failed.
    Sandbox(SandboxError),
}

impl fmt::Display for SubagentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loop(error) => write!(formatter, "{error}"),
            Self::ToolNotAllowed { name } => write!(formatter, "tool not allowed: {name}"),
            Self::Draft(error) => write!(formatter, "{error}"),
            Self::Sandbox(error) => write!(formatter, "{error}"),
        }
    }
}

impl fmt::Debug for SubagentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loop(error) => write!(formatter, "Loop({error})"),
            Self::ToolNotAllowed { name } => {
                write!(formatter, "ToolNotAllowed {{ name: {name} }}")
            }
            Self::Draft(error) => write!(formatter, "Draft({error})"),
            Self::Sandbox(error) => write!(formatter, "Sandbox({error})"),
        }
    }
}

impl std::error::Error for SubagentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Loop(error) => Some(error),
            Self::Draft(error) => Some(error),
            Self::Sandbox(error) => Some(error),
            Self::ToolNotAllowed { .. } => None,
        }
    }
}

/// The only tools a child loop may call.
#[must_use]
pub fn allowed_tools() -> &'static [&'static str] {
    &["draft.query", "sandbox.run"]
}

/// A child loop bound to one draft and one sandbox.
pub struct Subagent<'a, S> {
    draft: &'a DraftTools,
    sandbox: &'a mut S,
}

impl<'a, S: Sandbox> Subagent<'a, S> {
    /// A child that can read `draft` and run `sandbox`.
    #[must_use]
    pub fn new(draft: &'a DraftTools, sandbox: &'a mut S) -> Self {
        Self { draft, sandbox }
    }

    /// Query and run the sandbox, then return proposed requests.
    pub fn run<M: Model>(
        &mut self,
        model: &mut M,
        cap: u32,
        ask: &str,
    ) -> Result<SubagentResult, SubagentError> {
        let definitions = tool_definitions();
        let mut sink = ChildSink {
            draft: self.draft,
            sandbox: &mut *self.sandbox,
            disallowed: None,
        };
        let turn = match crate::r#loop::run(model, &mut sink, cap, ask, &definitions) {
            Ok(turn) => turn,
            Err(error) => return Err(SubagentError::Loop(error)),
        };
        if let Some(name) = sink.disallowed {
            return Err(SubagentError::ToolNotAllowed { name });
        }
        Ok(SubagentResult {
            requests: turn.completion.proposed,
        })
    }
}

/// Commit the child result as one combo on the parent draft.
pub fn parent_apply(
    draft: &mut DraftTools,
    result: &SubagentResult,
    request_id: &str,
    expected_revision: u64,
) -> Result<ApplyResult, DraftError> {
    let request_id = if request_id.is_empty() {
        None
    } else {
        Some(request_id.to_owned())
    };
    draft.apply(ApplyInput {
        request_id,
        expected_revision,
        requests: result.requests.clone(),
        mode: ApplyMode::Commit,
    })
}

fn tool_definitions() -> Vec<ToolDefinition> {
    allowed_tools()
        .iter()
        .map(|name| ToolDefinition {
            name: (*name).to_owned(),
            description: match *name {
                "draft.query" => "Read a compact draft view.",
                "sandbox.run" => "Run source with a JSON input.",
                _ => "Run one allowed tool.",
            }
            .to_owned(),
        })
        .collect()
}

struct ChildSink<'a, S> {
    draft: &'a DraftTools,
    sandbox: &'a mut S,
    disallowed: Option<String>,
}

impl<S: Sandbox> ToolSink for ChildSink<'_, S> {
    fn run_tool(&mut self, call: &ToolCall) -> ToolOutcome {
        if !allowed_tools().iter().any(|name| *name == call.name) {
            return self.refuse(&call.name);
        }
        match call.name.as_str() {
            "draft.query" => self.query(&call.arguments),
            "sandbox.run" => self.sandbox_run(&call.arguments),
            _ => self.refuse(&call.name),
        }
    }
}

impl<S: Sandbox> ChildSink<'_, S> {
    fn query(&self, arguments: &str) -> ToolOutcome {
        let query = match arguments.trim() {
            "project" => DraftQuery::Project,
            "full" => DraftQuery::Full,
            _ => {
                return ToolOutcome {
                    content: "unsupported query".to_owned(),
                    is_error: true,
                };
            }
        };
        match self.draft.query(&query) {
            Ok(view) => ToolOutcome {
                content: view.to_string(),
                is_error: false,
            },
            Err(error) => ToolOutcome {
                content: error.to_string(),
                is_error: true,
            },
        }
    }

    fn sandbox_run(&mut self, arguments: &str) -> ToolOutcome {
        let (source, input_json) = match arguments.split_once('\n') {
            Some((source, input_json)) => (source, input_json),
            None => (arguments, "{}"),
        };
        match self.sandbox.run(source, input_json) {
            Ok(output) => ToolOutcome {
                content: output,
                is_error: false,
            },
            Err(error) => ToolOutcome {
                content: error.to_string(),
                is_error: true,
            },
        }
    }

    fn refuse(&mut self, name: &str) -> ToolOutcome {
        if self.disallowed.is_none() {
            self.disallowed = Some(name.to_owned());
        }
        ToolOutcome {
            content: format!("tool not allowed: {name}"),
            is_error: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, Segment, SegmentId, TimeRange, TrackId};
    use cutlass_session::{Request, Session};

    use crate::draft::DraftTools;
    use crate::r#loop::LoopError;
    use crate::model::{Completion, CompletionRequest, LocalModel, Model, ModelError, ToolCall};
    use crate::sandbox::{LocalSandbox, SandboxError, SandboxExecutor, SandboxPolicy};

    use super::{Subagent, SubagentError, allowed_tools, parent_apply};

    struct Echo {
        calls: u32,
    }

    impl SandboxExecutor for Echo {
        fn execute(&mut self, _source: &str, input_json: &str) -> Result<String, SandboxError> {
            self.calls += 1;
            Ok(input_json.to_owned())
        }
    }

    struct QueryUntil {
        left: u32,
    }

    impl Model for QueryUntil {
        fn complete(&mut self, _request: &CompletionRequest) -> Result<Completion, ModelError> {
            if self.left == 0 {
                return Err(ModelError::NoMoreSteps);
            }
            self.left -= 1;
            Ok(Completion {
                text: String::new(),
                calls: vec![ToolCall {
                    id: "q".to_owned(),
                    name: "draft.query".to_owned(),
                    arguments: "project".to_owned(),
                }],
                proposed: Vec::new(),
            })
        }
    }

    fn open_main() -> DraftTools {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        DraftTools::new(Session::open(draft))
    }

    fn empty_sandbox() -> LocalSandbox<Echo> {
        LocalSandbox::new(
            SandboxPolicy {
                project_path: String::new(),
                home_path: String::new(),
            },
            Echo { calls: 0 },
        )
    }

    fn done(text: &str, proposed: Vec<Request>) -> Completion {
        Completion {
            text: text.to_owned(),
            calls: Vec::new(),
            proposed,
        }
    }

    fn call(name: &str, arguments: &str) -> Completion {
        Completion {
            text: String::new(),
            calls: vec![ToolCall {
                id: "1".to_owned(),
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            }],
            proposed: Vec::new(),
        }
    }

    #[test]
    fn child_tool_list_is_query_and_sandbox() {
        assert_eq!(allowed_tools(), ["draft.query", "sandbox.run"]);
        let tools = open_main();
        let mut sandbox = empty_sandbox();
        let mut model = LocalModel::scripted(vec![done("done", Vec::new())]);
        let mut child = Subagent::new(&tools, &mut sandbox);
        child.run(&mut model, 4, "look").unwrap();
        let seen = model.seen();
        assert_eq!(seen.len(), 1);
        let names: Vec<&str> = seen[0]
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect();
        assert_eq!(names, ["draft.query", "sandbox.run"]);
    }

    #[test]
    fn child_query_does_not_apply() {
        let tools = open_main();
        let mut sandbox = empty_sandbox();
        let mut model = LocalModel::scripted(vec![
            call("draft.query", "project"),
            done("done", Vec::new()),
        ]);
        let result = {
            let mut child = Subagent::new(&tools, &mut sandbox);
            child.run(&mut model, 4, "what is the project").unwrap()
        };
        assert!(result.requests.is_empty());
        assert_eq!(tools.session().revision(), 0);
        assert!(tools.session().draft().tracks[0].segments.is_empty());
        assert_eq!(sandbox.executor().calls, 0);
    }

    #[test]
    fn child_sandbox_runs_and_returns_requests() {
        let tools = open_main();
        let mut sandbox = empty_sandbox();
        let remove = Request::Remove {
            segment_id: SegmentId::new("gap"),
        };
        let mut model = LocalModel::scripted(vec![
            call("sandbox.run", "return input\n{}"),
            done("ready", vec![remove]),
        ]);
        let result = {
            let mut child = Subagent::new(&tools, &mut sandbox);
            child.run(&mut model, 4, "find the gaps").unwrap()
        };
        assert_eq!(result.requests.len(), 1);
        assert!(matches!(
            result.requests.as_slice(),
            [Request::Remove { segment_id }] if segment_id.as_str() == "gap"
        ));
        assert_eq!(tools.session().revision(), 0);
        assert_eq!(sandbox.executor().calls, 1);
    }

    #[test]
    fn child_apply_is_refused_and_the_draft_stays() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.tracks[0]
            .segments
            .push(Segment::new("gap", "media", TimeRange::new(0, 1_000_000)));
        let tools = DraftTools::new(Session::open(draft));
        let mut sandbox = empty_sandbox();
        let mut model = LocalModel::scripted(vec![
            call("draft.apply", "place"),
            done(
                "ready",
                vec![Request::Remove {
                    segment_id: SegmentId::new("gap"),
                }],
            ),
        ]);
        let error = {
            let mut child = Subagent::new(&tools, &mut sandbox);
            child.run(&mut model, 4, "apply the trim").unwrap_err()
        };
        assert!(matches!(
            error,
            SubagentError::ToolNotAllowed { name } if name == "draft.apply"
        ));
        assert_eq!(tools.session().revision(), 0);
        assert_eq!(tools.session().draft().tracks[0].segments.len(), 1);
        assert_eq!(
            tools.session().draft().tracks[0].segments[0].id.as_str(),
            "gap"
        );
    }

    #[test]
    fn parent_apply_is_the_only_commit() {
        let mut tools = open_main();
        let mut sandbox = empty_sandbox();
        let place = Request::Place {
            track_id: TrackId::new("main"),
            segment: Box::new(Segment::new("clip", "media", TimeRange::new(0, 1_000_000))),
        };
        let mut model = LocalModel::scripted(vec![done("ready", vec![place])]);
        let result = {
            let mut child = Subagent::new(&tools, &mut sandbox);
            child.run(&mut model, 4, "place the clip").unwrap()
        };
        assert_eq!(result.requests.len(), 1);
        assert_eq!(tools.session().revision(), 0);
        assert!(tools.session().draft().tracks[0].segments.is_empty());
        let first = parent_apply(&mut tools, &result, "child", 0).unwrap();
        assert_eq!(tools.session().revision(), 1);
        assert_eq!(tools.session().draft().tracks[0].segments.len(), 1);
        assert_eq!(
            tools.session().draft().tracks[0].segments[0].id.as_str(),
            "clip"
        );
        let second = parent_apply(&mut tools, &result, "child", 0).unwrap();
        assert_eq!(second, first);
        assert_eq!(tools.session().revision(), 1);
        assert_eq!(tools.session().draft().tracks[0].segments.len(), 1);
    }

    #[test]
    fn child_stops_at_the_step_cap() {
        let tools = open_main();
        let mut sandbox = empty_sandbox();
        let mut model = QueryUntil { left: 4 };
        let error = {
            let mut child = Subagent::new(&tools, &mut sandbox);
            child.run(&mut model, 1, "query").unwrap_err()
        };
        assert!(
            matches!(error, SubagentError::Loop(LoopError::StepCap { cap: 1 })),
            "{error:?}"
        );
        assert_eq!(tools.session().revision(), 0);
        assert!(tools.session().draft().tracks[0].segments.is_empty());
    }
}
