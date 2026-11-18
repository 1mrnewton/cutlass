//! `agent.delegate`: hand a look-up to a child with its own conversation.
//!
//! The child starts fresh, may only read (the draft, settings, media, the
//! catalog, skills, a speech job's words, and sandbox scripts; not
//! `track.poll`, which writes the path it reads),
//! and gives back its last answer as the tool's text. Its tool calls run
//! through the same sink as the parent's, so the host shows them and asks
//! for any allow the same way. The parent's conversation keeps only the
//! answer, not the child's reading, which keeps long look-ups out of it.
//!
//! The helpers of one turn share one allowance of rounds, the size of the
//! turn's own step cap. A helper is given what is left of it, and with
//! none left the hand-off is refused. A turn then makes at most twice the
//! cap's requests to the model, where each helper once got the whole cap
//! again and the cap bounded nothing.

use crate::r#loop::{ToolOutcome, ToolSink, run};
use crate::model::{Completion, CompletionRequest, Model, ModelError, ToolCall, ToolDefinition};
use crate::tools::tool_definitions;

/// The tool's name.
pub const DELEGATE: &str = "agent.delegate";

/// The tools a child may call: none of them changes the project.
pub const CHILD_TOOLS: &[&str] = &[
    "draft.query",
    "settings.get",
    "media.probe",
    "speech.poll",
    "catalog.search",
    "skill.load",
    "sandbox.run",
];

/// What the parent model is told the tool does.
pub const DELEGATE_ABOUT: &str = "Hand a look-up to a helper with a fresh conversation, and get back only its answer. \
     input: the task in full, since the helper sees nothing of this chat. The helper can read the project, \
     settings, media, the catalog and skills and run sandbox scripts, but cannot change anything. Use it for \
     reading a lot to answer a narrow question, such as which clips are shorter than a second.";

/// The definition the parent model sees.
#[must_use]
pub fn delegate_definition() -> ToolDefinition {
    ToolDefinition {
        name: DELEGATE.to_owned(),
        description: DELEGATE_ABOUT.to_owned(),
    }
}

/// The parent's tools with [`DELEGATE`] after them.
#[must_use]
pub fn definitions_with_delegate() -> Vec<ToolDefinition> {
    let mut definitions = tool_definitions();
    definitions.push(delegate_definition());
    definitions
}

/// A sink that runs [`DELEGATE`] calls as a child loop on a model `child`
/// makes, and passes every other call to `inner`.
pub struct Delegating<S, F> {
    inner: S,
    child: F,
    /// The rounds the turn's helpers have left between them.
    left: u32,
}

impl<S, F> Delegating<S, F> {
    /// The helpers of the turn this sink serves get `cap` rounds between
    /// them.
    pub fn new(inner: S, child: F, cap: u32) -> Self {
        Self {
            inner,
            child,
            left: cap,
        }
    }

    /// The rounds the turn's helpers have left.
    #[must_use]
    pub fn rounds_left(&self) -> u32 {
        self.left
    }

    /// The sink the other calls go to.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }
}

impl<S, F, M> ToolSink for Delegating<S, F>
where
    S: ToolSink,
    F: FnMut() -> Result<M, String>,
    M: Model,
{
    fn run_tool(&mut self, call: &ToolCall) -> ToolOutcome {
        if call.name != DELEGATE {
            return self.inner.run_tool(call);
        }
        self.inner.helper_started(call);
        let outcome = self.help(call.arguments.trim());
        self.inner.helper_finished(call, &outcome);
        outcome
    }

    fn heard(&mut self, message: &crate::model::Message) {
        self.inner.heard(message);
    }
}

impl<S, F, M> Delegating<S, F>
where
    S: ToolSink,
    F: FnMut() -> Result<M, String>,
    M: Model,
{
    fn help(&mut self, task: &str) -> ToolOutcome {
        if task.is_empty() {
            return failed("agent.delegate needs the task as its input".to_owned());
        }
        if self.left == 0 {
            return failed(
                "this turn's helpers have used all their rounds; read what you need yourself"
                    .to_owned(),
            );
        }
        let model = match (self.child)() {
            Ok(model) => model,
            Err(error) => return failed(format!("the helper did not start: {error}")),
        };
        let allowance = self.left;
        let mut used = 0;
        let mut model = Counted {
            inner: model,
            used: &mut used,
        };
        let definitions: Vec<ToolDefinition> = tool_definitions()
            .into_iter()
            .filter(|definition| CHILD_TOOLS.contains(&definition.name.as_str()))
            .collect();
        let mut sink = ReadOnly {
            inner: &mut self.inner,
        };
        let ask = format!(
            "Another assistant asks you this. Read what you need, then answer in full in plain text; \
             you cannot change the project, and only your last answer reaches it.\n\n{task}"
        );
        let ran = run(&mut model, &mut sink, allowance, &ask, &definitions);
        // What the helper asked of the model comes off the allowance,
        // however it ended.
        self.left = self.left.saturating_sub(used);
        match ran {
            Ok(turn) if turn.completion.text.trim().is_empty() => {
                failed("the helper finished without an answer".to_owned())
            }
            Ok(turn) => ToolOutcome {
                content: turn.completion.text,
                is_error: false,
                picture: None,
            },
            Err(error) => failed(format!("the helper stopped: {error}")),
        }
    }
}

/// A helper's model, counting the requests made of it.
struct Counted<'a, M> {
    inner: M,
    used: &'a mut u32,
}

impl<M: Model> Model for Counted<'_, M> {
    fn complete(&mut self, request: &CompletionRequest) -> Result<Completion, ModelError> {
        *self.used = self.used.saturating_add(1);
        self.inner.complete(request)
    }
}

/// The child's sink: [`CHILD_TOOLS`] pass, anything else is refused.
struct ReadOnly<'a, S> {
    inner: &'a mut S,
}

impl<S: ToolSink> ToolSink for ReadOnly<'_, S> {
    fn run_tool(&mut self, call: &ToolCall) -> ToolOutcome {
        if CHILD_TOOLS.contains(&call.name.as_str()) {
            return self.inner.run_tool(call);
        }
        failed(format!(
            "{} is not open to a helper, which only reads; say what should change in your answer instead",
            call.name
        ))
    }
}

fn failed(content: String) -> ToolOutcome {
    ToolOutcome {
        content,
        is_error: true,
        picture: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{CHILD_TOOLS, DELEGATE, Delegating, definitions_with_delegate};
    use crate::r#loop::{ToolOutcome, ToolSink, run};
    use crate::model::{Completion, LocalModel, ToolCall};
    use crate::tools::TOOL_NAMES;

    /// Records each call and answers with its name, and counts the
    /// messages it hears.
    #[derive(Default)]
    struct Seen(Vec<String>, Vec<crate::model::Message>);

    impl ToolSink for Seen {
        fn run_tool(&mut self, call: &ToolCall) -> ToolOutcome {
            self.0.push(call.name.clone());
            ToolOutcome {
                content: format!("ran {}", call.name),
                is_error: false,
                picture: None,
            }
        }

        fn heard(&mut self, message: &crate::model::Message) {
            self.1.push(message.clone());
        }
    }

    fn calls(name: &str, arguments: &str) -> Completion {
        Completion {
            text: String::new(),
            calls: vec![ToolCall {
                id: format!("c-{name}"),
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            }],
            proposed: Vec::new(),
        }
    }

    fn says(text: &str) -> Completion {
        Completion {
            text: text.to_owned(),
            calls: Vec::new(),
            proposed: Vec::new(),
        }
    }

    #[test]
    fn the_helpers_of_a_turn_share_one_allowance_of_rounds() {
        // Three hand-offs in a turn whose helpers have three rounds.
        let mut parent = LocalModel::scripted(vec![
            calls(DELEGATE, "first look-up"),
            calls(DELEGATE, "second look-up"),
            calls(DELEGATE, "third look-up"),
            says("Done."),
        ]);
        let started = std::cell::Cell::new(0u32);
        // Each helper reads once, then answers: two rounds.
        let child = || {
            started.set(started.get() + 1);
            Ok::<_, String>(LocalModel::scripted(vec![
                calls("draft.query", "project"),
                says("read it"),
            ]))
        };
        let mut sink = Delegating::new(Seen::default(), child, 3);
        assert_eq!(sink.rounds_left(), 3);
        let turn = run(
            &mut parent,
            &mut sink,
            8,
            "look things up",
            &definitions_with_delegate(),
        )
        .unwrap();
        let answers: Vec<(&str, bool)> = turn
            .messages
            .iter()
            .filter_map(|message| match message {
                crate::model::Message::Tool {
                    name,
                    content,
                    is_error,
                    ..
                } if name == DELEGATE => Some((content.as_str(), *is_error)),
                _ => None,
            })
            .collect();
        assert_eq!(answers.len(), 3);
        // The first has room for both its rounds.
        assert_eq!(answers[0], ("read it", false));
        // The second is given the one round left: it reads, and is
        // stopped before it can answer.
        assert!(answers[1].1, "{}", answers[1].0);
        assert!(
            answers[1].0.contains("the step cap 1 stopped the turn"),
            "{}",
            answers[1].0
        );
        // The third is refused before any model is made for it.
        assert!(answers[2].1);
        assert!(
            answers[2].0.contains("used all their rounds"),
            "{}",
            answers[2].0
        );
        assert_eq!(started.get(), 2);
        assert_eq!(sink.rounds_left(), 0);
        // The helpers read twice between them, and the parent went on.
        assert_eq!(sink.inner_mut().0, ["draft.query", "draft.query"]);
        assert_eq!(turn.completion.text, "Done.");
    }

    #[test]
    fn a_child_reads_is_refused_writes_and_only_its_answer_comes_back() {
        let mut parent = LocalModel::scripted(vec![
            calls(DELEGATE, "Which clips are short?"),
            says("Two short clips."),
        ]);
        let child = || {
            Ok::<_, String>(LocalModel::scripted(vec![
                calls("draft.query", "summary"),
                calls("draft.apply", "{}"),
                says("c1 and c2 are under a second."),
            ]))
        };
        let mut sink = Delegating::new(Seen::default(), child, 8);
        let turn = run(
            &mut parent,
            &mut sink,
            8,
            "tidy up",
            &definitions_with_delegate(),
        )
        .unwrap();
        assert_eq!(turn.completion.text, "Two short clips.");
        // The child read; its write never reached the host.
        assert_eq!(sink.inner_mut().0, ["draft.query"]);
        // The parent heard the child's answer and nothing of its reading.
        let heard: Vec<&str> = turn
            .messages
            .iter()
            .filter_map(|message| match message {
                crate::model::Message::Tool { name, content, .. } if name == DELEGATE => {
                    Some(content.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(heard, ["c1 and c2 are under a second."]);
        // The host heard the parent's turn, message by message, and none of
        // the child's.
        assert_eq!(sink.inner_mut().1, turn.messages);
    }

    #[test]
    fn a_helper_may_not_poll_a_track_which_writes_and_its_task_is_in_the_schema() {
        assert!(!CHILD_TOOLS.contains(&"track.poll"));
        let mut parent = LocalModel::scripted(vec![calls(DELEGATE, "follow it"), says("done")]);
        let child = || {
            Ok::<_, String>(LocalModel::scripted(vec![
                calls("track.poll", "track-1"),
                says("could not"),
            ]))
        };
        let mut sink = Delegating::new(Seen::default(), child, 8);
        run(
            &mut parent,
            &mut sink,
            8,
            "go",
            &definitions_with_delegate(),
        )
        .unwrap();
        assert!(sink.inner_mut().0.is_empty(), "the poll reached the host");
        // The model is told where the task goes.
        let schema = crate::mcp_server::input_schema(DELEGATE);
        assert_eq!(schema["required"], serde_json::json!(["input"]));
        assert_eq!(schema["properties"]["input"]["type"], "string");
    }

    #[test]
    fn an_empty_task_or_a_child_that_cannot_start_is_an_error() {
        let mut sink = Delegating::new(
            Seen::default(),
            || Err::<LocalModel, _>("no model".to_owned()),
            4,
        );
        let call = |arguments: &str| ToolCall {
            id: "1".to_owned(),
            name: DELEGATE.to_owned(),
            arguments: arguments.to_owned(),
        };
        assert!(sink.run_tool(&call(" ")).is_error);
        let outcome = sink.run_tool(&call("look"));
        assert!(outcome.is_error && outcome.content.contains("no model"));
        // Other calls pass straight through.
        let outcome = sink.run_tool(&ToolCall {
            id: "2".to_owned(),
            name: "draft.apply".to_owned(),
            arguments: "{}".to_owned(),
        });
        assert!(!outcome.is_error);
    }

    #[test]
    fn every_child_tool_is_a_real_tool() {
        for name in CHILD_TOOLS {
            assert!(TOOL_NAMES.contains(name), "{name}");
        }
        assert!(!TOOL_NAMES.contains(&DELEGATE));
    }
}
