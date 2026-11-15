//! The assistant's conversation: what was said, each tool call as a step,
//! and the turn running now.
//!
//! The model loop runs on its own thread, off the frame path. Each tool
//! call comes back to the window's thread, which runs it on the open
//! project between frames and sends the result back. Stop lets the turn go
//! at once: the thread finds out at its next call and ends.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};

use cutlass_agent::{
    ApprovalAction, ChatModel, CloudModel, Completion, CompletionRequest, Delegating, LoopbackHttp,
    Message, Model, ModelError, ToolCall, ToolOutcome, ToolSink, definitions_with_delegate,
    run_after, tool_definitions,
};
use cutlass_settings::Settings;

use crate::shell::Shell;

/// One line of the conversation as the panel shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Entry {
    /// What the user asked.
    Asked(String),
    /// What the model said.
    Said(String),
    /// A tool the model ran, what it was given, and what came back.
    Step {
        tool: String,
        input: String,
        result: Option<String>,
        failed: bool,
    },
    /// The turn stopped or failed, and why.
    Notice(String),
    /// A call waits for the user's allow: the question, then the answer
    /// once given.
    Ask {
        question: String,
        allowed: Option<bool>,
    },
}

/// A tool call waiting for the user's allow.
struct Pending {
    call: ToolCall,
    /// What the user is asked now.
    action: ApprovalAction,
    /// What they already allowed for this call; a call may need more than
    /// one allow, and asks for each in turn.
    allowed: Vec<ApprovalAction>,
    refused: ToolOutcome,
    answer: Sender<ToolOutcome>,
}

/// From the model's thread to the window's.
enum Event {
    Said(String),
    /// A message joined the turn's transcript.
    Heard(Message),
    Tool {
        call: ToolCall,
        answer: Sender<ToolOutcome>,
    },
    /// A helper took a look-up, then gave its answer.
    Helper {
        call: ToolCall,
        outcome: Option<ToolOutcome>,
    },
    Finished(Result<(String, Vec<Message>), String>),
}

struct Running {
    events: Receiver<Event>,
    stop: Arc<AtomicBool>,
}

/// The conversation and the turn running now.
#[derive(Default)]
pub struct Assistant {
    /// The lines shown.
    pub entries: Vec<Entry>,
    /// The words in the box, not sent yet.
    pub draft: String,
    /// The panel is showing.
    pub open: bool,
    history: Vec<Message>,
    /// The messages of the turn running now, as the loop adds them, so a
    /// turn that fails or is stopped is still remembered.
    turn: Vec<Message>,
    running: Option<Running>,
    pending: Option<Pending>,
    /// Where this conversation is kept; `None` keeps it nowhere.
    chat_file: Option<PathBuf>,
}

impl Assistant {
    /// A turn is running.
    #[must_use]
    pub fn busy(&self) -> bool {
        self.running.is_some()
    }

    /// A call waits for the user to allow it or not.
    #[must_use]
    pub fn asking(&self) -> bool {
        self.pending.is_some()
    }

    /// Answer the call waiting for an allow. Allowed, it runs; refused, the
    /// model hears that the user said no.
    pub fn decide(&mut self, shell: &mut Shell, allow: bool) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        if let Some(Entry::Ask { allowed, .. }) = self
            .entries
            .iter_mut()
            .rev()
            .find(|entry| matches!(entry, Entry::Ask { allowed: None, .. }))
        {
            *allowed = Some(allow);
        }
        let outcome = if allow {
            let mut allowed = pending.allowed;
            allowed.push(pending.action);
            let run = shell.run_agent_tool(&pending.call, &allowed);
            if let Some(action) = run.wants {
                // The call needs another allow: ask for that one too.
                self.entries.push(Entry::Ask {
                    question: action.prompt(),
                    allowed: None,
                });
                self.pending = Some(Pending {
                    call: pending.call,
                    action,
                    allowed,
                    refused: run.outcome,
                    answer: pending.answer,
                });
                return;
            }
            run.outcome
        } else {
            ToolOutcome {
                content: format!("the user said no: {}", pending.refused.content),
                is_error: true,
            }
        };
        self.entries.push(Entry::Step {
            tool: pending.call.name.clone(),
            input: pending.call.arguments.clone(),
            result: Some(outcome.content.clone()),
            failed: outcome.is_error,
        });
        let _ = pending.answer.send(outcome);
    }

    /// Send the words in the box with the model `settings` choose, telling
    /// it the user's `notes` while Memory is on.
    pub fn send(&mut self, settings: &Settings, notes: &str) {
        let ask = self.draft.trim().to_owned();
        if ask.is_empty() || self.busy() {
            return;
        }
        self.draft.clear();
        self.entries.push(Entry::Asked(ask.clone()));
        let model = match build_model(settings, notes) {
            Ok(model) => model,
            Err(notice) => {
                self.entries.push(Entry::Notice(notice));
                return;
            }
        };
        // A helper the model hands a look-up to runs on a model of its own,
        // the same kind, made when it is asked for.
        let cap = settings.step_cap;
        let (settings, notes) = (settings.clone(), notes.to_owned());
        let child = move || build_model(&settings, &notes);
        self.launch(model, Some(child), cap, &ask);
    }

    /// Run `ask` on `model` on its own thread, with no helper to hand
    /// look-ups to.
    pub fn start<M: Model + Send + 'static>(&mut self, model: M, cap: u32, ask: &str) {
        self.launch(model, None::<fn() -> Result<M, String>>, cap, ask);
    }

    /// Run `ask` on `model` on its own thread. With `child`, the model may
    /// hand a look-up to a helper on a model `child` makes.
    pub fn launch<M, C, F>(&mut self, model: M, child: Option<F>, cap: u32, ask: &str)
    where
        M: Model + Send + 'static,
        C: Model + 'static,
        F: FnMut() -> Result<C, String> + Send + 'static,
    {
        let (events, received) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        self.turn.clear();
        let history = self.history.clone();
        let ask = ask.to_owned();
        let watched = Watched {
            inner: model,
            stop: Arc::clone(&stop),
            events: events.clone(),
        };
        let relay = Relay {
            stop: Arc::clone(&stop),
            events: events.clone(),
        };
        let halt = Arc::clone(&stop);
        let spawned = std::thread::Builder::new()
            .name("assistant".to_owned())
            .spawn(move || {
                let mut watched = watched;
                let result = match child {
                    Some(mut child) => {
                        // A helper stops with the turn, as the parent does.
                        let made = move || {
                            child().map(|inner| Halting {
                                inner,
                                stop: Arc::clone(&halt),
                            })
                        };
                        let mut sink = Delegating::new(relay, made, cap);
                        run_after(
                            &mut watched,
                            &mut sink,
                            cap,
                            history,
                            &ask,
                            &definitions_with_delegate(),
                        )
                    }
                    None => {
                        let mut relay = relay;
                        run_after(
                            &mut watched,
                            &mut relay,
                            cap,
                            history,
                            &ask,
                            &tool_definitions(),
                        )
                    }
                }
                .map(|turn| (turn.completion.text, turn.messages))
                .map_err(|error| error.to_string());
                let _ = events.send(Event::Finished(result));
            });
        match spawned {
            Ok(_) => {
                self.running = Some(Running {
                    events: received,
                    stop,
                });
            }
            Err(error) => self.entries.push(Entry::Notice(format!(
                "the assistant did not start: {error}"
            ))),
        }
    }

    /// Stop the turn running now. What it already changed stays, each
    /// change its own undo step.
    pub fn stop(&mut self) {
        self.pending = None;
        for entry in &mut self.entries {
            if let Entry::Ask {
                allowed: allowed @ None,
                ..
            } = entry
            {
                *allowed = Some(false);
            }
        }
        if let Some(running) = self.running.take() {
            running.stop.store(true, Ordering::Relaxed);
            // What the loop already sent is part of the turn.
            for event in running.events.try_iter() {
                if let Event::Heard(message) = event {
                    self.turn.push(message);
                }
            }
            self.keep_turn();
            for entry in &mut self.entries {
                if let Entry::Step {
                    result: result @ None,
                    failed,
                    ..
                } = entry
                {
                    *result = Some("stopped".to_owned());
                    *failed = true;
                }
            }
            self.entries.push(Entry::Notice("Stopped.".to_owned()));
        }
    }

    /// Start over: nothing said, and the model forgets the earlier turns.
    /// The kept conversation is forgotten too.
    pub fn clear(&mut self) {
        self.stop();
        self.entries.clear();
        self.history.clear();
        self.turn.clear();
        self.keep();
    }

    /// Remember the turn that ended without an answer, failed or stopped:
    /// what was asked, what the model said and each tool it ran, so the
    /// next turn knows what was already changed. A call that never ran is
    /// answered as stopped, since a transcript may not leave one open. A
    /// turn with only its ask did nothing and is not kept.
    fn keep_turn(&mut self) {
        let mut turn = std::mem::take(&mut self.turn);
        if turn.len() <= 1 {
            return;
        }
        let last_reply = turn
            .iter()
            .rposition(|message| matches!(message, Message::Assistant { .. }));
        if let Some(at) = last_reply
            && let Message::Assistant { calls, .. } = &turn[at]
        {
            let unanswered: Vec<ToolCall> = calls
                .iter()
                .filter(|call| {
                    !turn[at + 1..].iter().any(|message| {
                        matches!(message, Message::Tool { call_id, .. } if *call_id == call.id)
                    })
                })
                .cloned()
                .collect();
            for call in unanswered {
                turn.push(Message::Tool {
                    call_id: call.id,
                    name: call.name,
                    content: "stopped before it ran".to_owned(),
                    is_error: true,
                });
            }
        }
        self.history.append(&mut turn);
        self.keep();
    }

    /// Keep the conversation in `file` from now on, the open project's,
    /// picking up what was said there before. `None` keeps it nowhere and
    /// leaves what is on screen. A turn running holds on to the file it
    /// started with.
    pub fn follow(&mut self, file: Option<PathBuf>) {
        if self.busy() || file == self.chat_file {
            return;
        }
        if file.is_none() {
            self.chat_file = None;
            return;
        }
        self.pending = None;
        self.history = file
            .as_deref()
            .map(|file| {
                cutlass_agent::load_chat(file).unwrap_or_else(|error| {
                    cutlass_logger::warn!("assistant chat: {error}");
                    Vec::new()
                })
            })
            .unwrap_or_default();
        self.entries = shown(&self.history);
        self.chat_file = file;
    }

    /// Write the conversation to its file; an empty one removes the file.
    fn keep(&self) {
        let Some(file) = &self.chat_file else {
            return;
        };
        if let Err(error) = cutlass_agent::save_chat(file, &self.history) {
            cutlass_logger::warn!("assistant chat: {error}");
        }
    }

    /// Take what the model's thread sent: run its tool calls on `shell`
    /// and show what it said. True while a turn is still running.
    pub fn poll(&mut self, shell: &mut Shell) -> bool {
        loop {
            let Some(running) = self.running.as_ref() else {
                return false;
            };
            if self.pending.is_some() {
                return true;
            }
            let event = match running.events.try_recv() {
                Ok(event) => event,
                Err(mpsc::TryRecvError::Empty) => return true,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.running = None;
                    self.entries.push(Entry::Notice(
                        "the assistant ended without an answer".to_owned(),
                    ));
                    return false;
                }
            };
            match event {
                Event::Said(text) => self.entries.push(Entry::Said(text)),
                Event::Heard(message) => self.turn.push(message),
                Event::Helper {
                    call,
                    outcome: None,
                } => self.entries.push(Entry::Step {
                    tool: call.name,
                    input: call.arguments,
                    result: None,
                    failed: false,
                }),
                Event::Helper {
                    call,
                    outcome: Some(outcome),
                } => {
                    if let Some(Entry::Step { result, failed, .. }) =
                        self.entries.iter_mut().rev().find(|entry| {
                            matches!(entry, Entry::Step { tool, result: None, .. } if *tool == call.name)
                        })
                    {
                        *result = Some(outcome.content);
                        *failed = outcome.is_error;
                    }
                }
                Event::Tool { call, answer } => {
                    let run = shell.run_agent_tool(&call, &[]);
                    if let Some(action) = run.wants {
                        self.entries.push(Entry::Ask {
                            question: action.prompt(),
                            allowed: None,
                        });
                        self.pending = Some(Pending {
                            call,
                            action,
                            allowed: Vec::new(),
                            refused: run.outcome,
                            answer,
                        });
                        return true;
                    }
                    let outcome = run.outcome;
                    self.entries.push(Entry::Step {
                        tool: call.name.clone(),
                        input: call.arguments.clone(),
                        result: Some(outcome.content.clone()),
                        failed: outcome.is_error,
                    });
                    let _ = answer.send(outcome);
                }
                Event::Finished(result) => {
                    self.running = None;
                    match result {
                        Ok((text, messages)) => {
                            if text.is_empty() {
                                // A model that ends on nothing leaves the
                                // user looking at a turn that just stopped.
                                self.entries.push(Entry::Notice(
                                    "The assistant finished without saying anything. Ask again, or try another model."
                                        .to_owned(),
                                ));
                            } else {
                                self.entries.push(Entry::Said(text));
                            }
                            self.turn.clear();
                            self.history = messages;
                            self.keep();
                        }
                        Err(error) => {
                            self.entries.push(Entry::Notice(explain(error)));
                            self.keep_turn();
                        }
                    }
                    return false;
                }
            }
        }
    }
}

/// The panel's lines for a conversation read back.
fn shown(history: &[Message]) -> Vec<Entry> {
    let mut entries = Vec::new();
    for message in history {
        match message {
            Message::User { text } => entries.push(Entry::Asked(text.clone())),
            Message::Assistant { text, calls } => {
                if !text.is_empty() {
                    entries.push(Entry::Said(text.clone()));
                }
                for call in calls {
                    entries.push(Entry::Step {
                        tool: call.name.clone(),
                        input: call.arguments.clone(),
                        result: None,
                        failed: false,
                    });
                }
            }
            Message::Tool {
                name,
                content,
                is_error,
                ..
            } => {
                if let Some(Entry::Step { result, failed, .. }) = entries.iter_mut().rev().find(
                    |entry| matches!(entry, Entry::Step { tool, result: None, .. } if tool == name),
                ) {
                    *result = Some(content.clone());
                    *failed = *is_error;
                }
            }
        }
    }
    entries
}

/// A turn's error, with what to do when the server lacks the model.
fn explain(error: String) -> String {
    let lower = error.to_ascii_lowercase();
    if lower.contains("not found") && lower.contains("model") {
        return format!(
            "{error}. Pick a model the server has with Find models in Settings → Assistant."
        );
    }
    error
}

/// The local or the cloud model, whichever `settings` choose, told the
/// user's `notes` while Memory is on.
fn build_model(settings: &Settings, notes: &str) -> Result<AnyModel, String> {
    let context = cutlass_agent::prompt_context(settings, notes);
    if settings.model.cloud {
        let key = crate::cloud::key(&crate::cloud::keys_file(), &settings.model.cloud_provider);
        let model = crate::cloud::model(settings, &key, &context)?;
        return Ok(AnyModel::Cloud(CloudModel::from_settings(settings, model)));
    }
    if !settings.model.local {
        return Err(
            "Both models are off. Turn on the local model, or the cloud model with your own key, in Settings → Assistant."
                .to_owned(),
        );
    }
    let transport =
        LoopbackHttp::new(&settings.model.local_url).map_err(|error| error.to_string())?;
    Ok(AnyModel::Local(
        ChatModel::new(transport, &settings.model.local_name).with_context(&context),
    ))
}

/// The model a turn runs on.
enum AnyModel {
    Local(ChatModel<LoopbackHttp>),
    Cloud(CloudModel<crate::cloud::Cloud>),
}

impl Model for AnyModel {
    fn complete(&mut self, request: &CompletionRequest) -> Result<Completion, ModelError> {
        match self {
            Self::Local(model) => model.complete(request),
            Self::Cloud(model) => model.complete(request),
        }
    }
}

/// A helper's model, stopped with the turn.
struct Halting<M> {
    inner: M,
    stop: Arc<AtomicBool>,
}

impl<M: Model> Model for Halting<M> {
    fn complete(&mut self, request: &CompletionRequest) -> Result<Completion, ModelError> {
        if self.stop.load(Ordering::Relaxed) {
            return Err(ModelError::Inner("stopped".to_owned()));
        }
        self.inner.complete(request)
    }
}

/// The model, stopped when the user stops the turn; its words between tool
/// calls are shown as they come.
struct Watched<M> {
    inner: M,
    stop: Arc<AtomicBool>,
    events: Sender<Event>,
}

impl<M: Model> Model for Watched<M> {
    fn complete(&mut self, request: &CompletionRequest) -> Result<Completion, ModelError> {
        if self.stop.load(Ordering::Relaxed) {
            return Err(ModelError::Inner("stopped".to_owned()));
        }
        let completion = self.inner.complete(request)?;
        if !completion.calls.is_empty() && !completion.text.is_empty() {
            let _ = self.events.send(Event::Said(completion.text.clone()));
        }
        Ok(completion)
    }
}

/// Tool calls, sent to the window's thread to run.
struct Relay {
    stop: Arc<AtomicBool>,
    events: Sender<Event>,
}

impl ToolSink for Relay {
    fn run_tool(&mut self, call: &ToolCall) -> ToolOutcome {
        let stopped = || ToolOutcome {
            content: "the user stopped the turn".to_owned(),
            is_error: true,
        };
        if self.stop.load(Ordering::Relaxed) {
            return stopped();
        }
        let (answer, answered) = mpsc::channel();
        if self
            .events
            .send(Event::Tool {
                call: call.clone(),
                answer,
            })
            .is_err()
        {
            return stopped();
        }
        answered.recv().unwrap_or_else(|_| stopped())
    }

    fn heard(&mut self, message: &Message) {
        let _ = self.events.send(Event::Heard(message.clone()));
    }

    fn helper_started(&mut self, call: &ToolCall) {
        let _ = self.events.send(Event::Helper {
            call: call.clone(),
            outcome: None,
        });
    }

    fn helper_finished(&mut self, call: &ToolCall, outcome: &ToolOutcome) {
        let _ = self.events.send(Event::Helper {
            call: call.clone(),
            outcome: Some(outcome.clone()),
        });
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use cutlass_agent::{Completion, LocalModel, Message, ToolCall};
    use cutlass_model::{Draft, Micros, Segment, TimeRange};
    use cutlass_store::Folder;

    use super::{Assistant, Entry};
    use crate::shell::Shell;

    fn shell_with_project(name: &str) -> (Shell, std::path::PathBuf) {
        let path =
            std::env::temp_dir().join(format!("cutlass-assistant-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let mut draft = Draft::empty("draft");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "missing", TimeRange::new(0, 3_000_000)));
        Folder::new(draft).save(&path).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, cutlass_settings::Settings::default());
        shell.open_folder(&path).unwrap();
        (shell, path)
    }

    fn finish(assistant: &mut Assistant, shell: &mut Shell) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while assistant.poll(shell) {
            assert!(Instant::now() < deadline, "the turn did not finish");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    fn completion(text: &str, calls: Vec<ToolCall>) -> Completion {
        Completion {
            text: text.to_owned(),
            calls,
            proposed: Vec::new(),
        }
    }

    #[test]
    fn a_turn_runs_its_tools_on_the_window_and_remembers_what_was_said() {
        let (mut shell, path) = shell_with_project("turn");
        let mut assistant = Assistant::default();
        let model = LocalModel::scripted(vec![
            completion(
                "Moving to two seconds.",
                vec![ToolCall {
                    id: "1".to_owned(),
                    name: "player.seek".to_owned(),
                    arguments: "2000000".to_owned(),
                }],
            ),
            completion("Done.", Vec::new()),
        ]);
        assistant.start(model, 8, "go to 2s");
        assert!(assistant.busy());
        finish(&mut assistant, &mut shell);
        assert!(!assistant.busy());
        assert_eq!(shell.playhead(), Micros(2_000_000));
        assert_eq!(
            assistant.entries,
            vec![
                Entry::Said("Moving to two seconds.".to_owned()),
                Entry::Step {
                    tool: "player.seek".to_owned(),
                    input: "2000000".to_owned(),
                    result: Some("ok".to_owned()),
                    failed: false,
                },
                Entry::Said("Done.".to_owned()),
            ]
        );
        // The next turn hands the model what came before.
        let model = LocalModel::scripted(vec![completion("Still here.", Vec::new())]);
        assistant.start(model, 8, "and now?");
        finish(&mut assistant, &mut shell);
        assert_eq!(assistant.history.len(), 6);
        assert_eq!(
            assistant.entries.last(),
            Some(&Entry::Said("Still here.".to_owned()))
        );
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn a_conversation_is_kept_per_file_and_read_back_with_its_steps() {
        let (mut shell, path) = shell_with_project("kept");
        let first = path.join("chats").join("a.json");
        let second = path.join("chats").join("b.json");
        let mut assistant = Assistant::default();
        assistant.follow(Some(first.clone()));
        let model = LocalModel::scripted(vec![
            completion(
                "Seeking.",
                vec![ToolCall {
                    id: "1".to_owned(),
                    name: "player.seek".to_owned(),
                    arguments: "1000000".to_owned(),
                }],
            ),
            completion("Done.", Vec::new()),
        ]);
        assistant.start(model, 8, "go to 1s");
        finish(&mut assistant, &mut shell);
        let live = assistant.entries.clone();
        assert!(first.exists());
        // Another project starts empty; coming back reads it all again.
        assistant.follow(Some(second.clone()));
        assert!(assistant.entries.is_empty() && assistant.history.is_empty());
        assistant.follow(Some(first.clone()));
        let mut expected = vec![Entry::Asked("go to 1s".to_owned())];
        expected.extend(live);
        assert_eq!(assistant.entries, expected);
        assert_eq!(assistant.history.len(), 4);
        // Memory off keeps what is shown and writes nothing more.
        assistant.follow(None);
        assert_eq!(assistant.entries.len(), expected.len());
        // Clear forgets the kept file.
        assistant.follow(Some(first.clone()));
        assistant.clear();
        assert!(!first.exists());
        assert!(!second.exists());
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn a_helper_reads_on_the_window_cannot_write_and_answers_the_parent() {
        let (mut shell, path) = shell_with_project("helper");
        let mut assistant = Assistant::default();
        let call = |name: &str, arguments: &str| {
            vec![ToolCall {
                id: name.to_owned(),
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            }]
        };
        let parent = LocalModel::scripted(vec![
            completion("", call("agent.delegate", "How long is the draft?")),
            completion("The helper says it is short.", Vec::new()),
        ]);
        let mut helper = Some(LocalModel::scripted(vec![
            completion("", call("draft.query", "project")),
            completion("", call("player.seek", "2000000")),
            completion("It is short.", Vec::new()),
        ]));
        let child = move || helper.take().ok_or_else(|| "one helper only".to_owned());
        assistant.launch(parent, Some(child), 8, "how long?");
        finish(&mut assistant, &mut shell);
        // The helper's read ran on the window; its seek never did.
        assert_eq!(shell.playhead(), Micros(0));
        let steps: Vec<(&str, bool)> = assistant
            .entries
            .iter()
            .filter_map(|entry| match entry {
                Entry::Step { tool, failed, .. } => Some((tool.as_str(), *failed)),
                _ => None,
            })
            .collect();
        // The hand-off shows first, then the helper's read.
        assert_eq!(steps, [("agent.delegate", false), ("draft.query", false)]);
        assert!(assistant.entries.iter().any(|entry| matches!(
            entry,
            Entry::Step { tool, result: Some(answer), .. }
                if tool == "agent.delegate" && answer == "It is short."
        )));
        assert_eq!(
            assistant.entries.last(),
            Some(&Entry::Said("The helper says it is short.".to_owned()))
        );
        assert!(assistant.history.iter().any(|message| matches!(
            message,
            Message::Tool { name, content, .. }
                if name == "agent.delegate" && content == "It is short."
        )));
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn a_call_that_needs_an_allow_waits_for_the_users_answer() {
        let (mut shell, path) = shell_with_project("ask");
        let taken = path.join("taken.png");
        std::fs::write(&taken, b"old").unwrap();
        let replace = || {
            completion(
                "",
                vec![ToolCall {
                    id: "1".to_owned(),
                    name: "export.start".to_owned(),
                    arguments: format!("replace {}", taken.display()),
                }],
            )
        };
        let mut assistant = Assistant::default();
        let model = LocalModel::scripted(vec![replace(), completion("Left it.", Vec::new())]);
        assistant.start(model, 8, "export over it");
        let deadline = Instant::now() + Duration::from_secs(10);
        while !assistant.asking() {
            assert!(assistant.poll(&mut shell), "the turn ended without asking");
            assert!(Instant::now() < deadline, "no question came");
            std::thread::sleep(Duration::from_millis(2));
        }
        // While it waits, nothing more runs.
        assert!(assistant.poll(&mut shell));
        assert!(matches!(
            assistant.entries.last(),
            Some(Entry::Ask { question, allowed: None }) if question.contains("Replace")
        ));
        assistant.decide(&mut shell, false);
        finish(&mut assistant, &mut shell);
        assert_eq!(std::fs::read(&taken).unwrap(), b"old");
        assert!(assistant.entries.iter().any(|entry| matches!(
            entry,
            Entry::Step { failed: true, result: Some(text), .. } if text.contains("the user said no")
        )));
        assert_eq!(
            assistant.entries.last(),
            Some(&Entry::Said("Left it.".to_owned()))
        );

        // Allowed, it runs.
        let model = LocalModel::scripted(vec![replace(), completion("Replaced.", Vec::new())]);
        assistant.start(model, 8, "now do");
        while !assistant.asking() {
            assert!(assistant.poll(&mut shell), "the turn ended without asking");
            std::thread::sleep(Duration::from_millis(2));
        }
        assistant.decide(&mut shell, true);
        finish(&mut assistant, &mut shell);
        assert_ne!(std::fs::read(&taken).unwrap(), b"old");
        assert!(assistant.entries.iter().any(|entry| matches!(
            entry,
            Entry::Ask {
                allowed: Some(true),
                ..
            }
        )));
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn a_model_error_and_a_stop_are_shown() {
        let (mut shell, path) = shell_with_project("stop");
        let mut assistant = Assistant::default();
        assistant.start(LocalModel::scripted(Vec::new()), 8, "hello");
        finish(&mut assistant, &mut shell);
        assert!(
            matches!(assistant.entries.last(), Some(Entry::Notice(text)) if text.contains("no more steps")),
            "{:?}",
            assistant.entries
        );
        assert!(assistant.history.is_empty());

        // A stopped turn changes nothing more, and says so.
        let model = LocalModel::scripted(vec![completion(
            "",
            vec![ToolCall {
                id: "1".to_owned(),
                name: "player.seek".to_owned(),
                arguments: "1000000".to_owned(),
            }],
        )]);
        assistant.start(model, 8, "seek");
        assistant.stop();
        assert!(!assistant.busy());
        assert!(!assistant.poll(&mut shell));
        assert_eq!(
            assistant.entries.last(),
            Some(&Entry::Notice("Stopped.".to_owned()))
        );
        assert_eq!(shell.playhead(), Micros(0));

        // The model off, or an address off this machine, says why.
        let mut settings = cutlass_settings::Settings::default();
        settings.model.local = false;
        assistant.draft = "hi".to_owned();
        assistant.send(&settings, "");
        assert!(
            matches!(assistant.entries.last(), Some(Entry::Notice(text)) if text.contains("Both models are off"))
        );
        // The cloud model with no name never posts; it says what to fill in.
        settings.model.cloud = true;
        settings.model.cloud_name.clear();
        assistant.draft = "hi".to_owned();
        assistant.send(&settings, "");
        assert!(
            matches!(assistant.entries.last(), Some(Entry::Notice(text)) if text.contains("Settings → Assistant"))
        );
        assert!(!assistant.busy());
        settings.model.cloud = false;
        settings.model.local = true;
        settings.model.local_url = "http://example.com/v1/chat/completions".to_owned();
        assistant.draft = "hi".to_owned();
        assistant.send(&settings, "");
        assert!(
            matches!(assistant.entries.last(), Some(Entry::Notice(text)) if text.contains("this machine"))
        );
        assert!(!assistant.busy());
        assistant.clear();
        assert!(assistant.entries.is_empty());
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn a_turn_that_hits_the_step_cap_or_is_stopped_is_remembered() {
        let (mut shell, path) = shell_with_project("remember");
        let mut assistant = Assistant::default();
        let seek = |id: &str, time: &str| {
            completion(
                "",
                vec![ToolCall {
                    id: id.to_owned(),
                    name: "player.seek".to_owned(),
                    arguments: time.to_owned(),
                }],
            )
        };
        // Two rounds allowed, and the model wants a third.
        let model = LocalModel::scripted(vec![
            seek("1", "1000000"),
            seek("2", "2000000"),
            completion("never said", Vec::new()),
        ]);
        assistant.start(model, 2, "move along");
        finish(&mut assistant, &mut shell);
        assert!(
            matches!(assistant.entries.last(), Some(Entry::Notice(text)) if text.contains("step cap")),
            "{:?}",
            assistant.entries
        );
        assert_eq!(shell.playhead(), Micros(2_000_000));
        // The ask, both replies and both results are kept for the next turn.
        assert_eq!(assistant.history.len(), 5, "{:?}", assistant.history);
        assert!(matches!(&assistant.history[0], Message::User { text } if text == "move along"));
        assert!(matches!(
            &assistant.history[4],
            Message::Tool { call_id, is_error: false, .. } if call_id == "2"
        ));
        // The next turn's model is sent them.
        let next = LocalModel::scripted(vec![completion("Carrying on.", Vec::new())]);
        assistant.start(next, 8, "and now?");
        finish(&mut assistant, &mut shell);
        assert_eq!(assistant.history.len(), 7);

        // A turn stopped while a call waits for an allow keeps the call,
        // answered as stopped, so the transcript has no call left open.
        let taken = path.join("taken.png");
        std::fs::write(&taken, b"old").unwrap();
        let model = LocalModel::scripted(vec![completion(
            "Replacing it.",
            vec![ToolCall {
                id: "r".to_owned(),
                name: "export.start".to_owned(),
                arguments: format!("replace {}", taken.display()),
            }],
        )]);
        assistant.start(model, 8, "export over it");
        let deadline = Instant::now() + Duration::from_secs(10);
        while !assistant.asking() {
            assert!(assistant.poll(&mut shell), "the turn ended without asking");
            assert!(Instant::now() < deadline, "no question came");
            std::thread::sleep(Duration::from_millis(2));
        }
        assistant.stop();
        assert_eq!(assistant.history.len(), 10, "{:?}", assistant.history);
        assert!(matches!(
            assistant.history.last(),
            Some(Message::Tool { call_id, content, is_error: true, .. })
                if call_id == "r" && content.contains("stopped")
        ));
        assert_eq!(std::fs::read(&taken).unwrap(), b"old");
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn an_empty_answer_is_said_and_a_call_asks_for_each_allow_it_needs() {
        let (mut shell, path) = shell_with_project("allows");
        let mut assistant = Assistant::default();
        assistant.start(
            LocalModel::scripted(vec![completion("", Vec::new())]),
            8,
            "hello",
        );
        finish(&mut assistant, &mut shell);
        assert!(
            matches!(assistant.entries.last(), Some(Entry::Notice(text)) if text.contains("without saying anything")),
            "{:?}",
            assistant.entries
        );

        // One settings change that needs two allows asks twice, then lands.
        let file = path.join("settings-home").join("settings.json");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        shell.keep_settings_file(file.clone(), 0);
        let model = LocalModel::scripted(vec![
            completion(
                "",
                vec![ToolCall {
                    id: "s".to_owned(),
                    name: "settings.set".to_owned(),
                    arguments: r#"{"expected_generation":0,"changes":{"model":{"cloud":true},"sandbox":{"online":true}}}"#.to_owned(),
                }],
            ),
            completion("Both on.", Vec::new()),
        ]);
        assistant.start(model, 8, "turn both on");
        let mut asked = 0;
        let deadline = Instant::now() + Duration::from_secs(10);
        while assistant.poll(&mut shell) {
            assert!(Instant::now() < deadline, "the turn did not finish");
            if assistant.asking() {
                asked += 1;
                assert!(!shell.settings().model.cloud, "written before every allow");
                assistant.decide(&mut shell, true);
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(asked, 2);
        assert!(shell.settings().model.cloud && shell.settings().sandbox.online);
        assert_eq!(
            assistant.entries.last(),
            Some(&Entry::Said("Both on.".to_owned()))
        );
        let _ = std::fs::remove_dir_all(&path);
    }

    /// A real model on this machine, named by `CUTLASS_LOCAL_MODEL`, reads
    /// the project and answers. Run by hand with Ollama up:
    /// `CUTLASS_LOCAL_MODEL=gemma4 cargo test -p cutlass real_local -- --ignored`.
    #[test]
    #[ignore = "needs a model running on this machine"]
    fn a_real_local_model_reads_the_project_and_answers() {
        let (mut shell, path) = shell_with_project("real");
        let mut settings = cutlass_settings::Settings::default();
        if let Ok(name) = std::env::var("CUTLASS_LOCAL_MODEL") {
            settings.model.local_name = name;
        }
        let mut assistant = Assistant {
            draft: "How many clips are on the main track? Read the project first.".to_owned(),
            ..Assistant::default()
        };
        assistant.send(&settings, "");
        let deadline = Instant::now() + Duration::from_secs(600);
        while assistant.poll(&mut shell) {
            assert!(Instant::now() < deadline, "the model did not answer");
            std::thread::sleep(Duration::from_millis(20));
        }
        eprintln!("{:#?}", assistant.entries);
        assert!(
            assistant
                .entries
                .iter()
                .any(|entry| matches!(entry, Entry::Step { tool, failed: false, .. } if tool == "draft.query")),
            "{:?}",
            assistant.entries
        );
        assert!(
            matches!(assistant.entries.last(), Some(Entry::Said(_))),
            "{:?}",
            assistant.entries
        );

        // And an edit: one split, one undo step in the window.
        assistant.entries.clear();
        assistant.draft = "Split clip a at 1 second.".to_owned();
        assistant.send(&settings, "");
        let deadline = Instant::now() + Duration::from_secs(600);
        while assistant.poll(&mut shell) {
            assert!(Instant::now() < deadline, "the model did not answer");
            std::thread::sleep(Duration::from_millis(20));
        }
        eprintln!("{:#?}", assistant.entries);
        let clips = shell.draft().unwrap().tracks[0].segments.len();
        assert_eq!(clips, 2, "{:?}", assistant.entries);
        shell.undo().unwrap();
        assert_eq!(shell.draft().unwrap().tracks[0].segments.len(), 1);

        // And a skill: it looks the bundled ones up before answering.
        assistant.entries.clear();
        assistant.draft = "Which skills can you load? Look them up.".to_owned();
        assistant.send(&settings, "");
        let deadline = Instant::now() + Duration::from_secs(600);
        while assistant.poll(&mut shell) {
            assert!(Instant::now() < deadline, "the model did not answer");
            std::thread::sleep(Duration::from_millis(20));
        }
        eprintln!("{:#?}", assistant.entries);
        assert!(
            assistant.entries.iter().any(|entry| matches!(
                entry,
                Entry::Step { tool, failed: false, .. } if tool == "skill.load"
            )),
            "{:?}",
            assistant.entries
        );

        // And a helper: the look-up is handed off and its answer comes back.
        assistant.entries.clear();
        assistant.draft =
            "Use a helper (agent.delegate) to find how long the main track is, then tell me."
                .to_owned();
        assistant.send(&settings, "");
        let deadline = Instant::now() + Duration::from_secs(600);
        while assistant.poll(&mut shell) {
            assert!(Instant::now() < deadline, "the model did not answer");
            std::thread::sleep(Duration::from_millis(20));
        }
        eprintln!("{:#?}", assistant.entries);
        assert!(
            assistant.history.iter().any(|message| matches!(
                message,
                Message::Tool { name, is_error: false, .. } if name == "agent.delegate"
            )),
            "{:?}",
            assistant.entries
        );
        let _ = std::fs::remove_dir_all(&path);
    }
}
