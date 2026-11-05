//! The assistant's conversation: what was said, each tool call as a step,
//! and the turn running now.
//!
//! The model loop runs on its own thread, off the frame path. Each tool
//! call comes back to the window's thread, which runs it on the open
//! project between frames and sends the result back. Stop lets the turn go
//! at once: the thread finds out at its next call and ends.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};

use cutlass_agent::{
    ChatModel, Completion, CompletionRequest, LoopbackHttp, Message, Model, ModelError, ToolCall,
    ToolOutcome, ToolSink, run_after, tool_definitions,
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
}

/// From the model's thread to the window's.
enum Event {
    Said(String),
    Tool {
        call: ToolCall,
        answer: Sender<ToolOutcome>,
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
    running: Option<Running>,
}

impl Assistant {
    /// A turn is running.
    #[must_use]
    pub fn busy(&self) -> bool {
        self.running.is_some()
    }

    /// Send the words in the box with the model `settings` choose.
    pub fn send(&mut self, settings: &Settings) {
        let ask = self.draft.trim().to_owned();
        if ask.is_empty() || self.busy() {
            return;
        }
        self.draft.clear();
        self.entries.push(Entry::Asked(ask.clone()));
        if !settings.model.local {
            self.entries.push(Entry::Notice(
                "The local model is off. Turn it on in Settings → Assistant; the cloud model needs an account, which is not set up yet."
                    .to_owned(),
            ));
            return;
        }
        let transport = match LoopbackHttp::new(&settings.model.local_url) {
            Ok(transport) => transport,
            Err(error) => {
                self.entries.push(Entry::Notice(error.to_string()));
                return;
            }
        };
        let model = ChatModel::new(transport, &settings.model.local_name);
        self.start(model, settings.step_cap, &ask);
    }

    /// Run `ask` on `model` on its own thread.
    pub fn start<M: Model + Send + 'static>(&mut self, model: M, cap: u32, ask: &str) {
        let (events, received) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
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
        let spawned = std::thread::Builder::new()
            .name("assistant".to_owned())
            .spawn(move || {
                let (mut watched, mut relay) = (watched, relay);
                let definitions = tool_definitions();
                let result = run_after(&mut watched, &mut relay, cap, history, &ask, &definitions)
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
        if let Some(running) = self.running.take() {
            running.stop.store(true, Ordering::Relaxed);
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
    pub fn clear(&mut self) {
        self.stop();
        self.entries.clear();
        self.history.clear();
    }

    /// Take what the model's thread sent: run its tool calls on `shell`
    /// and show what it said. True while a turn is still running.
    pub fn poll(&mut self, shell: &mut Shell) -> bool {
        loop {
            let Some(running) = self.running.as_ref() else {
                return false;
            };
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
                Event::Tool { call, answer } => {
                    let outcome = shell.run_agent_tool(&call);
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
                            if !text.is_empty() {
                                self.entries.push(Entry::Said(text));
                            }
                            self.history = messages;
                        }
                        Err(error) => self.entries.push(Entry::Notice(error)),
                    }
                    return false;
                }
            }
        }
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
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use cutlass_agent::{Completion, LocalModel, ToolCall};
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
        assistant.send(&settings);
        assert!(
            matches!(assistant.entries.last(), Some(Entry::Notice(text)) if text.contains("local model is off"))
        );
        settings.model.local = true;
        settings.model.local_url = "http://example.com/v1/chat/completions".to_owned();
        assistant.draft = "hi".to_owned();
        assistant.send(&settings);
        assert!(
            matches!(assistant.entries.last(), Some(Entry::Notice(text)) if text.contains("this machine"))
        );
        assert!(!assistant.busy());
        assistant.clear();
        assert!(assistant.entries.is_empty());
        let _ = std::fs::remove_dir_all(&path);
    }
}
