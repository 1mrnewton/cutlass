//! The assistant on a phone: the cloud model with the user's own key, the
//! tools the phone serves, and the conversation as lines to show.
//!
//! The model runs on its own thread. Each tool call goes to the project's
//! thread as any other edit does, and its line updates when it answers. The
//! phone asks for the lines while the assistant is busy; nothing calls
//! back into Kotlin or Swift.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, MutexGuard};

use cutlass_agent::{
    CloudModel, Completion, CompletionRequest, Delegating, Message, Model, ModelError,
    SettingsTool, ToolCall, ToolOutcome, ToolSink, delegate_definition, run_after,
};
use cutlass_ffi::{Home, PhoneHost};
use cutlass_settings::{ModelChoice, Settings};

use crate::{Editor, EditorError, Job, open_home};

/// What a line of the conversation is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum LineKind {
    /// What the user asked.
    Asked,
    /// What the model said.
    Said,
    /// A tool the model ran.
    Step,
    /// The turn stopped or failed, and why.
    Notice,
}

/// One line of the conversation.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Record)]
pub struct AssistantLine {
    pub kind: LineKind,
    /// The words, or for a step what the tool was given.
    pub text: String,
    /// The tool, for a step.
    pub tool: String,
    /// What the tool answered; empty while it runs.
    pub result: String,
    pub running: bool,
    pub failed: bool,
}

/// The cloud model as the phone's assistant settings show it. The key is
/// never read back; `has_key` says whether one is kept.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Record)]
pub struct CloudChoice {
    pub on: bool,
    /// `claude` or `openai` (any chat completions service).
    pub provider: String,
    pub name: String,
    /// The chat completions address for `openai`; empty takes OpenAI's.
    pub url: String,
    pub has_key: bool,
}

struct State {
    /// Where the conversation is kept while Memory is on.
    chat_file: Option<PathBuf>,
    lines: Vec<AssistantLine>,
    history: Vec<Message>,
    busy: bool,
    stop: Arc<AtomicBool>,
}

type Shared = Arc<Mutex<State>>;

fn lock(state: &Shared) -> MutexGuard<'_, State> {
    // A panic on the model's thread leaves the lines as they were.
    state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The assistant for one open project.
#[derive(uniffi::Object)]
pub struct Assistant {
    jobs: Sender<Job>,
    home: Home,
    state: Shared,
}

#[uniffi::export]
impl Assistant {
    /// The assistant for `editor`'s project, with its settings and keys in
    /// the Cutlass folder at `root`.
    #[uniffi::constructor]
    pub fn new(editor: &Arc<Editor>, root: String) -> Result<Arc<Self>, EditorError> {
        let home = open_home(&root)?;
        let project = editor.run(|host| host.root().to_path_buf())?;
        let memory = SettingsTool::open(home.settings_file())
            .map(|file| file.settings().memory)
            .unwrap_or(true);
        // Memory on, the project's conversation picks up where it was.
        let chat_file = memory.then(|| home.assistant_chat_file(&project));
        let history = chat_file
            .as_deref()
            .map(|file| {
                cutlass_agent::load_chat(file).unwrap_or_else(|error| {
                    cutlass_logger::warn!("assistant chat: {error}");
                    Vec::new()
                })
            })
            .unwrap_or_default();
        let lines = shown(&history);
        Ok(Arc::new(Self {
            jobs: editor.jobs.clone(),
            home,
            state: Arc::new(Mutex::new(State {
                chat_file,
                lines,
                history,
                busy: false,
                stop: Arc::new(AtomicBool::new(false)),
            })),
        }))
    }

    /// The conversation so far.
    pub fn lines(&self) -> Vec<AssistantLine> {
        lock(&self.state).lines.clone()
    }

    /// A turn is running.
    pub fn busy(&self) -> bool {
        lock(&self.state).busy
    }

    /// Ask. The answer arrives in [`Self::lines`]; nothing happens while a
    /// turn is running.
    pub fn send(&self, ask: String) {
        let ask = ask.trim().to_owned();
        let mut state = lock(&self.state);
        if ask.is_empty() || state.busy {
            return;
        }
        state.lines.push(line(LineKind::Asked, &ask));
        let settings = self.settings();
        let model = match build(&self.home, &settings) {
            Ok(model) => model,
            Err(notice) => {
                state.lines.push(line(LineKind::Notice, &notice));
                return;
            }
        };
        drop(state);
        let (home, child_settings) = (self.home.clone(), settings.clone());
        let child = move || build(&home, &child_settings);
        self.launch(ask, model, child, settings.step_cap);
    }

    /// Stop the turn running now. What it already changed stays, each
    /// change its own undo step.
    pub fn stop(&self) {
        lock(&self.state).stop.store(true, Ordering::Relaxed);
    }

    /// Start over: nothing said, and the model forgets the earlier turns.
    pub fn clear(&self) {
        let mut state = lock(&self.state);
        if state.busy {
            state.stop.store(true, Ordering::Relaxed);
        }
        state.lines.clear();
        state.history.clear();
        keep(&state);
    }

    /// The cloud model's settings.
    pub fn cloud(&self) -> CloudChoice {
        let choice = self.settings().model;
        let key =
            cutlass_agent_cloud::key(&self.home.assistant_keys_file(), &choice.cloud_provider);
        CloudChoice {
            on: choice.cloud,
            provider: choice.cloud_provider,
            name: choice.cloud_name,
            url: choice.cloud_url,
            has_key: !key.is_empty(),
        }
    }

    /// Change the cloud model's settings. `key`, when given, is kept for
    /// the provider in its own file, apart from the settings; an empty one
    /// forgets it. Turning the cloud model on sends what is asked off the
    /// phone, so the phone asks the user before calling this.
    pub fn set_cloud(&self, choice: CloudChoice, key: Option<String>) -> Result<(), EditorError> {
        let failed = |message: String| EditorError::Failed { message };
        if choice.provider != ModelChoice::CLAUDE && choice.provider != ModelChoice::OPENAI {
            return Err(failed(format!("no cloud service {}", choice.provider)));
        }
        if let Some(key) = key {
            cutlass_agent_cloud::set_key(&self.home.assistant_keys_file(), &choice.provider, &key)
                .map_err(failed)?;
        }
        let mut file = SettingsTool::open(self.home.settings_file())
            .map_err(|error| failed(error.to_string()))?;
        let mut settings = file.settings().clone();
        settings.model.cloud = choice.on;
        settings.model.cloud_provider = choice.provider;
        settings.model.cloud_name = choice.name.trim().to_owned();
        settings.model.cloud_url = choice.url.trim().to_owned();
        let generation = file.get().generation;
        file.set(generation, settings)
            .map(|_| ())
            .map_err(|error| failed(error.to_string()))
    }
}

impl Assistant {
    /// Run `ask` on `model` on its own thread; a helper runs on a model
    /// `child` makes. The ask is already shown.
    pub(crate) fn launch<M, C, F>(&self, ask: String, model: M, mut child: F, cap: u32)
    where
        M: Model + Send + 'static,
        C: Model + 'static,
        F: FnMut() -> Result<C, String> + Send + 'static,
    {
        let stop = Arc::new(AtomicBool::new(false));
        let history = {
            let mut state = lock(&self.state);
            state.stop = Arc::clone(&stop);
            state.busy = true;
            state.history.clone()
        };
        let shared = Arc::clone(&self.state);
        let sink = Relay {
            jobs: self.jobs.clone(),
            settings_file: self.home.settings_file(),
            state: Arc::clone(&self.state),
            stop: Arc::clone(&stop),
        };
        let child_stop = Arc::clone(&stop);
        let child = move || {
            child().map(|inner| Watched {
                inner,
                stop: Arc::clone(&child_stop),
                state: None,
            })
        };
        let mut watched = Watched {
            inner: model,
            stop,
            state: Some(Arc::clone(&self.state)),
        };
        let spawned = std::thread::Builder::new()
            .name("cutlass-assistant".to_owned())
            .spawn(move || {
                let mut tools = cutlass_ffi::phone_tools();
                tools.push(delegate_definition());
                let mut sink = Delegating::new(sink, child, cap);
                let result = run_after(&mut watched, &mut sink, cap, history, &ask, &tools);
                let mut state = lock(&shared);
                state.busy = false;
                match result {
                    Ok(turn) => {
                        if !turn.completion.text.is_empty() {
                            state
                                .lines
                                .push(line(LineKind::Said, &turn.completion.text));
                        }
                        state.history = turn.messages;
                        keep(&state);
                    }
                    Err(error) => {
                        let text = if state.stop.load(Ordering::Relaxed) {
                            "Stopped.".to_owned()
                        } else {
                            error.to_string()
                        };
                        state.lines.push(line(LineKind::Notice, &text));
                    }
                }
            });
        if let Err(error) = spawned {
            let mut state = lock(&self.state);
            state.busy = false;
            state.lines.push(line(
                LineKind::Notice,
                &format!("the assistant did not start: {error}"),
            ));
        }
    }

    fn settings(&self) -> Settings {
        SettingsTool::open(self.home.settings_file())
            .map(|file| file.settings().clone())
            .unwrap_or_default()
    }
}

/// The cloud model `settings` choose, with the kept key. A model on the
/// phone itself is not offered yet.
fn build(
    home: &Home,
    settings: &Settings,
) -> Result<CloudModel<cutlass_agent_cloud::Cloud>, String> {
    if !settings.model.cloud {
        return Err(
            "Turn on the cloud model with your own key in the assistant's settings. A model on the phone itself is not available yet."
                .to_owned(),
        );
    }
    let key = cutlass_agent_cloud::key(&home.assistant_keys_file(), &settings.model.cloud_provider);
    let mut context = cutlass_agent::prompt_context(settings, "");
    context.push_str(
        "\nThis is the phone app: you can read and change the draft, read and change settings, load skills and run scripts. Playback, export, speech, tracking and MCP servers are on the desktop app only.",
    );
    cutlass_agent_cloud::model(settings, &key, &context)
        .map(|model| CloudModel::from_settings(settings, model))
}

/// Write the conversation to its file, when it has one.
fn keep(state: &State) {
    if let Some(file) = &state.chat_file
        && let Err(error) = cutlass_agent::save_chat(file, &state.history)
    {
        cutlass_logger::warn!("assistant chat: {error}");
    }
}

/// The lines for a conversation read back: each tool call with its answer.
fn shown(history: &[Message]) -> Vec<AssistantLine> {
    let mut lines: Vec<AssistantLine> = Vec::new();
    for message in history {
        match message {
            Message::User { text } => lines.push(line(LineKind::Asked, text)),
            Message::Assistant { text, calls } => {
                if !text.is_empty() {
                    lines.push(line(LineKind::Said, text));
                }
                for call in calls {
                    lines.push(AssistantLine {
                        kind: LineKind::Step,
                        text: call.arguments.clone(),
                        tool: call.name.clone(),
                        result: String::new(),
                        running: true,
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
                if let Some(step) = lines
                    .iter_mut()
                    .rev()
                    .find(|line| line.running && line.tool == *name)
                {
                    step.running = false;
                    step.result.clone_from(content);
                    step.failed = *is_error;
                }
            }
        }
    }
    // A call with no answer saved did not finish.
    for step in &mut lines {
        step.running = false;
    }
    lines
}

fn line(kind: LineKind, text: &str) -> AssistantLine {
    AssistantLine {
        kind,
        text: text.to_owned(),
        tool: String::new(),
        result: String::new(),
        running: false,
        failed: false,
    }
}

/// The model, stopped with the turn; the parent's words between tool calls
/// are shown as they come.
struct Watched<M> {
    inner: M,
    stop: Arc<AtomicBool>,
    state: Option<Shared>,
}

impl<M: Model> Model for Watched<M> {
    fn complete(&mut self, request: &CompletionRequest) -> Result<Completion, ModelError> {
        if self.stop.load(Ordering::Relaxed) {
            return Err(ModelError::Inner("stopped".to_owned()));
        }
        let completion = self.inner.complete(request)?;
        if let Some(state) = &self.state
            && !completion.calls.is_empty()
            && !completion.text.is_empty()
        {
            lock(state)
                .lines
                .push(line(LineKind::Said, &completion.text));
        }
        Ok(completion)
    }
}

/// Tool calls, run on the project's thread; each shows as a line.
struct Relay {
    jobs: Sender<Job>,
    settings_file: PathBuf,
    state: Shared,
    stop: Arc<AtomicBool>,
}

impl Relay {
    fn begin(&self, call: &ToolCall) {
        lock(&self.state).lines.push(AssistantLine {
            kind: LineKind::Step,
            text: call.arguments.clone(),
            tool: call.name.clone(),
            result: String::new(),
            running: true,
            failed: false,
        });
    }

    fn end(&self, call: &ToolCall, outcome: &ToolOutcome) {
        let mut state = lock(&self.state);
        if let Some(step) = state
            .lines
            .iter_mut()
            .rev()
            .find(|line| line.running && line.tool == call.name)
        {
            step.running = false;
            step.result.clone_from(&outcome.content);
            step.failed = outcome.is_error;
        }
    }
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
        self.begin(call);
        let (reply, answer) = mpsc::sync_channel(1);
        let (job_call, settings_file) = (call.clone(), self.settings_file.clone());
        let job: Job = Box::new(move |host: &mut PhoneHost| {
            let _ = reply.send(host.run_agent_tool(&job_call, &settings_file));
        });
        let outcome = if self.jobs.send(job).is_ok() {
            answer.recv().unwrap_or_else(|_| stopped())
        } else {
            stopped()
        };
        self.end(call, &outcome);
        outcome
    }

    fn helper_started(&mut self, call: &ToolCall) {
        self.begin(call);
    }

    fn helper_finished(&mut self, call: &ToolCall, outcome: &ToolOutcome) {
        self.end(call, outcome);
    }
}
