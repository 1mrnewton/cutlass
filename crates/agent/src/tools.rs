//! The tool registry for the open draft, settings, and host services.
//!
//! This crate does not open media files and does not start downloads.
//! The traits are the host.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use cutlass_model::{Micros, SegmentId, TrackId};
use cutlass_settings::{McpServer, Settings};

use crate::approval::{ApprovalAction, Approver, authorize, authorize_then};
use crate::draft::{ApplyInput, DraftQuery, DraftTools};
use crate::r#loop::{ToolOutcome, ToolSink};
use crate::mcp::{McpClient, McpError, McpTransport};
use crate::model::{ToolCall, ToolDefinition};
use crate::sandbox::Sandbox;
use crate::settings_tool::{SettingsTool, SettingsToolError};
use crate::skill::{Skill, SkillLibrary};

/// The model has no approval flag. Host approval ignores this value.
const MODEL_SAYS_APPROVED: bool = false;

/// Tool names the model may call, in registry order.
pub const TOOL_NAMES: &[&str] = &[
    "draft.query",
    "draft.apply",
    "draft.undo",
    "draft.redo",
    "settings.get",
    "settings.set",
    "player.seek",
    "player.play",
    "player.pause",
    "media.probe",
    "speech.start",
    "speech.poll",
    "catalog.search",
    "package.ensure",
    "export.start",
    "sandbox.run",
    "skill.load",
    "mcp.call",
];

/// One definition per [`TOOL_NAMES`] entry, in the same order.
#[must_use]
pub fn tool_definitions() -> Vec<ToolDefinition> {
    TOOL_NAMES
        .iter()
        .map(|name| ToolDefinition {
            name: (*name).to_owned(),
            description: description(name).to_owned(),
        })
        .collect()
}

/// The open player. These calls do not push a draft undo step.
pub trait Player {
    /// Moves the playhead to `time_us` microseconds.
    fn seek(&mut self, time_us: i64) -> Result<(), String>;
    /// Starts playback.
    fn play(&mut self) -> Result<(), String>;
    /// Pauses playback.
    fn pause(&mut self) -> Result<(), String>;
}

/// Duration, picture size, and stream names from the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaInfo {
    /// Duration in microseconds.
    pub duration_us: i64,
    /// Picture width in pixels.
    pub width: i32,
    /// Picture height in pixels.
    pub height: i32,
    /// Stream names, in host order.
    pub streams: Vec<String>,
}

/// Probes a media file. The registry does not open the path.
pub trait Media {
    /// Returns duration, size, and streams for `path`.
    fn probe(&mut self, path: &str) -> Result<MediaInfo, String>;
}

/// A transcription job and the draft revision it started on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechJob {
    /// The id the host assigned.
    pub id: String,
    /// `draft.session().revision()` when the job started.
    pub revision: u64,
}

/// How a transcription poll resolved against the stored revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpeechPoll {
    /// The host has not finished the job.
    Pending,
    /// Word text from a job whose revision still matches.
    Ready {
        /// The words the host returned.
        words: String,
    },
    /// The draft revision moved, so the words are ignored.
    Stale,
}

/// Starts and polls transcription. The registry stores the draft revision.
pub trait Speech {
    /// Starts a job for `[start_us, start_us + duration_us)` and returns its id.
    fn start(&mut self, start_us: i64, duration_us: i64) -> Result<String, String>;
    /// Returns the words for `job_id`, or an error string from the host.
    fn poll(&mut self, job_id: &str) -> Result<String, String>;
}

/// Searches bundled effect and animation ids.
pub trait Catalog {
    /// Returns ids that match `query`.
    fn search(&mut self, query: &str) -> Result<Vec<String>, String>;
}

/// Makes a package available. The registry does not download it.
pub trait Packages {
    /// Asks the host to make `package_id` available.
    fn ensure(&mut self, package_id: &str) -> Result<(), String>;
}

/// Starts an export job.
pub trait Export {
    /// Starts an export to `path` and returns the job id.
    fn start(&mut self, path: &str) -> Result<String, String>;
}

/// One registry call, after arguments have been parsed or built by the host.
#[derive(Clone, Debug, PartialEq)]
pub enum Invocation {
    /// Read a compact draft view.
    DraftQuery(DraftQuery),
    /// Apply one combo, or preview it.
    DraftApply(ApplyInput),
    /// Undo the last committed combo.
    DraftUndo,
    /// Redo the combo that undo removed.
    DraftRedo,
    /// Read the settings generation and the switches that leave the machine.
    SettingsGet,
    /// Write settings when `expected` is the current generation.
    SettingsSet {
        /// The generation the caller read.
        expected: u64,
        /// The settings to store.
        settings: Settings,
    },
    /// Seek the open player.
    PlayerSeek {
        /// Playhead time in microseconds.
        time_us: i64,
    },
    /// Start playback.
    PlayerPlay,
    /// Pause playback.
    PlayerPause,
    /// Probe one media path through the host.
    MediaProbe {
        /// The path the host probes.
        path: String,
    },
    /// Start a transcription job.
    SpeechStart {
        /// Range start in microseconds.
        start_us: i64,
        /// Range length in microseconds.
        duration_us: i64,
    },
    /// Poll a transcription job.
    SpeechPoll {
        /// The id returned by start.
        job_id: String,
    },
    /// Search the catalog.
    CatalogSearch {
        /// The search text.
        query: String,
    },
    /// Ask the host to make a package available.
    PackageEnsure {
        /// The package id.
        package_id: String,
    },
    /// Start an export. `replace` asks before a file would be overwritten.
    ExportStart {
        /// The export path.
        path: String,
        /// Whether an existing file would be replaced.
        replace: bool,
    },
    /// Run source in the injected sandbox.
    SandboxRun {
        /// The source text.
        source: String,
        /// The JSON input.
        input_json: String,
    },
    /// Load the one skill that matches `ask`.
    SkillLoad {
        /// The user's ask.
        ask: String,
    },
    /// Add a skill folder from outside the app. This is not a model tool name.
    AddExternalSkill {
        /// The skill path.
        path: String,
    },
    /// Call a tool on one MCP server.
    McpCall {
        /// The server name.
        server: String,
        /// The tool name.
        tool: String,
        /// Opaque arguments.
        arguments: String,
    },
}

/// Why a registry call did not return a tool result.
#[derive(Debug, PartialEq, Eq)]
pub enum ToolError {
    /// The tool name is not in the registry.
    Unknown(String),
    /// The draft tool failed.
    Draft(crate::draft::DraftError),
    /// The settings tool failed.
    Settings(SettingsToolError),
    /// The host refused an approval.
    Approval(crate::approval::ApprovalError),
    /// A skill could not be added.
    Skill(crate::skill::SkillError),
    /// The sandbox did not return JSON.
    Sandbox(crate::sandbox::SandboxError),
    /// The MCP call failed.
    Mcp(McpError),
    /// A host trait returned an error string.
    Host(String),
    /// No skill matched the ask.
    SkillNotFound,
    /// `speech.poll` named a job this registry did not start.
    SpeechUnknown {
        /// The id that was asked for.
        job_id: String,
    },
    /// The argument text is not the grammar for that tool.
    BadArguments(String),
}

impl fmt::Display for ToolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(name) => write!(formatter, "unknown tool {name}"),
            Self::Draft(error) => write!(formatter, "{error}"),
            Self::Settings(error) => write!(formatter, "{error}"),
            Self::Approval(error) => write!(formatter, "{error}"),
            Self::Skill(error) => write!(formatter, "{error}"),
            Self::Sandbox(error) => write!(formatter, "{error}"),
            Self::Mcp(error) => write!(formatter, "{error}"),
            Self::Host(message) => write!(formatter, "{message}"),
            Self::SkillNotFound => write!(formatter, "skill not found"),
            Self::SpeechUnknown { job_id } => write!(formatter, "unknown speech job {job_id}"),
            Self::BadArguments(message) => write!(formatter, "{message}"),
        }
    }
}

impl std::error::Error for ToolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Draft(error) => Some(error),
            Self::Settings(error) => Some(error),
            Self::Approval(error) => Some(error),
            Self::Skill(error) => Some(error),
            Self::Sandbox(error) => Some(error),
            Self::Mcp(error) => Some(error),
            Self::Unknown(_)
            | Self::Host(_)
            | Self::SkillNotFound
            | Self::SpeechUnknown { .. }
            | Self::BadArguments(_) => None,
        }
    }
}

/// Draft, settings, skills, and the host services the registry calls.
pub struct Registry<P, Med, Sp, Cat, Pkg, Exp, Sand, Appr, Tr> {
    /// Draft queries and combos.
    pub draft: DraftTools,
    /// The settings file and its generation.
    pub settings: SettingsTool,
    /// Bundled skills and skills the host has allowed.
    pub skills: SkillLibrary,
    /// The open player.
    pub player: P,
    /// Media probes.
    pub media: Med,
    /// Transcription.
    pub speech: Sp,
    /// Catalog search.
    pub catalog: Cat,
    /// Package installs.
    pub packages: Pkg,
    /// Export jobs.
    pub export: Exp,
    /// The sandbox runner.
    pub sandbox: Sand,
    /// Host approval.
    pub approver: Appr,
    /// MCP calls.
    pub mcp: McpClient<Tr>,
    speech_jobs: BTreeMap<String, SpeechJob>,
}

impl<P, Med, Sp, Cat, Pkg, Exp, Sand, Appr, Tr> Registry<P, Med, Sp, Cat, Pkg, Exp, Sand, Appr, Tr>
where
    P: Player,
    Med: Media,
    Sp: Speech,
    Cat: Catalog,
    Pkg: Packages,
    Exp: Export,
    Sand: Sandbox,
    Appr: Approver,
    Tr: McpTransport,
{
    /// Builds a registry from the draft, settings, player, media, speech, catalog, and `(packages, export, sandbox, approver, mcp)`.
    #[must_use]
    pub fn new(
        draft: DraftTools,
        settings: SettingsTool,
        player: P,
        media: Med,
        speech: Sp,
        catalog: Cat,
        (packages, export, sandbox, approver, mcp): (Pkg, Exp, Sand, Appr, McpClient<Tr>),
    ) -> Self {
        Self {
            draft,
            settings,
            skills: SkillLibrary::bundled(),
            player,
            media,
            speech,
            catalog,
            packages,
            export,
            sandbox,
            approver,
            mcp,
            speech_jobs: BTreeMap::new(),
        }
    }

    /// Runs one call and returns its text.
    pub fn invoke(&mut self, call: Invocation) -> Result<String, ToolError> {
        match call {
            Invocation::DraftQuery(query) => self
                .draft
                .query(&query)
                .map(|view| view.to_string())
                .map_err(ToolError::Draft),
            Invocation::DraftApply(input) => self
                .draft
                .apply(input)
                .map(|result| {
                    format!(
                        "revision={} committed={}",
                        result.revision, result.committed
                    )
                })
                .map_err(ToolError::Draft),
            Invocation::DraftUndo => self
                .draft
                .undo()
                .map(|revision| format!("revision={revision}"))
                .map_err(ToolError::Draft),
            Invocation::DraftRedo => self
                .draft
                .redo()
                .map(|revision| format!("revision={revision}"))
                .map_err(ToolError::Draft),
            Invocation::SettingsGet => Ok(show_settings(&self.settings)),
            Invocation::SettingsSet { expected, settings } => {
                set_settings(&mut self.settings, &mut self.approver, expected, settings)
            }
            Invocation::PlayerSeek { time_us } => host_unit(self.player.seek(time_us)),
            Invocation::PlayerPlay => host_unit(self.player.play()),
            Invocation::PlayerPause => host_unit(self.player.pause()),
            Invocation::MediaProbe { path } => self
                .media
                .probe(&path)
                .map(|info| format_probe(&info))
                .map_err(ToolError::Host),
            Invocation::SpeechStart {
                start_us,
                duration_us,
            } => start_speech(
                &self.draft,
                &mut self.speech,
                &mut self.speech_jobs,
                start_us,
                duration_us,
            ),
            Invocation::SpeechPoll { job_id } => {
                poll_speech(&self.draft, &mut self.speech, &self.speech_jobs, &job_id)
            }
            Invocation::CatalogSearch { query } => self
                .catalog
                .search(&query)
                .map(|ids| format!("ids={}", ids.join(",")))
                .map_err(ToolError::Host),
            Invocation::PackageEnsure { package_id } => self
                .packages
                .ensure(&package_id)
                .map(|()| "ok".to_owned())
                .map_err(ToolError::Host),
            Invocation::ExportStart { path, replace } => {
                start_export(&mut self.export, &mut self.approver, &path, replace)
            }
            Invocation::SandboxRun { source, input_json } => self
                .sandbox
                .run(&source, &input_json)
                .map_err(ToolError::Sandbox),
            Invocation::SkillLoad { ask } => self
                .skills
                .load(&ask)
                .map(format_skill)
                .ok_or(ToolError::SkillNotFound),
            Invocation::AddExternalSkill { path } => {
                add_skill(&mut self.skills, &mut self.approver, Path::new(&path))
                    .map(|()| "ok".to_owned())
            }
            Invocation::McpCall {
                server,
                tool,
                arguments,
            } => call_mcp(
                &mut self.mcp,
                self.settings.settings(),
                &server,
                &tool,
                &arguments,
            ),
        }
    }

    /// Parses a model tool call. `draft.apply` and `settings.set` are rejected.
    pub fn parse(&self, name: &str, arguments: &str) -> Result<Invocation, ToolError> {
        let arguments = arguments.trim();
        match name {
            "draft.query" => parse_query(arguments),
            "draft.apply" => Err(bad("draft.apply")),
            "draft.undo" => bare(arguments, "draft.undo", Invocation::DraftUndo),
            "draft.redo" => bare(arguments, "draft.redo", Invocation::DraftRedo),
            "settings.get" => bare(arguments, "settings.get", Invocation::SettingsGet),
            "settings.set" => Err(bad("settings.set")),
            "player.seek" => {
                one_i64(arguments, "player.seek").map(|time_us| Invocation::PlayerSeek { time_us })
            }
            "player.play" => bare(arguments, "player.play", Invocation::PlayerPlay),
            "player.pause" => bare(arguments, "player.pause", Invocation::PlayerPause),
            "media.probe" => {
                text_arg(arguments, "media.probe").map(|path| Invocation::MediaProbe { path })
            }
            "speech.start" => {
                let (start_us, duration_us) = two_i64(arguments, "speech.start")?;
                Ok(Invocation::SpeechStart {
                    start_us,
                    duration_us,
                })
            }
            "speech.poll" => {
                text_arg(arguments, "speech.poll").map(|job_id| Invocation::SpeechPoll { job_id })
            }
            "catalog.search" => Ok(Invocation::CatalogSearch {
                query: arguments.to_owned(),
            }),
            "package.ensure" => text_arg(arguments, "package.ensure")
                .map(|package_id| Invocation::PackageEnsure { package_id }),
            "export.start" => parse_export(arguments),
            "sandbox.run" => Ok(parse_sandbox(arguments)),
            "skill.load" => Ok(Invocation::SkillLoad {
                ask: arguments.to_owned(),
            }),
            "mcp.call" => parse_mcp(arguments),
            other => Err(ToolError::Unknown(other.to_owned())),
        }
    }

    /// Adds a skill from outside the app after the host approves it.
    pub fn add_skill_outside(&mut self, path: &Path) -> Result<(), ToolError> {
        add_skill(&mut self.skills, &mut self.approver, path)
    }
}

impl<P, Med, Sp, Cat, Pkg, Exp, Sand, Appr, Tr> ToolSink
    for Registry<P, Med, Sp, Cat, Pkg, Exp, Sand, Appr, Tr>
where
    P: Player,
    Med: Media,
    Sp: Speech,
    Cat: Catalog,
    Pkg: Packages,
    Exp: Export,
    Sand: Sandbox,
    Appr: Approver,
    Tr: McpTransport,
{
    fn run_tool(&mut self, call: &ToolCall) -> ToolOutcome {
        let parsed = self.parse(&call.name, &call.arguments);
        let result = match parsed {
            Ok(invocation) => self.invoke(invocation),
            Err(error) => Err(error),
        };
        match result {
            Ok(content) => ToolOutcome {
                content,
                is_error: false,
            },
            Err(error) => ToolOutcome {
                content: error.to_string(),
                is_error: true,
            },
        }
    }
}

fn description(name: &str) -> &'static str {
    match name {
        "draft.query" => "Compact view of the open draft plus its revision.",
        "draft.apply" => "Session requests as one combo, or a dry run.",
        "draft.undo" => "Restore the draft from before the last committed combo.",
        "draft.redo" => "Reapply the combo that undo removed.",
        "settings.get" => "Read the settings generation and the switches that leave the machine.",
        "settings.set" => "Write user settings when the generation matches.",
        "player.seek" => "Move the open player to a time in microseconds.",
        "player.play" => "Start the open player.",
        "player.pause" => "Pause the open player.",
        "media.probe" => "Duration, size, and streams reported by the host for one path.",
        "speech.start" => "Start a transcription job and record the draft revision.",
        "speech.poll" => "Read a transcription job, ignoring it when the revision moved.",
        "catalog.search" => "Search bundled effect and animation ids.",
        "package.ensure" => "Ask the host to make one package available.",
        "export.start" => "Start an export. Replacing a file asks the user.",
        "sandbox.run" => "Run source in the injected sandbox and return JSON.",
        "skill.load" => "Return the one skill whose description matches the ask.",
        "mcp.call" => "Call a tool on a server the user enabled.",
        _ => "",
    }
}

fn show_settings(tool: &SettingsTool) -> String {
    let view = tool.get();
    format!(
        "generation={} cloud={} online={} step_cap={}",
        view.generation,
        view.settings.model.cloud,
        view.settings.sandbox.online,
        view.settings.step_cap
    )
}

fn set_settings(
    settings_tool: &mut SettingsTool,
    approver: &mut dyn Approver,
    expected: u64,
    next: Settings,
) -> Result<String, ToolError> {
    if expected != settings_tool.generation() {
        let actual = settings_tool.generation();
        return Err(ToolError::Settings(SettingsToolError::GenerationMismatch {
            expected,
            actual,
        }));
    }
    let actions = pending_approvals(settings_tool.settings(), &next);
    for action in actions {
        authorize(approver, action, MODEL_SAYS_APPROVED).map_err(ToolError::Approval)?;
    }
    let generation = settings_tool
        .set(expected, next)
        .map_err(ToolError::Settings)?;
    Ok(format!("generation={generation}"))
}

fn pending_approvals(current: &Settings, next: &Settings) -> Vec<ApprovalAction> {
    let mut actions = Vec::new();
    if !current.model.cloud && next.model.cloud {
        actions.push(ApprovalAction::EnableCloudModel);
    }
    if !current.sandbox.online && next.sandbox.online {
        actions.push(ApprovalAction::EnableOnlineSandbox);
    }
    for server in &next.mcp_servers {
        if server.kind == McpServer::REMOTE
            && server.enabled
            && !already_enabled_remote(current, &server.name)
        {
            actions.push(ApprovalAction::EnableRemoteMcp {
                server: server.name.clone(),
            });
        }
    }
    actions
}

fn already_enabled_remote(settings: &Settings, name: &str) -> bool {
    settings
        .mcp_servers
        .iter()
        .any(|server| server.name == name && server.kind == McpServer::REMOTE && server.enabled)
}

fn host_unit(result: Result<(), String>) -> Result<String, ToolError> {
    result.map(|()| "ok".to_owned()).map_err(ToolError::Host)
}

fn format_probe(info: &MediaInfo) -> String {
    format!(
        "duration={} width={} height={} streams={}",
        info.duration_us,
        info.width,
        info.height,
        info.streams.join(",")
    )
}

fn start_speech(
    draft: &DraftTools,
    speech: &mut impl Speech,
    jobs: &mut BTreeMap<String, SpeechJob>,
    start_us: i64,
    duration_us: i64,
) -> Result<String, ToolError> {
    let id = speech
        .start(start_us, duration_us)
        .map_err(ToolError::Host)?;
    let job = SpeechJob {
        id,
        revision: draft.session().revision(),
    };
    let text = format!("job={} revision={}", job.id, job.revision);
    jobs.insert(job.id.clone(), job);
    Ok(text)
}

fn poll_speech(
    draft: &DraftTools,
    speech: &mut impl Speech,
    jobs: &BTreeMap<String, SpeechJob>,
    job_id: &str,
) -> Result<String, ToolError> {
    let Some(job) = jobs.get(job_id) else {
        return Err(ToolError::SpeechUnknown {
            job_id: job_id.to_owned(),
        });
    };
    let stored = job.revision;
    let polled = speech.poll(job_id);
    if draft.session().revision() != stored {
        return Ok(format_poll(SpeechPoll::Stale));
    }
    let words = polled.map_err(ToolError::Host)?;
    Ok(format_poll(SpeechPoll::Ready { words }))
}

fn format_poll(poll: SpeechPoll) -> String {
    match poll {
        SpeechPoll::Pending => "pending".to_owned(),
        SpeechPoll::Ready { words } => format!("words={words}"),
        SpeechPoll::Stale => "stale".to_owned(),
    }
}

fn start_export(
    export: &mut impl Export,
    approver: &mut dyn Approver,
    path: &str,
    replace: bool,
) -> Result<String, ToolError> {
    if replace {
        let started = authorize_then(
            approver,
            ApprovalAction::ExportReplace {
                path: path.to_owned(),
            },
            MODEL_SAYS_APPROVED,
            || export.start(path),
        )
        .map_err(ToolError::Approval)?;
        return job_text(started);
    }
    job_text(export.start(path))
}

fn job_text(result: Result<String, String>) -> Result<String, ToolError> {
    result
        .map(|id| format!("job={id}"))
        .map_err(ToolError::Host)
}

fn add_skill(
    skills: &mut SkillLibrary,
    approver: &mut dyn Approver,
    path: &Path,
) -> Result<(), ToolError> {
    skills
        .add_outside(path, approver, MODEL_SAYS_APPROVED)
        .map_err(ToolError::Skill)
}

fn call_mcp(
    mcp: &mut McpClient<impl McpTransport>,
    settings: &Settings,
    server: &str,
    tool: &str,
    arguments: &str,
) -> Result<String, ToolError> {
    mcp.call(settings, server, tool, arguments)
        .map_err(ToolError::Mcp)
}

fn format_skill(skill: &Skill) -> String {
    format!(
        "title={}\ndescription={}\ntools={}\n{}",
        skill.title,
        skill.description,
        skill.allowed_tools.join(", "),
        skill.body
    )
}

fn parse_query(arguments: &str) -> Result<Invocation, ToolError> {
    if arguments == "project" {
        return Ok(Invocation::DraftQuery(DraftQuery::Project));
    }
    if arguments == "full" {
        return Ok(Invocation::DraftQuery(DraftQuery::Full));
    }
    if let Some(id) = headed(arguments, "track") {
        return Ok(Invocation::DraftQuery(DraftQuery::Track {
            track_id: TrackId::new(id),
        }));
    }
    if let Some(id) = headed(arguments, "segment") {
        return Ok(Invocation::DraftQuery(DraftQuery::Segment {
            segment_id: SegmentId::new(id),
        }));
    }
    if let Some(rest) = headed(arguments, "range") {
        let (start, end) = two_i64(rest, "draft.query")?;
        return Ok(Invocation::DraftQuery(DraftQuery::Range {
            start: Micros(start),
            end: Micros(end),
        }));
    }
    Err(bad("draft.query"))
}

fn parse_export(arguments: &str) -> Result<Invocation, ToolError> {
    if let Some(path) = headed(arguments, "replace") {
        return Ok(Invocation::ExportStart {
            path: path.to_owned(),
            replace: true,
        });
    }
    if let Some(path) = headed(arguments, "new") {
        return Ok(Invocation::ExportStart {
            path: path.to_owned(),
            replace: false,
        });
    }
    Err(bad("export.start"))
}

fn parse_sandbox(arguments: &str) -> Invocation {
    match arguments.split_once('\n') {
        Some((source, input_json)) => Invocation::SandboxRun {
            source: source.to_owned(),
            input_json: input_json.to_owned(),
        },
        None => Invocation::SandboxRun {
            source: arguments.to_owned(),
            input_json: "{}".to_owned(),
        },
    }
}

fn parse_mcp(arguments: &str) -> Result<Invocation, ToolError> {
    let Some((server, rest)) = arguments.split_once('\n') else {
        return Err(bad("mcp.call"));
    };
    let Some((tool, payload)) = rest.split_once('\n') else {
        return Err(bad("mcp.call"));
    };
    let server = server.trim();
    let tool = tool.trim();
    if server.is_empty() || tool.is_empty() {
        return Err(bad("mcp.call"));
    }
    Ok(Invocation::McpCall {
        server: server.to_owned(),
        tool: tool.to_owned(),
        arguments: payload.to_owned(),
    })
}

fn headed<'a>(arguments: &'a str, keyword: &str) -> Option<&'a str> {
    let rest = arguments.strip_prefix(keyword)?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim();
    if rest.is_empty() { None } else { Some(rest) }
}

fn bare(arguments: &str, name: &str, call: Invocation) -> Result<Invocation, ToolError> {
    if arguments.is_empty() {
        Ok(call)
    } else {
        Err(bad(name))
    }
}

fn text_arg(arguments: &str, name: &str) -> Result<String, ToolError> {
    if arguments.is_empty() {
        Err(bad(name))
    } else {
        Ok(arguments.to_owned())
    }
}

fn one_i64(arguments: &str, name: &str) -> Result<i64, ToolError> {
    let mut parts = arguments.split_whitespace();
    let Some(token) = parts.next() else {
        return Err(bad(name));
    };
    if parts.next().is_some() {
        return Err(bad(name));
    }
    parse_i64(token, name)
}

fn two_i64(arguments: &str, name: &str) -> Result<(i64, i64), ToolError> {
    let mut parts = arguments.split_whitespace();
    let Some(first) = parts.next() else {
        return Err(bad(name));
    };
    let Some(second) = parts.next() else {
        return Err(bad(name));
    };
    if parts.next().is_some() {
        return Err(bad(name));
    }
    Ok((parse_i64(first, name)?, parse_i64(second, name)?))
}

fn parse_i64(token: &str, name: &str) -> Result<i64, ToolError> {
    token.parse::<i64>().map_err(|_| bad(name))
}

fn bad(name: &str) -> ToolError {
    ToolError::BadArguments(format!("bad arguments for {name}"))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use cutlass_model::{Draft, Segment, TimeRange, TrackId};
    use cutlass_session::{Request, Session};
    use cutlass_settings::{McpServer, Settings, Shortcut};

    use crate::approval::{ApprovalAction, Approver};
    use crate::draft::{ApplyInput, ApplyMode, DraftError, DraftTools};
    use crate::r#loop::ToolSink;
    use crate::mcp::{McpClient, McpError, McpTransport};
    use crate::model::ToolCall;
    use crate::sandbox::{Sandbox, SandboxError};
    use crate::settings_tool::SettingsTool;
    use crate::skill::SkillError;

    use super::{
        Catalog, Export, Invocation, Media, MediaInfo, Packages, Player, Registry, Speech,
        TOOL_NAMES, ToolError, tool_definitions,
    };

    struct Temp {
        dir: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir()
                .join(format!("cutlass-agent-tools-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            Self { dir }
        }

        fn file(&self) -> PathBuf {
            self.dir.join("settings.json")
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[derive(Default)]
    struct CountPlayer {
        seek: u32,
        play: u32,
        pause: u32,
        time_us: i64,
    }

    impl Player for CountPlayer {
        fn seek(&mut self, time_us: i64) -> Result<(), String> {
            self.seek += 1;
            self.time_us = time_us;
            Ok(())
        }

        fn play(&mut self) -> Result<(), String> {
            self.play += 1;
            Ok(())
        }

        fn pause(&mut self) -> Result<(), String> {
            self.pause += 1;
            Ok(())
        }
    }

    struct Probe {
        calls: u32,
        path: String,
        info: MediaInfo,
    }

    impl Probe {
        fn sample() -> Self {
            Self {
                calls: 0,
                path: String::new(),
                info: MediaInfo {
                    duration_us: 1_500_000,
                    width: 1280,
                    height: 720,
                    streams: vec!["video".to_owned(), "audio".to_owned()],
                },
            }
        }
    }

    impl Media for Probe {
        fn probe(&mut self, path: &str) -> Result<MediaInfo, String> {
            self.calls += 1;
            self.path = path.to_owned();
            Ok(self.info.clone())
        }
    }

    struct Talk {
        starts: u32,
        polls: u32,
        start_us: i64,
        duration_us: i64,
        job_id: String,
        polled_id: String,
        words: String,
    }

    impl Talk {
        fn scripted() -> Self {
            Self {
                starts: 0,
                polls: 0,
                start_us: 0,
                duration_us: 0,
                job_id: "job-1".to_owned(),
                polled_id: String::new(),
                words: "hello from speech".to_owned(),
            }
        }
    }

    impl Speech for Talk {
        fn start(&mut self, start_us: i64, duration_us: i64) -> Result<String, String> {
            self.starts += 1;
            self.start_us = start_us;
            self.duration_us = duration_us;
            Ok(self.job_id.clone())
        }

        fn poll(&mut self, job_id: &str) -> Result<String, String> {
            self.polls += 1;
            self.polled_id = job_id.to_owned();
            Ok(self.words.clone())
        }
    }

    struct Search {
        calls: u32,
        query: String,
        ids: Vec<String>,
    }

    impl Search {
        fn fixed() -> Self {
            Self {
                calls: 0,
                query: String::new(),
                ids: vec!["wipe".to_owned(), "title".to_owned()],
            }
        }
    }

    impl Catalog for Search {
        fn search(&mut self, query: &str) -> Result<Vec<String>, String> {
            self.calls += 1;
            self.query = query.to_owned();
            Ok(self.ids.clone())
        }
    }

    #[derive(Default)]
    struct Pack {
        calls: u32,
        id: String,
    }

    impl Packages for Pack {
        fn ensure(&mut self, package_id: &str) -> Result<(), String> {
            self.calls += 1;
            self.id = package_id.to_owned();
            Ok(())
        }
    }

    #[derive(Default)]
    struct Exporter {
        calls: u32,
        path: String,
    }

    impl Export for Exporter {
        fn start(&mut self, path: &str) -> Result<String, String> {
            self.calls += 1;
            self.path = path.to_owned();
            Ok("export-1".to_owned())
        }
    }

    struct Runner {
        calls: u32,
        source: String,
        input: String,
        output: String,
    }

    impl Runner {
        fn json() -> Self {
            Self {
                calls: 0,
                source: String::new(),
                input: String::new(),
                output: "{}".to_owned(),
            }
        }
    }

    impl Sandbox for Runner {
        fn run(&mut self, source: &str, input_json: &str) -> Result<String, SandboxError> {
            self.calls += 1;
            self.source = source.to_owned();
            self.input = input_json.to_owned();
            Ok(self.output.clone())
        }
    }

    struct Gate {
        allow: bool,
        actions: Vec<ApprovalAction>,
    }

    impl Gate {
        fn refuse() -> Self {
            Self {
                allow: false,
                actions: Vec::new(),
            }
        }

        fn allow() -> Self {
            Self {
                allow: true,
                actions: Vec::new(),
            }
        }
    }

    impl Approver for Gate {
        fn decide(&mut self, action: &ApprovalAction) -> bool {
            self.actions.push(action.clone());
            self.allow
        }
    }

    struct CountMcp {
        calls: u32,
        seen: Vec<(String, String, String)>,
        output: String,
    }

    impl CountMcp {
        fn ready() -> Self {
            Self {
                calls: 0,
                seen: Vec::new(),
                output: "pong".to_owned(),
            }
        }
    }

    impl McpTransport for CountMcp {
        fn call(
            &mut self,
            server_name: &str,
            tool: &str,
            arguments: &str,
        ) -> Result<String, McpError> {
            self.calls += 1;
            self.seen.push((
                server_name.to_owned(),
                tool.to_owned(),
                arguments.to_owned(),
            ));
            Ok(self.output.clone())
        }
    }

    type TestRegistry =
        Registry<CountPlayer, Probe, Talk, Search, Pack, Exporter, Runner, Gate, CountMcp>;

    fn open(temp: &Temp, approver: Gate) -> TestRegistry {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        Registry::new(
            DraftTools::new(Session::open(draft)),
            SettingsTool::open(temp.file()).unwrap(),
            CountPlayer::default(),
            Probe::sample(),
            Talk::scripted(),
            Search::fixed(),
            (
                Pack::default(),
                Exporter::default(),
                Runner::json(),
                approver,
                McpClient::new(CountMcp::ready()),
            ),
        )
    }

    fn place() -> Request {
        Request::Place {
            track_id: TrackId::new("main"),
            segment: Box::new(Segment::new("a", "media", TimeRange::new(0, 1_000_000))),
        }
    }

    fn titles(registry: &TestRegistry, ask: &str) -> Vec<String> {
        registry
            .skills
            .for_prompt(ask)
            .into_iter()
            .map(|skill| skill.title.clone())
            .collect()
    }

    fn server(name: &str, enabled: bool, kind: &str, command: &str, url: &str) -> McpServer {
        McpServer {
            name: name.to_owned(),
            enabled,
            kind: kind.to_owned(),
            command: command.to_owned(),
            url: url.to_owned(),
            ..McpServer::default()
        }
    }

    #[test]
    fn registry_lists_exactly_the_tools() {
        assert_eq!(
            TOOL_NAMES,
            &[
                "draft.query",
                "draft.apply",
                "draft.undo",
                "draft.redo",
                "settings.get",
                "settings.set",
                "player.seek",
                "player.play",
                "player.pause",
                "media.probe",
                "speech.start",
                "speech.poll",
                "catalog.search",
                "package.ensure",
                "export.start",
                "sandbox.run",
                "skill.load",
                "mcp.call",
            ]
        );
        assert_eq!(TOOL_NAMES.len(), 18);
    }

    #[test]
    fn definitions_match_the_registry() {
        let definitions = tool_definitions();
        assert_eq!(definitions.len(), TOOL_NAMES.len());
        let mut seen = Vec::new();
        for (definition, name) in definitions.iter().zip(TOOL_NAMES.iter()) {
            assert_eq!(&definition.name, name);
            assert!(!definition.description.is_empty());
            assert!(!seen.contains(&definition.description.as_str()));
            seen.push(definition.description.as_str());
        }
    }

    #[test]
    fn draft_query_and_apply_round_trip() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        let project = registry.parse("draft.query", "project").unwrap();
        let text = registry.invoke(project).unwrap();
        assert!(text.contains("revision=0"));
        assert!(!text.contains("target_timerange"));
        assert_eq!(registry.draft.session().revision(), 0);
        assert!(matches!(
            registry.parse("draft.apply", "place"),
            Err(ToolError::BadArguments(_))
        ));
        assert!(matches!(
            registry.parse("draft.undo", "").unwrap(),
            Invocation::DraftUndo
        ));
        assert!(matches!(
            registry.parse("draft.redo", " \n").unwrap(),
            Invocation::DraftRedo
        ));
        assert!(matches!(
            registry.parse("draft.query", "range 1"),
            Err(ToolError::BadArguments(_))
        ));

        let applied = registry
            .invoke(Invocation::DraftApply(ApplyInput {
                request_id: None,
                expected_revision: 0,
                requests: vec![place()],
                mode: ApplyMode::Commit,
            }))
            .unwrap();
        assert_eq!(applied, "revision=1 committed=true");
        assert_eq!(registry.draft.session().revision(), 1);

        let track = registry
            .invoke(registry.parse("draft.query", "track main").unwrap())
            .unwrap();
        assert!(track.split_whitespace().any(|word| word == "a"));
        assert!(track.contains("track=main"));
        let segment = registry
            .invoke(registry.parse("draft.query", "segment a").unwrap())
            .unwrap();
        assert!(segment.contains("segment=a"));
        let range = registry
            .invoke(registry.parse("draft.query", "range 0 1000000").unwrap())
            .unwrap();
        assert!(range.split_whitespace().any(|word| word == "a"));
        let full = registry
            .invoke(registry.parse("draft.query", "full").unwrap())
            .unwrap();
        assert!(full.contains("revision=1"));
        assert!(full.split_whitespace().any(|word| word == "full"));

        assert_eq!(
            registry.invoke(Invocation::DraftUndo).unwrap(),
            "revision=2"
        );
        assert!(
            registry.draft.session().draft().tracks[0]
                .segments
                .is_empty()
        );
        assert_eq!(
            registry.invoke(Invocation::DraftRedo).unwrap(),
            "revision=3"
        );
        assert_eq!(registry.draft.session().draft().tracks[0].segments.len(), 1);
    }

    #[test]
    fn player_does_not_push_undo() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        assert!(matches!(
            registry.parse("player.seek", "5").unwrap(),
            Invocation::PlayerSeek { time_us: 5 }
        ));
        assert!(matches!(
            registry.parse("player.play", "").unwrap(),
            Invocation::PlayerPlay
        ));
        assert!(matches!(
            registry.parse("player.pause", "  ").unwrap(),
            Invocation::PlayerPause
        ));
        assert_eq!(
            registry
                .invoke(Invocation::PlayerSeek { time_us: 5 })
                .unwrap(),
            "ok"
        );
        assert_eq!(registry.invoke(Invocation::PlayerPlay).unwrap(), "ok");
        assert_eq!(registry.invoke(Invocation::PlayerPause).unwrap(), "ok");
        assert_eq!(registry.player.seek, 1);
        assert_eq!(registry.player.play, 1);
        assert_eq!(registry.player.pause, 1);
        assert_eq!(registry.player.time_us, 5);
        assert_eq!(registry.draft.session().revision(), 0);
        assert!(matches!(
            registry.invoke(Invocation::DraftUndo),
            Err(ToolError::Draft(DraftError::EmptyUndo))
        ));
    }

    #[test]
    fn media_probe_uses_the_injected_trait() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        let call = registry.parse("media.probe", "my clip.mov").unwrap();
        let text = registry.invoke(call).unwrap();
        assert_eq!(
            text,
            "duration=1500000 width=1280 height=720 streams=video,audio"
        );
        assert_eq!(registry.media.calls, 1);
        assert_eq!(registry.media.path, "my clip.mov");
        assert_eq!(registry.draft.session().revision(), 0);
    }

    #[test]
    fn speech_poll_ignores_a_stale_revision() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        assert!(matches!(
            registry.invoke(Invocation::SpeechPoll {
                job_id: "missing".to_owned(),
            }),
            Err(ToolError::SpeechUnknown { job_id }) if job_id == "missing"
        ));
        assert_eq!(registry.speech.polls, 0);
        let started = registry
            .invoke(registry.parse("speech.start", "10 20").unwrap())
            .unwrap();
        assert_eq!(started, "job=job-1 revision=0");
        assert_eq!(registry.speech.starts, 1);
        assert_eq!(registry.speech.start_us, 10);
        assert_eq!(registry.speech.duration_us, 20);
        let ready = registry
            .invoke(Invocation::SpeechPoll {
                job_id: "job-1".to_owned(),
            })
            .unwrap();
        assert_eq!(ready, "words=hello from speech");
        registry
            .invoke(Invocation::DraftApply(ApplyInput {
                request_id: None,
                expected_revision: 0,
                requests: vec![place()],
                mode: ApplyMode::Commit,
            }))
            .unwrap();
        assert_eq!(registry.draft.session().revision(), 1);
        let stale = registry
            .invoke(Invocation::SpeechPoll {
                job_id: "job-1".to_owned(),
            })
            .unwrap();
        assert_eq!(stale, "stale");
        assert!(!stale.contains("hello"));
        assert_eq!(registry.speech.polls, 2);
        assert_eq!(registry.speech.polled_id, "job-1");
    }

    #[test]
    fn catalog_search_returns_ids() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        let text = registry
            .invoke(registry.parse("catalog.search", "soft wipe").unwrap())
            .unwrap();
        assert_eq!(text, "ids=wipe,title");
        assert_eq!(registry.catalog.calls, 1);
        assert_eq!(registry.catalog.query, "soft wipe");
    }

    #[test]
    fn package_ensure_does_not_download() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        let text = registry
            .invoke(registry.parse("package.ensure", "titles pack").unwrap())
            .unwrap();
        assert_eq!(text, "ok");
        assert_eq!(registry.packages.calls, 1);
        assert_eq!(registry.packages.id, "titles pack");
        assert_eq!(registry.draft.session().revision(), 0);
        assert_eq!(registry.settings.generation(), 0);
        assert!(!registry.settings.path().exists());
    }

    #[test]
    fn export_replace_refusal_does_not_start() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        let call = registry
            .parse("export.start", "replace out file.mp4")
            .unwrap();
        let error = registry.invoke(call).unwrap_err();
        assert!(error.to_string().contains("refused"));
        assert!(matches!(
            error,
            ToolError::Approval(refused) if refused.action == ApprovalAction::ExportReplace { path: "out file.mp4".to_owned() }
        ));
        assert_eq!(registry.export.calls, 0);
        assert_eq!(registry.draft.session().revision(), 0);
        assert_eq!(
            registry.approver.actions,
            vec![ApprovalAction::ExportReplace {
                path: "out file.mp4".to_owned(),
            }]
        );
    }

    #[test]
    fn export_replace_allow_starts() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::allow());
        let created = registry
            .invoke(registry.parse("export.start", "new fresh.mp4").unwrap())
            .unwrap();
        assert_eq!(created, "job=export-1");
        assert!(registry.approver.actions.is_empty());
        assert_eq!(registry.export.calls, 1);
        assert_eq!(registry.export.path, "fresh.mp4");
        let replaced = registry
            .invoke(
                registry
                    .parse("export.start", "replace final cut.mp4")
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(replaced, "job=export-1");
        assert_eq!(registry.export.calls, 2);
        assert_eq!(registry.export.path, "final cut.mp4");
        assert_eq!(registry.draft.session().revision(), 0);
        assert_eq!(
            registry.approver.actions,
            vec![ApprovalAction::ExportReplace {
                path: "final cut.mp4".to_owned(),
            }]
        );
    }

    #[test]
    fn cloud_model_refusal_does_not_write_settings() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        let before = registry.invoke(Invocation::SettingsGet).unwrap();
        assert_eq!(
            before,
            format!(
                "generation=0 cloud=false online=false step_cap={}",
                cutlass_settings::DEFAULT_STEP_CAP
            )
        );
        let mut settings = registry.settings.settings().clone();
        settings.model.cloud = true;
        let error = registry
            .invoke(Invocation::SettingsSet {
                expected: 0,
                settings,
            })
            .unwrap_err();
        assert!(error.to_string().contains("refused"));
        assert_eq!(
            registry.approver.actions,
            vec![ApprovalAction::EnableCloudModel]
        );
        assert_eq!(registry.settings.generation(), 0);
        assert!(!registry.settings.settings().model.cloud);
        assert!(!registry.settings.path().exists());
        let after = registry.invoke(Invocation::SettingsGet).unwrap();
        assert_eq!(after, before);
    }

    #[test]
    fn online_sandbox_refusal_does_not_write_settings() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        let mut settings = registry.settings.settings().clone();
        settings.sandbox.online = true;
        let error = registry
            .invoke(Invocation::SettingsSet {
                expected: 0,
                settings,
            })
            .unwrap_err();
        assert!(matches!(
            error,
            ToolError::Approval(refused) if refused.action == ApprovalAction::EnableOnlineSandbox
        ));
        assert_eq!(registry.settings.generation(), 0);
        assert!(!registry.settings.settings().sandbox.online);
        assert!(registry.settings.settings().sandbox.local);
        assert!(!registry.settings.settings().model.cloud);
        assert!(!registry.settings.path().exists());
    }

    #[test]
    fn remote_mcp_refusal_does_not_write_settings() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        let mut settings = registry.settings.settings().clone();
        settings.mcp_servers.push(server(
            "far",
            true,
            McpServer::REMOTE,
            "",
            "http://127.0.0.1:9/mcp",
        ));
        let error = registry
            .invoke(Invocation::SettingsSet {
                expected: 0,
                settings,
            })
            .unwrap_err();
        assert!(error.to_string().contains("refused"));
        assert_eq!(
            registry.approver.actions,
            vec![ApprovalAction::EnableRemoteMcp {
                server: "far".to_owned(),
            }]
        );
        assert_eq!(registry.settings.generation(), 0);
        assert!(registry.settings.settings().mcp_servers.is_empty());
        assert!(!registry.settings.settings().model.cloud);
        assert!(!registry.settings.settings().sandbox.online);
        assert!(!registry.settings.path().exists());
        assert_eq!(registry.mcp.transport().calls, 0);
    }

    #[test]
    fn enabling_stdio_does_not_ask() {
        let temp = Temp::new();
        let mut initial = Settings::default();
        initial.mcp_servers.push(server(
            "far",
            true,
            McpServer::REMOTE,
            "",
            "http://127.0.0.1:9/mcp",
        ));
        cutlass_settings::save(&temp.file(), 0, &initial).unwrap();
        let mut registry = open(&temp, Gate::refuse());
        assert_eq!(registry.settings.generation(), 1);
        let mut next = registry.settings.settings().clone();
        next.step_cap = 40;
        next.shortcuts.push(Shortcut {
            action: "split".to_owned(),
            keys: "s".to_owned(),
            ..Shortcut::default()
        });
        next.mcp_servers
            .push(server("local", true, McpServer::STDIO, "tool", ""));
        let text = registry
            .invoke(Invocation::SettingsSet {
                expected: 1,
                settings: next,
            })
            .unwrap();
        assert_eq!(text, "generation=2");
        assert!(registry.approver.actions.is_empty());
        assert!(!registry.approver.allow);
        assert_eq!(registry.settings.generation(), 2);
        assert_eq!(registry.settings.settings().step_cap, 40);
        assert!(!registry.settings.settings().model.cloud);
        assert!(!registry.settings.settings().sandbox.online);
        let loaded = SettingsTool::open(registry.settings.path()).unwrap();
        assert_eq!(loaded.generation(), 2);
        assert_eq!(loaded.settings().step_cap, 40);
        assert_eq!(loaded.settings().shortcuts.len(), 1);
        assert_eq!(loaded.settings().shortcuts[0].action, "split");
        assert_eq!(loaded.settings().shortcuts[0].keys, "s");
        assert!(
            loaded.settings().mcp_servers.iter().any(|item| {
                item.name == "local" && item.enabled && item.kind == McpServer::STDIO
            })
        );
        assert!(
            loaded.settings().mcp_servers.iter().any(|item| {
                item.name == "far" && item.enabled && item.kind == McpServer::REMOTE
            })
        );
    }

    #[test]
    fn model_approved_text_does_not_skip_the_gate() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        assert!(matches!(
            registry.parse("settings.set", "approved=true"),
            Err(ToolError::BadArguments(_))
        ));
        assert!(matches!(
            registry.parse("settings.set", ""),
            Err(ToolError::BadArguments(_))
        ));
        assert!(registry.approver.actions.is_empty());
        assert_eq!(registry.settings.generation(), 0);
        assert!(!registry.settings.settings().model.cloud);
        let mut settings = registry.settings.settings().clone();
        settings.model.cloud = true;
        let error = registry
            .invoke(Invocation::SettingsSet {
                expected: 0,
                settings,
            })
            .unwrap_err();
        assert!(matches!(
            error,
            ToolError::Approval(refused) if refused.action == ApprovalAction::EnableCloudModel
        ));
        assert!(!registry.settings.settings().model.cloud);
        assert_eq!(registry.settings.generation(), 0);
        assert_eq!(
            registry.approver.actions,
            vec![ApprovalAction::EnableCloudModel]
        );
    }

    #[test]
    fn sandbox_run_uses_the_injected_runner() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        registry.sandbox.output = "{\"ran\":true}".to_owned();
        assert_eq!(
            registry.parse("sandbox.run", "only source").unwrap(),
            Invocation::SandboxRun {
                source: "only source".to_owned(),
                input_json: "{}".to_owned(),
            }
        );
        let call = registry.parse("sandbox.run", "gaps\n{\"a\":1}").unwrap();
        let text = registry.invoke(call).unwrap();
        assert_eq!(text, "{\"ran\":true}");
        assert_eq!(registry.sandbox.calls, 1);
        assert_eq!(registry.sandbox.source, "gaps");
        assert_eq!(registry.sandbox.input, "{\"a\":1}");
    }

    #[test]
    fn skill_load_returns_one_skill() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        let call = registry.parse("skill.load", "remove pauses").unwrap();
        let text = registry.invoke(call).unwrap();
        assert!(text.contains("Remove pauses"));
        assert!(!text.contains("Motion graphics"));
        assert!(!text.contains("Product demo"));
        assert!(!text.contains("Captions"));
        assert!(!text.contains("Project setup"));
    }

    #[test]
    fn external_skill_refusal_does_not_add() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        let pauses = titles(&registry, "remove pauses");
        let hello = titles(&registry, "hello");
        assert_eq!(pauses, vec!["Remove pauses".to_owned()]);
        assert!(hello.is_empty());
        let path = temp.dir.join("outside").join("SKILL.md");
        let error = registry.add_skill_outside(&path).unwrap_err();
        assert!(error.to_string().contains("refused"));
        assert!(matches!(
            error,
            ToolError::Skill(SkillError::ApprovalRefused)
        ));
        let again = registry
            .invoke(Invocation::AddExternalSkill {
                path: path.display().to_string(),
            })
            .unwrap_err();
        assert!(matches!(
            again,
            ToolError::Skill(SkillError::ApprovalRefused)
        ));
        assert_eq!(titles(&registry, "remove pauses"), pauses);
        assert_eq!(titles(&registry, "hello"), hello);
        assert!(registry.skills.load("remove pauses").is_some());
        assert_eq!(registry.approver.actions.len(), 2);
        assert!(registry.approver.actions.iter().all(|action| {
            matches!(action, ApprovalAction::AddExternalSkill { path } if path.contains("SKILL.md"))
        }));
    }

    #[test]
    fn mcp_call_reaches_only_an_enabled_server() {
        let disabled = Temp::new();
        let mut off = Settings::default();
        off.mcp_servers.push(server(
            "far",
            false,
            McpServer::REMOTE,
            "",
            "http://127.0.0.1:9/mcp",
        ));
        cutlass_settings::save(&disabled.file(), 0, &off).unwrap();
        let mut blocked = open(&disabled, Gate::refuse());
        let error = blocked
            .invoke(Invocation::McpCall {
                server: "far".to_owned(),
                tool: "ping".to_owned(),
                arguments: "hello".to_owned(),
            })
            .unwrap_err();
        assert!(error.to_string().contains("disabled"));
        assert!(matches!(
            error,
            ToolError::Mcp(McpError::Disabled { name, kind })
                if name == "far" && kind == McpServer::REMOTE
        ));
        assert_eq!(blocked.mcp.transport().calls, 0);

        let enabled = Temp::new();
        let mut on = Settings::default();
        on.mcp_servers.push(server(
            "local",
            true,
            McpServer::STDIO,
            "do-not-run",
            "http://127.0.0.1:9/mcp",
        ));
        cutlass_settings::save(&enabled.file(), 0, &on).unwrap();
        let mut registry = open(&enabled, Gate::refuse());
        assert_eq!(
            registry
                .parse("mcp.call", "local\nping\nhello\nthere")
                .unwrap(),
            Invocation::McpCall {
                server: "local".to_owned(),
                tool: "ping".to_owned(),
                arguments: "hello\nthere".to_owned(),
            }
        );
        let text = registry
            .invoke(Invocation::McpCall {
                server: "local".to_owned(),
                tool: "ping".to_owned(),
                arguments: "approved=true".to_owned(),
            })
            .unwrap();
        assert_eq!(text, "pong");
        assert_eq!(registry.mcp.transport().calls, 1);
        assert_eq!(
            registry.mcp.transport().seen,
            vec![(
                "local".to_owned(),
                "ping".to_owned(),
                "approved=true".to_owned()
            )]
        );
        assert!(registry.approver.actions.is_empty());
    }

    #[test]
    fn sink_runs_a_parsed_tool() {
        let temp = Temp::new();
        let mut registry = open(&temp, Gate::refuse());
        let outcome = registry.run_tool(&ToolCall {
            id: "1".to_owned(),
            name: "draft.query".to_owned(),
            arguments: "project".to_owned(),
        });
        assert!(!outcome.is_error);
        assert!(outcome.content.contains("revision=0"));
        assert_eq!(registry.draft.session().revision(), 0);
        let missing = registry.run_tool(&ToolCall {
            id: "2".to_owned(),
            name: "not.a.tool".to_owned(),
            arguments: String::new(),
        });
        assert!(missing.is_error);
        assert!(missing.content.contains("unknown"));
        assert!(missing.content.contains("not.a.tool"));
        let rejected = registry.run_tool(&ToolCall {
            id: "3".to_owned(),
            name: "draft.apply".to_owned(),
            arguments: "approved=true".to_owned(),
        });
        assert!(rejected.is_error);
        assert!(rejected.content.contains("bad arguments"));
        assert_eq!(registry.draft.session().revision(), 0);
    }
}
