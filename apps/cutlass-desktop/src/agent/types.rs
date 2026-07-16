use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;

use crossbeam_channel::Sender;
use cutlass_ai::{EditorContext, WireCommand};
use cutlass_jobs::JobManager;

use crate::cache_registry::CacheRegistry;
use crate::{AgentStore, AppWindow};

/// An entity id the sandbox allocated while rehearsing a command. Replay
/// maps it onto the id the live engine allocates for the same step.
#[derive(Debug, Clone, Copy)]
pub enum AgentCreated {
    Clip(u64),
    Track(u64),
    Marker(u64),
}

/// One rehearsed command, ready for live replay.
#[derive(Debug, Clone)]
pub struct AgentPlanStep {
    pub command: WireCommand,
    /// Sandbox id this step created (`split_clip`'s right half,
    /// `add_track`'s lane, …), if any.
    pub created: Option<AgentCreated>,
}

pub(crate) enum AgentRequest {
    Prompt {
        prompt: String,
        context: EditorContext,
        dry_run: bool,
    },
    ApplyPlan,
    DiscardPlan,
    NewChat,
    SelectChat {
        id: String,
    },
    /// Persist the outgoing draft's conversation and restore the incoming
    /// draft. A missing path means no app-owned project is active.
    SwitchProject {
        path: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ApprovalChoice {
    Approve,
    Deny,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ApprovalDecision {
    pub(crate) request_id: u64,
    pub(crate) choice: ApprovalChoice,
}

#[derive(Clone)]
pub struct AgentHandle {
    pub(crate) tx: Sender<AgentRequest>,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) approval_tx: Sender<ApprovalDecision>,
    pub(crate) pending_approval_id: Arc<AtomicU64>,
}

impl AgentHandle {
    pub fn prompt(&self, prompt: String, context: EditorContext, dry_run: bool) {
        let _ = self.tx.send(AgentRequest::Prompt {
            prompt,
            context,
            dry_run,
        });
    }

    pub fn apply_plan(&self) {
        let _ = self.tx.send(AgentRequest::ApplyPlan);
    }

    pub fn discard_plan(&self) {
        let _ = self.tx.send(AgentRequest::DiscardPlan);
    }

    pub fn new_chat(&self) {
        let _ = self.tx.send(AgentRequest::NewChat);
    }

    pub fn select_chat(&self, id: String) {
        let _ = self.tx.send(AgentRequest::SelectChat { id });
    }

    /// Persist the outgoing session and restore the incoming draft's
    /// conversation. Fired after the worker publishes a new project path.
    pub fn switch_project(&self, path: Option<PathBuf>) {
        let _ = self.tx.send(AgentRequest::SwitchProject { path });
    }

    /// Cooperative cancel: the provider checks this flag between stream
    /// chunks, so a running prompt aborts within one network read.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Approve only the System-tier call that is pending right now. The id
    /// travels with the decision so a delayed duplicate click can never
    /// authorize a later call.
    pub fn approve_system_tool(&self) {
        self.decide_system_tool(ApprovalChoice::Approve);
    }

    /// Decline only the System-tier call that is pending right now.
    pub fn deny_system_tool(&self) {
        self.decide_system_tool(ApprovalChoice::Deny);
    }

    fn decide_system_tool(&self, choice: ApprovalChoice) {
        let request_id = self.pending_approval_id.load(Ordering::Acquire);
        if request_id != 0 {
            let _ = self
                .approval_tx
                .send(ApprovalDecision { request_id, choice });
        }
    }
}

pub struct AgentWorker {
    pub(crate) handle: AgentHandle,
    pub(crate) _join: JoinHandle<()>,
}

pub(crate) struct AgentRuntimeHandles {
    pub(crate) store: slint::Weak<AgentStore<'static>>,
    pub(crate) app: slint::Weak<AppWindow>,
    pub(crate) cache_registry: CacheRegistry,
    pub(crate) job_manager: JobManager,
}
