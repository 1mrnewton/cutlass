use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crossbeam_channel::Receiver;
use cutlass_ai::{HostToolSpec, ToolHost, ToolOutput, ToolTier};
use cutlass_jobs::JobManager;
use cutlass_settings::Autonomy;
use tracing::error;

use crate::cache_registry::CacheRegistry;
use crate::preview_worker::WorkerHandle;
use crate::{AgentStore, AppWindow};

use super::approval::{
    APPROVAL_WAIT_SLICE, ApprovalWaitOutcome, allocate_approval_request_id, clear_approval_card,
    publish_approval_card, wait_for_system_tool_approval,
};
use super::types::{AgentRuntimeHandles, ApprovalDecision};

#[derive(Clone)]
pub(crate) struct DesktopToolHandles {
    pub(crate) store: slint::Weak<AgentStore<'static>>,
    pub(crate) app: slint::Weak<AppWindow>,
    pub(crate) cache_registry: Option<CacheRegistry>,
    pub(crate) job_manager: JobManager,
    pub(crate) worker: Option<WorkerHandle>,
}

impl DesktopToolHandles {
    pub(super) fn from_runtime(runtime: &AgentRuntimeHandles, worker: &WorkerHandle) -> Self {
        Self {
            store: runtime.store.clone(),
            app: runtime.app.clone(),
            cache_registry: Some(runtime.cache_registry.clone()),
            job_manager: runtime.job_manager.clone(),
            worker: Some(worker.clone()),
        }
    }
}

pub(crate) struct ApprovedProjectImport {
    pub(crate) arguments: serde_json::Value,
    pub(crate) validated: crate::agent_project::ValidatedImportMedia,
}

/// The desktop host-tool surface: app and job controls plus the approval
/// broker that gates every System-tier call.
pub struct DesktopToolHost {
    pub(crate) autonomy: Autonomy,
    pub(crate) runtime: DesktopToolHandles,
    pub(crate) approval_rx: Receiver<ApprovalDecision>,
    pub(crate) pending_approval_id: Arc<AtomicU64>,
    pub(crate) approval_id_allocator: Arc<AtomicU64>,
    pub(crate) ordinary_host_call_attempted: bool,
    pub(crate) approved_project_import: Option<ApprovedProjectImport>,
}

impl DesktopToolHost {
    pub(super) fn new(
        autonomy: Autonomy,
        runtime: DesktopToolHandles,
        approval_rx: Receiver<ApprovalDecision>,
        pending_approval_id: Arc<AtomicU64>,
        approval_id_allocator: Arc<AtomicU64>,
    ) -> Self {
        Self {
            autonomy,
            runtime,
            approval_rx,
            pending_approval_id,
            approval_id_allocator,
            ordinary_host_call_attempted: false,
            approved_project_import: None,
        }
    }

    pub(super) fn ordinary_host_call_attempted(&self) -> bool {
        self.ordinary_host_call_attempted
    }
}

impl ToolHost for DesktopToolHost {
    fn tools(&self) -> Vec<HostToolSpec> {
        let mut specs = crate::agent_app_control::specs();
        specs.extend(crate::agent_project::specs());
        specs.extend(crate::agent_jobs::specs());
        specs.extend(crate::agent_system::specs());
        specs
    }

    fn authorize(
        &mut self,
        name: &str,
        arguments: &serde_json::Value,
        tier: ToolTier,
        cancel: &AtomicBool,
    ) -> Result<(), String> {
        // Authorization and dispatch are synchronous in the agent loop. Clear
        // any stale binding before considering a new call.
        self.approved_project_import = None;
        if tier != ToolTier::System || self.autonomy == Autonomy::Full {
            return Ok(());
        }
        if cancel.load(Ordering::Acquire) {
            return Err("cancelled before the system tool could run".into());
        }
        let validated_import = match cutlass_ai::namespace(name) {
            "project" => {
                crate::agent_project::validate_request(name, arguments)?;
                if name == crate::agent_project::PROJECT_IMPORT_MEDIA {
                    Some(crate::agent_project::validated_import_media(arguments)?)
                } else {
                    None
                }
            }
            "system" => {
                crate::agent_system::validate_request(name, arguments)?;
                None
            }
            _ => None,
        };
        if cancel.load(Ordering::Acquire) {
            return Err("cancelled before the system tool could run".into());
        }

        let request_id = allocate_approval_request_id(&self.approval_id_allocator)?;
        self.pending_approval_id
            .compare_exchange(0, request_id, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| "another system tool approval is already pending".to_string())?;
        if let Err(error) = publish_approval_card(
            &self.runtime.store,
            name,
            arguments,
            self.runtime.cache_registry.as_ref(),
            validated_import.as_ref(),
        ) {
            let _ = self.pending_approval_id.compare_exchange(
                request_id,
                0,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            clear_approval_card(&self.runtime.store);
            return Err(error);
        }

        let outcome = wait_for_system_tool_approval(
            &self.approval_rx,
            request_id,
            cancel,
            APPROVAL_WAIT_SLICE,
        );
        let _ = self.pending_approval_id.compare_exchange(
            request_id,
            0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        clear_approval_card(&self.runtime.store);

        match outcome {
            ApprovalWaitOutcome::Approved if cancel.load(Ordering::Acquire) => {
                Err("cancelled before the system tool could run".into())
            }
            ApprovalWaitOutcome::Approved => {
                if let Some(validated) = validated_import {
                    self.approved_project_import = Some(ApprovedProjectImport {
                        arguments: arguments.clone(),
                        validated,
                    });
                }
                Ok(())
            }
            ApprovalWaitOutcome::Declined => Err(format!(
                "the user declined system tool '{name}'; the tool was not run"
            )),
            ApprovalWaitOutcome::Cancelled => {
                Err("cancelled while waiting for system tool approval; the tool was not run".into())
            }
            ApprovalWaitOutcome::ChannelClosed => {
                Err("system tool approval closed; the tool was not run".into())
            }
        }
    }

    fn call(
        &mut self,
        name: &str,
        arguments: &serde_json::Value,
        cancel: &AtomicBool,
    ) -> Result<ToolOutput, String> {
        let namespace = cutlass_ai::namespace(name);
        if matches!(namespace, "app" | "system")
            || name == "job_cancel"
            || crate::agent_project::mutates_live_project(name)
        {
            // Set this before dispatch: an error can still follow a partial
            // host-side effect, so abort messaging must be conservative.
            self.ordinary_host_call_attempted = true;
        }
        let approved_import = if name == crate::agent_project::PROJECT_IMPORT_MEDIA
            && self.autonomy == Autonomy::Ask
        {
            let approval = self.approved_project_import.take().ok_or_else(|| {
                "project_import_media approval could not be confirmed; not started".to_string()
            })?;
            if approval.arguments != *arguments {
                error!("project media-import arguments changed after approval");
                return Err(
                    "project_import_media approval did not match this request; not started".into(),
                );
            }
            Some(approval.validated)
        } else {
            self.approved_project_import = None;
            None
        };
        match namespace {
            "app" => {
                crate::agent_app_control::call(self.runtime.app.clone(), name, arguments, cancel)
            }
            "project" => match approved_import.as_ref() {
                Some(approved) => crate::agent_project::call_with_approved_import(
                    self.runtime.worker.as_ref(),
                    name,
                    arguments,
                    Some(approved),
                    cancel,
                ),
                None => crate::agent_project::call(
                    self.runtime.worker.as_ref(),
                    name,
                    arguments,
                    cancel,
                ),
            },
            "job" => crate::agent_jobs::call(&self.runtime.job_manager, name, arguments, cancel),
            "system" => crate::agent_system::call(
                self.runtime.cache_registry.as_ref(),
                name,
                arguments,
                cancel,
            ),
            other => Err(format!("unsupported desktop tool namespace '{other}'")),
        }
    }
}

pub(crate) fn abort_status_message(reason: &str, ordinary_host_call_attempted: bool) -> String {
    if !ordinary_host_call_attempted {
        return if reason == "cancelled" {
            "Stopped — nothing was applied.".to_string()
        } else {
            format!("{reason} — nothing was applied.")
        };
    }

    let effect_notice = "Timeline edits staged by this prompt were rolled back and were not \
                         applied; any host actions that already completed remain in effect.";
    if reason == "cancelled" {
        format!("Stopped — {effect_notice}")
    } else {
        format!("{reason} — {effect_notice}")
    }
}
