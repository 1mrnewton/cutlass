use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use crossbeam_channel::{Receiver, RecvTimeoutError, bounded};
use slint::SharedString;

use crate::AgentStore;
use crate::cache_registry::CacheRegistry;

use super::transcript::with_store;
use super::types::{ApprovalChoice, ApprovalDecision};

pub(crate) const APPROVAL_WAIT_SLICE: Duration = Duration::from_millis(50);
pub(crate) const APPROVAL_CARD_PUBLISH_TIMEOUT: Duration = Duration::from_secs(2);
pub(crate) const APPROVAL_DETAIL_MAX_CHARS: usize = 2_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ApprovalWaitOutcome {
    Approved,
    Declined,
    Cancelled,
    ChannelClosed,
}

/// Wait for one exact approval generation. Decisions for earlier requests
/// are consumed and ignored, so they cannot leak into a later authorization.
pub(crate) fn wait_for_system_tool_approval(
    approval_rx: &Receiver<ApprovalDecision>,
    request_id: u64,
    cancel: &AtomicBool,
    wait_slice: Duration,
) -> ApprovalWaitOutcome {
    loop {
        if cancel.load(Ordering::Acquire) {
            return ApprovalWaitOutcome::Cancelled;
        }
        match approval_rx.recv_timeout(wait_slice) {
            Ok(decision) if decision.request_id != request_id => continue,
            Ok(decision) => {
                // Stop wins if it raced with a click that was already queued.
                if cancel.load(Ordering::Acquire) {
                    return ApprovalWaitOutcome::Cancelled;
                }
                return match decision.choice {
                    ApprovalChoice::Approve => ApprovalWaitOutcome::Approved,
                    ApprovalChoice::Deny => ApprovalWaitOutcome::Declined,
                };
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                return ApprovalWaitOutcome::ChannelClosed;
            }
        }
    }
}

pub(crate) fn allocate_approval_request_id(allocator: &AtomicU64) -> Result<u64, String> {
    allocator
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map(|previous| previous + 1)
        .map_err(|_| "system tool approval request id space is exhausted".into())
}

pub(crate) fn publish_approval_card(
    store: &slint::Weak<AgentStore<'static>>,
    name: &str,
    arguments: &serde_json::Value,
    cache_registry: Option<&CacheRegistry>,
    validated_import: Option<&crate::agent_project::ValidatedImportMedia>,
) -> Result<(), String> {
    let title = approval_title(name);
    let detail = approval_detail(name, arguments, cache_registry, validated_import);
    let (published_tx, published_rx) = bounded(1);
    let store = store.clone();
    slint::invoke_from_event_loop(move || {
        let published = store.upgrade().is_some_and(|store| {
            store.set_approval_title(title.into());
            store.set_approval_detail(detail.into());
            store.set_approval_pending(true);
            true
        });
        let _ = published_tx.send(published);
    })
    .map_err(|error| format!("could not show system tool approval: {error}"))?;
    match published_rx.recv_timeout(APPROVAL_CARD_PUBLISH_TIMEOUT) {
        Ok(true) => Ok(()),
        Ok(false) => Err("could not show system tool approval because the UI is closed".into()),
        Err(RecvTimeoutError::Timeout) => {
            Err("timed out while showing the system tool approval".into())
        }
        Err(RecvTimeoutError::Disconnected) => {
            Err("system tool approval UI closed before it could be shown".into())
        }
    }
}

pub(crate) fn approval_title(name: &str) -> String {
    match name {
        "project_open" => "Open this project draft?".into(),
        "project_import_media" => "Import this media file?".into(),
        "system_cache_list" => "Let the assistant inspect cache usage?".into(),
        "system_cache_clear" => "Clear this cache?".into(),
        "system_cache_relocate" => "Move this cache?".into(),
        "system_reveal" => "Reveal this path?".into(),
        "system_open_external" => "Open this outside Cutlass?".into(),
        "app_close" => "Close Cutlass?".into(),
        _ => format!("Run {name}?"),
    }
}

pub(crate) fn approval_detail(
    name: &str,
    arguments: &serde_json::Value,
    cache_registry: Option<&CacheRegistry>,
    validated_import: Option<&crate::agent_project::ValidatedImportMedia>,
) -> String {
    if name == "project_open" {
        let draft_id = arguments
            .get("draft_id")
            .and_then(serde_json::Value::as_str)
            .filter(|draft_id| {
                !draft_id.is_empty()
                    && draft_id.chars().count() <= crate::agent_project::MAX_DRAFT_ID_CHARS
                    && draft_id.bytes().all(|byte| {
                        byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte) || byte == b'-'
                    })
            })
            .unwrap_or("<invalid draft ID>");
        return bound_approval_detail(format!(
            "Draft ID: {draft_id}\n\nOpening this draft replaces the current session and may discard unsaved work."
        ));
    }

    if name == crate::agent_project::PROJECT_IMPORT_MEDIA {
        // Normal authorization supplies an opaque validated token retained
        // for dispatch. If a future caller bypasses that flow, validate the
        // raw arguments here rather than copying hostile text into the card.
        let revalidated = crate::agent_project::validated_import_media(arguments).ok();
        let display_path = validated_import
            .or(revalidated.as_ref())
            .map(crate::agent_project::ValidatedImportMedia::canonical_path);
        let path = display_path
            .and_then(crate::agent_project::import_path_approval_display)
            .unwrap_or_else(|| "<invalid media path>".into());
        return bound_approval_detail(format!(
            "Canonical file: {path}\n\nCutlass adds a reference to this file rather than copying the source. Moving or deleting it can make the media missing."
        ));
    }

    if name == "system_cache_clear"
        && let Some(id) = arguments
            .get("cache_id")
            .and_then(serde_json::Value::as_str)
            .and_then(|id| cutlass_storage::CacheId::parse(id).ok())
        && let Some(registry) = cache_registry
    {
        return bound_approval_detail(registry.clear_approval_detail(id));
    }

    if name == "system_cache_relocate"
        && let Some(id) = arguments
            .get("cache_id")
            .and_then(serde_json::Value::as_str)
            .and_then(|id| cutlass_storage::CacheId::parse(id).ok())
        && let Some(destination) = arguments
            .get("destination")
            .and_then(serde_json::Value::as_str)
        && let Some(registry) = cache_registry
        && let Ok(current_path) = registry.cache_path(id)
    {
        return bound_approval_detail(format_cache_relocation_approval_detail(
            id,
            &current_path,
            Path::new(destination),
        ));
    }

    let detail = match arguments.as_object() {
        Some(arguments) if arguments.is_empty() => "No arguments.".to_string(),
        _ => serde_json::to_string_pretty(arguments).unwrap_or_else(|_| arguments.to_string()),
    };
    bound_approval_detail(detail)
}

pub(crate) fn bound_approval_detail(detail: String) -> String {
    let mut bounded: String = detail.chars().take(APPROVAL_DETAIL_MAX_CHARS).collect();
    if detail.chars().count() > APPROVAL_DETAIL_MAX_CHARS {
        bounded.push('…');
    }
    bounded
}

pub(crate) fn format_cache_relocation_approval_detail(
    id: cutlass_storage::CacheId,
    current_path: &Path,
    destination: &Path,
) -> String {
    format!(
        "Cache: {}\nCurrent path: {}\nRequested destination: {}\n\nThe move may be refused when projects reference cache-owned files.",
        id.descriptor().label,
        current_path.display(),
        destination.display()
    )
}

pub(crate) fn clear_approval_card(store: &slint::Weak<AgentStore<'static>>) {
    with_store(store, |store| {
        store.set_approval_pending(false);
        store.set_approval_title(SharedString::default());
        store.set_approval_detail(SharedString::default());
    });
}
