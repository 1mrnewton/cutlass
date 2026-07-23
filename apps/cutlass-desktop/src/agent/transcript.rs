use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;

use crossbeam_channel::bounded;
use cutlass_ai::Message;
use slint::{Model, ModelRc, SharedString, VecModel};
use tracing::warn;

use crate::agent_session::{AgentSession, ChatMeta, TranscriptEntry};
use crate::{AgentEntry, AgentStore};

pub(crate) fn entry(kind: &str, text: impl Into<SharedString>) -> AgentEntry {
    AgentEntry {
        kind: kind.into(),
        text: text.into(),
        image: Default::default(),
        image_aspect: 0.0,
        usage: Default::default(),
    }
}

pub(crate) fn with_transcript(
    store: &slint::Weak<AgentStore<'static>>,
    f: impl FnOnce(&VecModel<AgentEntry>) + Send + 'static,
) {
    let store = store.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(store) = store.upgrade() {
            let transcript = store.get_transcript();
            if let Some(model) = transcript.as_any().downcast_ref::<VecModel<AgentEntry>>() {
                f(model);
            }
        }
    });
}

pub(crate) fn push_entry(
    store: &slint::Weak<AgentStore<'static>>,
    kind: &'static str,
    text: String,
) {
    with_transcript(store, move |model| model.push(entry(kind, text)));
}

/// Current transcript row count, observed on the Slint thread.
pub(crate) fn transcript_row_count(store: &slint::Weak<AgentStore<'static>>) -> Option<usize> {
    let (tx, rx) = bounded(1);
    let store = store.clone();
    slint::invoke_from_event_loop(move || {
        let n = store
            .upgrade()
            .map(|store| store.get_transcript().row_count())
            .unwrap_or(0);
        let _ = tx.send(n);
    })
    .ok()?;
    rx.recv_timeout(Duration::from_secs(2)).ok()
}

/// Drop discarded dry-run rehearsal rows from the chat UI while keeping the
/// already-pushed replacement user prompt (and any status lines after it).
pub(crate) fn trim_transcript_after_stale_plan_discard(
    store: &slint::Weak<AgentStore<'static>>,
    checkpoint_len: usize,
) {
    with_transcript(store, move |model| {
        let count = model.row_count();
        let rows: Vec<AgentEntry> = (0..count).filter_map(|i| model.row_data(i)).collect();
        model.set_vec(trim_rows_after_stale_plan_discard(
            rows,
            checkpoint_len,
            |row| row.kind == "user",
        ));
    });
}

/// Pure splice used by [`trim_transcript_after_stale_plan_discard`]: keep
/// `[0..checkpoint_len)` plus the trailing segment that starts at the latest
/// `user` row at or after the checkpoint.
pub(crate) fn trim_rows_after_stale_plan_discard<T>(
    mut rows: Vec<T>,
    checkpoint_len: usize,
    is_user: impl Fn(&T) -> bool,
) -> Vec<T> {
    let count = rows.len();
    let checkpoint_len = checkpoint_len.min(count);
    let keep_from = (checkpoint_len..count)
        .rev()
        .find(|&i| is_user(&rows[i]))
        .unwrap_or(count);
    if keep_from > checkpoint_len {
        rows.drain(checkpoint_len..keep_from);
    } else if keep_from < checkpoint_len {
        rows.truncate(checkpoint_len);
    }
    rows
}

pub(crate) fn push_image_entry(
    store: &slint::Weak<AgentStore<'static>>,
    image: cutlass_ai::ImagePart,
) {
    let label = transcript_image_label(&image.label);
    let frame = match decode_transcript_image(&image) {
        Ok(frame) => frame,
        Err(error) => {
            push_entry(
                store,
                "status",
                format!("Could not display image '{label}': {error}"),
            );
            return;
        }
    };
    let aspect = frame.width as f32 / frame.height as f32;
    let (width, height, pixels) = (frame.width, frame.height, frame.pixels);
    with_transcript(store, move |model| {
        let buffer =
            slint::SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(&pixels, width, height);
        model.push(AgentEntry {
            kind: "image".into(),
            text: label.into(),
            image: slint::Image::from_rgba8(buffer),
            image_aspect: aspect,
            usage: Default::default(),
        });
    });
}

/// Attach a finished prompt's usage line to the current exchange: prefer the
/// last assistant row after the latest user prompt; otherwise a status line.
pub(crate) fn attach_usage_line(store: &slint::Weak<AgentStore<'static>>, usage: String) {
    with_transcript(store, move |model| {
        let count = model.row_count();
        let mut last_user = None;
        for index in 0..count {
            if model.row_data(index).is_some_and(|row| row.kind == "user") {
                last_user = Some(index);
            }
        }
        let start = last_user.map(|index| index + 1).unwrap_or(0);
        let mut assistant_idx = None;
        for index in (start..count).rev() {
            if model
                .row_data(index)
                .is_some_and(|row| row.kind == "assistant")
            {
                assistant_idx = Some(index);
                break;
            }
        }
        if let Some(index) = assistant_idx {
            if let Some(mut row) = model.row_data(index) {
                row.usage = usage.into();
                model.set_row_data(index, row);
            }
        } else {
            model.push(entry("status", usage));
        }
    });
}

pub(crate) fn decode_transcript_image(
    image: &cutlass_ai::ImagePart,
) -> Result<cutlass_render::RgbaImage, String> {
    const MAX_EDGE: u32 = 2_048;
    const MAX_PIXELS: u64 = 4 * 1024 * 1024;

    let frame = match image.media_type.as_str() {
        "image/png" => {
            cutlass_render::decode_png(image.data.as_slice()).map_err(|error| error.to_string())?
        }
        "image/jpeg" => cutlass_decoder::decode_image_bytes(image.data.as_slice())
            .map_err(|error| error.to_string())?,
        media_type => return Err(format!("unsupported transcript image type '{media_type}'")),
    };
    let pixels = u64::from(frame.width)
        .checked_mul(u64::from(frame.height))
        .ok_or_else(|| "transcript image dimensions overflow".to_string())?;
    if frame.width == 0
        || frame.height == 0
        || frame.width > MAX_EDGE
        || frame.height > MAX_EDGE
        || pixels > MAX_PIXELS
    {
        return Err(format!(
            "transcript image dimensions {}x{} exceed the display bound",
            frame.width, frame.height
        ));
    }
    if !frame.is_well_formed() {
        return Err("transcript image has a malformed RGBA buffer".into());
    }
    Ok(frame)
}

pub(crate) fn transcript_image_label(label: &str) -> String {
    const MAX_CHARS: usize = 160;
    let mut safe = String::with_capacity(label.len().min(MAX_CHARS));
    for character in label.chars().take(MAX_CHARS) {
        safe.push(if character.is_control() {
            '\u{fffd}'
        } else {
            character
        });
    }
    if label.chars().count() > MAX_CHARS {
        safe.push('…');
    }
    if safe.trim().is_empty() {
        "Agent image".to_string()
    } else {
        safe
    }
}

pub(crate) fn append_assistant_text(store: &slint::Weak<AgentStore<'static>>, delta: String) {
    with_transcript(store, move |model| {
        append_transcript_text(model, "assistant", delta);
    });
}

pub(crate) fn append_reasoning_text(store: &slint::Weak<AgentStore<'static>>, delta: String) {
    with_transcript(store, move |model| {
        append_transcript_text(model, "reasoning", delta);
    });
}

pub(crate) fn append_transcript_text(model: &VecModel<AgentEntry>, kind: &str, delta: String) {
    let last = model.row_count().wrapping_sub(1);
    match model.row_data(last) {
        Some(row) if row.kind == kind => {
            let mut row = row;
            row.text = format!("{}{}", row.text, delta).into();
            model.set_row_data(last, row);
        }
        _ => model.push(entry(kind, delta)),
    }
}

pub(crate) fn with_store(
    store: &slint::Weak<AgentStore<'static>>,
    f: impl FnOnce(AgentStore<'_>) + Send + 'static,
) {
    let store = store.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(store) = store.upgrade() {
            f(store);
        }
    });
}

/// Snapshot the visible transcript on the Slint thread. Calls are made only
/// from the dedicated agent worker; the timeout prevents shutdown from
/// hanging if the event loop has already stopped.
pub(crate) fn transcript_snapshot(
    store: &slint::Weak<AgentStore<'static>>,
) -> Result<Vec<TranscriptEntry>, String> {
    let (tx, rx) = bounded(1);
    let store = store.clone();
    slint::invoke_from_event_loop(move || {
        let rows = store
            .upgrade()
            .map(|store| {
                let model = store.get_transcript();
                (0..model.row_count())
                    .filter_map(|index| model.row_data(index))
                    .map(|row| TranscriptEntry {
                        kind: row.kind.to_string(),
                        text: row.text.to_string(),
                        usage: row.usage.to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let _ = tx.send(rows);
    })
    .map_err(|error| format!("failed to schedule agent transcript snapshot: {error}"))?;
    rx.recv_timeout(Duration::from_secs(2))
        .map_err(|error| format!("agent transcript snapshot timed out: {error}"))
}

pub(crate) fn replace_transcript(
    store: &slint::Weak<AgentStore<'static>>,
    mut transcript: Vec<TranscriptEntry>,
    restore_error: Option<String>,
) {
    if let Some(error) = restore_error {
        warn!(error, "agent session could not be restored");
        transcript.push(TranscriptEntry {
            kind: "error".into(),
            text: "The previous agent conversation could not be restored.".into(),
            usage: String::new(),
        });
    }
    with_store(store, move |store| {
        // Build Slint image-bearing rows on the UI thread: `slint::Image`
        // is intentionally not Send even when it is empty.
        let rows: Vec<AgentEntry> = transcript
            .into_iter()
            .map(|saved| {
                if saved.kind == "image" {
                    let caption = if saved.text.is_empty() {
                        "Image attachment from the previous session.".to_string()
                    } else {
                        format!("Image attachment from the previous session: {}", saved.text)
                    };
                    entry("status", caption)
                } else {
                    let mut row = entry(&saved.kind, saved.text);
                    if !saved.usage.is_empty() {
                        row.usage = saved.usage.into();
                    }
                    row
                }
            })
            .collect();
        store.set_transcript(ModelRc::new(VecModel::from(rows)));
    });
}

pub(crate) fn persist_session(
    project: Option<&Path>,
    chat_id: Option<&str>,
    history: &[Message],
    store: &slint::Weak<AgentStore<'static>>,
) {
    let (Some(project), Some(chat_id)) = (project, chat_id) else {
        return;
    };
    let transcript = match transcript_snapshot(store) {
        Ok(transcript) => transcript,
        Err(error) => {
            warn!(error, "agent session transcript was not captured");
            return;
        }
    };
    let session = AgentSession {
        history: history.to_vec(),
        transcript,
    };
    if session.history.is_empty() && session.transcript.is_empty() {
        return;
    }
    if let Err(error) = crate::agent_session::save_chat(project, chat_id, &session) {
        warn!(
            error,
            project = %project.display(),
            chat_id,
            "agent chat was not saved"
        );
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ChatChoice {
    pub(crate) id: String,
    pub(crate) label: String,
}

pub(crate) fn chat_choices(
    mut chats: Vec<ChatMeta>,
    active_chat_id: Option<&str>,
) -> Vec<ChatChoice> {
    if let Some(active_id) = active_chat_id
        && !chats.iter().any(|chat| chat.id == active_id)
    {
        chats.insert(
            0,
            ChatMeta {
                id: active_id.to_string(),
                title: "New chat".to_string(),
                updated_millis: u64::MAX,
            },
        );
    }

    let mut used = HashSet::new();
    chats
        .into_iter()
        .map(|chat| {
            let base = chat.title;
            let mut label = base.clone();
            let mut suffix = 2;
            while !used.insert(label.clone()) {
                label = format!("{base} · {suffix}");
                suffix += 1;
            }
            ChatChoice { id: chat.id, label }
        })
        .collect()
}

pub(crate) fn publish_chat_list(
    store: &slint::Weak<AgentStore<'static>>,
    project: Option<&Path>,
    active_chat_id: Option<&str>,
) {
    let chats = match project {
        Some(project) => match crate::agent_session::list_chats(project) {
            Ok(chats) => chats,
            Err(error) => {
                warn!(error, project = %project.display(), "agent chats could not be listed");
                Vec::new()
            }
        },
        None => Vec::new(),
    };
    let choices = chat_choices(chats, active_chat_id);
    let active_label = active_chat_id
        .and_then(|active_id| {
            choices
                .iter()
                .find(|choice| choice.id == active_id)
                .map(|choice| choice.label.clone())
        })
        .unwrap_or_default();
    let labels: Vec<SharedString> = choices
        .iter()
        .map(|choice| choice.label.as_str().into())
        .collect();
    let ids: Vec<SharedString> = choices.into_iter().map(|choice| choice.id.into()).collect();
    with_store(store, move |store| {
        store.set_chat_labels(ModelRc::new(VecModel::from(labels)));
        store.set_chat_ids(ModelRc::new(VecModel::from(ids)));
        store.set_active_chat_label(active_label.into());
    });
}
