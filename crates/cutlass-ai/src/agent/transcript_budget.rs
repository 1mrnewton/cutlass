//! Keep the in-flight and session-history transcripts bounded: image
//! budgets, and collapsing superseded `describe_project` dumps.

use crate::provider::{ImagePart, Message};

/// Placeholder written over a collapsed `describe_project` tool result.
/// Shared by the in-prompt collapse and session-history collection so
/// the wording cannot drift.
pub(super) const DESCRIBE_PROJECT_RESULT_PLACEHOLDER: &str =
    "(stale project snapshot removed; call describe_project for the current state)";

/// Replace the content of every `ToolResult` whose `call_id` is in
/// `describe_call_ids` with [`DESCRIBE_PROJECT_RESULT_PLACEHOLDER`].
/// Images are left untouched.
pub(super) fn collapse_describe_project_results(
    messages: &mut [Message],
    describe_call_ids: &[String],
) {
    if describe_call_ids.is_empty() {
        return;
    }
    for message in messages {
        if let Message::ToolResult {
            call_id, content, ..
        } = message
            && describe_call_ids.iter().any(|id| id == call_id)
        {
            *content = DESCRIBE_PROJECT_RESULT_PLACEHOLDER.to_string();
        }
    }
}

/// Bound a single extensible tool result before it reaches either the
/// transcript or the request history. Count and encoded-byte limits both keep
/// the newest attachments, matching the whole-request policy below.
pub(super) fn enforce_tool_output_image_budget(
    content: &mut String,
    images: &mut Vec<ImagePart>,
    max_images: usize,
    max_bytes: usize,
) {
    let mut count = images.len();
    let mut bytes = images
        .iter()
        .map(|image| image.data.len())
        .fold(0usize, usize::saturating_add);
    let mut drop_count = 0usize;
    for image in images.iter() {
        if count <= max_images && bytes <= max_bytes {
            break;
        }
        count = count.saturating_sub(1);
        bytes = bytes.saturating_sub(image.data.len());
        drop_count += 1;
    }
    for dropped in images.drain(..drop_count) {
        content.push_str(&format!(
            "\n[image not attached because it exceeded the request budget: {}]",
            dropped.label
        ));
    }
}

/// Keep only the newest `max_images` images across the request; older
/// ones are dropped in place and noted with a text placeholder carrying
/// the label, so the model knows what it saw and can re-request it.
/// Newest-wins matches how the agent works with vision: screenshot, look,
/// act — a stale frame is cheaper to re-take than to carry.
pub(super) fn enforce_image_budget(messages: &mut [Message], max_images: usize, max_bytes: usize) {
    let mut image_total: usize = messages.iter().map(image_count).sum();
    let mut byte_total: usize = messages
        .iter()
        .flat_map(message_images)
        .map(|image| image.data.len())
        .fold(0usize, usize::saturating_add);
    if image_total <= max_images && byte_total <= max_bytes {
        return;
    }

    // Oldest first. Count how much of each image vector to drain before
    // mutating it, keeping this O(number of images) rather than repeatedly
    // removing index zero.
    for message in messages.iter_mut() {
        if image_total <= max_images && byte_total <= max_bytes {
            break;
        }
        let (content, images) = match message {
            Message::User { content, images } => (content, images),
            Message::ToolResult {
                content, images, ..
            } => (content, images),
            _ => continue,
        };
        let mut drop_count = 0usize;
        for image in images.iter() {
            if image_total <= max_images && byte_total <= max_bytes {
                break;
            }
            image_total = image_total.saturating_sub(1);
            byte_total = byte_total.saturating_sub(image.data.len());
            drop_count += 1;
        }
        for dropped in images.drain(..drop_count) {
            content.push_str(&format!("\n[image no longer attached: {}]", dropped.label));
        }
    }
}

pub(super) fn image_count(message: &Message) -> usize {
    match message {
        Message::User { images, .. } | Message::ToolResult { images, .. } => images.len(),
        _ => 0,
    }
}

fn message_images(message: &Message) -> &[ImagePart] {
    match message {
        Message::User { images, .. } | Message::ToolResult { images, .. } => images,
        _ => &[],
    }
}

/// Session history is text-only: raw image bytes would bloat every later
/// request and the persisted session file for no benefit — the agent can
/// always re-screenshot the *current* state. A labeled placeholder keeps
/// the narrative ("looked at the timeline here") without the payload.
fn strip_images(content: &mut String, images: &mut Vec<ImagePart>) {
    for image in images.drain(..) {
        content.push_str(&format!("\n[image: {}]", image.label));
    }
}

/// This turn's slice of the conversation (`messages[turn_start..]`: the
/// user prompt plus every assistant/tool message the loop appended), with
/// the final text answer added (it isn't pushed during the loop),
/// `describe_project` results collapsed to a placeholder, and images
/// stripped to labels (history is text-only). This is what the session
/// appends to its history so the next prompt remembers the turn.
pub(super) fn collect_turn_messages(
    messages: Vec<Message>,
    turn_start: usize,
    describe_call_ids: &[String],
    final_text: &str,
) -> Vec<Message> {
    let mut turn: Vec<Message> = messages.into_iter().skip(turn_start).collect();
    collapse_describe_project_results(&mut turn, describe_call_ids);
    for message in &mut turn {
        match message {
            Message::ToolResult {
                content, images, ..
            } => strip_images(content, images),
            Message::User { content, images } => strip_images(content, images),
            _ => {}
        }
    }
    if !final_text.is_empty() {
        turn.push(Message::Assistant {
            content: final_text.to_string(),
            tool_calls: Vec::new(),
        });
    }
    turn
}
