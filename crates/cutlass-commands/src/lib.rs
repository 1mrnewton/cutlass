//! Editor command layer: a closed set of structured, deterministic edits.
//!
//! Every edit is an explicit, serializable value — auditable, replayable, and
//! undo-friendly once wired through `cutlass-engine`.

mod command;

pub use command::{Command, EditCommand, EditOutcome, ProjectCommand, TemplatePick};
pub use cutlass_models::{ClipId, Generator, MediaId, RationalTime, TimeRange, TrackId};

use tracing::info;

pub fn init() {
    info!("cutlass-commands ready");
}
