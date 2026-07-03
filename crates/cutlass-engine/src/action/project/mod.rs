//! Session-level project commands (Import/RelinkMedia/Save/Open/Load and the
//! template pair SaveTemplate/ApplyTemplate).
//!
//! Import and relink probe media through the native, FFmpeg-free decoder
//! ([`crate::import::import_media`]) — the same codec used for preview/export
//! reports the metadata.

use std::path::PathBuf;

use cutlass_models::Project;

use crate::action::ApplyContext;
use crate::error::EngineError;

pub mod apply_template;
pub mod import;
pub mod load;
pub mod open;
pub mod relink;
pub mod save;
pub mod save_template;

/// Replace the session project from a `.cutlass` file. With `strict`, every
/// referenced media path must exist on disk (Open); otherwise missing media is
/// tolerated (Load). History is cleared — a fresh document has no undo past.
pub(crate) fn load_session(
    ctx: &mut ApplyContext<'_>,
    path: PathBuf,
    strict: bool,
) -> Result<(), EngineError> {
    let loaded = Project::load_from_file(&path)?;
    if strict {
        for media in loaded.media_iter() {
            if !media.path().exists() {
                return Err(EngineError::MissingMedia(media.path().display().to_string()));
            }
        }
    }
    *ctx.project = loaded;
    *ctx.project_path = Some(path);
    ctx.history.clear();
    Ok(())
}
