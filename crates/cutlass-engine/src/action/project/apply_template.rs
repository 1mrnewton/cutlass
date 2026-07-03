use std::path::PathBuf;

use cutlass_commands::TemplatePick;
use cutlass_models::{Pick, Template};

use crate::action::ApplyContext;
use crate::error::EngineError;
use crate::import::import_media;

/// Replace the session with a `.cutlasst` template filled by `picks` (CapCut
/// "use template").
///
/// Atomic by construction: the template load, every pick probe, and the fill
/// all happen against local values, and the session is only replaced once the
/// filled project exists — a failure anywhere leaves project, path, and
/// history untouched. Like `Open`/`Load` the history is cleared (a fresh
/// document has no undo past), but the result is a *new unsaved* project, so
/// the project path resets. Sample media whose files are missing is tolerated
/// (matches `Load`); the UI relinks or fills those slots.
pub fn execute(
    ctx: &mut ApplyContext<'_>,
    path: PathBuf,
    picks: Vec<TemplatePick>,
) -> Result<(), EngineError> {
    let template = Template::load_from_file(&path)?;

    let mut resolved = Vec::with_capacity(picks.len());
    for pick in picks {
        let media_path = pick.path.canonicalize().map_err(EngineError::Io)?;
        let media = import_media(&media_path)?;
        resolved.push(Pick {
            media,
            source_in: pick.source_in,
        });
    }

    let filled = template.apply(&resolved)?;
    *ctx.project = filled;
    *ctx.project_path = None;
    ctx.history.clear();
    Ok(())
}
