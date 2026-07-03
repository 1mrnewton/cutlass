use std::path::PathBuf;

use cutlass_models::{Template, TemplateMeta};

use crate::action::ApplyContext;
use crate::error::EngineError;

/// Write the session project as a `.cutlasst` template file. The session is
/// untouched: unlike `Save`, the template is a different document kind, so
/// neither the project path nor the dirty baseline moves. A project with no
/// replaceable slots is allowed — text-only templates (editable titles over a
/// locked look) are a real CapCut category.
pub fn execute(
    ctx: &mut ApplyContext<'_>,
    path: PathBuf,
    meta: TemplateMeta,
) -> Result<(), EngineError> {
    let template = Template::from_project(ctx.project.clone(), meta);
    template.save_to_file(&path)?;
    Ok(())
}
