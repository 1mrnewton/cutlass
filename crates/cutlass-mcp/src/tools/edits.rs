//! Validated timeline edit tools — compact meta-tools over the wire vocabulary.
//!
//! Agents discover commands via `edit_commands_list` / `edit_schema_get`, then
//! mutate through `edit_apply` (never raw engine commands). Undo/redo operate
//! on whole `edit_apply` batches.

use cutlass_ai::{TOOL_SCHEMA_VERSION, ToolSpec, tool_specs};
use rmcp::{handler::server::wrapper::Parameters, tool, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::server::CutlassMcp;

/// Names offered to in-app assistants but not to external MCP agents.
///
/// Defense-in-depth: `describe_project` / `read_skill` are separate specs in
/// cutlass-ai today (not in [`tool_specs`]), so this filter currently removes
/// nothing. Kept so a future fold-in stays excluded without MCP churn.
const EXCLUDED_COMMANDS: &[&str] = &["describe_project", "read_skill"];

#[derive(Debug, Deserialize, JsonSchema)]
pub struct EditSchemaGetParams {
    /// Wire command names to fetch schemas for (e.g. `["split_clip"]`).
    pub commands: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct EditApplyParams {
    /// Batch of wire edits. Each item is `{"command": "<name>", ...arguments}`.
    /// The whole batch is one undo group: validated sequentially against live
    /// project state, all-or-nothing on failure.
    pub edits: Vec<Value>,
}

/// Wire edit commands exposed over MCP (filters in-app-only names).
pub(crate) fn exposed_tool_specs() -> Vec<ToolSpec> {
    let mut specs: Vec<ToolSpec> = tool_specs()
        .into_iter()
        .filter(|s| !EXCLUDED_COMMANDS.contains(&s.name.as_str()))
        .collect();
    specs.sort_by(|a, b| a.name.cmp(&b.name));
    specs
}

fn format_commands_list(specs: &[ToolSpec]) -> String {
    let mut lines = Vec::with_capacity(specs.len() + 1);
    lines.push(format!(
        "{} edit commands (tool schema v{TOOL_SCHEMA_VERSION}). \
         Call edit_schema_get before first use of a command.",
        specs.len()
    ));
    for spec in specs {
        lines.push(format!("{} — {}", spec.name, spec.description));
    }
    lines.join("\n")
}

fn lookup_schemas(names: &[String]) -> Result<Value, String> {
    let specs = exposed_tool_specs();
    let mut out = Map::new();
    for name in names {
        match specs.iter().find(|s| s.name == *name) {
            Some(spec) => {
                out.insert(
                    name.clone(),
                    json!({
                        "description": spec.description,
                        "parameters": spec.parameters,
                    }),
                );
            }
            None => {
                let available: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
                let hint = close_names(name, &available);
                return Err(format!(
                    "unknown edit command '{name}'. {hint}Available: {}",
                    available.join(", ")
                ));
            }
        }
    }
    Ok(Value::Object(out))
}

/// A few names that share a prefix / contain the query, for readable errors.
fn close_names(query: &str, available: &[&str]) -> String {
    let q = query.to_ascii_lowercase();
    let mut close: Vec<&str> = available
        .iter()
        .copied()
        .filter(|n| {
            let n = n.to_ascii_lowercase();
            n.contains(&q) || q.contains(&n) || n.starts_with(&q) || q.starts_with(&n)
        })
        .take(5)
        .collect();
    if close.is_empty() {
        close = available.iter().copied().take(5).collect();
        format!("Did you mean one of: {}? ", close.join(", "))
    } else {
        format!("Close matches: {}. ", close.join(", "))
    }
}

#[tool_router(router = edits_router, vis = "pub(crate)")]
impl CutlassMcp {
    /// List every wire edit command name and one-line description.
    #[tool(
        description = "List validated Cutlass edit commands (name — description). Call edit_schema_get for argument shapes before first use.",
        annotations(read_only_hint = true)
    )]
    fn edit_commands_list(&self) -> String {
        format_commands_list(&exposed_tool_specs())
    }

    /// Fetch JSON Schema + description for named wire commands.
    #[tool(
        description = "Get description + JSON Schema parameters for named edit commands (call before first use of each)",
        annotations(read_only_hint = true)
    )]
    fn edit_schema_get(
        &self,
        Parameters(params): Parameters<EditSchemaGetParams>,
    ) -> Result<String, String> {
        let schemas = lookup_schemas(&params.commands)?;
        serde_json::to_string_pretty(&schemas).map_err(|e| e.to_string())
    }

    /// Apply a batch of validated wire edits as one undo group.
    ///
    /// Each item is `{"command": "<name>", ...arguments}`. Edits are
    /// validated sequentially against live project state (so edit N can
    /// reference entities created by edit N-1 when ids are known). On any
    /// failure the whole batch rolls back. Created ids for chaining are in
    /// each outcome; call `edit_schema_get` for argument shapes.
    #[tool(
        description = "Apply validated wire edits as one all-or-nothing undo group. Each item: {\"command\": \"<name>\", ...arguments}. Call edit_schema_get for shapes; use returned outcome ids to chain follow-ups."
    )]
    async fn edit_apply(
        &self,
        Parameters(params): Parameters<EditApplyParams>,
    ) -> Result<String, String> {
        let batch = self.host.apply_edits(params.edits).await?;
        serde_json::to_string_pretty(&batch).map_err(|e| e.to_string())
    }

    /// Undo one history step (one `edit_apply` batch is one step).
    #[tool(description = "Undo one step — one edit_apply batch is one undo step")]
    async fn edit_undo(&self) -> Result<String, String> {
        let result = self.host.undo().await?;
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
    }

    /// Redo one history step (one `edit_apply` batch is one step).
    #[tool(description = "Redo one step — one edit_apply batch is one redo step")]
    async fn edit_redo(&self) -> Result<String, String> {
        let result = self.host.redo().await?;
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposed_commands_include_split_exclude_in_app_only() {
        let names: Vec<String> = exposed_tool_specs().into_iter().map(|s| s.name).collect();
        assert!(names.contains(&"split_clip".into()), "{names:?}");
        assert!(names.contains(&"add_generated".into()), "{names:?}");
        assert!(!names.iter().any(|n| n == "read_skill"), "{names:?}");
        assert!(!names.iter().any(|n| n == "describe_project"), "{names:?}");

        // Today the filter is a no-op; if cutlass-ai folds these into
        // `tool_specs()`, this assert flags the change instead of silently
        // relying on EXCLUDED_COMMANDS.
        let all: Vec<String> = tool_specs().into_iter().map(|s| s.name).collect();
        assert!(
            !all.iter().any(|n| n == "read_skill"),
            "tool_specs() gained read_skill — update EXCLUDED_COMMANDS comment: {all:?}"
        );
        assert!(
            !all.iter().any(|n| n == "describe_project"),
            "tool_specs() gained describe_project — update EXCLUDED_COMMANDS comment: {all:?}"
        );
    }

    #[test]
    fn commands_list_header_mentions_schema_version() {
        let text = format_commands_list(&exposed_tool_specs());
        assert!(
            text.contains(&format!("tool schema v{TOOL_SCHEMA_VERSION}")),
            "{text}"
        );
        assert!(text.contains("edit_schema_get"), "{text}");
        assert!(text.contains("split_clip —"), "{text}");
    }
}
