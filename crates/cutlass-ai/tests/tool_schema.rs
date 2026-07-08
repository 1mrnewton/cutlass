//! Tool-schema snapshot: the prompt-visible surface is a checked-in,
//! reviewed artifact. Any change to the wire types shows up as a diff in
//! `tests/snapshots/tools.json` (and should bump `TOOL_SCHEMA_VERSION`
//! when it changes shape).
//!
//! Regenerate with: `BLESS_TOOL_SCHEMA=1 cargo test -p cutlass-ai`

use std::path::PathBuf;

fn snapshot_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots/tools.json")
}

fn current_schema() -> serde_json::Value {
    let mut tools = cutlass_ai::tool_specs();
    tools.push(cutlass_ai::wire::describe_project_spec());
    serde_json::json!({
        "version": cutlass_ai::TOOL_SCHEMA_VERSION,
        "tools": tools
            .into_iter()
            .map(|spec| serde_json::json!({
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }))
            .collect::<Vec<_>>(),
    })
}

#[test]
fn tool_schema_matches_snapshot() {
    let current = current_schema();
    let path = snapshot_path();

    if std::env::var_os("BLESS_TOOL_SCHEMA").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, serde_json::to_string_pretty(&current).unwrap()).unwrap();
        return;
    }

    let stored = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing tool-schema snapshot at {} ({e}); \
             run with BLESS_TOOL_SCHEMA=1 to create it",
            path.display()
        )
    });
    let stored: serde_json::Value = serde_json::from_str(&stored).unwrap();

    assert_eq!(
        stored, current,
        "the prompt-visible tool schema changed; review the diff, bump \
         TOOL_SCHEMA_VERSION if the shape changed, and re-bless with \
         BLESS_TOOL_SCHEMA=1"
    );
}
