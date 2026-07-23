# cutlass-mcp

Headless [MCP](https://modelcontextprotocol.io) server that exposes Cutlass to
external agents (Cursor, Claude Code, and similar hosts). Agents open or create
`.cutlass` projects, inspect timelines, apply validated edits, grab preview
frames, and export video — without embedding the editor UI.

## Trust model

Every mutation flows through the same gate as the in-app agent:

`WireCommand` → `cutlass-ai` validate → `cutlass-engine` apply

There is no raw engine door. Reads and renders are Workspace-tier; edit batches
are all-or-nothing undo groups against live project state. The engine runs on a
dedicated OS thread (`EngineHost`); async tool handlers round-trip requests over
a channel because the engine is not safely shared across tokio workers.

## Build and run

```bash
cargo build -p cutlass-mcp --release
```

Binary: `target/release/cutlass-mcp`. Transport is stdio JSON-RPC (stdout is the
wire; logs go to stderr).

### Host config

Cursor (`~/.cursor/mcp.json`) and Claude Code use the same shape:

```json
{
  "mcpServers": {
    "cutlass": {
      "command": "/absolute/path/to/cutlass/target/release/cutlass-mcp"
    }
  }
}
```

## Tools

Agent workflow: `project_new` / `project_open` → `edit_commands_list` /
`edit_schema_get` → `edit_apply` (one undo group per batch) → `project_get` /
`frame_get` to verify → `project_save` / `export_video`.

### Probe

| Tool | Description |
|------|-------------|
| `version` | Server version and wire tool-schema version |

### Project

| Tool | Description |
|------|-------------|
| `project_new` | Create an empty project with a Main video track |
| `project_open` | Open a `.cutlass` file (tolerates missing media) |
| `project_save` | Save the open project to a `.cutlass` path |
| `project_get` | Session meta plus compact timeline/media summary |
| `media_import` | Register absolute-path media in the pool (no clips) |

### Edits

| Tool | Description |
|------|-------------|
| `edit_commands_list` | List validated wire command names and one-liners |
| `edit_schema_get` | JSON Schema + description for named commands |
| `edit_apply` | Apply a batch of wire edits as one undo group |
| `edit_undo` | Undo one `edit_apply` batch |
| `edit_redo` | Redo one `edit_apply` batch |

### Render

| Tool | Description |
|------|-------------|
| `frame_get` | Composited PNG at a timeline time (image + caption) |
| `export_video` | Sync H.264/AAC MP4 export of the whole timeline |

## Current limits

- Headless only — does not attach to a running desktop session.
- Stdio transport only (no streamable HTTP yet).
- `export_video` is synchronous and can block for minutes.
- One open project per server process.

## Testing

```bash
cargo test -p cutlass-mcp
```

Unit tests cover the engine host; `tests/mcp_client.rs` drives a real MCP client
against the server in-process over a duplex pipe (schemas, annotations, content
types). Frame grabs need a GPU-capable environment.
