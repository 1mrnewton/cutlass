//! The registry served over MCP: JSON-RPC 2.0, one message a line.
//!
//! The host reads a line, hands it to [`McpServer::handle`], and writes the
//! answer, if any, as one line. The model loop stays in the client. A tool
//! name is the registry name with `_` for `.`, since clients refuse dots.
//! `draft_apply` takes its JSON object as the arguments; every other tool
//! takes its text in `input`.

use serde_json::{Map, Value, json};

use crate::r#loop::ToolSink;
use crate::model::ToolCall;
use crate::tools::{TOOL_NAMES, tool_definitions};

/// The protocol version this server answers with when the client names none.
pub const MCP_PROTOCOL_VERSION: &str = "2025-06-18";

/// Answers MCP requests by running tools on `sink`.
#[derive(Debug)]
pub struct McpServer<S> {
    sink: S,
    name: String,
    version: String,
}

impl<S: ToolSink> McpServer<S> {
    /// Serves `sink`, naming itself `name` at `version` to the client.
    #[must_use]
    pub fn new(sink: S, name: &str, version: &str) -> Self {
        Self {
            sink,
            name: name.to_owned(),
            version: version.to_owned(),
        }
    }

    /// The tools the server runs.
    pub fn sink(&self) -> &S {
        &self.sink
    }

    /// The tools the server runs, to change between messages.
    pub fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }

    /// One message in, its answer out. A notification has no answer.
    pub fn handle(&mut self, line: &str) -> Option<String> {
        let message: Value = match serde_json::from_str(line) {
            Ok(message) => message,
            Err(error) => {
                return Some(failure(
                    &Value::Null,
                    -32700,
                    &format!("parse error: {error}"),
                ));
            }
        };
        let id = message.get("id").cloned();
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            // A response from the client, or nothing we know.
            return id.map(|id| failure(&id, -32600, "invalid request"));
        };
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        let result = match method {
            "initialize" => Ok(self.initialize(&params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": tool_list() })),
            "tools/call" => self.call(&params),
            other if other.starts_with("notifications/") => return None,
            other => Err((-32601, format!("method not found: {other}"))),
        };
        let id = id?;
        Some(match result {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string(),
            Err((code, message)) => failure(&id, code, &message),
        })
    }

    fn initialize(&self, params: &Value) -> Value {
        let version = params
            .get("protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or(MCP_PROTOCOL_VERSION);
        json!({
            "protocolVersion": version,
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": { "name": self.name, "version": self.version },
            "instructions": "Read with draft_query first: every write sends the revision it read. \
                Times are microseconds. draft_apply runs its requests as one undo step.",
        })
    }

    fn call(&mut self, params: &Value) -> Result<Value, (i64, String)> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| (-32602, "tools/call needs a name".to_owned()))?;
        let Some(tool) = registry_name(name) else {
            return Err((-32602, format!("unknown tool {name}")));
        };
        let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);
        let arguments = if tool == "draft.apply" {
            arguments.to_string()
        } else {
            arguments
                .get("input")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        let outcome = self.sink.run_tool(&ToolCall {
            id: String::new(),
            name: tool.to_owned(),
            arguments,
        });
        Ok(json!({
            "content": [{ "type": "text", "text": outcome.content }],
            "isError": outcome.is_error,
        }))
    }
}

fn failure(id: &Value, code: i64, message: &str) -> String {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    })
    .to_string()
}

/// The registry name for an MCP tool name.
fn registry_name(name: &str) -> Option<&'static str> {
    TOOL_NAMES
        .iter()
        .copied()
        .find(|tool| tool.replace('.', "_") == name)
}

/// Each tool with its input schema.
fn tool_list() -> Vec<Value> {
    tool_definitions()
        .into_iter()
        .map(|definition| {
            json!({
                "name": definition.name.replace('.', "_"),
                "description": definition.description,
                "inputSchema": input_schema(&definition.name),
            })
        })
        .collect()
}

fn input_schema(tool: &str) -> Value {
    if tool == "draft.apply" {
        return json!({
            "type": "object",
            "properties": {
                "expected_revision": {
                    "type": "integer",
                    "description": "The revision draft_query returned.",
                },
                "requests": {
                    "type": "array",
                    "description": "Session requests, each with its kind in op, such as \
                        {\"op\":\"move\",\"segment_id\":\"a\",\"track_id\":\"t\",\"start\":0}.",
                    "items": { "type": "object" },
                },
                "dry_run": { "type": "boolean" },
                "request_id": { "type": "string" },
            },
            "required": ["expected_revision", "requests"],
        });
    }
    let hint = input_hint(tool);
    if hint.is_empty() {
        return json!({ "type": "object", "properties": {} });
    }
    let mut properties = Map::new();
    properties.insert(
        "input".to_owned(),
        json!({ "type": "string", "description": hint }),
    );
    json!({ "type": "object", "properties": properties, "required": ["input"] })
}

/// How a tool reads its text, or empty for a tool that takes none.
fn input_hint(tool: &str) -> &'static str {
    match tool {
        "draft.query" => "project | full | track <id> | segment <id> | range <start_us> <end_us>",
        "player.seek" => "<time_us>",
        "media.probe" => "A file path.",
        "speech.start" => "<start_us> <duration_us>",
        "speech.poll" => "The job id speech_start returned.",
        "catalog.search" => "Words to search for.",
        "package.ensure" => "A package id.",
        "export.start" => "new <path> | replace <path>",
        "sandbox.run" => "Source on the first line, then the input JSON.",
        "skill.load" => "A skill name, or what the skill is for.",
        "mcp.call" => "The server, the tool and the arguments, one to a line.",
        "settings.set" => "Settings change only in the app.",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::McpServer;
    use crate::r#loop::{ToolOutcome, ToolSink};
    use crate::model::ToolCall;
    use crate::tools::TOOL_NAMES;

    #[derive(Default)]
    struct Echo {
        calls: Vec<ToolCall>,
    }

    impl ToolSink for Echo {
        fn run_tool(&mut self, call: &ToolCall) -> ToolOutcome {
            self.calls.push(call.clone());
            ToolOutcome {
                content: format!("ran {}", call.name),
                is_error: call.arguments == "fail",
            }
        }
    }

    fn ask(server: &mut McpServer<Echo>, message: &Value) -> Value {
        serde_json::from_str(&server.handle(&message.to_string()).unwrap()).unwrap()
    }

    #[test]
    fn initialize_lists_tools_and_answers_ping() {
        let mut server = McpServer::new(Echo::default(), "cutlass", "0.1.0");
        let reply = ask(
            &mut server,
            &json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}),
        );
        assert_eq!(reply["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(reply["result"]["serverInfo"]["name"], "cutlass");
        assert!(
            server
                .handle(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
                .is_none()
        );
        let tools = ask(
            &mut server,
            &json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        );
        let tools = tools["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), TOOL_NAMES.len());
        for tool in tools {
            let name = tool["name"].as_str().unwrap();
            assert!(
                name.chars()
                    .all(|letter| letter.is_ascii_alphanumeric() || letter == '_'),
                "{name}"
            );
            assert_eq!(tool["inputSchema"]["type"], "object");
        }
        let ping = ask(
            &mut server,
            &json!({"jsonrpc":"2.0","id":"p","method":"ping"}),
        );
        assert_eq!(ping["id"], "p");
        assert_eq!(ping["result"], json!({}));
    }

    #[test]
    fn a_call_runs_the_registry_tool_with_its_text() {
        let mut server = McpServer::new(Echo::default(), "cutlass", "0.1.0");
        let reply = ask(
            &mut server,
            &json!({"jsonrpc":"2.0","id":3,"method":"tools/call",
                "params":{"name":"draft_query","arguments":{"input":"project"}}}),
        );
        assert_eq!(reply["result"]["content"][0]["text"], "ran draft.query");
        assert_eq!(reply["result"]["isError"], false);
        let apply = json!({"expected_revision":0,"requests":[]});
        ask(
            &mut server,
            &json!({"jsonrpc":"2.0","id":4,"method":"tools/call",
                "params":{"name":"draft_apply","arguments":apply}}),
        );
        let failed = ask(
            &mut server,
            &json!({"jsonrpc":"2.0","id":5,"method":"tools/call",
                "params":{"name":"player_seek","arguments":{"input":"fail"}}}),
        );
        assert_eq!(failed["result"]["isError"], true);
        let calls = &server.sink().calls;
        assert_eq!(calls[0].name, "draft.query");
        assert_eq!(calls[0].arguments, "project");
        assert_eq!(calls[1].name, "draft.apply");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[1].arguments).unwrap(),
            apply
        );
    }

    #[test]
    fn unknown_methods_tools_and_bad_lines_are_errors() {
        let mut server = McpServer::new(Echo::default(), "cutlass", "0.1.0");
        let unknown = ask(
            &mut server,
            &json!({"jsonrpc":"2.0","id":6,"method":"resources/list"}),
        );
        assert_eq!(unknown["error"]["code"], -32601);
        let tool = ask(
            &mut server,
            &json!({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"draft.query"}}),
        );
        assert_eq!(tool["error"]["code"], -32602);
        let parse: Value = serde_json::from_str(&server.handle("{").unwrap()).unwrap();
        assert_eq!(parse["error"]["code"], -32700);
        assert!(server.sink().calls.is_empty());
    }
}
