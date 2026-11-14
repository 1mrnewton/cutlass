//! The client half of MCP over one connection: the handshake, listing a
//! server's tools, and calling one.
//!
//! The host owns the bytes: a child process's stdin and stdout, or HTTP
//! posts. It hands this module one JSON-RPC message at a time through
//! [`McpWire`]. This module does not spawn a process or open a socket.

use serde_json::{Value, json};

use crate::mcp::McpError;
use crate::mcp_server::MCP_PROTOCOL_VERSION;

/// One JSON-RPC message at a time, each way.
pub trait McpWire {
    /// Send one message, a line of JSON.
    fn send(&mut self, message: &str) -> Result<(), McpError>;
    /// The next message from the server. The host gives up after its own
    /// time limit and answers with an error.
    fn receive(&mut self) -> Result<String, McpError>;
}

/// The tool name `mcp.call` takes to list a server's tools.
pub const LIST_TOOLS: &str = "tools/list";

/// One server, connected through `wire`. The handshake runs on first use.
#[derive(Debug)]
pub struct McpConnection<W> {
    wire: W,
    next_id: u64,
    ready: bool,
    server: String,
}

impl<W: McpWire> McpConnection<W> {
    /// A connection to the server named `server` that has not spoken yet.
    #[must_use]
    pub fn new(wire: W, server: &str) -> Self {
        Self {
            wire,
            next_id: 1,
            ready: false,
            server: server.to_owned(),
        }
    }

    /// The host's half.
    pub fn wire_mut(&mut self) -> &mut W {
        &mut self.wire
    }

    /// `tool` with `arguments`, a JSON object or empty; [`LIST_TOOLS`]
    /// lists the server's tools instead. The answer is the tool's text.
    pub fn call(&mut self, tool: &str, arguments: &str) -> Result<String, McpError> {
        self.ensure_ready()?;
        if tool == LIST_TOOLS || tool == "list" {
            let result = self.request("tools/list", &json!({}))?;
            return Ok(show_tools(&result));
        }
        let arguments = if arguments.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str::<Value>(arguments).map_err(|error| {
                McpError::Transport(format!(
                    "the arguments for {tool} on {} are not JSON: {error}",
                    self.server
                ))
            })?
        };
        let result = self.request(
            "tools/call",
            &json!({ "name": tool, "arguments": arguments }),
        )?;
        let text = content_text(&result);
        if result.get("isError").and_then(Value::as_bool) == Some(true) {
            return Err(McpError::Transport(format!(
                "{tool} on {} failed: {text}",
                self.server
            )));
        }
        Ok(text)
    }

    fn ensure_ready(&mut self) -> Result<(), McpError> {
        if self.ready {
            return Ok(());
        }
        self.request(
            "initialize",
            &json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": { "name": "cutlass", "version": env!("CARGO_PKG_VERSION") },
            }),
        )?;
        self.wire.send(
            &json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }).to_string(),
        )?;
        self.ready = true;
        Ok(())
    }

    fn request(&mut self, method: &str, params: &Value) -> Result<Value, McpError> {
        let id = self.next_id;
        self.next_id += 1;
        self.wire.send(
            &json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }).to_string(),
        )?;
        loop {
            let line = self.wire.receive()?;
            let Ok(message) = serde_json::from_str::<Value>(&line) else {
                // A server may print something that is not a message.
                continue;
            };
            let has_method = message.get("method").is_some();
            match message.get("id") {
                Some(answer) if !has_method && answer.as_u64() == Some(id) => {
                    if let Some(error) = message.get("error") {
                        let text = error
                            .get("message")
                            .and_then(Value::as_str)
                            .map_or_else(|| error.to_string(), str::to_owned);
                        return Err(McpError::Transport(format!(
                            "{} answered {method} with an error: {text}",
                            self.server
                        )));
                    }
                    return Ok(message.get("result").cloned().unwrap_or(Value::Null));
                }
                Some(asked) if has_method => {
                    // The server asks the client something (sampling, roots,
                    // a ping). Only a ping is answered; the rest are refused.
                    let reply = if message.get("method").and_then(Value::as_str) == Some("ping") {
                        json!({ "jsonrpc": "2.0", "id": asked, "result": {} })
                    } else {
                        json!({
                            "jsonrpc": "2.0",
                            "id": asked,
                            "error": { "code": -32601, "message": "Cutlass does not offer that" },
                        })
                    };
                    self.wire.send(&reply.to_string())?;
                }
                // A notification, or an answer to something else.
                _ => {}
            }
        }
    }
}

/// The text items of a tool result, joined; other items are named.
fn content_text(result: &Value) -> String {
    let Some(items) = result.get("content").and_then(Value::as_array) else {
        return result.to_string();
    };
    let mut out = Vec::new();
    for item in items {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => out.push(
                item.get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            ),
            Some(other) => out.push(format!("[{other} content left out]")),
            None => out.push(item.to_string()),
        }
    }
    if out.is_empty()
        && let Some(structured) = result.get("structuredContent")
    {
        return structured.to_string();
    }
    out.join("\n")
}

/// One line a tool: its name and description, then its input schema.
fn show_tools(result: &Value) -> String {
    let mut out = String::new();
    for tool in result
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = tool.get("name").and_then(Value::as_str).unwrap_or_default();
        let about = tool
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        out.push_str(name);
        if !about.is_empty() {
            out.push_str(": ");
            out.push_str(&about);
        }
        if let Some(schema) = tool.get("inputSchema") {
            out.push_str("\n  arguments: ");
            out.push_str(&schema.to_string());
        }
        out.push('\n');
    }
    if out.is_empty() {
        return "the server lists no tools".to_owned();
    }
    out.trim_end().to_owned()
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use serde_json::{Value, json};

    use super::{McpConnection, McpWire};
    use crate::mcp::McpError;

    /// A server written as a function of each request it is sent.
    struct Scripted {
        sent: Vec<Value>,
        queue: VecDeque<String>,
    }

    impl McpWire for Scripted {
        fn send(&mut self, message: &str) -> Result<(), McpError> {
            let value: Value = serde_json::from_str(message).unwrap();
            if let (Some(id), Some(method)) = (
                value.get("id").and_then(Value::as_u64),
                value.get("method").and_then(Value::as_str),
            ) {
                // Noise and a server request come before each answer.
                self.queue.push_back("not json".to_owned());
                self.queue.push_back(
                    json!({"jsonrpc":"2.0","method":"notifications/message","params":{}})
                        .to_string(),
                );
                let result = match method {
                    "initialize" => json!({ "protocolVersion": "2025-06-18", "capabilities": {} }),
                    "tools/list" => json!({ "tools": [{
                        "name": "weather",
                        "description": "The weather\n in a city.",
                        "inputSchema": { "type": "object" },
                    }]}),
                    "tools/call" if value["params"]["name"] == "weather" => json!({
                        "content": [{ "type": "text", "text": format!("sunny in {}", value["params"]["arguments"]["city"].as_str().unwrap_or("?")) }],
                    }),
                    "tools/call" => json!({
                        "content": [{ "type": "text", "text": "no such tool" }],
                        "isError": true,
                    }),
                    _ => json!({}),
                };
                if method == "tools/call" {
                    self.queue
                        .push_back(json!({"jsonrpc":"2.0","id":"s1","method":"ping"}).to_string());
                }
                self.queue
                    .push_back(json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string());
            }
            self.sent.push(value);
            Ok(())
        }

        fn receive(&mut self) -> Result<String, McpError> {
            self.queue
                .pop_front()
                .ok_or_else(|| McpError::Transport("closed".to_owned()))
        }
    }

    #[test]
    fn a_connection_shakes_hands_lists_and_calls() {
        let mut connection = McpConnection::new(
            Scripted {
                sent: Vec::new(),
                queue: VecDeque::new(),
            },
            "local",
        );
        let tools = connection.call("tools/list", "").unwrap();
        assert!(
            tools.starts_with("weather: The weather in a city."),
            "{tools}"
        );
        assert!(
            tools.contains("arguments: {\"type\":\"object\"}"),
            "{tools}"
        );
        let answer = connection.call("weather", r#"{"city":"Accra"}"#).unwrap();
        assert_eq!(answer, "sunny in Accra");
        let failed = connection.call("other", "{}").unwrap_err();
        assert!(failed.to_string().contains("no such tool"), "{failed}");
        assert!(connection.call("weather", "not json").is_err());
        let sent = &connection.wire_mut().sent;
        let methods: Vec<&str> = sent
            .iter()
            .filter_map(|message| message.get("method").and_then(Value::as_str))
            .collect();
        assert_eq!(
            methods,
            [
                "initialize",
                "notifications/initialized",
                "tools/list",
                "tools/call",
                "tools/call",
            ],
            "the handshake runs once"
        );
        // The server's ping was answered.
        assert!(
            sent.iter()
                .any(|message| message["id"] == "s1" && message.get("result").is_some())
        );
    }
}
