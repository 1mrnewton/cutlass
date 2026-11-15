//! A model served on this machine over the chat completions protocol that
//! local runners share (Ollama, llama.cpp's server, LM Studio): one JSON
//! POST a round, tools as functions.
//!
//! Tool names go out with `_` for `.`, as for MCP, since some runners
//! refuse dots. `draft.apply` takes its JSON object as the arguments; every
//! other tool takes its text in `input`.
//!
//! The transport speaks plain HTTP/1.1 to a loopback address only, so the
//! prompt never leaves the machine on this path.

use std::io::{BufRead, BufReader, Write};
use std::net::{IpAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use serde_json::{Value, json};

use crate::mcp_server::{input_schema, registry_name, takes_object};
use crate::model::{Completion, CompletionRequest, Message, Model, ModelError, ToolCall};

/// Sends one request body and returns the reply body.
pub trait ChatTransport {
    /// POSTs `body`, JSON, and returns the answer's body.
    fn post(&mut self, body: &str) -> Result<String, ModelError>;
}

/// A chat runner on this machine, such as Ollama at
/// `http://127.0.0.1:11434/v1/chat/completions`.
#[derive(Debug)]
pub struct ChatModel<T> {
    transport: T,
    model: String,
    system: String,
    /// The model refused a request with a picture and took the same one
    /// without: it is sent no more pictures.
    blind: bool,
}

/// What the model is told before the user's first words.
pub const SYSTEM_PROMPT: &str = "You edit a video project in Cutlass with tools. \
Act with the tools; do not ask the user for ids you can read. \
Read with draft_query first (\"project\" lists the tracks, \"track <id>\" lists its clips as id@start+length, \"segment <id>\" shows one clip with what it stores and its text or media, \"materials\" lists the imported media and the texts): \
every draft_apply sends the revision it read, and a stale revision is refused, so read again. \
Times are microseconds: 1 second is 1000000. draft_apply runs its requests as one undo step; if one fails, none land. \
Requests, each with its kind in op, for example: \
{\"op\":\"split\",\"segment_id\":\"a\",\"at\":1000000,\"right_id\":\"a-2\"} (at is on the timeline; right_id names the new right piece); \
{\"op\":\"move\",\"segment_id\":\"a\",\"track_id\":\"main\",\"start\":0}; \
{\"op\":\"trim\",\"segment_id\":\"a\",\"edge\":\"start\",\"time\":500000} (edge start or end); \
{\"op\":\"remove\",\"segment_id\":\"a\"}. \
There are over a hundred kinds (text, captions, shapes, effects, filters, color, masks, keyframes, transitions, speed, canvas): \
draft_query \"ops\" lists them all, and draft_query \"ops add_text set_position\" gives the exact JSON of the ones you need. Look a kind up before you use it the first time. \
For a bigger job (captions, removing pauses, motion graphics, a product demo, setting up a project), skill_load it first and follow it. \
settings_get and settings_set read and change the app's own settings, not the project: its canvas, frame rate and clips are draft_apply requests. \
player_look shows you the frame at a time as a picture: after a change to how things look (a title, a position, a color, an effect), look and fix what is off. \
An ask may end with a note in brackets, [Editor now: ...]: where the playhead is, the clips the user selected and the clips at the playhead, each as id@start+length:track. \
\"This\", \"here\" and \"the selected clip\" mean those. The note is from when the user asked, so read again once you have edited. \
When done, say briefly what you changed.";

/// What the model is told about this user beyond [`SYSTEM_PROMPT`]: the
/// MCP servers they turned on and, while Memory is on, their notes.
#[must_use]
pub fn prompt_context(settings: &cutlass_settings::Settings, notes: &str) -> String {
    let mut out = String::new();
    let servers: Vec<&str> = settings
        .mcp_servers
        .iter()
        .filter(|server| server.enabled && !server.name.is_empty())
        .map(|server| server.name.as_str())
        .collect();
    if !servers.is_empty() {
        out.push_str(&format!(
            "MCP servers the user turned on, reached with mcp_call (tool tools/list lists a server's tools): {}.",
            servers.join(", ")
        ));
    }
    if settings.memory && !notes.trim().is_empty() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str("Notes the user keeps for you:\n");
        out.push_str(notes.trim());
    }
    out
}

impl<T: ChatTransport> ChatModel<T> {
    /// Adds `context` after the system prompt, such as [`prompt_context`].
    #[must_use]
    pub fn with_context(mut self, context: &str) -> Self {
        if !context.trim().is_empty() {
            self.system.push_str("\n\n");
            self.system.push_str(context.trim());
        }
        self
    }

    /// Talks to `model` through `transport`.
    #[must_use]
    pub fn new(transport: T, model: &str) -> Self {
        Self {
            transport,
            model: model.to_owned(),
            system: SYSTEM_PROMPT.to_owned(),
            blind: false,
        }
    }

    /// The transport, to look at in tests.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// The request body for `request`.
    ///
    /// A tool message is text only in this protocol, so a tool's picture
    /// follows the results of its round in a user message, as an
    /// `image_url` part holding the file. A model that refused pictures
    /// is told there was one instead.
    #[must_use]
    pub fn body(&self, request: &CompletionRequest) -> Value {
        let mut messages = vec![json!({ "role": "system", "content": self.system })];
        // The pictures of the tool results since the last other message.
        let mut shown: Vec<Value> = Vec::new();
        for message in &request.messages {
            if !matches!(message, Message::Tool { .. }) && !shown.is_empty() {
                messages.push(json!({ "role": "user", "content": std::mem::take(&mut shown) }));
            }
            if let Message::Tool {
                name,
                picture: Some(picture),
                ..
            } = message
            {
                let tool = name.replace('.', "_");
                if self.blind {
                    shown.push(json!({
                        "type": "text",
                        "text": format!(
                            "{tool} drew a picture, but this model does not take pictures, so it \
                             was not sent. Say so if the answer needs it."
                        ),
                    }));
                } else {
                    shown.push(json!({
                        "type": "text",
                        "text": format!("The picture {tool} drew:"),
                    }));
                    shown.push(json!({
                        "type": "image_url",
                        "image_url": { "url": picture.data_url() },
                    }));
                }
            }
            messages.push(match message {
                Message::User { text } => json!({ "role": "user", "content": text }),
                Message::Assistant { text, calls } => {
                    let mut out = json!({ "role": "assistant", "content": text });
                    if !calls.is_empty() {
                        out["tool_calls"] = calls
                            .iter()
                            .map(|call| {
                                json!({
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name.replace('.', "_"),
                                        "arguments": wire_arguments(&call.name, &call.arguments),
                                    },
                                })
                            })
                            .collect();
                    }
                    out
                }
                Message::Tool {
                    call_id,
                    name,
                    content,
                    is_error,
                    ..
                } => json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name.replace('.', "_"),
                    "content": if *is_error { format!("error: {content}") } else { content.clone() },
                }),
            });
        }
        if !shown.is_empty() {
            messages.push(json!({ "role": "user", "content": shown }));
        }
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name.replace('.', "_"),
                        "description": tool.description,
                        "parameters": input_schema(&tool.name),
                    },
                })
            })
            .collect();
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        body
    }
}

impl<T: ChatTransport> Model for ChatModel<T> {
    fn complete(&mut self, request: &CompletionRequest) -> Result<Completion, ModelError> {
        let body = self.body(request).to_string();
        let first = self
            .transport
            .post(&body)
            .and_then(|reply| parse_reply(&reply));
        let pictures = request.messages.iter().any(|message| {
            matches!(
                message,
                Message::Tool {
                    picture: Some(_),
                    ..
                }
            )
        });
        let Err(refused) = first else {
            return first;
        };
        if self.blind || !pictures {
            return Err(refused);
        }
        // Many models on this protocol take no pictures, and nothing says
        // which before one is sent. The same request without them tells:
        // if it is answered, the pictures were the cause.
        self.blind = true;
        let body = self.body(request).to_string();
        let again = self
            .transport
            .post(&body)
            .and_then(|reply| parse_reply(&reply));
        match again {
            Ok(completion) => Ok(completion),
            Err(_) => {
                // Not the pictures: the first error is the one to show.
                self.blind = false;
                Err(refused)
            }
        }
    }
}

/// The completion in a chat completions reply.
pub fn parse_reply(reply: &str) -> Result<Completion, ModelError> {
    let value: Value = serde_json::from_str(reply)
        .map_err(|error| ModelError::Inner(format!("the model's reply is not JSON: {error}")))?;
    if let Some(error) = value.get("error") {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .map_or_else(|| error.to_string(), str::to_owned);
        return Err(ModelError::Inner(format!("the model refused: {message}")));
    }
    let message = value
        .pointer("/choices/0/message")
        .ok_or_else(|| ModelError::Inner("the model's reply has no message".to_owned()))?;
    let text = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned();
    let mut calls = Vec::new();
    for (index, call) in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let wire = call
            .pointer("/function/name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let name = registry_name(wire).map_or_else(|| wire.to_owned(), str::to_owned);
        let arguments = call
            .pointer("/function/arguments")
            .cloned()
            .unwrap_or(Value::Null);
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .map_or_else(|| format!("call-{index}"), str::to_owned);
        calls.push(ToolCall {
            arguments: registry_arguments(&name, &arguments),
            id,
            name,
        });
    }
    Ok(Completion {
        text,
        calls,
        proposed: Vec::new(),
    })
}

/// The arguments a tool call carries on the wire, as a JSON string.
fn wire_arguments(tool: &str, arguments: &str) -> String {
    if takes_object(tool) {
        return arguments.to_owned();
    }
    if arguments.is_empty() {
        return "{}".to_owned();
    }
    json!({ "input": arguments }).to_string()
}

/// The registry's text for arguments from the wire: a JSON string, or an
/// object some runners send as is.
fn registry_arguments(tool: &str, arguments: &Value) -> String {
    let object = match arguments {
        Value::String(text) => serde_json::from_str(text).unwrap_or(Value::String(text.clone())),
        other => other.clone(),
    };
    if takes_object(tool) {
        return match object {
            Value::String(text) => text,
            other => other.to_string(),
        };
    }
    match object {
        Value::Object(map) => map
            .get("input")
            .map(|input| match input {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            })
            .unwrap_or_default(),
        Value::String(text) => text,
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Plain HTTP/1.1 to a runner on a loopback address.
#[derive(Clone, Debug)]
pub struct LoopbackHttp {
    host: String,
    port: u16,
    path: String,
    timeout: Duration,
}

impl LoopbackHttp {
    /// The runner at `url`, such as
    /// `http://127.0.0.1:11434/v1/chat/completions`. Only `http` to this
    /// machine is taken.
    pub fn new(url: &str) -> Result<Self, ModelError> {
        let rest = url.strip_prefix("http://").ok_or_else(|| {
            ModelError::Inner(format!(
                "the local model address must start with http://, not {url}"
            ))
        })?;
        let (authority, path) = rest
            .find('/')
            .map_or((rest, "/"), |at| (&rest[..at], &rest[at..]));
        let (host, port) = match authority.rsplit_once(':') {
            Some((host, port)) if !host.ends_with(']') || authority.starts_with('[') => (
                host,
                port.parse::<u16>()
                    .map_err(|_| ModelError::Inner(format!("the port in {url} is not a number")))?,
            ),
            _ => (authority, 80),
        };
        let host = host.trim_start_matches('[').trim_end_matches(']');
        let local = host == "localhost"
            || host
                .parse::<IpAddr>()
                .is_ok_and(|address| address.is_loopback());
        if !local {
            return Err(ModelError::Inner(format!(
                "the local model must run on this machine, not {host}"
            )));
        }
        Ok(Self {
            host: host.to_owned(),
            port,
            path: path.to_owned(),
            timeout: Duration::from_secs(600),
        })
    }

    /// Wait at most `timeout` to connect and for each read.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    fn unreachable(&self, error: &std::io::Error) -> ModelError {
        ModelError::Inner(format!(
            "no local model answered at {}:{} ({error}); start one, such as Ollama, or change the address in Settings",
            self.host, self.port
        ))
    }
}

impl LoopbackHttp {
    /// Send one request for `path` with `body`, if any, and read the answer.
    fn exchange(&self, method: &str, path: &str, body: &str) -> Result<String, ModelError> {
        let address = (self.host.as_str(), self.port)
            .to_socket_addrs()
            .map_err(|error| self.unreachable(&error))?
            .find(|address| address.ip().is_loopback())
            .ok_or_else(|| ModelError::Inner(format!("{} is not this machine", self.host)))?;
        let mut stream = TcpStream::connect_timeout(&address, self.timeout)
            .map_err(|error| self.unreachable(&error))?;
        stream
            .set_read_timeout(Some(self.timeout))
            .and_then(|()| stream.set_write_timeout(Some(self.timeout)))
            .map_err(|error| self.unreachable(&error))?;
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.host,
            self.port,
            body.len()
        );
        stream
            .write_all(head.as_bytes())
            .and_then(|()| stream.write_all(body.as_bytes()))
            .and_then(|()| stream.flush())
            .map_err(|error| self.unreachable(&error))?;
        read_response(BufReader::new(stream))
    }

    /// The models the runner has, by the names it takes, from its
    /// `/v1/models` beside the chat address (Ollama, llama.cpp's server and
    /// LM Studio all answer it).
    pub fn models(&self) -> Result<Vec<String>, ModelError> {
        let path = self
            .path
            .strip_suffix("/chat/completions")
            .map_or_else(|| "/v1/models".to_owned(), |base| format!("{base}/models"));
        let reply = self.exchange("GET", &path, "")?;
        let value: Value = serde_json::from_str(&reply)
            .map_err(|error| ModelError::Inner(format!("the model list is not JSON: {error}")))?;
        let mut names: Vec<String> = value
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|model| model.get("id").and_then(Value::as_str))
            .map(str::to_owned)
            .collect();
        names.sort();
        names.dedup();
        Ok(names)
    }
}

impl ChatTransport for LoopbackHttp {
    fn post(&mut self, body: &str) -> Result<String, ModelError> {
        self.exchange("POST", &self.path, body)
    }
}

/// The body of an HTTP/1.1 response: by its length, chunked, or to the end.
fn read_response(mut reader: impl BufRead) -> Result<String, ModelError> {
    let broken =
        |error: std::io::Error| ModelError::Inner(format!("the model's reply broke off: {error}"));
    let mut status = String::new();
    reader.read_line(&mut status).map_err(broken)?;
    let code: u16 = status
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .ok_or_else(|| {
            ModelError::Inner(format!("the model's reply is not HTTP: {}", status.trim()))
        })?;
    let mut length = None;
    let mut chunked = false;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).map_err(broken)? == 0 {
            break;
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            let value = value.trim();
            if name.eq_ignore_ascii_case("content-length") {
                length = value.parse::<usize>().ok();
            } else if name.eq_ignore_ascii_case("transfer-encoding")
                && value.to_ascii_lowercase().contains("chunked")
            {
                chunked = true;
            }
        }
    }
    let mut body = Vec::new();
    if chunked {
        loop {
            let mut size = String::new();
            reader.read_line(&mut size).map_err(broken)?;
            let size = usize::from_str_radix(size.trim().split(';').next().unwrap_or("0"), 16)
                .map_err(|_| ModelError::Inner("the model's reply has a bad chunk".to_owned()))?;
            if size == 0 {
                break;
            }
            let mut chunk = vec![0; size];
            reader.read_exact(&mut chunk).map_err(broken)?;
            body.extend_from_slice(&chunk);
            let mut end = String::new();
            reader.read_line(&mut end).map_err(broken)?;
        }
    } else if let Some(length) = length {
        body.resize(length, 0);
        reader.read_exact(&mut body).map_err(broken)?;
    } else {
        reader.read_to_end(&mut body).map_err(broken)?;
    }
    let text = String::from_utf8(body)
        .map_err(|_| ModelError::Inner("the model's reply is not text".to_owned()))?;
    if !(200..300).contains(&code) {
        let detail = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|value| {
                value
                    .pointer("/error/message")
                    .or_else(|| value.get("error"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| text.chars().take(200).collect());
        return Err(ModelError::Inner(format!(
            "the local model answered {code}: {detail}"
        )));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;

    use serde_json::{Value, json};

    use super::{ChatModel, ChatTransport, LoopbackHttp, parse_reply, read_response};
    use crate::model::{CompletionRequest, Message, Model, ModelError, ToolCall, ToolDefinition};

    struct Canned {
        replies: Vec<String>,
        bodies: Vec<Value>,
    }

    impl ChatTransport for Canned {
        fn post(&mut self, body: &str) -> Result<String, ModelError> {
            self.bodies.push(serde_json::from_str(body).unwrap());
            if self.replies.is_empty() {
                return Err(ModelError::NoMoreSteps);
            }
            Ok(self.replies.remove(0))
        }
    }

    fn reply(message: &Value) -> String {
        json!({ "choices": [{ "index": 0, "message": message }] }).to_string()
    }

    #[test]
    fn a_request_goes_out_with_tools_and_calls_come_back_by_registry_name() {
        let canned = Canned {
            replies: vec![reply(&json!({
                "role": "assistant",
                "content": "Looking.",
                "tool_calls": [
                    { "id": "a", "type": "function",
                      "function": { "name": "draft_query", "arguments": "{\"input\":\"project\"}" } },
                    { "id": "b", "type": "function",
                      "function": { "name": "draft_apply",
                                    "arguments": { "expected_revision": 3, "requests": [] } } },
                ],
            }))],
            bodies: Vec::new(),
        };
        let mut model = ChatModel::new(canned, "qwen");
        let request = CompletionRequest {
            messages: vec![
                Message::User {
                    text: "trim it".to_owned(),
                },
                Message::Assistant {
                    text: String::new(),
                    calls: vec![ToolCall {
                        id: "x".to_owned(),
                        name: "player.seek".to_owned(),
                        arguments: "5".to_owned(),
                    }],
                },
                Message::Tool {
                    call_id: "x".to_owned(),
                    name: "player.seek".to_owned(),
                    content: "no player".to_owned(),
                    is_error: true,
                    picture: None,
                },
            ],
            tools: vec![ToolDefinition {
                name: "draft.query".to_owned(),
                description: "Read the draft.".to_owned(),
            }],
        };
        let completion = model.complete(&request).unwrap();
        assert_eq!(completion.text, "Looking.");
        assert_eq!(completion.calls[0].name, "draft.query");
        assert_eq!(completion.calls[0].arguments, "project");
        assert_eq!(completion.calls[1].name, "draft.apply");
        let applied: Value = serde_json::from_str(&completion.calls[1].arguments).unwrap();
        assert_eq!(applied["expected_revision"], 3);

        let body = &model.transport().bodies[0];
        assert_eq!(body["model"], "qwen");
        assert_eq!(body["stream"], false);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["content"], "trim it");
        let call = &body["messages"][2]["tool_calls"][0];
        assert_eq!(call["function"]["name"], "player_seek");
        assert_eq!(call["function"]["arguments"], "{\"input\":\"5\"}");
        assert_eq!(body["messages"][3]["role"], "tool");
        assert_eq!(body["messages"][3]["tool_call_id"], "x");
        assert_eq!(body["messages"][3]["content"], "error: no player");
        assert_eq!(body["tools"][0]["function"]["name"], "draft_query");
        assert_eq!(
            body["tools"][0]["function"]["parameters"]["required"][0],
            "input"
        );
    }

    #[test]
    fn a_refusal_and_a_broken_reply_are_errors() {
        assert!(parse_reply("not json").is_err());
        let refused = parse_reply(r#"{"error":{"message":"model not found"}}"#).unwrap_err();
        assert!(refused.to_string().contains("model not found"), "{refused}");
        assert!(parse_reply(r#"{"choices":[]}"#).is_err());
        let plain = parse_reply(&reply(
            &json!({ "role": "assistant", "content": " Done. " }),
        ))
        .unwrap();
        assert_eq!(plain.text, "Done.");
        assert!(plain.calls.is_empty());
    }

    #[test]
    fn only_this_machine_is_taken() {
        assert!(LoopbackHttp::new("http://127.0.0.1:11434/v1/chat/completions").is_ok());
        assert!(LoopbackHttp::new("http://localhost:8080/v1/chat/completions").is_ok());
        assert!(LoopbackHttp::new("http://[::1]:8080/x").is_ok());
        assert!(LoopbackHttp::new("https://127.0.0.1/x").is_err());
        assert!(LoopbackHttp::new("http://example.com/v1/chat/completions").is_err());
        assert!(LoopbackHttp::new("http://10.0.0.2:11434/x").is_err());
        assert!(LoopbackHttp::new("http://127.0.0.1:port/x").is_err());
    }

    #[test]
    fn a_response_is_read_by_length_by_chunks_or_to_the_end() {
        let sized = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhiXX";
        assert_eq!(read_response(sized.as_bytes()).unwrap(), "hi");
        let chunked = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2;x=1\r\nde\r\n0\r\n\r\n";
        assert_eq!(read_response(chunked.as_bytes()).unwrap(), "abcde");
        let open = "HTTP/1.0 200 OK\r\n\r\nall of it";
        assert_eq!(read_response(open.as_bytes()).unwrap(), "all of it");
        let detail = r#"{"error":{"message":"model missing"}}"#;
        let missing = format!(
            "HTTP/1.1 404 Not Found\r\nContent-Length: {}\r\n\r\n{detail}",
            detail.len()
        );
        let error = read_response(missing.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("404: model missing"), "{error}");
    }

    fn looked(call_id: &str, picture: Option<crate::Picture>) -> Message {
        Message::Tool {
            call_id: call_id.to_owned(),
            name: "player.look".to_owned(),
            content: format!("frame {call_id}"),
            is_error: false,
            picture,
        }
    }

    fn look_turn(picture: &crate::Picture) -> Vec<Message> {
        vec![
            Message::User {
                text: "look".to_owned(),
            },
            Message::Assistant {
                text: String::new(),
                calls: ["a", "b"]
                    .iter()
                    .map(|id| ToolCall {
                        id: (*id).to_owned(),
                        name: "player.look".to_owned(),
                        arguments: String::new(),
                    })
                    .collect(),
            },
            looked("a", Some(picture.clone())),
            looked("b", None),
            Message::Assistant {
                text: "Seen.".to_owned(),
                calls: Vec::new(),
            },
        ]
    }

    #[test]
    fn a_tools_picture_follows_the_results_of_its_round_in_a_user_message() {
        let picture = crate::Picture::jpeg(b"frame");
        let model = ChatModel::new(
            Canned {
                replies: Vec::new(),
                bodies: Vec::new(),
            },
            "small",
        );
        let body = model.body(&CompletionRequest {
            messages: look_turn(&picture),
            tools: Vec::new(),
        });
        let messages = body["messages"].as_array().unwrap();
        let roles: Vec<&str> = messages
            .iter()
            .map(|message| message["role"].as_str().unwrap())
            .collect();
        // Both tool messages stay together, after the call that asked.
        assert_eq!(
            roles,
            [
                "system",
                "user",
                "assistant",
                "tool",
                "tool",
                "user",
                "assistant"
            ]
        );
        assert_eq!(messages[3]["content"], "frame a");
        assert_eq!(
            messages[5]["content"],
            json!([
                { "type": "text", "text": "The picture player_look drew:" },
                { "type": "image_url",
                  "image_url": { "url": "data:image/jpeg;base64,ZnJhbWU=" } },
            ])
        );
        // A turn that ends on the look still sends the picture.
        let mut open = look_turn(&picture);
        open.pop();
        let body = model.body(&CompletionRequest {
            messages: open,
            tools: Vec::new(),
        });
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 6);
        assert_eq!(messages[5]["content"][1]["type"], "image_url");
        // No picture: nothing is added.
        let body = model.body(&CompletionRequest {
            messages: vec![looked("a", None)],
            tools: Vec::new(),
        });
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn a_model_that_refuses_pictures_is_asked_again_without_them() {
        let picture = crate::Picture::jpeg(b"frame");
        let refusal = json!({ "error": { "message": "this model does not support images" } });
        let mut model = ChatModel::new(
            Canned {
                replies: vec![
                    refusal.to_string(),
                    reply(&json!({ "role": "assistant", "content": "I cannot see it." })),
                    reply(&json!({ "role": "assistant", "content": "Still here." })),
                ],
                bodies: Vec::new(),
            },
            "small",
        );
        let mut open = look_turn(&picture);
        open.pop();
        let request = CompletionRequest {
            messages: open,
            tools: Vec::new(),
        };
        let completion = model.complete(&request).unwrap();
        assert_eq!(completion.text, "I cannot see it.");
        let bodies = &model.transport().bodies;
        assert_eq!(bodies.len(), 2);
        assert_eq!(bodies[0]["messages"][5]["content"][1]["type"], "image_url");
        // The second try says there was a picture and sends none.
        let told = &bodies[1]["messages"][5]["content"];
        assert_eq!(told.as_array().unwrap().len(), 1);
        assert!(
            told[0]["text"]
                .as_str()
                .unwrap()
                .contains("does not take pictures")
        );
        assert!(!bodies[1].to_string().contains("image_url"));
        // The next round goes without pictures at once.
        model.complete(&request).unwrap();
        let bodies = &model.transport().bodies;
        assert_eq!(bodies.len(), 3);
        assert!(!bodies[2].to_string().contains("image_url"));
    }

    #[test]
    fn an_error_that_is_not_the_pictures_is_shown_and_pictures_are_still_sent() {
        let picture = crate::Picture::jpeg(b"frame");
        let down = json!({ "error": { "message": "overloaded" } });
        let mut model = ChatModel::new(
            Canned {
                replies: vec![
                    down.to_string(),
                    down.to_string(),
                    reply(&json!({ "role": "assistant", "content": "Seen." })),
                ],
                bodies: Vec::new(),
            },
            "small",
        );
        let mut open = look_turn(&picture);
        open.pop();
        let request = CompletionRequest {
            messages: open,
            tools: Vec::new(),
        };
        let error = model.complete(&request).unwrap_err();
        assert_eq!(
            error,
            ModelError::Inner("the model refused: overloaded".to_owned())
        );
        // Both tries failed, so the model is not taken for blind.
        assert_eq!(model.complete(&request).unwrap().text, "Seen.");
        let bodies = &model.transport().bodies;
        assert_eq!(bodies[2]["messages"][5]["content"][1]["type"], "image_url");
        // A request with no picture fails once and is not sent again.
        let mut plain = ChatModel::new(
            Canned {
                replies: vec![down.to_string()],
                bodies: Vec::new(),
            },
            "small",
        );
        let request = CompletionRequest {
            messages: vec![Message::User {
                text: "hi".to_owned(),
            }],
            tools: Vec::new(),
        };
        assert!(plain.complete(&request).is_err());
        assert_eq!(plain.transport().bodies.len(), 1);
    }

    #[test]
    fn a_round_trips_through_a_server_on_this_machine() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut length = 0;
            let mut first = String::new();
            reader.read_line(&mut first).unwrap();
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line.trim().is_empty() {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let sent: Value = serde_json::from_slice(&body).unwrap();
            let answer = reply(&json!({ "role": "assistant", "content": "hello" }));
            let mut stream = stream;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{answer}",
                answer.len()
            )
            .unwrap();
            (first, sent)
        });
        let transport =
            LoopbackHttp::new(&format!("http://127.0.0.1:{port}/v1/chat/completions")).unwrap();
        let mut model = ChatModel::new(transport, "small");
        let completion = model
            .complete(&CompletionRequest {
                messages: vec![Message::User {
                    text: "hi".to_owned(),
                }],
                tools: Vec::new(),
            })
            .unwrap();
        assert_eq!(completion.text, "hello");
        let (first, sent) = server.join().unwrap();
        assert!(
            first.starts_with("POST /v1/chat/completions HTTP/1.1"),
            "{first}"
        );
        assert_eq!(sent["model"], "small");

        // Nothing listening is a clear error.
        let closed = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = closed.local_addr().unwrap().port();
        drop(closed);
        let mut gone = LoopbackHttp::new(&format!("http://127.0.0.1:{port}/x")).unwrap();
        let error = gone.post("{}").unwrap_err();
        assert!(
            error.to_string().contains("no local model answered"),
            "{error}"
        );
    }

    #[test]
    fn each_request_the_prompt_shows_is_one_the_session_reads() {
        let mut shown = 0;
        let mut rest = super::SYSTEM_PROMPT;
        while let Some(at) = rest.find("{\"op\"") {
            let tail = &rest[at..];
            let end = tail.find('}').unwrap() + 1;
            let request: Result<cutlass_session::Request, _> = serde_json::from_str(&tail[..end]);
            assert!(request.is_ok(), "{}: {request:?}", &tail[..end]);
            shown += 1;
            rest = &tail[end..];
        }
        assert!(shown >= 4, "{shown}");
    }

    #[test]
    fn the_prompt_names_enabled_servers_and_carries_notes_only_with_memory_on() {
        let mut settings = cutlass_settings::Settings::default();
        settings.mcp_servers.push(cutlass_settings::McpServer {
            name: "files".to_owned(),
            enabled: true,
            ..Default::default()
        });
        settings.mcp_servers.push(cutlass_settings::McpServer {
            name: "off".to_owned(),
            ..Default::default()
        });
        let context = super::prompt_context(&settings, "Titles in Montserrat.");
        assert!(context.contains(": files."), "{context}");
        assert!(!context.contains("off"), "{context}");
        assert!(context.contains("Titles in Montserrat."), "{context}");
        settings.memory = false;
        let context = super::prompt_context(&settings, "Titles in Montserrat.");
        assert!(!context.contains("Montserrat"), "{context}");
        let model = ChatModel::new(LoopbackHttp::new("http://127.0.0.1:1/").unwrap(), "m")
            .with_context("Extra.");
        let body = model.body(&CompletionRequest {
            messages: Vec::new(),
            tools: Vec::new(),
        });
        let system = body["messages"][0]["content"].as_str().unwrap();
        assert!(system.starts_with(super::SYSTEM_PROMPT), "{system}");
        assert!(system.ends_with("\n\nExtra."), "{system}");
    }

    #[test]
    fn the_runner_lists_its_models() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut first = String::new();
            reader.read_line(&mut first).unwrap();
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line.trim().is_empty() {
                    break;
                }
            }
            let answer =
                r#"{"object":"list","data":[{"id":"qwen3.8:27b"},{"id":"gemma4:latest"}]}"#;
            let mut stream = stream;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{answer}",
                answer.len()
            )
            .unwrap();
            first
        });
        let transport =
            LoopbackHttp::new(&format!("http://127.0.0.1:{port}/v1/chat/completions")).unwrap();
        let models = transport.models().unwrap();
        assert_eq!(models, ["gemma4:latest", "qwen3.8:27b"]);
        let first = server.join().unwrap();
        assert!(first.starts_with("GET /v1/models HTTP/1.1"), "{first}");
    }
}
