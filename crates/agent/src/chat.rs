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

use crate::mcp_server::{input_schema, registry_name};
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
}

/// What the model is told before the user's first words.
pub const SYSTEM_PROMPT: &str = "You edit a video project in Cutlass with tools. \
Read with draft_query first: every draft_apply sends the revision it read. \
Times are microseconds. draft_apply runs its requests as one undo step. \
Say briefly what you changed.";

impl<T: ChatTransport> ChatModel<T> {
    /// Talks to `model` through `transport`.
    #[must_use]
    pub fn new(transport: T, model: &str) -> Self {
        Self {
            transport,
            model: model.to_owned(),
            system: SYSTEM_PROMPT.to_owned(),
        }
    }

    /// The transport, to look at in tests.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// The request body for `request`.
    #[must_use]
    pub fn body(&self, request: &CompletionRequest) -> Value {
        let mut messages = vec![json!({ "role": "system", "content": self.system })];
        for message in &request.messages {
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
                } => json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name.replace('.', "_"),
                    "content": if *is_error { format!("error: {content}") } else { content.clone() },
                }),
            });
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
        let reply = self.transport.post(&body)?;
        parse_reply(&reply)
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
    if tool == "draft.apply" {
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
    if tool == "draft.apply" {
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

impl ChatTransport for LoopbackHttp {
    fn post(&mut self, body: &str) -> Result<String, ModelError> {
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
            "POST {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.path,
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
}
