//! The MCP servers the user added in Settings → Assistant, reached for
//! `mcp.call`: a local program spoken to on its stdin and stdout, or a
//! remote URL spoken to with HTTP posts.
//!
//! A server starts on its first call and stays up for the next ones. It is
//! stopped when its settings change, when it fails, and when Cutlass
//! quits. Whether a server may be called at all is the agent's check
//! (`McpClient`), against the enabled switch the user set; turning a
//! remote one on asked the user then.

use std::collections::{BTreeMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::Duration;

use cutlass_agent::{McpConnection, McpError, McpTransport, McpWire};
use cutlass_settings::McpServer;

/// How long a server may take to answer one message.
const ANSWER_WITHIN: Duration = Duration::from_secs(120);

/// The servers started so far, by name, each with the settings it was
/// started with.
#[derive(Default)]
pub struct McpServers {
    known: Vec<McpServer>,
    live: BTreeMap<String, (McpServer, McpConnection<Wire>)>,
}

impl McpServers {
    /// Take the servers in `servers` as the ones that may be started. A
    /// running server whose settings changed or went away is stopped.
    pub fn configure(&mut self, servers: &[McpServer]) {
        self.known = servers.to_vec();
        self.live.retain(|name, (started, _)| {
            servers
                .iter()
                .any(|server| &server.name == name && same_place(server, started))
        });
    }

    /// The names of the servers running now.
    #[must_use]
    pub fn running(&self) -> Vec<String> {
        self.live.keys().cloned().collect()
    }
}

/// Whether `a` and `b` reach the same server the same way.
fn same_place(a: &McpServer, b: &McpServer) -> bool {
    a.kind == b.kind && a.command == b.command && a.args == b.args && a.url == b.url
}

impl McpTransport for McpServers {
    fn call(&mut self, server_name: &str, tool: &str, arguments: &str) -> Result<String, McpError> {
        if !self.live.contains_key(server_name) {
            let server = self
                .known
                .iter()
                .find(|server| server.name == server_name)
                .cloned()
                .ok_or_else(|| McpError::Unknown {
                    name: server_name.to_owned(),
                })?;
            let wire = Wire::start(&server)?;
            cutlass_logger::info!("mcp server {server_name} started ({})", server.kind);
            self.live.insert(
                server_name.to_owned(),
                (server, McpConnection::new(wire, server_name)),
            );
        }
        let Some((_, connection)) = self.live.get_mut(server_name) else {
            return Err(McpError::Unknown {
                name: server_name.to_owned(),
            });
        };
        let answer = connection.call(tool, arguments);
        if let Err(error) = &answer
            && connection.wire_mut().broken()
        {
            cutlass_logger::warn!("mcp server {server_name} stopped: {error}");
            self.live.remove(server_name);
        }
        answer
    }
}

/// One server's bytes.
pub enum Wire {
    Process(Process),
    Remote(Remote),
}

impl Wire {
    fn start(server: &McpServer) -> Result<Self, McpError> {
        if server.kind == McpServer::REMOTE {
            return Remote::new(&server.url).map(Wire::Remote);
        }
        Process::start(&server.command, &server.args).map(Wire::Process)
    }

    /// Whether the connection can no longer be used.
    fn broken(&mut self) -> bool {
        match self {
            Self::Process(process) => process.broken(),
            Self::Remote(remote) => remote.broken,
        }
    }
}

impl McpWire for Wire {
    fn send(&mut self, message: &str) -> Result<(), McpError> {
        match self {
            Self::Process(process) => process.send(message),
            Self::Remote(remote) => remote.send(message),
        }
    }

    fn receive(&mut self) -> Result<String, McpError> {
        match self {
            Self::Process(process) => process.receive(),
            Self::Remote(remote) => remote.receive(),
        }
    }
}

/// A server program, one JSON message a line each way.
pub struct Process {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
    name: String,
    closed: bool,
}

impl Process {
    fn start(command: &str, args: &[String]) -> Result<Self, McpError> {
        if command.trim().is_empty() {
            return Err(McpError::Transport(
                "the server has no program to start; name one in Settings → Assistant".to_owned(),
            ));
        }
        let mut child = Command::new(command.trim())
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| McpError::Transport(format!("{command} did not start: {error}")))?;
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            let _ = child.kill();
            return Err(McpError::Transport(format!(
                "{command} started without its pipes"
            )));
        };
        let (sender, lines) = mpsc::channel();
        let reader = std::thread::Builder::new()
            .name("mcp-server".to_owned())
            .spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let Ok(line) = line else { break };
                    if sender.send(line).is_err() {
                        break;
                    }
                }
            });
        if let Err(error) = reader {
            let _ = child.kill();
            return Err(McpError::Transport(format!(
                "could not read {command}: {error}"
            )));
        }
        Ok(Self {
            child,
            stdin,
            lines,
            name: command.to_owned(),
            closed: false,
        })
    }

    fn send(&mut self, message: &str) -> Result<(), McpError> {
        let written = self
            .stdin
            .write_all(message.as_bytes())
            .and_then(|()| self.stdin.write_all(b"\n"))
            .and_then(|()| self.stdin.flush());
        written.map_err(|error| {
            self.closed = true;
            McpError::Transport(format!("{} stopped listening: {error}", self.name))
        })
    }

    fn receive(&mut self) -> Result<String, McpError> {
        match self.lines.recv_timeout(ANSWER_WITHIN) {
            Ok(line) => Ok(line),
            Err(RecvTimeoutError::Timeout) => {
                self.closed = true;
                Err(McpError::Transport(format!(
                    "{} did not answer within {} seconds",
                    self.name,
                    ANSWER_WITHIN.as_secs()
                )))
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.closed = true;
                Err(McpError::Transport(format!("{} closed", self.name)))
            }
        }
    }

    fn broken(&mut self) -> bool {
        self.closed || !matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A server at a URL: each message is a POST, and the answers come back in
/// its body as JSON or as server-sent events.
pub struct Remote {
    http: cutlass_download::Http,
    url: String,
    session: Option<String>,
    answers: VecDeque<String>,
    broken: bool,
}

impl Remote {
    fn new(url: &str) -> Result<Self, McpError> {
        let url = url.trim();
        // Tool arguments go to this address: over https, or in the clear
        // only to this machine.
        if !(url.starts_with("https://") || on_this_machine(url)) {
            return Err(McpError::Transport(format!(
                "the server's address must start with https:// (or http:// for one on this \
                 machine), not {url}"
            )));
        }
        Ok(Self {
            http: cutlass_download::Http::patient(ANSWER_WITHIN),
            url: url.to_owned(),
            session: None,
            answers: VecDeque::new(),
            broken: false,
        })
    }

    fn send(&mut self, message: &str) -> Result<(), McpError> {
        let mut headers = vec![
            ("Content-Type", "application/json"),
            ("Accept", "application/json, text/event-stream"),
            ("MCP-Protocol-Version", cutlass_agent::MCP_PROTOCOL_VERSION),
        ];
        let session = self.session.clone();
        if let Some(session) = &session {
            headers.push(("Mcp-Session-Id", session));
        }
        let posted = self
            .http
            .post(&self.url, &headers, message)
            .map_err(|error| {
                self.broken = true;
                McpError::Transport(error)
            })?;
        if let Some(session) = posted.header("mcp-session-id") {
            self.session = Some(session.to_owned());
        }
        if !(200..300).contains(&posted.status) {
            if posted.status == 404 && self.session.is_some() {
                // The server forgot the session; start over next time.
                self.broken = true;
            }
            return Err(McpError::Transport(format!(
                "{} answered {}: {}",
                self.url,
                posted.status,
                posted.body.chars().take(300).collect::<String>()
            )));
        }
        let events = posted
            .header("content-type")
            .is_some_and(|kind| kind.starts_with("text/event-stream"));
        self.answers.extend(messages_in(&posted.body, events));
        Ok(())
    }

    fn receive(&mut self) -> Result<String, McpError> {
        self.answers
            .pop_front()
            .ok_or_else(|| McpError::Transport(format!("{} sent no answer", self.url)))
    }
}

/// Whether `url` is plain http to a loopback address.
fn on_this_machine(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("http://") else {
        return false;
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let host = match authority.strip_prefix('[') {
        // An IPv6 address in brackets, with or without a port.
        Some(bracketed) => bracketed.split(']').next().unwrap_or_default(),
        None => authority
            .rsplit_once(':')
            .map_or(authority, |(host, _)| host),
    };
    host == "localhost"
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

/// The JSON-RPC messages in a reply body: one object, a batch, or the
/// `data:` lines of server-sent events.
fn messages_in(body: &str, events: bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut push = |text: &str| match serde_json::from_str::<serde_json::Value>(text) {
        Ok(serde_json::Value::Array(batch)) => {
            out.extend(batch.iter().map(serde_json::Value::to_string));
        }
        Ok(message) => out.push(message.to_string()),
        Err(_) => {}
    };
    if events {
        let mut data = String::new();
        for line in body.lines().chain(std::iter::once("")) {
            if let Some(rest) = line.strip_prefix("data:") {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(rest.trim_start());
            } else if line.trim().is_empty() && !data.is_empty() {
                push(&data);
                data.clear();
            }
        }
    } else if !body.trim().is_empty() {
        push(body);
    }
    out
}

#[cfg(test)]
mod tests {
    use cutlass_agent::{McpClient, McpTransport};
    use cutlass_settings::{McpServer, Settings};

    use super::{McpServers, messages_in};

    /// A stdio MCP server in a few lines of shell: it answers initialize,
    /// tools/list and one tool by the id it was sent.
    const SERVER: &str = r#"
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  case "$line" in
    *'"initialize"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-06-18","capabilities":{}}}\n' "$id" ;;
    *'"tools/list"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"Say it back","inputSchema":{"type":"object"}}]}}\n' "$id" ;;
    *'"tools/call"'*) printf 'a log line\n{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"echo: ok"}]}}\n' "$id" ;;
  esac
done
"#;

    fn settings(command: &str) -> Settings {
        let mut settings = Settings::default();
        settings.mcp_servers.push(McpServer {
            name: "local".to_owned(),
            enabled: true,
            command: command.to_owned(),
            args: vec!["-c".to_owned(), SERVER.to_owned()],
            ..McpServer::default()
        });
        settings
    }

    #[cfg(unix)]
    #[test]
    fn a_local_server_is_started_once_listed_and_called() {
        let settings = settings("/bin/sh");
        let mut servers = McpServers::default();
        servers.configure(&settings.mcp_servers);
        let mut client = McpClient::new(servers);
        let tools = client.call(&settings, "local", "tools/list", "").unwrap();
        assert!(tools.starts_with("echo: Say it back"), "{tools}");
        let said = client
            .call(&settings, "local", "echo", r#"{"text":"hi"}"#)
            .unwrap();
        assert_eq!(said, "echo: ok");
        assert_eq!(client.transport().running(), ["local"]);
        // Changing how it starts stops it.
        let mut moved = settings.clone();
        moved.mcp_servers[0].args.push("again".to_owned());
        client.transport_mut().configure(&moved.mcp_servers);
        assert!(client.transport().running().is_empty());
    }

    #[test]
    fn a_program_that_is_not_there_is_an_error() {
        let settings = settings("/no/such/program");
        let mut servers = McpServers::default();
        servers.configure(&settings.mcp_servers);
        let error = servers.call("local", "tools/list", "").unwrap_err();
        assert!(error.to_string().contains("did not start"), "{error}");
        assert!(servers.call("other", "tools/list", "").is_err());
    }

    #[test]
    fn a_remote_server_is_reached_over_https_or_on_this_machine_only() {
        let mut settings = Settings::default();
        for (name, url) in [
            ("clear", "http://example.com/mcp"),
            ("spoof", "http://localhost.example.com/mcp"),
            ("file", "ftp://example.com/mcp"),
        ] {
            settings.mcp_servers.push(McpServer {
                name: name.to_owned(),
                enabled: true,
                kind: McpServer::REMOTE.to_owned(),
                url: url.to_owned(),
                ..McpServer::default()
            });
        }
        let mut servers = McpServers::default();
        servers.configure(&settings.mcp_servers);
        for name in ["clear", "spoof", "file"] {
            let error = servers.call(name, "tools/list", "").unwrap_err();
            assert!(error.to_string().contains("https://"), "{name}: {error}");
        }
        for (url, local) in [
            ("http://127.0.0.1:8080/mcp", true),
            ("http://localhost/mcp", true),
            ("http://[::1]:9000/mcp", true),
            ("http://10.0.0.5/mcp", false),
            ("https://127.0.0.1/mcp", false),
        ] {
            assert_eq!(super::on_this_machine(url), local, "{url}");
        }
    }

    #[test]
    fn answers_come_as_json_a_batch_or_events() {
        assert_eq!(messages_in(r#"{"id":1}"#, false), [r#"{"id":1}"#]);
        assert_eq!(messages_in(r#"[{"id":1},{"id":2}]"#, false).len(), 2);
        let events = "event: message\ndata: {\"id\":1}\n\n: comment\ndata: {\"id\":\ndata: 2}\n";
        assert_eq!(messages_in(events, true), [r#"{"id":1}"#, r#"{"id":2}"#]);
        assert!(messages_in("", false).is_empty());
    }
}
