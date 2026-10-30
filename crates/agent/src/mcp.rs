//! An MCP client that calls only servers the user has enabled.
//!
//! The argument string is opaque. This client does not open a socket and does
//! not spawn a server process.

use std::fmt;

use cutlass_settings::Settings;

/// Sends one tool call to a named server.
pub trait McpTransport {
    /// Delivers `tool` and opaque `arguments` to `server_name`.
    fn call(&mut self, server_name: &str, tool: &str, arguments: &str) -> Result<String, McpError>;
}

/// Why an MCP call did not return a tool result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McpError {
    /// No server in settings uses this name.
    Unknown {
        /// The name that was asked for.
        name: String,
    },
    /// The server is stored and the user has not enabled it.
    Disabled {
        /// The server name.
        name: String,
        /// The stored kind, such as `stdio` or `remote`.
        kind: String,
    },
    /// The injected transport failed.
    Transport(String),
}

impl fmt::Display for McpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown { name } => write!(formatter, "unknown MCP server {name}"),
            Self::Disabled { name, kind } => {
                write!(formatter, "MCP server {name} ({kind}) is disabled")
            }
            Self::Transport(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for McpError {}

/// Calls tools on MCP servers listed in settings.
#[derive(Debug)]
pub struct McpClient<T> {
    transport: T,
}

impl<T> McpClient<T> {
    /// Wraps the host transport.
    #[must_use]
    pub const fn new(transport: T) -> Self {
        Self { transport }
    }

    /// The transport the host injected.
    #[must_use]
    pub const fn transport(&self) -> &T {
        &self.transport
    }

    /// The transport the host injected.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }
}

impl<T: McpTransport> McpClient<T> {
    /// Calls `tool` when `server_name` is an enabled server.
    pub fn call(
        &mut self,
        settings: &Settings,
        server_name: &str,
        tool: &str,
        arguments: &str,
    ) -> Result<String, McpError> {
        let Some(server) = settings
            .mcp_servers
            .iter()
            .find(|server| server.name == server_name)
        else {
            return Err(McpError::Unknown {
                name: server_name.to_owned(),
            });
        };
        if !server.enabled {
            return Err(McpError::Disabled {
                name: server.name.clone(),
                kind: server.kind.clone(),
            });
        }
        self.transport.call(server_name, tool, arguments)
    }
}

#[cfg(test)]
mod tests {
    use cutlass_settings::{McpServer, Settings};

    use super::{McpClient, McpError, McpTransport};

    #[derive(Debug, Default)]
    struct Count {
        calls: u32,
        seen: Vec<(String, String, String)>,
    }

    impl McpTransport for Count {
        fn call(
            &mut self,
            server_name: &str,
            tool: &str,
            arguments: &str,
        ) -> Result<String, McpError> {
            self.calls += 1;
            self.seen.push((
                server_name.to_owned(),
                tool.to_owned(),
                arguments.to_owned(),
            ));
            Ok("ok".to_owned())
        }
    }

    fn server(name: &str, enabled: bool, kind: &str, command: &str, url: &str) -> McpServer {
        McpServer {
            name: name.to_owned(),
            enabled,
            kind: kind.to_owned(),
            command: command.to_owned(),
            url: url.to_owned(),
            ..McpServer::default()
        }
    }

    fn settings(servers: Vec<McpServer>) -> Settings {
        Settings {
            mcp_servers: servers,
            ..Settings::default()
        }
    }

    #[test]
    fn disabled_stdio_is_not_called() {
        let mut client = McpClient::new(Count::default());
        let settings = settings(vec![server("local", false, McpServer::STDIO, "tool", "")]);
        let error = client
            .call(&settings, "local", "ping", "hello")
            .unwrap_err();
        assert_eq!(client.transport().calls, 0);
        assert!(client.transport().seen.is_empty());
        assert_eq!(
            error,
            McpError::Disabled {
                name: "local".to_owned(),
                kind: McpServer::STDIO.to_owned(),
            }
        );
        let text = error.to_string();
        assert!(text.contains("disabled"));
        assert!(text.contains("stdio"));
        assert!(text.contains("local"));
    }

    #[test]
    fn enabled_stdio_reaches_the_transport() {
        let mut client = McpClient::new(Count::default());
        let settings = settings(vec![server("local", true, McpServer::STDIO, "tool", "")]);
        let text = client.call(&settings, "local", "ping", "hello").unwrap();
        assert_eq!(text, "ok");
        assert_eq!(client.transport().calls, 1);
        assert_eq!(
            client.transport().seen,
            vec![("local".to_owned(), "ping".to_owned(), "hello".to_owned())]
        );
    }

    #[test]
    fn disabled_remote_is_not_called() {
        let mut client = McpClient::new(Count::default());
        let settings = settings(vec![server(
            "far",
            false,
            McpServer::REMOTE,
            "",
            "http://127.0.0.1:9/mcp",
        )]);
        let error = client.call(&settings, "far", "ping", "hello").unwrap_err();
        assert_eq!(client.transport().calls, 0);
        assert!(client.transport().seen.is_empty());
        assert_eq!(
            error,
            McpError::Disabled {
                name: "far".to_owned(),
                kind: McpServer::REMOTE.to_owned(),
            }
        );
        let text = error.to_string();
        assert!(text.contains("disabled"));
        assert!(text.contains("remote"));
        assert!(text.contains("far"));
    }

    #[test]
    fn enabled_remote_uses_the_transport_and_not_a_url() {
        let url = "http://127.0.0.1:9/mcp";
        let mut client = McpClient::new(Count::default());
        let settings = settings(vec![server("far", true, McpServer::REMOTE, "", url)]);
        let text = client.call(&settings, "far", "ping", "hello").unwrap();
        assert_eq!(text, "ok");
        assert_eq!(client.transport().calls, 1);
        let seen = &client.transport().seen[0];
        assert_eq!(
            seen,
            &("far".to_owned(), "ping".to_owned(), "hello".to_owned())
        );
        assert!(!seen.0.contains(url));
        assert!(!seen.1.contains(url));
        assert!(!seen.2.contains(url));
    }

    #[test]
    fn unknown_server_is_an_error() {
        let mut client = McpClient::new(Count::default());
        let settings = settings(vec![server("local", true, McpServer::STDIO, "tool", "")]);
        let error = client
            .call(&settings, "missing", "ping", "hello")
            .unwrap_err();
        assert_eq!(client.transport().calls, 0);
        assert_eq!(
            error,
            McpError::Unknown {
                name: "missing".to_owned(),
            }
        );
        let text = error.to_string();
        assert!(text.contains("missing"));
        assert!(text.contains("unknown"));
    }

    #[test]
    fn arguments_are_opaque() {
        let mut client = McpClient::new(Count::default());
        let settings = settings(vec![server("local", true, McpServer::STDIO, "tool", "")]);
        let arguments = "not a protocol";
        let text = client.call(&settings, "local", "echo", arguments).unwrap();
        assert_eq!(text, "ok");
        assert_eq!(client.transport().calls, 1);
        assert_eq!(client.transport().seen[0].2, arguments);
    }
}
