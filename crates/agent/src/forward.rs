//! Forwarding tool calls to the app that holds a project.
//!
//! The app that holds a project listens on a loopback socket and names it,
//! with a token, in the project's `.locked`. A second host, such as
//! `cutlass mcp`, sends its tool calls there instead of opening the project
//! itself, so the window applies each edit and both see one project.
//!
//! The wire is a line of JSON each way: `{"token","id","name","arguments"}`
//! in, `{"content","is_error"}` out. A call with the wrong token is answered
//! with an error and nothing runs.

use std::hash::{BuildHasher, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde_json::{Value, json};

use crate::{ToolCall, ToolOutcome, ToolSink};

/// How long a forwarded call waits for the app before it gives up.
const ANSWER_WAIT: Duration = Duration::from_secs(600);

/// Wakes the app's loop when a call arrives.
pub type Wake = Arc<dyn Fn() + Send + Sync>;

/// A call that came in on the socket, waiting for the app to run it.
pub struct Forwarded {
    pub call: ToolCall,
    reply: Sender<ToolOutcome>,
}

impl Forwarded {
    /// Send the outcome back to the caller.
    pub fn answer(self, outcome: ToolOutcome) {
        let _ = self.reply.send(outcome);
    }
}

/// The loopback socket an app that holds a project listens on.
pub struct ProjectSocket {
    address: SocketAddr,
    token: String,
    calls: Mutex<Receiver<Forwarded>>,
    stop: Arc<AtomicBool>,
}

impl ProjectSocket {
    /// Listen on a free loopback port. `wake` is called, from another
    /// thread, whenever a call is waiting.
    pub fn open(wake: Wake) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let address = listener.local_addr()?;
        let token = fresh_token();
        let (sender, calls) = channel();
        let stop = Arc::new(AtomicBool::new(false));
        let watch = Arc::clone(&stop);
        let expected = token.clone();
        thread::Builder::new()
            .name("project socket".to_owned())
            .spawn(move || {
                for stream in listener.incoming() {
                    if watch.load(Ordering::Acquire) {
                        break;
                    }
                    let Ok(stream) = stream else {
                        continue;
                    };
                    let sender = sender.clone();
                    let wake = Arc::clone(&wake);
                    let token = expected.clone();
                    let _ = thread::Builder::new()
                        .name("project caller".to_owned())
                        .spawn(move || serve_caller(stream, &token, &sender, &wake));
                }
            })?;
        Ok(Self {
            address,
            token,
            calls: Mutex::new(calls),
            stop,
        })
    }

    /// Where it listens, as `.locked` names it.
    #[must_use]
    pub fn address(&self) -> String {
        self.address.to_string()
    }

    /// The token a caller must send.
    #[must_use]
    pub fn token(&self) -> &str {
        &self.token
    }

    /// The calls waiting now, oldest first.
    pub fn take(&self) -> Vec<Forwarded> {
        let Ok(calls) = self.calls.lock() else {
            return Vec::new();
        };
        calls.try_iter().collect()
    }
}

impl Drop for ProjectSocket {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        // Wake the accept loop so it sees the stop.
        let _ = TcpStream::connect_timeout(&self.address, Duration::from_millis(200));
    }
}

fn serve_caller(stream: TcpStream, token: &str, calls: &Sender<Forwarded>, wake: &Wake) {
    let Ok(mut writer) = stream.try_clone() else {
        return;
    };
    for line in BufReader::new(stream).lines() {
        let Ok(line) = line else {
            return;
        };
        if line.trim().is_empty() {
            continue;
        }
        let outcome = match read_call(&line, token) {
            Err(refused) => refused,
            Ok(call) => {
                let (reply, answer) = channel();
                if calls.send(Forwarded { call, reply }).is_err() {
                    return;
                }
                wake();
                answer
                    .recv_timeout(ANSWER_WAIT)
                    .unwrap_or_else(|_| ToolOutcome {
                        content: "the Cutlass window did not answer".to_owned(),
                        is_error: true,
                    })
            }
        };
        let line = json!({"content": outcome.content, "is_error": outcome.is_error});
        if writeln!(writer, "{line}")
            .and_then(|()| writer.flush())
            .is_err()
        {
            return;
        }
    }
}

fn read_call(line: &str, token: &str) -> Result<ToolCall, ToolOutcome> {
    let refuse = |content: &str| ToolOutcome {
        content: content.to_owned(),
        is_error: true,
    };
    let message: Value =
        serde_json::from_str(line).map_err(|_| refuse("a forwarded call is not JSON"))?;
    if message.get("token").and_then(Value::as_str) != Some(token) {
        return Err(refuse("the call's token does not match this project"));
    }
    let text = |key: &str| {
        message
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    let name = text("name");
    if name.is_empty() {
        return Err(refuse("a forwarded call names no tool"));
    }
    Ok(ToolCall {
        id: text("id"),
        name,
        arguments: text("arguments"),
    })
}

/// A random token, from the process's random hash keys and the clock.
fn fresh_token() -> String {
    let half = || {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos());
        hasher.write_u128(now);
        hasher.write_u32(std::process::id());
        hasher.finish()
    };
    format!("{:016x}{:016x}", half(), half())
}

/// Runs each tool call in the app that holds the project.
pub struct Forward {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    token: String,
}

impl Forward {
    /// Connect to the socket at `address`, which must be on this machine.
    pub fn connect(address: &str, token: &str) -> Result<Self, String> {
        let address: SocketAddr = address
            .parse()
            .map_err(|_| format!("{address} is not a socket address"))?;
        if !address.ip().is_loopback() {
            return Err(format!("{address} is not on this machine"));
        }
        let writer = TcpStream::connect_timeout(&address, Duration::from_secs(5))
            .map_err(|error| format!("the Cutlass window did not answer: {error}"))?;
        let reader = BufReader::new(writer.try_clone().map_err(|error| error.to_string())?);
        Ok(Self {
            reader,
            writer,
            token: token.to_owned(),
        })
    }

    fn send(&mut self, call: &ToolCall) -> Result<ToolOutcome, String> {
        let line = json!({
            "token": self.token,
            "id": call.id,
            "name": call.name,
            "arguments": call.arguments,
        });
        writeln!(self.writer, "{line}")
            .and_then(|()| self.writer.flush())
            .map_err(|error| error.to_string())?;
        let mut answer = String::new();
        self.reader
            .read_line(&mut answer)
            .map_err(|error| error.to_string())?;
        let answer: Value = serde_json::from_str(&answer)
            .map_err(|_| "the Cutlass window closed the connection".to_owned())?;
        Ok(ToolOutcome {
            content: answer
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            is_error: answer
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        })
    }
}

impl ToolSink for Forward {
    fn run_tool(&mut self, call: &ToolCall) -> ToolOutcome {
        self.send(call).unwrap_or_else(|error| ToolOutcome {
            content: format!("the call did not reach the Cutlass window: {error}"),
            is_error: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_call_reaches_the_owner_and_its_answer_comes_back() {
        let woken = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&woken);
        let socket =
            ProjectSocket::open(Arc::new(move || flag.store(true, Ordering::Release))).unwrap();
        assert!(socket.address().starts_with("127.0.0.1:"));
        assert_eq!(socket.token().len(), 32);
        let address = socket.address();
        let token = socket.token().to_owned();
        let caller = thread::spawn(move || {
            let mut forward = Forward::connect(&address, &token).unwrap();
            let call = ToolCall {
                id: "1".to_owned(),
                name: "draft.query".to_owned(),
                arguments: "project".to_owned(),
            };
            let first = forward.run_tool(&call);
            let second = forward.run_tool(&call);
            (first, second)
        });
        let mut answered = 0;
        while answered < 2 {
            for waiting in socket.take() {
                assert_eq!(waiting.call.name, "draft.query");
                assert_eq!(waiting.call.arguments, "project");
                answered += 1;
                waiting.answer(ToolOutcome {
                    content: format!("answer {answered}"),
                    is_error: false,
                });
            }
            thread::sleep(Duration::from_millis(2));
        }
        let (first, second) = caller.join().unwrap();
        assert_eq!(first.content, "answer 1");
        assert_eq!(second.content, "answer 2");
        assert!(!first.is_error);
        assert!(woken.load(Ordering::Acquire));
    }

    #[test]
    fn a_wrong_token_runs_nothing() {
        let socket = ProjectSocket::open(Arc::new(|| {})).unwrap();
        let mut forward = Forward::connect(&socket.address(), "not the token").unwrap();
        let outcome = forward.run_tool(&ToolCall {
            id: String::new(),
            name: "draft.apply".to_owned(),
            arguments: "{}".to_owned(),
        });
        assert!(outcome.is_error);
        assert!(outcome.content.contains("token"));
        assert!(socket.take().is_empty());
        assert!(
            Forward::connect("10.0.0.1:9", "t").is_err(),
            "only this machine"
        );
    }
}
