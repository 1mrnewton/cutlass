//! Forwarding tool calls to the app that holds a project.
//!
//! The app that holds a project listens on a loopback socket and names it,
//! with a token, in the project's `.locked`. A second host, such as
//! `cutlass mcp`, sends its tool calls there instead of opening the project
//! itself, so the window applies each edit and both see one project.
//!
//! The wire is a line of JSON each way: `{"token","id","name","arguments"}`
//! in, `{"content","is_error"}` out, with a `picture` when the tool made
//! one. A call with the wrong token is answered with an error and nothing
//! runs.
//!
//! Any program on the machine can connect, and the token is in the call
//! it sends, so what is read before the token is known is bounded: a call
//! is at most [`CALL_LIMIT`] bytes, and at most [`CALLER_LIMIT`] callers
//! are connected at once.
//!
//! A call waits in a queue until the app gets to it. A caller that stops
//! waiting meanwhile, by hanging up or by running out of time, was told
//! the call failed or never heard of it, so the call is dropped from the
//! queue and does not run later behind the caller's back.

use std::hash::{BuildHasher, Hasher};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::{ToolCall, ToolOutcome, ToolSink};

/// How long a forwarded call waits for the app before it gives up.
const ANSWER_WAIT: Duration = Duration::from_secs(600);

/// The longest call a caller may send, in bytes. A combo of thousands of
/// requests is far below it; a caller with no token cannot make the app
/// hold more than this for it.
const CALL_LIMIT: usize = 16 * 1024 * 1024;

/// The most callers connected at once. Each is a `cutlass mcp` started by
/// an MCP client, and a handful is many.
const CALLER_LIMIT: usize = 16;

/// How often a waiting call looks whether its caller is still there.
const CALLER_LOOK: Duration = Duration::from_millis(100);

/// The call a window sends the holder of a project it wants to open: a
/// host with no window saves, lets the project go and forwards to the
/// window from then on. A window never lets go; to it this is no tool.
pub const YIELD: &str = "host.yield";

/// Wakes the app's loop when a call arrives.
pub type Wake = Arc<dyn Fn() + Send + Sync>;

/// A call that came in on the socket, waiting for the app to run it.
pub struct Forwarded {
    pub call: ToolCall,
    reply: Sender<ToolOutcome>,
    /// Set once the caller stopped waiting for the answer.
    left: Arc<AtomicBool>,
}

impl Forwarded {
    /// Send the outcome back to the caller.
    pub fn answer(self, outcome: ToolOutcome) {
        let _ = self.reply.send(outcome);
    }

    /// Whether the caller stopped waiting: it hung up, or it waited as
    /// long as a call may take and was told the call failed. Such a call
    /// must not run: its caller takes it not to have. An app that works
    /// through several calls asks this before each one, as a caller may
    /// leave while the calls ahead of its own run.
    #[must_use]
    pub fn caller_left(&self) -> bool {
        self.left.load(Ordering::Acquire)
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
        Self::open_waiting(wake, ANSWER_WAIT)
    }

    /// [`Self::open`], with each call waiting `answer_wait` for the app.
    fn open_waiting(wake: Wake, answer_wait: Duration) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let address = listener.local_addr()?;
        let token = fresh_token();
        let (sender, calls) = channel();
        let stop = Arc::new(AtomicBool::new(false));
        let watch = Arc::clone(&stop);
        let expected = token.clone();
        let seated = Arc::new(AtomicUsize::new(0));
        thread::Builder::new()
            .name("project socket".to_owned())
            .spawn(move || {
                for stream in listener.incoming() {
                    if watch.load(Ordering::Acquire) {
                        break;
                    }
                    let Ok(mut stream) = stream else {
                        continue;
                    };
                    // A seat is held until the caller's thread ends.
                    let seat = Seat(Arc::clone(&seated));
                    if seated.fetch_add(1, Ordering::AcqRel) >= CALLER_LIMIT {
                        let _ = answer_line(&mut stream, &refused(TOO_MANY_CALLERS));
                        continue;
                    }
                    let sender = sender.clone();
                    let wake = Arc::clone(&wake);
                    let token = expected.clone();
                    let _ = thread::Builder::new()
                        .name("project caller".to_owned())
                        .spawn(move || {
                            let _seat = seat;
                            serve_caller(stream, &token, &sender, &wake, answer_wait);
                        });
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

    /// The calls waiting now, oldest first, without the ones whose caller
    /// stopped waiting.
    pub fn take(&self) -> Vec<Forwarded> {
        let Ok(calls) = self.calls.lock() else {
            return Vec::new();
        };
        calls
            .try_iter()
            .filter(|call| !call.caller_left())
            .collect()
    }
}

impl Drop for ProjectSocket {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        // Wake the accept loop so it sees the stop.
        let _ = TcpStream::connect_timeout(&self.address, Duration::from_millis(200));
    }
}

fn serve_caller(
    stream: TcpStream,
    token: &str,
    calls: &Sender<Forwarded>,
    wake: &Wake,
    answer_wait: Duration,
) {
    let Ok(mut writer) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(stream);
    loop {
        let line = match read_line(&mut reader, CALL_LIMIT) {
            Line::Text(line) => line,
            Line::End => return,
            Line::TooLong => {
                // The rest of it is not read: the caller is let go.
                let _ = answer_line(&mut writer, &refused(CALL_TOO_LONG));
                return;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let outcome = match read_call(&line, token) {
            Err(refused) => refused,
            Ok(call) => {
                let (reply, answer) = channel();
                let left = Arc::new(AtomicBool::new(false));
                let waiting = Forwarded {
                    call,
                    reply,
                    left: Arc::clone(&left),
                };
                if calls.send(waiting).is_err() {
                    return;
                }
                wake();
                match await_answer(&writer, &answer, answer_wait) {
                    Waited::Answered(outcome) => outcome,
                    Waited::CallerLeft => {
                        left.store(true, Ordering::Release);
                        return;
                    }
                    Waited::TooLong => {
                        // The caller is told it failed, so it must not
                        // run once the app gets to it.
                        left.store(true, Ordering::Release);
                        unanswered()
                    }
                    Waited::Dropped => unanswered(),
                }
            }
        };
        if answer_line(&mut writer, &outcome).is_err() {
            return;
        }
    }
}

/// What a caller past [`CALLER_LIMIT`] is told.
const TOO_MANY_CALLERS: &str =
    "too many callers are connected to the app that holds this project; close one and call again";

/// What a caller whose call is past [`CALL_LIMIT`] is told.
const CALL_TOO_LONG: &str = "a forwarded call is too long to read";

/// One caller's place among the [`CALLER_LIMIT`], given back when dropped.
struct Seat(Arc<AtomicUsize>);

impl Drop for Seat {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A refusal that ran nothing.
fn refused(content: &str) -> ToolOutcome {
    ToolOutcome {
        content: content.to_owned(),
        is_error: true,
        picture: None,
    }
}

/// Write `outcome` to a caller as one line.
fn answer_line(writer: &mut TcpStream, outcome: &ToolOutcome) -> std::io::Result<()> {
    let mut line = json!({"content": outcome.content, "is_error": outcome.is_error});
    if let Some(picture) = &outcome.picture {
        line["picture"] = json!(picture);
    }
    writeln!(writer, "{line}").and_then(|()| writer.flush())
}

/// The next line a caller sent.
enum Line {
    /// A line of text, without its line break.
    Text(String),
    /// The caller hung up, or sent bytes that are not text.
    End,
    /// The line is longer than the limit; its rest is unread.
    TooLong,
}

/// Read one line of at most `limit` bytes, so a caller that never sends a
/// line break cannot make the app hold what it sends without end.
fn read_line(reader: &mut BufReader<TcpStream>, limit: usize) -> Line {
    let mut bytes = Vec::new();
    let most = u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1);
    match reader.by_ref().take(most).read_until(b'\n', &mut bytes) {
        Err(_) | Ok(0) => return Line::End,
        Ok(_) => {}
    }
    if bytes.last() == Some(&b'\n') {
        bytes.pop();
        if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
    } else if bytes.len() > limit {
        return Line::TooLong;
    }
    String::from_utf8(bytes).map_or(Line::End, Line::Text)
}

/// What a caller is told when the app did not answer its call.
fn unanswered() -> ToolOutcome {
    ToolOutcome {
        content: "the Cutlass window did not answer".to_owned(),
        is_error: true,
        picture: None,
    }
}

/// How a call's wait for the app ended.
enum Waited {
    /// The app ran the call.
    Answered(ToolOutcome),
    /// The caller hung up before the app got to the call.
    CallerLeft,
    /// The app did not answer within the time a call may take.
    TooLong,
    /// The app let the call go unanswered: it closed the project.
    Dropped,
}

/// Wait up to `limit` for the app's answer to a call from `caller`,
/// looking every [`CALLER_LOOK`] whether the caller is still connected.
fn await_answer(caller: &TcpStream, answer: &Receiver<ToolOutcome>, limit: Duration) -> Waited {
    let asked = Instant::now();
    loop {
        match answer.recv_timeout(CALLER_LOOK) {
            Ok(outcome) => return Waited::Answered(outcome),
            Err(RecvTimeoutError::Disconnected) => return Waited::Dropped,
            Err(RecvTimeoutError::Timeout) => {}
        }
        if hung_up(caller) {
            return Waited::CallerLeft;
        }
        if asked.elapsed() >= limit {
            return Waited::TooLong;
        }
    }
}

/// Whether the other end of `stream` closed it. Nothing is read: a look
/// that would wait means the caller is there and quiet, and bytes mean it
/// has sent its next call.
fn hung_up(stream: &TcpStream) -> bool {
    if stream.set_nonblocking(true).is_err() {
        return false;
    }
    let looked = stream.peek(&mut [0u8; 1]);
    // This caller's lines are read with waits, as before the look. A
    // stream that cannot go back to that is of no more use.
    if stream.set_nonblocking(false).is_err() {
        return true;
    }
    match looked {
        Ok(0) => true,
        Ok(_) => false,
        Err(error) => !matches!(
            error.kind(),
            std::io::ErrorKind::WouldBlock
                | std::io::ErrorKind::TimedOut
                | std::io::ErrorKind::Interrupted
        ),
    }
}

fn read_call(line: &str, token: &str) -> Result<ToolCall, ToolOutcome> {
    let refuse = |content: &str| ToolOutcome {
        content: content.to_owned(),
        is_error: true,
        picture: None,
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

    /// Give up on an answer after `wait`, for a call that must not hang
    /// the caller, such as [`YIELD`] from a window that is opening.
    pub fn set_answer_wait(&mut self, wait: Duration) -> Result<(), String> {
        self.reader
            .get_ref()
            .set_read_timeout(Some(wait))
            .map_err(|error| error.to_string())
    }

    /// Run `call` in the app that holds the project. An error is the
    /// connection's: the app is gone or did not answer, and the caller
    /// should find the project's holder again.
    pub fn try_call(&mut self, call: &ToolCall) -> Result<ToolOutcome, String> {
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
            .map_err(|_| "the app that held the project closed the connection".to_owned())?;
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
            // A picture that does not read is left out; the text stands.
            picture: answer
                .get("picture")
                .and_then(|picture| serde_json::from_value(picture.clone()).ok()),
        })
    }
}

impl ToolSink for Forward {
    fn run_tool(&mut self, call: &ToolCall) -> ToolOutcome {
        self.try_call(call).unwrap_or_else(|error| ToolOutcome {
            content: format!("the call did not reach the Cutlass window: {error}"),
            is_error: true,
            picture: None,
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
                    // The second answer comes with a picture.
                    picture: (answered == 2).then(|| crate::Picture::jpeg(b"frame")),
                });
            }
            thread::sleep(Duration::from_millis(2));
        }
        let (first, second) = caller.join().unwrap();
        assert_eq!(first.content, "answer 1");
        assert_eq!(second.content, "answer 2");
        assert!(!first.is_error);
        assert_eq!(first.picture, None);
        assert_eq!(second.picture, Some(crate::Picture::jpeg(b"frame")));
        assert!(woken.load(Ordering::Acquire));
    }

    /// One call as a caller writes it on the wire.
    fn call_line(token: &str, name: &str) -> String {
        format!(
            "{}\n",
            json!({"token": token, "id": "1", "name": name, "arguments": ""})
        )
    }

    /// A socket, and a count of the calls that woke its app.
    fn counted_socket(answer_wait: Duration) -> (ProjectSocket, Arc<AtomicUsize>) {
        let woken = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&woken);
        let socket = ProjectSocket::open_waiting(
            Arc::new(move || {
                count.fetch_add(1, Ordering::AcqRel);
            }),
            answer_wait,
        )
        .unwrap();
        (socket, woken)
    }

    /// Wait until `done`, or fail saying `what` did not happen.
    fn until(what: &str, mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !done() {
            assert!(Instant::now() < deadline, "{what}");
            thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn a_call_whose_caller_hung_up_is_not_run() {
        let (socket, woken) = counted_socket(ANSWER_WAIT);
        let address: SocketAddr = socket.address().parse().unwrap();
        // The app has the call in hand when its caller goes, as when it
        // works through the calls ahead of it: the call says so.
        let mut caller = TcpStream::connect(address).unwrap();
        caller
            .write_all(call_line(socket.token(), YIELD).as_bytes())
            .unwrap();
        until("the call did not arrive", || {
            woken.load(Ordering::Acquire) == 1
        });
        let mut waiting = socket.take();
        assert_eq!(waiting.len(), 1);
        let held = waiting.remove(0);
        assert!(!held.caller_left(), "the caller is still there");
        drop(caller);
        until("the call did not hear its caller go", || held.caller_left());

        // A call still in the queue when its caller goes never reaches the
        // app: a window that gave up on its ask does not get the project
        // dropped for it minutes later.
        let mut caller = TcpStream::connect(address).unwrap();
        caller
            .write_all(call_line(socket.token(), YIELD).as_bytes())
            .unwrap();
        until("the second call did not arrive", || {
            woken.load(Ordering::Acquire) == 2
        });
        drop(caller);
        thread::sleep(CALLER_LOOK * 15);
        assert!(socket.take().is_empty());
    }

    #[test]
    fn a_call_the_app_was_too_slow_for_is_answered_as_failed_and_not_run() {
        let (socket, woken) = counted_socket(Duration::from_millis(250));
        let mut forward = Forward::connect(&socket.address(), socket.token()).unwrap();
        let call = ToolCall {
            id: "1".to_owned(),
            name: "draft.apply".to_owned(),
            arguments: "{}".to_owned(),
        };
        // No one takes the call in time: the caller hears that it failed.
        let outcome = forward.run_tool(&call);
        assert!(outcome.is_error);
        assert_eq!(outcome.content, "the Cutlass window did not answer");
        assert_eq!(woken.load(Ordering::Acquire), 1);
        // So it is gone from the queue, and does not land later.
        assert!(socket.take().is_empty());
        // The caller is still connected, and its next call is served.
        let caller = thread::spawn(move || forward.run_tool(&call));
        until("the next call did not arrive", || {
            woken.load(Ordering::Acquire) == 2
        });
        let mut waiting = socket.take();
        assert_eq!(waiting.len(), 1);
        waiting.remove(0).answer(ToolOutcome {
            content: "done".to_owned(),
            is_error: false,
            picture: None,
        });
        let outcome = caller.join().unwrap();
        assert_eq!(outcome.content, "done");
        assert!(!outcome.is_error);
    }

    /// The one line a caller is answered with, or nothing once the app
    /// hung up.
    fn answer_to(caller: &TcpStream) -> String {
        let mut line = String::new();
        let _ = BufReader::new(caller.try_clone().unwrap()).read_line(&mut line);
        line
    }

    #[test]
    fn a_call_with_no_end_is_cut_off_before_its_token_is_known() {
        let (socket, woken) = counted_socket(ANSWER_WAIT);
        let address: SocketAddr = socket.address().parse().unwrap();
        let mut caller = TcpStream::connect(address).unwrap();
        // One byte more than a call may have, and no line break.
        let flood = vec![b'a'; CALL_LIMIT + 1];
        caller.write_all(&flood).unwrap();
        let answer = answer_to(&caller);
        assert!(answer.contains(CALL_TOO_LONG), "{answer}");
        assert!(answer.contains("\"is_error\":true"), "{answer}");
        // The app let the caller go, and nothing reached its queue.
        assert_eq!(answer_to(&caller), "");
        assert_eq!(woken.load(Ordering::Acquire), 0);
        assert!(socket.take().is_empty());
        // A call of the longest length still reads: a line break is not
        // counted, and neither is the wrong token's refusal a hang-up.
        let mut caller = TcpStream::connect(address).unwrap();
        let mut long = vec![b' '; CALL_LIMIT];
        long.push(b'\n');
        caller.write_all(&long).unwrap();
        caller
            .write_all(call_line("not the token", "draft.apply").as_bytes())
            .unwrap();
        let answer = answer_to(&caller);
        assert!(answer.contains("token"), "{answer}");
    }

    #[test]
    fn only_so_many_callers_are_connected_at_once() {
        let (socket, woken) = counted_socket(ANSWER_WAIT);
        let address: SocketAddr = socket.address().parse().unwrap();
        let mut seated: Vec<TcpStream> = (0..CALLER_LIMIT)
            .map(|_| TcpStream::connect(address).unwrap())
            .collect();
        // The next one is told why and let go.
        let late = TcpStream::connect(address).unwrap();
        let answer = answer_to(&late);
        assert!(answer.contains(TOO_MANY_CALLERS), "{answer}");
        assert_eq!(answer_to(&late), "");
        // Those seated are served as before.
        seated[0]
            .write_all(call_line("not the token", "draft.apply").as_bytes())
            .unwrap();
        assert!(answer_to(&seated[0]).contains("token"));
        // A seat that is left is free for the next caller.
        drop(seated.pop());
        until("the seat was not given back", || {
            let mut next = TcpStream::connect(address).unwrap();
            next.write_all(call_line("not the token", "draft.apply").as_bytes())
                .unwrap();
            answer_to(&next).contains("token")
        });
        assert_eq!(woken.load(Ordering::Acquire), 0);
        assert!(socket.take().is_empty());
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
