//! The process log.
//!
//! Every level is written to `cutlass.log`. `cargo run` and
//! `cargo run --release` also write those lines to stderr. A public build
//! passes the `public-release` feature and does not write the console.
//! Stdout is never written, because `cutlass mcp` speaks on stdout.
//!
//! A call formats one line, writes it, and flushes before it returns.
//! Keep it off the per-frame and per-sample paths. Call [`init`] or
//! [`init_default`] once at process start. A record before that is discarded.
//! A failed write does not fail the caller. The next record tries again.

#![forbid(unsafe_code)]

mod dir;
mod time;

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::thread;
use std::time::SystemTime;

pub use dir::default_directory;

/// Live log size that triggers rotation, in bytes.
pub const MAX_FILE_BYTES: u64 = 8 * 1024 * 1024;

/// Message bytes kept on one line, before the truncation marker.
pub const MAX_MESSAGE_BYTES: usize = 16 * 1024;

/// Log file name inside the log directory.
pub const FILE_NAME: &str = "cutlass.log";

/// Previous log kept after rotation.
pub const PREVIOUS_NAME: &str = "cutlass.log.1";

const MAX_TARGET_BYTES: usize = 200;
const TRUNCATED: &str = " (truncated)";

static STATE: Mutex<Option<Logger>> = Mutex::new(None);

/// Why the log file could not be opened.
#[derive(Debug, PartialEq, Eq)]
pub enum InitError {
    /// No home or state directory is set, so there is nowhere to put the file.
    NoDirectory,
    /// The log path is a symlink. It is not followed.
    Symlink {
        /// The path that was refused.
        path: PathBuf,
    },
    /// The directory or the file could not be opened.
    Io {
        /// The path that failed.
        path: PathBuf,
        /// The filesystem error text.
        message: String,
    },
}

impl fmt::Display for InitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDirectory => formatter.write_str("no log directory is available"),
            Self::Symlink { path } => {
                write!(formatter, "log file {} is a symlink", path.display())
            }
            Self::Io { path, message } => {
                write!(formatter, "could not open {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for InitError {}

/// How serious a record is. None of these are filtered out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// The operation failed.
    Error,
    /// The operation continued with a problem.
    Warn,
    /// A normal step.
    Info,
    /// Detail for a development run.
    Debug,
    /// The finest detail.
    Trace,
}

impl Level {
    fn as_str(self) -> &'static str {
        match self {
            Self::Error => "ERROR",
            Self::Warn => "WARN",
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Trace => "TRACE",
        }
    }
}

impl fmt::Display for Level {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// This build writes stderr after a successful [`init`].
///
/// `public-release` compiles the stderr sink out. `cargo run --release`
/// does not pass that feature, so it still returns true.
#[must_use]
pub const fn console_enabled() -> bool {
    cfg!(not(feature = "public-release"))
}

/// Open `cutlass.log` in `directory`, creating the directory if needed.
///
/// The returned path is the live file. A second call switches to that
/// directory. On failure the error is also recorded when a console sink
/// is available.
pub fn init(directory: &Path) -> Result<PathBuf, InitError> {
    let result = install(directory, process_console(), MAX_FILE_BYTES);
    report_open(&result);
    result
}

/// Open `cutlass.log` in [`default_directory`].
pub fn init_default() -> Result<PathBuf, InitError> {
    match default_directory() {
        Ok(directory) => init(&directory),
        Err(error) => {
            install_console_only(process_console());
            record(
                Level::Error,
                "cutlass_logger",
                &format!("log file unavailable: {error}"),
            );
            Err(error)
        }
    }
}

/// Write one record. Before [`init`], this does nothing.
pub fn record(level: Level, target: &str, message: &str) {
    let line = format_line(level, target, message);
    let mut guard = lock();
    let Some(logger) = guard.as_mut() else {
        return;
    };
    if let Some(file) = logger.file.as_mut() {
        let _ = write_file(file, &line);
    }
    write_console(&mut logger.console, &line);
}

/// Record an error.
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {{
        let message = ::std::format!($($arg)*);
        $crate::record($crate::Level::Error, module_path!(), &message);
    }};
}

/// Record a warning.
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {{
        let message = ::std::format!($($arg)*);
        $crate::record($crate::Level::Warn, module_path!(), &message);
    }};
}

/// Record an ordinary step.
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {{
        let message = ::std::format!($($arg)*);
        $crate::record($crate::Level::Info, module_path!(), &message);
    }};
}

/// Record development detail.
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {{
        let message = ::std::format!($($arg)*);
        $crate::record($crate::Level::Debug, module_path!(), &message);
    }};
}

/// Record the finest detail.
#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {{
        let message = ::std::format!($($arg)*);
        $crate::record($crate::Level::Trace, module_path!(), &message);
    }};
}

struct Logger {
    file: Option<LogFile>,
    console: Console,
}

struct LogFile {
    path: PathBuf,
    file: Option<File>,
    bytes: u64,
    max_bytes: u64,
}

enum Console {
    #[cfg(feature = "public-release")]
    Off,
    #[cfg(not(feature = "public-release"))]
    Stderr,
    #[cfg(test)]
    Capture(Vec<u8>),
}

fn process_console() -> Console {
    #[cfg(not(feature = "public-release"))]
    {
        Console::Stderr
    }
    #[cfg(feature = "public-release")]
    {
        Console::Off
    }
}

fn lock() -> MutexGuard<'static, Option<Logger>> {
    STATE.lock().unwrap_or_else(|error| error.into_inner())
}

fn install(directory: &Path, console: Console, max_bytes: u64) -> Result<PathBuf, InitError> {
    match open_log(directory, max_bytes) {
        Ok(file) => {
            let path = file.path.clone();
            *lock() = Some(Logger {
                file: Some(file),
                console,
            });
            Ok(path)
        }
        Err(error) => {
            if console_is_on(&console) {
                let mut guard = lock();
                if guard.is_none() {
                    *guard = Some(Logger {
                        file: None,
                        console,
                    });
                }
            }
            Err(error)
        }
    }
}

fn install_console_only(console: Console) {
    if !console_is_on(&console) {
        return;
    }
    let mut guard = lock();
    if guard.is_none() {
        *guard = Some(Logger {
            file: None,
            console,
        });
    }
}

fn console_is_on(console: &Console) -> bool {
    match console {
        #[cfg(feature = "public-release")]
        Console::Off => false,
        #[cfg(not(feature = "public-release"))]
        Console::Stderr => true,
        #[cfg(test)]
        Console::Capture(_) => true,
    }
}

fn report_open(result: &Result<PathBuf, InitError>) {
    match result {
        Ok(path) => record(
            Level::Info,
            "cutlass_logger",
            &format!("log file {}", path.display()),
        ),
        Err(error) => record(
            Level::Error,
            "cutlass_logger",
            &format!("log file unavailable: {error}"),
        ),
    }
}

fn open_log(directory: &Path, max_bytes: u64) -> Result<LogFile, InitError> {
    fs::create_dir_all(directory).map_err(|error| io_error(directory, error))?;
    restrict_directory(directory)?;
    let path = directory.join(FILE_NAME);
    refuse_symlink(&path)?;
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .read(false)
        .open(&path)
        .map_err(|error| io_error(&path, error))?;
    restrict_file(&file, &path)?;
    let bytes = file
        .metadata()
        .map_err(|error| io_error(&path, error))?
        .len();
    Ok(LogFile {
        path,
        file: Some(file),
        bytes,
        max_bytes,
    })
}

fn refuse_symlink(path: &Path) -> Result<(), InitError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(InitError::Symlink {
            path: path.to_path_buf(),
        }),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io_error(path, error)),
    }
}

fn io_error(path: &Path, error: std::io::Error) -> InitError {
    InitError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

fn format_line(level: Level, target: &str, message: &str) -> String {
    let time = time::format_system_time(SystemTime::now());
    let thread = format!("{:?}", thread::current().id());
    let target = sanitize_target(target);
    let message = sanitize_message(message);
    format!("{time} {level} {thread} {target} {message}")
}

fn sanitize_target(target: &str) -> String {
    if target.is_empty() {
        return "-".to_owned();
    }
    let mut out = String::new();
    for ch in target.chars() {
        let next = if ch.is_whitespace() { '_' } else { ch };
        if out.len() + next.len_utf8() > MAX_TARGET_BYTES {
            break;
        }
        out.push(next);
    }
    if out.is_empty() { "-".to_owned() } else { out }
}

fn sanitize_message(message: &str) -> String {
    let mut out = String::new();
    let mut truncated = false;
    for ch in message.chars() {
        let piece = if ch == '\n' || ch == '\r' {
            2
        } else {
            ch.len_utf8()
        };
        if out.len() + piece > MAX_MESSAGE_BYTES {
            truncated = true;
            break;
        }
        if ch == '\n' || ch == '\r' {
            out.push_str("\\n");
        } else {
            out.push(ch);
        }
    }
    if truncated {
        out.push_str(TRUNCATED);
    }
    out
}

fn write_file(log: &mut LogFile, line: &str) -> std::io::Result<()> {
    let extra = u64::try_from(line.len().saturating_add(1)).unwrap_or(u64::MAX);
    if log.bytes > 0 && log.bytes.saturating_add(extra) > log.max_bytes {
        rotate(log)?;
    }
    let bytes = terminated(line);
    let Some(file) = log.file.as_mut() else {
        return Err(std::io::Error::other("log file is closed"));
    };
    file.write_all(&bytes)?;
    file.flush()?;
    log.bytes = log.bytes.saturating_add(extra);
    Ok(())
}

fn terminated(line: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(line.len() + 1);
    bytes.extend_from_slice(line.as_bytes());
    bytes.push(b'\n');
    bytes
}

fn rotate(log: &mut LogFile) -> std::io::Result<()> {
    if let Some(file) = log.file.as_mut() {
        file.flush()?;
    }
    log.file.take();
    let previous = log.path.with_file_name(PREVIOUS_NAME);
    if let Err(error) = remove_if_present(&previous) {
        reopen(log)?;
        return Err(error);
    }
    if let Err(error) = fs::rename(&log.path, &previous) {
        reopen(log)?;
        return Err(error);
    }
    reopen(log)?;
    Ok(())
}

fn remove_if_present(path: &Path) -> std::io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn reopen(log: &mut LogFile) -> std::io::Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log.path)?;
    restrict_file(&file, &log.path).map_err(std::io::Error::other)?;
    log.bytes = file.metadata()?.len();
    log.file = Some(file);
    Ok(())
}

#[cfg(unix)]
fn restrict_directory(path: &Path) -> Result<(), InitError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .map_err(|error| io_error(path, error))
}

#[cfg(not(unix))]
fn restrict_directory(_path: &Path) -> Result<(), InitError> {
    Ok(())
}

#[cfg(unix)]
fn restrict_file(file: &File, path: &Path) -> Result<(), InitError> {
    use std::os::unix::fs::PermissionsExt;
    file.set_permissions(fs::Permissions::from_mode(0o600))
        .map_err(|error| io_error(path, error))
}

#[cfg(not(unix))]
fn restrict_file(_file: &File, _path: &Path) -> Result<(), InitError> {
    Ok(())
}

fn write_console(console: &mut Console, line: &str) {
    match console {
        #[cfg(feature = "public-release")]
        Console::Off => {
            let _ = line;
        }
        #[cfg(not(feature = "public-release"))]
        Console::Stderr => write_stderr(&terminated(line)),
        #[cfg(test)]
        Console::Capture(buffer) => buffer.extend_from_slice(&terminated(line)),
    }
}

#[cfg(not(feature = "public-release"))]
fn write_stderr(bytes: &[u8]) {
    let mut sink = std::io::stderr().lock();
    let _ = sink.write_all(bytes);
}

#[cfg(test)]
fn init_capture(directory: &Path, max_bytes: u64) -> Result<PathBuf, InitError> {
    let result = install(directory, Console::Capture(Vec::new()), max_bytes);
    report_open(&result);
    result
}

#[cfg(test)]
fn console_bytes() -> Vec<u8> {
    let guard = lock();
    match guard.as_ref() {
        Some(Logger {
            console: Console::Capture(buffer),
            ..
        }) => buffer.clone(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
fn console_kind() -> &'static str {
    let guard = lock();
    match guard.as_ref().map(|logger| &logger.console) {
        None => "none",
        #[cfg(feature = "public-release")]
        Some(Console::Off) => "off",
        #[cfg(not(feature = "public-release"))]
        Some(Console::Stderr) => "stderr",
        Some(Console::Capture(_)) => "capture",
    }
}

#[cfg(test)]
fn installed_max_bytes() -> Option<u64> {
    lock()
        .as_ref()
        .and_then(|logger| logger.file.as_ref().map(|file| file.max_bytes))
}

#[cfg(test)]
fn reset() {
    *lock() = None;
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, MutexGuard};

    use super::{
        FILE_NAME, InitError, Level, MAX_FILE_BYTES, MAX_MESSAGE_BYTES, PREVIOUS_NAME,
        console_bytes, console_enabled, console_kind, init, init_capture, installed_max_bytes,
        record, reset,
    };

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("cutlass-logger-{}-{n}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    struct Env {
        temp: Temp,
        _guard: MutexGuard<'static, ()>,
    }

    impl Env {
        fn new() -> Self {
            let guard = test_lock();
            reset();
            Self {
                temp: Temp::new(),
                _guard: guard,
            }
        }
    }

    impl Drop for Env {
        fn drop(&mut self) {
            reset();
        }
    }

    fn test_lock() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|error| error.into_inner())
    }

    fn valid_record(line: &str) -> bool {
        let mut parts = line.splitn(5, ' ');
        let Some(time) = parts.next() else {
            return false;
        };
        let Some(level) = parts.next() else {
            return false;
        };
        let Some(thread) = parts.next() else {
            return false;
        };
        let Some(target) = parts.next() else {
            return false;
        };
        let Some(message) = parts.next() else {
            return false;
        };
        time.len() >= 27
            && time.ends_with('Z')
            && time.contains('T')
            && matches!(level, "ERROR" | "WARN" | "INFO" | "DEBUG" | "TRACE")
            && thread.starts_with("ThreadId(")
            && thread.ends_with(')')
            && !target.is_empty()
            && !target.contains(' ')
            && !message.contains('\n')
    }

    #[test]
    fn every_level_reaches_the_file_and_the_console_sink() {
        let env = Env::new();
        let path = init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap();
        error!("err-token");
        warn!("warn-token");
        info!("info-token");
        debug!("debug-token");
        trace!("trace-token");
        let file = fs::read(&path).unwrap();
        assert_eq!(file, console_bytes());
        let text = String::from_utf8(file).unwrap();
        for (level, token) in [
            ("ERROR", "err-token"),
            ("WARN", "warn-token"),
            ("INFO", "info-token"),
            ("DEBUG", "debug-token"),
            ("TRACE", "trace-token"),
        ] {
            assert!(text.contains(level), "{level} missing in {text}");
            assert!(text.contains(token), "{token} missing in {text}");
        }
        for line in text.lines() {
            assert!(valid_record(line), "{line}");
        }
    }

    #[test]
    fn a_newline_stays_inside_one_record() {
        let env = Env::new();
        let path = init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap();
        info!("alpha\nbeta\rgamma");
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("alpha\\nbeta\\ngamma"));
        assert_eq!(
            text.lines().filter(|line| line.contains("alpha")).count(),
            1
        );
        assert!(!text.contains("beta\n"));
    }

    #[test]
    fn a_long_message_is_cut_on_a_character_boundary() {
        let env = Env::new();
        let path = init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap();
        let message = "é".repeat(MAX_MESSAGE_BYTES);
        info!("{message}");
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("(truncated)"));
        assert_eq!(text.matches('é').count(), MAX_MESSAGE_BYTES / 2);
        assert!(text.lines().all(valid_record));
    }

    #[test]
    fn an_empty_target_and_message_are_still_one_record() {
        let env = Env::new();
        let path = init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap();
        record(Level::Warn, "a b\nc", "");
        record(Level::Info, "", "");
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains(" WARN "));
        assert!(text.contains(" a_b_c "));
        assert!(text.contains(" INFO "));
        assert!(text.contains(" - "));
        for line in text.lines() {
            assert!(valid_record(line), "{line}");
        }
    }

    #[test]
    fn a_record_before_init_is_discarded() {
        let _env = Env::new();
        record(Level::Error, "cutlass_logger", "no file yet");
        assert!(super::lock().is_none());
    }

    #[test]
    fn an_oversized_log_rotates_before_the_next_record() {
        let env = Env::new();
        let path = env.temp.path.join(FILE_NAME);
        fs::write(&path, vec![b'x'; 300]).unwrap();
        init_capture(&env.temp.path, 100).unwrap();
        let previous = fs::read(env.temp.path.join(PREVIOUS_NAME)).unwrap();
        assert_eq!(previous, vec![b'x'; 300]);
        let live = fs::read_to_string(&path).unwrap();
        assert!(live.contains("log file"));
        assert!(!live.contains('x'));
    }

    #[test]
    fn rotation_drops_the_oldest_and_stays_within_the_cap() {
        let env = Env::new();
        let max = 4 * 1024;
        let path = init_capture(&env.temp.path, max).unwrap();
        let mut count = 0;
        loop {
            info!("mark-{count:04}");
            count += 1;
            let combined = combined_text(&env.temp.path);
            if count > 20
                && !combined.contains("mark-0000")
                && combined.contains(&format!("mark-{:04}", count - 1))
            {
                break;
            }
            assert!(count < 500, "rotation did not drop the oldest record");
        }
        let live = fs::metadata(&path).unwrap().len();
        let previous = fs::metadata(env.temp.path.join(PREVIOUS_NAME))
            .unwrap()
            .len();
        assert!(live <= max, "live {live} exceeds {max}");
        assert!(previous <= max, "previous {previous} exceeds {max}");
        let text = combined_text(&env.temp.path);
        for line in text.lines() {
            assert!(valid_record(line), "{line}");
        }
    }

    #[test]
    fn records_stay_inside_the_log_directory() {
        let env = Env::new();
        init_capture(&env.temp.path, 256).unwrap();
        for n in 0..30 {
            info!("inside-{n}");
        }
        for entry in fs::read_dir(&env.temp.path).unwrap() {
            let name = entry.unwrap().file_name();
            let text = name.to_string_lossy();
            assert!(text == FILE_NAME || text == PREVIOUS_NAME, "{text}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_log_is_refused() {
        let env = Env::new();
        let outside = env.temp.path.join("outside");
        fs::write(&outside, b"keep").unwrap();
        let link = env.temp.path.join(FILE_NAME);
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        let error = init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap_err();
        assert_eq!(error, InitError::Symlink { path: link });
        assert_eq!(fs::read(&outside).unwrap(), b"keep");
        let text = String::from_utf8(console_bytes()).unwrap();
        assert!(text.contains("symlink"));
        assert!(text.contains("ERROR"));
    }

    #[test]
    fn a_file_where_the_directory_should_be_is_an_error() {
        let env = Env::new();
        let blocker = env.temp.path.join("blocker");
        fs::write(&blocker, b"x").unwrap();
        let nested = blocker.join("logs");
        let error = init_capture(&nested, MAX_FILE_BYTES).unwrap_err();
        assert!(matches!(error, InitError::Io { .. }));
        assert!(!nested.exists());
        assert_eq!(fs::read(&blocker).unwrap(), b"x");
        let text = String::from_utf8(console_bytes()).unwrap();
        assert!(text.contains("log file unavailable"));
    }

    #[test]
    fn a_directory_named_as_the_log_file_is_an_error() {
        let env = Env::new();
        fs::create_dir(env.temp.path.join(FILE_NAME)).unwrap();
        let error = init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap_err();
        assert!(matches!(error, InitError::Io { .. }));
        assert!(env.temp.path.join(FILE_NAME).is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn the_log_directory_and_file_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let env = Env::new();
        let path = init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap();
        let directory_mode = fs::metadata(&env.temp.path).unwrap().permissions().mode() & 0o777;
        let file_mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(directory_mode, 0o700);
        assert_eq!(file_mode, 0o600);
    }

    #[test]
    fn process_init_uses_the_file_cap_and_the_build_console() {
        let env = Env::new();
        let path = init(&env.temp.path).unwrap();
        assert_eq!(installed_max_bytes(), Some(MAX_FILE_BYTES));
        assert_eq!(console_enabled(), cfg!(not(feature = "public-release")));
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("log file"));
        assert!(text.contains("INFO"));
        if console_enabled() {
            assert_eq!(console_kind(), "stderr");
        } else {
            assert_eq!(console_kind(), "off");
            assert!(console_bytes().is_empty());
        }
    }

    #[test]
    fn concurrent_records_stay_intact() {
        let env = Env::new();
        let path = init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap();
        std::thread::scope(|scope| {
            for id in 0..8 {
                scope.spawn(move || {
                    for n in 0..80 {
                        info!("mark-{id}-{n}");
                    }
                });
            }
        });
        let file = fs::read(&path).unwrap();
        assert_eq!(file, console_bytes());
        let text = String::from_utf8(file).unwrap();
        let mut marks = std::collections::BTreeSet::new();
        for line in text.lines() {
            assert!(valid_record(line), "{line}");
            if let Some(mark) = line
                .split_whitespace()
                .find(|part| part.starts_with("mark-"))
            {
                assert!(marks.insert(mark.to_owned()), "duplicate {mark}");
            }
        }
        assert_eq!(marks.len(), 8 * 80);
        assert_eq!(text.matches("mark-").count(), marks.len());
    }

    fn combined_text(directory: &std::path::Path) -> String {
        let mut text = String::new();
        let previous = directory.join(PREVIOUS_NAME);
        if previous.is_file() {
            text.push_str(&fs::read_to_string(&previous).unwrap());
        }
        text.push_str(&fs::read_to_string(directory.join(FILE_NAME)).unwrap());
        text
    }
}
