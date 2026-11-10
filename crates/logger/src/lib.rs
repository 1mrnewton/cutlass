//! The process log.
//!
//! Records at the current level and above are written to `cutlass.log`.
//! The level is INFO unless `CUTLASS_LOG` names another one when [`init`]
//! runs, and [`set_level`] changes it later. `cargo run` and
//! `cargo run --release` also write those lines to stderr. A public build
//! passes the `public-release` feature and does not write the console.
//! Stdout is never written, because `cutlass mcp` speaks on stdout.
//! Crates that report through the `log` facade (wgpu, eframe, burn) land
//! here too, at WARN and above only.
//!
//! A call formats one line, writes it, and flushes before it returns. A
//! record below the level is dropped before its message is formatted.
//! Keep it off the per-frame and per-sample paths. Call [`init`] once at
//! process start with the logs folder of the Cutlass root. A record before
//! that is discarded.
//! A failed write does not fail the caller. The next record tries again.

#![forbid(unsafe_code)]

mod time;

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::thread;
use std::time::SystemTime;

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

/// Environment variable that sets the level when [`init`] runs.
pub const LEVEL_ENV: &str = "CUTLASS_LOG";

/// The level when `CUTLASS_LOG` is unset.
pub const DEFAULT_LEVEL: Level = Level::Info;

static STATE: Mutex<Option<Logger>> = Mutex::new(None);
static MAX_LEVEL: AtomicU8 = AtomicU8::new(DEFAULT_LEVEL as u8);

/// Why the log file could not be opened.
#[derive(Debug, PartialEq, Eq)]
pub enum InitError {
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

/// How serious a record is, most serious first. A record is kept when
/// its level is at or above the current one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Level {
    /// The operation failed.
    Error = 1,
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

/// A `CUTLASS_LOG` value that names no level.
#[derive(Debug, PartialEq, Eq)]
pub struct ParseLevelError {
    /// The value that was refused.
    pub value: String,
}

impl fmt::Display for ParseLevelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{LEVEL_ENV}={:?} is not error, warn, info, debug or trace",
            self.value
        )
    }
}

impl std::error::Error for ParseLevelError {}

impl FromStr for Level {
    type Err = ParseLevelError;

    /// The level name in any case, with surrounding space ignored.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "error" => Ok(Self::Error),
            "warn" | "warning" => Ok(Self::Warn),
            "info" => Ok(Self::Info),
            "debug" => Ok(Self::Debug),
            "trace" => Ok(Self::Trace),
            _ => Err(ParseLevelError {
                value: value.to_owned(),
            }),
        }
    }
}

impl Level {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Error,
            2 => Self::Warn,
            3 => Self::Info,
            4 => Self::Debug,
            _ => Self::Trace,
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

/// Open `cutlass.log` in `directory`, creating the directory if needed,
/// and take the level from `CUTLASS_LOG`, or INFO when it is unset.
///
/// The returned path is the live file. A second call switches to that
/// directory. On failure the error is also recorded when a console sink
/// is available. A `CUTLASS_LOG` that names no level keeps INFO and is
/// recorded as a warning.
pub fn init(directory: &Path) -> Result<PathBuf, InitError> {
    let chosen = level_from_env(std::env::var(LEVEL_ENV).ok().as_deref());
    set_level(*chosen.as_ref().unwrap_or(&DEFAULT_LEVEL));
    install_bridge();
    let result = install(directory, process_console(), MAX_FILE_BYTES);
    report_open(&result);
    if let Err(error) = chosen {
        record(Level::Warn, "cutlass_logger", &error.to_string());
    }
    result
}

/// The level a `CUTLASS_LOG` value asks for. Unset or empty is INFO.
pub fn level_from_env(value: Option<&str>) -> Result<Level, ParseLevelError> {
    match value {
        None => Ok(DEFAULT_LEVEL),
        Some(value) if value.trim().is_empty() => Ok(DEFAULT_LEVEL),
        Some(value) => value.parse(),
    }
}

/// Keep records at `level` and above from now on.
pub fn set_level(level: Level) {
    MAX_LEVEL.store(level as u8, Ordering::Relaxed);
}

/// The least serious level that is kept.
#[must_use]
pub fn level() -> Level {
    Level::from_u8(MAX_LEVEL.load(Ordering::Relaxed))
}

/// A record at `level` would be kept. The macros check this before they
/// format the message.
#[must_use]
#[inline]
pub fn enabled(level: Level) -> bool {
    level as u8 <= MAX_LEVEL.load(Ordering::Relaxed)
}

/// The folder the installed log writes to, or `None` before [`init`] or
/// when the file did not open.
#[must_use]
pub fn directory() -> Option<PathBuf> {
    lock()
        .as_ref()
        .and_then(|logger| logger.file.as_ref())
        .and_then(|file| file.path.parent().map(Path::to_path_buf))
}

/// Copy the log in `directory`, the previous file first, into one file at
/// `destination`. The logger is held meanwhile, so no record is cut in half
/// and a rotation cannot move a file away mid-copy. The copy is written
/// beside the destination and renamed over it, so a failed export leaves
/// no partial file. A directory with no log gives an empty file.
pub fn export(directory: &Path, destination: &Path) -> std::io::Result<()> {
    let _guard = lock();
    let mut bytes = Vec::new();
    for name in [PREVIOUS_NAME, FILE_NAME] {
        match fs::read(directory.join(name)) {
            Ok(part) => bytes.extend_from_slice(&part),
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    let name = destination
        .file_name()
        .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidInput, "no file name"))?;
    let mut partial = name.to_os_string();
    partial.push(".partial");
    let partial = destination.with_file_name(partial);
    let written = File::create(&partial)
        .and_then(|mut file| file.write_all(&bytes).and_then(|()| file.sync_all()))
        .and_then(|()| fs::rename(&partial, destination));
    if written.is_err() {
        let _ = fs::remove_file(&partial);
    }
    written
}

/// Write one record. Before [`init`], or below the level, this does
/// nothing.
pub fn record(level: Level, target: &str, message: &str) {
    if !enabled(level) {
        return;
    }
    let line = format_line(level, target, message);
    write_line(&mut lock(), &line);
}

/// Record an error.
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {{
        if $crate::enabled($crate::Level::Error) {
            let message = ::std::format!($($arg)*);
            $crate::record($crate::Level::Error, ::std::module_path!(), &message);
        }
    }};
}

/// Record a warning.
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {{
        if $crate::enabled($crate::Level::Warn) {
            let message = ::std::format!($($arg)*);
            $crate::record($crate::Level::Warn, ::std::module_path!(), &message);
        }
    }};
}

/// Record an ordinary step.
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {{
        if $crate::enabled($crate::Level::Info) {
            let message = ::std::format!($($arg)*);
            $crate::record($crate::Level::Info, ::std::module_path!(), &message);
        }
    }};
}

/// Record development detail.
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {{
        if $crate::enabled($crate::Level::Debug) {
            let message = ::std::format!($($arg)*);
            $crate::record($crate::Level::Debug, ::std::module_path!(), &message);
        }
    }};
}

/// Record the finest detail.
#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {{
        if $crate::enabled($crate::Level::Trace) {
            let message = ::std::format!($($arg)*);
            $crate::record($crate::Level::Trace, ::std::module_path!(), &message);
        }
    }};
}

/// Record every panic at ERROR, with its thread, place, message and a
/// backtrace, before the hook that was set before runs. Call once at
/// process start, after [`init`].
///
/// A panic inside the logger itself, with the log held on that thread,
/// gives up on the record after a short wait rather than deadlocking. A
/// second call changes nothing, so a panic is never recorded twice.
pub fn install_panic_hook() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(chain_panic_hook);
}

fn chain_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        record_panic(info);
        previous(info);
    }));
}

fn record_panic(info: &std::panic::PanicHookInfo<'_>) {
    let payload = info.payload();
    let message = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("(a payload that is not text)");
    let place = info.location().map_or_else(
        || "an unknown place".to_owned(),
        |location| {
            format!(
                "{}:{}:{}",
                location.file(),
                location.line(),
                location.column()
            )
        },
    );
    let current = thread::current();
    let name = current.name().unwrap_or("unnamed");
    let backtrace = std::backtrace::Backtrace::force_capture();
    let line = format_line(
        Level::Error,
        "panic",
        &format!("thread {name} panicked at {place}: {message}\nbacktrace:\n{backtrace}"),
    );
    // The log may already be held on this thread, if the panic came from
    // inside a write. Waiting for it then would never end.
    for _ in 0..PANIC_LOCK_TRIES {
        match STATE.try_lock() {
            Ok(mut guard) => return write_line(&mut guard, &line),
            Err(std::sync::TryLockError::Poisoned(error)) => {
                return write_line(&mut error.into_inner(), &line);
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

const PANIC_LOCK_TRIES: u32 = 200;

fn write_line(guard: &mut Option<Logger>, line: &str) {
    let Some(logger) = guard.as_mut() else {
        return;
    };
    if let Some(file) = logger.file.as_mut() {
        let _ = write_file(file, line);
    }
    write_console(&mut logger.console, line);
}

/// The most detail kept from crates that report through the `log` facade
/// (wgpu, eframe, burn). Their INFO and below runs every frame.
pub const OTHER_CRATES_LEVEL: Level = Level::Warn;

struct Bridge;

static BRIDGE: Bridge = Bridge;

impl Bridge {
    fn level(level: log::Level) -> Level {
        match level {
            log::Level::Error => Level::Error,
            log::Level::Warn => Level::Warn,
            log::Level::Info => Level::Info,
            log::Level::Debug => Level::Debug,
            log::Level::Trace => Level::Trace,
        }
    }
}

impl log::Log for Bridge {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        let level = Self::level(metadata.level());
        level <= OTHER_CRATES_LEVEL && enabled(level)
    }

    fn log(&self, entry: &log::Record<'_>) {
        if log::Log::enabled(self, entry.metadata()) {
            let text = entry.args().to_string();
            // A kernel the GPU refused before it ran is kept as detail.
            let level = if refused_kernel(entry.target(), &text) {
                Level::Debug
            } else {
                Self::level(entry.level())
            };
            if enabled(level) {
                record(level, entry.target(), &text);
            }
        }
    }

    fn flush(&self) {}
}

/// Whether a record from another crate is burn's GPU runtime saying a
/// kernel asked for more than this GPU gives one, such as more than
/// Metal's 32 KB of shared memory, so the kernel was refused before it ran.
///
/// The runtime tries several ways to run an operation and keeps one that
/// the GPU takes; each refusal is still reported as a warning, once a size.
/// Nothing went wrong: the models' GPU output is checked against PyTorch's
/// with these refusals happening. A kernel that fails any other way, to
/// compile or while it runs, is not matched and stays a warning.
fn refused_kernel(target: &str, text: &str) -> bool {
    target.starts_with("cubecl_runtime")
        && text.contains("Too many resources were requested during launch")
}

/// Send `log` facade records at WARN and above into this log. A second
/// call, or a process that set another `log` logger first, changes nothing.
fn install_bridge() {
    if log::set_logger(&BRIDGE).is_ok() {
        log::set_max_level(log::LevelFilter::Warn);
    }
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
            &format!("log file {}, level {}", path.display(), level()),
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
    set_level(DEFAULT_LEVEL);
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, MutexGuard};

    use super::{
        DEFAULT_LEVEL, FILE_NAME, InitError, Level, MAX_FILE_BYTES, MAX_MESSAGE_BYTES,
        PREVIOUS_NAME, ParseLevelError, console_bytes, console_enabled, console_kind, directory,
        enabled, export, init, init_capture, installed_max_bytes, level, level_from_env, record,
        reset, set_level,
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
    fn every_level_reaches_the_file_and_the_console_sink_at_trace() {
        let env = Env::new();
        let path = init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap();
        set_level(Level::Trace);
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
    fn the_default_level_keeps_info_and_drops_debug_and_trace() {
        let env = Env::new();
        let path = init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap();
        assert_eq!(level(), DEFAULT_LEVEL);
        assert_eq!(DEFAULT_LEVEL, Level::Info);
        error!("err-token");
        warn!("warn-token");
        info!("info-token");
        debug!("debug-token");
        trace!("trace-token");
        let text = fs::read_to_string(&path).unwrap();
        for token in ["err-token", "warn-token", "info-token"] {
            assert!(text.contains(token), "{token} missing in {text}");
        }
        for token in ["debug-token", "trace-token"] {
            assert!(!text.contains(token), "{token} kept in {text}");
        }
        assert!(!text.contains(" DEBUG ") && !text.contains(" TRACE "));
        assert_eq!(fs::read(&path).unwrap(), console_bytes());
    }

    #[test]
    fn set_level_moves_the_cut_both_ways() {
        let env = Env::new();
        let path = init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap();
        set_level(Level::Warn);
        info!("quiet-info");
        warn!("loud-warn");
        set_level(Level::Debug);
        debug!("now-debug");
        trace!("still-no-trace");
        set_level(Level::Error);
        warn!("quiet-warn");
        error!("loud-error");
        record(Level::Info, "direct", "quiet-direct");
        let text = fs::read_to_string(&path).unwrap();
        for token in ["loud-warn", "now-debug", "loud-error"] {
            assert!(text.contains(token), "{token} missing in {text}");
        }
        for token in ["quiet-info", "still-no-trace", "quiet-warn", "quiet-direct"] {
            assert!(!text.contains(token), "{token} kept in {text}");
        }
    }

    #[test]
    fn a_dropped_record_is_never_formatted() {
        use std::cell::Cell;
        use std::fmt;

        struct Counted<'a>(&'a Cell<u32>);
        impl fmt::Display for Counted<'_> {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.set(self.0.get() + 1);
                formatter.write_str("counted")
            }
        }

        let env = Env::new();
        init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap();
        let calls = Cell::new(0);
        debug!("{}", Counted(&calls));
        trace!("{}", Counted(&calls));
        assert_eq!(calls.get(), 0);
        info!("{}", Counted(&calls));
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn other_crates_reach_the_file_at_warn_and_above_only() {
        let env = Env::new();
        let path = init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap();
        super::install_bridge();
        set_level(Level::Trace);
        log::error!(target: "wgpu_core::device", "bridged-error");
        log::warn!(target: "eframe", "bridged-warn");
        log::info!(target: "wgpu_core::device", "bridged-info");
        log::debug!(target: "eframe", "bridged-debug");
        set_level(Level::Error);
        log::warn!(target: "eframe", "bridged-quiet-warn");
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains(" ERROR "), "{text}");
        assert!(text.contains(" wgpu_core::device bridged-error"), "{text}");
        assert!(text.contains(" eframe bridged-warn"), "{text}");
        for token in ["bridged-info", "bridged-debug", "bridged-quiet-warn"] {
            assert!(!text.contains(token), "{token} kept in {text}");
        }
    }

    #[test]
    fn a_kernel_the_gpu_refused_is_detail_and_any_other_failure_a_warning() {
        let env = Env::new();
        let path = init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap();
        super::install_bridge();
        let refusal = "A launch error happened\nCaused by:\n  Too many resources were requested \
                       during launch\nToo much shared memory requested.\nRequested 34160 bytes, \
                       maximum 32768 bytes available.";
        set_level(Level::Info);
        log::warn!(target: "cubecl_runtime::logging::server", "{refusal} first");
        // The same words from another crate, and another failure from the
        // runtime, stay warnings.
        log::warn!(target: "eframe", "{refusal} elsewhere");
        log::warn!(
            target: "cubecl_runtime::logging::server",
            "A launch error happened\nCaused by:\n  the shader did not compile"
        );
        set_level(Level::Debug);
        log::warn!(target: "cubecl_runtime::logging::server", "{refusal} second");
        let text = fs::read_to_string(&path).unwrap();
        assert!(!text.contains("first"), "{text}");
        assert!(text.contains("elsewhere"), "{text}");
        assert!(text.contains("the shader did not compile"), "{text}");
        let detail = text
            .lines()
            .find(|line| line.contains("cubecl_runtime") && line.contains(" DEBUG "));
        assert!(detail.is_some(), "{text}");
        assert!(text.contains("second"), "{text}");
        assert!(super::refused_kernel("cubecl_runtime", refusal));
        assert!(!super::refused_kernel("cubecl_runtime", ""));
        assert!(!super::refused_kernel("", refusal));
    }

    #[test]
    fn levels_order_from_most_to_least_serious() {
        let _env = Env::new();
        assert!(Level::Error < Level::Warn);
        assert!(Level::Warn < Level::Info);
        assert!(Level::Info < Level::Debug);
        assert!(Level::Debug < Level::Trace);
        set_level(Level::Info);
        assert!(enabled(Level::Error) && enabled(Level::Warn) && enabled(Level::Info));
        assert!(!enabled(Level::Debug) && !enabled(Level::Trace));
        set_level(Level::Trace);
        assert_eq!(level(), Level::Trace);
        assert!(enabled(Level::Trace));
    }

    #[test]
    fn the_env_value_names_a_level_or_is_refused() {
        assert_eq!(level_from_env(None), Ok(Level::Info));
        assert_eq!(level_from_env(Some("")), Ok(Level::Info));
        assert_eq!(level_from_env(Some("  ")), Ok(Level::Info));
        assert_eq!(level_from_env(Some("DEBUG")), Ok(Level::Debug));
        assert_eq!(level_from_env(Some(" trace ")), Ok(Level::Trace));
        assert_eq!(level_from_env(Some("Warning")), Ok(Level::Warn));
        assert_eq!(level_from_env(Some("error")), Ok(Level::Error));
        assert_eq!(
            level_from_env(Some("loud")),
            Err(ParseLevelError {
                value: "loud".to_owned()
            })
        );
        assert!(
            ParseLevelError {
                value: "loud".to_owned()
            }
            .to_string()
            .contains("CUTLASS_LOG")
        );
    }

    #[test]
    fn export_joins_the_previous_file_and_the_current_one() {
        let env = Env::new();
        assert_eq!(directory(), None);
        let path = init_capture(&env.temp.path, MAX_FILE_BYTES).unwrap();
        assert_eq!(directory().as_deref(), path.parent());
        fs::write(env.temp.path.join(PREVIOUS_NAME), "older\n").unwrap();
        info!("newer-token");
        let out = env.temp.path.join("export").join("all.log");
        assert!(export(&env.temp.path, &out).is_err());
        fs::create_dir_all(out.parent().unwrap()).unwrap();
        export(&env.temp.path, &out).unwrap();
        let text = fs::read_to_string(&out).unwrap();
        assert!(text.starts_with("older\n"));
        assert!(text.ends_with(&fs::read_to_string(&path).unwrap()));
        assert!(text.contains("newer-token"));
        assert!(!out.with_file_name("all.log.partial").exists());
        let empty = Temp::new();
        export(&empty.path, &out).unwrap();
        assert!(fs::read(&out).unwrap().is_empty());
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
        assert!(text.contains(&format!("level {}", level())));
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
