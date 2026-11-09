//! A child process is the console check. The parent reads its stdout, stderr, and log file.

use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_logger::{FILE_NAME, LEVEL_ENV, console_enabled};

struct Temp {
    path: PathBuf,
}

impl Temp {
    fn try_new() -> io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("cutlass-logger-child-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path)?;
        Ok(Self { path })
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[test]
fn the_child_writes_the_file_and_stderr_only_outside_a_public_build() -> io::Result<()> {
    let dir = Temp::try_new()?;
    let output = Command::new(env!("CARGO_BIN_EXE_emit"))
        .arg(&dir.path)
        .env_remove(LEVEL_ENV)
        .output()?;
    assert!(output.status.success(), "{output:?}");
    assert!(
        output.stdout.is_empty(),
        "stdout {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let file = fs::read(dir.path.join(FILE_NAME))?;
    let text = String::from_utf8_lossy(&file);
    assert_eq!(text.as_bytes(), file.as_slice(), "log file is not utf-8");
    assert!(text.contains("child"), "{text}");
    assert!(text.contains("ERROR"), "{text}");
    assert!(text.contains("log file"), "{text}");
    assert_eq!(console_enabled(), cfg!(not(feature = "public-release")));
    if console_enabled() {
        assert_eq!(output.stderr, file);
    } else {
        assert!(
            output.stderr.is_empty(),
            "stderr {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

/// Run the child with `CUTLASS_LOG` set to `value` and give its log text.
fn child_log(value: &str) -> io::Result<String> {
    let dir = Temp::try_new()?;
    let output = Command::new(env!("CARGO_BIN_EXE_emit"))
        .arg(&dir.path)
        .env(LEVEL_ENV, value)
        .output()?;
    assert!(output.status.success(), "{output:?}");
    assert!(output.stdout.is_empty());
    fs::read_to_string(dir.path.join(FILE_NAME))
}

#[test]
fn the_env_level_decides_what_the_child_keeps() -> io::Result<()> {
    let info = child_log("info")?;
    assert!(info.contains("child"), "{info}");
    assert!(!info.contains("child-debug"), "{info}");
    assert!(info.contains("level INFO"), "{info}");
    let debug = child_log("DEBUG")?;
    assert!(debug.contains(" DEBUG "), "{debug}");
    assert!(debug.contains("child-debug"), "{debug}");
    assert!(debug.contains("level DEBUG"), "{debug}");
    let error = child_log("error")?;
    assert!(error.contains("child"), "{error}");
    assert!(!error.contains("log file"), "{error}");
    Ok(())
}

#[test]
fn an_unknown_env_level_keeps_info_and_says_so() -> io::Result<()> {
    let text = child_log("loud")?;
    assert!(text.contains("level INFO"), "{text}");
    assert!(text.contains(" WARN "), "{text}");
    assert!(text.contains("CUTLASS_LOG"), "{text}");
    assert!(!text.contains("child-debug"), "{text}");
    Ok(())
}

#[test]
fn a_panic_is_logged_before_the_process_dies() -> io::Result<()> {
    let dir = Temp::try_new()?;
    let output = Command::new(env!("CARGO_BIN_EXE_emit"))
        .arg(&dir.path)
        .arg("panic")
        .env_remove(LEVEL_ENV)
        .output()?;
    assert!(!output.status.success(), "{output:?}");
    assert!(output.stdout.is_empty());
    let text = fs::read_to_string(dir.path.join(FILE_NAME))?;
    let line = text
        .lines()
        .find(|line| line.contains("index out of bounds"))
        .unwrap_or_default();
    assert!(line.contains(" ERROR "), "{text}");
    assert!(line.contains(" panic "), "{text}");
    assert!(line.contains("thread main panicked at"), "{text}");
    assert!(line.contains("emit.rs:"), "{text}");
    assert!(line.contains("backtrace:"), "{text}");
    // The child installs the hook twice; the panic is still one record.
    assert_eq!(text.matches("index out of bounds").count(), 1, "{text}");
    // The hook set before still runs, so the panic still reaches stderr.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("panicked at"), "{stderr}");
    Ok(())
}

#[test]
fn a_panic_on_a_named_thread_is_logged_and_the_process_goes_on() -> io::Result<()> {
    let dir = Temp::try_new()?;
    let output = Command::new(env!("CARGO_BIN_EXE_emit"))
        .arg(&dir.path)
        .arg("thread-panic")
        .env_remove(LEVEL_ENV)
        .output()?;
    assert!(output.status.success(), "{output:?}");
    let text = fs::read_to_string(dir.path.join(FILE_NAME))?;
    assert!(
        text.contains("thread worker panicked at") && text.contains("the index is 7"),
        "{text}"
    );
    let panic_at = text.find("the index is 7").unwrap_or(usize::MAX);
    let after_at = text.find(" after").unwrap_or(0);
    assert!(panic_at < after_at, "{text}");
    Ok(())
}
