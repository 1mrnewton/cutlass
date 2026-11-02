//! A child process is the console check. The parent reads its stdout, stderr, and log file.

use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_logger::{FILE_NAME, console_enabled};

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
