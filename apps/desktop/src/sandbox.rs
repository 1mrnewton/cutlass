//! The local sandbox: each `sandbox.run` call is a fresh `cutlass sandbox`
//! child with no environment, an empty scratch folder, and ten seconds.
//! The script is Rhai, which has no file, socket or process calls; the
//! project folder and the home folder are refused by name before it runs.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

use cutlass_agent::{LocalSandbox, ProcessExecutor, SandboxPolicy};

/// How long one script may run before it is stopped.
pub const SCRIPT_LIMIT: Duration = Duration::from_secs(10);

/// The runner for a project at `project` (or none open), started from
/// `program` (the running `cutlass` binary in the app).
#[must_use]
pub fn local_sandbox(program: PathBuf, project: Option<&Path>) -> LocalSandbox<ProcessExecutor> {
    let text = |path: Option<PathBuf>| {
        path.map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    LocalSandbox::new(
        SandboxPolicy {
            project_path: text(project.map(Path::to_path_buf)),
            home_path: text(std::env::home_dir()),
        },
        ProcessExecutor::new(program, vec![OsString::from("sandbox")], SCRIPT_LIMIT),
    )
}

/// The running binary, which answers `cutlass sandbox`.
#[must_use]
pub fn this_program() -> PathBuf {
    std::env::current_exe().unwrap_or_else(|_| PathBuf::from("cutlass"))
}

/// `cutlass sandbox`: run the one call on stdin and answer on stdout.
pub fn serve() -> Result<(), String> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    cutlass_agent::serve_script(&mut stdin.lock(), &mut stdout.lock())
        .map_err(|error| error.to_string())
}
