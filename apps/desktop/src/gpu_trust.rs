//! Whether the vision models may use this machine's GPU.
//!
//! The Mac's Metal path is trusted from the start. On Windows and Linux the
//! models use Vulkan only once [`cutlass_vision::check_gpu`] has passed on
//! this machine, and the check runs in a process of its own
//! (`cutlass gpu-check`): a broken driver can abort the process that opens
//! it, and that must not be the editor. A job that loads a model asks
//! [`compute`], which waits for a check still running.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use cutlass_vision::{Compute, GpuCheck};

/// The argument that runs the check instead of the editor.
pub const CHECK_ARGUMENT: &str = "gpu-check";

/// What a passing check prints: the GPU gives the answers in 32-bit floats,
/// and in 16-bit ones too.
const PASSED: &str = "gpu passed";
const PASSED_HALF: &str = "gpu passed half";

/// Longest the check may take: a first launch compiles its kernels, which
/// is slow on a weak GPU.
const LONGEST: Duration = Duration::from_secs(240);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// No check was started: the GPU is as the vision crate starts it.
    Idle,
    Running,
    Done,
}

static STATE: Mutex<State> = Mutex::new(State::Idle);
static SETTLED: Condvar = Condvar::new();

/// Start checking this machine's GPU in a process of its own, when the
/// build has a GPU path that is not trusted from the start. Call it once,
/// early; a later call does nothing.
pub fn start_check() {
    if !cutlass_vision::gpu_available() || cutlass_vision::gpu_trusted() {
        return;
    }
    let Ok(mut state) = STATE.lock() else {
        return;
    };
    if *state != State::Idle {
        return;
    }
    *state = State::Running;
    drop(state);
    let started = std::thread::Builder::new()
        .name("gpu-check".into())
        .spawn(|| {
            let found = run_check();
            match &found {
                Ok(check) => cutlass_logger::info!(
                    "the GPU passed the vision check; 16-bit floats: {}",
                    check.half
                ),
                Err(reason) => {
                    cutlass_logger::warn!("vision models stay on the CPU: {reason}");
                }
            }
            cutlass_vision::trust_gpu(found.ok());
            settle();
        });
    if started.is_err() {
        cutlass_logger::warn!("vision models stay on the CPU: the GPU check could not start");
        settle();
    }
}

fn settle() {
    if let Ok(mut state) = STATE.lock() {
        *state = State::Done;
    }
    SETTLED.notify_all();
}

/// Where a model runs: the GPU once it is trusted, waiting for a check that
/// is still running. Called from a job, never the window's thread.
#[must_use]
pub fn compute() -> Compute {
    if let Ok(state) = STATE.lock() {
        let _ = SETTLED.wait_timeout_while(state, LONGEST, |state| *state == State::Running);
    }
    Compute::best()
}

/// Run `cutlass gpu-check` and read its answer.
fn run_check() -> Result<GpuCheck, String> {
    let program = std::env::current_exe().map_err(|error| format!("no program path: {error}"))?;
    let mut child = Command::new(program)
        .arg(CHECK_ARGUMENT)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("the check did not start: {error}"))?;
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() > LONGEST => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("the check took over {} s", LONGEST.as_secs()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(error) => return Err(format!("the check was lost: {error}")),
        }
    };
    let mut said = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut said);
    }
    let mut complaint = String::new();
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_string(&mut complaint);
    }
    if !status.success() {
        let complaint = complaint.trim();
        return Err(if complaint.is_empty() {
            format!("the check failed ({status})")
        } else {
            format!("the check failed ({status}): {complaint}")
        });
    }
    answer(&said).ok_or_else(|| format!("the check said {:?}", said.trim()))
}

/// What a check that exited cleanly found, from its output.
fn answer(said: &str) -> Option<GpuCheck> {
    match said.trim() {
        PASSED_HALF => Some(GpuCheck { half: true }),
        PASSED => Some(GpuCheck { half: false }),
        _ => None,
    }
}

/// `cutlass gpu-check`: run the check on this process's GPU, keeping its
/// compiled kernels in `kernels`, and say whether it passed. A driver that
/// aborts ends this process, which the editor reads as a failure.
pub fn serve(kernels: &Path) -> Result<String, String> {
    if let Err(error) = cutlass_vision::keep_kernels_in(kernels) {
        cutlass_logger::warn!("kernel cache: {error}");
    }
    let check = cutlass_vision::check_gpu().map_err(|error| error.to_string())?;
    Ok(if check.half { PASSED_HALF } else { PASSED }.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_clean_pass_is_read_as_one() {
        assert_eq!(answer("gpu passed half\n"), Some(GpuCheck { half: true }));
        assert_eq!(answer("gpu passed"), Some(GpuCheck { half: false }));
        assert_eq!(answer(""), None);
        assert_eq!(answer("gpu passed maybe"), None);
    }

    #[test]
    fn with_no_check_started_a_job_does_not_wait() {
        let started = Instant::now();
        let compute = compute();
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(compute, Compute::best());
    }
}
