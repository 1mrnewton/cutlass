#![forbid(unsafe_code)]

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let first = args.next();
    if first.as_ref().is_some_and(|first| first == "sandbox") {
        // One agent script, from `sandbox.run`: no log, no window, no
        // project, only stdin and stdout.
        return match cutlass::sandbox::serve() {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ExitCode::FAILURE,
        };
    }
    let _opened = cutlass::boot(None);
    if first.as_ref().is_some_and(|first| first == "mcp") {
        let path = args.next().map(std::path::PathBuf::from);
        return match cutlass::mcp::run(path.as_deref()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => fail("cutlass mcp", &error),
        };
    }
    if first.as_ref().is_some_and(|first| first == "new") {
        // `cutlass new <folder>`: write an empty project under the folder and
        // print where it went, for scripts and the CI smoke test.
        let Some(root) = args.next().map(std::path::PathBuf::from) else {
            return fail(
                "cutlass new",
                "name the folder: cutlass new /path/to/folder",
            );
        };
        return match cutlass::shell::Shell::create_project(&root) {
            Ok(folder) => {
                let _ = std::io::Write::write_all(
                    &mut std::io::stdout(),
                    format!("{}\n", folder.display()).as_bytes(),
                );
                ExitCode::SUCCESS
            }
            Err(error) => fail("cutlass new", &error.to_string()),
        };
    }
    // `cutlass studio`: the window, open on the effect studio.
    let opened = if first.as_ref().is_some_and(|first| first == "studio") {
        cutlass::window::run_studio()
    } else {
        cutlass::window::run()
    };
    if let Err(error) = opened {
        cutlass::report_error(&error.to_string());
    }
    ExitCode::SUCCESS
}

/// Log `error`, say it on stderr, and give the failing status.
fn fail(command: &str, error: &str) -> ExitCode {
    cutlass::report_error(error);
    let _ = std::io::Write::write_all(
        &mut std::io::stderr(),
        format!("{command}: {error}\n").as_bytes(),
    );
    ExitCode::FAILURE
}
