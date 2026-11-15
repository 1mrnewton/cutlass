#![forbid(unsafe_code)]

use std::process::ExitCode;

use cutlass::cli::Command;

fn main() -> ExitCode {
    let command = cutlass::cli::parse(std::env::args_os().skip(1));
    match &command {
        // Answered on the terminal: no log, no window, no project.
        Command::Help => return say(&cutlass::cli::usage()),
        Command::Version => return say(&cutlass::cli::version()),
        Command::Unknown(option) => {
            let _ = std::io::Write::write_all(
                &mut std::io::stderr(),
                format!("cutlass: no option {option}\n\n{}", cutlass::cli::usage()).as_bytes(),
            );
            return ExitCode::from(2);
        }
        Command::Sandbox => {
            // One agent script, from `sandbox.run`: no log, no window, no
            // project, only stdin and stdout.
            return match cutlass::sandbox::serve() {
                Ok(()) => ExitCode::SUCCESS,
                Err(_) => ExitCode::FAILURE,
            };
        }
        Command::Window | Command::Studio | Command::Mcp(_) | Command::New(_) => {}
    }
    let _opened = cutlass::boot(None);
    let opened = match command {
        Command::Mcp(path) => {
            return match cutlass::mcp::run(path.as_deref()) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => fail("cutlass mcp", &error),
            };
        }
        // `cutlass new <folder>`: write an empty project under the folder and
        // print where it went, for scripts and the CI smoke test.
        Command::New(None) => {
            return fail(
                "cutlass new",
                "name the folder: cutlass new /path/to/folder",
            );
        }
        Command::New(Some(root)) => {
            return match cutlass::shell::Shell::create_project(&root) {
                Ok(folder) => say(&format!("{}\n", folder.display())),
                Err(error) => fail("cutlass new", &error.to_string()),
            };
        }
        // `cutlass studio`: the window, open on the effect studio.
        Command::Studio => cutlass::window::run_studio(),
        _ => cutlass::window::run(),
    };
    if let Err(error) = opened {
        cutlass::report_error(&error.to_string());
    }
    ExitCode::SUCCESS
}

/// Write `text` on stdout and give the passing status.
fn say(text: &str) -> ExitCode {
    let _ = std::io::Write::write_all(&mut std::io::stdout(), text.as_bytes());
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
