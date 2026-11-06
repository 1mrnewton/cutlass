//! Runs the bench from a shell. See the library docs for the flags.

#![forbid(unsafe_code)]

use std::io::Write;

fn main() -> std::process::ExitCode {
    match ml_bench::run(std::env::args().skip(1)) {
        Ok(report) => match writeln!(std::io::stdout().lock(), "{report}") {
            Ok(()) => std::process::ExitCode::SUCCESS,
            Err(_) => std::process::ExitCode::FAILURE,
        },
        Err(error) => {
            // The bench is a standalone tool with no logger; the error is its output.
            let _ = writeln!(std::io::stderr().lock(), "ml-bench: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
