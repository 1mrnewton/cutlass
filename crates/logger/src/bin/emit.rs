//! The process the console test runs. It is not a user command.
//!
//! With a directory argument it opens the log there and writes one error.
//! Stdout stays empty.

fn main() {
    let mut args = std::env::args();
    let _program = args.next();
    let Some(directory) = args.next() else {
        return;
    };
    if cutlass_logger::init(std::path::Path::new(&directory)).is_ok() {
        cutlass_logger::error!("child");
    }
}
