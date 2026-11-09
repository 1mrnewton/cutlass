//! The process the console test runs. It is not a user command.
//!
//! With a directory argument it opens the log there and writes one error
//! and one debug record. Stdout stays empty. A second argument of `panic`
//! then panics on the main thread; `thread-panic` panics on a thread named
//! `worker` and writes `after` once that thread is joined. The panic is an
//! index past the end of an empty list, as the lints refuse `panic!`.

fn main() {
    let mut args = std::env::args();
    let _program = args.next();
    let Some(directory) = args.next() else {
        return;
    };
    if cutlass_logger::init(std::path::Path::new(&directory)).is_err() {
        return;
    }
    cutlass_logger::install_panic_hook();
    cutlass_logger::install_panic_hook();
    cutlass_logger::error!("child");
    cutlass_logger::debug!("child-debug");
    match args.next().as_deref() {
        Some("panic") => out_of_bounds(args.len()),
        Some("thread-panic") => {
            let worker = std::thread::Builder::new()
                .name("worker".to_owned())
                .spawn(move || out_of_bounds(7));
            if let Ok(worker) = worker {
                let _ = worker.join();
            }
            cutlass_logger::info!("after");
        }
        _ => {}
    }
}

fn out_of_bounds(index: usize) {
    let empty: Vec<u8> = Vec::new();
    let _ = empty[index];
}
