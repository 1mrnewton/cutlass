#![forbid(unsafe_code)]

fn main() {
    let _opened = cutlass::boot(None);
    if let Err(error) = cutlass::window::run() {
        cutlass::report_error(&error.to_string());
    }
}
