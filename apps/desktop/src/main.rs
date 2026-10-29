#![forbid(unsafe_code)]

fn main() {
    if let Err(error) = cutlass::window::run() {
        #[allow(clippy::print_stderr)]
        {
            eprintln!("cutlass: {error}");
        }
    }
}
