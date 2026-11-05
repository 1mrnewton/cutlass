#![forbid(unsafe_code)]

fn main() {
    let _opened = cutlass::boot(None);
    let mut args = std::env::args_os().skip(1);
    if args.next().is_some_and(|first| first == "mcp") {
        let path = args.next().map(std::path::PathBuf::from);
        if let Err(error) = cutlass::mcp::run(path.as_deref()) {
            cutlass::report_error(&error);
            let _ = std::io::Write::write_all(
                &mut std::io::stderr(),
                format!("cutlass mcp: {error}\n").as_bytes(),
            );
            std::process::exit(1);
        }
        return;
    }
    if let Err(error) = cutlass::window::run() {
        cutlass::report_error(&error.to_string());
    }
}
