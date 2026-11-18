//! The rustc wrapper `scripts/pgo.py` hands Cargo as
//! `RUSTC_WORKSPACE_WRAPPER`, so `-Cprofile-generate` and `-Cprofile-use`
//! reach the workspace's own crates and no dependency.
//!
//! Given in RUSTFLAGS, the profile reached every crate, and a dependency
//! with no counts in it could take far longer to optimise: on Windows,
//! `pliron` (under burn's Vulkan backend) compiled in about a minute
//! without it and ran past three hours with it. The profile only holds the
//! counts of workspace crates (the H.264 codec, and the AV1 decoder's from
//! `PGO_MERGE`), so no dependency loses anything.
//!
//! Cargo runs `pgo_rustc <rustc> <args...>`; this runs
//! `<rustc> <args...> $PGO_RUSTC_FLAG` and exits as it did. It is compiled
//! by `pgo.py` with one rustc call and has no dependencies.

use std::env;
use std::process::{exit, Command};

fn main() {
    let mut args = env::args_os().skip(1);
    let Some(rustc) = args.next() else {
        eprintln!("pgo_rustc: run by Cargo as RUSTC_WORKSPACE_WRAPPER, with rustc first");
        exit(2);
    };
    let mut command = Command::new(rustc);
    command.args(args);
    if let Some(flag) = env::var_os("PGO_RUSTC_FLAG") {
        command.arg(flag);
    }
    match command.status() {
        Ok(status) => exit(status.code().unwrap_or(1)),
        Err(error) => {
            eprintln!("pgo_rustc: could not run rustc: {error}");
            exit(1);
        }
    }
}
