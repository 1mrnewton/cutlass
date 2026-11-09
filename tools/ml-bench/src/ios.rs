//! The iPhone entry point. An app has no shell to pass flags, so this runs
//! a fixed set: both reference frames, both engines, on the CPU and on the
//! GPU. The models and reference files ship in the app bundle. Each report
//! is printed to the console and written under the app's `Documents/`.
//!
//! `ML_BENCH_ONLY=<name>` in the launch environment runs one entry of the
//! plan, and `ML_BENCH_ENGINE=candle|burn` runs one engine, so a crash in
//! one does not hide the others.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::BenchError;

/// Name, reference file, GPU, encodes, clicks.
const PLAN: [(&str, &str, bool, &str, &str); 4] = [
    ("gpu_landscape", "reference_landscape", true, "10", "30"),
    ("gpu_portrait", "reference_portrait", true, "10", "30"),
    ("cpu_landscape", "reference_landscape", false, "3", "10"),
    ("cpu_portrait", "reference_portrait", false, "3", "10"),
];

/// Runs the plan. Returns 0 when every run finished, 1 otherwise.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn ml_bench_run_ios() -> i32 {
    match run_plan() {
        Ok(()) => 0,
        Err(error) => {
            let _ = writeln!(std::io::stdout().lock(), "ml-bench: {error}");
            1
        }
    }
}

fn run_plan() -> Result<(), BenchError> {
    let io = |error: std::io::Error| BenchError::Io(error.to_string());
    let exe = std::env::current_exe().map_err(io)?;
    let bundle = exe
        .parent()
        .ok_or_else(|| BenchError::Io("the app has no bundle folder".into()))?;
    let home = std::env::var_os("HOME").ok_or_else(|| BenchError::Io("HOME is not set".into()))?;
    let documents = Path::new(&home).join("Documents");
    let only = std::env::var("ML_BENCH_ONLY").ok();
    for (name, reference, gpu, runs, clicks) in PLAN {
        if only.as_deref().is_some_and(|only| only != name) {
            continue;
        }
        let reference: PathBuf = bundle.join(format!("{reference}.safetensors"));
        let mut args = vec![
            "--reference".to_owned(),
            reference.display().to_string(),
            "--models".to_owned(),
            bundle.display().to_string(),
            "--out".to_owned(),
            documents.join(name).display().to_string(),
            "--runs".to_owned(),
            runs.to_owned(),
            "--clicks".to_owned(),
            clicks.to_owned(),
        ];
        if gpu {
            args.push("--gpu".to_owned());
        }
        if let Ok(engine) = std::env::var("ML_BENCH_ENGINE") {
            args.extend(["--engine".to_owned(), engine]);
        }
        let report = crate::run(args)?;
        let _ = writeln!(
            std::io::stdout().lock(),
            "=== ml-bench {name}\n{report}\n=== end {name}"
        );
    }
    Ok(())
}
