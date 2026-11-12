//! CAF open, sequential read and seek against libavformat, compared by
//! retired instructions.
//!
//! `cargo bench -p cutlass-caf` makes 20 minutes of 48 kHz stereo sound
//! with ffmpeg, writes it as AAC, ALAC, Opus and 16-bit PCM CAF with
//! afconvert (ffmpeg's FLAC and ALAC where afconvert is missing), and
//! builds `benches/ffmpeg_caf.c` against the FFmpeg pkg-config finds. Both
//! sides then do the same three jobs in one process each, `repeat` times:
//! open (header and packet table, no packet read), read (open and every
//! packet, dropped) and seek (open once, then seeks to the same
//! pseudo-random times, each followed by one packet read). Before anything
//! is timed, both sides must report the same packets, bytes and landings.
//!
//! Each job runs under `/usr/bin/time -l` with `repeat` and with 0, five
//! rounds taking turns; the median of the zero runs (process start-up,
//! library loading) is taken off the median of the full runs, so neither
//! side is charged for starting. `CAF_BENCH_WORK=<folder>` keeps the files.
//! The bench prints `skip:` and exits 0 when a tool is missing.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_caf::CafDemux;
use cutlass_container::{Demux, Landing, SeekTo};

const ROUNDS: usize = 5;
const MINUTES: u32 = 20;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--side") {
        return rust_side(&args[1..]);
    }
    if !cfg!(target_os = "macos") {
        println!("skip: instruction counts come from macOS /usr/bin/time -l");
        return Ok(());
    }
    for tool in ["ffmpeg", "cc", "pkg-config"] {
        if !on_path(tool) {
            println!("skip: {tool} is not on PATH");
            return Ok(());
        }
    }
    let work = std::env::var_os("CAF_BENCH_WORK").map_or_else(
        || std::env::temp_dir().join("cutlass-caf-bench"),
        PathBuf::from,
    );
    std::fs::create_dir_all(&work).map_err(|error| error.to_string())?;
    let harness = build_harness(&work)?;
    let files = make_inputs(&work)?;
    let me = std::env::current_exe().map_err(|error| error.to_string())?;
    println!(
        "| File | Job | Repeat | Cutlass instructions per job | libavformat instructions per job | Ratio | Cutlass peak RSS | libavformat peak RSS |"
    );
    println!("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |");
    for (name, file) in &files {
        let range = valid_frames(file)?;
        for (job, repeat) in [("open", 2_000_u64), ("read", 20), ("seek", 20_000)] {
            let ours = |repeat: u64| -> Vec<String> {
                vec![
                    me.display().to_string(),
                    "--side".into(),
                    job.into(),
                    file.display().to_string(),
                    repeat.to_string(),
                    range.to_string(),
                ]
            };
            let theirs = |repeat: u64| -> Vec<String> {
                vec![
                    harness.display().to_string(),
                    job.into(),
                    file.display().to_string(),
                    repeat.to_string(),
                    range.to_string(),
                ]
            };
            let a = run(&ours(1))?.stdout;
            let b = run(&theirs(1))?.stdout;
            if job != "open" && a != b {
                return Err(format!(
                    "{name} {job}: Cutlass says {a:?}, libavformat {b:?}"
                ));
            }
            let mut rows: [Vec<Run>; 4] = Default::default();
            for _ in 0..ROUNDS {
                rows[0].push(run(&ours(repeat))?);
                rows[1].push(run(&ours(0))?);
                rows[2].push(run(&theirs(repeat))?);
                rows[3].push(run(&theirs(0))?);
            }
            let net = |full: &[Run], empty: &[Run]| {
                let full = median(full.iter().map(|run| run.instructions).collect());
                let empty = median(empty.iter().map(|run| run.instructions).collect());
                full.saturating_sub(empty) / repeat
            };
            let ours_per = net(&rows[0], &rows[1]);
            let theirs_per = net(&rows[2], &rows[3]);
            let rss = |runs: &[Run]| median(runs.iter().map(|run| run.rss).collect());
            println!(
                "| {name} | {job} | {repeat} | {ours_per} | {theirs_per} | {:.2} | {:.1} MB | {:.1} MB |",
                ours_per as f64 / theirs_per.max(1) as f64,
                rss(&rows[0]) as f64 / 1e6,
                rss(&rows[2]) as f64 / 1e6,
            );
        }
    }
    if std::env::var_os("CAF_BENCH_WORK").is_none() {
        let _ = std::fs::remove_dir_all(&work);
    }
    Ok(())
}

fn on_path(tool: &str) -> bool {
    Command::new("which")
        .arg(tool)
        .output()
        .is_ok_and(|out| out.status.success())
}

fn shell(program: &str, args: &[&str]) -> Result<String, String> {
    let out = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("{program}: {error}"))?;
    if !out.status.success() {
        return Err(format!(
            "{program} {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn build_harness(work: &Path) -> Result<PathBuf, String> {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("benches/ffmpeg_caf.c");
    let out = work.join("ffmpeg_caf");
    let flags = shell(
        "pkg-config",
        &[
            "--cflags",
            "--libs",
            "libavformat",
            "libavcodec",
            "libavutil",
        ],
    )?;
    let mut args = vec![
        "-O2",
        "-o",
        out.to_str().ok_or("path")?,
        source.to_str().ok_or("path")?,
    ];
    args.extend(flags.split_whitespace());
    shell("cc", &args)?;
    Ok(out)
}

fn make_inputs(work: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let wav = work.join("source.wav");
    if !wav.exists() {
        let seconds = (MINUTES * 60).to_string();
        let graph = format!(
            "sine=f=330:r=48000:d={seconds},aformat=channel_layouts=mono[a];\
             anoisesrc=r=48000:a=0.05:d={seconds}[b];[a][b]amerge=inputs=2"
        );
        shell(
            "ffmpeg",
            &[
                "-v",
                "error",
                "-y",
                "-filter_complex",
                &graph,
                "-c:a",
                "pcm_s16le",
                wav.to_str().ok_or("path")?,
            ],
        )?;
    }
    let source = wav.to_str().ok_or("path")?;
    let mut files = Vec::new();
    let apple = on_path("afconvert");
    let jobs: Vec<(&str, &str, Vec<&str>)> = if apple {
        vec![
            (
                "AAC",
                "afconvert",
                vec!["-f", "caff", "-d", "aac", "-b", "128000"],
            ),
            ("ALAC", "afconvert", vec!["-f", "caff", "-d", "alac"]),
            ("Opus", "afconvert", vec!["-f", "caff", "-d", "opus"]),
            ("PCM s16", "afconvert", vec!["-f", "caff", "-d", "LEI16"]),
        ]
    } else {
        vec![
            ("ALAC", "ffmpeg", vec!["-c:a", "alac"]),
            ("FLAC", "ffmpeg", vec!["-c:a", "flac"]),
            ("PCM s16", "ffmpeg", vec!["-c:a", "pcm_s16le"]),
        ]
    };
    for (name, tool, options) in jobs {
        let out = work.join(format!("{}.caf", name.replace(' ', "-")));
        if !out.exists() {
            let target = out.to_str().ok_or("path")?;
            let mut args: Vec<&str> = Vec::new();
            if tool == "ffmpeg" {
                args.extend(["-v", "error", "-y", "-i", source]);
                args.extend(&options);
                args.push(target);
            } else {
                args.extend(&options);
                args.extend([source, target]);
            }
            shell(tool, &args)?;
        }
        files.push((name.to_owned(), out));
    }
    Ok(files)
}

/// Frames a seek may aim at, counted from the first packet on both sides:
/// every coded frame but the remainder at the end.
fn valid_frames(file: &Path) -> Result<u64, String> {
    let demux = open(file)?;
    let total = demux.total_frames().ok_or("no frame count")?;
    let remainder = demux
        .packet_table()
        .map_or(0, |header| u64::from(header.remainder_frames));
    Ok(total - remainder)
}

struct Run {
    stdout: String,
    instructions: u64,
    rss: u64,
}

fn run(command: &[String]) -> Result<Run, String> {
    let out = Command::new("/usr/bin/time")
        .arg("-l")
        .args(command)
        .output()
        .map_err(|error| error.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "{}: {}",
            command.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let report = String::from_utf8_lossy(&out.stderr);
    let field = |name: &str| -> u64 {
        report
            .lines()
            .find(|line| line.trim_end().ends_with(name))
            .and_then(|line| line.split_whitespace().next())
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    };
    Ok(Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        instructions: field("instructions retired"),
        rss: field("maximum resident set size"),
    })
}

fn median(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn open(file: &Path) -> Result<CafDemux<File>, String> {
    let handle = File::open(file).map_err(|error| error.to_string())?;
    let length = handle.metadata().map_err(|error| error.to_string())?.len();
    CafDemux::open(handle, length).map_err(|error| error.to_string())
}

/// The same xorshift as `ffmpeg_caf.c`.
struct Times(u64);

impl Times {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// One job, `repeat` times, printing what the C harness prints.
fn rust_side(args: &[String]) -> Result<(), String> {
    let [job, file, repeat, range] = args else {
        return Err("--side <open|read|seek> <file> <repeat> <range>".into());
    };
    let file = Path::new(file);
    let repeat: u64 = repeat.parse().map_err(|_| "repeat")?;
    let range: u64 = range.parse().map_err(|_| "range")?;
    let (mut packets, mut bytes, mut check) = (0_u64, 0_u64, 0_i64);
    match job.as_str() {
        "open" | "read" => {
            for _ in 0..repeat {
                let mut demux = open(file)?;
                packets = 0;
                bytes = 0;
                if job == "open" {
                    continue;
                }
                while let Some(packet) = demux.read_packet().map_err(|error| error.to_string())? {
                    packets += 1;
                    bytes += packet.bytes().len() as u64;
                    check += i64::from(packet.bytes()[0]);
                }
            }
        }
        "seek" => {
            let mut demux = open(file)?;
            let mut times = Times(0x5EED_CAFE);
            for _ in 0..repeat {
                let target = i64::try_from(times.next() % range).map_err(|_| "time")?;
                let landing = demux
                    .seek(SeekTo {
                        track: 0,
                        time: target,
                        end: None,
                    })
                    .map_err(|error| error.to_string())?;
                if landing.landing == Landing::End {
                    return Err(format!("seek to {target} landed at the end"));
                }
                let packet = demux
                    .read_packet()
                    .map_err(|error| error.to_string())?
                    .ok_or("no packet after a seek")?;
                packets += 1;
                bytes += packet.bytes().len() as u64;
                check += packet.timestamp().presentation().unwrap_or(0);
            }
        }
        other => return Err(format!("no job {other}")),
    }
    println!("packets {packets} bytes {bytes} check {check}");
    Ok(())
}
