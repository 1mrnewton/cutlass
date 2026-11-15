//! Times the editor's three hot paths on a three-layer cut.
//!
//! `editor-bench` builds a draft of three H.264 movies stacked on the
//! canvas, a full-frame one and two smaller overlays, one of them half
//! transparent, and times:
//!
//! - **playback**: every frame of the cut in order, as the phone preview
//!   draws it (decode, eval, the GPU frame, the readback), in frames per
//!   second. The first frame opens the GPU and the files and is left out.
//! - **scrub**: 40 frames at scattered times, each a seek, as a scrub drags
//!   the playhead. The median, the 95th percentile and the worst, in
//!   milliseconds.
//! - **export**: the whole cut written as an H.264 MP4 at the canvas size,
//!   against real time. 2.0 means a 4-second cut took 2 seconds.
//!
//! The movies are made once with ffmpeg (`testsrc2` through libx264, a
//! keyframe every two seconds, so the decoder sees B-frames and motion like
//! a camera file's) and kept in `--work`. Nothing else uses ffmpeg.
//!
//! `--check <profile>` compares the results with that profile in
//! `targets.json` beside this crate and exits 1 when one misses.
//! `--json <path>` writes the results.
//!
//! ```text
//! cargo run --release -p editor-bench -- --sides 1080,2160 --check reference
//! ```

#![forbid(unsafe_code)]

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};

use cutlass_codec::{CodecId, ContainerId};
use cutlass_eval::NoFonts;
use cutlass_export::{ExportRequest, MovieSources, ProresProfile, write_movie};
use cutlass_ffi::Preview;
use cutlass_model::{
    Draft, FrameRate, MaterialVideo, Micros, Segment, TimeRange, Track, TrackFlag,
};
use cutlass_render::Renderer;
use cutlass_session::Session;
use serde_json::{Value, json};

/// A result line on stdout. A closed pipe is not an error worth stopping for.
macro_rules! say {
    ($($arg:tt)*) => {
        let _ = writeln!(std::io::stdout(), $($arg)*);
    };
}

const FPS: i64 = 30;
const SCRUB_FRAMES: usize = 40;

struct Args {
    sides: Vec<u32>,
    seconds: u32,
    work: PathBuf,
    check: Option<String>,
    json: Option<PathBuf>,
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(message) => {
            let _ = writeln!(std::io::stderr(), "editor-bench: {message}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool, String> {
    let args = parse(std::env::args().skip(1))?;
    std::fs::create_dir_all(&args.work).map_err(|err| err.to_string())?;
    let mut results = serde_json::Map::new();
    for side in &args.sides {
        let size = canvas(*side)?;
        let sources = sources(&args.work, size, args.seconds)?;
        let draft = three_layers(size, args.seconds, &sources);
        let session = Session::open(draft.clone());
        let playback = playback(&session, &args.work)?;
        let scrub = scrub(&session, &args.work, args.seconds)?;
        let export = export(&draft, &args.work, *side, args.seconds)?;
        say!(
            "{side}p three layers: playback {playback:.1} fps, scrub p50 {:.0} ms p95 {:.0} ms max {:.0} ms, export {export:.2}x real time",
            scrub.p50,
            scrub.p95,
            scrub.max
        );
        results.insert(
            side.to_string(),
            json!({
                "playback_fps": round(playback, 1),
                "scrub_p50_ms": round(scrub.p50, 1),
                "scrub_p95_ms": round(scrub.p95, 1),
                "scrub_max_ms": round(scrub.max, 1),
                "export_realtime": round(export, 2),
            }),
        );
    }
    let results = Value::Object(results);
    if let Some(path) = &args.json {
        let text = serde_json::to_string_pretty(&results).map_err(|err| err.to_string())?;
        std::fs::write(path, text).map_err(|err| err.to_string())?;
    }
    match &args.check {
        Some(profile) => check(&results, profile),
        None => Ok(true),
    }
}

fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut parsed = Args {
        sides: vec![1080],
        seconds: 4,
        work: std::env::temp_dir().join("cutlass-editor-bench"),
        check: None,
        json: None,
    };
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--sides" => {
                parsed.sides = value()?
                    .split(',')
                    .map(|side| side.parse().map_err(|_| format!("not a side: {side}")))
                    .collect::<Result<_, _>>()?;
            }
            "--seconds" => {
                parsed.seconds = value()?.parse().map_err(|_| "--seconds is a number")?;
            }
            "--work" => parsed.work = PathBuf::from(value()?),
            "--check" => parsed.check = Some(value()?),
            "--json" => parsed.json = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if parsed.seconds < 2 {
        return Err("--seconds is at least 2".to_owned());
    }
    Ok(parsed)
}

fn canvas(side: u32) -> Result<(u32, u32), String> {
    match side {
        720 => Ok((1280, 720)),
        1080 => Ok((1920, 1080)),
        2160 => Ok((3840, 2160)),
        _ => Err(format!("no canvas for side {side}; use 720, 1080 or 2160")),
    }
}

/// Three movies of the canvas size, made once.
fn sources(work: &Path, size: (u32, u32), seconds: u32) -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::new();
    for (index, pattern) in ["testsrc2", "testsrc2,hue=h=120", "testsrc2,negate"]
        .iter()
        .enumerate()
    {
        let path = work.join(format!(
            "source-{}x{}-{seconds}s-{index}.mp4",
            size.0, size.1
        ));
        if !path.is_file() {
            let (source, filter) = pattern.split_once(',').unwrap_or((pattern, "null"));
            let status = Command::new("ffmpeg")
                .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
                .arg(format!(
                    "{source}=size={}x{}:rate={FPS}:duration={seconds}",
                    size.0, size.1
                ))
                .args(["-vf", filter])
                .args([
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                ])
                .args(["-g", "60", "-b:v"])
                .arg(if size.1 > 1080 { "20M" } else { "8M" })
                .arg(&path)
                .status()
                .map_err(|err| format!("ffmpeg makes the source movies: {err}"))?;
            if !status.success() {
                return Err(format!("ffmpeg failed to make {}", path.display()));
            }
        }
        paths.push(path);
    }
    Ok(paths)
}

/// A full-frame movie, an overlay at 60 % in the top left, and a half
/// transparent one at 40 % in the bottom right.
fn three_layers(size: (u32, u32), seconds: u32, sources: &[PathBuf]) -> Draft {
    let mut draft = Draft::empty("bench");
    draft.canvas_config.width = i32::try_from(size.0).unwrap_or(i32::MAX);
    draft.canvas_config.height = i32::try_from(size.1).unwrap_or(i32::MAX);
    draft.fps = 30.0;
    let length = i64::from(seconds) * 1_000_000;
    draft.duration = Micros(length);
    let span = TimeRange::new(0, length);
    let places = [
        (1.0, 0.0, 0.0, 1.0),
        (0.6, -0.4, -0.4, 1.0),
        (0.4, 0.6, 0.6, 0.5),
    ];
    for (index, (path, (scale, x, y, alpha))) in sources.iter().zip(places).enumerate() {
        let id = format!("layer{index}");
        let mut video = MaterialVideo::new(id.as_str());
        video.path = path.display().to_string();
        video.width = draft.canvas_config.width;
        video.height = draft.canvas_config.height;
        video.duration = Micros(length);
        draft.materials.videos.push(video);
        let mut segment = Segment::new(id.as_str(), id.as_str(), span.clone());
        segment.source_range = Some(span.clone());
        segment.clip.scale.x = scale;
        segment.clip.scale.y = scale;
        segment.clip.transform.x = x;
        segment.clip.transform.y = y;
        segment.clip.alpha = alpha;
        segment.render_index = i32::try_from(index).unwrap_or(0);
        if index == 0 {
            draft.tracks[0].segments.push(segment);
        } else {
            let mut track = Track::main(id.as_str());
            track.flag = TrackFlag::SubVideo;
            track.segments.push(segment);
            draft.tracks.push(track);
        }
    }
    draft
}

/// The first microsecond of frame `index`, as export and the playhead step.
fn frame_time(index: i64) -> Micros {
    FrameRate::FPS_30
        .frame_start(index)
        .unwrap_or(Micros(index * 1_000_000 / FPS))
}

/// Frames per second over every frame after the first.
fn playback(session: &Session, root: &Path) -> Result<f64, String> {
    let mut preview = Preview::new();
    let frames = session.draft().duration.0 * FPS / 1_000_000;
    let mut last = draw(&mut preview, session, root, frame_time(0))?;
    let start = Instant::now();
    for index in 1..frames {
        let next = draw(&mut preview, session, root, frame_time(index))?;
        // The sources move every frame, so a repeated picture is a frame
        // that was not drawn again.
        if next == last {
            return Err(format!("frame {index} repeats the one before it"));
        }
        last = next;
    }
    let seconds = start.elapsed().as_secs_f64();
    #[allow(clippy::cast_precision_loss)]
    Ok((frames - 1) as f64 / seconds)
}

struct Scrub {
    p50: f64,
    p95: f64,
    max: f64,
}

/// Seeks to scattered times. The same times every run.
fn scrub(session: &Session, root: &Path, seconds: u32) -> Result<Scrub, String> {
    let mut preview = Preview::new();
    draw(&mut preview, session, root, frame_time(0))?;
    let frames = u64::from(seconds) * u64::try_from(FPS).unwrap_or(30);
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut took = Vec::with_capacity(SCRUB_FRAMES);
    for _ in 0..SCRUB_FRAMES {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let index = i64::try_from(state % frames).unwrap_or(0);
        let start = Instant::now();
        draw(&mut preview, session, root, frame_time(index))?;
        took.push(start.elapsed());
    }
    took.sort();
    let at = |fraction: f64| {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let index = ((took.len() - 1) as f64 * fraction).round() as usize;
        millis(took[index])
    };
    Ok(Scrub {
        p50: at(0.5),
        p95: at(0.95),
        max: at(1.0),
    })
}

/// Draws the frame at `time` and returns a hash of its pixels.
fn draw(
    preview: &mut Preview,
    session: &Session,
    root: &Path,
    time: Micros,
) -> Result<u64, String> {
    let picture = preview
        .frame(session, root, &NoFonts, time)
        .map_err(|err| format!("frame at {} us: {err}", time.0))?;
    if picture.rgba.iter().all(|byte| *byte == 0) {
        return Err(format!("frame at {} us drew nothing", time.0));
    }
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&picture.rgba, &mut hash);
    Ok(std::hash::Hasher::finish(&hash))
}

/// Seconds of cut written per second of wall time.
fn export(draft: &Draft, work: &Path, side: u32, seconds: u32) -> Result<f64, String> {
    let readers = cutlass_ffi::readers();
    let encoders = cutlass_ffi::encoders();
    let renderer = Renderer::headless().map_err(|err| err.to_string())?;
    let request = ExportRequest {
        range: TimeRange::new(0, draft.duration.0),
        short_side: side,
        codec: CodecId::H264,
        bitrate: cutlass_export::video_bitrate(side),
        frame_rate: FrameRate::FPS_30,
        container: ContainerId::Isobmff,
        audio: None,
        sidecar: None,
        late_by: 0,
        constant_rate: false,
        prores: ProresProfile::Standard,
    };
    let destination = work.join(format!("export-{side}.mp4"));
    let start = Instant::now();
    let result = write_movie(
        &MovieSources {
            draft,
            root: Some(work),
            readers: &readers,
            fonts: &NoFonts,
            renderer: Some(&renderer),
            encoders: &encoders,
        },
        &request,
        &destination,
        &mut |_, _| true,
    )
    .map_err(|err| format!("export: {err}"))?;
    let wall = start.elapsed().as_secs_f64();
    let expected = usize::try_from(i64::from(seconds) * FPS).unwrap_or(0);
    if result.frames.len() != expected {
        return Err(format!(
            "export wrote {} frames, not {expected}",
            result.frames.len()
        ));
    }
    let _ = std::fs::remove_file(&destination);
    Ok(f64::from(seconds) / wall)
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn round(value: f64, places: i32) -> f64 {
    let scale = 10f64.powi(places);
    (value * scale).round() / scale
}

/// Whether every number in `results` meets `profile` in `targets.json`.
/// A `_fps` or `_realtime` target is a floor; a `_ms` target is a ceiling.
fn check(results: &Value, profile: &str) -> Result<bool, String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("targets.json");
    let text = std::fs::read_to_string(&path).map_err(|err| err.to_string())?;
    let targets: Value = serde_json::from_str(&text).map_err(|err| err.to_string())?;
    let Some(sides) = targets.get(profile).and_then(|entry| entry.get("sides")) else {
        return Err(format!("no profile {profile} in {}", path.display()));
    };
    let mut met = true;
    let Some(results) = results.as_object() else {
        return Err("results are not an object".to_owned());
    };
    for (side, measured) in results {
        let Some(wanted) = sides.get(side).and_then(Value::as_object) else {
            say!("{side}p: no {profile} target");
            continue;
        };
        for (name, target) in wanted {
            let (Some(target), Some(got)) =
                (target.as_f64(), measured.get(name).and_then(Value::as_f64))
            else {
                return Err(format!("{side}p {name} is not a number"));
            };
            let ceiling = name.ends_with("_ms");
            let ok = if ceiling {
                got <= target
            } else {
                got >= target
            };
            let sign = if ceiling { "<=" } else { ">=" };
            say!(
                "{} {side}p {name} {got} (target {sign} {target})",
                if ok { "ok  " } else { "MISS" }
            );
            met &= ok;
        }
    }
    Ok(met)
}
