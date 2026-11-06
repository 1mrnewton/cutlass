//! Times MobileSAM, a click-to-select segmentation model, on candle and on
//! burn, on this machine's CPU and GPU, and checks both give the same masks.
//!
//! ```text
//! ml-bench --image frame.png --click 1010,650 [--engine both] [--gpu]
//! ml-bench --reference reference_landscape.safetensors [--engine both] [--gpu]
//! ```
//!
//! The report goes to stdout and to `<out>/report.json`. Each engine's best
//! mask goes to `<out>/<engine>_mask.png`.

#![forbid(unsafe_code)]

mod burn_engine;
mod candle_engine;
mod compare;
mod engine;
mod frame;
mod stats;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use safetensors::SafeTensors;
use serde_json::{Value, json};

use crate::engine::{BenchError, EMBEDDING_LEN, Engine, MASK_SIDE, MASKS, Masks, Point};
use crate::frame::Frame;

struct Args {
    engines: Vec<String>,
    gpu: bool,
    runs: usize,
    clicks: usize,
    image: Option<PathBuf>,
    click: Option<(f32, f32)>,
    reference: Option<PathBuf>,
    models: PathBuf,
    out: PathBuf,
}

const USAGE: &str = "usage: ml-bench (--image PATH --click X,Y | --reference PATH) \
[--engine candle|burn|both] [--gpu] [--runs N] [--clicks N] [--models DIR] [--out DIR]";

fn parse_args() -> Result<Args, BenchError> {
    let mut args = Args {
        engines: vec!["candle".into(), "burn".into()],
        gpu: false,
        runs: 10,
        clicks: 30,
        image: None,
        click: None,
        reference: None,
        models: PathBuf::from("models"),
        out: PathBuf::from("out"),
    };
    let mut it = std::env::args().skip(1);
    let usage = |what: &str| BenchError::Usage(format!("{what}\n{USAGE}"));
    while let Some(flag) = it.next() {
        let mut value = || {
            it.next()
                .ok_or_else(|| usage(&format!("{flag} needs a value")))
        };
        match flag.as_str() {
            "--engine" => {
                args.engines = match value()?.as_str() {
                    "both" => vec!["candle".into(), "burn".into()],
                    one @ ("candle" | "burn") => vec![one.into()],
                    other => return Err(usage(&format!("unknown engine {other}"))),
                };
            }
            "--gpu" => args.gpu = true,
            "--runs" => args.runs = parse_count(&value()?)?,
            "--clicks" => args.clicks = parse_count(&value()?)?,
            "--image" => args.image = Some(PathBuf::from(value()?)),
            "--click" => {
                let text = value()?;
                let (x, y) = text
                    .split_once(',')
                    .ok_or_else(|| usage("--click is X,Y"))?;
                let x = x
                    .trim()
                    .parse()
                    .map_err(|_| usage("--click X is not a number"))?;
                let y = y
                    .trim()
                    .parse()
                    .map_err(|_| usage("--click Y is not a number"))?;
                args.click = Some((x, y));
            }
            "--reference" => args.reference = Some(PathBuf::from(value()?)),
            "--models" => args.models = PathBuf::from(value()?),
            "--out" => args.out = PathBuf::from(value()?),
            other => return Err(usage(&format!("unknown flag {other}"))),
        }
    }
    if args.reference.is_none() && (args.image.is_none() || args.click.is_none()) {
        return Err(usage("give --reference, or --image with --click"));
    }
    Ok(args)
}

fn parse_count(text: &str) -> Result<usize, BenchError> {
    match text.parse::<usize>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(BenchError::Usage(format!(
            "{text} is not a count above zero"
        ))),
    }
}

/// What onnxruntime produced for one frame and one click.
struct Reference {
    frame: Frame,
    points: Vec<Point>,
    embeddings: Vec<f32>,
    masks: Masks,
}

fn read_f32(tensors: &SafeTensors<'_>, name: &str) -> Result<(Vec<f32>, Vec<usize>), BenchError> {
    let view = tensors
        .tensor(name)
        .map_err(|error| BenchError::Shape(format!("reference {name}: {error}")))?;
    if view.dtype() != safetensors::Dtype::F32 {
        return Err(BenchError::Shape(format!("reference {name} is not f32")));
    }
    let values = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok((values, view.shape().to_vec()))
}

fn load_reference(path: &Path) -> Result<Reference, BenchError> {
    let bytes = std::fs::read(path)
        .map_err(|error| BenchError::Io(format!("{}: {error}", path.display())))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .map_err(|error| BenchError::Shape(format!("{}: {error}", path.display())))?;
    let (hwc, shape) = read_f32(&tensors, "resized")?;
    let [height, width, 3] = shape[..] else {
        return Err(BenchError::Shape(format!(
            "reference frame shape {shape:?}"
        )));
    };
    let frame = Frame::from_hwc(hwc, height, width, 1.0)?;
    let (coords, _) = read_f32(&tensors, "point_coords")?;
    let (labels, _) = read_f32(&tensors, "point_labels")?;
    let points = coords
        .as_chunks::<2>()
        .0
        .iter()
        .zip(&labels)
        .map(|([x, y], label)| Point {
            x: *x,
            y: *y,
            label: *label,
        })
        .collect();
    let (embeddings, _) = read_f32(&tensors, "embeddings")?;
    let (scores, _) = read_f32(&tensors, "iou")?;
    let (logits, _) = read_f32(&tensors, "low_res_masks")?;
    if embeddings.len() != EMBEDDING_LEN {
        return Err(BenchError::Shape(
            "reference embeddings have the wrong size".into(),
        ));
    }
    let masks = Masks::from_all_four(&scores, &logits)?;
    Ok(Reference {
        frame,
        points,
        embeddings,
        masks,
    })
}

/// Everything one engine measured.
struct Run {
    label: String,
    load: Duration,
    first_encode: Duration,
    encodes: Vec<Duration>,
    first_decode: Duration,
    decodes: Vec<Duration>,
    embeddings: Vec<f32>,
    masks: Masks,
}

fn time<T>(f: impl FnOnce() -> Result<T, BenchError>) -> Result<(T, Duration), BenchError> {
    let start = Instant::now();
    let value = f()?;
    Ok((value, start.elapsed()))
}

fn bench(name: &str, args: &Args, frame: &Frame, points: &[Point]) -> Result<Run, BenchError> {
    let (mut engine, load): (Box<dyn Engine>, Duration) = match name {
        "candle" => {
            let weights = args.models.join("mobile_sam-tiny-vitt.safetensors");
            let (engine, load) = time(|| candle_engine::CandleEngine::load(&weights, args.gpu))?;
            (Box::new(engine), load)
        }
        _ => {
            let (engine, load) = time(|| burn_engine::BurnEngine::load(args.gpu))?;
            (Box::new(engine), load)
        }
    };
    // The first run pays for kernel compiles and tuning; later runs do not.
    let ((), first_encode) = time(|| engine.encode(frame))?;
    let mut encodes = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        encodes.push(time(|| engine.encode(frame))?.1);
    }
    let embeddings = engine.embeddings()?;
    let (masks, first_decode) = time(|| engine.decode(points))?;
    let mut decodes = Vec::with_capacity(args.clicks);
    for _ in 0..args.clicks {
        decodes.push(time(|| engine.decode(points))?.1);
    }
    if embeddings.len() != EMBEDDING_LEN
        || masks.scores.len() != MASKS
        || masks.logits.len() != MASKS * MASK_SIDE * MASK_SIDE
    {
        return Err(BenchError::Shape(format!(
            "{} returned outputs of the wrong size",
            engine.label()
        )));
    }
    Ok(Run {
        label: engine.label(),
        load,
        first_encode,
        encodes,
        first_decode,
        decodes,
        embeddings,
        masks,
    })
}

fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1_000_000.0).round() / 1000.0
}

fn summary_json(timings: &[Duration]) -> Value {
    stats::summarize(timings).map_or(Value::Null, |s| {
        json!({ "runs": s.runs, "min_ms": s.min, "median_ms": s.median, "p90_ms": s.p90, "max_ms": s.max })
    })
}

/// How far `b` is from `a`: the embeddings, the scores, and each mask.
fn agreement(a: (&[f32], &Masks), b: (&[f32], &Masks)) -> Value {
    let emb = compare::drift(a.0, b.0);
    let scores = compare::drift(&a.1.scores, &b.1.scores);
    let ious: Vec<Value> = (0..MASKS)
        .map(|i| compare::mask_iou(a.1.mask(i), b.1.mask(i)).map_or(Value::Null, |v| json!(v)))
        .collect();
    json!({
        "embeddings_max_abs": emb.map(|d| d.max),
        "embeddings_mean_abs": emb.map(|d| d.mean),
        "scores_max_abs": scores.map(|d| d.max),
        "mask_iou": ious,
        "same_best_mask": a.1.best() == b.1.best(),
    })
}

/// The best mask, cut to the frame's part of the 256 square.
fn write_mask(path: &Path, masks: &Masks, frame: &Frame) -> Result<(), BenchError> {
    let width = (frame.width / 4).max(1);
    let height = (frame.height / 4).max(1);
    let best = masks.mask(masks.best());
    let mut pixels = Vec::with_capacity(width * height);
    for row in 0..height {
        for col in 0..width {
            let inside = best.get(row * MASK_SIDE + col).is_some_and(|v| *v > 0.0);
            pixels.push(if inside { 255 } else { 0 });
        }
    }
    let (w, h) = (
        u32::try_from(width).map_err(|_| BenchError::Shape("mask too wide".into()))?,
        u32::try_from(height).map_err(|_| BenchError::Shape("mask too tall".into()))?,
    );
    image::GrayImage::from_raw(w, h, pixels)
        .ok_or_else(|| BenchError::Shape("mask buffer".into()))?
        .save(path)
        .map_err(|error| BenchError::Io(format!("{}: {error}", path.display())))
}

fn run() -> Result<(), BenchError> {
    let args = parse_args()?;
    let reference = args.reference.as_deref().map(load_reference).transpose()?;
    let (frame, points) = match &reference {
        Some(r) => (r.frame.clone(), r.points.clone()),
        None => {
            let path = args
                .image
                .as_deref()
                .ok_or_else(|| BenchError::Usage(USAGE.into()))?;
            let frame = Frame::open(path)?;
            let (x, y) = args.click.unwrap_or_default();
            let points = vec![Point {
                x: x * frame.scale,
                y: y * frame.scale,
                label: 1.0,
            }];
            (frame, points)
        }
    };
    std::fs::create_dir_all(&args.out)
        .map_err(|error| BenchError::Io(format!("{}: {error}", args.out.display())))?;

    let mut runs = Vec::new();
    for name in &args.engines {
        let run = bench(name, &args, &frame, &points)?;
        let file = format!("{}_mask.png", run.label.replace(' ', "_"));
        write_mask(&args.out.join(file), &run.masks, &frame)?;
        runs.push(run);
    }

    let mut engines = Vec::new();
    for run in &runs {
        let mut entry = json!({
            "engine": run.label,
            "load_ms": ms(run.load),
            "first_encode_ms": ms(run.first_encode),
            "encode": summary_json(&run.encodes),
            "first_decode_ms": ms(run.first_decode),
            "decode": summary_json(&run.decodes),
            "scores": run.masks.scores,
            "best_mask": run.masks.best(),
        });
        if let (Some(r), Some(obj)) = (&reference, entry.as_object_mut()) {
            obj.insert(
                "vs_onnxruntime".into(),
                agreement((&r.embeddings, &r.masks), (&run.embeddings, &run.masks)),
            );
        }
        engines.push(entry);
    }
    let mut report = json!({
        "machine": { "os": std::env::consts::OS, "arch": std::env::consts::ARCH },
        "frame": { "width": frame.width, "height": frame.height },
        "points": points.iter().map(|p| json!([p.x, p.y, p.label])).collect::<Vec<_>>(),
        "engines": engines,
    });
    if let ([a, b], Some(obj)) = (&runs[..], report.as_object_mut()) {
        obj.insert(
            "candle_vs_burn".into(),
            agreement((&a.embeddings, &a.masks), (&b.embeddings, &b.masks)),
        );
    }
    let text = serde_json::to_string_pretty(&report)
        .map_err(|error| BenchError::Io(format!("report: {error}")))?;
    let path = args.out.join("report.json");
    std::fs::write(&path, &text)
        .map_err(|error| BenchError::Io(format!("{}: {error}", path.display())))?;
    writeln!(std::io::stdout().lock(), "{text}")
        .map_err(|error| BenchError::Io(format!("stdout: {error}")))
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            // The bench is a standalone tool with no logger; the error is its output.
            let _ = writeln!(std::io::stderr().lock(), "ml-bench: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
