//! Release throughput of the WebP decoder and encoder, beside libwebp.
//!
//! `cargo bench -p cutlass-webp` runs it. The picture is one 1920x1080
//! frame of the first video in `../local-files`, through ffmpeg; without
//! them it is a synthetic gradient with hard edges. `cwebp` writes the
//! files the decoder reads, and its own `-v` timings are printed beside
//! ours: they are in-process times, without process startup. Each of our
//! rows is the median of five passes after one warmup. Our decode of each
//! file is checked against `dwebp` first; a mismatch fails the run.
//! A missing `cwebp` or `dwebp` prints a skip line and exits 0.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use cutlass_decode::{CodecId, CodecParams, Decoder, Packet};
use cutlass_encode::{BitDepth, ColorType, Encoder, PictureInput, StillImage};
use cutlass_webp::{WebpDecoder, WebpEncoder, WebpOptions};

const REPS: usize = 5;

fn main() -> Result<(), String> {
    let mut out = std::io::stdout();
    let say = |out: &mut std::io::Stdout, line: String| {
        let _ = writeln!(out, "{line}");
    };
    let (Some(cwebp), Some(dwebp)) = (tool("cwebp"), tool("dwebp")) else {
        say(&mut out, "skip: cwebp and dwebp are not installed".into());
        return Ok(());
    };
    let dir = std::env::temp_dir().join(format!("cutlass-webp-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let (label, width, height, rgba, png) = picture(&dir)?;
    let megapixels = f64::from(width * height) / 1e6;
    say(&mut out, format!("picture: {label}, {width}x{height}"));

    for (name, args) in [
        ("lossy q75 m4", &["-q", "75", "-m", "4"][..]),
        ("lossless z6", &["-lossless", "-z", "6"][..]),
    ] {
        let file = dir.join(format!("{}.webp", name.replace(' ', "-")));
        let theirs = cwebp_run(&cwebp, &png, args, &file)?;
        let bytes = std::fs::read(&file).map_err(|e| e.to_string())?;
        let ours = decode(&bytes)?;
        let reference = dwebp_rgba(&dwebp, &file, &dir)?;
        if ours != reference {
            return Err(format!("{name}: decode differs from dwebp"));
        }
        let decode_time = median(|| decode(&bytes).map(|_| ()))?;
        let dwebp_time = dwebp_time(&dwebp, &file)?;
        say(
            &mut out,
            format!(
                "decode {name}: {} bytes, cutlass {:.1} ms ({:.1} MP/s), dwebp {:.1} ms; cwebp encode {:.1} ms",
                bytes.len(),
                ms(decode_time),
                megapixels / decode_time.as_secs_f64(),
                ms(dwebp_time),
                ms(theirs)
            ),
        );
    }

    for (name, options, args) in [
        (
            "lossy q75 m0",
            WebpOptions {
                lossless: false,
                quality: 75,
                effort: 0,
                exact: false,
            },
            &["-q", "75", "-m", "0"][..],
        ),
        (
            "lossy q75 m4",
            WebpOptions::default(),
            &["-q", "75", "-m", "4"][..],
        ),
        (
            "lossy q75 m6",
            WebpOptions {
                lossless: false,
                quality: 75,
                effort: 6,
                exact: false,
            },
            &["-q", "75", "-m", "6"][..],
        ),
        (
            "lossless z0",
            WebpOptions {
                lossless: true,
                quality: 75,
                effort: 0,
                exact: true,
            },
            &["-lossless", "-z", "0", "-exact"][..],
        ),
        (
            "lossless z6",
            WebpOptions::lossless(),
            &["-lossless", "-z", "6"][..],
        ),
    ] {
        let size = encode(options, width, height, &rgba)?.len();
        let encode_time = median(|| encode(options, width, height, &rgba).map(|_| ()))?;
        let file = dir.join("theirs.webp");
        let theirs = cwebp_run(&cwebp, &png, args, &file)?;
        let their_size = std::fs::metadata(&file).map_err(|e| e.to_string())?.len();
        say(
            &mut out,
            format!(
                "encode {name}: cutlass {size} bytes {:.1} ms ({:.1} MP/s), cwebp {their_size} bytes {:.1} ms ({:.1} MP/s)",
                ms(encode_time),
                megapixels / encode_time.as_secs_f64(),
                ms(theirs),
                megapixels / theirs.as_secs_f64()
            ),
        );
    }
    say(
        &mut out,
        format!("median of {REPS} after 1 warmup, one thread"),
    );
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn median(mut run: impl FnMut() -> Result<(), String>) -> Result<Duration, String> {
    run()?;
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        run()?;
        times.push(start.elapsed());
    }
    times.sort();
    Ok(times[REPS / 2])
}

fn decode(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut decoder =
        WebpDecoder::open(&CodecParams::video(CodecId::Webp, 0, 0)).map_err(|e| e.to_string())?;
    let pictures = decoder
        .decode_picture(&Packet::video(bytes.to_vec(), 0, 0, 0, true))
        .map_err(|e| e.to_string())?;
    let image = &pictures.first().ok_or("no picture")?.image;
    let channels = image.samples.len() / (image.width * image.height) as usize;
    Ok(image
        .samples
        .chunks(channels)
        .flat_map(|p| {
            let alpha = p.get(3).copied().unwrap_or(255);
            [p[0] as u8, p[1] as u8, p[2] as u8, alpha as u8]
        })
        .collect())
}

fn encode(options: WebpOptions, width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    let mut params = CodecParams::video(CodecId::Webp, width, height);
    params.extra = options.to_extra();
    let mut encoder = WebpEncoder::open(&params).map_err(|e| e.to_string())?;
    let samples = rgba.iter().map(|&s| u16::from(s)).collect();
    let image = StillImage::from_samples(width, height, ColorType::Rgba, BitDepth::Eight, samples);
    let packets = encoder
        .push_picture(&PictureInput { image })
        .map_err(|e| e.to_string())?;
    Ok(packets.into_iter().next().ok_or("no packet")?.data)
}

fn tool(name: &str) -> Option<PathBuf> {
    let homebrew = PathBuf::from("/opt/homebrew/bin").join(name);
    if homebrew.is_file() {
        return Some(homebrew);
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(name))
            .find(|path| path.is_file())
    })
}

/// The picture as RGBA and as a PNG for `cwebp`.
fn picture(dir: &Path) -> Result<(String, u32, u32, Vec<u8>, PathBuf), String> {
    let png = dir.join("picture.png");
    let local = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    let video = std::fs::read_dir(&local).ok().and_then(|entries| {
        entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "mp4"))
            .min()
    });
    if let (Some(video), Some(ffmpeg)) = (video, tool("ffmpeg")) {
        let ok = Command::new(&ffmpeg)
            .args(["-v", "error", "-y", "-ss", "1", "-i"])
            .arg(&video)
            .args([
                "-frames:v",
                "1",
                "-vf",
                "scale=1920:1080",
                "-pix_fmt",
                "rgba",
            ])
            .arg(&png)
            .status()
            .is_ok_and(|status| status.success());
        let raw = Command::new(&ffmpeg)
            .args(["-v", "error", "-i"])
            .arg(&png)
            .args(["-f", "rawvideo", "-pix_fmt", "rgba", "-"])
            .output()
            .map_err(|e| e.to_string())?;
        if ok && raw.status.success() && raw.stdout.len() == 1920 * 1080 * 4 {
            let name = video
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            return Ok((format!("frame of {name}"), 1920, 1080, raw.stdout, png));
        }
    }
    let (width, height) = (1920u32, 1080u32);
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let edge = (x / 64 + y / 64) % 2 == 0;
            rgba.extend_from_slice(&[
                (x * 255 / width) as u8,
                (y * 255 / height) as u8,
                if edge { 220 } else { 30 },
                255,
            ]);
        }
    }
    let ffmpeg = tool("ffmpeg").ok_or("ffmpeg is needed for the synthetic picture's PNG")?;
    let raw = dir.join("picture.rgba");
    std::fs::write(&raw, &rgba).map_err(|e| e.to_string())?;
    let ok = Command::new(ffmpeg)
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            "1920x1080",
            "-i",
        ])
        .arg(&raw)
        .arg(&png)
        .status()
        .is_ok_and(|status| status.success());
    if !ok {
        return Err("ffmpeg could not write the PNG".into());
    }
    Ok(("synthetic".into(), width, height, rgba, png))
}

/// `cwebp`'s own encode time, from `-v`.
fn cwebp_run(cwebp: &Path, png: &Path, args: &[&str], out: &Path) -> Result<Duration, String> {
    let output = Command::new(cwebp)
        .args(args)
        .arg("-v")
        .arg(png)
        .arg("-o")
        .arg(out)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!("cwebp {args:?} failed"));
    }
    seconds(
        &String::from_utf8_lossy(&output.stderr),
        "Time to encode picture:",
    )
}

/// `dwebp`'s own decode time, from `-v`.
fn dwebp_time(dwebp: &Path, file: &Path) -> Result<Duration, String> {
    let mut best = Duration::MAX;
    for _ in 0..REPS {
        let output = Command::new(dwebp)
            .arg("-v")
            .arg(file)
            .args(["-pam", "-o", "/dev/null"])
            .output()
            .map_err(|e| e.to_string())?;
        let time = seconds(
            &String::from_utf8_lossy(&output.stderr),
            "Time to decode picture:",
        )?;
        best = best.min(time);
    }
    Ok(best)
}

fn seconds(text: &str, key: &str) -> Result<Duration, String> {
    let line = text
        .lines()
        .find(|line| line.contains(key))
        .ok_or_else(|| format!("no `{key}` in {text}"))?;
    let value: f64 = line
        .split(':')
        .nth(1)
        .and_then(|v| v.trim().trim_end_matches('s').parse().ok())
        .ok_or_else(|| format!("unreadable `{line}`"))?;
    Ok(Duration::from_secs_f64(value))
}

fn dwebp_rgba(dwebp: &Path, file: &Path, dir: &Path) -> Result<Vec<u8>, String> {
    let pam = dir.join("reference.pam");
    let ok = Command::new(dwebp)
        .arg("-quiet")
        .arg(file)
        .arg("-pam")
        .arg("-o")
        .arg(&pam)
        .status()
        .is_ok_and(|status| status.success());
    if !ok {
        return Err("dwebp failed".into());
    }
    let bytes = std::fs::read(&pam).map_err(|e| e.to_string())?;
    let marker = b"ENDHDR\n";
    let at = bytes
        .windows(marker.len())
        .position(|w| w == marker)
        .ok_or("no PAM header")?;
    Ok(bytes[at + marker.len()..].to_vec())
}
