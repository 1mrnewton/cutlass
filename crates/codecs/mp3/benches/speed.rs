//! Release decode and encode speed on the local MP3 files, beside ffmpeg
//! and lame on the same input.
//!
//! `cargo bench -p cutlass-mp3 --bench speed`
//!
//! The local media folder (`../local-files` next to the checkout) is
//! optional: without it the bench prints a note and exits 0. Times are one
//! thread, packets and PCM already in memory, best of `RUNS`. ffmpeg's and
//! lame's times include process launch and file reading.

use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use cutlass_decode::{ChannelLayout, CodecId, CodecParams, Decoder, Output, Packet, Pcm};
use cutlass_encode::{AudioInput, Encoder};
use cutlass_mp3::{Mp3Decoder, Mp3Encoder, Mp3EncoderConfig, Mp3Rate};
use cutlass_mp3_container::{Demux, Mp3Demux};

const RUNS: usize = 3;

fn layout(channels: usize) -> ChannelLayout {
    if channels == 1 {
        ChannelLayout::mono()
    } else {
        ChannelLayout::stereo()
    }
}

struct Input {
    rate: u32,
    channels: usize,
    packets: Vec<Packet>,
}

fn read(path: &Path) -> Result<Input, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let mut demux = Mp3Demux::open(Cursor::new(bytes))?;
    let stream = demux.streams()[0].clone();
    let rate = stream.params.sample_rate;
    let channels = usize::from(stream.params.channels);
    let mut packets = Vec::new();
    while let Some(packet) = demux.next_packet()? {
        packets.push(Packet::audio(
            packet.data,
            packet.pts,
            packet.duration,
            rate,
        ));
    }
    Ok(Input {
        rate,
        channels,
        packets,
    })
}

fn decode(input: &Input) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let params = CodecParams::audio(CodecId::Mp3, input.rate, layout(input.channels));
    let mut decoder = Mp3Decoder::open(&params)?;
    let mut pcm = Vec::new();
    for packet in &input.packets {
        for output in decoder.push(packet)? {
            if let Output::Audio(audio) = output {
                pcm.extend_from_slice(audio.pcm.as_f32().unwrap_or(&[]));
            }
        }
    }
    Ok(pcm)
}

fn encode(
    rate: u32,
    channels: usize,
    mode: Mp3Rate,
    pcm: &[f32],
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut encoder = Mp3Encoder::new(Mp3EncoderConfig {
        sample_rate: rate,
        channels,
        rate: mode,
    })?;
    let mut bytes = 0;
    // One second per push, as export hands it over.
    for chunk in pcm.chunks(rate as usize * channels) {
        let input = AudioInput {
            pcm: Pcm::from_f32(rate, layout(channels), chunk.to_vec())
                .map_err(|error| format!("pcm: {error:?}"))?,
        };
        bytes += encoder
            .push(&input)?
            .iter()
            .map(|p| p.data.len())
            .sum::<usize>();
    }
    bytes += encoder.drain()?.iter().map(|p| p.data.len()).sum::<usize>();
    Ok(bytes)
}

fn best<T>(mut run: impl FnMut() -> T) -> (f64, T) {
    let mut fastest = f64::MAX;
    let mut last = None;
    for _ in 0..RUNS {
        let started = Instant::now();
        let value = run();
        fastest = fastest.min(started.elapsed().as_secs_f64());
        last = Some(value);
    }
    match last {
        Some(value) => (fastest, value),
        None => (fastest, run()),
    }
}

fn tool(name: &str) -> Option<PathBuf> {
    ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]
        .iter()
        .map(|dir| Path::new(dir).join(name))
        .find(|path| path.exists())
}

fn timed(command: &mut Command) -> Option<f64> {
    let started = Instant::now();
    let status = command
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .ok()?;
    status.success().then(|| started.elapsed().as_secs_f64())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = std::io::stdout().lock();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        writeln!(out, "skip: {} is missing", dir.display())?;
        return Ok(());
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "mp3"))
        .collect();
    files.sort();
    let ffmpeg = tool("ffmpeg");
    let lame = tool("lame");
    let scratch = std::env::temp_dir().join(format!("cutlass-mp3-speed-{}", std::process::id()));
    std::fs::create_dir_all(&scratch)?;
    writeln!(
        out,
        "one thread, best of {RUNS}; real-time factor is audio seconds per second"
    )?;
    for path in files {
        let name = path
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        let input = read(&path)?;
        let (decode_s, pcm) = best(|| decode(&input));
        let pcm = pcm?;
        let frames = pcm.len() / input.channels;
        let seconds = frames as f64 / f64::from(input.rate);
        writeln!(
            out,
            "{name}: {seconds:.1} s, {} Hz, {} ch",
            input.rate, input.channels
        )?;
        writeln!(
            out,
            "  decode  cutlass {decode_s:.3} s ({:.0}x)",
            seconds / decode_s
        )?;
        if let Some(ffmpeg) = &ffmpeg {
            let mut command = Command::new(ffmpeg);
            command
                .args(["-v", "error", "-threads", "1", "-i"])
                .arg(&path)
                .args(["-f", "f32le", "-y", "/dev/null"]);
            if let Some(time) = timed(&mut command) {
                writeln!(
                    out,
                    "  decode  ffmpeg  {time:.3} s ({:.0}x)",
                    seconds / time
                )?;
            }
        }
        let wav = scratch.join("in.wav");
        if let (Some(ffmpeg), Some(_)) = (&ffmpeg, &lame) {
            let _ = Command::new(ffmpeg)
                .args(["-v", "error", "-y", "-i"])
                .arg(&path)
                .args(["-c:a", "pcm_s16le"])
                .arg(&wav)
                .status();
        }
        for (label, mode, lame_args) in [
            ("cbr 128", Mp3Rate::Constant(128), ["-b", "128"]),
            ("vbr V2 ", Mp3Rate::Variable(2), ["-V", "2"]),
        ] {
            let (time, bytes) = best(|| encode(input.rate, input.channels, mode, &pcm));
            let bytes = bytes?;
            writeln!(
                out,
                "  encode  {label} cutlass {time:.3} s ({:.0}x), {:.1} kbps",
                seconds / time,
                bytes as f64 * 8.0 / seconds / 1000.0
            )?;
            if let Some(lame) = &lame
                && wav.exists()
            {
                let mut command = Command::new(lame);
                command
                    .args(["--quiet"])
                    .args(lame_args)
                    .arg(&wav)
                    .arg(scratch.join("out.mp3"));
                if let Some(time) = timed(&mut command) {
                    writeln!(
                        out,
                        "  encode  {label} lame    {time:.3} s ({:.0}x)",
                        seconds / time
                    )?;
                }
            }
        }
        let _ = std::fs::remove_file(&wav);
    }
    let _ = std::fs::remove_dir_all(&scratch);
    Ok(())
}
