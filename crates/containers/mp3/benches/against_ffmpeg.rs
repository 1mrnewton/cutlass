//! Read one ffmpeg MP3 and compare frame sizes and timestamps with ffprobe.
//!
//! `cargo bench -p cutlass-mp3-container`
//!
//! ffmpeg writes a two-second stereo file. Packet count, sizes, and
//! presentation timestamps are checked before anything is timed. The timed
//! work is a full open and read, median of three after one warmup. The
//! ffprobe side is a new process each pass, so that time includes process
//! startup. This crate returns frames, not PCM, so the bench does not decode
//! a waveform. When ffmpeg is missing the process prints
//! `skip: ffmpeg is not installed` and exits 0.

#![allow(clippy::print_stdout)]

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use cutlass_mp3_container::{CodecId, CodecParams, Demux, Mp3Demux, Mp3Mux, Mux, Packet};

const SAMPLE_RATE: u32 = 48_000;
const CHANNELS: u16 = 2;
const BITRATE_KBPS: u32 = 128;
const SAMPLES_PER_FRAME: i64 = 1_152;
const MUX_FRAMES: u16 = 8;

struct Frame {
    size: u64,
    pts: i64,
}

struct ProbePacket {
    size: u64,
    pts_samples: i64,
}

struct Probe {
    sample_rate: u32,
    channels: u16,
    packets: Vec<ProbePacket>,
}

struct Compared {
    packets: usize,
    dropped_leading_tag: bool,
}

struct Times {
    samples_ns: [u128; 3],
    median_ns: u128,
}

struct Report<'a> {
    ffmpeg: &'a Path,
    sample_bytes: u64,
    sample_rate: u32,
    channels: u16,
    compared: &'a Compared,
    cutlass: &'a Times,
    ffprobe: &'a Times,
    mux: &'a Compared,
}

fn main() -> Result<(), String> {
    let Some((ffmpeg, ffprobe)) = locate() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let dir = std::env::temp_dir().join(format!(
        "cutlass-mp3-container-bench-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
    let outcome = run(&ffmpeg, &ffprobe, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn run(ffmpeg: &Path, ffprobe: &Path, dir: &Path) -> Result<(), String> {
    let sample = dir.join("sample.mp3");
    write_sample(ffmpeg, &sample)?;
    let sample_bytes = std::fs::metadata(&sample)
        .map_err(|err| err.to_string())?
        .len();
    let ours = read_frames(&sample)?;
    let probed = read_ffprobe(ffprobe, &sample)?;
    check_sample(&probed)?;
    let compared = compare_frames(&ours, &probed)?;
    let total_bytes = byte_total(&ours)?;

    let mux_path = dir.join("mux.mp3");
    let written = write_mux(&mux_path)?;
    let mux_probe = read_ffprobe(ffprobe, &mux_path)?;
    let mux_compared = compare_frames(&written, &mux_probe)?;

    let cutlass = measure(|| time_read(&sample, compared.packets, total_bytes))?;
    let ffprobe_time = measure(|| time_ffprobe(ffprobe, &sample, probed.packets.len()))?;
    print_report(&Report {
        ffmpeg,
        sample_bytes,
        sample_rate: probed.sample_rate,
        channels: probed.channels,
        compared: &compared,
        cutlass: &cutlass,
        ffprobe: &ffprobe_time,
        mux: &mux_compared,
    });
    Ok(())
}

fn print_report(report: &Report<'_>) {
    println!("ffmpeg: {}", report.ffmpeg.display());
    println!(
        "sample: ffmpeg libmp3lame, 2 s, stereo, {} Hz, {BITRATE_KBPS} kbit/s",
        report.sample_rate
    );
    println!("sample_bytes: {}", report.sample_bytes);
    println!("channels: {}", report.channels);
    println!(
        "compare: packets={} sizes=match pts=match dropped_leading_tag={}",
        report.compared.packets,
        yes_no(report.compared.dropped_leading_tag)
    );
    print_times("cutlass", report.cutlass);
    print_times("ffprobe", report.ffprobe);
    println!("timing: median of 3 after 1 warmup");
    println!("ffprobe_time_includes_process_startup: yes");
    println!(
        "mux: packets={} sizes=match pts=match dropped_leading_tag={}",
        report.mux.packets,
        yes_no(report.mux.dropped_leading_tag)
    );
}

fn print_times(name: &str, times: &Times) {
    println!(
        "{name}_samples_ns: {} {} {}",
        times.samples_ns[0], times.samples_ns[1], times.samples_ns[2]
    );
    println!("{name}_median_ns: {}", times.median_ns);
    println!("{name}_median_s: {}", seconds(times.median_ns));
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn seconds(ns: u128) -> String {
    let whole = ns / 1_000_000_000;
    let frac = ns % 1_000_000_000;
    format!("{whole}.{frac:09}")
}

fn measure(mut once: impl FnMut() -> Result<u128, String>) -> Result<Times, String> {
    let _warmup = once()?;
    let mut samples_ns = [once()?, once()?, once()?];
    samples_ns.sort_unstable();
    Ok(Times {
        samples_ns,
        median_ns: samples_ns[1],
    })
}

fn time_read(path: &Path, expect_packets: usize, expect_bytes: u64) -> Result<u128, String> {
    let started = Instant::now();
    let file = File::open(path).map_err(|err| err.to_string())?;
    let mut demux = Mp3Demux::open(file).map_err(|err| err.to_string())?;
    let mut count = 0usize;
    let mut bytes = 0u64;
    while let Some(packet) = demux.next_packet().map_err(|err| err.to_string())? {
        count += 1;
        let len = u64::try_from(packet.data.len()).map_err(|_| "packet length".to_owned())?;
        bytes = bytes
            .checked_add(len)
            .ok_or_else(|| "byte count overflow".to_owned())?;
    }
    let elapsed = started.elapsed().as_nanos();
    if count != expect_packets || bytes != expect_bytes {
        return Err(format!(
            "timed read saw {count} packets and {bytes} bytes, expected {expect_packets} and {expect_bytes}"
        ));
    }
    Ok(elapsed)
}

fn time_ffprobe(ffprobe: &Path, path: &Path, raw_packets: usize) -> Result<u128, String> {
    let started = Instant::now();
    let output = ffprobe_cmd(ffprobe, path)?;
    let elapsed = started.elapsed().as_nanos();
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let found = text
        .lines()
        .filter(|line| line.starts_with("packet|"))
        .count();
    if found != raw_packets {
        return Err(format!(
            "ffprobe timed read saw {found} packets, expected {raw_packets}"
        ));
    }
    Ok(elapsed)
}

fn check_sample(probed: &Probe) -> Result<(), String> {
    if probed.sample_rate != SAMPLE_RATE || probed.channels != CHANNELS {
        return Err(format!(
            "sample is {} Hz {} ch, expected {SAMPLE_RATE} Hz {CHANNELS} ch",
            probed.sample_rate, probed.channels
        ));
    }
    Ok(())
}

fn byte_total(frames: &[Frame]) -> Result<u64, String> {
    let mut total = 0u64;
    for frame in frames {
        total = total
            .checked_add(frame.size)
            .ok_or_else(|| "byte count overflow".to_owned())?;
    }
    Ok(total)
}

fn compare_frames(ours: &[Frame], probed: &Probe) -> Result<Compared, String> {
    if probed.sample_rate == 0 {
        return Err("ffprobe reported no sample rate".to_owned());
    }
    let (aligned, dropped_leading_tag) = align(ours, &probed.packets)?;
    if aligned.len() != ours.len() {
        return Err(format!(
            "packet count ours {} ffprobe {}",
            ours.len(),
            aligned.len()
        ));
    }
    let base = aligned
        .first()
        .map(|packet| packet.pts_samples)
        .ok_or_else(|| "ffprobe returned no audio packets".to_owned())?;
    for (index, (frame, theirs)) in ours.iter().zip(aligned).enumerate() {
        if frame.size != theirs.size {
            return Err(format!(
                "size at {index}: ours {} ffprobe {}",
                frame.size, theirs.size
            ));
        }
        let pts = theirs
            .pts_samples
            .checked_sub(base)
            .ok_or_else(|| format!("pts underflow at {index}"))?;
        if frame.pts != pts {
            return Err(format!("pts at {index}: ours {} ffprobe {pts}", frame.pts));
        }
    }
    Ok(Compared {
        packets: ours.len(),
        dropped_leading_tag,
    })
}

fn align<'a>(
    ours: &[Frame],
    theirs: &'a [ProbePacket],
) -> Result<(&'a [ProbePacket], bool), String> {
    let our_sizes: Vec<u64> = ours.iter().map(|frame| frame.size).collect();
    let their_sizes: Vec<u64> = theirs.iter().map(|packet| packet.size).collect();
    if their_sizes == our_sizes {
        return Ok((theirs, false));
    }
    if their_sizes.len() == our_sizes.len() + 1 && their_sizes[1..] == our_sizes {
        return Ok((&theirs[1..], true));
    }
    Err(format!(
        "packet sizes diverged: ours {our_sizes:?} ffprobe {their_sizes:?}"
    ))
}

fn read_frames(path: &Path) -> Result<Vec<Frame>, String> {
    let file = File::open(path).map_err(|err| err.to_string())?;
    let mut demux = Mp3Demux::open(file).map_err(|err| err.to_string())?;
    let mut frames = Vec::new();
    while let Some(packet) = demux.next_packet().map_err(|err| err.to_string())? {
        let size = u64::try_from(packet.data.len()).map_err(|_| "packet length".to_owned())?;
        frames.push(Frame {
            size,
            pts: packet.pts,
        });
    }
    if frames.is_empty() {
        return Err("demux returned no frames".to_owned());
    }
    Ok(frames)
}

fn read_ffprobe(ffprobe: &Path, path: &Path) -> Result<Probe, String> {
    let output = ffprobe_cmd(ffprobe, path)?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    parse_probe(&String::from_utf8_lossy(&output.stdout))
}

fn ffprobe_cmd(ffprobe: &Path, path: &Path) -> Result<std::process::Output, String> {
    Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,time_base",
            "-show_entries",
            "packet=size,pts",
            "-of",
            "compact",
        ])
        .arg(path)
        .output()
        .map_err(|err| err.to_string())
}

fn parse_probe(text: &str) -> Result<Probe, String> {
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut time_num = 0i64;
    let mut time_den = 0i64;
    let mut raw = Vec::new();
    let mut saw_stream = false;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("stream|") {
            if saw_stream {
                return Err("ffprobe returned more than one stream".to_owned());
            }
            saw_stream = true;
            parse_stream(
                rest,
                &mut sample_rate,
                &mut channels,
                &mut time_num,
                &mut time_den,
            )?;
        } else if let Some(rest) = line.strip_prefix("packet|") {
            raw.push(parse_packet(rest)?);
        }
    }
    if !saw_stream || sample_rate == 0 || time_den == 0 {
        return Err(format!("ffprobe stream missing in {text}"));
    }
    let mut packets = Vec::with_capacity(raw.len());
    for (size, pts) in raw {
        packets.push(ProbePacket {
            size,
            pts_samples: scale_pts(pts, sample_rate, time_num, time_den)?,
        });
    }
    Ok(Probe {
        sample_rate,
        channels,
        packets,
    })
}

fn parse_stream(
    rest: &str,
    sample_rate: &mut u32,
    channels: &mut u16,
    time_num: &mut i64,
    time_den: &mut i64,
) -> Result<(), String> {
    for part in rest.split('|') {
        if let Some(value) = part.strip_prefix("sample_rate=") {
            *sample_rate = value
                .parse()
                .map_err(|_| format!("bad sample rate {value}"))?;
        } else if let Some(value) = part.strip_prefix("channels=") {
            *channels = value.parse().map_err(|_| format!("bad channels {value}"))?;
        } else if let Some(value) = part.strip_prefix("time_base=") {
            let (num, den) = value
                .split_once('/')
                .ok_or_else(|| format!("bad time base {value}"))?;
            *time_num = num.parse().map_err(|_| format!("bad time base {value}"))?;
            *time_den = den.parse().map_err(|_| format!("bad time base {value}"))?;
        }
    }
    Ok(())
}

fn parse_packet(rest: &str) -> Result<(u64, i64), String> {
    let mut size = None;
    let mut pts = None;
    for part in rest.split('|') {
        if let Some(value) = part.strip_prefix("size=") {
            size = Some(
                value
                    .parse::<u64>()
                    .map_err(|_| format!("bad size {value}"))?,
            );
        } else if let Some(value) = part.strip_prefix("pts=") {
            pts = Some(
                value
                    .parse::<i64>()
                    .map_err(|_| format!("bad pts {value}"))?,
            );
        }
    }
    let size = size.ok_or_else(|| format!("packet missing size: {rest}"))?;
    let pts = pts.ok_or_else(|| format!("packet missing pts: {rest}"))?;
    Ok((size, pts))
}

fn scale_pts(pts: i64, sample_rate: u32, time_num: i64, time_den: i64) -> Result<i64, String> {
    if time_den == 0 {
        return Err("time base denominator is zero".to_owned());
    }
    let samples = i128::from(pts)
        .checked_mul(i128::from(time_num))
        .and_then(|value| value.checked_mul(i128::from(sample_rate)))
        .and_then(|value| value.checked_div(i128::from(time_den)))
        .ok_or_else(|| "pts conversion overflowed".to_owned())?;
    i64::try_from(samples).map_err(|_| "pts does not fit an i64".to_owned())
}

fn write_sample(ffmpeg: &Path, path: &Path) -> Result<(), String> {
    let output = Command::new(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2:sample_rate=48000",
            "-ac",
            "2",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
        ])
        .arg(path)
        .output()
        .map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "ffmpeg encode failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

fn write_mux(path: &Path) -> Result<Vec<Frame>, String> {
    let file = File::create(path).map_err(|err| err.to_string())?;
    let mut mux = Mp3Mux::new(file);
    mux.add_stream(CodecParams::audio(CodecId::Mp3, SAMPLE_RATE, CHANNELS))
        .map_err(|err| err.to_string())?;
    let mut written = Vec::new();
    for index in 0..MUX_FRAMES {
        let frame = layer3_frame(index)?;
        let pts = i64::from(index) * SAMPLES_PER_FRAME;
        let size = u64::try_from(frame.len()).map_err(|_| "frame length".to_owned())?;
        written.push(Frame { size, pts });
        let packet = Packet::audio(frame, pts, SAMPLES_PER_FRAME, SAMPLE_RATE);
        mux.write_packet(&packet).map_err(|err| err.to_string())?;
    }
    mux.finish().map_err(|err| err.to_string())?;
    let _file = mux.into_inner().map_err(|err| err.to_string())?;
    Ok(written)
}

/// MPEG-1 Layer III, bitrate index 9, sample-rate index 1, stereo, no CRC.
fn layer3_frame(index: u16) -> Result<Vec<u8>, String> {
    let bitrate_index = 9u32;
    let rate_index = 1u32;
    let raw = (0x7FFu32 << 21)
        | (0b11 << 19)
        | (0b01 << 17)
        | (1 << 16)
        | (bitrate_index << 12)
        | (rate_index << 10);
    let size = usize::try_from(BITRATE_KBPS * 144_000 / SAMPLE_RATE)
        .map_err(|_| "frame size does not fit".to_owned())?;
    if size < 8 {
        return Err("frame shorter than a header".to_owned());
    }
    let mut frame = vec![0u8; size];
    frame[..4].copy_from_slice(&raw.to_be_bytes());
    let mark = index.to_be_bytes();
    frame[5] = mark[0];
    frame[6] = mark[1];
    Ok(frame)
}

fn locate() -> Option<(PathBuf, PathBuf)> {
    let mut candidates = vec![
        PathBuf::from("/opt/homebrew/bin/ffmpeg"),
        PathBuf::from("/usr/local/bin/ffmpeg"),
    ];
    if let Some(found) = on_path("ffmpeg") {
        candidates.push(found);
    }
    for ffmpeg in candidates {
        if !ffmpeg.is_file() {
            continue;
        }
        let ffprobe = ffmpeg.with_file_name("ffprobe");
        if ffprobe.is_file() {
            return Some((ffmpeg, ffprobe));
        }
    }
    None
}

fn on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| {
        let candidate = dir.join(name);
        if candidate.is_file() {
            Some(candidate)
        } else {
            None
        }
    })
}
