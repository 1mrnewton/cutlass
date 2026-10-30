//! Read one ffmpeg-written ADTS AAC file and time that read against ffprobe.
//!
//! The accuracy check runs before the clock starts. Frame count, on-disk
//! size, and presentation time have to match ffprobe. ffprobe's packet size
//! is the ADTS frame, header included. This demuxer returns the raw access
//! unit, so the size check is payload length plus header length.
//!
//! Timing is one untimed warmup and then three full reads. The reported
//! value is the median. The ffprobe samples include process startup.

#![allow(clippy::print_stdout)]

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use cutlass_adts::{AdtsFile, AdtsMuxer, Muxer};
use cutlass_codec::CodecId;
use cutlass_codec::access::{ChannelLayout, CodecParams, Packet};
use cutlass_container::Demux;

const LAVFI: &str = "sine=frequency=440:sample_rate=48000:duration=0.5";

struct Frame {
    pts: i64,
    duration: i64,
    timescale: u32,
    bytes: Vec<u8>,
}

struct Opened {
    sample_rate: u32,
    channels: u16,
    frames: Vec<Frame>,
}

struct ProbePacket {
    pts: i64,
    duration: i64,
    size: usize,
}

struct StreamClock {
    sample_rate: u32,
    channels: u16,
    time_base_num: i64,
    time_base_den: i64,
}

struct Row {
    pts: i64,
    duration: i64,
    payload: usize,
    header: usize,
    ffprobe_pts: i64,
    ffprobe_duration: i64,
    ffprobe_size: usize,
}

struct Times {
    median: u128,
    min: u128,
    max: u128,
    samples: Vec<u128>,
}

fn main() -> Result<(), String> {
    let Some(ffmpeg) = find_bin("ffmpeg") else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let Some(ffprobe) = find_bin("ffprobe") else {
        return Err("ffprobe is not installed".to_owned());
    };
    let dir = std::env::temp_dir().join(format!("cutlass-adts-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
    let outcome = run(&dir, &ffmpeg, &ffprobe);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn run(dir: &Path, ffmpeg: &Path, ffprobe: &Path) -> Result<(), String> {
    let source = dir.join("source.aac");
    let remux = dir.join("remux.aac");
    write_sample(ffmpeg, &source)?;
    let file_bytes = std::fs::read(&source).map_err(|err| err.to_string())?;
    let opened = open_all(&source)?;
    let clock = probe_stream(ffprobe, &source)?;
    let probed = probe_packets(ffprobe, &source)?;
    check_stream(&opened, &clock)?;
    let rows = check_frames(&file_bytes, &opened.frames, &probed, &clock)?;

    let mux_bytes = mux_adts(&opened)?;
    std::fs::write(&remux, &mux_bytes).map_err(|err| err.to_string())?;
    let remuxed = open_all(&remux)?;
    same_units(&opened, &remuxed)?;
    let mux_clock = probe_stream(ffprobe, &remux)?;
    let mux_probed = probe_packets(ffprobe, &remux)?;
    check_stream(&remuxed, &mux_clock)?;
    let mux_rows = check_frames(&mux_bytes, &remuxed.frames, &mux_probed, &mux_clock)?;

    let cutlass = measure(|| checksum(&source))?;
    let ffprobe_time = measure(|| ffprobe_read(ffprobe, &source))?;
    print_report(
        &file_bytes,
        &clock,
        &rows,
        &mux_bytes,
        &mux_rows,
        &cutlass,
        &ffprobe_time,
    );
    Ok(())
}

fn write_sample(ffmpeg: &Path, dest: &Path) -> Result<(), String> {
    let status = Command::new(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            LAVFI,
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-f",
            "adts",
        ])
        .arg(dest)
        .status()
        .map_err(|err| err.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("ffmpeg exited with {status}"))
    }
}

fn open_all(path: &Path) -> Result<Opened, String> {
    let file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut demux = AdtsFile::open(file).map_err(|err| err.to_string())?;
    let (sample_rate, channels) = stream_clock(&demux)?;
    let mut frames = Vec::new();
    while let Some(packet) = demux.read_packet().map_err(|err| err.to_string())? {
        let pts = packet
            .timestamp()
            .presentation()
            .ok_or_else(|| "packet has no presentation time".to_owned())?;
        let duration = packet
            .timestamp()
            .duration()
            .ok_or_else(|| "packet has no duration".to_owned())?;
        frames.push(Frame {
            pts,
            duration,
            timescale: packet.timescale().ticks_per_second(),
            bytes: packet.bytes().to_vec(),
        });
    }
    if frames.is_empty() {
        return Err("adts file has no frames".to_owned());
    }
    Ok(Opened {
        sample_rate,
        channels,
        frames,
    })
}

fn stream_clock(demux: &AdtsFile<File>) -> Result<(u32, u16), String> {
    let stream = Demux::streams(demux)
        .first()
        .ok_or_else(|| "adts file has no stream".to_owned())?;
    let sample_rate = stream
        .params()
        .sample_rate()
        .ok_or_else(|| "stream has no sample rate".to_owned())?;
    let channels = stream
        .params()
        .channels()
        .ok_or_else(|| "stream has no channel count".to_owned())?;
    Ok((sample_rate, channels))
}

fn check_stream(opened: &Opened, clock: &StreamClock) -> Result<(), String> {
    if opened.sample_rate != clock.sample_rate {
        return Err(format!(
            "sample rate cutlass {} ffprobe {}",
            opened.sample_rate, clock.sample_rate
        ));
    }
    if opened.channels != clock.channels {
        return Err(format!(
            "channels cutlass {} ffprobe {}",
            opened.channels, clock.channels
        ));
    }
    Ok(())
}

fn check_frames(
    file: &[u8],
    frames: &[Frame],
    probed: &[ProbePacket],
    clock: &StreamClock,
) -> Result<Vec<Row>, String> {
    if frames.len() != probed.len() {
        return Err(format!(
            "frame count cutlass {} ffprobe {}",
            frames.len(),
            probed.len()
        ));
    }
    let mut rows = Vec::with_capacity(frames.len());
    let mut offset = 0usize;
    for (index, (frame, probe)) in frames.iter().zip(probed.iter()).enumerate() {
        if frame.timescale != clock.sample_rate {
            return Err(format!(
                "frame {index} timescale {} != sample rate {}",
                frame.timescale, clock.sample_rate
            ));
        }
        let end = offset
            .checked_add(probe.size)
            .ok_or_else(|| format!("frame {index} size overflows"))?;
        let slice = file
            .get(offset..end)
            .ok_or_else(|| format!("frame {index} exceeds the file"))?;
        let frame_len = adts_frame_len(slice)?;
        if frame_len != probe.size {
            return Err(format!(
                "frame {index} header frame_length {frame_len} != ffprobe size {}",
                probe.size
            ));
        }
        let header = adts_header_len(slice)?;
        let payload = probe
            .size
            .checked_sub(header)
            .ok_or_else(|| format!("frame {index} header exceeds ffprobe size"))?;
        if payload != frame.bytes.len() {
            return Err(format!(
                "frame {index} payload {} != ffprobe size {} minus header {header}",
                frame.bytes.len(),
                probe.size
            ));
        }
        if !same_ticks(frame.pts, frame.timescale, probe.pts, clock) {
            return Err(format!(
                "frame {index} pts {}/{} != ffprobe pts {} at {}/{}",
                frame.pts, frame.timescale, probe.pts, clock.time_base_num, clock.time_base_den
            ));
        }
        if !same_ticks(frame.duration, frame.timescale, probe.duration, clock) {
            return Err(format!(
                "frame {index} duration {}/{} != ffprobe duration {}",
                frame.duration, frame.timescale, probe.duration
            ));
        }
        rows.push(Row {
            pts: frame.pts,
            duration: frame.duration,
            payload,
            header,
            ffprobe_pts: probe.pts,
            ffprobe_duration: probe.duration,
            ffprobe_size: probe.size,
        });
        offset = end;
    }
    if offset != file.len() {
        return Err(format!(
            "ffprobe sizes cover {offset} bytes, file is {}",
            file.len()
        ));
    }
    Ok(rows)
}

fn same_ticks(ticks: i64, timescale: u32, probe_ticks: i64, clock: &StreamClock) -> bool {
    let left = i128::from(ticks) * i128::from(clock.time_base_den);
    let right = i128::from(probe_ticks) * i128::from(clock.time_base_num) * i128::from(timescale);
    left == right
}

fn adts_frame_len(frame: &[u8]) -> Result<usize, String> {
    let header = frame
        .get(..7)
        .ok_or_else(|| "adts header is short".to_owned())?;
    let sync = (u16::from(header[0]) << 8) | u16::from(header[1]);
    if sync & 0xFFF0 != 0xFFF0 {
        return Err("adts sync missing".to_owned());
    }
    let layer = (header[1] >> 1) & 0b11;
    if layer != 0 {
        return Err("adts layer is not 0".to_owned());
    }
    let len = (usize::from(header[3] & 0b11) << 11)
        | (usize::from(header[4]) << 3)
        | (usize::from(header[5]) >> 5);
    if len != frame.len() {
        return Err(format!("adts frame_length {len} != buffer {}", frame.len()));
    }
    Ok(len)
}

fn adts_header_len(frame: &[u8]) -> Result<usize, String> {
    let header = frame
        .get(..7)
        .ok_or_else(|| "adts header is short".to_owned())?;
    let protection_absent = header[1] & 1 == 1;
    let blocks = usize::from(header[6] & 0b11);
    let len = if protection_absent { 7 } else { 9 + 2 * blocks };
    if len > frame.len() {
        return Err("adts header exceeds the frame".to_owned());
    }
    Ok(len)
}

fn same_units(source: &Opened, remuxed: &Opened) -> Result<(), String> {
    if source.sample_rate != remuxed.sample_rate || source.channels != remuxed.channels {
        return Err("remux stream parameters differ".to_owned());
    }
    if source.frames.len() != remuxed.frames.len() {
        return Err(format!(
            "remux frame count {} != source {}",
            remuxed.frames.len(),
            source.frames.len()
        ));
    }
    for (index, (left, right)) in source.frames.iter().zip(remuxed.frames.iter()).enumerate() {
        if left.bytes != right.bytes
            || left.pts != right.pts
            || left.duration != right.duration
            || left.timescale != right.timescale
        {
            return Err(format!("remux frame {index} differs from the source"));
        }
    }
    Ok(())
}

fn mux_adts(opened: &Opened) -> Result<Vec<u8>, String> {
    let mut mux = AdtsMuxer::new();
    mux.add_stream(CodecParams::audio(
        CodecId::Aac,
        opened.sample_rate,
        layout(opened.channels)?,
    ))
    .map_err(|err| err.to_string())?;
    for frame in &opened.frames {
        let packet = Packet::audio(
            frame.bytes.clone(),
            frame.pts,
            frame.duration,
            frame.timescale,
        );
        mux.write_packet(&packet).map_err(|err| err.to_string())?;
    }
    mux.finish().map_err(|err| err.to_string())
}

fn layout(channels: u16) -> Result<ChannelLayout, String> {
    match channels {
        1 => Ok(ChannelLayout::mono()),
        2 => Ok(ChannelLayout::stereo()),
        _ => Err(format!("bench sample has {channels} channels")),
    }
}

fn measure(mut read: impl FnMut() -> Result<u64, String>) -> Result<Times, String> {
    let warmup = read()?;
    std::hint::black_box(warmup);
    let mut samples = Vec::with_capacity(3);
    for _ in 0..3 {
        let started = Instant::now();
        let value = read()?;
        let elapsed = started.elapsed();
        std::hint::black_box(value);
        samples.push(elapsed.as_micros());
    }
    let median = median(&samples)?;
    let min = samples
        .iter()
        .copied()
        .min()
        .ok_or_else(|| "no samples".to_owned())?;
    let max = samples
        .iter()
        .copied()
        .max()
        .ok_or_else(|| "no samples".to_owned())?;
    Ok(Times {
        median,
        min,
        max,
        samples,
    })
}

fn median(samples: &[u128]) -> Result<u128, String> {
    let mut ordered = samples.to_vec();
    ordered.sort_unstable();
    ordered
        .get(ordered.len() / 2)
        .copied()
        .ok_or_else(|| "no samples".to_owned())
}

fn checksum(path: &Path) -> Result<u64, String> {
    let opened = open_all(path)?;
    let mut sum = u64::try_from(opened.frames.len()).map_err(|_| "frame count".to_owned())?;
    for frame in &opened.frames {
        let len = u64::try_from(frame.bytes.len()).map_err(|_| "payload length".to_owned())?;
        let pts = u64::try_from(frame.pts).map_err(|_| "negative pts".to_owned())?;
        sum = sum.wrapping_add(len).wrapping_add(pts);
    }
    Ok(sum)
}

fn ffprobe_read(ffprobe: &Path, path: &Path) -> Result<u64, String> {
    let output = packet_command(ffprobe, path)
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "ffprobe read failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    u64::try_from(output.stdout.len()).map_err(|_| "ffprobe stdout length".to_owned())
}

fn probe_packets(ffprobe: &Path, path: &Path) -> Result<Vec<ProbePacket>, String> {
    let output = packet_command(ffprobe, path)
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "ffprobe packets failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let text = String::from_utf8(output.stdout).map_err(|err| err.to_string())?;
    let mut packets = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        let pts = parse_i64(parts.next(), index, "pts")?;
        let duration = parse_i64(parts.next(), index, "duration")?;
        let size = parse_usize(parts.next(), index, "size")?;
        if parts.next().is_some() {
            return Err(format!("ffprobe packet {index} has extra fields"));
        }
        packets.push(ProbePacket {
            pts,
            duration,
            size,
        });
    }
    Ok(packets)
}

fn packet_command(ffprobe: &Path, path: &Path) -> Command {
    let mut command = Command::new(ffprobe);
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_entries",
            "packet=pts,duration,size",
            "-of",
            "csv=p=0",
        ])
        .arg(path);
    command
}

fn probe_stream(ffprobe: &Path, path: &Path) -> Result<StreamClock, String> {
    let output = Command::new(ffprobe)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_entries",
            "stream=sample_rate,channels,time_base",
            "-of",
            "default=nw=1",
        ])
        .arg(path)
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "ffprobe stream failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let text = String::from_utf8(output.stdout).map_err(|err| err.to_string())?;
    let mut sample_rate = None;
    let mut channels = None;
    let mut time_base = None;
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("sample_rate=") {
            sample_rate = Some(
                value
                    .parse::<u32>()
                    .map_err(|_| format!("sample_rate {value}"))?,
            );
        } else if let Some(value) = line.strip_prefix("channels=") {
            channels = Some(
                value
                    .parse::<u16>()
                    .map_err(|_| format!("channels {value}"))?,
            );
        } else if let Some(value) = line.strip_prefix("time_base=") {
            time_base = Some(parse_time_base(value)?);
        }
    }
    let sample_rate = sample_rate.ok_or_else(|| "ffprobe stream missing sample_rate".to_owned())?;
    let channels = channels.ok_or_else(|| "ffprobe stream missing channels".to_owned())?;
    let (time_base_num, time_base_den) =
        time_base.ok_or_else(|| "ffprobe stream missing time_base".to_owned())?;
    Ok(StreamClock {
        sample_rate,
        channels,
        time_base_num,
        time_base_den,
    })
}

fn parse_time_base(value: &str) -> Result<(i64, i64), String> {
    let (num, den) = value
        .split_once('/')
        .ok_or_else(|| format!("time_base {value}"))?;
    let num = num
        .parse::<i64>()
        .map_err(|_| format!("time_base numerator {value}"))?;
    let den = den
        .parse::<i64>()
        .map_err(|_| format!("time_base denominator {value}"))?;
    if den == 0 {
        return Err("time_base denominator is 0".to_owned());
    }
    Ok((num, den))
}

fn parse_i64(value: Option<&str>, index: usize, field: &str) -> Result<i64, String> {
    let value = value.ok_or_else(|| format!("ffprobe packet {index} missing {field}"))?;
    value
        .parse::<i64>()
        .map_err(|_| format!("ffprobe packet {index} {field} {value}"))
}

fn parse_usize(value: Option<&str>, index: usize, field: &str) -> Result<usize, String> {
    let value = value.ok_or_else(|| format!("ffprobe packet {index} missing {field}"))?;
    value
        .parse::<usize>()
        .map_err(|_| format!("ffprobe packet {index} {field} {value}"))
}

fn print_report(
    file: &[u8],
    clock: &StreamClock,
    rows: &[Row],
    mux_bytes: &[u8],
    mux_rows: &[Row],
    cutlass: &Times,
    ffprobe_time: &Times,
) {
    println!("sample lavfi {LAVFI} codec aac bitrate 64k format adts");
    println!("file_bytes {}", file.len());
    println!("frames {}", rows.len());
    println!(
        "sample_rate {} channels {} time_base {}/{}",
        clock.sample_rate, clock.channels, clock.time_base_num, clock.time_base_den
    );
    println!("size_check payload_plus_header_equals_ffprobe_size");
    println!("pts_check presentation_ticks_match");
    println!("index pts duration payload header ffprobe_pts ffprobe_duration ffprobe_size");
    for (index, row) in rows.iter().enumerate() {
        print_row("frame", index, row);
    }
    println!("mux_bytes {}", mux_bytes.len());
    println!("mux_frames {}", mux_rows.len());
    println!("mux_size_check payload_plus_header_equals_ffprobe_size");
    println!("mux_pts_check presentation_ticks_match");
    for (index, row) in mux_rows.iter().enumerate() {
        print_row("mux", index, row);
    }
    println!("timing median of 3 after 1 warmup");
    print_times("cutlass_read_us", cutlass);
    print_times("ffprobe_read_us", ffprobe_time);
    println!("ffprobe time includes process startup");
}

fn print_row(kind: &str, index: usize, row: &Row) {
    println!(
        "{kind} {index} {} {} {} {} {} {} {}",
        row.pts,
        row.duration,
        row.payload,
        row.header,
        row.ffprobe_pts,
        row.ffprobe_duration,
        row.ffprobe_size
    );
}

fn print_times(label: &str, times: &Times) {
    let samples = times
        .samples
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "{label} median {} min {} max {} samples {samples}",
        times.median, times.min, times.max
    );
}

fn find_bin(name: &str) -> Option<PathBuf> {
    ["/opt/homebrew/bin", "/usr/local/bin", ""]
        .into_iter()
        .map(|dir| {
            if dir.is_empty() {
                PathBuf::from(name)
            } else {
                PathBuf::from(dir).join(name)
            }
        })
        .find(|path| tool_runs(path))
}

fn tool_runs(path: &Path) -> bool {
    Command::new(path)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}
