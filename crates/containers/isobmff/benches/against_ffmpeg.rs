//! Packet count, size, and presentation time against ffprobe.
//!
//! ffmpeg writes one short MP4. This bench opens it with `IsoBmff`, checks
//! every packet against `ffprobe -show_packets`, then times a fresh open and a
//! full read. The ffprobe time includes process startup. Pixels are not decoded.

#![allow(clippy::print_stdout)]

use cutlass_isobmff::{Demux, FileSource, IsoBmff, IsoBmffMux, Mux, MuxOptions, Packet, Stream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

const FFMPEG: &str = "/opt/homebrew/bin/ffmpeg";
const FFPROBE: &str = "/opt/homebrew/bin/ffprobe";
const MUX_PACKETS: usize = 4;

fn main() -> Result<(), String> {
    let ffmpeg = PathBuf::from(FFMPEG);
    if !ffmpeg.is_file() {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    }
    let ffprobe = PathBuf::from(FFPROBE);
    if !ffprobe.is_file() {
        return Err("ffprobe is missing beside ffmpeg".to_string());
    }

    let version = first_line(&ffmpeg, &["-version"])?;
    let cpu = first_line(Path::new("sysctl"), &["-n", "machdep.cpu.brand_string"])?;
    let today = first_line(Path::new("date"), &[])?;

    let dir = std::env::temp_dir().join("cutlass-isobmff-bench");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
    let sample = dir.join("sample.mp4");
    let command = write_sample(&ffmpeg, &sample)?;

    let listed = ffprobe_packets(&ffprobe, &sample)?;
    let (streams, packets) = read_all(&sample)?;
    let groups = groups_of(&streams, &packets)?;
    compare_listed(&groups, &listed)?;
    let expected_bytes: usize = packets.iter().map(|packet| packet.payload.len()).sum();

    let _warmup = time_read(&sample, packets.len(), expected_bytes)?;
    let cutlass = [
        time_read(&sample, packets.len(), expected_bytes)?,
        time_read(&sample, packets.len(), expected_bytes)?,
        time_read(&sample, packets.len(), expected_bytes)?,
    ];
    let _probe_warmup = time_ffprobe(&ffprobe, &sample)?;
    let probe = [
        time_ffprobe(&ffprobe, &sample)?,
        time_ffprobe(&ffprobe, &sample)?,
        time_ffprobe(&ffprobe, &sample)?,
    ];
    let muxed = mux_prefix(&ffprobe, &dir, &streams, &packets)?;

    println!("ffmpeg_version: {version}");
    println!("cpu: {cpu}");
    println!("date: {today}");
    println!("command: {command}");
    println!("streams: {}", streams.len());
    println!("packets: {}", packets.len());
    for (index, (stream, group)) in streams.iter().zip(&groups).enumerate() {
        println!(
            "stream {index}: {:?} timescale {} packets {}",
            stream.params.codec,
            stream.params.timescale.get(),
            group.len()
        );
    }
    println!("compare: packet count, sizes, and pts match ffprobe -show_packets");
    println!("sample: 1 file, median of 3 after 1 warmup");
    println!("cutlass_open_read_s: {}", show(cutlass));
    println!("ffprobe_show_packets_s: {}", show(probe));
    println!("ffmpeg time includes process startup");
    println!("mux_packets: {muxed}");
    println!("mux_compare: packet count, sizes, and pts match ffprobe -show_packets");
    Ok(())
}

fn write_sample(ffmpeg: &Path, path: &Path) -> Result<String, String> {
    let path_arg = path.to_str().ok_or("sample path is not utf-8")?;
    let args = [
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=64x64:rate=10:duration=0.4",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:sample_rate=48000:duration=0.4",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "10",
        "-bf",
        "0",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-shortest",
        "-use_editlist",
        "0",
        path_arg,
    ];
    let command = join_command(ffmpeg, &args);
    let output = Command::new(ffmpeg)
        .args(args)
        .output()
        .map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(command)
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

fn read_all(path: &Path) -> Result<(Vec<Stream>, Vec<Packet>), String> {
    let mut source = FileSource::open(path).map_err(|err| err.to_string())?;
    let mut movie = IsoBmff::open(&mut source).map_err(|err| err.to_string())?;
    let streams = movie.streams().to_vec();
    let mut packets = Vec::new();
    while let Some(packet) = movie
        .read_packet(&mut source)
        .map_err(|err| err.to_string())?
    {
        packets.push(packet);
    }
    if packets.is_empty() {
        return Err("the sample has no packets".to_string());
    }
    Ok((streams, packets))
}

fn time_read(
    path: &Path,
    expected_count: usize,
    expected_bytes: usize,
) -> Result<Duration, String> {
    let started = Instant::now();
    let mut source = FileSource::open(path).map_err(|err| err.to_string())?;
    let mut movie = IsoBmff::open(&mut source).map_err(|err| err.to_string())?;
    let mut count = 0usize;
    let mut bytes = 0usize;
    while let Some(packet) = movie
        .read_packet(&mut source)
        .map_err(|err| err.to_string())?
    {
        count += 1;
        bytes += packet.payload.len();
    }
    let elapsed = started.elapsed();
    if count != expected_count || bytes != expected_bytes {
        return Err(format!(
            "read {count} packets and {bytes} bytes, expected {expected_count} packets and {expected_bytes} bytes"
        ));
    }
    Ok(elapsed)
}

fn time_ffprobe(ffprobe: &Path, path: &Path) -> Result<Duration, String> {
    let started = Instant::now();
    let output = ffprobe_command(ffprobe, path)
        .output()
        .map_err(|err| err.to_string())?;
    let elapsed = started.elapsed();
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let text = String::from_utf8(output.stdout).map_err(|err| err.to_string())?;
    if !text.contains("[PACKET]") {
        return Err("ffprobe listed no packets".to_string());
    }
    Ok(elapsed)
}

fn ffprobe_packets(ffprobe: &Path, path: &Path) -> Result<Vec<Listed>, String> {
    let output = ffprobe_command(ffprobe, path)
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let text = String::from_utf8(output.stdout).map_err(|err| err.to_string())?;
    parse_packets(&text)
}

fn ffprobe_command(ffprobe: &Path, path: &Path) -> Command {
    let mut command = Command::new(ffprobe);
    command
        .args(["-hide_banner", "-v", "error", "-show_packets"])
        .arg(path);
    command
}

struct Listed {
    stream_index: usize,
    pts: i64,
    size: usize,
}

struct Seen {
    pts: i64,
    size: usize,
}

fn groups_of(streams: &[Stream], packets: &[Packet]) -> Result<Vec<Vec<Seen>>, String> {
    let mut groups = Vec::new();
    let mut assigned = 0usize;
    for stream in streams {
        let mut group = Vec::new();
        for packet in packets {
            if packet.track_id == stream.params.track_id {
                group.push(Seen {
                    pts: packet.pts,
                    size: packet.payload.len(),
                });
                assigned += 1;
            }
        }
        groups.push(group);
    }
    if assigned != packets.len() {
        return Err("a packet track is not in the movie".to_string());
    }
    Ok(groups)
}

fn compare_listed(groups: &[Vec<Seen>], listed: &[Listed]) -> Result<(), String> {
    let expected: usize = groups.iter().map(Vec::len).sum();
    if expected != listed.len() {
        return Err(format!(
            "packet count cutlass {expected} ffprobe {}",
            listed.len()
        ));
    }
    if listed
        .iter()
        .any(|packet| packet.stream_index >= groups.len())
    {
        return Err("ffprobe listed a stream this demuxer did not open".to_string());
    }
    for (stream_index, group) in groups.iter().enumerate() {
        let mut found = 0usize;
        for packet in listed
            .iter()
            .filter(|packet| packet.stream_index == stream_index)
        {
            let Some(ours) = group.get(found) else {
                return Err(format!("stream {stream_index} has an extra ffprobe packet"));
            };
            if ours.pts != packet.pts || ours.size != packet.size {
                return Err(format!(
                    "stream {stream_index} packet {found} cutlass pts {} size {} ffprobe pts {} size {}",
                    ours.pts, ours.size, packet.pts, packet.size
                ));
            }
            found += 1;
        }
        if found != group.len() {
            return Err(format!(
                "stream {stream_index} count cutlass {} ffprobe {found}",
                group.len()
            ));
        }
    }
    Ok(())
}

fn parse_packets(text: &str) -> Result<Vec<Listed>, String> {
    let mut listed = Vec::new();
    let mut stream_index = None;
    let mut pts = None;
    let mut size = None;
    let mut open = false;
    for line in text.lines() {
        let line = line.trim();
        if line == "[PACKET]" {
            if open {
                return Err("nested packet".to_string());
            }
            open = true;
            stream_index = None;
            pts = None;
            size = None;
            continue;
        }
        if line == "[/PACKET]" {
            if !open {
                return Err("packet ended before it started".to_string());
            }
            listed.push(Listed {
                stream_index: stream_index.ok_or("packet missing stream_index")?,
                pts: pts.ok_or("packet missing pts")?,
                size: size.ok_or("packet missing size")?,
            });
            open = false;
            continue;
        }
        if !open {
            continue;
        }
        if let Some(value) = field(line, "stream_index") {
            stream_index = Some(parse_usize(value)?);
        } else if let Some(value) = field(line, "pts") {
            pts = Some(parse_i64(value)?);
        } else if let Some(value) = field(line, "size") {
            size = Some(parse_usize(value)?);
        }
    }
    if open {
        return Err("unclosed packet".to_string());
    }
    if listed.is_empty() {
        return Err("ffprobe listed no packets".to_string());
    }
    Ok(listed)
}

fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let (name, value) = line.split_once('=')?;
    (name == key).then_some(value)
}

fn parse_i64(value: &str) -> Result<i64, String> {
    value.parse().map_err(|_| format!("bad integer {value}"))
}

fn parse_usize(value: &str) -> Result<usize, String> {
    value.parse().map_err(|_| format!("bad integer {value}"))
}

fn mux_prefix(
    ffprobe: &Path,
    dir: &Path,
    streams: &[Stream],
    packets: &[Packet],
) -> Result<usize, String> {
    let mut mux = IsoBmffMux::new(std::io::Cursor::new(Vec::new()), MuxOptions::default());
    let mut plan = Vec::new();
    for stream in streams {
        let mut prefix = Vec::new();
        for packet in packets
            .iter()
            .filter(|packet| packet.track_id == stream.params.track_id)
            .take(MUX_PACKETS)
        {
            prefix.push(packet.clone());
        }
        let prefix = rebase(&prefix)?;
        let id = mux
            .add_stream(&stream.params)
            .map_err(|err| err.to_string())?;
        plan.push((id, prefix));
    }
    let mut groups = Vec::new();
    let mut written = 0usize;
    for (id, prefix) in &plan {
        let mut seen = Vec::new();
        for packet in prefix {
            let mut packet = packet.clone();
            packet.track_id = *id;
            mux.write_packet(&packet).map_err(|err| err.to_string())?;
            seen.push(Seen {
                pts: packet.pts,
                size: packet.payload.len(),
            });
            written += 1;
        }
        groups.push(seen);
    }
    let bytes = mux
        .finish_into()
        .map_err(|err| err.to_string())?
        .into_inner();
    let path = dir.join("remux.mp4");
    std::fs::write(&path, bytes).map_err(|err| err.to_string())?;
    let listed = ffprobe_packets(ffprobe, &path)?;
    compare_listed(&groups, &listed)?;
    Ok(written)
}

fn rebase(packets: &[Packet]) -> Result<Vec<Packet>, String> {
    let Some(first) = packets.first() else {
        return Err("a stream has no packets to mux".to_string());
    };
    let origin = first.dts;
    let mut next = 0i64;
    let mut out = Vec::with_capacity(packets.len());
    for packet in packets {
        let dts = packet
            .dts
            .checked_sub(origin)
            .ok_or("decode time underflow")?;
        if dts != next {
            return Err(format!("decode time {dts} does not follow {next}"));
        }
        let pts = packet
            .pts
            .checked_sub(origin)
            .ok_or("presentation time underflow")?;
        next = dts
            .checked_add(i64::from(packet.duration))
            .ok_or("duration overflow")?;
        let mut copy = packet.clone();
        copy.dts = dts;
        copy.pts = pts;
        out.push(copy);
    }
    Ok(out)
}

fn show(samples: [Duration; 3]) -> String {
    format!(
        "{} {} {} median {}",
        secs(samples[0]),
        secs(samples[1]),
        secs(samples[2]),
        secs(median(samples))
    )
}

fn median(samples: [Duration; 3]) -> Duration {
    let mut ordered = samples;
    ordered.sort_unstable();
    ordered[1]
}

fn secs(duration: Duration) -> String {
    format!("{:.6}", duration.as_secs_f64())
}

fn join_command(bin: &Path, args: &[&str]) -> String {
    let mut parts = Vec::with_capacity(args.len().saturating_add(1));
    parts.push(bin.display().to_string());
    for arg in args {
        parts.push((*arg).to_string());
    }
    parts.join(" ")
}

fn first_line(program: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "{} failed: {}",
            program.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let text = String::from_utf8(output.stdout).map_err(|err| err.to_string())?;
    let line = text.lines().next().unwrap_or("").trim();
    if line.is_empty() {
        return Err(format!("{} printed nothing", program.display()));
    }
    Ok(line.to_string())
}
