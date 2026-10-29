//! Time one full WAV read against ffmpeg.
//!
//! ffmpeg writes one short 16-bit stereo file. Stream parameters, the
//! timestamp span, and payload bytes are compared to ffprobe and ffmpeg
//! before anything is timed. The ffmpeg row includes process startup.
//! A missing ffmpeg prints `skip: ffmpeg is not installed` and exits 0.

#![allow(clippy::print_stdout)]

use std::fs::File;
use std::io::{Cursor, Read, Seek};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration, TrackId};
use cutlass_riff::{WavDemux, WavMux};

const REPS: usize = 3;

struct Opened {
    rate: u32,
    channels: u16,
    packets: u64,
    samples: u64,
    first_pts: i64,
    last_pts: i64,
    payload: Vec<u8>,
}

struct Probe {
    sample_rate: u32,
    channels: u16,
    duration_ts: u64,
    start_pts: Option<i64>,
    codec_name: String,
    packets: u64,
    first_pts: i64,
    last_pts: i64,
    duration_sum: u64,
}

struct Timed {
    median: Duration,
    min: Duration,
    max: Duration,
    bytes: u64,
}

struct Meta {
    cpu: String,
    date: String,
    version: String,
    ffmpeg: PathBuf,
    ffprobe: PathBuf,
}

fn main() -> Result<(), String> {
    let Some(ffmpeg) = find_ffmpeg() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let dir = std::env::temp_dir().join(format!("cutlass-riff-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
    let outcome = measure(&dir, &ffmpeg);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn measure(dir: &Path, ffmpeg: &Path) -> Result<(), String> {
    let ffprobe = find_ffprobe(ffmpeg)?;
    let wav = dir.join("sample.wav");
    write_sample(ffmpeg, &wav)?;
    let opened = read_wav(&wav)?;
    let probed = probe_file(&ffprobe, &wav)?;
    let foreign = ffmpeg_pcm(ffmpeg, &wav)?;
    compare(&opened, &probed, &foreign)?;
    let muxed = dir.join("muxed.wav");
    let bytes = mux_pcm(&opened)?;
    std::fs::write(&muxed, &bytes).map_err(|err| err.to_string())?;
    let mux_probe = probe_file(&ffprobe, &muxed)?;
    compare_mux(&opened, &mux_probe)?;
    let cutlass = time_reps(|| read_all(&wav))?;
    let ffmpeg_time = time_reps(|| {
        let pcm = ffmpeg_pcm(ffmpeg, &wav)?;
        byte_len(&pcm)
    })?;
    same_length("cutlass", cutlass.bytes, &opened.payload)?;
    same_length("ffmpeg", ffmpeg_time.bytes, &foreign)?;
    let meta = Meta {
        cpu: command_text("sysctl", &["-n", "machdep.cpu.brand_string"])?,
        date: command_text("date", &[])?,
        version: ffmpeg_version(ffmpeg)?,
        ffmpeg: ffmpeg.to_path_buf(),
        ffprobe,
    };
    print_report(&meta, &opened, &probed, &mux_probe, &cutlass, &ffmpeg_time);
    Ok(())
}

fn write_sample(ffmpeg: &Path, wav: &Path) -> Result<(), String> {
    let mut command = Command::new(ffmpeg);
    command.args([
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1:sample_rate=44100",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
    ]);
    command.arg(wav);
    let _ = ffmpeg_output(command)?;
    Ok(())
}

fn read_wav(path: &Path) -> Result<Opened, String> {
    let mut demux = open_wav(path)?;
    let (rate, channels) = stream_audio(&demux)?;
    let mut packets = 0u64;
    let mut samples = 0u64;
    let mut first_pts = None;
    let mut last_pts = 0i64;
    let mut payload = Vec::new();
    while let Some(packet) = demux.read_packet().map_err(|err| err.to_string())? {
        let start = packet
            .timestamp()
            .presentation()
            .ok_or_else(|| "packet has no presentation time".to_owned())?;
        let duration = packet_duration(&packet)?;
        if packet.timescale().ticks_per_second() != rate {
            return Err(format!(
                "packet timescale {} is not {rate}",
                packet.timescale().ticks_per_second()
            ));
        }
        let at = u64::try_from(start).map_err(|_| format!("presentation {start} is negative"))?;
        if at != samples {
            return Err(format!(
                "presentation {start} does not follow sample {samples}"
            ));
        }
        if first_pts.is_none() {
            first_pts = Some(start);
        }
        last_pts = start;
        samples = samples
            .checked_add(duration)
            .ok_or_else(|| "sample count overflow".to_owned())?;
        packets = packets
            .checked_add(1)
            .ok_or_else(|| "packet count overflow".to_owned())?;
        payload.extend_from_slice(packet.bytes());
    }
    let first_pts = first_pts.ok_or_else(|| "wav has no packets".to_owned())?;
    Ok(Opened {
        rate,
        channels,
        packets,
        samples,
        first_pts,
        last_pts,
        payload,
    })
}

fn read_all(path: &Path) -> Result<u64, String> {
    let mut demux = open_wav(path)?;
    let mut total = 0u64;
    while let Some(packet) = demux.read_packet().map_err(|err| err.to_string())? {
        total = total
            .checked_add(byte_len(packet.bytes())?)
            .ok_or_else(|| "payload size overflow".to_owned())?;
    }
    Ok(std::hint::black_box(total))
}

fn open_wav(path: &Path) -> Result<WavDemux<File>, String> {
    let file = File::open(path).map_err(|err| err.to_string())?;
    let length = file.metadata().map_err(|err| err.to_string())?.len();
    WavDemux::open(file, length).map_err(|err| err.to_string())
}

fn stream_audio<R: Read + Seek>(demux: &WavDemux<R>) -> Result<(u32, u16), String> {
    let stream = demux
        .streams()
        .first()
        .ok_or_else(|| "wav has no audio stream".to_owned())?;
    let rate = stream
        .params()
        .sample_rate()
        .ok_or_else(|| "stream has no sample rate".to_owned())?;
    let channels = stream
        .params()
        .channels()
        .ok_or_else(|| "stream has no channel count".to_owned())?;
    if stream.timescale().ticks_per_second() != rate {
        return Err(format!(
            "stream timescale {} is not {rate}",
            stream.timescale().ticks_per_second()
        ));
    }
    Ok((rate, channels))
}

fn packet_duration(packet: &Packet) -> Result<u64, String> {
    let duration = packet
        .timestamp()
        .duration()
        .ok_or_else(|| "packet has no duration".to_owned())?;
    u64::try_from(duration).map_err(|_| "packet duration is negative".to_owned())
}

fn compare(opened: &Opened, probe: &Probe, foreign: &[u8]) -> Result<(), String> {
    if probe.codec_name != "pcm_s16le" {
        return Err(format!("ffprobe codec is {}", probe.codec_name));
    }
    if opened.rate != probe.sample_rate {
        return Err(format!(
            "sample rate {} != ffprobe {}",
            opened.rate, probe.sample_rate
        ));
    }
    if opened.channels != probe.channels {
        return Err(format!(
            "channels {} != ffprobe {}",
            opened.channels, probe.channels
        ));
    }
    if opened.samples != probe.duration_ts || opened.samples != probe.duration_sum {
        return Err(format!(
            "samples {} != ffprobe duration_ts {} duration_sum {}",
            opened.samples, probe.duration_ts, probe.duration_sum
        ));
    }
    if opened.first_pts != probe.first_pts {
        return Err(format!(
            "first pts {} != ffprobe {}",
            opened.first_pts, probe.first_pts
        ));
    }
    let end = span_end(opened.first_pts, opened.samples)?;
    let foreign_end = span_end(probe.first_pts, probe.duration_sum)?;
    if end != foreign_end {
        return Err(format!(
            "timestamp span ends at {end}, ffprobe ends at {foreign_end}"
        ));
    }
    if opened.payload != foreign {
        return Err(format!(
            "payload {} bytes != ffmpeg {} bytes",
            opened.payload.len(),
            foreign.len()
        ));
    }
    Ok(())
}

fn compare_mux(opened: &Opened, probe: &Probe) -> Result<(), String> {
    if probe.codec_name != "pcm_s16le" {
        return Err(format!("mux ffprobe codec is {}", probe.codec_name));
    }
    if probe.sample_rate != opened.rate
        || probe.channels != opened.channels
        || probe.duration_ts != opened.samples
        || probe.duration_sum != opened.samples
        || probe.first_pts != opened.first_pts
    {
        return Err(format!(
            "mux ffprobe rate {} channels {} duration_ts {} duration_sum {} first_pts {}",
            probe.sample_rate,
            probe.channels,
            probe.duration_ts,
            probe.duration_sum,
            probe.first_pts
        ));
    }
    Ok(())
}

fn mux_pcm(opened: &Opened) -> Result<Vec<u8>, String> {
    let ticks = i64::try_from(opened.samples).map_err(|_| "frame count does not fit".to_owned())?;
    let params = StreamParams::try_audio(
        StreamCodec::Known(CodecId::Pcm),
        opened.rate,
        opened.channels,
        Vec::new(),
    )
    .map_err(|err| err.to_string())?;
    let timestamp =
        Timestamp::try_new(Some(0), Some(0), Some(ticks)).map_err(|err| err.to_string())?;
    let packet = Packet::try_new(
        0,
        opened.payload.clone(),
        timestamp,
        Timescale::new(opened.rate),
        true,
    )
    .map_err(|err| err.to_string())?;
    let mut mux = WavMux::new(Cursor::new(Vec::new()));
    mux.add_stream(NewStream {
        params,
        timescale: Timescale::new(opened.rate),
        language: None,
    })
    .map_err(|err| err.to_string())?;
    mux.write_packet(&packet).map_err(|err| err.to_string())?;
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(opened.rate),
        ticks,
    }])
    .map_err(|err| err.reason().to_string())?;
    Ok(mux.into_inner().into_inner())
}

fn ffmpeg_pcm(ffmpeg: &Path, wav: &Path) -> Result<Vec<u8>, String> {
    let mut command = Command::new(ffmpeg);
    command
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(wav)
        .args(["-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]);
    Ok(ffmpeg_output(command)?.stdout)
}

fn probe_file(ffprobe: &Path, wav: &Path) -> Result<Probe, String> {
    let mut command = Command::new(ffprobe);
    command.args([
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,duration_ts,codec_name,start_pts:packet=pts,duration",
        "-of",
        "compact",
    ]);
    command.arg(wav);
    let output = ffmpeg_output(command)?;
    let text = String::from_utf8(output.stdout).map_err(|_| "ffprobe output is not utf-8")?;
    parse_probe(&text)
}

fn parse_probe(text: &str) -> Result<Probe, String> {
    let mut sample_rate = None;
    let mut channels = None;
    let mut duration_ts = None;
    let mut start_pts = None;
    let mut codec_name = None;
    let mut packets = 0u64;
    let mut first_pts = None;
    let mut last_pts = None;
    let mut duration_sum = 0u64;
    let mut cursor = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("stream|") {
            read_stream(
                rest,
                &mut sample_rate,
                &mut channels,
                &mut duration_ts,
                &mut start_pts,
                &mut codec_name,
            )?;
        } else if let Some(rest) = line.strip_prefix("packet|") {
            let (pts, duration) = read_packet(rest)?;
            if let Some(expected) = cursor
                && pts != expected
            {
                return Err(format!("ffprobe pts {pts} does not follow {expected}"));
            }
            if first_pts.is_none() {
                first_pts = Some(pts);
            }
            last_pts = Some(pts);
            duration_sum = duration_sum
                .checked_add(duration)
                .ok_or_else(|| "ffprobe duration overflow".to_owned())?;
            packets = packets
                .checked_add(1)
                .ok_or_else(|| "ffprobe packet count overflow".to_owned())?;
            cursor = Some(
                pts.checked_add(i64::try_from(duration).map_err(|_| "duration does not fit")?)
                    .ok_or_else(|| "ffprobe timestamp overflow".to_owned())?,
            );
        }
    }
    Ok(Probe {
        sample_rate: sample_rate.ok_or_else(|| "ffprobe missed sample_rate".to_owned())?,
        channels: channels.ok_or_else(|| "ffprobe missed channels".to_owned())?,
        duration_ts: duration_ts.ok_or_else(|| "ffprobe missed duration_ts".to_owned())?,
        start_pts,
        codec_name: codec_name.ok_or_else(|| "ffprobe missed codec_name".to_owned())?,
        packets,
        first_pts: first_pts.ok_or_else(|| "ffprobe returned no packets".to_owned())?,
        last_pts: last_pts.ok_or_else(|| "ffprobe returned no packets".to_owned())?,
        duration_sum,
    })
}

fn read_stream(
    rest: &str,
    sample_rate: &mut Option<u32>,
    channels: &mut Option<u16>,
    duration_ts: &mut Option<u64>,
    start_pts: &mut Option<i64>,
    codec_name: &mut Option<String>,
) -> Result<(), String> {
    for part in rest.split('|') {
        if let Some(value) = part.strip_prefix("sample_rate=") {
            *sample_rate = Some(parse_u32(value)?);
        } else if let Some(value) = part.strip_prefix("channels=") {
            *channels = Some(parse_u16(value)?);
        } else if let Some(value) = part.strip_prefix("duration_ts=") {
            *duration_ts = Some(parse_u64(value)?);
        } else if let Some(value) = part.strip_prefix("codec_name=") {
            *codec_name = Some(value.to_owned());
        } else if let Some(value) = part.strip_prefix("start_pts=") {
            *start_pts = parse_optional_i64(value)?;
        }
    }
    Ok(())
}

fn read_packet(rest: &str) -> Result<(i64, u64), String> {
    let mut pts = None;
    let mut duration = None;
    for part in rest.split('|') {
        if let Some(value) = part.strip_prefix("pts=") {
            pts = Some(parse_i64(value)?);
        } else if let Some(value) = part.strip_prefix("duration=") {
            duration = Some(parse_u64(value)?);
        }
    }
    Ok((
        pts.ok_or_else(|| "ffprobe packet has no pts".to_owned())?,
        duration.ok_or_else(|| "ffprobe packet has no duration".to_owned())?,
    ))
}

fn time_reps(mut once: impl FnMut() -> Result<u64, String>) -> Result<Timed, String> {
    let _ = once()?;
    let mut times = Vec::with_capacity(REPS);
    let mut bytes = 0u64;
    for _ in 0..REPS {
        let started = Instant::now();
        bytes = once()?;
        times.push(started.elapsed());
    }
    let min = times.iter().copied().min().unwrap_or_default();
    let max = times.iter().copied().max().unwrap_or_default();
    Ok(Timed {
        median: median(&mut times),
        min,
        max,
        bytes,
    })
}

fn median(times: &mut [Duration]) -> Duration {
    times.sort();
    times.get(times.len() / 2).copied().unwrap_or_default()
}

fn print_report(
    meta: &Meta,
    opened: &Opened,
    probe: &Probe,
    mux: &Probe,
    cutlass: &Timed,
    ffmpeg: &Timed,
) {
    println!("cpu: {}", meta.cpu);
    println!("date: {}", meta.date);
    println!("command: cargo bench -p cutlass-riff");
    println!("ffmpeg: {}", meta.ffmpeg.display());
    println!("ffprobe: {}", meta.ffprobe.display());
    println!("{}", meta.version);
    println!("sample: one 16-bit stereo WAV written by ffmpeg");
    println!("reps: 3 after 1 warmup, median of the timed passes");
    println!("ffmpeg time includes process startup");
    println!("accuracy codec cutlass=pcm ffprobe={}", probe.codec_name);
    println!(
        "accuracy sample_rate cutlass={} ffprobe={}",
        opened.rate, probe.sample_rate
    );
    println!(
        "accuracy channels cutlass={} ffprobe={}",
        opened.channels, probe.channels
    );
    println!(
        "accuracy packets cutlass={} ffprobe={}",
        opened.packets, probe.packets
    );
    println!(
        "accuracy samples={} ffprobe_duration_ts={} ffprobe_duration_sum={}",
        opened.samples, probe.duration_ts, probe.duration_sum
    );
    println!(
        "accuracy first_pts cutlass={} ffprobe={} last_pts cutlass={} ffprobe={} ffprobe_start_pts={}",
        opened.first_pts,
        probe.first_pts,
        opened.last_pts,
        probe.last_pts,
        pts_text(probe.start_pts)
    );
    println!(
        "accuracy span cutlass={}+{} ffprobe={}+{}",
        opened.first_pts, opened.samples, probe.first_pts, probe.duration_sum
    );
    println!(
        "accuracy payload_bytes cutlass={} ffmpeg={}",
        cutlass.bytes, ffmpeg.bytes
    );
    println!(
        "mux codec={} sample_rate={} channels={} duration_ts={} packets={} first_pts={}",
        mux.codec_name, mux.sample_rate, mux.channels, mux.duration_ts, mux.packets, mux.first_pts
    );
    println!();
    println!(
        "{:<8} {:<8} {:>12} {:>12} {:>12} {:>12}",
        "side", "op", "median", "min", "max", "bytes"
    );
    print_row("cutlass", cutlass);
    print_row("ffmpeg", ffmpeg);
}

fn print_row(side: &str, timed: &Timed) {
    println!(
        "{:<8} {:<8} {:>12} {:>12} {:>12} {:>12}",
        side,
        "read",
        micros(timed.median),
        micros(timed.min),
        micros(timed.max),
        timed.bytes
    );
}

fn micros(duration: Duration) -> String {
    format!("{} µs", duration.as_micros())
}

fn pts_text(value: Option<i64>) -> String {
    value
        .map(|pts| pts.to_string())
        .unwrap_or_else(|| "N/A".to_owned())
}

fn same_length(side: &str, timed: u64, bytes: &[u8]) -> Result<(), String> {
    let expected = byte_len(bytes)?;
    if timed == expected {
        Ok(())
    } else {
        Err(format!("{side} timed {timed} bytes, comparison {expected}"))
    }
}

fn span_end(first: i64, samples: u64) -> Result<i64, String> {
    let samples = i64::try_from(samples).map_err(|_| "sample count does not fit".to_owned())?;
    first
        .checked_add(samples)
        .ok_or_else(|| "timestamp end overflow".to_owned())
}

fn byte_len(bytes: &[u8]) -> Result<u64, String> {
    u64::try_from(bytes.len()).map_err(|_| "byte count does not fit".to_owned())
}

fn parse_u32(value: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|_| format!("bad integer {value}"))
}

fn parse_u16(value: &str) -> Result<u16, String> {
    value
        .parse::<u16>()
        .map_err(|_| format!("bad integer {value}"))
}

fn parse_u64(value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("bad integer {value}"))
}

fn parse_i64(value: &str) -> Result<i64, String> {
    value
        .parse::<i64>()
        .map_err(|_| format!("bad integer {value}"))
}

fn parse_optional_i64(value: &str) -> Result<Option<i64>, String> {
    if value == "N/A" {
        Ok(None)
    } else {
        Ok(Some(parse_i64(value)?))
    }
}

fn find_ffmpeg() -> Option<PathBuf> {
    [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ]
    .into_iter()
    .find(|candidate| tool_runs(candidate))
    .map(PathBuf::from)
}

fn find_ffprobe(ffmpeg: &Path) -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Some(parent) = ffmpeg.parent() {
        candidates.push(parent.join("ffprobe"));
    }
    candidates.push(PathBuf::from("/opt/homebrew/bin/ffprobe"));
    candidates.push(PathBuf::from("/usr/local/bin/ffprobe"));
    candidates.push(PathBuf::from("ffprobe"));
    candidates
        .into_iter()
        .find(|candidate| candidate.to_str().is_some_and(tool_runs))
        .ok_or_else(|| "ffprobe is not installed".to_owned())
}

fn tool_runs(candidate: &str) -> bool {
    Command::new(candidate)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn ffmpeg_output(mut command: Command) -> Result<std::process::Output, String> {
    command.stdin(Stdio::null());
    let output = command.output().map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(output)
    } else {
        Err(format!(
            "command exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn command_text(program: &str, args: &[&str]) -> Result<String, String> {
    let mut command = Command::new(program);
    command.args(args);
    let output = ffmpeg_output(command)?;
    let text =
        String::from_utf8(output.stdout).map_err(|_| format!("{program} output is not utf-8"))?;
    Ok(text.trim().to_owned())
}

fn ffmpeg_version(ffmpeg: &Path) -> Result<String, String> {
    let mut command = Command::new(ffmpeg);
    command.arg("-version");
    let output = ffmpeg_output(command)?;
    let text = String::from_utf8(output.stdout).map_err(|_| "ffmpeg version is not utf-8")?;
    text.lines()
        .next()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| "ffmpeg version is empty".to_owned())
}
