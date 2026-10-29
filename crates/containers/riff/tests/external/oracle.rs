//! Files from ffmpeg, read back by this crate, ffprobe, and Python's wave module.
//!
//! ffprobe is libavformat. Python's `wave` module is a separate reader.
//! `afinfo` is AudioToolbox when it is on PATH. GStreamer is tried with a
//! short timeout and skipped when the plugin scanner does not finish.

use std::io::Cursor;
use std::time::Duration;

use crate::support::{ffprobe_stream, output, output_timeout, python_wave, tool_on_path};
use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration, TrackId};
use cutlass_riff::{WavDemux, WavMux};

#[test]
fn ffmpeg_pcm_matches_ffprobe_and_python_wave() {
    if !tool_on_path("ffmpeg") || !tool_on_path("ffprobe") {
        eprintln!("skipped: ffmpeg and ffprobe are not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join(format!("cutlass-riff-oracle-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp");

    let pcm = dir.join("tone.wav");
    ffmpeg_wav(
        &pcm,
        &[
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.05:sample_rate=44100",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
        ],
    );
    let ours = read_packets(&pcm);
    let probed = ffprobe_stream(&pcm);
    assert_eq!(probed.sample_rate, 44_100);
    assert_eq!(probed.channels, 2);
    assert_eq!(probed.codec_name, "pcm_s16le");
    assert_eq!(
        probed.duration_ts,
        Some(u64::try_from(ours.len()).expect("frames"))
    );
    assert!(
        ours.iter()
            .all(|packet| packet.timescale().ticks_per_second() == 44_100)
    );

    let side = dir.join("tone.raw");
    let wave = python_wave(&pcm, &side).expect("python wave");
    assert_eq!(wave, (2, 44_100, 2, u64::try_from(ours.len()).expect("n")));
    let raw = std::fs::read(&side).expect("raw");
    let mut gathered = Vec::new();
    for packet in &ours {
        gathered.extend_from_slice(packet.bytes());
    }
    assert_eq!(gathered, raw);

    if tool_on_path("afinfo") {
        let info = output("afinfo", &[&pcm.display().to_string()]);
        let text = String::from_utf8_lossy(&info.stdout);
        assert!(text.contains("44100"), "{text}");
    }

    try_gstreamer(&pcm);

    let floats = dir.join("float.wav");
    ffmpeg_wav(
        &floats,
        &[
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.02:sample_rate=48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_f32le",
        ],
    );
    let float_packets = read_packets(&floats);
    let float_probe = ffprobe_stream(&floats);
    assert_eq!(float_probe.sample_rate, 48_000);
    assert_eq!(float_probe.codec_name, "pcm_f32le");
    assert_eq!(
        float_probe.duration_ts,
        Some(u64::try_from(float_packets.len()).expect("float frames"))
    );
    let float_demux = open_file(&floats);
    assert!(float_demux.format().is_float());

    let muxed = dir.join("muxed.wav");
    write_mux(&muxed, &gathered);
    let mux_wave = python_wave(&muxed, &dir.join("muxed.raw")).expect("muxed wave");
    assert_eq!(mux_wave.0, 2);
    assert_eq!(mux_wave.1, 44_100);
    assert_eq!(
        std::fs::read(dir.join("muxed.raw")).expect("mux raw"),
        gathered
    );
    let mux_probe = ffprobe_stream(&muxed);
    assert_eq!(mux_probe.sample_rate, 44_100);
    assert_eq!(mux_probe.channels, 2);
    assert_eq!(mux_probe.duration_ts, Some(mux_wave.3));

    let rf64 = dir.join("rf64.wav");
    write_rf64(&rf64, &gathered);
    let rf64_probe = ffprobe_stream(&rf64);
    assert_eq!(rf64_probe.sample_rate, 44_100);
    assert_eq!(rf64_probe.channels, 2);
    assert_eq!(
        rf64_probe.duration_ts,
        Some(u64::try_from(ours.len()).expect("rf64"))
    );

    let _ = std::fs::remove_dir_all(&dir);
}

fn ffmpeg_wav(path: &std::path::Path, args: &[&str]) {
    let mut full = vec!["-y", "-hide_banner", "-loglevel", "error"];
    full.extend_from_slice(args);
    full.push(path.to_str().expect("path"));
    let made = output("ffmpeg", &full);
    assert!(
        made.status.success(),
        "{}",
        String::from_utf8_lossy(&made.stderr)
    );
}

fn open_file(path: &std::path::Path) -> WavDemux<std::fs::File> {
    let file = std::fs::File::open(path).expect("open");
    let len = file.metadata().expect("meta").len();
    WavDemux::open(file, len).expect("demux")
}

fn read_packets(path: &std::path::Path) -> Vec<Packet> {
    let mut demux = open_file(path);
    let mut packets = Vec::new();
    while let Some(packet) = demux.read_packet().expect("packet") {
        assert_ne!(packet.timescale().ticks_per_second(), 1_000_000);
        assert!(packet.keyframe());
        packets.push(packet);
    }
    assert!(!packets.is_empty());
    packets
}

fn write_mux(path: &std::path::Path, pcm: &[u8]) {
    let mut file = std::fs::File::create(path).expect("create");
    let bytes = mux_bytes(pcm, false);
    std::io::Write::write_all(&mut file, &bytes).expect("write");
}

fn write_rf64(path: &std::path::Path, pcm: &[u8]) {
    let mut file = std::fs::File::create(path).expect("create");
    let bytes = mux_bytes(pcm, true);
    std::io::Write::write_all(&mut file, &bytes).expect("write");
}

fn mux_bytes(pcm: &[u8], rf64: bool) -> Vec<u8> {
    let params = StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 44_100, 2, Vec::new())
        .expect("pcm");
    let frames = i64::try_from(pcm.len() / 4).expect("frames");
    let timestamp = Timestamp::try_new(Some(0), Some(0), Some(frames)).expect("time");
    let packet =
        Packet::try_new(0, pcm.to_vec(), timestamp, Timescale::new(44_100), true).expect("packet");
    let mut mux = if rf64 {
        WavMux::rf64(Cursor::new(Vec::new()))
    } else {
        WavMux::new(Cursor::new(Vec::new()))
    };
    mux.add_stream(NewStream {
        params,
        timescale: Timescale::new(44_100),
        language: None,
    })
    .expect("stream");
    mux.write_packet(&packet).expect("packet");
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(44_100),
        ticks: frames,
    }])
    .expect("finish");
    mux.into_inner().into_inner()
}

fn try_gstreamer(path: &std::path::Path) {
    if !tool_on_path("gst-launch-1.0") {
        eprintln!("gstreamer: not on PATH");
        return;
    }
    let Some(out) = output_timeout(
        "gst-launch-1.0",
        &[
            "-q",
            "filesrc",
            &format!("location={}", path.display()),
            "!",
            "wavparse",
            "!",
            "fakesink",
        ],
        Duration::from_secs(8),
    ) else {
        eprintln!("gstreamer: not available (plugin scanner did not finish)");
        return;
    };
    assert!(
        out.status.success(),
        "gst-launch: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
