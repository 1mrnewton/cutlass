//! The muxer's files read back by ffprobe, ffmpeg, GStreamer, and this
//! crate's demuxer.
//!
//! Packets come from ffmpeg-written Opus files (libopus and the native
//! encoder), demuxed by this crate and muxed again by [`OggMux`]. The
//! output must give libavformat the same packets, times, pre-skip, and
//! payload bytes as the source, and ffmpeg and GStreamer must decode the
//! same number of samples from both, so end trimming survives.

use std::fs::File;
use std::path::Path;

use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale};
use cutlass_container::{Demux, Mux, MuxStatus, NewStream, TrackDuration};
use cutlass_ogg::OggMux;

use crate::oracle::{VARIANTS, check_against_gstreamer, duration, encode, open, pts, read_all};
use crate::support::{
    environment, ffmpeg_decoded_samples, ffmpeg_payload, ffprobe_packets, ffprobe_stream, scratch,
};

/// How the muxer learns the OpusHead.
#[derive(Clone, Copy, Debug)]
enum Config {
    Extradata,
    SetConfig,
}

const CASES: &[(&str, Config)] = &[
    ("libopus-stereo-96k", Config::Extradata),
    ("libopus-mono-32k-60ms", Config::SetConfig),
    ("libopus-120ms-24k", Config::Extradata),
    ("native-opus-mono-64k", Config::SetConfig),
    ("native-opus-stereo-128k", Config::Extradata),
    ("libopus-5.1-family-1", Config::SetConfig),
];

fn remux(source: &Path, out: &Path, config: Config) -> Vec<Packet> {
    let mut demux = open(source);
    let stream = demux.streams()[0].clone();
    let packets = read_all(&mut demux);
    let head = stream.params().extradata().to_vec();
    let channels = stream.params().channels().expect("channels");
    let extradata = match config {
        Config::Extradata => head.clone(),
        Config::SetConfig => Vec::new(),
    };
    let mut mux = OggMux::new(File::create(out).expect("create"));
    let track = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(
                StreamCodec::Known(CodecId::Opus),
                48_000,
                channels,
                extradata,
            )
            .expect("params"),
            timescale: Timescale::new(48_000),
            language: None,
        })
        .expect("add stream");
    if matches!(config, Config::SetConfig) {
        mux.set_config(track, &head).expect("set config");
    }
    let mut ticks = 0;
    for packet in &packets {
        mux.write_packet(packet).expect("write");
        mux.interleave().expect("interleave");
        ticks += duration(packet);
    }
    let finished = mux
        .finish(&[TrackDuration {
            track,
            timescale: Timescale::new(48_000),
            ticks,
        }])
        .expect("finish");
    assert!(finished.is_complete());
    assert_eq!(mux.status(), MuxStatus::Finished);
    packets
}

#[test]
fn ffmpeg_and_gstreamer_read_remuxed_files_like_the_source() {
    let Some(root) = environment("ffmpeg_and_gstreamer_read_remuxed_files_like_the_source") else {
        return;
    };
    let dir = scratch("mux-oracle");
    let mut checked = 0;
    for &(name, config) in CASES {
        let variant = VARIANTS
            .iter()
            .find(|variant| variant.name == name)
            .expect("variant");
        let Some(source) = encode(&root, &dir, variant) else {
            eprintln!("skipped {name}: no source file");
            continue;
        };
        let out = dir.join(format!("{name}.remux.opus"));
        let packets = remux(&source, &out, config);
        check_remux(&source, &out, name, &packets, &dir);
        checked += 1;
    }
    assert!(checked >= 5, "only {checked} cases ran");
    let _ = std::fs::remove_dir_all(&dir);
}

fn check_remux(source: &Path, out: &Path, name: &str, packets: &[Packet], dir: &Path) {
    // libavformat: stream facts, packets, times, and payload bytes.
    let before = ffprobe_stream(source);
    let after = ffprobe_stream(out);
    assert_eq!(after, before, "{name} ffprobe stream");
    let theirs = ffprobe_packets(out);
    assert_eq!(theirs, ffprobe_packets(source), "{name} ffprobe packets");
    assert_eq!(theirs.len(), packets.len(), "{name} packet count");
    for (index, (mine, probe)) in packets.iter().zip(&theirs).enumerate() {
        assert_eq!(
            pts(mine),
            probe.pts + after.initial_padding,
            "{name} packet {index} pts"
        );
        assert_eq!(mine.bytes().len(), probe.size, "{name} packet {index} size");
    }
    assert_eq!(
        ffmpeg_payload(out),
        ffmpeg_payload(source),
        "{name} payload"
    );
    let decoded = ffmpeg_decoded_samples(out);
    assert_eq!(
        decoded,
        ffmpeg_decoded_samples(source),
        "{name} decoded samples"
    );

    // GStreamer: packet sizes, granule ends, decoded samples.
    check_against_gstreamer(out, name, dir, packets, decoded);

    // This crate: the same packets, trim, and edit as the source.
    let mut original = open(source);
    let mut again = open(out);
    assert_eq!(
        again.streams()[0].trim(),
        original.streams()[0].trim(),
        "{name} trim"
    );
    assert_eq!(
        again.streams()[0].edits(),
        original.streams()[0].edits(),
        "{name} edit"
    );
    assert_eq!(
        again.opus(0).expect("opus").head,
        original.opus(0).expect("opus").head,
        "{name} OpusHead"
    );
    assert_eq!(
        again.tags(0).expect("tags").vendor,
        cutlass_ogg::VENDOR,
        "{name} vendor"
    );
    let round = read_all(&mut again);
    assert_eq!(round, read_all(&mut original), "{name} round trip");
    let source_len = std::fs::metadata(source).expect("meta").len();
    let out_len = std::fs::metadata(out).expect("meta").len();
    eprintln!(
        "{name}: {} packets remuxed, decoded {decoded}, source {source_len} bytes, remux {out_len} bytes",
        packets.len()
    );
}

#[test]
fn ffmpeg_reads_two_opus_streams_from_one_mux() {
    let Some(root) = environment("ffmpeg_reads_two_opus_streams_from_one_mux") else {
        return;
    };
    let dir = scratch("mux-two");
    let mut sources = Vec::new();
    for name in ["libopus-stereo-96k", "native-opus-mono-64k"] {
        let variant = VARIANTS
            .iter()
            .find(|variant| variant.name == name)
            .expect("variant");
        let Some(source) = encode(&root, &dir, variant) else {
            eprintln!("skipped: no source for {name}");
            return;
        };
        sources.push(source);
    }
    let out = dir.join("two.opus");
    let mut mux = OggMux::new(File::create(&out).expect("create"));
    let mut inputs = Vec::new();
    for source in &sources {
        let mut demux = open(source);
        let stream = demux.streams()[0].clone();
        let track = mux
            .add_stream(NewStream {
                params: stream.params().clone(),
                timescale: Timescale::new(48_000),
                language: None,
            })
            .expect("add");
        inputs.push((track, read_all(&mut demux)));
    }
    // Interleave by time, as a real writer would.
    let mut order: Vec<(i64, usize, usize)> = inputs
        .iter()
        .enumerate()
        .flat_map(|(which, (_, packets))| {
            packets
                .iter()
                .enumerate()
                .map(move |(index, packet)| (pts(packet), which, index))
        })
        .collect();
    order.sort_unstable();
    for (_, which, index) in order {
        let (track, packets) = &inputs[which];
        let mut packet = packets[index].clone();
        packet = Packet::try_new(
            track.get(),
            packet.bytes().to_vec(),
            packet.timestamp(),
            packet.timescale(),
            true,
        )
        .expect("packet");
        mux.write_packet(&packet).expect("write");
    }
    let durations: Vec<TrackDuration> = inputs
        .iter()
        .map(|(track, packets)| TrackDuration {
            track: *track,
            timescale: Timescale::new(48_000),
            ticks: packets.iter().map(duration).sum(),
        })
        .collect();
    mux.finish(&durations).expect("finish");

    for (which, source) in sources.iter().enumerate() {
        let selector = format!("a:{which}");
        assert_eq!(
            probe_selected(&out, &selector),
            probe_selected(source, "a:0"),
            "stream {which} packets"
        );
    }
    let mut demux = open(&out);
    assert_eq!(demux.streams().len(), 2);
    let all = read_all(&mut demux);
    for (which, (_, packets)) in inputs.iter().enumerate() {
        let mine: Vec<&Packet> = all
            .iter()
            .filter(|packet| packet.stream() == u32::try_from(which).expect("track"))
            .collect();
        assert_eq!(mine.len(), packets.len(), "stream {which} count");
        for (left, right) in mine.iter().zip(packets) {
            assert_eq!(left.bytes(), right.bytes());
            assert_eq!(left.timestamp(), right.timestamp());
        }
    }
    eprintln!("two streams: {} packets", all.len());
    let _ = std::fs::remove_dir_all(&dir);
}

fn probe_selected(path: &Path, selector: &str) -> Vec<String> {
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            selector,
            "-show_entries",
            "packet=pts,duration,size",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .expect("ffprobe");
    assert!(out.status.success());
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_owned)
        .collect()
}
