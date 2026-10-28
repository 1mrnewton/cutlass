//! Files produced by LAME, read back by ffprobe and mpg123.
//!
//! ffprobe times are ticks of 1/14112000. They are converted with
//! `ticks * sample_rate / 14112000`, then shifted so the first audio frame
//! is presentation time zero. A leading Xing or Info frame is not an audio
//! packet in this crate. ffprobe may still emit that frame, so one leading
//! packet is dropped when that makes the size sequences equal.
//!
//! GStreamer is installed on this machine but its plugin scanner does not
//! finish, so it is not an oracle here.

use std::io::Cursor;
use std::process::Command;

use crate::support::{ProbePacket, ffprobe, marker, mpeg1_frame, output, tool_on_path};
use cutlass_codec::{CodecId, CodecParams, Packet};
use cutlass_container::{Demux, Mux};
use cutlass_mp3_container::{Mp3Demux, Mp3Mux};

#[test]
fn lame_and_ffmpeg_packets_match_ffprobe_and_mpg123() {
    if !tool_on_path("ffmpeg") || !tool_on_path("ffprobe") {
        eprintln!("skipped: ffmpeg and ffprobe are not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join(format!("cutlass-mp3-oracle-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp");

    let cbr = dir.join("cbr.mp3");
    let made = output(
        "ffmpeg",
        &[
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.12:sample_rate=48000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            cbr.to_str().expect("path"),
        ],
    );
    assert!(
        made.status.success(),
        "{}",
        String::from_utf8_lossy(&made.stderr)
    );
    compare_file(&cbr);

    if tool_on_path("lame") {
        let wav = dir.join("tone.wav");
        let wav_made = output(
            "ffmpeg",
            &[
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=0.2:sample_rate=44100",
                "-ac",
                "2",
                wav.to_str().expect("path"),
            ],
        );
        assert!(wav_made.status.success());
        let vbr = dir.join("lame-vbr.mp3");
        let encoded = output(
            "lame",
            &[
                "-V",
                "4",
                wav.to_str().expect("wav"),
                vbr.to_str().expect("mp3"),
            ],
        );
        assert!(
            encoded.status.success(),
            "{}",
            String::from_utf8_lossy(&encoded.stderr)
        );
        compare_file(&vbr);
    } else {
        eprintln!("skipped: lame is not on PATH");
    }

    compare_our_mux_with_ffprobe();
    let _ = std::fs::remove_dir_all(&dir);
}

fn compare_file(path: &std::path::Path) {
    let bytes = std::fs::read(path).expect("read");
    let mut demux = Mp3Demux::open(Cursor::new(bytes)).expect("open");
    let rate = demux.streams()[0].params.sample_rate;
    let channels = demux.streams()[0].params.channels;
    let mut ours = Vec::new();
    while let Some(packet) = demux.next_packet().expect("packet") {
        assert_eq!(packet.timescale, rate);
        assert!(packet.keyframe);
        assert_ne!(packet.timescale, 1_000_000);
        ours.push((
            u64::try_from(packet.data.len()).expect("len"),
            packet.pts,
            packet.duration,
        ));
    }
    assert!(!ours.is_empty());

    let probed = ffprobe(path);
    assert_eq!(probed.sample_rate, rate);
    assert_eq!(probed.channels, channels);
    let (theirs, dropped_tag) = align(&ours, &probed.packets);
    let base = theirs[0].pts_samples;
    for (index, (ours, theirs)) in ours.iter().zip(theirs).enumerate() {
        assert_eq!(
            ours.0, theirs.size,
            "size at {index}, tag dropped {dropped_tag}"
        );
        assert_eq!(
            ours.1,
            theirs.pts_samples - base,
            "pts at {index}, tag dropped {dropped_tag}"
        );
    }
    let mut clock = 0i64;
    for packet in &ours {
        assert_eq!(packet.1, clock);
        clock += packet.2;
    }

    if tool_on_path("mpg123") {
        let played = Command::new("mpg123")
            .args(["-t", "-v", &path.display().to_string()])
            .output()
            .expect("mpg123");
        let text = String::from_utf8_lossy(&played.stderr);
        assert!(
            text.contains(&rate.to_string()),
            "mpg123 did not report {rate}: {text}"
        );
        let count = mpg123_frame_count(&text);
        assert!(
            count == ours.len() || count == ours.len() + usize::from(dropped_tag),
            "mpg123 frames {count}, ours {}, tag {dropped_tag}, text {text}",
            ours.len()
        );
    }
}

fn align<'a>(ours: &[(u64, i64, i64)], theirs: &'a [ProbePacket]) -> (&'a [ProbePacket], bool) {
    let our_sizes: Vec<u64> = ours.iter().map(|packet| packet.0).collect();
    let their_sizes: Vec<u64> = theirs.iter().map(|packet| packet.size).collect();
    if their_sizes == our_sizes {
        return (theirs, false);
    }
    if their_sizes.len() == our_sizes.len() + 1 && their_sizes[1..] == our_sizes[..] {
        return (&theirs[1..], true);
    }
    panic!("packet sizes diverged\nours {our_sizes:?}\ntheirs {their_sizes:?}");
}

fn mpg123_frame_count(text: &str) -> usize {
    let mut count = 0usize;
    for line in text.split(['\n', '\r']) {
        let line = line.trim();
        let Some(rest) = line.strip_prefix('>') else {
            continue;
        };
        let Some(token) = rest.split_whitespace().next() else {
            continue;
        };
        let Some((played, left)) = token.split_once('+') else {
            continue;
        };
        if left == "0" {
            count = played.parse().unwrap_or(count);
        }
    }
    count
}

fn compare_our_mux_with_ffprobe() {
    let mut mux = Mp3Mux::new(Cursor::new(Vec::new()));
    mux.add_stream(CodecParams::audio(CodecId::Mp3, 48_000, 2))
        .expect("stream");
    for index in 0..6u16 {
        let frame = mpeg1_frame(9, 1, false, index);
        let pts = i64::from(index) * 1152;
        mux.write_packet(&Packet::audio(frame, pts, 1152, 48_000))
            .expect("write");
    }
    mux.finish().expect("finish");
    let bytes = mux.into_inner().expect("inner").into_inner();
    let path = std::env::temp_dir().join(format!("cutlass-mp3-mux-{}.mp3", std::process::id()));
    std::fs::write(&path, &bytes).expect("write");
    let mut demux = Mp3Demux::open(Cursor::new(bytes)).expect("open");
    let mut markers = Vec::new();
    while let Some(packet) = demux.next_packet().expect("packet") {
        markers.push(marker(&packet.data));
    }
    assert_eq!(markers, vec![0, 1, 2, 3, 4, 5]);
    compare_file(&path);
    let _ = std::fs::remove_file(&path);
}
