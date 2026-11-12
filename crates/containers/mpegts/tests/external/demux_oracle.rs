//! This demuxer against files FFmpeg writes, packet by packet against
//! ffprobe, and seeks against the keyframes ffprobe lists.
//!
//! Producers: FFmpeg 8.1's MPEG-TS muxer over its test sources (libx264,
//! libx265, native MPEG-2, AAC, AC-3, E-AC-3, MP2, libmp3lame, libopus),
//! and GStreamer's `mpegtsmux` when it is present. Oracle: libavformat's
//! MPEG-TS demuxer and parsers through ffprobe.
//!
//! What may differ, and why: ffprobe's AAC packets keep their 7-byte ADTS
//! header and this crate hands on the raw frame; a LATM packet is the
//! AudioMuxElement in ffprobe and the raw frame here, so its size is not
//! compared; audio frames between PES timestamps may differ by one tick,
//! as FFmpeg rounds each frame's length and this crate rounds the sum.

use std::path::{Path, PathBuf};

use cutlass_container::{Demux, Landing, SeekTo};

use crate::support::{Packet, audio, ffmpeg, ffmpeg_present, ffprobe, ours, scratch};

type Demuxer = cutlass_mpegts::TsDemux<std::io::BufReader<std::fs::File>>;

pub struct Variant {
    pub name: &'static str,
    pub args: &'static [&'static str],
}

const VIDEO: &str = "testsrc2=size=320x240:rate=25:duration=6";
const AUDIO: &str = "sine=frequency=700:sample_rate=48000:duration=6";
const AUDIO_44: &str = "sine=frequency=500:sample_rate=44100:duration=6";

pub const VARIANTS: &[Variant] = &[
    Variant {
        name: "h264_bframes_aac.ts",
        args: &[
            "-c:v", "libx264", "-preset", "veryfast", "-bf", "3", "-g", "50", "-c:a", "aac",
        ],
    },
    Variant {
        name: "hevc_ac3.ts",
        args: &[
            "-c:v",
            "libx265",
            "-x265-params",
            "log-level=none:bframes=3:keyint=40",
            "-c:a",
            "ac3",
        ],
    },
    Variant {
        name: "mpeg2_mp2.ts",
        args: &["-c:v", "mpeg2video", "-g", "15", "-bf", "2", "-c:a", "mp2"],
    },
    Variant {
        name: "mpeg2_interlaced_mp3.ts",
        args: &[
            "-c:v",
            "mpeg2video",
            "-flags",
            "+ildct+ilme",
            "-top",
            "1",
            "-g",
            "12",
            "-c:a",
            "libmp3lame",
        ],
    },
    Variant {
        name: "mpeg4_mp3.ts",
        args: &[
            "-c:v",
            "mpeg4",
            "-bf",
            "2",
            "-g",
            "12",
            "-c:a",
            "libmp3lame",
        ],
    },
    Variant {
        name: "h264_latm.ts",
        args: &[
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-g",
            "50",
            "-c:a",
            "aac",
            "-mpegts_flags",
            "latm",
        ],
    },
    Variant {
        name: "h264_opus.ts",
        args: &[
            "-c:v", "libx264", "-preset", "veryfast", "-g", "50", "-c:a", "libopus",
        ],
    },
    Variant {
        name: "h264_eac3.m2ts",
        args: &[
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-g",
            "50",
            "-c:a",
            "eac3",
            "-mpegts_m2ts_mode",
            "1",
        ],
    },
    // FFmpeg's M2TS mode writes AAC as private data with no descriptor
    // naming it, so the demuxer names it from its first frames.
    Variant {
        name: "h264_aac.m2ts",
        args: &[
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-g",
            "50",
            "-c:a",
            "aac",
            "-mpegts_m2ts_mode",
            "1",
        ],
    },
    Variant {
        name: "h264_ac3.m2ts",
        args: &[
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-bf",
            "2",
            "-g",
            "50",
            "-c:a",
            "ac3",
            "-mpegts_m2ts_mode",
            "1",
        ],
    },
];

/// Writes `variant` into `dir`, six seconds of test pattern and tone.
pub fn write(dir: &Path, variant: &Variant) -> PathBuf {
    let path = dir.join(variant.name);
    let mut args = vec!["-f", "lavfi", "-i", VIDEO, "-f", "lavfi", "-i", AUDIO];
    args.extend_from_slice(variant.args);
    let out = path.to_str().expect("path");
    args.push(out);
    ffmpeg(&args);
    path
}

/// Compares every stream of `path`; returns a description of each
/// difference, empty when the file matches.
pub fn compare(path: &Path) -> Vec<String> {
    let theirs = ffprobe(path, &[]);
    let (_, mine) = ours(path);
    let mut problems = Vec::new();
    for (pid, packets) in &mine {
        let Some(stream) = theirs.iter().find(|stream| stream.id == *pid) else {
            problems.push(format!("PID {pid:#x} is not in ffprobe's streams"));
            continue;
        };
        let reference = &stream.packets;
        if packets.len() != reference.len() {
            problems.push(format!(
                "PID {pid:#x} {}: {} packets, ffprobe {}",
                stream.codec,
                packets.len(),
                reference.len()
            ));
        }
        let tolerance = if audio(&stream.codec) { 1 } else { 0 };
        let close = |a: Option<i64>, b: Option<i64>| match (a, b) {
            (Some(a), Some(b)) => (a - b).abs() <= tolerance,
            (None, _) => true,
            (Some(_), None) => false,
        };
        for (index, (got, want)) in packets.iter().zip(reference).enumerate() {
            let want_size = if stream.codec == "aac" {
                want.size.saturating_sub(7)
            } else {
                want.size
            };
            let mut fields = Vec::new();
            if !close(got.pts, want.pts) {
                fields.push("pts");
            }
            if !close(got.dts, want.dts) {
                fields.push("dts");
            }
            if got.duration.is_some()
                && want.duration.is_some()
                && !close(got.duration, want.duration)
            {
                fields.push("duration");
            }
            if got.key != want.key {
                fields.push("keyframe");
            }
            if stream.codec != "aac_latm" && got.size != want_size {
                fields.push("size");
            }
            if !fields.is_empty() {
                problems.push(format!(
                    "PID {pid:#x} {} packet {index}: {fields:?} ours {got:?} ffprobe {want:?}",
                    stream.codec
                ));
            }
            if problems.len() > 12 {
                return problems;
            }
        }
    }
    problems
}

/// Seeks every stream of `path` to nine targets, before the start to past
/// the end. Each landing must be the stream's last keyframe at or before
/// the target (or its first keyframe), and the stream's next packet must be
/// that keyframe.
pub fn check_seeks(path: &Path) -> Vec<String> {
    let (mut demux, tracks) = ours(path);
    let mut problems = Vec::new();
    for (track, packets) in &tracks {
        problems.extend(
            check_track_seeks(&mut demux, *track, packets)
                .into_iter()
                .map(|problem| format!("stream {track:#x}: {problem}")),
        );
    }
    problems
}

fn check_track_seeks(demux: &mut Demuxer, track: u32, packets: &[Packet]) -> Vec<String> {
    let track = &track;
    let keys: Vec<i64> = {
        let mut keys: Vec<i64> = packets
            .iter()
            .filter(|packet| packet.key)
            .filter_map(|packet| packet.pts)
            .collect();
        keys.sort_unstable();
        keys
    };
    let start = demux.start();
    let end = demux.end().expect("random access knows the end");
    let mut problems = Vec::new();
    for fraction in [-0.1, 0.0, 0.013, 0.25, 0.5, 0.731, 0.9, 0.999, 1.2] {
        #[allow(clippy::cast_possible_truncation)]
        let target = start + ((end - start) as f64 * fraction) as i64;
        let landing = demux.seek(SeekTo {
            track: *track,
            time: target,
            end: Some(end),
        });
        let expected = keys
            .iter()
            .rev()
            .find(|&&key| key <= target)
            .or(keys.first())
            .copied();
        match landing.map(|landing| landing.landing) {
            Ok(Landing::Packet { time, .. }) => {
                let next = std::iter::from_fn(|| demux.read_packet().expect("read"))
                    .find(|packet| packet.stream() == *track);
                let next_pts = next.as_ref().and_then(|p| p.timestamp().presentation());
                let next_key = next.as_ref().is_some_and(cutlass_codec::Packet::keyframe);
                if Some(time) != expected || next_pts != Some(time) || !next_key {
                    problems.push(format!(
                        "target {target}: landed {time}, expected {expected:?}, next {next_pts:?} key {next_key}"
                    ));
                }
            }
            Ok(Landing::End) => {
                if target < end {
                    problems.push(format!(
                        "target {target} before the end {end} landed at End"
                    ));
                }
            }
            Err(error) => problems.push(format!("target {target}: {error}")),
        }
    }
    problems
}

/// The keyframe count is a sanity check that the variants are not all one
/// GOP, so the seeks above choose between keyframes.
fn keyframes(packets: &[Packet]) -> usize {
    packets.iter().filter(|packet| packet.key).count()
}

#[test]
fn ffmpeg_files_read_as_ffprobe_reads_them() {
    if !ffmpeg_present("ffmpeg_files_read_as_ffprobe_reads_them") {
        return;
    }
    let dir = scratch("demux");
    let mut failures = Vec::new();
    for variant in VARIANTS {
        let path = write(&dir, variant);
        let problems = compare(&path);
        if !problems.is_empty() {
            failures.push(format!("{}:\n  {}", variant.name, problems.join("\n  ")));
        }
    }
    std::fs::remove_dir_all(&dir).ok();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn ffmpeg_files_seek_to_the_keyframe_before() {
    if !ffmpeg_present("ffmpeg_files_seek_to_the_keyframe_before") {
        return;
    }
    let dir = scratch("seek");
    let mut failures = Vec::new();
    for variant in VARIANTS {
        let path = write(&dir, variant);
        let (_, tracks) = ours(&path);
        assert!(keyframes(&tracks[0].1) >= 2, "{}: one GOP", variant.name);
        let problems = check_seeks(&path);
        if !problems.is_empty() {
            failures.push(format!("{}:\n  {}", variant.name, problems.join("\n  ")));
        }
    }
    std::fs::remove_dir_all(&dir).ok();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Two programs, the second with 44.1 kHz audio: the first program opens,
/// the second opens by number, and each matches ffprobe.
#[test]
fn ffmpeg_two_programs_open_by_number() {
    if !ffmpeg_present("ffmpeg_two_programs_open_by_number") {
        return;
    }
    let dir = scratch("programs");
    let path = dir.join("programs.ts");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        VIDEO,
        "-f",
        "lavfi",
        "-i",
        AUDIO,
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=30:duration=6",
        "-f",
        "lavfi",
        "-i",
        AUDIO_44,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-map",
        "2:v",
        "-map",
        "3:a",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-program",
        "title=One:st=0:st=1",
        "-program",
        "title=Two:st=2:st=3",
        path.to_str().unwrap(),
    ]);
    let theirs = ffprobe(&path, &[]);
    let (demux, _) = ours(&path);
    assert_eq!(demux.programs().len(), 2);
    let second = demux.programs()[1].number;
    let file = std::fs::File::open(&path).unwrap();
    let length = file.metadata().unwrap().len();
    let mut other =
        cutlass_mpegts::TsDemux::open_program(std::io::BufReader::new(file), length, second)
            .unwrap();
    let pids: Vec<u32> = other.streams().iter().map(|s| s.id().get()).collect();
    assert_eq!(pids.len(), 2);
    let mut counts = vec![0_usize; pids.len()];
    while let Some(packet) = other.read_packet().unwrap() {
        counts[pids.iter().position(|&pid| pid == packet.stream()).unwrap()] += 1;
    }
    for (pid, count) in pids.iter().zip(counts) {
        let stream = theirs.iter().find(|stream| stream.id == *pid).unwrap();
        assert_eq!(count, stream.packets.len(), "PID {pid:#x}");
    }
    let problems = compare(&path);
    std::fs::remove_dir_all(&dir).ok();
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// GStreamer's `mpegtsmux` lays packets out differently (PIDs 0x41 on,
/// PCR on its own cadence, PTS starting at an hour).
#[test]
fn gstreamer_files_read_as_ffprobe_reads_them() {
    if !ffmpeg_present("gstreamer_files_read_as_ffprobe_reads_them") {
        return;
    }
    if !crate::support::tool_on_path("gst-launch-1.0") {
        eprintln!("skipped gstreamer_files_read_as_ffprobe_reads_them: no gst-launch-1.0");
        return;
    }
    let dir = scratch("gst");
    let mut failures = Vec::new();
    for (name, audio) in [
        ("gst_h264_aac.ts", "fdkaacenc ! aacparse"),
        ("gst_h264_mp3.ts", "lamemp3enc ! mpegaudioparse"),
    ] {
        let path = dir.join(name);
        let pipeline = format!(
            "videotestsrc num-buffers=150 ! video/x-raw,width=320,height=240,framerate=25/1 \
             ! x264enc bframes=2 key-int-max=25 ! h264parse ! mux. \
             audiotestsrc num-buffers=280 ! audio/x-raw,rate=48000,channels=2 ! {audio} ! mux. \
             mpegtsmux name=mux ! filesink location={}",
            path.display()
        );
        let status = std::process::Command::new("gst-launch-1.0")
            .arg("-q")
            .args(pipeline.split_whitespace())
            .output()
            .expect("gst-launch-1.0");
        if !status.status.success() {
            eprintln!("skipped {name}: the GStreamer pipeline did not run");
            continue;
        }
        let mut problems = compare(&path);
        problems.extend(check_seeks(&path));
        if !problems.is_empty() {
            failures.push(format!("{name}:\n  {}", problems.join("\n  ")));
        }
    }
    std::fs::remove_dir_all(&dir).ok();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
