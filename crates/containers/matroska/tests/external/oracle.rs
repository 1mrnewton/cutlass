//! Files from FFmpeg and GStreamer, read by this crate and by the two
//! independent demuxers (libavformat through ffprobe, GStreamer's
//! matroskademux), and files this crate writes, read back by both.
//!
//! Each fixture is one second, generated here with lavfi sources, so no
//! media is checked in. A case whose encoder is missing from this ffmpeg
//! build is skipped and named on stderr.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use cutlass_codec::{CodecId, StreamCodec};
use cutlass_container::{Demux, Landing, SeekTo};

use crate::support::{
    delays, ffmpeg, ffmpeg_payload, ffmpeg_tools, ffprobe_packets, gstreamer_packets, open,
    paddings, read_all, remux, scratch, tool_on_path,
};

struct Case {
    name: &'static str,
    args: &'static [&'static str],
    codecs: &'static [CodecId],
}

const VIDEO: &[&str] = &[
    "-f",
    "lavfi",
    "-i",
    "testsrc2=size=160x120:rate=25:duration=1",
];
const SOUND: &str = "sine=frequency=440:sample_rate=48000:duration=1";

const CASES: &[Case] = &[
    Case {
        name: "h264_aac.mkv",
        args: &["-c:v", "libx264", "-g", "10", "-bf", "2", "-c:a", "aac"],
        codecs: &[CodecId::H264, CodecId::Aac],
    },
    Case {
        name: "hevc_opus.mkv",
        args: &[
            "-c:v",
            "libx265",
            "-x265-params",
            "log-level=none",
            "-g",
            "10",
            "-c:a",
            "libopus",
        ],
        codecs: &[CodecId::Hevc, CodecId::Opus],
    },
    Case {
        name: "vp9_opus.webm",
        args: &[
            "-c:v",
            "libvpx-vp9",
            "-g",
            "10",
            "-b:v",
            "200k",
            "-c:a",
            "libopus",
        ],
        codecs: &[CodecId::Vp9, CodecId::Opus],
    },
    Case {
        name: "av1_opus.webm",
        args: &[
            "-c:v",
            "libsvtav1",
            "-preset",
            "12",
            "-g",
            "10",
            "-c:a",
            "libopus",
        ],
        codecs: &[CodecId::Av1, CodecId::Opus],
    },
    Case {
        name: "vp8_vorbis.webm",
        args: &[
            "-c:v", "libvpx", "-g", "10", "-c:a", "vorbis", "-strict", "-2", "-ac", "2",
        ],
        codecs: &[CodecId::Vp8, CodecId::Vorbis],
    },
    Case {
        name: "mpeg2_mp3.mkv",
        args: &["-c:v", "mpeg2video", "-g", "6", "-c:a", "libmp3lame"],
        codecs: &[CodecId::Mpeg2, CodecId::Mp3],
    },
    Case {
        name: "mjpeg_pcm16.mkv",
        args: &["-c:v", "mjpeg", "-c:a", "pcm_s16le"],
        codecs: &[CodecId::Mjpeg, CodecId::Pcm],
    },
    Case {
        name: "prores_pcm24.mkv",
        args: &["-c:v", "prores", "-c:a", "pcm_s24le"],
        codecs: &[CodecId::Prores, CodecId::Pcm],
    },
    Case {
        name: "mpeg4_ac3.mkv",
        args: &["-c:v", "mpeg4", "-g", "10", "-c:a", "ac3"],
        codecs: &[CodecId::Mpeg4, CodecId::Ac3],
    },
    Case {
        name: "h264_eac3_clusters.mkv",
        args: &[
            "-c:v",
            "libx264",
            "-g",
            "50",
            "-c:a",
            "eac3",
            "-cluster_time_limit",
            "100",
        ],
        codecs: &[CodecId::H264, CodecId::Eac3],
    },
    Case {
        name: "h264_cues_front.mkv",
        args: &[
            "-c:v",
            "libx264",
            "-g",
            "5",
            "-c:a",
            "aac",
            "-cues_to_front",
            "1",
            "-reserve_index_space",
            "2048",
        ],
        codecs: &[CodecId::H264, CodecId::Aac],
    },
    Case {
        name: "vp9_live.webm",
        args: &[
            "-c:v",
            "libvpx-vp9",
            "-g",
            "10",
            "-c:a",
            "libopus",
            "-live",
            "1",
        ],
        codecs: &[CodecId::Vp9, CodecId::Opus],
    },
    Case {
        name: "flac.mka",
        args: &["-vn", "-c:a", "flac"],
        codecs: &[CodecId::Flac],
    },
    Case {
        name: "alac.mka",
        args: &["-vn", "-c:a", "alac"],
        codecs: &[CodecId::Alac],
    },
    Case {
        name: "pcm_f32.mka",
        args: &["-vn", "-c:a", "pcm_f32le"],
        codecs: &[CodecId::Pcm],
    },
    Case {
        name: "pcm_s32be.mka",
        args: &["-vn", "-c:a", "pcm_s32be"],
        codecs: &[CodecId::Pcm],
    },
    Case {
        name: "pcm_u8.mka",
        args: &["-vn", "-c:a", "pcm_u8"],
        codecs: &[CodecId::Pcm],
    },
];

/// The generated fixtures, once per test run.
fn fixtures() -> &'static Vec<(PathBuf, &'static Case)> {
    static MADE: OnceLock<Vec<(PathBuf, &'static Case)>> = OnceLock::new();
    MADE.get_or_init(|| {
        let dir = scratch("oracle-fixtures");
        let mut made = Vec::new();
        for case in CASES {
            let path = dir.join(case.name);
            let mut args: Vec<&str> = Vec::new();
            if case.codecs.len() > 1 {
                args.extend_from_slice(VIDEO);
            }
            args.extend_from_slice(&["-f", "lavfi", "-i", SOUND]);
            args.extend_from_slice(case.args);
            let target = path.display().to_string();
            if case.name.ends_with(".mka") {
                args.extend_from_slice(&["-f", "matroska"]);
            }
            args.push(&target);
            if ffmpeg(&args) {
                made.push((path, case));
            } else {
                eprintln!("skipped {}: this ffmpeg cannot write it", case.name);
            }
        }
        made
    })
}

/// Our packets against ffprobe's, packet by packet in file order, and our
/// payloads against `ffmpeg -c copy -f data` stream by stream.
pub fn agrees_with_libavformat(path: &Path, dir: &Path, codecs: Option<&[CodecId]>) {
    let mut demux = open(path);
    let streams = demux.streams().to_vec();
    if let Some(codecs) = codecs {
        let ours: Vec<StreamCodec> = streams.iter().map(|s| s.params().codec().clone()).collect();
        let want: Vec<StreamCodec> = codecs.iter().map(|c| StreamCodec::Known(*c)).collect();
        assert_eq!(ours, want, "{}", path.display());
    }
    let delay = delays(&demux);
    let padding = paddings(&demux);
    let ours = read_all(&mut demux);
    let theirs = ffprobe_packets(path);
    // Each track's trailing padding is its last packet's DiscardPadding.
    for (index, stream) in streams.iter().enumerate() {
        let last = theirs
            .iter()
            .rev()
            .find(|probed| probed.stream == index)
            .map_or(0, |probed| probed.discard);
        let want = (stream.params().sample_rate().is_some() && last > 0).then_some(last);
        // Without Cues the last Cluster is not looked for at open.
        let read = has_cues(path);
        assert_eq!(
            padding[index],
            want.filter(|_| read),
            "{} stream {index} padding",
            path.display()
        );
    }
    assert_eq!(ours.len(), theirs.len(), "{} packet count", path.display());
    for (index, (mine, probed)) in ours.iter().zip(&theirs).enumerate() {
        let stream = streams
            .iter()
            .position(|s| s.id().get() == mine.stream())
            .expect("stream");
        let what = format!("{} packet {index}", path.display());
        assert_eq!(stream, probed.stream, "{what} stream");
        assert_eq!(mine.bytes().len(), probed.size, "{what} size");
        assert_eq!(mine.keyframe(), probed.key, "{what} keyframe");
        let pts = mine
            .timestamp()
            .presentation()
            .map(|pts| pts - delay[stream]);
        assert_eq!(pts, probed.pts, "{what} pts");
        // FFmpeg's parsers fill durations the file does not state, and
        // give a trimmed last audio frame its whole length again.
        let last = !ours[index + 1..]
            .iter()
            .any(|p| p.stream() == mine.stream());
        if let Some(duration) = mine.timestamp().duration()
            && duration > 0
            && probed.discard == 0
            && !last
        {
            assert_eq!(Some(duration), probed.duration, "{what} duration");
        }
    }
    for (index, stream) in streams.iter().enumerate() {
        let joined: Vec<u8> = ours
            .iter()
            .filter(|packet| packet.stream() == stream.id().get())
            .flat_map(|packet| packet.bytes().to_vec())
            .collect();
        assert!(
            joined == ffmpeg_payload(path, index, dir),
            "{} stream {index} payload differs from libavformat",
            path.display()
        );
    }
}

/// The file has a Cues element (its ID, at the top level or not).
fn has_cues(path: &Path) -> bool {
    let bytes = std::fs::read(path).expect("read");
    bytes
        .windows(4)
        .any(|window| window == [0x1C, 0x53, 0xBB, 0x6B])
}

#[test]
fn ffmpeg_files_read_as_libavformat_reads_them() {
    if !ffmpeg_tools() {
        return;
    }
    let dir = scratch("oracle-ffmpeg");
    for (path, case) in fixtures() {
        agrees_with_libavformat(path, &dir, Some(case.codecs));
    }
}

#[test]
fn gstreamer_files_read_as_both_demuxers_read_them() {
    if !ffmpeg_tools() || !tool_on_path("gst-launch-1.0") {
        eprintln!("skipped: GStreamer is not on PATH");
        return;
    }
    let dir = scratch("oracle-gstreamer");
    let video = "videotestsrc num-buffers=25 ! video/x-raw,width=160,height=120,framerate=25/1";
    let sound = "audiotestsrc num-buffers=50 ! audio/x-raw,rate=48000,channels=2 ! audioconvert";
    let pipelines = [
        (
            "gst_vp8_vorbis.webm",
            format!("{video} ! vp8enc keyframe-max-dist=10 ! queue ! mux. {sound} ! vorbisenc ! queue ! mux. webmmux name=mux"),
        ),
        (
            "gst_live_opus.webm",
            format!("{video} ! vp8enc ! queue ! mux. {sound} ! opusenc ! queue ! mux. webmmux streamable=true name=mux"),
        ),
        ("gst_flac.mka", format!("{sound} ! flacenc ! matroskamux")),
        ("gst_mp3.mka", format!("{sound} ! lamemp3enc ! matroskamux")),
        (
            "gst_pcm_be.mka",
            "audiotestsrc num-buffers=50 ! audio/x-raw,format=S16BE,rate=44100,channels=2 ! matroskamux".to_owned(),
        ),
    ];
    for (name, pipeline) in pipelines {
        let path = dir.join(name);
        let mut args: Vec<String> = vec!["-q".to_owned()];
        args.extend(pipeline.split(' ').map(str::to_owned));
        args.extend([
            "!".to_owned(),
            "filesink".to_owned(),
            format!("location={}", path.display()),
        ]);
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let made = crate::support::output_timeout(
            "gst-launch-1.0",
            &refs,
            std::time::Duration::from_secs(60),
        )
        .is_some_and(|out| out.status.success());
        if !made {
            eprintln!("skipped {name}: this GStreamer cannot write it");
            continue;
        }
        agrees_with_libavformat(&path, &dir, None);
        agrees_with_gstreamer(&path);
    }
}

/// Our packet sizes, per track, against GStreamer's matroskademux pads.
fn agrees_with_gstreamer(path: &Path) {
    let mut demux = open(path);
    let streams = demux.streams().to_vec();
    let ours = read_all(&mut demux);
    let mut videos = 0;
    let mut audios = 0;
    for stream in &streams {
        let pad = if stream.params().size().is_some() {
            videos += 1;
            format!("video_{}", videos - 1)
        } else {
            audios += 1;
            format!("audio_{}", audios - 1)
        };
        let Some(theirs) = gstreamer_packets(path, &pad) else {
            eprintln!("skipped GStreamer on {}: no answer", path.display());
            return;
        };
        let mine: Vec<usize> = ours
            .iter()
            .filter(|packet| packet.stream() == stream.id().get())
            .map(|packet| packet.bytes().len())
            .collect();
        let sizes: Vec<usize> = theirs.iter().map(|(size, _)| *size).collect();
        // GStreamer hands Vorbis, FLAC and Opus headers out as buffers too;
        // they come first, before any frame.
        let skip = sizes.len().saturating_sub(mine.len());
        assert_eq!(&sizes[skip..], mine.as_slice(), "{} {pad}", path.display());
    }
}

#[test]
fn files_this_crate_writes_read_in_ffprobe_and_gstreamer() {
    if !ffmpeg_tools() {
        return;
    }
    let dir = scratch("oracle-written");
    for (path, case) in fixtures() {
        let webm = case.name.ends_with(".webm");
        let out = dir.join(case.name);
        remux(path, &out, webm);
        // The copy holds the same packets as the source, as we read them.
        let mut source = open(path);
        let source_delay = delays(&source);
        let before = read_all(&mut source);
        let mut copy = open(&out);
        let after_delay = delays(&copy);
        let after = read_all(&mut copy);
        assert_eq!(before.len(), after.len(), "{}", case.name);
        // Track by track: the copy interleaves by time, which may not be
        // the source's order.
        for track in 1..=u32::try_from(case.codecs.len()).expect("tracks") {
            let a: Vec<_> = before.iter().filter(|p| p.stream() == track).collect();
            let b: Vec<_> = after.iter().filter(|p| p.stream() == track).collect();
            assert_eq!(a.len(), b.len(), "{} track {track}", case.name);
            for (a, b) in a.iter().zip(&b) {
                assert!(a.bytes() == b.bytes(), "{} track {track} bytes", case.name);
                assert_eq!(a.keyframe(), b.keyframe(), "{}", case.name);
                assert_eq!(
                    a.timestamp().presentation(),
                    b.timestamp().presentation(),
                    "{}",
                    case.name
                );
            }
        }
        // libavformat reads the copy the same way we do.
        agrees_with_libavformat(&out, &dir, Some(case.codecs));
        // CodecDelay survives: Opus's is in the OpusHead, and the others
        // come back as the stream's encoder delay, which the copy states.
        assert_eq!(source_delay, after_delay, "{}", case.name);
        // So does the end padding, where the source's could be read at open.
        if has_cues(path) {
            assert_eq!(paddings(&source), paddings(&copy), "{}", case.name);
        }
        agrees_with_gstreamer(&out);
        // FFmpeg seeks the copy through its Cues.
        if case.codecs.len() > 1 {
            seeks_in_ffprobe(&out);
        }
    }
}

/// `ffprobe -read_intervals 0.6%+#1` seeks before reading: the first video
/// packet must be a keyframe at or before 0.6 s.
fn seeks_in_ffprobe(path: &Path) {
    let out = crate::support::output_timeout(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            "0.6%+#1",
            "-show_entries",
            "packet=pts,flags",
            "-of",
            "csv=p=0",
            &path.display().to_string(),
        ],
        std::time::Duration::from_secs(30),
    )
    .expect("ffprobe");
    let text = String::from_utf8_lossy(&out.stdout);
    let first = text.lines().next().unwrap_or_default().to_owned();
    let mut fields = first.split(',');
    let pts: i64 = fields.next().and_then(|v| v.parse().ok()).unwrap_or(-1);
    let flags = fields.next().unwrap_or_default();
    assert!(
        (0..=600).contains(&pts) && flags.starts_with('K'),
        "{}: ffprobe seek landed on {first:?}",
        path.display()
    );
}

#[test]
fn seeks_land_on_the_keyframe_libavformat_lists() {
    if !ffmpeg_tools() {
        return;
    }
    for (path, _) in fixtures() {
        let mut demux = open(path);
        let delay = delays(&demux);
        let theirs = ffprobe_packets(path);
        let tracks: Vec<u32> = demux.streams().iter().map(|s| s.id().get()).collect();
        for (stream, (track, delay)) in tracks.into_iter().zip(delay).enumerate() {
            let keys: Vec<i64> = theirs
                .iter()
                .filter(|p| p.stream == stream && p.key)
                .filter_map(|p| p.pts)
                .map(|pts| pts + delay)
                .collect();
            let Some(first) = keys.first().copied() else {
                continue;
            };
            for time in [0, 1, 99, 333, 500, 640, 901, 990] {
                let landing = demux
                    .seek(SeekTo {
                        track,
                        time,
                        end: None,
                    })
                    .expect("seek");
                let want = keys
                    .iter()
                    .copied()
                    .filter(|key| *key <= time)
                    .max()
                    .unwrap_or(first);
                match landing.landing {
                    Landing::Packet { time: got, .. } => {
                        assert_eq!(got, want, "{} track {track} at {time}", path.display());
                        let next = demux.read_packet().expect("read").expect("packet");
                        assert_eq!(next.stream(), track);
                        assert_eq!(next.timestamp().presentation(), Some(want));
                    }
                    Landing::End => panic!("{} track {track} at {time}: end", path.display()),
                }
                assert_eq!(landing.payload_scans, 0);
            }
        }
    }
}
