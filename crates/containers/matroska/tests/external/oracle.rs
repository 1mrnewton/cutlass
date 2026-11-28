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
    paddings, read_all, remux, remux_with_cover, scratch, tool_on_path,
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

/// The streams ffprobe lists in `path`, one line each: index, codec type,
/// codec name, the `attached_pic` disposition, and the file name and media
/// type tags.
fn probed_streams(path: &Path) -> Vec<String> {
    let out = crate::support::output_timeout(
        "ffprobe",
        &[
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name:stream_disposition=attached_pic:stream_tags=filename,mimetype",
            "-of",
            "csv=p=0",
            &path.display().to_string(),
        ],
        std::time::Duration::from_secs(60),
    )
    .expect("ffprobe");
    assert!(out.status.success(), "ffprobe failed on {}", path.display());
    assert!(
        out.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_owned)
        .collect()
}

/// libavformat reads a Matroska cover as one more stream: Motion JPEG with
/// `attached_pic` set, tagged `cover.jpg` and `image/jpeg`, its packet the
/// picture. The movie's own streams and packets are as in the file without
/// it. A WebM file has no cover.
#[test]
fn ffprobe_reads_the_cover_as_an_attached_picture() {
    if !ffmpeg_tools() {
        return;
    }
    let dir = scratch("oracle-cover");
    let picture = dir.join("cover.jpg");
    let target = picture.display().to_string();
    assert!(ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=16x16",
        "-frames:v",
        "1",
        "-c:v",
        "mjpeg",
        "-f",
        "image2",
        &target,
    ]));
    let jpeg = std::fs::read(&picture).expect("cover");
    let mut checked = 0;
    for (path, case) in fixtures() {
        if case.codecs.len() < 2 {
            continue;
        }
        let webm = case.name.ends_with(".webm");
        let plain = dir.join(format!("plain-{}", case.name));
        let covered = dir.join(case.name);
        remux(path, &plain, webm);
        remux_with_cover(path, &covered, webm, Some(jpeg.clone()));
        let before = probed_streams(&plain);
        let after = probed_streams(&covered);
        if webm {
            assert_eq!(after, before, "{}", case.name);
            continue;
        }
        assert_eq!(after.len(), before.len() + 1, "{}: {after:?}", case.name);
        assert_eq!(&after[..before.len()], before.as_slice(), "{}", case.name);
        assert_eq!(
            after[before.len()],
            format!("{},mjpeg,video,1,cover.jpg,image/jpeg", before.len()),
            "{}",
            case.name
        );
        for index in 0..before.len() {
            assert_eq!(
                ffmpeg_payload(&covered, index, &dir),
                ffmpeg_payload(&plain, index, &dir),
                "{} stream {index}",
                case.name
            );
        }
        assert_eq!(
            ffmpeg_payload(&covered, before.len(), &dir),
            jpeg,
            "{}",
            case.name
        );
        checked += 1;
    }
    if checked == 0 {
        eprintln!("skipped: this ffmpeg made no Matroska fixture");
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

/// The body of the first `kind` box in an MP4 file.
fn box_body<'a>(bytes: &'a [u8], kind: &[u8; 4]) -> &'a [u8] {
    let at = bytes
        .windows(4)
        .position(|window| window == kind)
        .expect("box");
    let size = u32::from_be_bytes(bytes[at - 4..at].try_into().expect("size")) as usize;
    &bytes[at + 4..at - 4 + size]
}

/// FFmpeg's MD5 of the decoded sound of `path`, or what it said on failure.
fn pcm_md5(path: &Path) -> String {
    let path = path.display().to_string();
    let out = crate::support::output_timeout(
        "ffmpeg",
        &["-v", "error", "-i", &path, "-map", "0:a", "-f", "md5", "-"],
        std::time::Duration::from_secs(60),
    )
    .expect("ffmpeg");
    if !out.status.success() {
        return format!("failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    String::from_utf8_lossy(&out.stdout).trim().to_owned()
}

/// Writes `packets` of `stream` as the one track of a Matroska file, with
/// `config` as its extradata.
fn write_track(
    path: &Path,
    stream: &cutlass_container::Stream,
    config: Vec<u8>,
    packets: &[cutlass_codec::Packet],
) {
    use cutlass_container::{Mux, NewStream, TrackDuration};
    let params = cutlass_codec::StreamParams::try_audio(
        stream.params().codec().clone(),
        stream.params().sample_rate().expect("rate"),
        stream.params().channels().expect("channels"),
        config,
    )
    .expect("params");
    let file = std::io::BufWriter::new(std::fs::File::create(path).expect("create"));
    let mut mux = cutlass_matroska::MatroskaMux::new(file);
    let track = mux
        .add_stream(NewStream {
            params,
            timescale: stream.timescale(),
            language: None,
        })
        .expect("add stream");
    let mut ticks = 0;
    for (index, packet) in packets.iter().enumerate() {
        let pts = packet.timestamp().presentation().expect("pts");
        let duration = packet.timestamp().duration().unwrap_or_else(|| {
            packets
                .get(index + 1)
                .and_then(|next| next.timestamp().presentation())
                .map_or(0, |next| next - pts)
        });
        ticks += duration;
        let stamped =
            cutlass_codec::Timestamp::try_new(None, Some(pts), Some(duration)).expect("time");
        let copy = cutlass_codec::Packet::try_new(
            track.get(),
            packet.bytes().to_vec(),
            stamped,
            packet.timescale(),
            packet.keyframe(),
        )
        .expect("packet");
        mux.write_packet(&copy).expect("write");
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: stream.timescale(),
        ticks,
    }])
    .expect("finish");
}

/// FFmpeg copies a six-channel FLAC file into Matroska and MP4. The
/// Matroska track is written again with the configuration each other
/// container hands over: the `dfLa` body from MP4, the bare STREAMINFO
/// from a native file, and its own CodecPrivate (STREAMINFO and the
/// channel-mask comment FFmpeg adds for six channels). Each copy's
/// CodecPrivate is the marker and the blocks, and FFmpeg decodes each to
/// the same samples.
#[test]
fn flac_configuration_from_mp4_or_a_native_file_becomes_a_codec_private() {
    if !ffmpeg_tools() {
        return;
    }
    let dir = scratch("oracle-flac-config");
    let source = dir.join("source.flac");
    let mka = dir.join("copy.mka");
    let mp4 = dir.join("copy.mp4");
    let source_arg = source.display().to_string();
    assert!(ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2:sample_rate=48000",
        "-ac",
        "6",
        "-c:a",
        "flac",
        &source_arg,
    ]));
    for copy in [&mka, &mp4] {
        assert!(ffmpeg(&[
            "-i",
            &source_arg,
            "-c",
            "copy",
            &copy.display().to_string()
        ]));
    }
    let dfla = box_body(&std::fs::read(&mp4).expect("mp4"), b"dfLa").to_vec();
    assert_eq!(&dfla[..8], &[0, 0, 0, 0, 0x80, 0, 0, 0x22]);
    let mut demux = open(&mka);
    let stream = demux.streams()[0].clone();
    assert_eq!(stream.params().codec(), &StreamCodec::Known(CodecId::Flac));
    let private = stream.params().extradata().to_vec();
    assert_eq!(&private[..4], b"fLaC");
    assert_eq!(private[42] & 0x7F, 4, "a comment block follows STREAMINFO");
    let packets = read_all(&mut demux);
    let want = pcm_md5(&mka);
    assert!(want.starts_with("MD5="), "{want}");
    let mut from_dfla = b"fLaC".to_vec();
    from_dfla.extend_from_slice(&dfla[4..]);
    for (name, config, expected) in [
        ("dfla", dfla.clone(), from_dfla.clone()),
        ("streaminfo", dfla[8..].to_vec(), from_dfla),
        ("private", private.clone(), private.clone()),
    ] {
        let out = dir.join(format!("from-{name}.mka"));
        write_track(&out, &stream, config, &packets);
        assert_eq!(pcm_md5(&out), want, "{name}");
        let mut copy = open(&out);
        assert_eq!(
            copy.streams()[0].params().extradata(),
            expected.as_slice(),
            "{name}"
        );
        let again = read_all(&mut copy);
        assert_eq!(again.len(), packets.len(), "{name}");
        assert!(
            again
                .iter()
                .zip(&packets)
                .all(|(a, b)| a.bytes() == b.bytes()),
            "{name}"
        );
    }
}
