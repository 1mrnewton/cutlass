//! Real files under `../local-files`, copied into AVI by FFmpeg
//! (`-c copy`, the first ten seconds). Skips when that folder or FFmpeg is
//! missing. The copies live under the target directory; the originals are
//! only read.

use std::path::{Path, PathBuf};

use cutlass_avi::AviMux;
use cutlass_container::{Demux, Landing, Mux, NewStream, SeekTo, TrackDuration};

use crate::oracle::{check_against_ffprobe, check_written, pcm_block};
use crate::support::{drain, open_path, output, scratch, tool_on_path};

fn local_root() -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    path.is_dir().then_some(path)
}

fn codecs(path: &Path) -> Vec<String> {
    let out = output(
        "ffprobe",
        &[
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            &path.display().to_string(),
        ],
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|line| line.trim().trim_end_matches(',').to_owned())
        .collect()
}

/// Up to two files of each kind AVI can hold after a stream copy.
fn pick(root: &Path) -> Vec<PathBuf> {
    let mut names: Vec<PathBuf> = std::fs::read_dir(root)
        .expect("local-files")
        .map(|entry| entry.expect("entry").path())
        .collect();
    names.sort();
    let mut kinds: Vec<(String, u32)> = Vec::new();
    let mut out = Vec::new();
    for path in names {
        let Some(ext) = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_ascii_lowercase)
        else {
            continue;
        };
        if !matches!(ext.as_str(), "mp4" | "mov" | "m4a" | "wav" | "mkv" | "webm") {
            continue;
        }
        let found = codecs(&path);
        let fits = !found.is_empty()
            && found.iter().all(|codec| {
                matches!(
                    codec.as_str(),
                    "h264"
                        | "hevc"
                        | "mpeg4"
                        | "mjpeg"
                        | "vp8"
                        | "aac"
                        | "mp3"
                        | "ac3"
                        | "pcm_s16le"
                        | "pcm_s24le"
                )
            });
        if !fits {
            continue;
        }
        let kind = found.join("+");
        match kinds.iter_mut().find(|(name, _)| *name == kind) {
            Some((_, count)) if *count >= 2 => continue,
            Some((_, count)) => *count += 1,
            None => kinds.push((kind, 1)),
        }
        out.push(path);
    }
    out
}

#[test]
fn local_files_copied_into_avi_match_ffprobe_seek_and_remux() {
    let Some(root) = local_root() else {
        eprintln!("skipped: ../local-files is missing");
        return;
    };
    if !tool_on_path("ffmpeg") || !tool_on_path("ffprobe") {
        eprintln!("skipped: ffmpeg or ffprobe is missing");
        return;
    }
    let dir = scratch("local");
    let picked = pick(&root);
    let mut checked = 0;
    for (index, source) in picked.iter().enumerate() {
        let avi = dir.join(format!("{index}.avi"));
        let name = avi.display().to_string();
        let found = codecs(source);
        let mut args = vec![
            "-v".to_owned(),
            "error".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            source.display().to_string(),
            "-t".to_owned(),
            "10".to_owned(),
            "-map".to_owned(),
            "0:v?".to_owned(),
            "-map".to_owned(),
            "0:a?".to_owned(),
            "-c".to_owned(),
            "copy".to_owned(),
        ];
        if found.iter().any(|codec| codec == "hevc") {
            args.extend(["-tag:v".to_owned(), "HEVC".to_owned()]);
        }
        args.push(name.clone());
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let made = output("ffmpeg", &refs);
        if !made.status.success() {
            eprintln!(
                "skipped {}: ffmpeg could not copy it into AVI",
                source.display()
            );
            continue;
        }
        let packets = check_against_ffprobe(&avi);
        assert!(!packets.is_empty(), "{}", source.display());
        seek_without_payload(&avi);
        remux(&avi, &dir.join(format!("{index}-remux.avi")));
        eprintln!(
            "local {} ({}): {} packets",
            source.display(),
            found.join("+"),
            packets.len()
        );
        checked += 1;
    }
    if checked == 0 {
        eprintln!("skipped: no local file copies into AVI");
    }
}

fn seek_without_payload(path: &Path) {
    let mut demux = open_path(path);
    let len = std::fs::metadata(path).expect("meta").len();
    let header = demux.consumed();
    eprintln!("  open read {header} of {len} bytes");
    for track in 0..u32::try_from(demux.streams().len()).expect("tracks") {
        let Some(end) = demux.duration_ticks(track) else {
            continue;
        };
        for time in [-1, 0, end / 3, end / 2, end - 1, end + 1] {
            let before = demux.consumed();
            let landing = demux
                .seek(SeekTo {
                    track,
                    time,
                    end: Some(end),
                })
                .expect("seek");
            assert_eq!(landing.payload_scans, 0);
            assert_eq!(demux.consumed(), before, "a seek reads nothing");
            match landing.landing {
                Landing::Packet {
                    time: landed,
                    keyframe,
                    ..
                } => {
                    assert!(keyframe);
                    assert!(landed <= time.max(0) || time < 0);
                    // Other streams resume at the landing time and may sit
                    // earlier in the file; this stream's first packet is the landing.
                    let next = std::iter::from_fn(|| demux.read_packet().expect("read"))
                        .take(10_000)
                        .find(|packet| packet.stream() == track)
                        .expect("landing packet");
                    assert!(next.keyframe());
                    assert_eq!(next.timestamp().decode(), Some(landed));
                }
                Landing::End => assert!(time >= end),
            }
        }
    }
}

/// Copies `input` through [`AviDemux`] and [`AviMux`] into `output_path`
/// and checks the copy with ffprobe and GStreamer.
pub fn remux(input: &Path, output_path: &Path) {
    let mut demux = open_path(input);
    let mut mux = AviMux::new(std::fs::File::create(output_path).expect("create"));
    let mut durations = Vec::new();
    let mut scales = Vec::new();
    let mut pads = Vec::new();
    let (mut video, mut audio) = (0, 0);
    for stream in demux.streams() {
        let track = mux
            .add_stream(NewStream {
                params: stream.params().clone(),
                timescale: stream.timescale(),
                language: None,
            })
            .expect("add");
        durations.push(TrackDuration {
            track,
            timescale: stream.timescale(),
            ticks: 0,
        });
        scales.push(stream.timescale().ticks_per_second());
        if stream.params().size().is_some() {
            pads.push(format!("video_{video}"));
            video += 1;
        } else {
            pads.push(format!("audio_{audio}"));
            audio += 1;
        }
    }
    let params: Vec<_> = demux
        .streams()
        .iter()
        .map(|stream| stream.params().clone())
        .collect();
    let packets = drain(&mut demux);
    for packet in &packets {
        let index = usize::try_from(packet.stream()).expect("stream");
        durations[index].ticks += packet.timestamp().duration().expect("duration");
        mux.write_packet(packet).expect("write");
    }
    mux.finish(&durations).expect("finish");
    let blocks: Vec<Option<usize>> = params
        .iter()
        .zip(&scales)
        .map(|(params, _)| packets.first().and_then(|packet| pcm_block(packet, params)))
        .collect();
    let pads: Vec<&str> = pads.iter().map(String::as_str).collect();
    check_written(output_path, &packets, &scales, &blocks, &pads, false);
}
