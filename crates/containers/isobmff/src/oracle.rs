//! Independent checks. FFmpeg and GStreamer are oracles, not dependencies.
//! A missing tool or the local media folder skips that check and says so.

use crate::{Demux, FileSource, IsoBmff, MediaKind, Mux, SliceSource, Stream};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

struct OraclePacket {
    dts: i64,
    pts: i64,
    duration: u32,
    size: usize,
    keyframe: bool,
    hash: String,
}

#[test]
fn ffmpeg_and_gstreamer_see_the_same_packets() {
    let Some(ffmpeg) = tool("ffmpeg") else {
        eprintln!("not available: ffmpeg");
        return;
    };
    let Some(ffprobe) = tool("ffprobe") else {
        eprintln!("not available: ffprobe");
        return;
    };
    let dir = std::env::temp_dir().join(format!("isobmff-oracle-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let progressive = dir.join("progressive.mp4");
    let fragmented = dir.join("fragmented.mp4");
    let quicktime = dir.join("movie.mov");
    run(
        &ffmpeg,
        &[
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=0.4:size=32x32:rate=10",
            "-c:v",
            "libx264",
            "-g",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-an",
            progressive.to_str().unwrap(),
        ],
    );
    run(
        &ffmpeg,
        &[
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=0.4:size=32x32:rate=10",
            "-c:v",
            "libx264",
            "-g",
            "5",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof",
            fragmented.to_str().unwrap(),
        ],
    );
    run(
        &ffmpeg,
        &[
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=0.2:size=16x16:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-f",
            "mov",
            quicktime.to_str().unwrap(),
        ],
    );
    compare_file(&ffmpeg, &ffprobe, &progressive);
    compare_file(&ffmpeg, &ffprobe, &fragmented);
    compare_file(&ffmpeg, &ffprobe, &quicktime);
    if let Some(gst) = tool("gst-discoverer-1.0") {
        let output = Command::new(gst)
            .arg(progressive.as_os_str())
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "gstreamer {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(text.contains("H.264") || text.contains("avc"), "{text}");
        assert!(text.contains("Width: 32"), "{text}");
    } else {
        eprintln!("not available: gst-discoverer-1.0");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ffmpeg_reads_files_this_muxer_wrote() {
    let Some(ffmpeg) = tool("ffmpeg") else {
        eprintln!("not available: ffmpeg");
        return;
    };
    let dir = std::env::temp_dir().join(format!("isobmff-mux-oracle-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("source.mp4");
    run(
        &ffmpeg,
        &[
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=0.5:size=32x32:rate=10",
            "-c:v",
            "libx264",
            "-bf",
            "2",
            "-g",
            "5",
            "-pix_fmt",
            "yuv420p",
            "-an",
            source.to_str().unwrap(),
        ],
    );
    let (mut params, mut packets) = packets_of(&source);
    rebase(&mut packets);
    let layouts = [
        (
            "moov-end.mp4",
            "v:0",
            crate::MuxOptions {
                moov_at_end: true,
                ..crate::MuxOptions::default()
            },
        ),
        (
            "moov-start.mp4",
            "v:0",
            crate::MuxOptions {
                moov_at_end: false,
                ..crate::MuxOptions::default()
            },
        ),
        (
            "fragmented.mp4",
            "v:0",
            crate::MuxOptions {
                fragmented: true,
                samples_per_fragment: 2,
                ..crate::MuxOptions::default()
            },
        ),
        (
            "movie.mov",
            "v:0",
            crate::MuxOptions {
                brand: crate::FileBrand::Mov,
                moov_at_end: true,
                ..crate::MuxOptions::default()
            },
        ),
    ];
    for (name, stream, options) in layouts {
        let path = dir.join(name);
        write_remux(&path, &params, &packets, options);
        assert_ffmpeg_sees_packets(&ffmpeg, &path, &packets, stream);
    }
    let audio_source = dir.join("source.m4a");
    run(
        &ffmpeg,
        &[
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-c:a",
            "aac",
            "-t",
            "0.2",
            audio_source.to_str().unwrap(),
        ],
    );
    (params, packets) = packets_of(&audio_source);
    rebase(&mut packets);
    let audio_path = dir.join("remux.m4a");
    write_remux(
        &audio_path,
        &params,
        &packets,
        crate::MuxOptions {
            brand: crate::FileBrand::M4a,
            ..crate::MuxOptions::default()
        },
    );
    assert_ffmpeg_sees_packets(&ffmpeg, &audio_path, &packets, "a:0");
    let _ = std::fs::remove_dir_all(&dir);
}

fn packets_of(path: &Path) -> (crate::StreamParams, Vec<crate::Packet>) {
    let bytes = std::fs::read(path).unwrap();
    let mut source = SliceSource::new(&bytes);
    let mut movie = IsoBmff::open(&mut source).unwrap();
    let params = movie.streams()[0].params.clone();
    let mut packets = Vec::new();
    while let Some(packet) = movie.read_packet(&mut source).unwrap() {
        if packet.track_id == params.track_id {
            packets.push(packet);
        }
    }
    assert!(!packets.is_empty(), "{}", path.display());
    (params, packets)
}

fn rebase(packets: &mut [crate::Packet]) {
    let Some(origin) = packets.iter().map(|packet| packet.dts).min() else {
        return;
    };
    for packet in packets {
        packet.dts -= origin;
        packet.pts -= origin;
    }
}

fn write_remux(
    path: &Path,
    params: &crate::StreamParams,
    packets: &[crate::Packet],
    options: crate::MuxOptions,
) {
    let mut mux = crate::IsoBmffMux::new(std::io::Cursor::new(Vec::new()), options);
    let mut stored = params.clone();
    stored.track_id = 0;
    let id = mux.add_stream(&stored).unwrap();
    for packet in packets {
        let mut packet = packet.clone();
        packet.track_id = id;
        packet.timescale = stored.timescale;
        mux.write_packet(&packet)
            .unwrap_or_else(|err| panic!("write {}: {err}", path.display()));
    }
    let bytes = mux
        .finish_into()
        .unwrap_or_else(|err| panic!("finish {}: {err}", path.display()))
        .into_inner();
    std::fs::write(path, bytes).unwrap();
}

fn assert_ffmpeg_sees_packets(ffmpeg: &Path, path: &Path, packets: &[crate::Packet], stream: &str) {
    let hashed = framehash(ffmpeg, path, stream);
    assert_eq!(hashed.len(), packets.len(), "{}", path.display());
    let shift = packets[0].dts - hashed[0].dts;
    for (packet, hashed) in packets.iter().zip(&hashed) {
        assert_eq!(sha256(&packet.payload), hashed.hash, "{}", path.display());
        assert_eq!(packet.duration, hashed.duration, "{}", path.display());
        assert_eq!(packet.dts - hashed.dts, shift, "{}", path.display());
        assert_eq!(packet.pts - hashed.pts, shift, "{}", path.display());
        assert_eq!(
            packet.pts - packet.dts,
            hashed.pts - hashed.dts,
            "{}",
            path.display()
        );
        assert_eq!(packet.keyframe, hashed.keyframe, "{}", path.display());
    }
}

fn compare_file(ffmpeg: &Path, ffprobe: &Path, path: &Path) {
    let bytes = std::fs::read(path).unwrap();
    let mut source = SliceSource::new(&bytes);
    let mut movie = IsoBmff::open(&mut source).unwrap_or_else(|err| {
        panic!("open {}: {err}", path.display());
    });
    let track_id = movie.streams()[0].params.track_id;
    let kind = movie.streams()[0].params.kind;
    assert!(matches!(kind, MediaKind::Video(_)));
    if let MediaKind::Video(video) = kind {
        assert!(video.width > 0 && video.height > 0);
    }
    let mut ours = Vec::new();
    while let Some(packet) = movie.read_packet(&mut source).unwrap() {
        if packet.track_id == track_id {
            ours.push(packet);
        }
    }
    let hashed = framehash(ffmpeg, path, "v:0");
    assert_eq!(ours.len(), hashed.len(), "{}", path.display());
    let shift = ours[0].dts - hashed[0].dts;
    for (ours, theirs) in ours.iter().zip(&hashed) {
        assert_eq!(ours.payload.len(), theirs.size, "{}", path.display());
        assert_eq!(sha256(&ours.payload), theirs.hash, "{}", path.display());
        assert_eq!(ours.duration, theirs.duration);
        assert_eq!(ours.dts - theirs.dts, shift);
        assert_eq!(ours.pts - theirs.pts, shift);
        assert_eq!(ours.pts - ours.dts, theirs.pts - theirs.dts);
        assert_eq!(ours.keyframe, theirs.keyframe);
    }
    let probed = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .unwrap();
    let line = String::from_utf8_lossy(&probed.stdout);
    assert!(line.contains("h264"), "{line}");
    let _ = ffprobe;
}

fn framehash(ffmpeg: &Path, path: &Path, stream: &str) -> Vec<OraclePacket> {
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-c", "copy", "-f", "framehash", "-hash", "sha256", "-"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout);
    let mut packets = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let fields: Vec<_> = line.split(',').map(str::trim).collect();
        if fields.len() < 6 {
            continue;
        }
        packets.push(OraclePacket {
            dts: fields[1].parse().unwrap(),
            pts: fields[2].parse().unwrap(),
            duration: fields[3].parse().unwrap(),
            size: fields[4].parse().unwrap(),
            hash: fields[5].to_string(),
            keyframe: false,
        });
    }
    let probed = Command::new(tool("ffprobe").unwrap())
        .args([
            "-v",
            "error",
            "-select_streams",
            stream,
            "-show_entries",
            "packet=flags",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .unwrap();
    let flags: Vec<_> = String::from_utf8_lossy(&probed.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.contains('K'))
        .collect();
    assert_eq!(flags.len(), packets.len());
    for (packet, keyframe) in packets.iter_mut().zip(flags) {
        packet.keyframe = keyframe;
    }
    packets
}

fn sha256(bytes: &[u8]) -> String {
    let mut child = Command::new("openssl")
        .args(["dgst", "-sha256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(bytes).unwrap();
    let output = child.wait_with_output().unwrap();
    let text = String::from_utf8_lossy(&output.stdout);
    text.split_whitespace().last().unwrap().trim().to_string()
}

#[test]
fn local_files_open_when_the_folder_is_present() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    if !root.is_dir() {
        eprintln!("not available: {}", root.display());
        return;
    }
    let mut opened = 0u32;
    for name in [
        "16929689_1920_1080_30fps.mp4",
        "IMG_8146.MP4",
        "IMG_9197.mov",
        "Expo - You Don't Need a Mac to Build iOS Apps Anymore.m4a",
    ] {
        let path = root.join(name);
        if !path.is_file() {
            eprintln!("not available: {}", path.display());
            continue;
        }
        let mut source = FileSource::open(&path).unwrap();
        let mut movie = IsoBmff::open(&mut source).unwrap_or_else(|err| {
            panic!("open {}: {err}", path.display());
        });
        assert!(!movie.streams().is_empty(), "{}", path.display());
        let mut packets = 0u32;
        while packets < 8 {
            match movie.read_packet(&mut source).unwrap() {
                Some(_) => packets += 1,
                None => break,
            }
        }
        assert!(packets > 0, "{}", path.display());
        let track = movie.streams()[0].params.track_id;
        let landed = movie.seek(&mut source, track, 0).unwrap();
        assert!(landed.used_index);
        opened += 1;
    }
    assert!(opened >= 3, "expected mp4, mov, and m4a");
    let webm = root.join("file_example_WEBM_1920_3_7MB.webm");
    if webm.is_file() {
        let mut source = FileSource::open(&webm).unwrap();
        assert!(IsoBmff::open(&mut source).is_err());
    }
}

/// Remux real files when `ISOBMFF_LOCAL_WRITE=1`.
///
/// Files at or under `ISOBMFF_LOCAL_MAX_BYTES` (default 300 MiB) are read,
/// written, and read back. Larger files are opened, sampled, and seeked.
/// `ISOBMFF_LOCAL_MATCH` limits the run to names that contain that string.
#[test]
fn local_files_round_trip_when_write_is_requested() {
    if std::env::var("ISOBMFF_LOCAL_WRITE").ok().as_deref() != Some("1") {
        eprintln!("not run: set ISOBMFF_LOCAL_WRITE=1 to remux real media");
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    if !root.is_dir() {
        eprintln!("not available: {}", root.display());
        return;
    }
    let max_bytes = std::env::var("ISOBMFF_LOCAL_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(300 * 1024 * 1024);
    let only = std::env::var("ISOBMFF_LOCAL_MATCH").ok();
    let ffmpeg = tool("ffmpeg");
    let dir = std::env::temp_dir().join(format!("isobmff-local-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut paths = iso_paths(&root);
    paths.sort();
    let mut failures = Vec::new();
    let mut opened = 0u32;
    let mut wrote = 0u32;
    for path in paths {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        if only.as_ref().is_some_and(|filter| !name.contains(filter)) {
            continue;
        }
        let size = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
        let started = Instant::now();
        let dest = dir.join(format!("out-{wrote}.mp4"));
        let result = if size <= max_bytes {
            wrote += 1;
            let outcome = remux_file(&path, &dest, ffmpeg.as_deref());
            if outcome.is_err() {
                let kept = std::env::temp_dir().join(format!("isobmff-fail-{wrote}.mp4"));
                let _ = std::fs::copy(&dest, &kept);
                eprintln!("kept {}", kept.display());
            }
            let _ = std::fs::remove_file(&dest);
            outcome
        } else {
            probe_file(&path)
        };
        match result {
            Ok(summary) => {
                opened += 1;
                eprintln!(
                    "ok {} ms {} {summary}",
                    started.elapsed().as_millis(),
                    path.display()
                );
            }
            Err(err) => {
                eprintln!("FAIL {} {err}", path.display());
                failures.push(format!("{}: {err}", path.display()));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "{} files failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
    if only.is_none() {
        assert!(
            opened >= 3 && wrote >= 1,
            "opened {opened}, remuxed {wrote}"
        );
    } else {
        assert!(opened >= 1, "opened {opened}, remuxed {wrote}");
    }
}

#[test]
fn open_and_seek_stay_under_a_coarse_bound() {
    let mut samples = Vec::new();
    for index in 0..4_000u32 {
        let dts = i64::from(index);
        samples.push(crate::Packet {
            track_id: 0,
            payload: vec![u8::try_from(index & 0xff).unwrap_or(0); 16],
            dts,
            pts: dts,
            duration: 1,
            timescale: crate::Timescale(1_000),
            keyframe: index.is_multiple_of(30),
        });
    }
    let params = crate::StreamParams::video(crate::CodecId::H264, crate::Timescale(1_000), 8, 8);
    let bytes = {
        let mut mux = crate::IsoBmffMux::new(
            std::io::Cursor::new(Vec::new()),
            crate::MuxOptions::default(),
        );
        let id = mux.add_stream(&params).unwrap();
        for mut packet in samples {
            packet.track_id = id;
            mux.write_packet(&packet).unwrap();
        }
        mux.finish_into().unwrap().into_inner()
    };
    let started = Instant::now();
    let mut source = SliceSource::new(&bytes);
    let mut movie = IsoBmff::open(&mut source).unwrap();
    let opened = started.elapsed();
    let _ = movie.read_packet(&mut source).unwrap();
    let seek_at = Instant::now();
    let landed = movie.seek(&mut source, 1, 3_000).unwrap();
    let sought = seek_at.elapsed();
    assert!(landed.used_index);
    assert!(landed.dts <= 3_000);
    assert!(opened < Duration::from_secs(2), "open took {opened:?}");
    assert!(sought < Duration::from_millis(500), "seek took {sought:?}");
    eprintln!("perf open {opened:?} seek {sought:?} bytes {}", bytes.len());
}

#[test]
fn fuzz_probe_open_read_seek_for_the_requested_budget() {
    let seconds: u64 = std::env::var("ISOBMFF_FUZZ_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if seconds == 0 {
        eprintln!("not run: set ISOBMFF_FUZZ_SECS to at least 1800");
        return;
    }
    let seed = std::env::var("ISOBMFF_FUZZ_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0x150B_F00D_u64);
    let mut rng = seed;
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut iterations = 0u64;
    let valid = sample_movie();
    while Instant::now() < deadline {
        iterations += 1;
        let mut bytes = if iterations.is_multiple_of(5) {
            valid.clone()
        } else {
            vec![0u8; 64 + (rng_next(&mut rng) % 2048) as usize]
        };
        let flips = 1 + (rng_next(&mut rng) % 16);
        for _ in 0..flips {
            if bytes.is_empty() {
                break;
            }
            let at = (rng_next(&mut rng) as usize) % bytes.len();
            bytes[at] ^= u8::try_from(rng_next(&mut rng) & 0xff).unwrap_or(1);
        }
        let _ = crate::probe(&bytes);
        let mut source = SliceSource::new(&bytes);
        let Ok(mut movie) = IsoBmff::open(&mut source) else {
            continue;
        };
        for _ in 0..4 {
            if movie.read_packet(&mut source).is_err() {
                break;
            }
        }
        let track = movie.streams().first().map(|stream| stream.params.track_id);
        if let Some(track) = track {
            let ticks = i64::try_from(rng_next(&mut rng) % 10_000).unwrap_or(0);
            let _ = movie.seek(&mut source, track, ticks);
        }
    }
    eprintln!("fuzz iterations {iterations} seed {seed:#x} seconds {seconds}");
    assert!(iterations > 0);
}

#[test]
fn fuzz_mux_bounded_packets_for_the_requested_budget() {
    let seconds: u64 = std::env::var("ISOBMFF_MUX_FUZZ_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if seconds == 0 {
        eprintln!("not run: set ISOBMFF_MUX_FUZZ_SECS for a bounded mux campaign");
        return;
    }
    let seed = std::env::var("ISOBMFF_MUX_FUZZ_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0x150B_F11D_u64);
    let mut rng = seed;
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut iterations = 0u64;
    while Instant::now() < deadline {
        iterations += 1;
        let step = rng;
        if let Err(err) = mux_once(&mut rng) {
            panic!("mux fuzz iteration {iterations} seed {seed:#x} step {step:#x}: {err}");
        }
    }
    eprintln!("mux fuzz iterations {iterations} seed {seed:#x} seconds {seconds}");
    assert!(iterations > 0);
}

fn mux_once(rng: &mut u64) -> Result<(), String> {
    let brand = match rng_next(rng) % 5 {
        0 => crate::FileBrand::Mp4,
        1 => crate::FileBrand::Mov,
        2 => crate::FileBrand::M4a,
        3 => crate::FileBrand::M4v,
        _ => crate::FileBrand::ThreeGp,
    };
    let options = crate::MuxOptions {
        brand,
        moov_at_end: !rng_next(rng).is_multiple_of(2),
        fragmented: rng_next(rng).is_multiple_of(2),
        samples_per_fragment: u32::try_from(1 + (rng_next(rng) % 8)).unwrap_or(1),
        movie_timescale: None,
    };
    let mut mux = crate::IsoBmffMux::new(std::io::Cursor::new(Vec::new()), options);
    let track_count = 1 + (rng_next(rng) % 3);
    let mut ids = Vec::new();
    let mut scales = Vec::new();
    for index in 0..track_count {
        let params = if index.is_multiple_of(2) {
            crate::StreamParams::video(crate::CodecId::H264, crate::Timescale(30), 16, 16)
        } else {
            crate::StreamParams::audio(
                crate::CodecId::Aac,
                crate::Timescale(48_000),
                48_000,
                2,
                16,
                crate::SampleFormat::S16Le,
            )
        };
        let id = mux.add_stream(&params).map_err(|err| err.to_string())?;
        ids.push(id);
        scales.push(params.timescale);
    }
    let Some(first) = ids.first().copied() else {
        return Err(String::from("no track"));
    };
    let Some(scale) = scales.first().copied() else {
        return Err(String::from("no timescale"));
    };
    let rejected = crate::Packet {
        track_id: first,
        payload: vec![9],
        dts: 5,
        pts: 5,
        duration: 1,
        timescale: scale,
        keyframe: true,
    };
    if mux.write_packet(&rejected).is_ok() {
        return Err(String::from("a decode time that skips 0 was accepted"));
    }
    let packets = 1 + (rng_next(rng) % 12);
    let mut next = vec![0i64; ids.len()];
    let mut expected = Vec::new();
    for step in 0..packets {
        let track = usize::try_from(step % track_count).unwrap_or(0);
        let duration = u32::try_from(1 + (rng_next(rng) % 30)).unwrap_or(1);
        let len = 1 + (rng_next(rng) % 24);
        let mut payload = Vec::new();
        for _ in 0..len {
            payload.push(u8::try_from(rng_next(rng) & 0xff).unwrap_or(1));
        }
        let dts = next[track];
        let packet = crate::Packet {
            track_id: ids[track],
            payload: payload.clone(),
            dts,
            pts: dts.saturating_add(i64::try_from(rng_next(rng) % 3).unwrap_or(0)),
            duration,
            timescale: scales[track],
            keyframe: dts == 0 || rng_next(rng).is_multiple_of(3),
        };
        mux.write_packet(&packet)
            .map_err(|err| format!("write {err}"))?;
        next[track] = dts.saturating_add(i64::from(duration));
        expected.push((ids[track], payload, packet.pts.saturating_sub(dts)));
    }
    let bytes = mux
        .finish_into()
        .map_err(|err| format!("finish {err}"))?
        .into_inner();
    let mut source = SliceSource::new(&bytes);
    let mut movie = IsoBmff::open(&mut source).map_err(|err| format!("open {err}"))?;
    let mut got = Vec::new();
    while let Some(packet) = movie
        .read_packet(&mut source)
        .map_err(|err| format!("read {err}"))?
    {
        got.push((
            packet.track_id,
            packet.payload,
            packet.pts.saturating_sub(packet.dts),
        ));
    }
    if got != expected {
        let wrote: Vec<u32> = expected.iter().map(|(id, _, _)| *id).collect();
        let read: Vec<u32> = got.iter().map(|(id, _, _)| *id).collect();
        return Err(format!("track order wrote {wrote:?} read {read:?}"));
    }
    Ok(())
}

fn sample_movie() -> Vec<u8> {
    let mut mux = crate::IsoBmffMux::new(
        std::io::Cursor::new(Vec::new()),
        crate::MuxOptions::default(),
    );
    let params = crate::StreamParams::video(crate::CodecId::H264, crate::Timescale(30), 2, 2);
    let id = mux.add_stream(&params).unwrap();
    for index in 0..4u32 {
        mux.write_packet(&crate::Packet {
            track_id: id,
            payload: vec![u8::try_from(index & 0xff).unwrap_or(0), 1, 2, 3],
            dts: i64::from(index) * 30,
            pts: i64::from(index) * 30,
            duration: 30,
            timescale: crate::Timescale(30),
            keyframe: true,
        })
        .unwrap();
    }
    mux.finish_into().unwrap().into_inner()
}

fn rng_next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn iso_paths(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && iso_extension(path))
        .collect()
}

fn iso_extension(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return false;
    };
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "mp4" | "mov" | "m4a" | "m4v" | "3gp"
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Seen {
    track_id: u32,
    dts: i64,
    pts: i64,
    duration: u32,
    keyframe: bool,
    hash: u128,
}

fn payload_hash(bytes: &[u8]) -> u128 {
    let mut high = 0xcbf2_9ce4_8422_2325u64;
    let mut low = 0x8422_2325_cbf2_9ce4u64;
    for &byte in bytes {
        high ^= u64::from(byte);
        high = high.wrapping_mul(0x100_0000_01b3);
        low = low.wrapping_mul(0x0100_0193).wrapping_add(u64::from(byte));
    }
    let length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    (u128::from(high) << 64) | u128::from(low ^ length)
}

fn seen_from(packet: &crate::Packet) -> Seen {
    Seen {
        track_id: packet.track_id,
        dts: packet.dts,
        pts: packet.pts,
        duration: packet.duration,
        keyframe: packet.keyframe,
        hash: payload_hash(&packet.payload),
    }
}

fn remux_file(path: &Path, dest: &Path, ffmpeg: Option<&Path>) -> Result<String, String> {
    let mut source = FileSource::open(path).map_err(|err| format!("open: {err}"))?;
    let mut movie = IsoBmff::open(&mut source).map_err(|err| format!("open: {err}"))?;
    let streams = movie.streams().to_vec();
    if streams.is_empty() {
        return Err("no streams".to_string());
    }
    let brands = brands_text(movie.brands());
    let summary = stream_summary(&streams);
    let movie_timescale = movie.movie_timescale();
    let mut options = crate::MuxOptions::default();
    if movie_timescale.get() != 0 {
        options.movie_timescale = Some(movie_timescale.get());
    }
    let mut mux = crate::IsoBmffMux::new(
        std::fs::File::create(dest).map_err(|err| format!("create: {err}"))?,
        options,
    );
    for stream in &streams {
        let id = mux
            .add_stream(&stream.params)
            .map_err(|err| format!("add stream: {err}"))?;
        if !stream.edits.is_empty() {
            mux.set_edits(id, stream.edits.clone())
                .map_err(|err| format!("edits: {err}"))?;
        }
    }
    let mut seen = Vec::new();
    loop {
        match movie.read_packet(&mut source) {
            Ok(Some(packet)) => {
                seen.push(seen_from(&packet));
                mux.write_packet(&packet)
                    .map_err(|err| format!("write packet: {err}"))?;
            }
            Ok(None) => break,
            Err(err) => return Err(format!("read: {err}")),
        }
    }
    if seen.is_empty() {
        return Err("no packets".to_string());
    }
    let file = mux.finish_into().map_err(|err| format!("finish: {err}"))?;
    file.sync_all().map_err(|err| format!("sync: {err}"))?;
    drop(file);
    let written = read_movie(dest)?;
    packets_match(&seen, &written.seen)?;
    streams_match(&streams, &written.streams)?;
    let mut again = FileSource::open(dest).map_err(|err| format!("reopen: {err}"))?;
    let mut reopened = IsoBmff::open(&mut again).map_err(|err| format!("reopen: {err}"))?;
    let track = seek_track(&written.streams).ok_or_else(|| "remux has no stream".to_string())?;
    let landed = reopened
        .seek(&mut again, track, 0)
        .map_err(|err| format!("remux seek: {err}"))?;
    if !landed.used_index {
        return Err("remux seek did not use an index".to_string());
    }
    let timing = edit_note(
        movie_timescale,
        written.movie_timescale,
        &streams,
        &written.streams,
        &seen,
    )?;
    let oracle = match ffmpeg {
        Some(ffmpeg) => match ffmpeg_payloads(ffmpeg, path, dest) {
            Ok(text) => text,
            Err(err) => return Err(format!("{err}; wrote {}", count_tracks(&seen))),
        },
        None => "ffmpeg not available".to_string(),
    };
    let bytes = std::fs::metadata(dest).map(|meta| meta.len()).unwrap_or(0);
    Ok(format!(
        "remux packets={} bytes={bytes} brands=[{brands}] {summary} | {timing} | {oracle}",
        seen.len()
    ))
}

struct MovieRead {
    movie_timescale: crate::Timescale,
    streams: Vec<Stream>,
    seen: Vec<Seen>,
}

fn read_movie(path: &Path) -> Result<MovieRead, String> {
    let mut source = FileSource::open(path).map_err(|err| format!("open: {err}"))?;
    let mut movie = IsoBmff::open(&mut source).map_err(|err| format!("open: {err}"))?;
    if movie.streams().is_empty() {
        return Err("no streams".to_string());
    }
    let movie_timescale = movie.movie_timescale();
    let streams = movie.streams().to_vec();
    let mut seen = Vec::new();
    loop {
        match movie.read_packet(&mut source) {
            Ok(Some(packet)) => seen.push(seen_from(&packet)),
            Ok(None) => break,
            Err(err) => return Err(format!("read: {err}")),
        }
    }
    if seen.is_empty() {
        return Err("no packets".to_string());
    }
    Ok(MovieRead {
        movie_timescale,
        streams,
        seen,
    })
}

fn count_tracks(packets: &[Seen]) -> String {
    let mut counts: Vec<(u32, usize)> = Vec::new();
    for packet in packets {
        if let Some(found) = counts
            .iter_mut()
            .find(|(track, _)| *track == packet.track_id)
        {
            found.1 += 1;
        } else {
            counts.push((packet.track_id, 1));
        }
    }
    counts
        .iter()
        .map(|(track, count)| format!("{track}:{count}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn packets_match(before: &[Seen], after: &[Seen]) -> Result<(), String> {
    if before.len() != after.len() {
        return Err(format!("packet count {} vs {}", before.len(), after.len()));
    }
    for (index, (before, after)) in before.iter().zip(after).enumerate() {
        if before.track_id != after.track_id
            || before.dts != after.dts
            || before.pts != after.pts
            || before.duration != after.duration
            || before.keyframe != after.keyframe
        {
            return Err(format!(
                "packet {index} timing track {} dts {} pts {} duration {} keyframe {}",
                before.track_id, before.dts, before.pts, before.duration, before.keyframe
            ));
        }
        if before.hash != after.hash {
            return Err(format!("packet {index} payload changed"));
        }
    }
    Ok(())
}

fn probe_file(path: &Path) -> Result<String, String> {
    let mut source = FileSource::open(path).map_err(|err| format!("open: {err}"))?;
    let mut movie = IsoBmff::open(&mut source).map_err(|err| format!("open: {err}"))?;
    let streams = movie.streams().to_vec();
    if streams.is_empty() {
        return Err("no streams".to_string());
    }
    let summary = stream_summary(&streams);
    let brands = brands_text(movie.brands());
    let mut packets = 0u32;
    loop {
        if packets == 8 {
            break;
        }
        match movie.read_packet(&mut source) {
            Ok(Some(_)) => packets += 1,
            Ok(None) => break,
            Err(err) => return Err(format!("read: {err}")),
        }
    }
    if packets == 0 {
        return Err("no packets".to_string());
    }
    let track = seek_track(&streams).ok_or_else(|| "no stream".to_string())?;
    let duration = streams
        .iter()
        .find(|stream| stream.params.track_id == track)
        .map(|stream| stream.params.duration_ticks)
        .unwrap_or(0);
    let mid = i64::try_from(duration / 2).unwrap_or(i64::MAX);
    let start = movie
        .seek(&mut source, track, 0)
        .map_err(|err| format!("seek 0: {err}"))?;
    if !start.used_index {
        return Err("seek to 0 did not use an index".to_string());
    }
    let landed = movie
        .seek(&mut source, track, mid)
        .map_err(|err| format!("seek mid: {err}"))?;
    if !landed.used_index {
        return Err("seek into the file did not use an index".to_string());
    }
    if landed.dts > mid {
        return Err(format!("seek landed at {} after {mid}", landed.dts));
    }
    match movie.read_packet(&mut source) {
        Ok(Some(_)) => {}
        Ok(None) => return Err("seek landed past the last packet".to_string()),
        Err(err) => return Err(format!("read after seek: {err}")),
    }
    Ok(format!(
        "read-only packets>={packets} brands=[{brands}] {summary} | seek dts {}",
        landed.dts
    ))
}

fn seek_track(streams: &[Stream]) -> Option<u32> {
    streams
        .iter()
        .find(|stream| matches!(stream.params.kind, MediaKind::Video(_)))
        .or_else(|| streams.first())
        .map(|stream| stream.params.track_id)
}

fn streams_match(before: &[Stream], after: &[Stream]) -> Result<(), String> {
    if before.len() != after.len() {
        return Err(format!("stream count {} vs {}", before.len(), after.len()));
    }
    for (index, (left, right)) in before.iter().zip(after).enumerate() {
        if left.params.track_id != right.params.track_id {
            return Err(format!("track {index} id changed"));
        }
        if left.params.codec != right.params.codec {
            return Err(format!(
                "track {index} codec {:?} vs {:?}",
                left.params.codec, right.params.codec
            ));
        }
        if left.params.timescale != right.params.timescale {
            return Err(format!("track {index} timescale changed"));
        }
        if left.params.sample_entry != right.params.sample_entry {
            return Err(format!(
                "track {index} sample entry {} vs {}",
                fourcc_text(&left.params.sample_entry),
                fourcc_text(&right.params.sample_entry)
            ));
        }
        if left.params.color != right.params.color {
            return Err(format!("track {index} color changed"));
        }
        match (&left.params.kind, &right.params.kind) {
            (MediaKind::Video(left_video), MediaKind::Video(right_video)) => {
                if left_video.width != right_video.width || left_video.height != right_video.height
                {
                    return Err(format!("track {index} dimensions changed"));
                }
                if left.params.extradata != right.params.extradata {
                    return Err(format!("track {index} video configuration changed"));
                }
            }
            (MediaKind::Audio(left_audio), MediaKind::Audio(right_audio)) => {
                if left_audio.sample_rate != right_audio.sample_rate
                    || left_audio.channels != right_audio.channels
                {
                    return Err(format!("track {index} audio geometry changed"));
                }
                if left.params.extradata != right.params.extradata {
                    return Err(format!("track {index} audio configuration changed"));
                }
            }
            (MediaKind::Data, MediaKind::Data) => {}
            _ => return Err(format!("track {index} kind changed")),
        }
    }
    Ok(())
}

fn edit_note(
    before_scale: crate::Timescale,
    after_scale: crate::Timescale,
    before: &[Stream],
    after: &[Stream],
    packets: &[Seen],
) -> Result<String, String> {
    if before_scale != after_scale {
        return Err(format!(
            "movie timescale {} -> {}",
            before_scale.get(),
            after_scale.get()
        ));
    }
    if before.len() != after.len() {
        return Err(format!("track count {} vs {}", before.len(), after.len()));
    }
    let mut filled = Vec::new();
    for (index, (left, right)) in before.iter().zip(after).enumerate() {
        match edit_change(left, right, before_scale.get(), packets) {
            EditChange::Same => {}
            EditChange::Filled => filled.push(index),
            EditChange::Changed => {
                return Err(format!(
                    "track {index} edits {:?} -> {:?}",
                    left.edits, right.edits
                ));
            }
        }
    }
    let note = if filled.is_empty() {
        String::new()
    } else {
        format!(", filled unknown edit duration on tracks {filled:?}")
    };
    Ok(format!("movie timescale {}{note}", before_scale.get()))
}

enum EditChange {
    Same,
    Filled,
    Changed,
}

/// A zero segment duration with a media time is an unknown length. Fragmented
/// files store it that way. The writer fills it from the samples it copied,
/// and `elst` version 0 keeps that duration in 32 bits.
fn edit_change(before: &Stream, after: &Stream, movie_scale: u32, packets: &[Seen]) -> EditChange {
    if before.edits.len() != after.edits.len() {
        return EditChange::Changed;
    }
    let ticks = media_ticks(packets, before.params.track_id);
    let stored = stored_edit_duration(ticks, before.params.timescale.get(), movie_scale);
    let mut filled = false;
    for (left, right) in before.edits.iter().zip(&after.edits) {
        if left.media_time != right.media_time || left.media_rate != right.media_rate {
            return EditChange::Changed;
        }
        if left.segment_duration == right.segment_duration {
            continue;
        }
        let unknown = left.segment_duration == 0
            && left.media_time.is_some()
            && ticks > 0
            && right.segment_duration == stored;
        if !unknown {
            return EditChange::Changed;
        }
        filled = true;
    }
    if filled {
        EditChange::Filled
    } else {
        EditChange::Same
    }
}

fn media_ticks(packets: &[Seen], track_id: u32) -> u64 {
    packets
        .iter()
        .filter(|packet| packet.track_id == track_id)
        .fold(0u64, |sum, packet| {
            sum.saturating_add(u64::from(packet.duration))
        })
}

fn stored_edit_duration(ticks: u64, from: u32, to: u32) -> u64 {
    if ticks == 0 {
        return 0;
    }
    let scaled = scaled_media(ticks, from, to);
    u64::from(u32::try_from(scaled).unwrap_or(u32::MAX))
}

fn scaled_media(ticks: u64, from: u32, to: u32) -> u64 {
    if from == 0 {
        return 0;
    }
    let value = u128::from(ticks) * u128::from(to) / u128::from(from);
    u64::try_from(value).unwrap_or(u64::MAX).max(1)
}

fn stream_summary(streams: &[Stream]) -> String {
    streams
        .iter()
        .map(|stream| {
            let codec = codec_text(&stream.params.codec);
            let kind = match stream.params.kind {
                MediaKind::Video(video) => format!("video {}x{}", video.width, video.height),
                MediaKind::Audio(audio) => {
                    format!("audio {}Hz ch{}", audio.sample_rate, audio.channels)
                }
                MediaKind::Data => "data".to_string(),
            };
            format!(
                "{codec}/{kind}/{}",
                fourcc_text(&stream.params.sample_entry)
            )
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

fn codec_text(codec: &crate::CodecId) -> String {
    match codec {
        crate::CodecId::Unknown(code) => format!("Unknown({})", fourcc_text(code)),
        other => format!("{other:?}"),
    }
}

fn brands_text(brands: &[[u8; 4]]) -> String {
    brands.iter().map(fourcc_text).collect::<Vec<_>>().join(",")
}

fn fourcc_text(code: &[u8; 4]) -> String {
    code.iter()
        .map(|byte| {
            if byte.is_ascii_graphic() || *byte == b' ' {
                char::from(*byte).to_string()
            } else {
                format!("\\x{byte:02x}")
            }
        })
        .collect()
}

struct FfPacket {
    dts: i64,
    pts: i64,
    duration: i64,
    size: u64,
    hash: String,
}

fn ffmpeg_payloads(ffmpeg: &Path, original: &Path, remux: &Path) -> Result<String, String> {
    let left = framehash_streams(ffmpeg, original)?;
    let right = framehash_streams(ffmpeg, remux)?;
    if left.len() != right.len() {
        return Err(format!("ffmpeg streams {} vs {}", left.len(), right.len()));
    }
    let mut shifts = Vec::new();
    let mut timing = true;
    for (index, (before, after)) in left.iter().zip(right.iter()).enumerate() {
        if before.len() != after.len() {
            return Err(format!(
                "ffmpeg stream {index} packets {} vs {}",
                before.len(),
                after.len()
            ));
        }
        let mut shift: Option<i64> = None;
        for (packet_index, (before, after)) in before.iter().zip(after).enumerate() {
            if before.size != after.size || before.hash != after.hash {
                return Err(format!(
                    "ffmpeg stream {index} packet {packet_index} payload mismatch"
                ));
            }
            if before.duration != after.duration || before.pts - before.dts != after.pts - after.dts
            {
                timing = false;
            }
            let step = after.dts.wrapping_sub(before.dts);
            match shift {
                None => shift = Some(step),
                Some(shift) if shift != step => timing = false,
                Some(_) => {}
            }
        }
        shifts.push(shift.unwrap_or(0));
    }
    if timing {
        Ok(format!("ffmpeg payloads match, dts shift {shifts:?}"))
    } else {
        Ok(format!(
            "ffmpeg payloads match, timestamps diverged, dts shift {shifts:?}"
        ))
    }
}

fn framehash_streams(ffmpeg: &Path, path: &Path) -> Result<Vec<Vec<FfPacket>>, String> {
    let output = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-ignore_editlist",
            "1",
            "-i",
        ])
        .arg(path)
        .args([
            "-map",
            "0",
            "-c",
            "copy",
            "-f",
            "framehash",
            "-hash",
            "sha256",
            "-",
        ])
        .output()
        .map_err(|err| format!("ffmpeg: {err}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ffmpeg: {}", clip_chars(stderr.trim(), 400)));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut streams: Vec<Vec<FfPacket>> = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split(',').map(str::trim).collect();
        if fields.len() < 6 {
            return Err(format!("short framehash line: {line}"));
        }
        let stream: usize = fields[0]
            .parse()
            .map_err(|_| format!("framehash stream: {line}"))?;
        let packet = FfPacket {
            dts: fields[1]
                .parse()
                .map_err(|_| format!("framehash dts: {line}"))?,
            pts: fields[2]
                .parse()
                .map_err(|_| format!("framehash pts: {line}"))?,
            duration: fields[3]
                .parse()
                .map_err(|_| format!("framehash duration: {line}"))?,
            size: fields[4]
                .parse()
                .map_err(|_| format!("framehash size: {line}"))?,
            hash: fields[5].to_string(),
        };
        if stream > 32 {
            return Err(format!("framehash stream index {stream}"));
        }
        while streams.len() <= stream {
            streams.push(Vec::new());
        }
        let Some(bucket) = streams.get_mut(stream) else {
            return Err("framehash stream missing".to_string());
        };
        bucket.push(packet);
    }
    streams.retain(|stream| !stream.is_empty());
    if streams.is_empty() {
        return Err("ffmpeg returned no packets".to_string());
    }
    Ok(streams)
}

fn clip_chars(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    &text[..end]
}

fn tool(name: &str) -> Option<PathBuf> {
    let output = Command::new("which").arg(name).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8(output.stdout).ok()?;
    let path = PathBuf::from(path.trim());
    path.is_file().then_some(path)
}

fn run(program: &Path, args: &[&str]) {
    let status = Command::new(program).args(args).status().unwrap();
    assert!(status.success(), "{} failed", program.display());
}
