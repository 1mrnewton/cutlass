//! This demuxer against Vorbis files from libvorbis and FFmpeg, read back
//! by libavformat and GStreamer.
//!
//! Producers: libvorbis through GStreamer's `vorbisenc ! oggmux`, FFmpeg's
//! native Vorbis encoder through libavformat's Ogg muxer, libavformat's
//! stream copy of the local WebM track, and libavformat's stream copy cut
//! (`-ss` with `-c copy`), whose first page completes more samples than its
//! granule. Oracles: libavformat (ffprobe packets and `-f data` payloads),
//! GStreamer's `oggdemux` (each packet's granule end), and the sample
//! counts `vorbis` in ffmpeg and `vorbisdec` in GStreamer decode.
//!
//! libavformat's parser starts each stream as if a short block came before
//! the first packet, so its first packet has a duration and starts that
//! much earlier. From the second packet on, both demuxers give the same
//! times, and every packet ends at the same sample.

use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_codec::{CodecId, Packet, StreamCodec};
use cutlass_container::{Demux, Landing, SeekTo};
use cutlass_ogg::VorbisComment;

use crate::oracle::{duration, open, pts, read_all};
use crate::support::{
    environment, ffmpeg, ffmpeg_decoded_samples, ffmpeg_payload, ffprobe_packets, ffprobe_stream,
    gstreamer_decoded_samples, gstreamer_on_path, gstreamer_packets, pick, scratch, sha256_hex,
};

/// How a file's granules relate to its samples.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Granules {
    /// Every granule is the end of its page's last packet.
    Exact,
    /// The remux rebuilt each granule from a millisecond timestamp.
    Rebuilt,
}

struct Produced {
    name: String,
    path: PathBuf,
    granules: Granules,
}

/// PCM from the local music file: `seconds` from 30 s in, with `filter`.
fn wav(root: &Path, dir: &Path, name: &str, seconds: &str, filter: &[&str]) -> Option<PathBuf> {
    let source = pick(root, &[".m4a"])?;
    let out = dir.join(format!("{name}.wav"));
    let mut args = vec!["-ss", "30", "-t", seconds, "-i", source.to_str()?, "-vn"];
    args.extend_from_slice(filter);
    args.push(out.to_str()?);
    ffmpeg(&args);
    Some(out)
}

/// libvorbis through GStreamer: `wavparse ! audioconvert ! [convert] !
/// vorbisenc props ! oggmux`.
fn libvorbis(wav: &Path, out: &Path, convert: &[&str], props: &[&str]) -> bool {
    let mut args = vec![
        "-q".to_owned(),
        "filesrc".to_owned(),
        format!("location={}", wav.display()),
        "!".to_owned(),
        "wavparse".to_owned(),
        "!".to_owned(),
        "audioconvert".to_owned(),
    ];
    for element in convert {
        args.push("!".to_owned());
        args.push((*element).to_owned());
    }
    args.push("!".to_owned());
    args.push("vorbisenc".to_owned());
    args.extend(props.iter().map(|prop| (*prop).to_owned()));
    args.extend(
        ["!", "oggmux", "!", "filesink"]
            .iter()
            .map(|part| (*part).to_owned()),
    );
    args.push(format!("location={}", out.display()));
    Command::new("gst-launch-1.0")
        .args(&args)
        .status()
        .is_ok_and(|status| status.success())
}

fn produce(root: &Path, dir: &Path) -> Vec<Produced> {
    let mut files = Vec::new();
    let mut add = |name: &str, path: PathBuf, granules| {
        files.push(Produced {
            name: name.to_owned(),
            path,
            granules,
        });
    };
    let Some(stereo) = wav(root, dir, "stereo", "12", &[]) else {
        return files;
    };
    if gstreamer_on_path() {
        let cases: [(&str, &[&str], &[&str]); 3] = [
            ("libvorbis-q4-stereo", &[], &["quality=0.4"]),
            (
                "libvorbis-q0-mono-22050",
                &["audioresample", "audio/x-raw,channels=1,rate=22050"],
                &["quality=0.0"],
            ),
            (
                "libvorbis-managed-96k-44100",
                &["audioresample", "audio/x-raw,rate=44100"],
                &["bitrate=96000", "managed=true"],
            ),
        ];
        for (name, convert, props) in cases {
            let out = dir.join(format!("{name}.ogg"));
            assert!(libvorbis(&stereo, &out, convert, props), "{name}");
            add(name, out, Granules::Exact);
        }
        if let Some(surround) = wav(
            root,
            dir,
            "surround",
            "8",
            &[
                "-af",
                "pan=5.1|FL=c0|FR=c1|FC=0.5*c0+0.5*c1|LFE=0.1*c0|BL=c0|BR=c1",
            ],
        ) {
            let out = dir.join("libvorbis-q3-5.1.ogg");
            assert!(libvorbis(&surround, &out, &[], &["quality=0.3"]), "5.1");
            add("libvorbis-q3-5.1", out, Granules::Exact);
        }
    } else {
        eprintln!("skipped the libvorbis files: gst-launch-1.0 is not on PATH");
    }
    let native = dir.join("ffmpeg-native-q4.ogg");
    ffmpeg(&[
        "-i",
        stereo.to_str().expect("utf8"),
        "-c:a",
        "vorbis",
        "-strict",
        "-2",
        "-q:a",
        "4",
        native.to_str().expect("utf8"),
    ]);
    // A stream copy of the libvorbis file from 1 s in: the first page
    // holds packets from before the cut, so the stream starts below zero.
    let libvorbis_q4 = dir.join("libvorbis-q4-stereo.ogg");
    if libvorbis_q4.is_file() {
        let cut = dir.join("ffmpeg-copy-cut.ogg");
        ffmpeg(&[
            "-ss",
            "1.0",
            "-i",
            libvorbis_q4.to_str().expect("utf8"),
            "-c",
            "copy",
            cut.to_str().expect("utf8"),
        ]);
        add("ffmpeg-copy-cut", cut, Granules::Exact);
    }
    add("ffmpeg-native-q4", native, Granules::Exact);
    if let Some(webm) = pick(root, &[".webm"]) {
        let out = dir.join("webm-copy.ogg");
        ffmpeg(&[
            "-i",
            webm.to_str().expect("utf8"),
            "-vn",
            "-c:a",
            "copy",
            out.to_str().expect("utf8"),
        ]);
        add("webm-copy", out, Granules::Rebuilt);
    }
    files
}

/// The three packets of Xiph-laced extradata.
fn xiph_split(bytes: &[u8]) -> [&[u8]; 3] {
    assert_eq!(bytes[0], 2, "Xiph lacing");
    let mut at = 1;
    let mut lengths = [0_usize; 2];
    for length in &mut lengths {
        loop {
            let byte = bytes[at];
            at += 1;
            *length += usize::from(byte);
            if byte < 255 {
                break;
            }
        }
    }
    let (first, rest) = bytes[at..].split_at(lengths[0]);
    let (second, third) = rest.split_at(lengths[1]);
    [first, second, third]
}

#[test]
fn vorbis_files_demux_to_the_same_packets_times_and_trim() {
    let Some(root) = environment("vorbis_files_demux_to_the_same_packets_times_and_trim") else {
        return;
    };
    let dir = scratch("vorbis-oracle");
    let files = produce(&root, &dir);
    for file in &files {
        check_against_ffmpeg(file, &dir);
        check_seeks(file);
    }
    assert!(files.len() >= 6, "only {} files ran", files.len());
    if let Some(cut) = files.iter().find(|file| file.name == "ffmpeg-copy-cut") {
        let demux = open(&cut.path);
        let start = demux.vorbis(0).and_then(|info| info.start).expect("start");
        assert!(start < 0, "the cut starts at {start}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

fn check_against_ffmpeg(file: &Produced, dir: &Path) {
    let (path, name) = (&file.path, file.name.as_str());
    let mut demux = open(path);
    let probe = ffprobe_stream(path);
    let stream = demux.streams()[0].clone();
    assert_eq!(probe.codec, "vorbis", "{name}");
    assert_eq!(
        stream.params().codec(),
        &StreamCodec::Known(CodecId::Vorbis)
    );
    let rate = stream.params().sample_rate().expect("rate");
    assert_eq!(rate, probe.sample_rate, "{name}");
    assert_eq!(stream.timescale().ticks_per_second(), rate);
    assert_eq!(stream.params().channels(), Some(probe.channels), "{name}");
    // libavformat keeps only the vendor string of the comment packet in its
    // extradata; this crate keeps the packet as stored.
    let [id, comment, setup] = xiph_split(stream.params().extradata());
    assert!(id.starts_with(b"\x01vorbis") && id.len() == 30, "{name}");
    assert!(comment.starts_with(b"\x03vorbis"), "{name}");
    assert!(setup.starts_with(b"\x05vorbis"), "{name}");
    let info = demux.vorbis(0).expect("vorbis info").clone();
    assert_eq!(comment, info.comment.encode(), "{name} comment packet");
    let vendor_only = VorbisComment {
        vendor: info.comment.vendor.clone(),
        ..VorbisComment::default()
    }
    .encode();
    let lace = |len: usize| len / 255 + 1;
    assert_eq!(
        1 + lace(id.len()) + lace(vendor_only.len()) + id.len() + vendor_only.len() + setup.len(),
        probe.extradata_size,
        "{name} extradata"
    );

    let packets = read_all(&mut demux);
    let expected = ffprobe_packets(path);
    assert_eq!(packets.len(), expected.len(), "{name} packet count");
    let payload = ffmpeg_payload(path);
    let blocks = info.id.blocksizes;
    let [short, long] = blocks.map(i64::from);
    // Rebuilt granules are rounded; this crate keeps the packet timeline,
    // which stays within the shortest packet of them.
    let slack = match file.granules {
        Granules::Exact => 0,
        Granules::Rebuilt => short / 2 - 1,
    };
    let near = |mine: i64, theirs: i64| (mine - theirs).abs() <= slack;
    let mut transitions = 0;
    let mut at = 0;
    for (index, (mine, theirs)) in packets.iter().zip(&expected).enumerate() {
        assert_eq!(
            mine.bytes().len(),
            theirs.size,
            "{name} packet {index} size"
        );
        assert_eq!(
            mine.bytes(),
            &payload[at..at + theirs.size],
            "{name} packet {index} bytes"
        );
        at += theirs.size;
        assert!(
            near(pts(mine) + duration(mine), theirs.pts + theirs.duration),
            "{name} packet {index} end: {} + {} against {} + {}",
            pts(mine),
            duration(mine),
            theirs.pts,
            theirs.duration
        );
        // ffprobe gives the first short packet after a long one half a
        // short block, where section 4.3.8 gives a quarter of each block.
        // Its packet ends and ffmpeg's decoded count follow the spec.
        let transition = duration(mine) == long / 4 + short / 4 && theirs.duration == short / 2;
        if index > 0 && !transition {
            assert!(near(pts(mine), theirs.pts), "{name} packet {index} pts");
        }
        transitions += usize::from(index > 0 && transition && pts(mine) != theirs.pts);
        assert!(mine.keyframe());
        assert_eq!(mine.timescale().ticks_per_second(), rate);
    }
    assert_eq!(duration(&packets[0]), 0, "{name} first packet");
    assert_eq!(at, payload.len(), "{name} payload length");
    for index in [0, packets.len() - 1] {
        assert_eq!(
            sha256_hex(packets[index].bytes(), dir),
            expected[index].hash,
            "{name} packet {index} sha256"
        );
    }

    let start = info.start.expect("start");
    let end = info.end.expect("end");
    assert_eq!(pts(&packets[0]), start, "{name}");
    let edit = stream.edits()[0];
    assert_eq!(edit.media_time(), Some(start.max(0)), "{name}");
    assert_eq!(edit.duration(), end - start.max(0), "{name}");
    assert_eq!(
        stream.trim().encoder_delay(),
        Some((-start).max(0).unsigned_abs()),
        "{name}"
    );
    assert!(demux.gaps().is_empty());
    assert_eq!(demux.crc_failures(), 0);
    assert_eq!(demux.lost_pages(), 0);

    for pair in packets.windows(2) {
        assert_eq!(
            pts(&pair[0]) + duration(&pair[0]),
            pts(&pair[1]),
            "{name} contiguous"
        );
    }
    let last = packets.last().expect("last");
    assert!(near(pts(last) + duration(last), end), "{name} end trim");
    if file.granules == Granules::Exact {
        assert_eq!(demux.granule_drift(), 0, "{name}");
    } else {
        assert!(demux.granule_drift() > 0, "{name}");
    }

    // Neither decoder drops the samples before zero that section A.2
    // discards, so both decode from the start to the end.
    let decoded = ffmpeg_decoded_samples(path);
    let span = u64::try_from(end - start).expect("span");
    if file.granules == Granules::Exact {
        assert_eq!(decoded, span, "{name} ffmpeg decoded samples");
    }
    if let Some(theirs) = gstreamer_packets(path) {
        assert_eq!(packets.len(), theirs.len(), "{name} GStreamer packet count");
        // On a first page that starts below zero, oggdemux counts its
        // packets from zero and gives only the last one the page granule.
        let mut shifted = 0;
        for (index, (mine, (size, packet_end))) in packets.iter().zip(&theirs).enumerate() {
            assert_eq!(
                mine.bytes().len(),
                *size,
                "{name} GStreamer packet {index} size"
            );
            let end = pts(mine) + duration(mine);
            if start < 0 && index == shifted && *packet_end == end - start {
                shifted += 1;
                continue;
            }
            assert!(
                near(end, *packet_end),
                "{name} GStreamer packet {index} end"
            );
        }
        if shifted > 0 {
            eprintln!("{name}: oggdemux counts the first {shifted} packets from zero");
        }
        let samples = gstreamer_decoded_samples(path, dir, "vorbisdec", rate).expect("gst decode");
        // libvorbis through vorbisdec returns one sample past the end
        // granule on some streams.
        assert!(
            samples == decoded || samples == decoded + 1,
            "{name} GStreamer decoded {samples}, ffmpeg {decoded}"
        );
    } else {
        eprintln!("{name}: GStreamer comparison skipped, gst-launch-1.0 is not on PATH");
    }
    eprintln!(
        "{name}: {} packets, rate {rate}, blocks {blocks:?}, start {start}, end {end}, trailing {:?}, decoded {decoded}, ffprobe long-to-short durations {transitions}",
        packets.len(),
        info.trailing
    );
}

/// Each seek lands on the packet that holds the target less half a long
/// block. The decoder returns nothing for that packet, and the next packet
/// starts at or before the target.
/// After a seek, packet times count from the landing page's granule. On
/// rebuilt granules they differ from a read from the start by the same
/// rounding as the granule.
fn check_seeks(file: &Produced) {
    let (path, name) = (&file.path, file.name.as_str());
    let mut demux = open(path);
    let blocks = demux.vorbis(0).expect("info").id.blocksizes;
    let preroll_samples = i64::from(blocks[1] / 2);
    let slack = match file.granules {
        Granules::Exact => 0,
        Granules::Rebuilt => i64::from(blocks[0] / 2) - 1,
    };
    let all = read_all(&mut demux);
    let start = pts(&all[0]);
    let last = all.last().expect("last");
    let end = pts(last) + duration(last);
    let targets = [
        start - 1000,
        start,
        start + 1,
        start + preroll_samples * 3,
        start + (end - start) / 3,
        start + (end - start) / 2,
        end - 4_000,
        end - 1,
    ];
    for &target in &targets {
        let landing = demux
            .seek(SeekTo {
                track: 0,
                time: target,
                end: None,
            })
            .expect("seek");
        let Landing::Packet { time, preroll, .. } = landing.landing else {
            panic!("{name} seek {target} landed at the end");
        };
        let aim = (target - preroll_samples).max(start);
        let next = demux.read_packet().expect("read").expect("packet");
        assert_eq!(pts(&next), time, "{name} no stale packet at {target}");
        let index = all
            .iter()
            .position(|packet| {
                (pts(packet) - time).abs() <= slack
                    && duration(packet) == duration(&next)
                    && packet.bytes() == next.bytes()
            })
            .expect("landing is a packet start");
        let held: &Packet = &all[index];
        assert!(
            time <= aim + slack && aim < time + duration(held).max(1) + slack,
            "{name} {target}: {time}"
        );
        let expected_preroll = all[index..]
            .iter()
            .take_while(|packet| pts(packet) < target)
            .count();
        if slack == 0 {
            assert_eq!(
                usize::try_from(preroll).expect("preroll"),
                expected_preroll,
                "{name} preroll at {target}"
            );
        }
        assert!(
            pts(&next) + duration(&next) <= target.max(start) + slack,
            "{name} the packet after the landing starts after {target}"
        );
    }
    let granule_end = demux.vorbis(0).and_then(|info| info.end).expect("end");
    assert!((granule_end - end).abs() <= slack, "{name}");
    let past = demux
        .seek(SeekTo {
            track: 0,
            time: granule_end,
            end: None,
        })
        .expect("seek past end");
    assert_eq!(past.landing, Landing::End, "{name}");
    assert!(demux.read_packet().expect("read").is_none());
}

#[test]
fn a_chained_vorbis_file_returns_its_first_link() {
    let Some(root) = environment("a_chained_vorbis_file_returns_its_first_link") else {
        return;
    };
    let dir = scratch("vorbis-chain");
    let Some(source) = wav(&root, &dir, "chain", "4", &[]) else {
        eprintln!("skipped: no m4a");
        return;
    };
    let first = dir.join("first.ogg");
    let second = dir.join("second.ogg");
    let src = source.to_str().expect("utf8");
    ffmpeg(&[
        "-i",
        src,
        "-t",
        "3",
        "-c:a",
        "vorbis",
        "-strict",
        "-2",
        first.to_str().expect("p"),
    ]);
    ffmpeg(&[
        "-i",
        src,
        "-t",
        "1",
        "-ar",
        "22050",
        "-c:a",
        "vorbis",
        "-strict",
        "-2",
        second.to_str().expect("p"),
    ]);
    let mut chained = std::fs::read(&first).expect("first");
    chained.extend(std::fs::read(&second).expect("second"));
    let chain = dir.join("chain.ogg");
    std::fs::write(&chain, &chained).expect("chain");

    let mut demux = open(&chain);
    assert!(demux.chained());
    assert_eq!(
        demux.link_end(),
        Some(std::fs::metadata(&first).expect("meta").len())
    );
    let packets = read_all(&mut demux);
    let alone = read_all(&mut open(&first));
    assert_eq!(packets.len(), alone.len());
    for (mine, theirs) in packets.iter().zip(&alone) {
        assert_eq!(mine.bytes(), theirs.bytes());
        assert_eq!(mine.timestamp(), theirs.timestamp());
    }
    assert_eq!(
        u64::try_from(demux.streams()[0].edits()[0].duration()).expect("edit"),
        ffmpeg_decoded_samples(&first)
    );
    let _ = std::fs::remove_dir_all(&dir);
}
