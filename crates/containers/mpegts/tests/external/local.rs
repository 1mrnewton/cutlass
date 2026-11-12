//! Real recordings from `../local-files`: phone and camera movies and
//! downloads, remuxed into transport streams by this crate, then read back
//! here and by ffprobe, checked for structure, and seeked. Nothing from
//! that folder is committed. `CUTLASS_LOCAL_FILES` names the folder when
//! the checkout is not beside it; `TS_LOCAL_FILES` sets how many files are
//! used (default 5), smallest first up to 150 MB.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use cutlass_codec::{CodecId, StreamCodec};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration};
use cutlass_isobmff::IsoBmffFile;
use cutlass_mpegts::TsMux;
use cutlass_mpegts::packet::PacketSize;

use crate::demux_oracle::{check_seeks, compare};
use crate::mux_oracle::{check_structure, structure};
use crate::support::{ffmpeg_present, local_root, ours, scratch};

const CARRIED: [CodecId; 8] = [
    CodecId::H264,
    CodecId::Hevc,
    CodecId::Mpeg2,
    CodecId::Aac,
    CodecId::Mp3,
    CodecId::Ac3,
    CodecId::Eac3,
    CodecId::Opus,
];

/// How far each track's decode times must move earlier so that no sample
/// shows before it decodes. An MP4 with negative composition offsets (as
/// macOS screen recordings have) gives samples a PTS before their DTS;
/// FFmpeg's MOV demuxer moves its DTS the same way.
fn decode_shifts(source: &Path) -> Option<Vec<(u32, i64)>> {
    let file = File::open(source).ok()?;
    let mut demux = IsoBmffFile::open(BufReader::new(file)).ok()?;
    let mut shifts: Vec<(u32, i64)> = demux
        .streams()
        .iter()
        .map(|stream| (stream.id().get(), 0))
        .collect();
    while let Some(packet) = demux.read_packet().ok()? {
        let time = packet.timestamp();
        if let (Some(dts), Some(pts)) = (time.decode(), time.presentation())
            && let Some(slot) = shifts.iter_mut().find(|(id, _)| *id == packet.stream())
        {
            slot.1 = slot.1.max(dts - pts);
        }
    }
    Some(shifts)
}

/// Remuxes the tracks the muxer carries; `None` when there is no video
/// among them or the source does not open.
fn remux_carried(source: &Path, out: &Path) -> Option<usize> {
    let shifts = decode_shifts(source)?;
    let file = File::open(source).ok()?;
    let mut demux = IsoBmffFile::open(BufReader::new(file)).ok()?;
    let mut mux = TsMux::new(BufWriter::new(File::create(out).expect("create")));
    let mut map = Vec::new();
    let mut video = false;
    for stream in demux.streams() {
        let StreamCodec::Known(codec) = stream.params().codec() else {
            continue;
        };
        if !CARRIED.contains(codec) {
            continue;
        }
        video |= stream.params().size().is_some();
        let track = mux
            .add_stream(NewStream {
                params: stream.params().clone(),
                timescale: stream.timescale(),
                language: stream.language().cloned(),
            })
            .ok()?;
        map.push((stream.id().get(), track));
    }
    if !video {
        return None;
    }
    let mut count = 0;
    while let Some(packet) = demux.read_packet().ok()? {
        let Some((_, track)) = map.iter().find(|(id, _)| *id == packet.stream()) else {
            continue;
        };
        let shift = shifts
            .iter()
            .find(|(id, _)| *id == packet.stream())
            .map_or(0, |slot| slot.1);
        let time = packet.timestamp();
        let time = cutlass_codec::Timestamp::try_new(
            time.decode().map(|dts| dts - shift),
            time.presentation(),
            time.duration(),
        )
        .ok()?;
        let packet = cutlass_codec::Packet::try_new(
            track.get(),
            packet.bytes().to_vec(),
            time,
            packet.timescale(),
            packet.keyframe(),
        )
        .ok()?;
        mux.write_packet(&packet).expect("write");
        count += 1;
    }
    let durations: [TrackDuration; 0] = [];
    mux.finish(&durations).expect("finish");
    Some(count)
}

fn candidates(root: &Path) -> Vec<PathBuf> {
    let limit: usize = std::env::var("TS_LOCAL_FILES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);
    let mut files: Vec<(u64, PathBuf)> = std::fs::read_dir(root)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    let name = path.to_string_lossy().to_lowercase();
                    name.ends_with(".mp4") || name.ends_with(".mov")
                })
                .filter_map(|path| {
                    let size = path.metadata().ok()?.len();
                    (size <= 150_000_000).then_some((size, path))
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
        .into_iter()
        .map(|(_, path)| path)
        .take(limit)
        .collect()
}

#[test]
fn local_movies_remux_read_back_and_seek() {
    if !ffmpeg_present("local_movies_remux_read_back_and_seek") {
        return;
    }
    let Some(root) = local_root() else {
        eprintln!("skipped local_movies_remux_read_back_and_seek: ../local-files is missing");
        return;
    };
    let dir = scratch("local");
    let mut failures = Vec::new();
    let mut done = 0;
    for (index, source) in candidates(&root).iter().enumerate() {
        let out = dir.join(format!("local{index}.ts"));
        let Some(written) = remux_carried(source, &out) else {
            eprintln!("skipped {}: no carried video", source.display());
            continue;
        };
        let mut problems = compare(&out);
        problems.extend(check_seeks(&out));
        problems.extend(check_structure(&structure(
            &std::fs::read(&out).unwrap(),
            PacketSize::Ts,
        )));
        let (_, read) = ours(&out);
        let total: usize = read.iter().map(|(_, packets)| packets.len()).sum();
        if total != written {
            problems.push(format!("read back {total} of {written} packets"));
        }
        eprintln!(
            "{}: {written} packets, {} problems",
            source.file_name().unwrap().to_string_lossy(),
            problems.len()
        );
        if !problems.is_empty() {
            failures.push(format!(
                "{}:\n  {}",
                source.display(),
                problems.join("\n  ")
            ));
        }
        std::fs::remove_file(&out).ok();
        done += 1;
    }
    std::fs::remove_dir_all(&dir).ok();
    eprintln!("{done} local movies checked");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
