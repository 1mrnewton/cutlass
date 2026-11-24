//! The writer's layouts: long durations, fast start, the recoverable file
//! and its truncations, fragment shape, chunking, and the edit and time
//! fields. FFmpeg is an oracle where it is installed; a missing tool skips
//! that check and says so.

use crate::{
    CodecId, Demux, Edit, Error, FileSource, IsoBmff, IsoBmffMux, MovieFile, Mp4Mux, Mp4Options,
    Mux, MuxOptions, Packet, SliceSource, StreamParams, Timescale,
};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

const VIDEO_SCALE: u32 = 60_000;
const FRAME: u32 = 1_001;
const AUDIO_SCALE: u32 = 44_100;
const AUDIO_FRAME: u32 = 1_024;

fn video() -> StreamParams {
    let mut params = StreamParams::video(CodecId::H264, Timescale(VIDEO_SCALE), 16, 8);
    params.extradata = vec![1, 66, 0, 10];
    params
}

fn audio() -> StreamParams {
    let mut params = StreamParams::audio(
        CodecId::Aac,
        Timescale(AUDIO_SCALE),
        AUDIO_SCALE,
        2,
        16,
        Default::default(),
    );
    params.extradata = vec![0x12, 0x10];
    params
}

/// A few bytes that differ from packet to packet and track to track.
fn payload(track: u32, number: u64) -> Vec<u8> {
    let len = 1 + usize::try_from(number * 7 % 5).unwrap();
    let seed = u8::try_from(number % 251).unwrap();
    (0..len)
        .map(|at| {
            seed.wrapping_mul(31)
                .wrapping_add(u8::try_from(at).unwrap())
                ^ (track as u8)
        })
        .collect()
}

/// Track 1 is 59.94 fps video with a keyframe every `gop` frames and
/// reordered frames between; track 2 is 44.1 kHz sound in 1024-sample
/// packets. Packets interleave by time, as export writes them.
fn movie(seconds: u64, gop: u64) -> Vec<Packet> {
    let frames = seconds * u64::from(VIDEO_SCALE) / u64::from(FRAME);
    let sound = seconds * u64::from(AUDIO_SCALE) / u64::from(AUDIO_FRAME);
    let mut out = Vec::new();
    let mut heard = 0u64;
    for frame in 0..frames {
        let dts = i64::try_from(frame).unwrap() * i64::from(FRAME);
        // Decode order I P B B P B B: a P frame is shown three frames on.
        let place = frame % gop;
        let cts = if place == 0 {
            1
        } else if place % 3 == 1 {
            3
        } else {
            0
        };
        out.push(Packet {
            track_id: 1,
            payload: payload(1, frame),
            dts,
            pts: dts + cts * i64::from(FRAME),
            duration: FRAME,
            timescale: Timescale(VIDEO_SCALE),
            keyframe: place == 0,
        });
        // Sound up to the end of this frame.
        while heard < sound
            && heard * u64::from(AUDIO_FRAME) * u64::from(VIDEO_SCALE)
                < (frame + 1) * u64::from(FRAME) * u64::from(AUDIO_SCALE)
        {
            let dts = i64::try_from(heard).unwrap() * i64::from(AUDIO_FRAME);
            out.push(Packet {
                track_id: 2,
                payload: payload(2, heard),
                dts,
                pts: dts,
                duration: AUDIO_FRAME,
                timescale: Timescale(AUDIO_SCALE),
                keyframe: true,
            });
            heard += 1;
        }
    }
    out
}

fn start<W: std::io::Write + std::io::Seek>(mux: &mut IsoBmffMux<W>) {
    assert_eq!(mux.add_stream(&video()).unwrap(), 1);
    assert_eq!(mux.add_stream(&audio()).unwrap(), 2);
}

fn write(options: MuxOptions, packets: &[Packet]) -> Vec<u8> {
    let mut mux = IsoBmffMux::with_reader(Cursor::new(Vec::new()), options);
    start(&mut mux);
    for packet in packets {
        mux.write_packet(packet).unwrap();
    }
    mux.finish_into().unwrap().into_inner()
}

/// What is on the output when the process dies before `finish`.
fn write_unfinished(options: MuxOptions, packets: &[Packet]) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut mux = IsoBmffMux::with_reader(&mut cursor, options);
        start(&mut mux);
        for packet in packets {
            mux.write_packet(packet).unwrap();
        }
    }
    cursor.into_inner()
}

fn by_track(mut packets: Vec<Packet>) -> Vec<Packet> {
    packets.sort_by_key(|packet| (packet.track_id, packet.dts));
    packets
}

fn read_all(bytes: &[u8]) -> Vec<Packet> {
    let mut source = SliceSource::new(bytes);
    let mut movie = IsoBmff::open(&mut source).expect("open");
    let mut out = Vec::new();
    while let Some(packet) = movie.read_packet(&mut source).expect("read") {
        out.push(packet);
    }
    out
}

/// Top-level boxes that lie whole inside `bytes`: kind, start, end.
fn top_level(bytes: &[u8]) -> Vec<([u8; 4], usize, usize)> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + 8 <= bytes.len() {
        let size32 = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap());
        let kind: [u8; 4] = bytes[at + 4..at + 8].try_into().unwrap();
        let size = match size32 {
            0 => bytes.len() - at,
            1 => {
                if at + 16 > bytes.len() {
                    break;
                }
                usize::try_from(u64::from_be_bytes(
                    bytes[at + 8..at + 16].try_into().unwrap(),
                ))
                .unwrap()
            }
            size => size as usize,
        };
        if size < 8 || at + size > bytes.len() {
            break;
        }
        out.push((kind, at, at + size));
        at += size;
    }
    out
}

fn kinds(bytes: &[u8]) -> Vec<String> {
    top_level(bytes)
        .iter()
        .map(|(kind, _, _)| String::from_utf8_lossy(kind).into_owned())
        .collect()
}

/// The payloads of every box of `kind` directly inside `parent`.
fn children<'a>(parent: &'a [u8], kind: &[u8; 4]) -> Vec<&'a [u8]> {
    top_level(parent)
        .into_iter()
        .filter(|(found, _, _)| found == kind)
        .map(|(_, start, end)| {
            let header = if parent[start..start + 4] == [0, 0, 0, 1] {
                16
            } else {
                8
            };
            &parent[start + header..end]
        })
        .collect()
}

/// The payload of the first box at `path`, each step inside the last.
fn find<'a>(bytes: &'a [u8], path: &[&[u8; 4]]) -> &'a [u8] {
    let mut inside = bytes;
    for kind in path {
        inside = children(inside, kind)
            .first()
            .copied()
            .unwrap_or_else(|| panic!("no {}", String::from_utf8_lossy(*kind)));
    }
    inside
}

fn be32(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn be64(bytes: &[u8], at: usize) -> u64 {
    u64::from_be_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// Version, timescale and duration of an `mvhd` or `mdhd` payload.
fn timed(payload: &[u8]) -> (u8, u32, u64) {
    if payload[0] == 1 {
        (1, be32(payload, 20), be64(payload, 24))
    } else {
        (0, be32(payload, 12), u64::from(be32(payload, 16)))
    }
}

/// Version and duration of a `tkhd` payload.
fn tkhd(payload: &[u8]) -> (u8, u64) {
    if payload[0] == 1 {
        (1, be64(payload, 28))
    } else {
        (0, u64::from(be32(payload, 20)))
    }
}

/// Each `trak`'s payload, in track order.
fn traks(bytes: &[u8]) -> Vec<&[u8]> {
    children(find(bytes, &[b"moov"]), b"trak")
}

fn durations(bytes: &[u8]) -> Vec<u64> {
    let mut source = SliceSource::new(bytes);
    let movie = IsoBmff::open(&mut source).expect("open");
    movie
        .streams()
        .iter()
        .map(|stream| stream.params.duration_ticks)
        .collect()
}

#[test]
fn a_long_ntsc_movie_with_441_sound_keeps_exact_durations() {
    // 500 s passes the 487 s where the least common multiple of 60000 and
    // 44100, 8 820 000 ticks a second, no longer fits 32 bits.
    let packets = movie(500, 60);
    let frames = packets.iter().filter(|p| p.track_id == 1).count() as u64;
    let sound = packets.iter().filter(|p| p.track_id == 2).count() as u64;
    assert_eq!(frames, 29_970);
    assert_eq!(sound, 21_533);
    let video_ticks = frames * u64::from(FRAME);
    let audio_ticks = sound * u64::from(AUDIO_FRAME);

    let bytes = write(MuxOptions::default(), &packets);
    // The movie timescale is the video's, so nothing clamps and the
    // headers stay version 0.
    let (version, scale, duration) = timed(find(&bytes, &[b"moov", b"mvhd"]));
    assert_eq!((version, scale), (0, VIDEO_SCALE));
    assert_eq!(duration, video_ticks);
    let traks = traks(&bytes);
    assert_eq!(tkhd(find(traks[0], &[b"tkhd"])), (0, video_ticks));
    // 22 049 792 samples at 44.1 kHz, rounded up to a 60000th.
    let audio_in_movie = (audio_ticks * 60_000).div_ceil(44_100);
    assert_eq!(tkhd(find(traks[1], &[b"tkhd"])), (0, audio_in_movie));
    assert_eq!(
        timed(find(traks[0], &[b"mdia", b"mdhd"])),
        (0, VIDEO_SCALE, video_ticks)
    );
    assert_eq!(
        timed(find(traks[1], &[b"mdia", b"mdhd"])),
        (0, AUDIO_SCALE, audio_ticks)
    );
    assert_eq!(durations(&bytes), [video_ticks, audio_ticks]);
    assert_eq!(by_track(read_all(&bytes)), by_track(packets));
}

#[test]
fn durations_past_32_bits_are_written_in_version_1_boxes() {
    // The old choice of timescale, forced: 500 s is 4 410 000 000 ticks.
    let packets = movie(500, 60);
    let frames = packets.iter().filter(|p| p.track_id == 1).count() as u64;
    let sound = packets.iter().filter(|p| p.track_id == 2).count() as u64;
    let bytes = write(
        MuxOptions {
            movie_timescale: Some(8_820_000),
            ..MuxOptions::default()
        },
        &packets,
    );
    let video_in_movie = frames * u64::from(FRAME) * 147;
    let audio_in_movie = sound * u64::from(AUDIO_FRAME) * 200;
    assert!(video_in_movie > u64::from(u32::MAX));
    assert_eq!(
        timed(find(&bytes, &[b"moov", b"mvhd"])),
        (1, 8_820_000, video_in_movie.max(audio_in_movie))
    );
    let traks = traks(&bytes);
    assert_eq!(tkhd(find(traks[0], &[b"tkhd"])), (1, video_in_movie));
    assert_eq!(tkhd(find(traks[1], &[b"tkhd"])), (1, audio_in_movie));
    // The media durations still fit, so `mdhd` stays version 0.
    assert_eq!(find(traks[0], &[b"mdia", b"mdhd"])[0], 0);
    assert_eq!(
        durations(&bytes),
        [frames * u64::from(FRAME), sound * u64::from(AUDIO_FRAME)]
    );
    let mut source = SliceSource::new(&bytes);
    let movie = IsoBmff::open(&mut source).unwrap();
    assert_eq!(movie.movie_timescale(), Timescale(8_820_000));
}

#[test]
fn a_track_longer_than_32_bits_of_its_own_ticks_reads_back_exactly() {
    // 80 000 one-second samples at 60000 ticks a second: 4.8e9 ticks, past
    // 32 bits in the media and in the default movie timescale alike.
    let mut mux = IsoBmffMux::new(Cursor::new(Vec::new()), MuxOptions::default());
    let id = mux.add_stream(&video()).unwrap();
    let count = 80_000u64;
    for number in 0..count {
        let dts = i64::try_from(number).unwrap() * 60_000;
        mux.write_packet(&Packet {
            track_id: id,
            payload: payload(1, number),
            dts,
            pts: dts,
            duration: 60_000,
            timescale: Timescale(VIDEO_SCALE),
            keyframe: true,
        })
        .unwrap();
    }
    // An edit that starts past 32 bits of media time, for 2^32 + 5 ticks.
    let edit = Edit {
        segment_duration: (1 << 32) + 5,
        media_time: Some((1 << 31) + 7),
        media_rate: 0x0001_0000,
    };
    mux.set_edits(id, vec![edit]).unwrap();
    let bytes = mux.finish_into().unwrap().into_inner();
    let ticks = count * 60_000;
    assert!(ticks > u64::from(u32::MAX));
    assert_eq!(
        timed(find(&bytes, &[b"moov", b"mvhd"])),
        (1, VIDEO_SCALE, edit.segment_duration)
    );
    let trak = traks(&bytes)[0];
    assert_eq!(tkhd(find(trak, &[b"tkhd"])), (1, edit.segment_duration));
    assert_eq!(
        timed(find(trak, &[b"mdia", b"mdhd"])),
        (1, VIDEO_SCALE, ticks)
    );
    let elst = find(trak, &[b"edts", b"elst"]);
    assert_eq!(elst[0], 1);
    assert_eq!(be64(elst, 8), edit.segment_duration);
    assert_eq!(be64(elst, 16), (1 << 31) + 7);

    let mut source = SliceSource::new(&bytes);
    let mut movie = IsoBmff::open(&mut source).unwrap();
    assert_eq!(movie.streams()[0].params.duration_ticks, ticks);
    assert_eq!(movie.streams()[0].edits, [edit]);
    let mut last = None;
    while let Some(packet) = movie.read_packet(&mut source).unwrap() {
        last = Some(packet);
    }
    assert_eq!(last.unwrap().dts, i64::try_from(ticks).unwrap() - 60_000);
}

#[test]
fn the_movie_timescale_is_exact_when_small_and_bounded_when_not() {
    let scale_of = |scales: &[(bool, u32)]| {
        let mut mux = IsoBmffMux::new(Cursor::new(Vec::new()), MuxOptions::default());
        for (picture, scale) in scales {
            let mut params = if *picture { video() } else { audio() };
            params.timescale = Timescale(*scale);
            mux.add_stream(&params).unwrap();
        }
        let bytes = mux.finish_into().unwrap().into_inner();
        timed(find(&bytes, &[b"moov", b"mvhd"])).1
    };
    // The least common multiple while it is at most 1 000 000.
    assert_eq!(scale_of(&[(true, 30), (false, 48_000)]), 48_000);
    assert_eq!(scale_of(&[(true, 30_000), (false, 48_000)]), 240_000);
    assert_eq!(scale_of(&[(true, 24_000), (false, 44_100)]), 24_000);
    // Past that, the video's own timescale.
    assert_eq!(scale_of(&[(true, 60_000), (false, 44_100)]), 60_000);
    assert_eq!(scale_of(&[(false, 44_100), (true, 30_000)]), 30_000);
    // Sound only, or a video timescale that is itself too large: 1000.
    assert_eq!(scale_of(&[(false, 999_983), (false, 44_100)]), 1000);
    assert_eq!(scale_of(&[(true, 90_000_000), (false, 44_100)]), 1000);
    assert_eq!(scale_of(&[(false, 44_100)]), 44_100);
}

#[test]
fn fast_start_puts_the_index_first_and_keeps_every_packet() {
    let packets = movie(4, 30);
    let last = write(MuxOptions::default(), &packets);
    assert_eq!(kinds(&last), ["ftyp", "mdat", "moov"]);
    let first = write(
        MuxOptions {
            moov_at_end: false,
            ..MuxOptions::default()
        },
        &packets,
    );
    assert_eq!(kinds(&first), ["ftyp", "moov", "mdat"]);
    assert_eq!(first.len(), last.len());
    assert_eq!(read_all(&first), read_all(&last));
    assert_eq!(by_track(read_all(&first)), by_track(packets.clone()));
    // The payload moved by the size of `moov`, a byte for a byte.
    let moov = top_level(&first)[1];
    let shift = moov.2 - moov.1;
    let mdat = top_level(&last)[1];
    assert_eq!(first[mdat.1 + shift..], last[mdat.1..mdat.2]);

    // An output that cannot be read back holds the payload instead, and
    // gives the same file.
    let mut mux = IsoBmffMux::new(
        Cursor::new(Vec::new()),
        MuxOptions {
            moov_at_end: false,
            ..MuxOptions::default()
        },
    );
    start(&mut mux);
    for packet in &packets {
        mux.write_packet(packet).unwrap();
    }
    let held = mux.finish_into().unwrap().into_inner();
    assert_eq!(kinds(&held), ["ftyp", "moov", "mdat"]);
    assert_eq!(read_all(&held), read_all(&first));
}

#[test]
fn fast_start_moves_a_payload_larger_than_one_block() {
    // Three samples of 3 MiB: the payload is moved in more than one block
    // of 4 MiB, the last of them partial.
    let mut mux = IsoBmffMux::with_reader(
        Cursor::new(Vec::new()),
        MuxOptions {
            moov_at_end: false,
            ..MuxOptions::default()
        },
    );
    let id = mux.add_stream(&video()).unwrap();
    let mut wrote = Vec::new();
    for number in 0..3u32 {
        let bytes: Vec<u8> = (0..3 * 1024 * 1024u32)
            .map(|at| (at.wrapping_mul(2_654_435_761).wrapping_add(number) >> 13) as u8)
            .collect();
        let packet = Packet {
            track_id: id,
            payload: bytes,
            dts: i64::from(number * FRAME),
            pts: i64::from(number * FRAME),
            duration: FRAME,
            timescale: Timescale(VIDEO_SCALE),
            keyframe: true,
        };
        mux.write_packet(&packet).unwrap();
        wrote.push(packet);
    }
    let bytes = mux.finish_into().unwrap().into_inner();
    assert_eq!(kinds(&bytes), ["ftyp", "moov", "mdat"]);
    assert_eq!(read_all(&bytes), wrote);
}

#[test]
fn a_recoverable_file_finishes_as_a_plain_movie() {
    let packets = movie(5, 30);
    for moov_at_end in [true, false] {
        let bytes = write(
            MuxOptions {
                recoverable: true,
                moov_at_end,
                ..MuxOptions::default()
            },
            &packets,
        );
        let expected: &[&str] = if moov_at_end {
            &["ftyp", "mdat", "moov"]
        } else {
            &["ftyp", "moov", "mdat"]
        };
        assert_eq!(kinds(&bytes), expected);
        // No movie-extends box: a reader sees a movie without fragments.
        assert!(children(find(&bytes, &[b"moov"]), b"mvex").is_empty());
        assert_eq!(by_track(read_all(&bytes)), by_track(packets.clone()));
        assert_eq!(
            durations(&bytes),
            durations(&write(MuxOptions::default(), &packets))
        );
    }
}

/// The fragments that lie whole in an unfinished recoverable file: for each
/// `moof` followed by its whole `mdat`, the samples its runs name.
fn whole_fragments(bytes: &[u8]) -> Vec<usize> {
    let boxes = top_level(bytes);
    let mut out = Vec::new();
    for pair in boxes.windows(2) {
        if &pair[0].0 == b"moof" && &pair[1].0 == b"mdat" {
            let moof = &bytes[pair[0].1 + 8..pair[0].2];
            let mut samples = 0;
            let mut payload = 0u64;
            for traf in children(moof, b"traf") {
                let tfhd = find(traf, &[b"tfhd"]);
                let flags = be32(tfhd, 0) & 0x00ff_ffff;
                let mut at = 8;
                if flags & 0x08 != 0 {
                    at += 4;
                }
                let default_size = (flags & 0x10 != 0).then(|| be32(tfhd, at));
                for trun in children(traf, b"trun") {
                    let flags = be32(trun, 0) & 0x00ff_ffff;
                    let count = be32(trun, 4) as usize;
                    samples += count;
                    let mut at = 12 + if flags & 0x004 != 0 { 4 } else { 0 };
                    for _ in 0..count {
                        if flags & 0x100 != 0 {
                            at += 4;
                        }
                        if flags & 0x200 != 0 {
                            payload += u64::from(be32(trun, at));
                            at += 4;
                        } else {
                            payload += u64::from(default_size.unwrap());
                        }
                        if flags & 0x400 != 0 {
                            at += 4;
                        }
                        if flags & 0x800 != 0 {
                            at += 4;
                        }
                    }
                }
            }
            // The runs account for every byte of the `mdat`.
            assert_eq!(payload, (pair[1].2 - pair[1].1 - 8) as u64);
            out.push(samples);
        }
    }
    out
}

#[test]
fn an_unfinished_recoverable_file_cut_anywhere_keeps_its_whole_fragments() {
    let packets = movie(6, 30);
    let options = MuxOptions {
        recoverable: true,
        fragment_millis: 500,
        ..MuxOptions::default()
    };
    let unfinished = write_unfinished(options, &packets);
    assert_eq!(kinds(&unfinished)[..4], ["ftyp", "free", "moov", "moof"]);
    let fragments = whole_fragments(&unfinished);
    // Half-second keyframes: eleven fragments are down; the twelfth was
    // still in memory when the process died.
    assert_eq!(fragments.len(), 11);
    let boxes = top_level(&unfinished);
    let mut ends: Vec<usize> = boxes
        .iter()
        .filter(|(kind, _, _)| kind == b"mdat")
        .map(|(_, _, end)| *end)
        .collect();
    let moov_end = boxes[2].2;
    ends.insert(0, moov_end);

    // Every cut from the end of the first `moov` on: the fragments that end
    // at or before the cut are whole, and their samples are all there.
    for cut in (moov_end..=unfinished.len()).step_by(7).chain(ends.clone()) {
        let whole = ends.iter().filter(|end| **end <= cut).count() - 1;
        let found = whole_fragments(&unfinished[..cut]);
        assert_eq!(found, fragments[..whole], "cut at {cut}");
    }

    // Cut on a fragment boundary, this crate's demuxer reads every packet
    // of the fragments before it, as written.
    let mut expected = 0;
    for (index, end) in ends.iter().enumerate() {
        if index > 0 {
            expected += fragments[index - 1];
        }
        let read = by_track(read_all(&unfinished[..*end]));
        assert_eq!(read.len(), expected, "cut at {end}");
        let mut wanted: Vec<Packet> = Vec::new();
        for track in [1, 2] {
            let have = read.iter().filter(|p| p.track_id == track).count();
            wanted.extend(
                packets
                    .iter()
                    .filter(|p| p.track_id == track)
                    .take(have)
                    .cloned(),
            );
        }
        assert_eq!(read, by_track(wanted), "cut at {end}");
    }
}

#[test]
fn fragments_are_cut_at_video_keyframes_with_one_traf_a_track() {
    // A keyframe every 30 frames, about half a second, and a target of
    // 400 ms: each fragment is one group of pictures.
    let packets = movie(3, 30);
    let bytes = write(
        MuxOptions {
            fragmented: true,
            fragment_millis: 400,
            ..MuxOptions::default()
        },
        &packets,
    );
    let boxes = top_level(&bytes);
    let names = kinds(&bytes);
    assert_eq!(names[..2], ["ftyp", "moov"]);
    assert_eq!(names.last().unwrap(), "mfra");
    let moofs: Vec<_> = boxes
        .iter()
        .filter(|(kind, _, _)| kind == b"moof")
        .collect();
    let frames = packets.iter().filter(|p| p.track_id == 1).count();
    assert_eq!(moofs.len(), frames.div_ceil(30));
    for (index, (_, start, end)) in moofs.iter().enumerate() {
        let moof = &bytes[start + 8..*end];
        assert_eq!(be32(find(moof, &[b"mfhd"]), 4) as usize, index + 1);
        let trafs = children(moof, b"traf");
        assert_eq!(trafs.len(), 2, "one traf for each track");
        assert_eq!(be32(find(trafs[0], &[b"tfhd"]), 4), 1);
        assert_eq!(be32(find(trafs[1], &[b"tfhd"]), 4), 2);
        // The video starts on its keyframe, at the decode time `tfdt` says.
        assert_eq!(
            be64(find(trafs[0], &[b"tfdt"]), 4),
            (index as u64) * 30 * u64::from(FRAME)
        );
        let video_samples: u32 = children(trafs[0], b"trun")
            .iter()
            .map(|trun| be32(trun, 4))
            .sum();
        assert_eq!(video_samples as usize, 30.min(frames - index * 30));
    }
    // Each fragment is `sidx`, `moof`, `mdat`, and the index names the
    // pair's size, its video duration, and that it opens on a keyframe.
    for window in boxes.windows(3) {
        if &window[1].0 != b"moof" {
            continue;
        }
        assert_eq!(&window[0].0, b"sidx");
        assert_eq!(&window[2].0, b"mdat");
        let sidx = &bytes[window[0].1 + 8..window[0].2];
        assert_eq!(sidx[0], 1);
        assert_eq!(be32(sidx, 4), 1, "the video track");
        assert_eq!(be32(sidx, 8), VIDEO_SCALE);
        assert_eq!(be64(sidx, 20), 0, "the moof follows at once");
        assert_eq!(be32(sidx, 28), 1, "one reference");
        assert_eq!(be32(sidx, 32) as usize, window[2].2 - window[1].1);
        assert_eq!(be32(sidx, 40), 0x9000_0000);
    }
    // `tfra` for the video: one row a fragment, at the keyframe's
    // presentation time, pointing at that fragment's `moof`.
    let mfra = find(&bytes, &[b"mfra"]);
    let tfra = children(mfra, b"tfra")[0];
    assert_eq!((tfra[0], be32(tfra, 4), be32(tfra, 8)), (1, 1, 0x0F));
    assert_eq!(be32(tfra, 12) as usize, moofs.len());
    for (index, (_, start, _)) in moofs.iter().enumerate() {
        let row = &tfra[16 + index * 25..];
        let key_dts = (index as u64) * 30 * u64::from(FRAME);
        assert_eq!(
            be64(row, 0),
            key_dts + u64::from(FRAME),
            "presentation time"
        );
        assert_eq!(be64(row, 8) as usize, *start);
        assert_eq!((row[16], be32(row, 17), be32(row, 21)), (1, 1, 1));
    }
    let mfro = find(mfra, &[b"mfro"]);
    assert_eq!(be32(mfro, 4) as usize, mfra.len() + 8);
    assert_eq!(by_track(read_all(&bytes)), by_track(packets));
}

#[test]
fn sound_alone_is_cut_by_duration_and_a_byte_bound_cuts_early() {
    let sound: Vec<Packet> = movie(4, 30)
        .into_iter()
        .filter(|p| p.track_id == 2)
        .collect();
    let write_sound = |options: MuxOptions| {
        let mut mux = IsoBmffMux::new(Cursor::new(Vec::new()), options);
        let mut params = audio();
        params.track_id = 2;
        mux.add_stream(&params).unwrap();
        for packet in &sound {
            mux.write_packet(packet).unwrap();
        }
        mux.finish_into().unwrap().into_inner()
    };
    let bytes = write_sound(MuxOptions {
        fragmented: true,
        ..MuxOptions::default()
    });
    // 44 packets of 1024 samples pass one second; the 172 packets of four
    // seconds make four fragments.
    let counts = whole_fragments(&bytes);
    assert_eq!(counts, [44, 44, 44, 40]);
    assert_eq!(read_all(&bytes), sound);
    // Every sample is a sync sample: `tfra` lists the first of each
    // fragment, not all 172.
    let tfra = children(find(&bytes, &[b"mfra"]), b"tfra")[0];
    assert_eq!(be32(tfra, 12), 4);
    assert_eq!(be64(tfra, 16 + 25), 44 * 1024);

    let small = write_sound(MuxOptions {
        fragmented: true,
        fragment_bytes: 16,
        ..MuxOptions::default()
    });
    let counts = whole_fragments(&small);
    assert!(counts.len() > 20, "{counts:?}");
    for (kind, start, end) in top_level(&small) {
        if &kind == b"mdat" {
            assert!(end - start - 8 <= 16);
        }
    }
    assert_eq!(read_all(&small), sound);
}

#[test]
fn a_fragment_does_not_wait_forever_for_a_keyframe() {
    // One keyframe, then none: the fragment is cut after ten target
    // lengths, not held until finish.
    let packets: Vec<Packet> = movie(3, 100_000)
        .into_iter()
        .filter(|p| p.track_id == 1)
        .collect();
    let mut mux = IsoBmffMux::new(
        Cursor::new(Vec::new()),
        MuxOptions {
            fragmented: true,
            fragment_millis: 100,
            ..MuxOptions::default()
        },
    );
    mux.add_stream(&video()).unwrap();
    for packet in &packets {
        mux.write_packet(packet).unwrap();
    }
    let bytes = mux.finish_into().unwrap().into_inner();
    // 60 frames are 1.001 s, the first count past ten times 100 ms.
    assert_eq!(whole_fragments(&bytes), [60, 60, 59]);
    assert_eq!(read_all(&bytes), packets);

    // A recoverable file waits only twice the target, 12 frames here, so a
    // crash between far keyframes loses little.
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut mux = IsoBmffMux::new(
            &mut cursor,
            MuxOptions {
                recoverable: true,
                fragment_millis: 100,
                ..MuxOptions::default()
            },
        );
        mux.add_stream(&video()).unwrap();
        for packet in &packets {
            mux.write_packet(packet).unwrap();
        }
    }
    let unfinished = cursor.into_inner();
    let fragments = whole_fragments(&unfinished);
    assert_eq!(fragments.len(), 179 / 12);
    assert!(fragments.iter().all(|samples| *samples == 12));
    assert_eq!(read_all(&unfinished), packets[..fragments.len() * 12]);
}

fn chunk_count(trak: &[u8]) -> u32 {
    be32(find(trak, &[b"mdia", b"minf", b"stbl", b"stco"]), 4)
}

#[test]
fn samples_that_lie_together_share_a_chunk() {
    let packets = movie(4, 30);
    let frames = packets.iter().filter(|p| p.track_id == 1).count() as u32;

    // Sound alone: every packet follows the last, so the track is one chunk
    // and `stsc` has one row for all of it.
    let sound: Vec<Packet> = packets
        .iter()
        .filter(|p| p.track_id == 2)
        .cloned()
        .collect();
    let mut mux = IsoBmffMux::new(Cursor::new(Vec::new()), MuxOptions::default());
    let mut params = audio();
    params.track_id = 2;
    mux.add_stream(&params).unwrap();
    for packet in &sound {
        mux.write_packet(packet).unwrap();
    }
    let bytes = mux.finish_into().unwrap().into_inner();
    let stbl = find(traks(&bytes)[0], &[b"mdia", b"minf", b"stbl"]);
    assert_eq!(be32(find(stbl, &[b"stco"]), 4), 1);
    let stsc = find(stbl, &[b"stsc"]);
    assert_eq!((be32(stsc, 4), be32(stsc, 8), be32(stsc, 12)), (1, 1, 172));
    assert_eq!(read_all(&bytes), sound);

    // Interleaved, a chunk is a run of frames with no sound between them:
    // one frame, or two where the 43 sound packets a second leave a gap.
    let runs = (0..packets.len())
        .filter(|&at| packets[at].track_id == 1 && (at == 0 || packets[at - 1].track_id != 1))
        .count() as u32;
    assert!(runs > frames / 2 && runs < frames);
    let plain = write(MuxOptions::default(), &packets);
    assert_eq!(chunk_count(traks(&plain)[0]), runs);
    // A recoverable file lays each track's part of a fragment together:
    // four one-second fragments, four chunks a track.
    let grouped = write(
        MuxOptions {
            recoverable: true,
            ..MuxOptions::default()
        },
        &packets,
    );
    assert_eq!(chunk_count(traks(&grouped)[0]), 4);
    assert_eq!(chunk_count(traks(&grouped)[1]), 4);
    let moov = |bytes: &[u8]| find(bytes, &[b"moov"]).len();
    assert!(moov(&grouped) < moov(&plain) - 4 * (runs as usize - 4));
    assert_eq!(by_track(read_all(&grouped)), by_track(packets));
}

#[test]
fn a_chunk_stops_at_one_mebibyte() {
    let mut mux = IsoBmffMux::new(Cursor::new(Vec::new()), MuxOptions::default());
    let id = mux.add_stream(&video()).unwrap();
    let mut wrote = Vec::new();
    for number in 0..5u32 {
        let packet = Packet {
            track_id: id,
            payload: vec![number as u8; 400 * 1024],
            dts: i64::from(number * FRAME),
            pts: i64::from(number * FRAME),
            duration: FRAME,
            timescale: Timescale(VIDEO_SCALE),
            keyframe: true,
        };
        mux.write_packet(&packet).unwrap();
        wrote.push(packet);
    }
    let bytes = mux.finish_into().unwrap().into_inner();
    let stbl = find(traks(&bytes)[0], &[b"mdia", b"minf", b"stbl"]);
    // 400 KiB samples: two fit under 1 MiB, the third starts a chunk.
    assert_eq!(be32(find(stbl, &[b"stco"]), 4), 3);
    let stsc = find(stbl, &[b"stsc"]);
    assert_eq!(be32(stsc, 4), 2, "two rows: chunks of 2, then one of 1");
    assert_eq!((be32(stsc, 8), be32(stsc, 12)), (1, 2));
    assert_eq!((be32(stsc, 20), be32(stsc, 24)), (3, 1));
    assert_eq!(read_all(&bytes), wrote);
}

#[test]
fn the_writer_refuses_what_it_cannot_store() {
    // A composition offset that does not fit 32 bits, refused when written.
    let mut mux = IsoBmffMux::new(Cursor::new(Vec::new()), MuxOptions::default());
    let id = mux.add_stream(&video()).unwrap();
    let mut packet = movie(1, 30).remove(0);
    packet.track_id = id;
    packet.pts = i64::from(i32::MAX) + 1;
    assert_eq!(
        mux.write_packet(&packet),
        Err(Error::Limit {
            what: "composition offset",
            value: 1 << 31,
        })
    );
    // The muxer took nothing: the same packet with a sound time goes in.
    packet.pts = 0;
    mux.write_packet(&packet).unwrap();

    // A recoverable file with its index first has to be read back.
    let mut mux = IsoBmffMux::new(
        Cursor::new(Vec::new()),
        MuxOptions {
            recoverable: true,
            moov_at_end: false,
            ..MuxOptions::default()
        },
    );
    let id = mux.add_stream(&video()).unwrap();
    packet.track_id = id;
    assert!(matches!(
        mux.write_packet(&packet),
        Err(Error::Unsupported { .. })
    ));
    assert!(matches!(mux.finish(), Err(Error::Unsupported { .. })));

    // An empty recoverable or fast-start movie is still a movie.
    for moov_at_end in [true, false] {
        let bytes = write(
            MuxOptions {
                recoverable: true,
                moov_at_end,
                ..MuxOptions::default()
            },
            &[],
        );
        assert_eq!(durations(&bytes), [0, 0]);
        assert!(read_all(&bytes).is_empty());
    }
}

#[test]
fn the_creation_time_is_written_and_widens_the_headers_when_it_must() {
    let packets = movie(1, 30);
    let at = |unix: Option<u64>| {
        write(
            MuxOptions {
                creation_time: unix,
                ..MuxOptions::default()
            },
            &packets,
        )
    };
    let none = at(None);
    let mvhd = find(&none, &[b"moov", b"mvhd"]);
    assert_eq!((mvhd[0], be32(mvhd, 4), be32(mvhd, 8)), (0, 0, 0));

    // 2026-11-24 00:00:00 UTC, in seconds since 1904.
    let bytes = at(Some(1_795_478_400));
    let stamp = 1_795_478_400 + 2_082_844_800u32;
    let mvhd = find(&bytes, &[b"moov", b"mvhd"]);
    assert_eq!((mvhd[0], be32(mvhd, 4), be32(mvhd, 8)), (0, stamp, stamp));
    for trak in traks(&bytes) {
        let tkhd = find(trak, &[b"tkhd"]);
        assert_eq!((be32(tkhd, 4), be32(tkhd, 8)), (stamp, stamp));
        let mdhd = find(trak, &[b"mdia", b"mdhd"]);
        assert_eq!((be32(mdhd, 4), be32(mdhd, 8)), (stamp, stamp));
    }
    assert_eq!(by_track(read_all(&bytes)), by_track(packets.clone()));

    // 2040-02-06 06:28:16 UTC is the first second past 32 bits.
    let late = at(Some((1 << 32) - 2_082_844_800));
    let mvhd = find(&late, &[b"moov", b"mvhd"]);
    assert_eq!(
        (mvhd[0], be64(mvhd, 4), be64(mvhd, 12)),
        (1, 1 << 32, 1 << 32)
    );
    assert_eq!(durations(&late), durations(&none));
    assert_eq!(by_track(read_all(&late)), by_track(packets));
}

mod export_mux {
    //! The same through `Mp4Mux`, the muxer export uses.

    use super::{be32, children, find, kinds, timed, tkhd, traks};
    use crate::{FileBrand, MovieFile, Mp4Mux, Mp4Options};
    use cutlass_codec::{
        CodecId as Id, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp,
    };
    use cutlass_container::{Demux, IndexPlace, Mux, NewStream, TrackDuration, TrackId, Trim};
    use std::io::{Read, Seek, Write};

    pub(super) struct Wrote {
        pub packets: Vec<Packet>,
        pub index: IndexPlace,
    }

    /// Two seconds of 30 fps H.264 and 48 kHz AAC through `mux`.
    pub(super) fn two_seconds<W: Write + Seek>(mux: &mut Mp4Mux<W>, trim: Option<Trim>) -> Wrote {
        let picture = mux
            .add_stream(NewStream {
                params: StreamParams::video(
                    StreamCodec::Known(Id::H264),
                    FrameSize::try_new(64, 32).unwrap(),
                    Vec::new(),
                ),
                timescale: Timescale::new(30),
                language: None,
            })
            .unwrap();
        let sound = mux
            .add_stream(NewStream {
                params: StreamParams::try_audio(StreamCodec::Known(Id::Aac), 48_000, 2, Vec::new())
                    .unwrap(),
                timescale: Timescale::new(48_000),
                language: None,
            })
            .unwrap();
        mux.set_config(picture, &[1, 66, 0, 10]).unwrap();
        mux.set_config(sound, &[0x11, 0x90]).unwrap();
        if let Some(trim) = trim {
            mux.set_trim(sound, trim).unwrap();
        }
        let mut packets = Vec::new();
        let mut heard = 0i64;
        for frame in 0..60i64 {
            let time = Timestamp::try_new(Some(frame), Some(frame), Some(1)).unwrap();
            let bytes = vec![frame as u8; 40 + (frame as usize % 7)];
            packets.push(
                Packet::try_new(
                    picture.get(),
                    bytes,
                    time,
                    Timescale::new(30),
                    frame % 15 == 0,
                )
                .unwrap(),
            );
            while heard * 1024 * 30 < (frame + 1) * 48_000 {
                let at = heard * 1024;
                let time = Timestamp::try_new(Some(at), Some(at), Some(1024)).unwrap();
                let bytes = vec![heard as u8 ^ 0x5A; 9 + (heard as usize % 3)];
                packets.push(
                    Packet::try_new(sound.get(), bytes, time, Timescale::new(48_000), true)
                        .unwrap(),
                );
                heard += 1;
            }
        }
        for packet in &packets {
            mux.write_packet(packet).unwrap();
        }
        let done = mux
            .finish(&[
                TrackDuration {
                    track: picture,
                    timescale: Timescale::new(30),
                    ticks: 60,
                },
                TrackDuration {
                    track: sound,
                    timescale: Timescale::new(48_000),
                    ticks: heard * 1024,
                },
            ])
            .unwrap();
        Wrote {
            packets,
            index: done.index(),
        }
    }

    pub(super) fn read_back<R: Read + Seek + Send>(reader: R) -> Vec<(u32, i64, Vec<u8>, bool)> {
        let mut file = crate::IsoBmffFile::open(reader).unwrap();
        let mut out = Vec::new();
        while let Some(packet) = file.read_packet().unwrap() {
            out.push((
                packet.stream(),
                packet.timestamp().presentation_or_decode().unwrap(),
                packet.bytes().to_vec(),
                packet.keyframe(),
            ));
        }
        out.sort();
        out
    }

    fn expected(packets: &[Packet]) -> Vec<(u32, i64, Vec<u8>, bool)> {
        let mut out: Vec<_> = packets
            .iter()
            .map(|packet| {
                (
                    packet.stream(),
                    packet.timestamp().presentation_or_decode().unwrap(),
                    packet.bytes().to_vec(),
                    packet.keyframe(),
                )
            })
            .collect();
        out.sort();
        out
    }

    #[test]
    fn every_layout_holds_the_same_packets() {
        let mut plain = Mp4Mux::new(std::io::Cursor::new(Vec::new()));
        let wrote = two_seconds(&mut plain, None);
        assert_eq!(wrote.index, IndexPlace::End);
        let plain = plain.into_writer().unwrap().into_inner();
        assert_eq!(kinds(&plain), ["ftyp", "mdat", "moov"]);
        assert_eq!(
            read_back(std::io::Cursor::new(plain)),
            expected(&wrote.packets)
        );

        for (fast_start, recoverable) in
            [(true, true), (true, false), (false, true), (false, false)]
        {
            let options = Mp4Options {
                fast_start,
                recoverable,
                ..Mp4Options::default()
            };
            let mut mux = Mp4Mux::with_options(std::io::Cursor::new(Vec::new()), options);
            let wrote = two_seconds(&mut mux, None);
            let bytes = mux.into_writer().unwrap().into_inner();
            if fast_start {
                assert_eq!(wrote.index, IndexPlace::Start);
                assert_eq!(kinds(&bytes), ["ftyp", "moov", "mdat"]);
            } else {
                assert_eq!(wrote.index, IndexPlace::End);
                assert_eq!(kinds(&bytes), ["ftyp", "mdat", "moov"]);
            }
            assert!(children(find(&bytes, &[b"moov"]), b"mvex").is_empty());
            assert_eq!(
                read_back(std::io::Cursor::new(bytes)),
                expected(&wrote.packets),
                "fast start {fast_start}, recoverable {recoverable}"
            );
        }
    }

    #[test]
    fn a_movie_file_on_disk_matches_one_in_memory() {
        let options = Mp4Options {
            brand: FileBrand::Mov,
            creation_time: Some(1_795_478_400),
            ..Mp4Options::default()
        };
        let mut memory = Mp4Mux::with_options(std::io::Cursor::new(Vec::new()), options);
        two_seconds(&mut memory, None);
        let memory = memory.into_writer().unwrap().into_inner();

        let path =
            std::env::temp_dir().join(format!("isobmff-movie-file-{}.mov", std::process::id()));
        let mut mux = Mp4Mux::with_options(MovieFile::create(&path).unwrap(), options);
        two_seconds(&mut mux, None);
        let mut out = mux.into_writer().unwrap().into_inner();
        out.flush().unwrap();
        drop(out);
        let disk = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(kinds(&disk), ["ftyp", "moov", "mdat"]);
        assert!(disk == memory, "the file differs from the in-memory movie");
    }

    #[test]
    fn an_aac_track_gets_an_edit_past_its_priming() {
        // 94 packets of 1024: 96 256 samples, of which the first 1024 are
        // the encoder's delay and the last 256 padding.
        let trim = Trim::new(Some(1024), Some(256));
        let mut mux = Mp4Mux::new(std::io::Cursor::new(Vec::new()));
        let wrote = two_seconds(&mut mux, Some(trim));
        let bytes = mux.into_writer().unwrap().into_inner();
        let sound = traks(&bytes)[1];
        let elst = find(sound, &[b"edts", b"elst"]);
        // One version 0 edit: 94 976 samples from sample 1024, at rate 1.
        assert_eq!((elst[0], be32(elst, 4)), (0, 1));
        assert_eq!(timed(find(&bytes, &[b"moov", b"mvhd"])).1, 48_000);
        assert_eq!(be32(elst, 8), 96_256 - 1024 - 256);
        assert_eq!(be32(elst, 12), 1024);
        assert_eq!(be32(elst, 16), 0x0001_0000);
        // The track presents what the edit plays. The media ends where the
        // padding starts: the last sample is 256 ticks short, which is how
        // FFmpeg knows to stop there.
        assert_eq!(tkhd(find(sound, &[b"tkhd"])), (0, 94_976));
        assert_eq!(timed(find(sound, &[b"mdia", b"mdhd"])), (0, 48_000, 96_000));
        let stts = find(sound, &[b"mdia", b"minf", b"stbl", b"stts"]);
        assert_eq!(be32(stts, 4), 2);
        assert_eq!((be32(stts, 8), be32(stts, 12)), (93, 1024));
        assert_eq!((be32(stts, 16), be32(stts, 20)), (1, 768));
        // The picture track has no edit, and the packets are untouched.
        assert!(children(traks(&bytes)[0], b"edts").is_empty());
        let file = crate::IsoBmffFile::open(std::io::Cursor::new(bytes.clone())).unwrap();
        assert_eq!(file.streams()[1].id(), TrackId::new(2));
        assert_eq!(
            read_back(std::io::Cursor::new(bytes)),
            expected(&wrote.packets)
        );
    }
}

fn tool(name: &str) -> Option<PathBuf> {
    let output = Command::new("which").arg(name).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let path = PathBuf::from(String::from_utf8(output.stdout).ok()?.trim());
    path.is_file().then_some(path)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("isobmff-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `ffprobe`'s view of a file: each packet as stream, pts, dts, duration,
/// size and flags in stream ticks; each stream's duration in ticks; and the
/// container duration in seconds.
struct Probe {
    packets: Vec<String>,
    stream_durations: Vec<String>,
    format_duration: f64,
}

fn probe(ffprobe: &Path, path: &Path) -> Probe {
    let output = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-show_packets",
        ])
        .args(["-of", "flat=s=_"])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        output.status.success() && output.stderr.is_empty(),
        "ffprobe {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let mut packets: Vec<[String; 6]> = Vec::new();
    let mut stream_durations = Vec::new();
    let mut format_duration = 0.0;
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim_matches('"').to_owned();
        if let Some(rest) = key.strip_prefix("packets_packet_") {
            let (number, field) = rest.split_once('_').unwrap();
            let number: usize = number.parse().unwrap();
            if packets.len() <= number {
                packets.resize(number + 1, Default::default());
            }
            let slot = match field {
                "stream_index" => 0,
                "pts" => 1,
                "dts" => 2,
                "duration" => 3,
                "size" => 4,
                "flags" => 5,
                _ => continue,
            };
            // The second flag marks a packet the edit list discards.
            packets[number][slot] = value;
        } else if key.starts_with("streams_stream_") && key.ends_with("_duration_ts") {
            stream_durations.push(value);
        } else if key == "format_duration" {
            format_duration = value.parse().unwrap();
        }
    }
    let mut packets: Vec<String> = packets.iter().map(|fields| fields.join(" ")).collect();
    packets.sort();
    Probe {
        packets,
        stream_durations,
        format_duration,
    }
}

/// A file's streams and packets through this crate's demuxer.
fn demux(path: &Path) -> (Vec<crate::Stream>, Timescale, Vec<Packet>) {
    let mut source = FileSource::open(path).unwrap();
    let mut movie = IsoBmff::open(&mut source).unwrap();
    let mut packets = Vec::new();
    while let Some(packet) = movie.read_packet(&mut source).unwrap() {
        packets.push(packet);
    }
    (movie.streams().to_vec(), movie.movie_timescale(), packets)
}

/// A second and a half of 59.94 fps H.264 with B-frames and 44.1 kHz AAC,
/// encoded by FFmpeg.
fn ffmpeg_source(ffmpeg: &Path, dir: &Path) -> PathBuf {
    let source = dir.join("source.mp4");
    let status = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args([
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1.5:size=64x48:rate=60000/1001",
        ])
        .args([
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=1.5",
        ])
        .args([
            "-c:v", "libx264", "-bf", "2", "-g", "24", "-pix_fmt", "yuv420p",
        ])
        .args(["-c:a", "aac", "-shortest"])
        .arg(&source)
        .status()
        .unwrap();
    assert!(status.success(), "ffmpeg could not make the source");
    source
}

/// Starts a muxer on the source's streams, with their edits, whose
/// durations go from the source's movie timescale to `scale`.
fn remux_start<W: std::io::Write + std::io::Seek>(
    mux: &mut IsoBmffMux<W>,
    streams: &[crate::Stream],
    source_scale: Timescale,
    scale: u32,
) {
    for stream in streams {
        let id = mux.add_stream(&stream.params).unwrap();
        let edits = stream
            .edits
            .iter()
            .map(|edit| Edit {
                segment_duration: (edit.segment_duration * u64::from(scale))
                    .div_ceil(u64::from(source_scale.get())),
                ..*edit
            })
            .collect();
        mux.set_edits(id, edits).unwrap();
    }
}

#[test]
fn ffprobe_agrees_with_every_layout_and_ffmpeg_decodes_it_cleanly() {
    let (Some(ffmpeg), Some(ffprobe)) = (tool("ffmpeg"), tool("ffprobe")) else {
        eprintln!("not available: ffmpeg and ffprobe");
        return;
    };
    let dir = scratch("layouts");
    let source = ffmpeg_source(&ffmpeg, &dir);
    let truth = probe(&ffprobe, &source);
    assert!(truth.packets.len() > 100);
    let (streams, scale, packets) = demux(&source);
    let theirs = Some(scale.get());
    let layouts = [
        // The muxer's own movie timescale, then FFmpeg's 1000.
        ("progressive", MuxOptions::default()),
        (
            "progressive-1000",
            MuxOptions {
                movie_timescale: theirs,
                ..MuxOptions::default()
            },
        ),
        (
            "fast-start",
            MuxOptions {
                moov_at_end: false,
                movie_timescale: theirs,
                ..MuxOptions::default()
            },
        ),
        (
            "recoverable",
            MuxOptions {
                recoverable: true,
                fragment_millis: 300,
                movie_timescale: theirs,
                ..MuxOptions::default()
            },
        ),
        (
            "recoverable-fast-start",
            MuxOptions {
                recoverable: true,
                moov_at_end: false,
                fragment_millis: 300,
                creation_time: Some(1_795_478_400),
                ..MuxOptions::default()
            },
        ),
        (
            "fragmented",
            MuxOptions {
                fragmented: true,
                fragment_millis: 300,
                movie_timescale: theirs,
                ..MuxOptions::default()
            },
        ),
    ];
    for (name, options) in layouts {
        let path = dir.join(format!("{name}.mp4"));
        // Left to itself the muxer picks the video's 60000 for this source:
        // with 44100 the least common multiple is 8 820 000.
        let ours = options.movie_timescale.unwrap_or(VIDEO_SCALE);
        let fragmented = options.fragmented;
        let mut mux = IsoBmffMux::with_reader(MovieFile::create(&path).unwrap(), options);
        remux_start(&mut mux, &streams, scale, ours);
        for packet in &packets {
            mux.write_packet(packet).unwrap();
        }
        let mut out = mux.finish_into().unwrap().into_inner();
        std::io::Write::flush(&mut out).unwrap();
        drop(out);

        let ours = probe(&ffprobe, &path);
        assert_eq!(ours.packets.len(), truth.packets.len(), "{name}: packets");
        for (ours, theirs) in ours.packets.iter().zip(&truth.packets) {
            if fragmented {
                // Reading fragments, its own too, FFmpeg never marks the
                // AAC priming packet "discard" and gives a track's last
                // packet the duration of the one before, where a movie's
                // tables say 614. So the duration and flags are compared
                // through this crate's demuxer below, and here the rest.
                let rest = |line: &str| {
                    let fields: Vec<&str> = line.split(' ').collect();
                    [fields[0], fields[1], fields[2], fields[4], &fields[5][..1]].join(" ")
                };
                assert_eq!(rest(ours), rest(theirs), "{name}: stream pts dts size key");
            } else {
                assert_eq!(ours, theirs, "{name}: stream pts dts duration size flags");
            }
        }
        if fragmented {
            // FFmpeg sums a fragmented track from its runs and that last
            // duration, so its durations are a few hundred samples off.
            assert!((ours.format_duration - truth.format_duration).abs() < 0.03);
        } else {
            assert_eq!(ours.stream_durations, truth.stream_durations, "{name}");
            assert!(
                (ours.format_duration - truth.format_duration).abs() < 0.002,
                "{name}: {} against {}",
                ours.format_duration,
                truth.format_duration
            );
        }
        // Decoding every frame logs nothing at the error level.
        let decode = Command::new(&ffmpeg)
            .args(["-v", "error", "-i"])
            .arg(&path)
            .args(["-f", "null", "-"])
            .output()
            .unwrap();
        assert!(
            decode.status.success() && decode.stderr.is_empty(),
            "{name}: {}",
            String::from_utf8_lossy(&decode.stderr)
        );
        // And this crate reads back what it was given.
        let (_, _, read) = demux(&path);
        assert_eq!(by_track(read), by_track(packets.clone()), "{name}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ffmpeg_plays_an_unfinished_recoverable_file_up_to_its_last_fragment() {
    let (Some(ffmpeg), Some(ffprobe)) = (tool("ffmpeg"), tool("ffprobe")) else {
        eprintln!("not available: ffmpeg and ffprobe");
        return;
    };
    let dir = scratch("recover");
    let source = ffmpeg_source(&ffmpeg, &dir);
    let (streams, scale, packets) = demux(&source);
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut mux = IsoBmffMux::with_reader(
            &mut cursor,
            MuxOptions {
                recoverable: true,
                fragment_millis: 300,
                ..MuxOptions::default()
            },
        );
        remux_start(&mut mux, &streams, scale, VIDEO_SCALE);
        for packet in &packets {
            mux.write_packet(packet).unwrap();
        }
        // The process dies here: no finish.
    }
    let unfinished = cursor.into_inner();
    let fragments = whole_fragments(&unfinished);
    assert!(fragments.len() >= 3, "{fragments:?}");
    let ends: Vec<usize> = top_level(&unfinished)
        .iter()
        .filter(|(kind, _, _)| kind == b"mdat")
        .map(|(_, _, end)| *end)
        .collect();
    let count = |cut: usize| {
        let path = dir.join(format!("cut-{cut}.mp4"));
        std::fs::write(&path, &unfinished[..cut]).unwrap();
        let output = Command::new(&ffprobe)
            .args([
                "-v",
                "error",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
            ])
            .args(["-of", "csv=p=0"])
            .arg(&path)
            .output()
            .unwrap();
        let total: usize = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| line.trim().trim_end_matches(',').parse::<usize>().ok())
            .sum();
        (path, total)
    };
    let mut whole = 0;
    for (index, end) in ends.iter().enumerate() {
        whole += fragments[index];
        // Cut on the boundary, FFmpeg reads every packet and decodes them
        // without complaint.
        let (path, total) = count(*end);
        assert_eq!(total, whole, "cut at {end}");
        let decode = Command::new(&ffmpeg)
            .args(["-v", "error", "-i"])
            .arg(&path)
            .args(["-f", "null", "-"])
            .output()
            .unwrap();
        assert!(
            decode.status.success() && decode.stderr.is_empty(),
            "cut at {end}: {}",
            String::from_utf8_lossy(&decode.stderr)
        );
        // Cut inside the next fragment, at a third and two thirds of it,
        // every whole fragment before the cut is still read.
        let next = ends.get(index + 1).copied().unwrap_or(unfinished.len());
        for cut in [end + (next - end) / 3, end + (next - end) * 2 / 3] {
            if cut > *end && cut < next {
                let (_, total) = count(cut);
                assert!(total >= whole, "cut at {cut}: {total} of {whole}");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_movie_file_reads_back_what_it_buffered() {
    use std::io::{Read, Seek, SeekFrom, Write};
    let path = std::env::temp_dir().join(format!("isobmff-read-back-{}", std::process::id()));
    let mut file = MovieFile::create(&path).unwrap();
    file.write_all(b"0123456789").unwrap();
    // Still in the write buffer: a read sees it all the same.
    file.seek(SeekFrom::Start(3)).unwrap();
    let mut four = [0u8; 4];
    file.read_exact(&mut four).unwrap();
    assert_eq!(&four, b"3456");
    file.seek(SeekFrom::Start(8)).unwrap();
    file.write_all(b"ab").unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut all = Vec::new();
    file.read_to_end(&mut all).unwrap();
    assert_eq!(all, b"01234567ab");
    drop(file);
    let _ = std::fs::remove_file(&path);
    // `Mp4Mux` and its options are what export names.
    let _ = Mp4Mux::with_options(Cursor::new(Vec::new()), Mp4Options::default());
}
