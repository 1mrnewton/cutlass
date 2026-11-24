//! Constant-size audio: PCM tracks read as packets of about 1024 frames.
//!
//! The fixtures are built here byte by byte: a classic QuickTime sound
//! description in each version, with the sample tables a camera or FFmpeg
//! writes. FFmpeg is an oracle where it is installed and is skipped where
//! it is not.

use crate::limits::MAX_SAMPLES;
use crate::{
    CodecId, Demux, Error, FileBrand, IsoBmff, IsoBmffFile, IsoBmffMux, MediaKind, Mux, MuxOptions,
    Packet, SampleFormat, SliceSource, StreamParams, Timescale,
};
use cutlass_container::{Landing, SeekTo};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

enum SizeTable {
    Constant { size: u32, count: u32 },
    Each(Vec<u32>),
}

struct Fixture {
    handler: [u8; 4],
    /// One sample entry box.
    entry: Vec<u8>,
    timescale: u32,
    /// `(count, delta)` rows.
    stts: Vec<(u32, u32)>,
    /// `(first chunk, samples per chunk)` rows.
    stsc: Vec<(u32, u32)>,
    sizes: SizeTable,
    /// Chunk offsets from the start of the payload.
    chunks: Vec<u32>,
    payload: Vec<u8>,
}

fn box_of(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = u32::try_from(payload.len() + 8)
        .unwrap()
        .to_be_bytes()
        .to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

fn table(rows: &[(u32, u32)], third: Option<u32>) -> Vec<u8> {
    let mut body = vec![0, 0, 0, 0];
    body.extend(u32::try_from(rows.len()).unwrap().to_be_bytes());
    for (a, b) in rows {
        body.extend(a.to_be_bytes());
        body.extend(b.to_be_bytes());
        if let Some(third) = third {
            body.extend(third.to_be_bytes());
        }
    }
    body
}

fn movie(fixture: &Fixture) -> Vec<u8> {
    let ftyp = box_of(b"ftyp", b"qt  \0\0\0\0qt  ");
    let data_at = u32::try_from(ftyp.len()).unwrap() + 8;
    let mut stbl = Vec::new();
    let mut stsd = vec![0, 0, 0, 0, 0, 0, 0, 1];
    stsd.extend_from_slice(&fixture.entry);
    stbl.extend(box_of(b"stsd", &stsd));
    stbl.extend(box_of(b"stts", &table(&fixture.stts, None)));
    stbl.extend(box_of(b"stsc", &table(&fixture.stsc, Some(1))));
    let mut stsz = vec![0, 0, 0, 0];
    match &fixture.sizes {
        SizeTable::Constant { size, count } => {
            stsz.extend(size.to_be_bytes());
            stsz.extend(count.to_be_bytes());
        }
        SizeTable::Each(rows) => {
            stsz.extend(0u32.to_be_bytes());
            stsz.extend(u32::try_from(rows.len()).unwrap().to_be_bytes());
            for row in rows {
                stsz.extend(row.to_be_bytes());
            }
        }
    }
    stbl.extend(box_of(b"stsz", &stsz));
    let mut stco = vec![0, 0, 0, 0];
    stco.extend(u32::try_from(fixture.chunks.len()).unwrap().to_be_bytes());
    for chunk in &fixture.chunks {
        stco.extend((data_at + chunk).to_be_bytes());
    }
    stbl.extend(box_of(b"stco", &stco));
    let mut mdhd = vec![0u8; 12];
    mdhd.extend(fixture.timescale.to_be_bytes());
    mdhd.extend([0; 8]);
    let mut hdlr = vec![0u8; 8];
    hdlr.extend(fixture.handler);
    let mut mdia = box_of(b"mdhd", &mdhd);
    mdia.extend(box_of(b"hdlr", &hdlr));
    mdia.extend(box_of(b"minf", &box_of(b"stbl", &stbl)));
    let mut tkhd = vec![0u8; 84];
    tkhd[15] = 1;
    let mut trak = box_of(b"tkhd", &tkhd);
    trak.extend(box_of(b"mdia", &mdia));
    let mut mvhd = vec![0u8; 12];
    mvhd.extend(1000u32.to_be_bytes());
    mvhd.extend([0; 4]);
    let mut moov = box_of(b"mvhd", &mvhd);
    moov.extend(box_of(b"trak", &trak));
    let mut file = ftyp;
    file.extend(box_of(b"mdat", &fixture.payload));
    file.extend(box_of(b"moov", &moov));
    file
}

/// A QuickTime sound description of `version` 0 or 1. Version 1 adds the
/// four fields after the rate, written the way QuickTime writes them for
/// uncompressed sound: one sample per packet.
fn sound_entry(
    code: &[u8; 4],
    version: u16,
    channels: u16,
    bits: u16,
    rate: u16,
    children: &[u8],
) -> Vec<u8> {
    let mut entry = vec![0, 0, 0, 0, 0, 0, 0, 1];
    entry.extend(version.to_be_bytes());
    entry.extend([0; 6]);
    entry.extend(channels.to_be_bytes());
    entry.extend(bits.to_be_bytes());
    entry.extend([0; 4]);
    entry.extend(rate.to_be_bytes());
    entry.extend([0, 0]);
    if version == 1 {
        let width = u32::from(bits / 8);
        entry.extend(1u32.to_be_bytes());
        entry.extend(width.to_be_bytes());
        entry.extend((width * u32::from(channels)).to_be_bytes());
        entry.extend(width.to_be_bytes());
    }
    entry.extend_from_slice(children);
    box_of(code, &entry)
}

/// A version-1 description of a compressed codec: `per_packet` frames in
/// `per_frame` bytes over all channels.
fn block_entry(code: &[u8; 4], channels: u16, per_packet: u32, per_frame: u32) -> Vec<u8> {
    let mut entry = vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 1];
    entry.extend([0; 6]);
    entry.extend(channels.to_be_bytes());
    entry.extend(16u16.to_be_bytes());
    entry.extend([0; 4]);
    entry.extend(44_100u16.to_be_bytes());
    entry.extend([0, 0]);
    entry.extend(per_packet.to_be_bytes());
    entry.extend((per_frame / u32::from(channels)).to_be_bytes());
    entry.extend(per_frame.to_be_bytes());
    entry.extend(2u32.to_be_bytes());
    box_of(code, &entry)
}

/// A version-2 `lpcm` description: the rate is a float and the bytes of one
/// frame are stated.
fn lpcm_entry(rate: f64, channels: u32, bits: u32, flags: u32, frame_bytes: u32) -> Vec<u8> {
    let mut entry = vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 2];
    entry.extend([0; 6]);
    entry.extend(3u16.to_be_bytes());
    entry.extend(16u16.to_be_bytes());
    entry.extend((-2i16).to_be_bytes());
    entry.extend([0, 0]);
    entry.extend(0x0001_0000u32.to_be_bytes());
    entry.extend(72u32.to_be_bytes());
    entry.extend(rate.to_bits().to_be_bytes());
    entry.extend(channels.to_be_bytes());
    entry.extend(0x7F00_0000u32.to_be_bytes());
    entry.extend(bits.to_be_bytes());
    entry.extend(flags.to_be_bytes());
    entry.extend(frame_bytes.to_be_bytes());
    entry.extend(1u32.to_be_bytes());
    box_of(b"lpcm", &entry)
}

/// An `esds` box whose decoder-specific info is `config`.
fn esds(config: &[u8]) -> Vec<u8> {
    let descriptor = |tag: u8, body: &[u8]| {
        let mut out = vec![tag, u8::try_from(body.len()).unwrap()];
        out.extend_from_slice(body);
        out
    };
    let mut decoder = vec![0x40, 0x15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    decoder.extend(descriptor(0x05, config));
    let mut es = vec![0, 1, 0];
    es.extend(descriptor(0x04, &decoder));
    let mut body = vec![0; 4];
    body.extend(descriptor(0x03, &es));
    box_of(b"esds", &body)
}

fn noise(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
}

fn read_all(bytes: &[u8]) -> Vec<Packet> {
    let mut source = SliceSource::new(bytes);
    let mut demux = IsoBmff::open(&mut source).expect("open");
    let mut out = Vec::new();
    while let Some(packet) = demux.read_packet(&mut source).expect("read") {
        out.push(packet);
    }
    out
}

fn open_error(bytes: &[u8]) -> Error {
    match IsoBmff::open(&mut SliceSource::new(bytes)) {
        Ok(_) => panic!("opened a file that should fail"),
        Err(error) => error,
    }
}

/// 2500 frames of 16-bit stereo in two chunks, 2000 then 500, with three
/// bytes of another track's data between them. `stsz` says what classic
/// QuickTime says: sample size 1.
fn classic(version: u16, stated_size: u32) -> (Fixture, Vec<u8>) {
    let payload = noise(2500 * 4 + 3, 7);
    let fixture = Fixture {
        handler: *b"soun",
        entry: sound_entry(b"sowt", version, 2, 16, 48_000, &[]),
        timescale: 48_000,
        stts: vec![(2500, 1)],
        stsc: vec![(1, 2000), (2, 500)],
        sizes: SizeTable::Constant {
            size: stated_size,
            count: 2500,
        },
        chunks: vec![0, 8003],
        payload: payload.clone(),
    };
    (fixture, payload)
}

#[test]
fn a_sowt_entry_with_sample_size_1_reads_whole_frames_in_both_versions() {
    // `stsz` 1 is classic QuickTime; 4 is what FFmpeg and this muxer write.
    for (version, stated) in [(0, 1), (1, 1), (0, 4), (1, 4)] {
        let (fixture, payload) = classic(version, stated);
        let bytes = movie(&fixture);
        let packets = read_all(&bytes);
        let seen: Vec<(i64, i64, u32, usize)> = packets
            .iter()
            .map(|packet| {
                (
                    packet.dts,
                    packet.pts,
                    packet.duration,
                    packet.payload.len(),
                )
            })
            .collect();
        // FFmpeg's grouping: 1024 frames, the rest of the chunk, then the
        // next chunk. A packet does not cross a chunk.
        assert_eq!(
            seen,
            [
                (0, 0, 1024, 4096),
                (1024, 1024, 976, 3904),
                (2000, 2000, 500, 2000)
            ],
            "version {version} stsz {stated}"
        );
        assert!(packets.iter().all(|packet| packet.keyframe));
        assert_eq!(packets[0].payload, payload[..4096]);
        assert_eq!(packets[1].payload, payload[4096..8000]);
        assert_eq!(packets[2].payload, payload[8003..]);

        let demux = IsoBmff::open(&mut SliceSource::new(&bytes)).unwrap();
        let params = &demux.streams()[0].params;
        assert_eq!(params.codec, CodecId::Pcm);
        assert_eq!(params.duration_ticks, 2500);
        let MediaKind::Audio(audio) = params.kind else {
            panic!("not audio");
        };
        assert_eq!(audio.format, SampleFormat::S16Le);
        assert_eq!((audio.sample_rate, audio.channels), (48_000, 2));
    }
}

#[test]
fn the_shared_demux_returns_the_same_groups_and_the_indexed_end() {
    let (fixture, payload) = classic(1, 1);
    let mut file = IsoBmffFile::open(Cursor::new(movie(&fixture))).unwrap();
    let track = cutlass_container::Demux::streams(&file)[0].id();
    assert_eq!(
        cutlass_container::Demux::indexed_end(&file, track),
        Some(2500)
    );
    let mut back = Vec::new();
    let mut times = Vec::new();
    while let Some(packet) = cutlass_container::Demux::read_packet(&mut file).unwrap() {
        let time = packet.timestamp();
        times.push((time.presentation(), time.duration()));
        back.extend_from_slice(packet.bytes());
    }
    assert_eq!(
        times,
        [
            (Some(0), Some(1024)),
            (Some(1024), Some(976)),
            (Some(2000), Some(500))
        ]
    );
    let mut expected = payload[..8000].to_vec();
    expected.extend_from_slice(&payload[8003..]);
    assert_eq!(back, expected);
}

#[test]
fn a_version_2_entry_frames_by_its_stated_bytes_per_packet() {
    // 24-bit stereo, little-endian, packed: six bytes a frame.
    let payload = noise(1500 * 6, 11);
    let bytes = movie(&Fixture {
        handler: *b"soun",
        entry: lpcm_entry(96_000.0, 2, 24, 4 | 8, 6),
        timescale: 96_000,
        stts: vec![(1500, 1)],
        stsc: vec![(1, 1500)],
        sizes: SizeTable::Constant {
            size: 1,
            count: 1500,
        },
        chunks: vec![0],
        payload: payload.clone(),
    });
    let packets = read_all(&bytes);
    assert_eq!(packets.len(), 2);
    assert_eq!(
        (packets[0].duration, packets[0].payload.len()),
        (1024, 6144)
    );
    assert_eq!((packets[1].dts, packets[1].duration), (1024, 476));
    let back: Vec<u8> = packets
        .iter()
        .flat_map(|packet| packet.payload.clone())
        .collect();
    assert_eq!(back, payload);
    let file = IsoBmffFile::open(Cursor::new(bytes)).unwrap();
    let params = cutlass_container::Demux::streams(&file)[0].params();
    assert_eq!(params.sample_rate(), Some(96_000));
    assert_eq!(params.extradata(), [2, 0, 0]);
}

#[test]
fn a_compressed_block_entry_is_cut_on_whole_blocks() {
    // IMA 4:1 as QuickTime describes it: 64 frames in 34 bytes a channel.
    let payload = noise(34 * 68, 3);
    let fixture = |per_chunk: u32, count: u32| Fixture {
        handler: *b"soun",
        entry: block_entry(b"ima4", 2, 64, 68),
        timescale: 44_100,
        stts: vec![(count, 1)],
        stsc: vec![(1, per_chunk)],
        sizes: SizeTable::Constant { size: 1, count },
        chunks: vec![0],
        payload: payload.clone(),
    };
    let packets = read_all(&movie(&fixture(34 * 64, 34 * 64)));
    let seen: Vec<(i64, u32, usize)> = packets
        .iter()
        .map(|packet| (packet.dts, packet.duration, packet.payload.len()))
        .collect();
    assert_eq!(
        seen,
        [
            (0, 1024, 16 * 68),
            (1024, 1024, 16 * 68),
            (2048, 128, 2 * 68)
        ]
    );
    let back: Vec<u8> = packets
        .iter()
        .flat_map(|packet| packet.payload.clone())
        .collect();
    assert_eq!(back, payload);
    // A chunk that ends inside a block has no byte length.
    assert_eq!(
        open_error(&movie(&fixture(100, 100))),
        Error::Unsupported {
            what: "audio chunk that splits a frame"
        }
    );
}

#[test]
fn a_rate_of_zero_is_read_from_the_codec_or_the_media_timescale() {
    // (entry, media timescale, expected rate)
    let aac_96k = sound_entry(b"mp4a", 0, 2, 16, 0, &esds(&[0x10, 0x10]));
    // AOT 5 (SBR) over a 48 kHz core, extension rate 96 kHz.
    let he_aac = sound_entry(b"mp4a", 0, 2, 16, 0, &esds(&[0x29, 0x90, 0x08]));
    let mut dfla = vec![0, 0, 0, 0, 0x80, 0, 0, 34];
    let mut info = vec![0u8; 34];
    // 88200 Hz in the 20 bits after the block and frame sizes.
    info[10..13].copy_from_slice(&[0x15, 0x88, 0x80]);
    dfla.extend(info);
    let flac = sound_entry(b"fLaC", 0, 2, 16, 0, &box_of(b"dfLa", &dfla));
    let bare = sound_entry(b"mp4a", 0, 2, 16, 0, &[]);
    for (entry, timescale, rate) in [
        (aac_96k, 96_000, 96_000),
        (he_aac, 48_000, 96_000),
        (flac, 88_200, 88_200),
        // No configuration: the media timescale is the rate.
        (bare, 88_200, 88_200),
    ] {
        let bytes = movie(&Fixture {
            handler: *b"soun",
            entry,
            timescale,
            stts: vec![(2, 1024)],
            stsc: vec![(1, 2)],
            sizes: SizeTable::Each(vec![5, 7]),
            chunks: vec![0],
            payload: noise(12, 1),
        });
        let file = IsoBmffFile::open(Cursor::new(bytes.clone())).expect("opens with rate 0");
        let params = cutlass_container::Demux::streams(&file)[0].params();
        assert_eq!(params.sample_rate(), Some(rate));
        // Compressed audio keeps one packet a sample.
        let packets = read_all(&bytes);
        let sizes: Vec<usize> = packets.iter().map(|packet| packet.payload.len()).collect();
        assert_eq!(sizes, [5, 7]);
        assert_eq!(packets[1].dts, 1024);
    }
}

#[test]
fn seek_lands_on_the_group_that_holds_the_frame() {
    // 8000 mono 16-bit frames in chunks of 5000 and 3000.
    let payload = noise(16_000, 5);
    let bytes = movie(&Fixture {
        handler: *b"soun",
        entry: sound_entry(b"twos", 0, 1, 16, 48_000, &[]),
        timescale: 48_000,
        stts: vec![(8000, 1)],
        stsc: vec![(1, 5000), (2, 3000)],
        sizes: SizeTable::Constant {
            size: 1,
            count: 8000,
        },
        chunks: vec![0, 10_000],
        payload: payload.clone(),
    });
    let mut file = IsoBmffFile::open(Cursor::new(bytes)).unwrap();
    let track = cutlass_container::Demux::streams(&file)[0].id().get();
    // Group starts: 0, 1024, 2048, 3072, 4096, then the second chunk at
    // 5000, 6024, 7048.
    for (frame, ordinal, start) in [
        (0_i64, 0_u32, 0_i64),
        (1023, 0, 0),
        (1024, 1, 1024),
        (4999, 4, 4096),
        (5000, 5, 5000),
        (6023, 5, 5000),
        (7999, 7, 7048),
    ] {
        let landing = cutlass_container::Demux::seek(
            &mut file,
            SeekTo {
                track,
                time: frame,
                end: None,
            },
        )
        .unwrap();
        assert!(landing.index_used);
        assert_eq!(landing.payload_scans, 0);
        let Landing::Packet {
            ordinal: landed,
            time,
            keyframe,
            preroll,
            ..
        } = landing.landing
        else {
            panic!("frame {frame} landed at the end");
        };
        assert_eq!((landed, time, keyframe, preroll), (ordinal, start, true, 0));
        // The asked frame is `frame - time` frames into the next packet.
        let packet = cutlass_container::Demux::read_packet(&mut file)
            .unwrap()
            .unwrap();
        assert_eq!(packet.timestamp().presentation(), Some(start));
        let within = usize::try_from(frame - start).unwrap() * 2;
        let at = usize::try_from(frame).unwrap() * 2;
        assert_eq!(packet.bytes()[within..within + 2], payload[at..at + 2]);
    }
    let end = cutlass_container::Demux::seek(
        &mut file,
        SeekTo {
            track,
            time: 8000,
            end: None,
        },
    )
    .unwrap();
    assert_eq!(end.landing, Landing::End);
    assert!(
        cutlass_container::Demux::read_packet(&mut file)
            .unwrap()
            .is_none()
    );
    // A refused seek leaves the read where it was.
    let _ = cutlass_container::Demux::seek(
        &mut file,
        SeekTo {
            track,
            time: 2048,
            end: None,
        },
    )
    .unwrap();
    assert!(
        cutlass_container::Demux::seek(
            &mut file,
            SeekTo {
                track: track + 9,
                time: 0,
                end: None,
            },
        )
        .is_err()
    );
    let packet = cutlass_container::Demux::read_packet(&mut file)
        .unwrap()
        .unwrap();
    assert_eq!(packet.timestamp().presentation(), Some(2048));
}

#[test]
fn a_long_pcm_movie_this_muxer_wrote_opens_again() {
    // More audio frames than `MAX_SAMPLES`, which a row per frame refused.
    const PACKET_FRAMES: u32 = 4096;
    let packets = u32::try_from(MAX_SAMPLES).unwrap() / PACKET_FRAMES + 3;
    let params = StreamParams::audio(
        CodecId::Pcm,
        Timescale(48_000),
        48_000,
        1,
        8,
        SampleFormat::U8,
    );
    let mut mux = IsoBmffMux::new(
        Cursor::new(Vec::new()),
        MuxOptions {
            brand: FileBrand::Mov,
            ..MuxOptions::default()
        },
    );
    let id = mux.add_stream(&params).unwrap();
    let mut written = Vec::new();
    for index in 0..packets {
        let payload = noise(usize::try_from(PACKET_FRAMES).unwrap(), index);
        mux.write_packet(&Packet {
            track_id: id,
            payload: payload.clone(),
            dts: i64::from(index) * i64::from(PACKET_FRAMES),
            pts: i64::from(index) * i64::from(PACKET_FRAMES),
            duration: PACKET_FRAMES,
            timescale: Timescale(48_000),
            keyframe: true,
        })
        .unwrap();
        written.extend(payload);
    }
    let bytes = mux.finish_into().unwrap().into_inner();
    let frames = u64::from(packets) * u64::from(PACKET_FRAMES);
    assert!(frames > MAX_SAMPLES);

    let mut source = SliceSource::new(&bytes);
    let mut demux = IsoBmff::open(&mut source).expect("a long PCM movie opens");
    assert_eq!(demux.streams()[0].params.duration_ticks, frames);
    let mut back = Vec::with_capacity(written.len());
    let mut count = 0u32;
    let mut next = 0i64;
    while let Some(packet) = demux.read_packet(&mut source).unwrap() {
        assert_eq!(packet.dts, next);
        assert_eq!(packet.duration, 1024);
        next += i64::from(packet.duration);
        count += 1;
        back.extend(packet.payload);
    }
    assert_eq!(count, packets * 4);
    assert_eq!(u64::try_from(next).unwrap(), frames);
    assert!(back == written, "the samples differ");
    // The last frame is still reachable by time.
    let last = i64::try_from(frames).unwrap() - 1;
    let at = demux.seek(&mut source, id, last).unwrap();
    assert_eq!(at.dts, last / 1024 * 1024);
    let packet = demux.read_packet(&mut source).unwrap().unwrap();
    assert_eq!(packet.payload.last(), written.last());
}

#[test]
fn tables_that_do_not_describe_the_payload_are_errors() {
    let base = |stsc: Vec<(u32, u32)>, count: u32, payload: usize| Fixture {
        handler: *b"soun",
        entry: sound_entry(b"raw ", 0, 1, 8, 8_000, &[]),
        timescale: 8_000,
        stts: vec![(count, 1)],
        stsc,
        sizes: SizeTable::Constant { size: 1, count },
        chunks: vec![0],
        payload: noise(payload, 2),
    };
    // The chunk holds fewer frames than `stsz` counts.
    assert_eq!(
        open_error(&movie(&base(vec![(1, 100)], 200, 200))),
        Error::Missing {
            what: "chunk covers samples"
        }
    );
    // A chunk of no samples covers nothing.
    assert_eq!(
        open_error(&movie(&base(vec![(1, 0)], 200, 200))),
        Error::Missing {
            what: "chunk covers samples"
        }
    );
    // Four billion frames in one chunk open without a row each, and the
    // first read past the media data is refused.
    let bytes = movie(&base(vec![(1, u32::MAX)], u32::MAX, 1500));
    let mut source = SliceSource::new(&bytes);
    let mut demux = IsoBmff::open(&mut source).expect("no allocation by count");
    assert_eq!(
        demux.read_packet(&mut source).unwrap().unwrap().payload,
        noise(1500, 2)[..1024]
    );
    assert!(matches!(
        demux.read_packet(&mut source),
        Err(Error::SampleExtendsPastMedia { size: 1024, .. })
    ));
    // The size table ends inside the chunk: the last packet is short.
    let packets = read_all(&movie(&base(vec![(1, 5000)], 1100, 1100)));
    let durations: Vec<u32> = packets.iter().map(|packet| packet.duration).collect();
    assert_eq!(durations, [1024, 76]);
    // A version-1 entry cut before its four extra fields.
    let mut short = sound_entry(b"sowt", 1, 2, 16, 48_000, &[]);
    short.truncate(short.len() - 8);
    let size = u32::try_from(short.len()).unwrap();
    short[..4].copy_from_slice(&size.to_be_bytes());
    let mut fixture = base(vec![(1, 100)], 100, 400);
    fixture.entry = short;
    assert!(matches!(
        open_error(&movie(&fixture)),
        Error::Truncated { .. }
    ));
}

#[test]
fn other_constant_size_tracks_keep_one_packet_a_sample() {
    // Constant-size audio whose samples are 160 ticks long is a codec's
    // frames, not PCM: one packet each.
    let bytes = movie(&Fixture {
        handler: *b"soun",
        entry: sound_entry(b"samr", 0, 1, 16, 8_000, &[]),
        timescale: 8_000,
        stts: vec![(3, 160)],
        stsc: vec![(1, 3)],
        sizes: SizeTable::Constant { size: 32, count: 3 },
        chunks: vec![0],
        payload: noise(96, 4),
    });
    let packets = read_all(&bytes);
    assert_eq!(packets.len(), 3);
    assert_eq!((packets[2].dts, packets[2].payload.len()), (320, 32));
    // A picture track still makes a row per sample, so its count is bounded.
    let count = u32::try_from(MAX_SAMPLES).unwrap() + 1;
    let mut entry = vec![0u8; 78];
    entry[7] = 1;
    let error = open_error(&movie(&Fixture {
        handler: *b"vide",
        entry: box_of(b"avc1", &entry),
        timescale: 30,
        stts: vec![(count, 1)],
        stsc: vec![(1, count)],
        sizes: SizeTable::Constant { size: 1, count },
        chunks: vec![0],
        payload: vec![0; 16],
    }));
    assert_eq!(
        error,
        Error::Limit {
            what: "samples",
            value: u64::from(count)
        }
    );
}

fn tool(name: &str) -> Option<PathBuf> {
    let output = Command::new("which").arg(name).output().ok()?;
    let path = PathBuf::from(String::from_utf8(output.stdout).ok()?.trim());
    path.is_file().then_some(path)
}

fn stdout_of(program: &Path, args: &[&str]) -> Vec<u8> {
    let output = Command::new(program).args(args).output().unwrap();
    assert!(
        output.status.success(),
        "{program:?} {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

/// FFmpeg writes three seconds of PCM in each layout and container. Every
/// packet's time, duration and size match `ffprobe -show_packets`, the
/// packet bytes match FFmpeg's stream copy, and for 16-bit little-endian
/// they match FFmpeg's decode to `s16le`.
#[test]
fn ffmpeg_pcm_files_give_the_packets_ffprobe_lists() {
    let (Some(ffmpeg), Some(ffprobe)) = (tool("ffmpeg"), tool("ffprobe")) else {
        eprintln!("not available: ffmpeg or ffprobe");
        return;
    };
    let dir = std::env::temp_dir().join(format!("isobmff-grouped-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for codec in ["pcm_s16le", "pcm_s24be", "pcm_f32le"] {
        for extension in ["mov", "mp4"] {
            let path = dir.join(format!("{codec}.{extension}"));
            let name = path.to_str().unwrap();
            let written = Command::new(&ffmpeg)
                .args(["-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi"])
                .args(["-i", "anoisesrc=sample_rate=48000:duration=3:seed=5"])
                .args(["-ac", "2", "-c:a", codec, name])
                .output()
                .unwrap();
            if !written.status.success() {
                // An FFmpeg before 6.1 does not write PCM into MP4.
                eprintln!("not available: {codec} in {extension}");
                continue;
            }
            let listed = String::from_utf8(stdout_of(
                &ffprobe,
                &[
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "packet=pts,dts,duration,size",
                    "-of",
                    "csv=p=0",
                    name,
                ],
            ))
            .unwrap();
            let copied = stdout_of(
                &ffmpeg,
                &[
                    "-v", "error", "-i", name, "-map", "0:a:0", "-c", "copy", "-f", "data", "-",
                ],
            );
            let mut file = IsoBmffFile::open(std::fs::File::open(&path).unwrap()).unwrap();
            let mut ours = String::new();
            let mut bytes = Vec::new();
            let mut count = 0u32;
            while let Some(packet) = cutlass_container::Demux::read_packet(&mut file).unwrap() {
                let time = packet.timestamp();
                ours.push_str(&format!(
                    "{},{},{},{}\n",
                    time.presentation().unwrap(),
                    time.decode().unwrap(),
                    time.duration().unwrap(),
                    packet.bytes().len()
                ));
                bytes.extend_from_slice(packet.bytes());
                count += 1;
            }
            let listed: Vec<&str> = listed.lines().map(str::trim).collect();
            let ours: Vec<&str> = ours.lines().collect();
            assert_eq!(ours, listed, "{name}");
            // About 47 packets a second at 48 kHz.
            assert_eq!(count, 141, "{name}");
            assert!(bytes == copied, "{name}: packet bytes differ");
            if codec == "pcm_s16le" {
                let decoded = stdout_of(&ffmpeg, &["-v", "error", "-i", name, "-f", "s16le", "-"]);
                assert!(bytes == decoded, "{name}: decoded samples differ");
            }
            // FFmpeg 6.1 labels 24-bit PCM in MP4 as 32-bit. The frame
            // size in `stsz` says what the bytes are.
            if codec == "pcm_s24be" {
                let layout = cutlass_container::Demux::streams(&file)[0]
                    .params()
                    .extradata()
                    .to_vec();
                assert_eq!(layout, [2, 1, 0], "{name}");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
