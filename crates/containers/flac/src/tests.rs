//! Mux and demux over hand-built frames: round trips, seeking, and every
//! refusal path.

use std::io::Cursor;

use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{
    Demux, Landing, Mux, MuxError, MuxStatus, NewStream, OpenError, ReadError, SeekTo,
    TrackDuration, TrackId,
};

use crate::format::{PLACEHOLDER, SeekPoint, StreamInfo, crc8, crc16};
use crate::{FlacDemux, FlacMux};

const RATE: u32 = 44_100;

/// Writes `value` as the UTF-8-style coded number.
fn coded(value: u64, out: &mut Vec<u8>) {
    if value < 0x80 {
        out.push(value as u8);
        return;
    }
    let mut bytes = 2;
    while value >> (6 * (bytes - 1)) >= 1 << (7 - bytes) {
        bytes += 1;
    }
    let lead = (0xFF00u16 >> bytes) as u8;
    out.push(lead | (value >> (6 * (bytes - 1))) as u8);
    for index in (0..bytes - 1).rev() {
        out.push(0x80 | ((value >> (6 * index)) & 0x3F) as u8);
    }
}

/// A mono 16-bit frame holding one constant subframe of `value`.
fn frame(number: u64, block: u32, variable: bool, value: i16) -> Vec<u8> {
    let mut out = vec![0xFF, 0xF8 | u8::from(variable), 0x79, 0x08];
    coded(number, &mut out);
    out.extend_from_slice(&u16::try_from(block - 1).expect("block").to_be_bytes());
    let crc = crc8(&out);
    out.push(crc);
    out.push(0x00);
    out.extend_from_slice(&value.to_be_bytes());
    let crc = crc16(&out);
    out.extend_from_slice(&crc.to_be_bytes());
    out
}

fn packet(data: Vec<u8>, pts: u64, block: u32) -> Packet {
    let timestamp = Timestamp::try_new(Some(pts as i64), Some(pts as i64), Some(i64::from(block)))
        .expect("timestamp");
    Packet::try_new(0, data, timestamp, Timescale::new(RATE), true).expect("packet")
}

fn stream(extra: Vec<u8>) -> NewStream {
    NewStream {
        params: StreamParams::try_audio(StreamCodec::Known(CodecId::Flac), RATE, 1, extra)
            .expect("params"),
        timescale: Timescale::new(RATE),
        language: None,
    }
}

/// Muxes `blocks` (sample counts per frame) and returns the file, the
/// packets written, and the finished STREAMINFO.
fn file(blocks: &[u32], variable: bool, slots: u32) -> (Vec<u8>, Vec<Packet>) {
    let mut mux = FlacMux::with_seek_points(Cursor::new(Vec::new()), slots);
    mux.add_stream(stream(Vec::new())).expect("stream");
    let mut packets = Vec::new();
    let mut sample = 0u64;
    for (index, block) in blocks.iter().enumerate() {
        let number = if variable { sample } else { index as u64 };
        let data = frame(number, *block, variable, index as i16);
        let packet = packet(data, sample, *block);
        mux.write_packet(&packet).expect("write");
        packets.push(packet);
        sample += u64::from(*block);
    }
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(RATE),
        ticks: sample as i64,
    }])
    .expect("finish");
    (mux.into_inner().into_inner(), packets)
}

fn open(bytes: &[u8]) -> Result<FlacDemux<Cursor<Vec<u8>>>, OpenError> {
    FlacDemux::open(Cursor::new(bytes.to_vec()), bytes.len() as u64)
}

fn read_all(demux: &mut FlacDemux<Cursor<Vec<u8>>>) -> Vec<Packet> {
    let mut out = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        out.push(packet);
    }
    out
}

fn blocks() -> Vec<u32> {
    let mut blocks = vec![4096; 200];
    blocks.push(1000);
    blocks
}

#[test]
fn a_fixed_stream_round_trips_with_streaminfo_and_seek_points() {
    let (bytes, written) = file(&blocks(), false, 16);
    let mut demux = open(&bytes).expect("open");
    let info = *demux.stream_info();
    assert_eq!(info.total_samples, 200 * 4096 + 1000);
    assert_eq!((info.min_block, info.max_block), (4096, 4096));
    assert_eq!(info.min_frame as usize, written[0].bytes().len());
    let points = demux.seek_points().to_vec();
    assert!(points.len() >= 2 && points.len() <= 16, "{points:?}");
    assert_eq!(
        points[0],
        SeekPoint {
            sample: 0,
            offset: 0,
            samples: 4096
        }
    );
    assert!(
        points
            .windows(2)
            .all(|pair| pair[0].sample < pair[1].sample)
    );
    let streams = demux.streams();
    assert_eq!(streams[0].params().extradata(), &info.to_bytes()[..]);
    let read = read_all(&mut demux);
    assert_eq!(read, written);
}

#[test]
fn a_variable_stream_keeps_sample_numbers() {
    let blocks = [4096u32, 1024, 512, 4096, 16, 333];
    let (bytes, written) = file(&blocks, true, 4);
    let mut demux = open(&bytes).expect("open");
    let info = *demux.stream_info();
    assert_eq!(info.max_block, 4096);
    assert_eq!(info.min_block, 16, "the last block is not a minimum");
    assert_eq!(read_all(&mut demux), written);
}

#[test]
fn seeks_land_on_the_frame_holding_the_target_with_and_without_an_index() {
    for slots in [0u32, 3, 128] {
        let (bytes, written) = file(&blocks(), false, slots);
        let mut demux = open(&bytes).expect("open");
        for target in [
            0i64, 1, 4095, 4096, 4097, 100_000, 409_600, 819_199, 819_200, 820_199,
        ] {
            let landing = demux
                .seek(SeekTo {
                    track: 0,
                    time: target,
                    end: None,
                })
                .expect("seek");
            let Landing::Packet { time, preroll, .. } = landing.landing else {
                panic!("slots {slots} target {target}: {landing:?}");
            };
            assert_eq!(preroll, 0);
            assert_eq!(time, target / 4096 * 4096, "slots {slots} target {target}");
            assert_eq!(landing.index_used, slots > 0);
            let next = demux.read_packet().expect("read").expect("packet");
            let index = (target / 4096) as usize;
            assert_eq!(next, written[index]);
        }
        let end = demux
            .seek(SeekTo {
                track: 0,
                time: 820_200,
                end: None,
            })
            .expect("seek");
        assert_eq!(end.landing, Landing::End);
        assert!(demux.read_packet().expect("read").is_none());
        let back = demux
            .seek(SeekTo {
                track: 0,
                time: -5,
                end: None,
            })
            .expect("seek");
        assert!(matches!(back.landing, Landing::Packet { time: 0, .. }));
    }
}

fn header_with(blocks: &[(u8, Vec<u8>)]) -> Vec<u8> {
    let mut out = b"fLaC".to_vec();
    for (index, (kind, body)) in blocks.iter().enumerate() {
        let last = if index + 1 == blocks.len() { 0x80 } else { 0 };
        let len = (body.len() as u32).to_be_bytes();
        out.extend_from_slice(&[last | kind, len[1], len[2], len[3]]);
        out.extend_from_slice(body);
    }
    out
}

fn info_body() -> Vec<u8> {
    StreamInfo {
        min_block: 4096,
        max_block: 4096,
        min_frame: 0,
        max_frame: 0,
        sample_rate: RATE,
        channels: 1,
        bits: 16,
        total_samples: 0,
        md5: [0; 16],
    }
    .to_bytes()
    .to_vec()
}

#[test]
fn malformed_metadata_is_refused() {
    let frame0 = frame(0, 4096, false, 1);
    let with_frame = |mut head: Vec<u8>| {
        head.extend_from_slice(&frame0);
        head
    };
    assert!(matches!(
        open(b"RIFF0000WAVE"),
        Err(OpenError::Unrecognized { .. })
    ));
    assert!(matches!(
        open(b"ID3\x04\x00\x00"),
        Err(OpenError::UnsupportedBrand { .. })
    ));
    assert!(open(b"fLa").is_err());
    let refused = [
        header_with(&[(4, vec![0; 8]), (0, info_body())]),
        header_with(&[(0, info_body()), (0, info_body())]),
        header_with(&[(0, info_body()[..33].to_vec())]),
        header_with(&[(0, info_body()), (127, Vec::new())]),
        header_with(&[(0, info_body()), (3, vec![0; 17])]),
        header_with(&[(0, info_body()), (3, Vec::new()), (3, Vec::new())]),
        header_with(&[(0, vec![0; 34])]),
    ];
    for (index, head) in refused.into_iter().enumerate() {
        assert!(open(&with_frame(head)).is_err(), "case {index}");
    }
    // Seek points out of order, or after a placeholder.
    let point = |sample: u64, offset: u64| {
        SeekPoint {
            sample,
            offset,
            samples: 4096,
        }
        .to_bytes()
    };
    let mut table = point(4096, 100).to_vec();
    table.extend_from_slice(&point(0, 0));
    assert!(open(&with_frame(header_with(&[(0, info_body()), (3, table)]))).is_err());
    let mut table = SeekPoint::placeholder().to_bytes().to_vec();
    table.extend_from_slice(&point(0, 0));
    assert!(open(&with_frame(header_with(&[(0, info_body()), (3, table)]))).is_err());
    // A block past the end of the file.
    let mut head = header_with(&[(0, info_body()), (6, vec![0; 10])]);
    head.truncate(head.len() - 4);
    assert!(open(&head).is_err());
    // Bytes that are not a frame after the metadata.
    let mut head = header_with(&[(0, info_body())]);
    head.extend_from_slice(&[0u8; 64]);
    assert!(open(&head).is_err());
}

#[test]
fn skipped_blocks_placeholders_and_an_id3v1_trailer_are_handled() {
    let mut table = SeekPoint {
        sample: 0,
        offset: 0,
        samples: 4096,
    }
    .to_bytes()
    .to_vec();
    table.extend_from_slice(&SeekPoint::placeholder().to_bytes());
    let mut bytes = header_with(&[
        (0, info_body()),
        (1, vec![0; 100]),
        (6, vec![9; 3000]),
        (3, table),
        (4, vec![0; 8]),
    ]);
    for index in 0..3 {
        bytes.extend_from_slice(&frame(index, 4096, false, index as i16));
    }
    let mut tag = b"TAG".to_vec();
    tag.resize(128, b'x');
    bytes.extend_from_slice(&tag);
    let mut demux = open(&bytes).expect("open");
    assert_eq!(demux.seek_points().len(), 1);
    assert_ne!(demux.seek_points()[0].sample, PLACEHOLDER);
    let packets = read_all(&mut demux);
    assert_eq!(packets.len(), 3);
    assert_eq!(packets[2].bytes(), frame(2, 4096, false, 2).as_slice());
}

#[test]
fn a_cut_last_frame_ends_the_stream_and_is_reported() {
    let (bytes, written) = file(&[4096, 4096, 4096], false, 0);
    let mut demux = open(&bytes[..bytes.len() - 3]).expect("open");
    let packets = read_all(&mut demux);
    assert_eq!(packets, written[..2]);
    assert!(demux.damaged_tail());
}

#[test]
fn damaged_frames_never_panic_the_demuxer() {
    let (bytes, _) = file(&[4096, 4096, 1024], false, 2);
    for at in 0..bytes.len() {
        for flip in [0x01u8, 0x80] {
            let mut damaged = bytes.clone();
            damaged[at] ^= flip;
            let Ok(mut demux) = open(&damaged) else {
                continue;
            };
            for _ in 0..8 {
                if !matches!(demux.read_packet(), Ok(Some(_))) {
                    break;
                }
            }
            let _ = demux.seek(SeekTo {
                track: 0,
                time: 5000,
                end: None,
            });
        }
    }
}

#[test]
fn cancel_stops_reads_and_seeks() {
    let (bytes, _) = file(&[4096, 4096], false, 0);
    let mut demux = open(&bytes).expect("open");
    demux.cancel();
    assert_eq!(demux.read_packet(), Err(ReadError::Cancelled));
    assert_eq!(
        demux
            .seek(SeekTo {
                track: 0,
                time: 0,
                end: None
            })
            .err(),
        Some(ReadError::Cancelled)
    );
}

fn duration(ticks: i64) -> [TrackDuration; 1] {
    [TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(RATE),
        ticks,
    }]
}

#[test]
fn the_muxer_refuses_frames_that_do_not_follow() {
    let try_write = |packets: &[Packet]| -> Result<(), MuxError> {
        let mut mux = FlacMux::new(Cursor::new(Vec::new()));
        mux.add_stream(stream(Vec::new()))?;
        for packet in packets {
            mux.write_packet(packet)?;
        }
        Ok(())
    };
    let good = packet(frame(0, 4096, false, 0), 0, 4096);
    assert!(try_write(std::slice::from_ref(&good)).is_ok());
    let mut bad_crc = frame(0, 4096, false, 0);
    let last = bad_crc.len() - 1;
    bad_crc[last] ^= 1;
    assert!(try_write(&[packet(bad_crc, 0, 4096)]).is_err());
    assert!(try_write(&[packet(frame(1, 4096, false, 0), 0, 4096)]).is_err());
    assert!(try_write(&[good.clone(), packet(frame(2, 4096, false, 0), 4096, 4096)]).is_err());
    assert!(try_write(&[good.clone(), packet(frame(1, 4096, false, 0), 4000, 4096)]).is_err());
    assert!(try_write(&[packet(frame(0, 4096, false, 0), 0, 4000)]).is_err());
    assert!(
        try_write(&[
            good.clone(),
            packet(frame(1, 1000, false, 0), 4096, 1000),
            packet(frame(2, 4096, false, 0), 5096, 4096),
        ])
        .is_err()
    );
    assert!(try_write(&[good.clone(), packet(frame(4096, 4096, true, 0), 4096, 4096)]).is_err());
    assert!(try_write(&[packet(b"not a frame".to_vec(), 0, 4096)]).is_err());

    let mut mux = FlacMux::new(Cursor::new(Vec::new()));
    let pcm = StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), RATE, 1, Vec::new())
        .expect("pcm");
    assert!(
        mux.add_stream(NewStream {
            params: pcm,
            timescale: Timescale::new(RATE),
            language: None
        })
        .is_err()
    );
    assert_eq!(mux.status(), MuxStatus::Failed);

    // A failed write stops the mux.
    let mut mux = FlacMux::new(Cursor::new(Vec::new()));
    mux.add_stream(stream(Vec::new())).expect("stream");
    assert!(
        mux.write_packet(&packet(b"junk".to_vec(), 0, 4096))
            .is_err()
    );
    assert!(mux.write_packet(&good).is_err());
    assert!(mux.finish(&duration(4096)).is_err());
}

#[test]
fn finish_checks_durations_and_config_supplies_the_signature() {
    let mut mux = FlacMux::new(Cursor::new(Vec::new()));
    mux.add_stream(stream(Vec::new())).expect("stream");
    let mut config = StreamInfo::parse(&info_body()).expect("info");
    config.md5 = [5; 16];
    config.total_samples = 8192;
    mux.set_config(TrackId::new(0), &config.to_bytes())
        .expect("config");
    let mut stereo = config;
    stereo.channels = 2;
    assert!(mux.set_config(TrackId::new(0), &stereo.to_bytes()).is_err());
    assert!(mux.set_config(TrackId::new(0), b"short").is_err());
    mux.write_packet(&packet(frame(0, 4096, false, 0), 0, 4096))
        .expect("0");
    mux.write_packet(&packet(frame(1, 4096, false, 0), 4096, 4096))
        .expect("1");
    let failed = mux.finish(&duration(4000)).expect_err("mismatch");
    assert!(!failed.is_complete());

    let mut mux = FlacMux::new(Cursor::new(Vec::new()));
    mux.add_stream(stream(config.to_bytes().to_vec()))
        .expect("stream");
    for index in 0..2u64 {
        mux.write_packet(&packet(frame(index, 4096, false, 0), index * 4096, 4096))
            .expect("write");
    }
    let done = mux.finish(&duration(8192)).expect("finish");
    assert_eq!(
        done.payload_bytes() as usize,
        2 * frame(0, 4096, false, 0).len()
    );
    let bytes = mux.into_inner().into_inner();
    let demux = open(&bytes).expect("open");
    assert_eq!(demux.stream_info().md5, [5; 16]);
    assert_eq!(demux.stream_info().total_samples, 8192);

    // A config whose sample count differs loses its signature.
    let mut mux = FlacMux::new(Cursor::new(Vec::new()));
    mux.add_stream(stream(config.to_bytes().to_vec()))
        .expect("stream");
    mux.write_packet(&packet(frame(0, 4096, false, 0), 0, 4096))
        .expect("write");
    mux.finish(&duration(4096)).expect("finish");
    let bytes = mux.into_inner().into_inner();
    assert_eq!(open(&bytes).expect("open").stream_info().md5, [0; 16]);
}

#[test]
fn a_cancelled_mux_leaves_a_playable_file_of_unknown_length() {
    let mut mux = FlacMux::new(Cursor::new(Vec::new()));
    mux.add_stream(stream(Vec::new())).expect("stream");
    mux.write_packet(&packet(frame(0, 4096, false, 0), 0, 4096))
        .expect("write");
    mux.cancel();
    assert_eq!(mux.status(), MuxStatus::Failed);
    assert!(mux.finish(&duration(4096)).is_err());
    let bytes = mux.into_inner().into_inner();
    let mut demux = open(&bytes).expect("open");
    assert_eq!(demux.stream_info().total_samples, 0);
    assert_eq!(read_all(&mut demux).len(), 1);
}

#[test]
fn an_empty_stream_finishes_as_a_header() {
    let mut mux = FlacMux::new(Cursor::new(Vec::new()));
    mux.add_stream(stream(info_body())).expect("stream");
    mux.finish(&duration(0)).expect("finish");
    let bytes = mux.into_inner().into_inner();
    let mut demux = open(&bytes).expect("open");
    assert!(demux.read_packet().expect("read").is_none());
}
