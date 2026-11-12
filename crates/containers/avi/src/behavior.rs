//! Behaviour on hand-built files, and on files this crate's muxer writes.

use std::io::Cursor;

use cutlass_codec::FrameSize;
use cutlass_codec::{
    CodecId, ContainerId, Packet, StreamCodec, StreamParams, Timescale, Timestamp,
};
use cutlass_container::{
    Demux, Landing, Limits, Mux, MuxError, MuxStatus, NewStream, OpenError, ReadError, SeekError,
    SeekTo, TrackDuration, TrackId,
};

use crate::{AviDemux, AviMux, AviMuxOptions, IndexSource};

// ---------------------------------------------------------------- builder

fn chunk(id: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
    out
}

fn list(form: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut payload = form.to_vec();
    payload.extend_from_slice(body);
    chunk(b"LIST", &payload)
}

fn le32(value: u32) -> [u8; 4] {
    value.to_le_bytes()
}

struct Spec {
    kind: [u8; 4],
    handler: [u8; 4],
    scale: u32,
    rate: u32,
    sample_size: u32,
    strf: Vec<u8>,
}

fn strh(spec: &Spec) -> Vec<u8> {
    let mut out = spec.kind.to_vec();
    out.extend_from_slice(&spec.handler);
    out.extend_from_slice(&[0; 12]);
    out.extend_from_slice(&le32(spec.scale));
    out.extend_from_slice(&le32(spec.rate));
    out.extend_from_slice(&[0; 8]);
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&le32(u32::MAX));
    out.extend_from_slice(&le32(spec.sample_size));
    out.extend_from_slice(&[0; 8]);
    out
}

fn bih(fourcc: &[u8; 4], width: u32, height: u32, extra: &[u8]) -> Vec<u8> {
    let mut out = le32(40 + u32::try_from(extra.len()).unwrap()).to_vec();
    out.extend_from_slice(&le32(width));
    out.extend_from_slice(&le32(height));
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&[0; 20]);
    out.extend_from_slice(extra);
    out
}

fn wfx(tag: u16, channels: u16, rate: u32, align: u16, bits: u16, extra: &[u8]) -> Vec<u8> {
    let mut out = tag.to_le_bytes().to_vec();
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&le32(rate));
    out.extend_from_slice(&le32(rate * u32::from(align)));
    out.extend_from_slice(&align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(&u16::try_from(extra.len()).unwrap().to_le_bytes());
    out.extend_from_slice(extra);
    out
}

fn mjpeg(scale: u32, rate: u32) -> Spec {
    Spec {
        kind: *b"vids",
        handler: *b"MJPG",
        scale,
        rate,
        sample_size: 0,
        strf: bih(b"MJPG", 64, 48, &[]),
    }
}

fn video(fourcc: &[u8; 4]) -> Spec {
    Spec {
        kind: *b"vids",
        handler: *fourcc,
        scale: 1,
        rate: 25,
        sample_size: 0,
        strf: bih(fourcc, 64, 48, &[]),
    }
}

fn pcm() -> Spec {
    Spec {
        kind: *b"auds",
        handler: [0; 4],
        scale: 2,
        rate: 16_000,
        sample_size: 2,
        strf: wfx(1, 1, 8_000, 2, 16, &[]),
    }
}

fn hdrl(specs: &[Spec]) -> Vec<u8> {
    let mut avih = vec![0u8; 56];
    avih[..4].copy_from_slice(&le32(40_000));
    avih[24..28].copy_from_slice(&le32(u32::try_from(specs.len()).unwrap()));
    let mut body = chunk(b"avih", &avih);
    for spec in specs {
        let mut strl = chunk(b"strh", &strh(spec));
        strl.extend_from_slice(&chunk(b"strf", &spec.strf));
        body.extend_from_slice(&list(b"strl", &strl));
    }
    list(b"hdrl", &body)
}

/// How a hand-built file stores `idx1`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Idx1 {
    Relative,
    Absolute,
    None,
}

struct Built {
    bytes: Vec<u8>,
    /// Payload position of each chunk, in `movi` order.
    data: Vec<u64>,
}

/// A file with `chunks` in `movi`. A chunk id of `LIST` is a `rec ` list
/// holding the chunks that follow it up to the next `LIST`.
fn avi(specs: &[Spec], chunks: &[(&[u8; 4], Vec<u8>, bool)], idx1: Idx1) -> Built {
    let header = hdrl(specs);
    let movi_at = 12 + header.len() + 8;
    let mut body = Vec::new();
    let mut rows = Vec::new();
    let mut data = Vec::new();
    for (id, payload, key) in chunks {
        let at = movi_at + 4 + body.len();
        data.push(u64::try_from(at + 8).unwrap());
        rows.push((**id, if *key { 0x10u32 } else { 0 }, at, payload.len()));
        body.extend_from_slice(&chunk(id, payload));
    }
    let mut out = b"RIFF\0\0\0\0AVI ".to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&list(b"movi", &body));
    if idx1 != Idx1::None {
        let mut index = Vec::new();
        for (id, flags, at, size) in rows {
            let offset = if idx1 == Idx1::Relative {
                at - movi_at
            } else {
                at
            };
            index.extend_from_slice(&id);
            index.extend_from_slice(&le32(flags));
            index.extend_from_slice(&le32(u32::try_from(offset).unwrap()));
            index.extend_from_slice(&le32(u32::try_from(size).unwrap()));
        }
        out.extend_from_slice(&chunk(b"idx1", &index));
    }
    let size = le32(u32::try_from(out.len() - 8).unwrap());
    out[4..8].copy_from_slice(&size);
    Built { bytes: out, data }
}

fn open(bytes: &[u8]) -> Result<AviDemux<Cursor<Vec<u8>>>, OpenError> {
    AviDemux::open(
        Cursor::new(bytes.to_vec()),
        u64::try_from(bytes.len()).unwrap(),
    )
}

fn drain(demux: &mut AviDemux<Cursor<Vec<u8>>>) -> Vec<Packet> {
    let mut out = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        out.push(packet);
    }
    out
}

fn jpeg(n: u8) -> Vec<u8> {
    vec![0xFF, 0xD8, n, 0xFF, 0xD9]
}

fn interleaved_mjpeg_pcm(idx1: Idx1) -> Built {
    avi(
        &[mjpeg(1, 25), pcm()],
        &[
            (b"00dc", jpeg(0), true),
            (b"01wb", vec![1; 640], true),
            (b"00dc", jpeg(1), true),
            (b"01wb", vec![2; 640], true),
            (b"00dc", jpeg(2), true),
        ],
        idx1,
    )
}

// ------------------------------------------------------------------ demux

#[test]
fn relative_and_absolute_idx1_give_the_same_packets() {
    for mode in [Idx1::Relative, Idx1::Absolute, Idx1::None] {
        let file = interleaved_mjpeg_pcm(mode);
        let mut demux = open(&file.bytes).expect("open");
        let expected = match mode {
            Idx1::None => IndexSource::Scan,
            _ => IndexSource::Idx1,
        };
        assert_eq!(demux.index_source(), expected);
        assert!(demux.interleaved() && !demux.truncated());
        assert_eq!(demux.family(), ContainerId::Avi);
        let streams = demux.streams();
        assert_eq!(streams.len(), 2);
        assert_eq!(
            streams[0].params().codec(),
            &StreamCodec::Known(CodecId::Mjpeg)
        );
        assert_eq!(streams[0].timescale(), Timescale::new(25));
        assert_eq!(streams[1].timescale(), Timescale::new(8_000));
        assert_eq!(streams[1].params().extradata().len(), 16);
        let packets = drain(&mut demux);
        let summary: Vec<_> = packets
            .iter()
            .map(|packet| {
                let stamp = packet.timestamp();
                (
                    packet.stream(),
                    stamp.decode(),
                    stamp.presentation(),
                    stamp.duration(),
                    packet.bytes().len(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            vec![
                (0, Some(0), Some(0), Some(1), 5),
                (1, Some(0), Some(0), Some(320), 640),
                (0, Some(1), Some(1), Some(1), 5),
                (1, Some(320), Some(320), Some(320), 640),
                (0, Some(2), Some(2), Some(1), 5),
            ]
        );
        assert!(packets.iter().all(Packet::keyframe));
        assert_eq!(demux.duration_ticks(0), Some(3));
        assert_eq!(demux.duration_ticks(1), Some(640));
    }
}

#[test]
fn a_scan_reads_keyframes_from_the_bitstream_and_skips_rec_lists() {
    let idr = vec![0, 0, 0, 1, 0x65, 0x88, 0x80];
    let slice = vec![0, 0, 0, 1, 0x41, 0x9A, 0x00];
    let header = hdrl(&[video(b"H264")]);
    let movi_at = 12 + header.len() + 8;
    let mut rec = b"rec ".to_vec();
    rec.extend_from_slice(&chunk(b"00dc", &idr));
    rec.extend_from_slice(&chunk(b"00dc", &slice));
    let mut body = chunk(b"LIST", &rec);
    body.extend_from_slice(&chunk(b"JUNK", &[0; 6]));
    body.extend_from_slice(&chunk(b"00dc", &slice));
    let mut bytes = b"RIFF\0\0\0\0AVI ".to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&list(b"movi", &body));
    let size = le32(u32::try_from(bytes.len() - 8).unwrap());
    bytes[4..8].copy_from_slice(&size);
    let mut demux = open(&bytes).expect("open");
    assert_eq!(demux.index_source(), IndexSource::Scan);
    let packets = drain(&mut demux);
    assert_eq!(packets.len(), 3);
    assert_eq!(
        packets.iter().map(Packet::keyframe).collect::<Vec<_>>(),
        [true, false, false]
    );
    assert_eq!(
        packets[0].timestamp().presentation(),
        None,
        "H.264 may reorder"
    );
    assert_eq!(packets[2].timestamp().decode(), Some(2));
    // The same file with an idx1 that lists the rec list itself.
    let mut index = Vec::new();
    let rec_at = movi_at + 4;
    for (id, flags, at, size) in [
        (*b"LIST", 0x01u32, rec_at, rec.len()),
        (*b"00dc", 0x10, rec_at + 12, idr.len()),
        (*b"00dc", 0, rec_at + 12 + 8 + idr.len() + 1, slice.len()),
    ] {
        index.extend_from_slice(&id);
        index.extend_from_slice(&le32(flags));
        index.extend_from_slice(&le32(u32::try_from(at - movi_at).unwrap()));
        index.extend_from_slice(&le32(u32::try_from(size).unwrap()));
    }
    let mut indexed = bytes.clone();
    indexed.extend_from_slice(&chunk(b"idx1", &index));
    let size = le32(u32::try_from(indexed.len() - 8).unwrap());
    indexed[4..8].copy_from_slice(&size);
    let mut demux = open(&indexed).expect("open");
    assert_eq!(demux.index_source(), IndexSource::Idx1);
    assert_eq!(
        drain(&mut demux).len(),
        2,
        "idx1 is the table when it is usable"
    );
}

#[test]
fn an_idx1_pointing_outside_movi_falls_back_to_a_scan() {
    let mut file = interleaved_mjpeg_pcm(Idx1::Relative);
    let at = file.bytes.len() - 16 * 5 + 8;
    let row = at + 16 * 2;
    file.bytes[row..row + 4].copy_from_slice(&le32(0x7FFF_0000));
    let mut demux = open(&file.bytes).expect("open");
    assert_eq!(demux.index_source(), IndexSource::Scan);
    assert_eq!(demux.fallbacks().len(), 1);
    assert_eq!(demux.fallbacks()[0].index, "idx1");
    assert_eq!(drain(&mut demux).len(), 5);
}

#[test]
fn a_cut_recording_reads_up_to_the_cut() {
    let file = interleaved_mjpeg_pcm(Idx1::None);
    let cut = usize::try_from(file.data[3]).unwrap() + 100;
    let mut demux = open(&file.bytes[..cut]).expect("open");
    assert!(demux.truncated());
    let packets = drain(&mut demux);
    assert_eq!(
        packets.len(),
        3,
        "the chunk the cut runs through is not a packet"
    );
}

#[test]
fn garbage_between_chunks_is_skipped_by_a_resync() {
    let header = hdrl(&[mjpeg(1, 25)]);
    let mut body = chunk(b"00dc", &jpeg(0));
    body.extend_from_slice(&[0xEE; 37]);
    body.extend_from_slice(&chunk(b"00dc", &jpeg(1)));
    let mut bytes = b"RIFF\0\0\0\0AVI ".to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&list(b"movi", &body));
    let size = le32(u32::try_from(bytes.len() - 8).unwrap());
    bytes[4..8].copy_from_slice(&size);
    let mut demux = open(&bytes).expect("open");
    assert!(demux.resyncs() >= 1);
    let packets = drain(&mut demux);
    assert_eq!(packets.len(), 2);
    assert_eq!(packets[1].bytes(), jpeg(1).as_slice());
}

#[test]
fn hostile_headers_are_errors_not_panics() {
    assert!(matches!(
        open(b"RIFF\x04\0\0\0WAVE"),
        Err(OpenError::Unrecognized { .. })
    ));
    assert!(matches!(
        open(b"RIFF\x04\0\0\0AVIX"),
        Err(OpenError::UnsupportedBrand { .. })
    ));
    assert!(matches!(
        open(b"RIFX\x04\0\0\0AVI "),
        Err(OpenError::UnsupportedBrand { .. })
    ));
    assert!(open(b"RIFF").is_err());
    assert!(open(b"RIFF\x04\0\0\0AVI ").is_err(), "no hdrl");
    // An hdrl list larger than its RIFF.
    let mut file = interleaved_mjpeg_pcm(Idx1::Relative).bytes;
    file[16..20].copy_from_slice(&le32(0xFFFF_FF00));
    assert!(open(&file).is_err());
    // More streams than the limit.
    let specs: Vec<Spec> = (0..70).map(|_| mjpeg(1, 25)).collect();
    let many = avi(&specs, &[(b"00dc", jpeg(0), true)], Idx1::None);
    assert!(matches!(
        open(&many.bytes),
        Err(OpenError::HostileSize { .. })
    ));
    // An index larger than the entry budget.
    let file = interleaved_mjpeg_pcm(Idx1::Relative);
    let mut limits = Limits::standard();
    limits.max_index_entries = 3;
    let len = u64::try_from(file.bytes.len()).unwrap();
    let result = AviDemux::open_with_limits(Cursor::new(file.bytes), len, limits);
    assert!(matches!(result, Err(OpenError::HostileSize { .. })));
}

#[test]
fn unknown_codecs_are_published_and_read() {
    let mut odd = video(b"ABCD");
    odd.strf = bih(b"ABCD", 8, 8, &[7, 7]);
    let audio = Spec {
        kind: *b"auds",
        handler: [0; 4],
        scale: 1,
        rate: 8_000,
        sample_size: 0,
        strf: wfx(0x1234, 1, 8_000, 1, 0, &[]),
    };
    let text = Spec {
        kind: *b"txts",
        handler: [0; 4],
        scale: 1,
        rate: 25,
        sample_size: 0,
        strf: Vec::new(),
    };
    let file = avi(
        &[odd, audio, text],
        &[
            (b"00dc", vec![1, 2, 3], true),
            (b"01wb", vec![4], true),
            (b"02tx", vec![5, 6], true),
        ],
        Idx1::Relative,
    );
    let mut demux = open(&file.bytes).expect("open");
    assert!(demux.streams()[0].params().codec().is_unknown());
    assert_eq!(demux.streams()[0].params().extradata(), &[7, 7]);
    assert!(demux.streams()[1].params().codec().is_unknown());
    assert!(demux.streams()[2].params().codec().is_unknown());
    let packets = drain(&mut demux);
    assert_eq!(packets.len(), 3);
    assert_eq!(packets[2].bytes(), &[5, 6]);
}

#[test]
fn an_empty_chunk_is_a_dropped_frame() {
    let file = avi(
        &[mjpeg(1, 25)],
        &[
            (b"00dc", jpeg(0), true),
            (b"00dc", Vec::new(), false),
            (b"00dc", jpeg(2), true),
        ],
        Idx1::Relative,
    );
    let mut demux = open(&file.bytes).expect("open");
    let packets = drain(&mut demux);
    assert_eq!(packets.len(), 2);
    assert_eq!(packets[1].timestamp().decode(), Some(2));
}

#[test]
fn variable_bitrate_mp3_counts_frames_by_block_align() {
    let audio = Spec {
        kind: *b"auds",
        handler: [0; 4],
        scale: 1152,
        rate: 48_000,
        sample_size: 0,
        strf: wfx(
            0x55,
            2,
            48_000,
            1152,
            0,
            &[1, 0, 2, 0, 0, 0, 0x80, 4, 1, 0, 0x71, 5],
        ),
    };
    let file = avi(
        &[audio],
        &[
            (b"00wb", vec![0; 400], true),
            (b"00wb", vec![0; 1200], true),
            (b"00wb", vec![0; 300], true),
        ],
        Idx1::Relative,
    );
    let mut demux = open(&file.bytes).expect("open");
    assert_eq!(demux.streams()[0].timescale(), Timescale::new(125));
    let times: Vec<_> = drain(&mut demux)
        .iter()
        .map(|packet| (packet.timestamp().decode(), packet.timestamp().duration()))
        .collect();
    assert_eq!(
        times,
        [(Some(0), Some(3)), (Some(3), Some(6)), (Some(9), Some(3))]
    );
}

/// MPEG-1 Layer III at 128 kbit/s and 48 kHz: 384 bytes, 1152 samples.
fn mp3_frame(fill: u8) -> Vec<u8> {
    let mut frame = vec![fill; 384];
    frame[..4].copy_from_slice(&[0xFF, 0xFB, 0x94, 0x00]);
    frame
}

#[test]
fn a_variable_bitrate_chunk_of_several_mp3_frames_is_cut_into_frames() {
    let audio = Spec {
        kind: *b"auds",
        handler: [0; 4],
        scale: 1152,
        rate: 48_000,
        sample_size: 0,
        strf: wfx(0x55, 2, 48_000, 1152, 0, &[]),
    };
    let mut two = mp3_frame(1);
    two.extend(mp3_frame(2));
    // Garbage, then a frame whose end is in the next chunk.
    let mut odd = vec![7; 10];
    odd.extend(mp3_frame(3));
    let split = odd.len() - 100;
    let tail = odd.split_off(split);
    let file = avi(
        &[audio],
        &[
            (b"00wb", two, true),
            (b"00wb", odd, true),
            (b"00wb", tail, true),
            (b"00wb", vec![9; 50], true),
        ],
        Idx1::Relative,
    );
    let mut demux = open(&file.bytes).expect("open");
    let packets = drain(&mut demux);
    let seen: Vec<_> = packets
        .iter()
        .map(|packet| {
            (
                packet.bytes().len(),
                packet.bytes()[4],
                packet.timestamp().decode(),
                packet.timestamp().duration(),
            )
        })
        .collect();
    // The clock is 125 Hz and a 1152-sample frame is 3 ticks. Each chunk
    // counts one unit, as the file's nBlockAlign says, so the frame that
    // starts in the second chunk takes that chunk's time, as FFmpeg gives
    // it; a chunk with no frame header is returned whole.
    assert_eq!(
        seen,
        [
            (384, 1, Some(0), Some(3)),
            (384, 2, Some(3), Some(3)),
            (384, 3, Some(3), Some(3)),
            (50, 9, Some(9), Some(3)),
        ]
    );
}

#[test]
fn a_non_interleaved_file_reads_in_time_order() {
    let mut chunks: Vec<(&[u8; 4], Vec<u8>, bool)> = Vec::new();
    for frame in 0..100u8 {
        chunks.push((b"00dc", jpeg(frame), true));
    }
    for _ in 0..4 {
        chunks.push((b"01wb", vec![0; 8_000], true));
    }
    let file = avi(&[mjpeg(1, 25), pcm()], &chunks, Idx1::Relative);
    let mut demux = open(&file.bytes).expect("open");
    assert!(!demux.interleaved());
    let packets = drain(&mut demux);
    assert_eq!(packets.len(), 104);
    assert_eq!(
        packets[1].stream(),
        1,
        "audio at time zero comes before the second frame"
    );
}

#[test]
fn seeks_land_from_the_index_without_reading_payloads() {
    let mut chunks: Vec<(&[u8; 4], Vec<u8>, bool)> = Vec::new();
    for frame in 0..10u8 {
        let fourcc: &[u8; 4] = b"00dc";
        let key = frame % 4 == 0;
        let nal = if key { 0x65 } else { 0x41 };
        chunks.push((fourcc, vec![0, 0, 0, 1, nal, frame], key));
        chunks.push((b"01wb", vec![frame; 640], true));
    }
    let file = avi(&[video(b"H264"), pcm()], &chunks, Idx1::Relative);
    let mut demux = open(&file.bytes).expect("open");
    let at = |time, end| SeekTo {
        track: 0,
        time,
        end,
    };
    let consumed = demux.consumed();
    let landing = demux.seek(at(6, None)).expect("seek");
    assert_eq!(landing.payload_scans, 0);
    assert!(landing.index_used);
    assert_eq!(demux.consumed(), consumed, "a seek reads nothing");
    let Landing::Packet {
        ordinal,
        time,
        keyframe,
        preroll,
        ..
    } = landing.landing
    else {
        panic!("landing");
    };
    assert_eq!((ordinal, time, keyframe, preroll), (4, 4, true, 2));
    let packets: Vec<_> = drain(&mut demux);
    assert_eq!(packets[0].stream(), 0);
    assert_eq!(packets[0].timestamp().decode(), Some(4));
    let audio: Vec<_> = packets
        .iter()
        .filter(|packet| packet.stream() == 1)
        .collect();
    assert_eq!(
        audio[0].timestamp().decode(),
        Some(4 * 320),
        "audio resumes at the landing time"
    );

    assert!(matches!(
        demux.seek(at(-5, None)).expect("before").landing,
        Landing::Packet { ordinal: 0, .. }
    ));
    assert!(matches!(
        demux.seek(at(9, None)).expect("last").landing,
        Landing::Packet { ordinal: 8, .. }
    ));
    assert_eq!(
        demux.seek(at(11, None)).expect("past").landing,
        Landing::End
    );
    assert!(demux.read_packet().expect("end").is_none());
    assert_eq!(
        demux.seek(at(10, Some(10))).expect("end").landing,
        Landing::End
    );
    assert!(matches!(
        demux.seek(SeekTo {
            track: 7,
            time: 0,
            end: None
        }),
        Err(ReadError::Seek(SeekError::UnknownTrack))
    ));
    demux.seek(at(0, None)).expect("start");
    assert_eq!(drain(&mut demux).len(), 20);
}

#[test]
fn cancel_stops_reads_and_seeks() {
    let file = interleaved_mjpeg_pcm(Idx1::Relative);
    let mut demux = open(&file.bytes).expect("open");
    demux.cancel();
    assert!(matches!(demux.read_packet(), Err(ReadError::Cancelled)));
    assert!(matches!(
        demux.seek(SeekTo {
            track: 0,
            time: 0,
            end: None
        }),
        Err(ReadError::Cancelled)
    ));
}

#[test]
fn a_chunk_the_index_misnames_is_an_error_and_the_next_read_goes_on() {
    let mut file = interleaved_mjpeg_pcm(Idx1::Relative);
    let at = usize::try_from(file.data[2]).unwrap() - 8;
    file.bytes[at..at + 4].copy_from_slice(b"JUNK");
    let mut demux = open(&file.bytes).expect("open");
    assert_eq!(demux.index_source(), IndexSource::Idx1);
    assert!(demux.read_packet().expect("first").is_some());
    assert!(demux.read_packet().expect("second").is_some());
    assert!(matches!(demux.read_packet(), Err(ReadError::Source { .. })));
    let rest = drain(&mut demux);
    assert_eq!(rest.len(), 2);
}

#[test]
fn a_packet_above_the_limit_is_refused_and_skipped() {
    let file = interleaved_mjpeg_pcm(Idx1::Relative);
    let mut limits = Limits::standard();
    limits.max_packet_bytes = 100;
    let len = u64::try_from(file.bytes.len()).unwrap();
    let mut demux = AviDemux::open_with_limits(Cursor::new(file.bytes), len, limits).expect("open");
    assert!(demux.read_packet().expect("jpeg").is_some());
    assert!(matches!(
        demux.read_packet(),
        Err(ReadError::HostileSize { .. })
    ));
    assert_eq!(
        demux.read_packet().expect("next").expect("jpeg").stream(),
        0
    );
}

// -------------------------------------------------------------------- mux

fn video_stream(codec: CodecId, extradata: Vec<u8>) -> NewStream {
    NewStream {
        params: StreamParams::video(
            StreamCodec::Known(codec),
            FrameSize::try_new(64, 48).unwrap(),
            extradata,
        ),
        timescale: Timescale::new(90_000),
        language: None,
    }
}

fn audio_stream(codec: CodecId, rate: u32, extradata: Vec<u8>) -> NewStream {
    NewStream {
        params: StreamParams::try_audio(StreamCodec::Known(codec), rate, 2, extradata).unwrap(),
        timescale: Timescale::new(rate),
        language: None,
    }
}

fn packet(stream: u32, bytes: Vec<u8>, time: i64, duration: i64, scale: u32, key: bool) -> Packet {
    Packet::try_new(
        stream,
        bytes,
        Timestamp::try_new(Some(time), Some(time), Some(duration)).unwrap(),
        Timescale::new(scale),
        key,
    )
    .unwrap()
}

fn duration(track: u32, scale: u32, ticks: i64) -> TrackDuration {
    TrackDuration {
        track: TrackId::new(track),
        timescale: Timescale::new(scale),
        ticks,
    }
}

/// H.264 at 25 fps on a 90 kHz clock and AAC at 48 kHz, written with
/// `riff_bytes` per segment.
fn write_movie(frames: u32, riff_bytes: u64) -> (Vec<u8>, u32) {
    let options = AviMuxOptions {
        riff_bytes,
        ..AviMuxOptions::default()
    };
    let mut mux = AviMux::with_options(Cursor::new(Vec::new()), options);
    let avcc = vec![1, 0x64, 0, 0x1F, 0xFF, 0xE0, 0, 0];
    mux.add_stream(video_stream(CodecId::H264, avcc)).unwrap();
    mux.add_stream(audio_stream(CodecId::Aac, 48_000, vec![0x11, 0x90]))
        .unwrap();
    let mut audio = 0i64;
    for frame in 0..frames {
        let time = i64::from(frame) * 3_600;
        let key = frame % 10 == 0;
        let size = 300 + usize::try_from(frame % 7).unwrap() * 50;
        let mut bytes = vec![u8::try_from(frame % 251).unwrap(); size];
        bytes[..5].copy_from_slice(&[0, 0, 0, 1, if key { 0x65 } else { 0x41 }]);
        mux.write_packet(&packet(0, bytes, time, 3_600, 90_000, key))
            .unwrap();
        while audio * 1024 * 90_000 / 48_000 <= time {
            mux.write_packet(&packet(1, vec![9; 200], audio * 1024, 1024, 48_000, true))
                .unwrap();
            audio += 1;
        }
    }
    mux.finish(&[
        duration(0, 90_000, i64::from(frames) * 3_600),
        duration(1, 48_000, audio * 1024),
    ])
    .expect("finish");
    let riffs = mux.riff_count();
    (mux.into_writer().unwrap().into_inner(), riffs)
}

#[test]
fn a_written_movie_reads_back_packet_for_packet() {
    let (bytes, riffs) = write_movie(50, 1 << 30);
    assert_eq!(riffs, 1);
    let mut demux = open(&bytes).expect("open");
    assert_eq!(demux.index_source(), IndexSource::OpenDml);
    assert_eq!(demux.streams()[0].timescale(), Timescale::new(25));
    assert_eq!(demux.streams()[0].params().extradata()[0], 1);
    assert_eq!(demux.streams()[1].params().extradata(), &[0x11, 0x90]);
    assert_eq!(demux.streams()[1].timescale(), Timescale::new(375));
    assert_eq!(demux.stream_header(0).unwrap().handler, *b"avc1");
    let packets = drain(&mut demux);
    let video: Vec<_> = packets
        .iter()
        .filter(|packet| packet.stream() == 0)
        .collect();
    assert_eq!(video.len(), 50);
    assert_eq!(video[13].timestamp().decode(), Some(13));
    assert_eq!(video.iter().filter(|packet| packet.keyframe()).count(), 5);
    assert_eq!(video[13].bytes().len(), 300 + 6 * 50);
    let audio: Vec<_> = packets
        .iter()
        .filter(|packet| packet.stream() == 1)
        .collect();
    assert_eq!(audio[3].timestamp().decode(), Some(24));
    assert_eq!(audio[3].timestamp().duration(), Some(8));
}

#[test]
fn a_long_movie_spans_riff_segments_and_every_index_agrees() {
    let (bytes, riffs) = write_movie(400, 32 * 1024);
    assert!(riffs > 3, "{riffs} segments");
    let mut demux = open(&bytes).expect("open");
    assert_eq!(demux.riff_count(), riffs);
    assert_eq!(demux.index_source(), IndexSource::OpenDml);
    assert_eq!(demux.odml_total_frames(), Some(400));
    let odml = drain(&mut demux);
    assert_eq!(
        odml.iter().filter(|packet| packet.stream() == 0).count(),
        400
    );
    // Without the super indexes: idx1 for the first RIFF, a scan after it.
    let mut plain = bytes.clone();
    let mut at = 0;
    while let Some(found) = plain[at..].windows(4).position(|window| window == b"indx") {
        plain[at + found..at + found + 4].copy_from_slice(b"JUNK");
        at += found + 4;
    }
    let mut demux = open(&plain).expect("open");
    assert_eq!(demux.index_source(), IndexSource::Idx1AndScan);
    let mixed = drain(&mut demux);
    assert_eq!(odml.len(), mixed.len());
    for (index, (left, right)) in odml.iter().zip(&mixed).enumerate() {
        assert!(
            left == right,
            "packet {index}: {:?} {:?} {} / {:?} {:?} {}",
            left.stream(),
            left.timestamp(),
            left.bytes().len(),
            right.stream(),
            right.timestamp(),
            right.bytes().len()
        );
    }
}

#[test]
fn durations_of_several_frames_write_dropped_frames() {
    let mut mux = AviMux::new(Cursor::new(Vec::new()));
    mux.add_stream(video_stream(CodecId::Mjpeg, Vec::new()))
        .unwrap();
    mux.write_packet(&packet(0, jpeg(0), 7_200, 3_600, 90_000, true))
        .unwrap();
    mux.write_packet(&packet(0, jpeg(1), 10_800, 10_800, 90_000, true))
        .unwrap();
    mux.write_packet(&packet(0, jpeg(2), 21_600, 3_600, 90_000, true))
        .unwrap();
    mux.finish(&[duration(0, 90_000, 18_000)]).expect("finish");
    let bytes = mux.into_writer().unwrap().into_inner();
    let mut demux = open(&bytes).expect("open");
    let times: Vec<_> = drain(&mut demux)
        .iter()
        .map(|packet| packet.timestamp().presentation())
        .collect();
    assert_eq!(
        times,
        [Some(2), Some(3), Some(6)],
        "dwStart holds the first two frames"
    );
    assert_eq!(demux.stream_header(0).unwrap().start, 2);
}

#[test]
fn pcm_is_written_at_a_constant_bitrate() {
    let mut mux = AviMux::new(Cursor::new(Vec::new()));
    mux.add_stream(audio_stream(CodecId::Pcm, 44_100, Vec::new()))
        .unwrap();
    mux.write_packet(&packet(0, vec![1; 4 * 1000], 0, 1000, 44_100, true))
        .unwrap();
    mux.write_packet(&packet(0, vec![2; 4 * 500], 1000, 500, 44_100, true))
        .unwrap();
    mux.finish(&[duration(0, 44_100, 1500)]).expect("finish");
    let bytes = mux.into_writer().unwrap().into_inner();
    let mut demux = open(&bytes).expect("open");
    let header = *demux.stream_header(0).unwrap();
    assert_eq!(
        (header.scale, header.rate, header.sample_size, header.length),
        (4, 176_400, 4, 1500)
    );
    let packets = drain(&mut demux);
    assert_eq!(packets[1].timestamp().decode(), Some(1000));
    assert_eq!(packets[1].timestamp().duration(), Some(500));
}

#[test]
fn the_muxer_refuses_what_avi_cannot_hold() {
    let refuse = |stream: NewStream| {
        let mut mux = AviMux::new(Cursor::new(Vec::new()));
        let error = mux.add_stream(stream).unwrap_err();
        assert_eq!(mux.status(), MuxStatus::Failed);
        error
    };
    let unknown = cutlass_codec::UnknownCodec::new(b"zzzz".to_vec()).unwrap();
    assert!(matches!(
        refuse(NewStream {
            params: StreamParams::opaque(StreamCodec::Unknown(unknown), Vec::new()),
            timescale: Timescale::new(25),
            language: None,
        }),
        MuxError::UnknownCodec(_)
    ));
    assert!(matches!(
        refuse(audio_stream(CodecId::Opus, 48_000, Vec::new())),
        MuxError::Rejected { .. }
    ));
    assert!(matches!(
        refuse(NewStream {
            params: StreamParams::opaque(StreamCodec::Known(CodecId::H264), Vec::new()),
            timescale: Timescale::new(25),
            language: None,
        }),
        MuxError::Rejected { .. }
    ));

    // A gap in PCM.
    let mut mux = AviMux::new(Cursor::new(Vec::new()));
    mux.add_stream(audio_stream(CodecId::Pcm, 8_000, Vec::new()))
        .unwrap();
    mux.write_packet(&packet(0, vec![0; 40], 0, 10, 8_000, true))
        .unwrap();
    assert!(
        mux.write_packet(&packet(0, vec![0; 40], 20, 10, 8_000, true))
            .is_err()
    );

    // AAC without its AudioSpecificConfig.
    let mut mux = AviMux::new(Cursor::new(Vec::new()));
    mux.add_stream(audio_stream(CodecId::Aac, 48_000, Vec::new()))
        .unwrap();
    assert!(
        mux.write_packet(&packet(0, vec![0; 10], 0, 1024, 48_000, true))
            .is_err()
    );

    // A packet on another clock, and a duration list that disagrees.
    let mut mux = AviMux::new(Cursor::new(Vec::new()));
    mux.add_stream(video_stream(CodecId::Mjpeg, Vec::new()))
        .unwrap();
    assert!(matches!(
        mux.write_packet(&packet(0, jpeg(0), 0, 1, 25, true)),
        Err(MuxError::TimescaleMismatch { .. })
    ));
    let mut mux = AviMux::new(Cursor::new(Vec::new()));
    mux.add_stream(video_stream(CodecId::Mjpeg, Vec::new()))
        .unwrap();
    mux.write_packet(&packet(0, jpeg(0), 0, 3_600, 90_000, true))
        .unwrap();
    let failed = mux.finish(&[duration(0, 90_000, 7_200)]).unwrap_err();
    assert!(matches!(
        failed.reason(),
        MuxError::DurationMismatch { track: 0 }
    ));

    // Cancel.
    let mut mux = AviMux::new(Cursor::new(Vec::new()));
    mux.add_stream(video_stream(CodecId::Mjpeg, Vec::new()))
        .unwrap();
    mux.cancel();
    assert!(matches!(
        mux.write_packet(&packet(0, jpeg(0), 0, 3_600, 90_000, true)),
        Err(MuxError::Cancelled)
    ));
    assert!(mux.finish(&[]).is_err());
}

#[test]
fn a_config_handed_over_with_the_first_packet_is_stored() {
    let mut mux = AviMux::new(Cursor::new(Vec::new()));
    let video = mux
        .add_stream(video_stream(CodecId::H264, Vec::new()))
        .unwrap();
    let audio = mux
        .add_stream(audio_stream(CodecId::Aac, 44_100, Vec::new()))
        .unwrap();
    mux.set_config(video, &[1, 2, 3]).unwrap();
    mux.write_packet(&packet(0, vec![0, 0, 0, 1, 0x65], 0, 3_600, 90_000, true))
        .unwrap();
    mux.set_config(audio, &[0x12, 0x10]).unwrap();
    mux.write_packet(&packet(1, vec![5; 9], 0, 1024, 44_100, true))
        .unwrap();
    for frame in 1..8 {
        mux.write_packet(&packet(
            0,
            vec![0, 0, 0, 1, 0x41],
            frame * 3_600,
            3_600,
            90_000,
            false,
        ))
        .unwrap();
    }
    assert!(
        mux.set_config(audio, &[0x12, 0x10, 0x56]).is_err(),
        "the header is written"
    );
    mux.set_config(audio, &[0x12, 0x08]).unwrap();
    mux.finish(&[duration(0, 90_000, 8 * 3_600), duration(1, 44_100, 1024)])
        .expect("finish");
    let bytes = mux.into_writer().unwrap().into_inner();
    let demux = open(&bytes).expect("open");
    assert_eq!(demux.streams()[0].params().extradata(), &[1, 2, 3]);
    assert_eq!(demux.streams()[1].params().extradata(), &[0x12, 0x08]);
}

#[test]
fn video_times_off_the_frame_grid_land_on_the_nearest_frame() {
    // 30 fps on a 1 kHz clock: 33 and 34 ms frames. The first packets set
    // the period to 33 ms, and every frame lands on the nearest one.
    let mut mux = AviMux::new(Cursor::new(Vec::new()));
    mux.add_stream(NewStream {
        params: StreamParams::video(
            StreamCodec::Known(CodecId::Mjpeg),
            FrameSize::try_new(16, 16).unwrap(),
            Vec::new(),
        ),
        timescale: Timescale::new(1_000),
        language: None,
    })
    .unwrap();
    let mut time = 0;
    let mut total = 0;
    for frame in 0..30u8 {
        let span = if frame % 3 == 2 { 34 } else { 33 };
        mux.write_packet(&packet(0, jpeg(frame), time, span, 1_000, true))
            .unwrap();
        time += span;
        total += span;
    }
    // A late frame after a pause of about three frames.
    mux.write_packet(&packet(0, jpeg(99), time + 100, 33, 1_000, true))
        .unwrap();
    total += 33;
    mux.finish(&[duration(0, 1_000, total)]).expect("finish");
    let bytes = mux.into_writer().unwrap().into_inner();
    let mut demux = open(&bytes).expect("open");
    assert_eq!(demux.streams()[0].timescale(), Timescale::new(1_000));
    let times: Vec<i64> = drain(&mut demux)
        .iter()
        .map(|packet| packet.timestamp().decode().unwrap())
        .collect();
    assert_eq!(times.len(), 31);
    for (frame, at) in times.iter().take(30).enumerate() {
        let exact = i64::try_from(frame).unwrap() * 1_000 / 30;
        assert!(
            (at - exact).abs() <= 17,
            "frame {frame} at {at}, exact {exact}"
        );
    }
    assert_eq!(times[30], 33 * 33, "the pause holds three empty chunks");
}
