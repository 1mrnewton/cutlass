//! Open, read, seek, and write behavior.

use std::io::Cursor;

use cutlass_codec::{
    CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp, UnknownCodec,
};
use cutlass_container::{
    Demux, Landing, Limits, Mux, MuxError, NewStream, OpenError, ReadError, SeekError, SeekTo,
    TrackDuration, TrackId,
};

use crate::rate::encode_rate;
use crate::{AiffDemux, AiffMux, FEATURES};

fn chunk(id: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(id);
    let size = u32::try_from(payload.len()).expect("chunk");
    out.extend(size.to_be_bytes());
    out.extend(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
    out
}

fn form(kind: &[u8; 4], chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut body = Vec::from(*kind);
    for chunk in chunks {
        body.extend(chunk);
    }
    let mut file = Vec::from(*b"FORM");
    let size = u32::try_from(body.len()).expect("form");
    file.extend(size.to_be_bytes());
    file.extend(body);
    file
}

fn comm(channels: u16, frames: u32, bits: u16, rate: u32, compression: Option<[u8; 4]>) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend(channels.to_be_bytes());
    payload.extend(frames.to_be_bytes());
    payload.extend(bits.to_be_bytes());
    payload.extend(encode_rate(rate).expect("rate"));
    if let Some(tag) = compression {
        payload.extend(tag);
    }
    chunk(b"COMM", &payload)
}

fn ssnd(offset: u32, block: u32, sound: &[u8]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend(offset.to_be_bytes());
    payload.extend(block.to_be_bytes());
    let junk = usize::try_from(offset).expect("offset");
    payload.extend(std::iter::repeat_n(0x5a, junk));
    payload.extend(sound);
    chunk(b"SSND", &payload)
}

fn pcm_file(sound: &[u8]) -> Vec<u8> {
    let frames = u32::try_from(sound.len() / 2).expect("frames");
    form(
        b"AIFF",
        &[comm(1, frames, 16, 44_100, None), ssnd(0, 0, sound)],
    )
}

fn open(bytes: &[u8]) -> AiffDemux<Cursor<Vec<u8>>> {
    let len = u64::try_from(bytes.len()).expect("len");
    AiffDemux::open(Cursor::new(bytes.to_vec()), len).expect("open")
}

fn open_err(bytes: &[u8]) -> OpenError {
    let len = u64::try_from(bytes.len()).expect("len");
    AiffDemux::open(Cursor::new(bytes.to_vec()), len).expect_err("error")
}

fn read_packets(demux: &mut AiffDemux<Cursor<Vec<u8>>>) -> Vec<Packet> {
    let mut packets = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        packets.push(packet);
    }
    assert!(demux.read_packet().expect("end").is_none());
    packets
}

fn mux_frames(extra: [u8; 3], channels: u16, rate: u32, frames: &[&[u8]]) -> Vec<u8> {
    let mut mux = AiffMux::new(Cursor::new(Vec::new()));
    add_pcm(&mut mux, extra, channels, rate);
    for (index, frame) in frames.iter().enumerate() {
        let pts = i64::try_from(index).expect("pts");
        write_frame(&mut mux, frame, pts, rate);
    }
    let ticks = i64::try_from(frames.len()).expect("ticks");
    finish_ok(&mut mux, rate, ticks);
    mux.into_inner().into_inner()
}

fn add_pcm(mux: &mut AiffMux<Cursor<Vec<u8>>>, extra: [u8; 3], channels: u16, rate: u32) {
    let params = StreamParams::try_audio(
        StreamCodec::Known(CodecId::Pcm),
        rate,
        channels,
        extra.to_vec(),
    )
    .expect("params");
    mux.add_stream(NewStream {
        params,
        timescale: Timescale::new(rate),
        language: None,
    })
    .expect("stream");
}

fn write_frame(mux: &mut AiffMux<Cursor<Vec<u8>>>, frame: &[u8], pts: i64, rate: u32) {
    let packet = Packet::try_new(
        0,
        frame.to_vec(),
        Timestamp::try_new(Some(pts), Some(pts), Some(1)).expect("time"),
        Timescale::new(rate),
        true,
    )
    .expect("packet");
    mux.write_packet(&packet).expect("write");
}

fn finish_ok(mux: &mut AiffMux<Cursor<Vec<u8>>>, rate: u32, ticks: i64) {
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(rate),
        ticks,
    }])
    .expect("finish");
}

#[test]
fn a_pcm_file_returns_one_frame_per_packet_in_sample_ticks() {
    let sound = [0x00, 0x01, 0xff, 0xfe];
    let mut demux = open(&pcm_file(&sound));
    assert_eq!(demux.family(), cutlass_codec::ContainerId::Aiff);
    assert!(demux.gaps().is_empty());
    assert!(!demux.is_aifc());
    assert_eq!(demux.sample_rate(), 44_100);
    assert_eq!(demux.channels(), 1);
    assert_eq!(demux.duration_samples(), 2);
    let stream = &demux.streams()[0];
    assert_eq!(stream.timescale().ticks_per_second(), 44_100);
    assert_eq!(stream.params().extradata(), &[1, 1, 0]);
    assert_eq!(stream.params().codec(), &StreamCodec::Known(CodecId::Pcm));
    let before = demux.consumed();
    let packets = read_packets(&mut demux);
    assert_eq!(packets.len(), 2);
    assert_eq!(packets[0].bytes(), &[0x00, 0x01]);
    assert_eq!(packets[1].bytes(), &[0xff, 0xfe]);
    assert_eq!(packets[0].timestamp().presentation(), Some(0));
    assert_eq!(packets[1].timestamp().presentation(), Some(1));
    assert_eq!(packets[0].timestamp().duration(), Some(1));
    assert!(packets[0].keyframe());
    assert_eq!(
        demux.consumed() - before,
        u64::try_from(sound.len()).expect("len")
    );
}

#[test]
fn samples_are_not_read_until_a_packet_is_asked_for() {
    let mut sound = vec![0u8; 256];
    sound[0] = 1;
    sound[1] = 2;
    let file = pcm_file(&sound);
    let mut demux = open(&file);
    let before = demux.consumed();
    assert!(
        before + u64::try_from(sound.len()).expect("len")
            <= u64::try_from(file.len()).expect("file")
    );
    let packet = demux.read_packet().expect("read").expect("frame");
    assert_eq!(packet.bytes(), &[1, 2]);
    assert_eq!(demux.consumed() - before, 2);
}

#[test]
fn comm_after_ssnd_still_opens_on_a_random_file() {
    let sound = [0x12, 0x34];
    let file = form(b"AIFF", &[ssnd(0, 0, &sound), comm(1, 1, 16, 8_000, None)]);
    let mut demux = open(&file);
    let packet = demux.read_packet().expect("read").expect("frame");
    assert_eq!(packet.bytes(), &sound);
    assert_eq!(packet.timestamp().presentation(), Some(0));
}

#[test]
fn an_odd_name_chunk_is_skipped_and_the_samples_remain() {
    let sound = [0x00, 0x10, 0x00, 0x20];
    let file = form(
        b"AIFF",
        &[
            comm(1, 2, 16, 44_100, None),
            chunk(b"NAME", b"A"),
            ssnd(0, 0, &sound),
        ],
    );
    let mut demux = open(&file);
    assert_eq!(demux.skipped_chunks(), 1);
    let packets = read_packets(&mut demux);
    assert_eq!(packets[0].bytes(), &[0x00, 0x10]);
    assert_eq!(packets[1].bytes(), &[0x00, 0x20]);
}

#[test]
fn ssnd_offset_is_skipped_and_block_size_is_not_a_frame() {
    let sound = [0x00, 0x01, 0x00, 0x02];
    let file = form(
        b"AIFF",
        &[comm(1, 2, 16, 16_000, None), ssnd(2, 512, &sound)],
    );
    let mut demux = open(&file);
    let packets = read_packets(&mut demux);
    assert_eq!(packets.len(), 2);
    assert_eq!(packets[0].bytes(), &[0x00, 0x01]);
    assert_eq!(packets[1].bytes(), &[0x00, 0x02]);
    assert!(!packets[0].bytes().contains(&0x5a));
}

#[test]
fn seek_lands_on_the_sample_without_reading_the_payload() {
    let mut demux = open(&pcm_file(&[0, 1, 0, 2, 0, 3]));
    let consumed = demux.consumed();
    let landing = demux
        .seek(SeekTo {
            track: 0,
            time: 1,
            end: None,
        })
        .expect("seek");
    assert_eq!(demux.consumed(), consumed);
    assert_eq!(landing.payload_scans, 0);
    assert!(landing.index_used);
    match landing.landing {
        Landing::Packet {
            ordinal,
            time,
            keyframe,
            preroll,
            ..
        } => {
            assert_eq!(ordinal, 1);
            assert_eq!(time, 1);
            assert!(keyframe);
            assert_eq!(preroll, 0);
        }
        Landing::End => panic!("middle landed past the end"),
    }
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(packet.bytes(), &[0, 2]);
    let last = demux
        .seek(SeekTo {
            track: 0,
            time: 2,
            end: None,
        })
        .expect("last");
    match last.landing {
        Landing::Packet { ordinal, time, .. } => {
            assert_eq!(ordinal, 2);
            assert_eq!(time, 2);
        }
        Landing::End => panic!("last frame was past the end"),
    }
    let end = demux
        .seek(SeekTo {
            track: 0,
            time: 3,
            end: None,
        })
        .expect("end");
    assert!(matches!(end.landing, Landing::End));
    assert!(demux.read_packet().expect("none").is_none());
    let first = demux
        .seek(SeekTo {
            track: 0,
            time: -4,
            end: None,
        })
        .expect("first");
    match first.landing {
        Landing::Packet { ordinal, time, .. } => {
            assert_eq!(ordinal, 0);
            assert_eq!(time, 0);
        }
        Landing::End => panic!("negative time left the track"),
    }
}

#[test]
fn a_failed_seek_does_not_move_the_next_packet() {
    let mut demux = open(&pcm_file(&[0, 1, 0, 2]));
    let _ = demux.read_packet().expect("first").expect("frame");
    let error = demux
        .seek(SeekTo {
            track: 4,
            time: 0,
            end: None,
        })
        .expect_err("track");
    assert!(matches!(error, ReadError::Seek(SeekError::UnknownTrack)));
    let next = demux.read_packet().expect("second").expect("frame");
    assert_eq!(next.bytes(), &[0, 2]);
}

#[test]
fn a_forward_reader_plays_in_order_and_refuses_seek() {
    let bytes = pcm_file(&[0, 1, 0, 2]);
    let mut demux = AiffDemux::open_forward(Cursor::new(bytes)).expect("forward");
    let error = demux
        .seek(SeekTo {
            track: 0,
            time: 0,
            end: None,
        })
        .expect_err("seek");
    assert!(matches!(error, ReadError::Seek(SeekError::ForwardOnly)));
    let packet = demux.read_packet().expect("read").expect("frame");
    assert_eq!(packet.bytes(), &[0, 1]);
}

#[test]
fn a_forward_reader_refuses_ssnd_before_comm() {
    let bytes = form(b"AIFF", &[ssnd(0, 0, &[0, 1]), comm(1, 1, 16, 8_000, None)]);
    let error = AiffDemux::open_forward(Cursor::new(bytes)).expect_err("order");
    assert!(error.to_string().contains("SSND before COMM"));
}

#[test]
fn a_forward_reader_skips_a_small_chunk_between_comm_and_ssnd() {
    let bytes = form(
        b"AIFF",
        &[
            comm(1, 1, 16, 8_000, None),
            chunk(b"NAME", b"hi"),
            ssnd(0, 0, &[0x0a, 0x0b]),
        ],
    );
    let mut demux = AiffDemux::open_forward(Cursor::new(bytes)).expect("forward");
    let packet = demux.read_packet().expect("read").expect("frame");
    assert_eq!(packet.bytes(), &[0x0a, 0x0b]);
}

#[test]
fn a_large_skipped_chunk_is_seeked_on_random_and_refused_forward() {
    let name = vec![b'a'; 80];
    let file = form(
        b"AIFF",
        &[
            comm(1, 1, 16, 8_000, None),
            chunk(b"NAME", &name),
            ssnd(0, 0, &[0, 1]),
        ],
    );
    let mut limits = Limits::standard();
    limits.max_header_bytes = 40;
    let len = u64::try_from(file.len()).expect("len");
    AiffDemux::open_limited(Cursor::new(file.clone()), len, limits).expect("random");
    let error = AiffDemux::open_forward_limited(Cursor::new(file), limits).expect_err("forward");
    assert!(matches!(error, OpenError::HostileSize { .. }));
}

#[test]
fn truncated_sound_and_a_form_past_the_file_fail_closed() {
    let short = form(b"AIFF", &[comm(1, 2, 16, 8_000, None), ssnd(0, 0, &[0, 1])]);
    assert_eq!(
        open_err(&short),
        OpenError::Truncated {
            expected: 4,
            got: 2
        }
    );
    let mut huge = pcm_file(&[0, 1]);
    huge[4..8].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(matches!(open_err(&huge), OpenError::HostileSize { .. }));
    let mut past = pcm_file(&[0, 1, 0, 2]);
    let size = u32::from_be_bytes([past[4], past[5], past[6], past[7]]);
    past[4..8].copy_from_slice(&(size - 4).to_be_bytes());
    assert!(matches!(open_err(&past), OpenError::HostileSize { .. }));
}

#[test]
fn a_short_file_an_unknown_form_and_a_zero_rate_are_refused() {
    assert!(matches!(
        open_err(b"FORM"),
        OpenError::Truncated { expected: 12, .. }
    ));
    assert!(matches!(
        open_err(b"FORM\x00\x00\x00\x00AIFF"),
        OpenError::Truncated {
            expected: 12,
            got: 8
        }
    ));
    assert!(matches!(
        open_err(b"RIFF\x00\x00\x00\x00WAVE"),
        OpenError::Unrecognized { .. }
    ));
    assert!(matches!(
        open_err(b"FORM\x00\x00\x00\x048SVX"),
        OpenError::Unrecognized { .. }
    ));
}

#[test]
fn duplicate_chunks_zero_channels_and_a_bad_width_are_refused() {
    let sound = [0u8, 1];
    let duplicate_comm = form(
        b"AIFF",
        &[
            comm(1, 1, 16, 8_000, None),
            comm(1, 1, 16, 8_000, None),
            ssnd(0, 0, &sound),
        ],
    );
    assert!(
        open_err(&duplicate_comm)
            .to_string()
            .contains("duplicate COMM")
    );
    let duplicate_ssnd = form(
        b"AIFF",
        &[
            comm(1, 1, 16, 8_000, None),
            ssnd(0, 0, &sound),
            ssnd(0, 0, &sound),
        ],
    );
    assert!(
        open_err(&duplicate_ssnd)
            .to_string()
            .contains("duplicate SSND")
    );
    let silent = form(b"AIFF", &[comm(0, 0, 16, 8_000, None), ssnd(0, 0, &[])]);
    assert!(open_err(&silent).to_string().contains("channel"));
    let wide = form(b"AIFF", &[comm(65, 0, 16, 8_000, None), ssnd(0, 0, &[])]);
    assert!(matches!(open_err(&wide), OpenError::HostileSize { .. }));
    let bits = form(b"AIFF", &[comm(1, 1, 12, 8_000, None), ssnd(0, 0, &[0, 1])]);
    assert!(open_err(&bits).to_string().contains("sample size"));
}

#[test]
fn too_many_chunks_are_refused() {
    let mut chunks = Vec::new();
    for _ in 0..4_095 {
        chunks.push(chunk(b"NAME", b""));
    }
    chunks.push(comm(1, 1, 16, 8_000, None));
    chunks.push(ssnd(0, 0, &[0, 1]));
    assert!(matches!(
        open_err(&form(b"AIFF", &chunks)),
        OpenError::HostileSize { .. }
    ));
}

#[test]
fn aifc_pcm_widths_keep_their_layout_bytes() {
    let s16 = [0x01, 0x02];
    let sowt = form(
        b"AIFC",
        &[comm(1, 1, 16, 8_000, Some(*b"sowt")), ssnd(0, 0, &s16)],
    );
    let mut demux = open(&sowt);
    assert!(demux.is_aifc());
    assert_eq!(demux.streams()[0].params().extradata(), &[1, 0, 0]);
    assert_eq!(read_packets(&mut demux)[0].bytes(), &s16);

    let bytes = [0x11, 0x22, 0x33];
    let in24 = form(
        b"AIFC",
        &[comm(1, 1, 24, 8_000, Some(*b"in24")), ssnd(0, 0, &bytes)],
    );
    assert_eq!(open(&in24).streams()[0].params().extradata(), &[2, 1, 0]);

    let float = 1.0f32.to_be_bytes();
    let fl32 = form(
        b"AIFC",
        &[comm(1, 1, 32, 8_000, Some(*b"fl32")), ssnd(0, 0, &float)],
    );
    let mut demux = open(&fl32);
    assert_eq!(demux.streams()[0].params().extradata(), &[4, 1, 0]);
    assert_eq!(read_packets(&mut demux)[0].bytes(), &float);

    let wide = 1.0f64.to_be_bytes();
    let fl64 = form(
        b"AIFC",
        &[comm(1, 1, 64, 8_000, Some(*b"fl64")), ssnd(0, 0, &wide)],
    );
    assert_eq!(open(&fl64).streams()[0].params().extradata(), &[5, 1, 0]);

    let eight = form(b"AIFF", &[comm(1, 1, 8, 8_000, None), ssnd(0, 0, &[0x80])]);
    assert_eq!(open(&eight).streams()[0].params().extradata(), &[6, 1, 0]);
}

#[test]
fn ulaw_and_ima4_are_packetized_without_being_decoded() {
    let ulaw = form(
        b"AIFC",
        &[
            comm(1, 2, 8, 8_000, Some(*b"ulaw")),
            ssnd(0, 0, &[0x11, 0x22]),
        ],
    );
    let mut demux = open(&ulaw);
    assert_eq!(
        demux.streams()[0].params().codec(),
        &StreamCodec::Unknown(UnknownCodec::new(b"ulaw").expect("tag"))
    );
    let packets = read_packets(&mut demux);
    assert_eq!(packets[0].bytes(), &[0x11]);
    assert_eq!(packets[1].bytes(), &[0x22]);
    assert_eq!(packets[1].timestamp().presentation(), Some(1));
    assert_eq!(packets[0].timestamp().duration(), Some(1));

    let block = vec![0x3c; 34];
    let ima = form(
        b"AIFC",
        &[comm(1, 64, 16, 44_100, Some(*b"ima4")), ssnd(0, 0, &block)],
    );
    let mut demux = open(&ima);
    let packets = read_packets(&mut demux);
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].bytes(), block.as_slice());
    assert_eq!(packets[0].timestamp().duration(), Some(64));
    assert_eq!(packets[0].timestamp().presentation(), Some(0));
}

#[test]
fn ima4_seek_uses_the_block_duration() {
    let sound = vec![0x11; 68];
    let file = form(
        b"AIFC",
        &[comm(1, 128, 16, 44_100, Some(*b"ima4")), ssnd(0, 0, &sound)],
    );
    let mut demux = open(&file);
    let early = demux
        .seek(SeekTo {
            track: 0,
            time: 63,
            end: None,
        })
        .expect("early");
    match early.landing {
        Landing::Packet { ordinal, time, .. } => {
            assert_eq!(ordinal, 0);
            assert_eq!(time, 0);
        }
        Landing::End => panic!("early block was past the end"),
    }
    let next = demux
        .seek(SeekTo {
            track: 0,
            time: 64,
            end: None,
        })
        .expect("next");
    match next.landing {
        Landing::Packet {
            ordinal,
            time,
            byte_position,
            ..
        } => {
            assert_eq!(ordinal, 1);
            assert_eq!(time, 64);
            assert_eq!(byte_position, early_byte(&early) + 34);
        }
        Landing::End => panic!("second block was past the end"),
    }
    let _ = early;
}

fn early_byte(landing: &cutlass_container::SeekLanding) -> u64 {
    match landing.landing {
        Landing::Packet { byte_position, .. } => byte_position,
        Landing::End => panic!("missing first block"),
    }
}

#[test]
fn mace_and_gsm_use_their_block_sizes() {
    let mac3 = form(
        b"AIFC",
        &[
            comm(1, 6, 8, 8_000, Some(*b"MAC3")),
            ssnd(0, 0, &[0xaa, 0xbb]),
        ],
    );
    let mut demux = open(&mac3);
    let packet = read_packets(&mut demux).pop().expect("packet");
    assert_eq!(packet.bytes(), &[0xaa, 0xbb]);
    assert_eq!(packet.timestamp().duration(), Some(6));

    let mac6 = form(
        b"AIFC",
        &[comm(1, 6, 8, 8_000, Some(*b"MAC6")), ssnd(0, 0, &[0xcc])],
    );
    let mut demux = open(&mac6);
    let packet = read_packets(&mut demux).pop().expect("packet");
    assert_eq!(packet.bytes(), &[0xcc]);
    assert_eq!(packet.timestamp().duration(), Some(6));

    let gsm = vec![0x7e; 33];
    let file = form(
        b"AIFC",
        &[comm(1, 160, 0, 8_000, Some(*b"GSM ")), ssnd(0, 0, &gsm)],
    );
    let mut demux = open(&file);
    let packet = read_packets(&mut demux).pop().expect("packet");
    assert_eq!(packet.bytes(), gsm.as_slice());
    assert_eq!(packet.timestamp().duration(), Some(160));
}

#[test]
fn an_unpacketized_compression_is_named_and_refused() {
    let file = form(
        b"AIFC",
        &[comm(1, 1, 16, 8_000, Some(*b"QDM2")), ssnd(0, 0, &[0, 1])],
    );
    let error = open_err(&file);
    assert!(error.to_string().contains("QDM2"));
    let partial = form(
        b"AIFC",
        &[comm(1, 63, 16, 8_000, Some(*b"ima4")), ssnd(0, 0, &[0; 34])],
    );
    assert!(open_err(&partial).to_string().contains("whole number"));
}

#[test]
fn cancel_stops_the_next_read() {
    let mut demux = open(&pcm_file(&[0, 1]));
    demux.cancel();
    assert!(matches!(demux.read_packet(), Err(ReadError::Cancelled)));
}

#[test]
fn a_mux_round_trip_keeps_big_endian_pcm_as_aiff() {
    let frames: [&[u8]; 2] = [&[0x00, 0x0a], &[0x00, 0x0b]];
    let bytes = mux_frames([1, 1, 0], 1, 48_000, &frames);
    assert_eq!(&bytes[0..4], b"FORM");
    assert_eq!(&bytes[8..12], b"AIFF");
    assert!(!bytes.windows(4).any(|window| window == b"sowt"));
    let mut demux = open(&bytes);
    assert!(!demux.is_aifc());
    assert_eq!(demux.streams()[0].params().extradata(), &[1, 1, 0]);
    let packets = read_packets(&mut demux);
    assert_eq!(packets[0].bytes(), frames[0]);
    assert_eq!(packets[1].bytes(), frames[1]);
    assert_eq!(packets[1].timestamp().presentation(), Some(1));
}

#[test]
fn sowt_and_float_are_written_as_aifc() {
    let sowt = mux_frames([1, 0, 0], 1, 48_000, &[&[0x01, 0x00]]);
    assert_eq!(&sowt[8..12], b"AIFC");
    assert!(sowt.windows(4).any(|window| window == b"sowt"));
    assert!(sowt.windows(4).any(|window| window == b"FVER"));
    let mut demux = open(&sowt);
    assert_eq!(demux.skipped_chunks(), 1);
    assert_eq!(demux.streams()[0].params().extradata(), &[1, 0, 0]);
    assert_eq!(read_packets(&mut demux)[0].bytes(), &[0x01, 0x00]);

    let sample = 1.0f32.to_be_bytes();
    let float = mux_frames([4, 1, 0], 1, 48_000, &[&sample]);
    assert!(float.windows(4).any(|window| window == b"fl32"));
    assert_eq!(open(&float).streams()[0].params().extradata(), &[4, 1, 0]);

    let wide = 1.0f64.to_be_bytes();
    let fl64 = mux_frames([5, 1, 0], 1, 48_000, &[&wide]);
    assert!(fl64.windows(4).any(|window| window == b"fl64"));
}

#[test]
fn an_empty_mux_opens_and_has_no_packets() {
    let bytes = mux_frames([1, 1, 0], 1, 48_000, &[]);
    let mut demux = open(&bytes);
    assert_eq!(demux.duration_samples(), 0);
    assert!(demux.read_packet().expect("read").is_none());
    let error = demux
        .seek(SeekTo {
            track: 0,
            time: 0,
            end: None,
        })
        .expect_err("empty");
    assert!(matches!(error, ReadError::Seek(SeekError::Empty)));
}

#[test]
fn a_duration_mismatch_does_not_write_a_file() {
    let mut mux = AiffMux::new(Cursor::new(Vec::new()));
    add_pcm(&mut mux, [1, 1, 0], 1, 48_000);
    write_frame(&mut mux, &[0, 1], 0, 48_000);
    let failed = mux
        .finish(&[TrackDuration {
            track: TrackId::new(0),
            timescale: Timescale::new(48_000),
            ticks: 5,
        }])
        .expect_err("duration");
    assert!(matches!(
        failed.reason(),
        MuxError::DurationMismatch { track: 0 }
    ));
    assert!(failed.durations().is_none());
    assert!(mux.writer().get_ref().is_empty());
}

#[test]
fn mux_rejects_a_bad_stream_before_any_bytes() {
    let mut mux = AiffMux::new(Cursor::new(Vec::new()));
    let params =
        StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 48_000, 1, vec![4, 0, 0])
            .expect("params");
    let error = mux
        .add_stream(NewStream {
            params,
            timescale: Timescale::new(48_000),
            language: None,
        })
        .expect_err("endian");
    assert!(error.to_string().contains("big-endian"));

    let mut mux = AiffMux::new(Cursor::new(Vec::new()));
    let params = StreamParams::try_audio(StreamCodec::Known(CodecId::Flac), 48_000, 1, vec![])
        .expect("flac");
    let error = mux
        .add_stream(NewStream {
            params,
            timescale: Timescale::new(48_000),
            language: None,
        })
        .expect_err("flac");
    assert!(matches!(error, MuxError::Rejected { .. }));

    let mut mux = AiffMux::new(Cursor::new(Vec::new()));
    let unknown = UnknownCodec::new(b"ulaw").expect("tag");
    let params =
        StreamParams::try_audio(StreamCodec::Unknown(unknown), 48_000, 1, vec![]).expect("unknown");
    let error = mux
        .add_stream(NewStream {
            params,
            timescale: Timescale::new(48_000),
            language: None,
        })
        .expect_err("unknown");
    assert!(matches!(error, MuxError::UnknownCodec(_)));
}

#[test]
fn a_short_packet_and_a_second_finish_fail() {
    let mut mux = AiffMux::new(Cursor::new(Vec::new()));
    add_pcm(&mut mux, [1, 1, 0], 1, 8_000);
    let packet = Packet::try_new(
        0,
        vec![0x00],
        Timestamp::try_new(Some(0), Some(0), Some(1)).expect("time"),
        Timescale::new(8_000),
        true,
    )
    .expect("packet");
    let error = mux.write_packet(&packet).expect_err("short");
    assert!(error.to_string().contains("one sample frame"));
    assert!(mux.writer().get_ref().is_empty());

    let bytes = mux_frames([1, 1, 0], 2, 8_000, &[&[0, 1, 0, 2]]);
    let mut demux = open(&bytes);
    assert_eq!(demux.channels(), 2);
    assert_eq!(read_packets(&mut demux)[0].bytes().len(), 4);

    let mut mux = AiffMux::new(Cursor::new(Vec::new()));
    add_pcm(&mut mux, [1, 1, 0], 1, 8_000);
    let duration = [TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(8_000),
        ticks: 0,
    }];
    mux.finish(&duration).expect("first");
    let again = mux.finish(&duration).expect_err("second");
    assert!(matches!(again.reason(), MuxError::AlreadyFinished));
}

#[test]
fn cancel_and_a_negative_duration_write_nothing() {
    let mut mux = AiffMux::new(Cursor::new(Vec::new()));
    add_pcm(&mut mux, [1, 1, 0], 1, 8_000);
    mux.cancel();
    let failed = mux
        .finish(&[TrackDuration {
            track: TrackId::new(0),
            timescale: Timescale::new(8_000),
            ticks: 0,
        }])
        .expect_err("cancel");
    assert!(matches!(failed.reason(), MuxError::Cancelled));
    assert!(mux.writer().get_ref().is_empty());

    let mut mux = AiffMux::new(Cursor::new(Vec::new()));
    add_pcm(&mut mux, [1, 1, 0], 1, 8_000);
    let failed = mux
        .finish(&[TrackDuration {
            track: TrackId::new(0),
            timescale: Timescale::new(8_000),
            ticks: -1,
        }])
        .expect_err("negative");
    assert!(matches!(failed.reason(), MuxError::NegativeDuration));
    assert!(mux.writer().get_ref().is_empty());
}

#[test]
fn mutated_bytes_fail_closed() {
    let original = pcm_file(&[0, 1, 0, 2, 0, 3, 0, 4]);
    let mut bytes = original.clone();
    for index in 0..original.len() {
        bytes[index] ^= 0xa5;
        let len = u64::try_from(bytes.len()).expect("len");
        if let Ok(mut demux) = AiffDemux::open(Cursor::new(bytes.clone()), len) {
            while let Ok(Some(_)) = demux.read_packet() {}
            let _ = demux.seek(SeekTo {
                track: 0,
                time: 1,
                end: None,
            });
        }
        let _ = AiffDemux::open(Cursor::new(bytes.clone()), 4);
        bytes[index] = original[index];
    }
    for len in 0..original.len() {
        let _ = AiffDemux::open(
            Cursor::new(original[..len].to_vec()),
            u64::try_from(len).expect("prefix"),
        );
    }
}

#[test]
fn the_support_matrix_names_each_row_once() {
    let mut names = Vec::new();
    for feature in FEATURES {
        assert!(!names.contains(&feature.name), "{}", feature.name);
        names.push(feature.name);
    }
    let row = |name: &str| {
        FEATURES
            .iter()
            .find(|feature| feature.name == name)
            .expect(name)
    };
    assert!(row("aiff").demux && row("aiff").mux);
    assert!(row("aifc").demux && row("aifc").mux);
    assert!(row("pcm-integer-be").demux && row("pcm-integer-be").mux);
    assert!(row("pcm-integer-le").demux && row("pcm-integer-le").mux);
    assert!(row("pcm-float").demux && row("pcm-float").mux);
    assert!(row("pcm-8").demux && row("pcm-8").mux);
    assert!(row("alaw-ulaw").demux && !row("alaw-ulaw").mux);
    assert!(row("ima4").demux && !row("ima4").mux);
    assert!(row("mace").demux && !row("mace").mux);
    assert!(row("gsm").demux && !row("gsm").mux);
    assert!(!row("other-compression").demux && !row("other-compression").mux);
    assert!(row("ssnd-offset").demux && !row("ssnd-offset").mux);
    assert!(row("unknown-chunk").demux && !row("unknown-chunk").mux);
}
