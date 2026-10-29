//! Behavior locked to the support matrix.

use std::io::Cursor;

use cutlass_codec::{
    CodecId, ContainerId, Packet, StreamCodec, StreamParams, Timescale, Timestamp,
};
use cutlass_container::{
    Demux, FailedMux, IndexPlace, Landing, Mux, MuxError, MuxMode, NewStream, OpenError, ReadError,
    SeekError, SeekTo, TrackDuration, TrackId,
};

use crate::{CHANNEL_MASK_OFFSET, MAX_KEPT_CHUNK, WavDemux, WavMux};

fn pcm_fmt(channels: u16, rate: u32, bits: u16) -> Vec<u8> {
    let align = (bits / 8) * channels;
    let avg = rate * u32::from(align);
    let mut out = Vec::new();
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&avg.to_le_bytes());
    out.extend_from_slice(&align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out
}

fn chunk(id: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    chunk_sized(id, u32::try_from(payload.len()).expect("size"), payload)
}

fn chunk_sized(id: &[u8; 4], size: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(id);
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(payload);
    if !payload.len().is_multiple_of(2) {
        out.push(0);
    }
    out
}

fn riff(body: &[u8]) -> Vec<u8> {
    let mut out = b"RIFF".to_vec();
    let size = u32::try_from(4 + body.len()).expect("riff");
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(body);
    out
}

fn open(bytes: &[u8]) -> Result<WavDemux<Cursor<Vec<u8>>>, OpenError> {
    let len = u64::try_from(bytes.len()).expect("len");
    WavDemux::open(Cursor::new(bytes.to_vec()), len)
}

fn must_open(bytes: &[u8]) -> WavDemux<Cursor<Vec<u8>>> {
    open(bytes).expect("open")
}

fn must_err(bytes: &[u8]) -> OpenError {
    match open(bytes) {
        Ok(_) => panic!("opened a file that should fail"),
        Err(error) => error,
    }
}

fn read_all(demux: &mut WavDemux<Cursor<Vec<u8>>>) -> Vec<Packet> {
    let mut packets = Vec::new();
    while let Some(packet) = demux.read_packet().expect("packet") {
        packets.push(packet);
    }
    packets
}

fn stamp(rate: u32, bytes: Vec<u8>, at: i64, duration: i64) -> Packet {
    let timestamp = Timestamp::try_new(Some(at), Some(at), Some(duration)).expect("time");
    Packet::try_new(0, bytes, timestamp, Timescale::new(rate), true).expect("packet")
}

fn track_duration(rate: u32, ticks: i64) -> TrackDuration {
    TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(rate),
        ticks,
    }
}

fn stereo_file(frames: usize) -> (Vec<u8>, Vec<u8>) {
    let mut data = Vec::new();
    for frame in 0..frames {
        let sample = u16::try_from(frame + 1).expect("sample");
        data.extend_from_slice(&sample.to_le_bytes());
        data.extend_from_slice(&sample.to_le_bytes());
    }
    let body = [
        chunk(b"fmt ", &pcm_fmt(2, 44_100, 16)),
        chunk(b"data", &data),
    ]
    .concat();
    (riff(&body), data)
}

fn seek_at(time: i64, end: Option<i64>) -> SeekTo {
    SeekTo {
        track: 0,
        time,
        end,
    }
}

fn packet_at(landing: Landing) -> (u32, i64) {
    match landing {
        Landing::Packet { ordinal, time, .. } => (ordinal, time),
        Landing::End => panic!("end"),
    }
}

#[test]
fn a_stereo_pcm_file_is_one_frame_per_packet_at_the_sample_rate() {
    let (file, data) = stereo_file(4);
    let mut demux = must_open(&file);
    assert_eq!(demux.family(), ContainerId::Riff);
    assert_eq!(demux.format().tag(), 1);
    assert_eq!(demux.format().sample_rate(), 44_100);
    assert_ne!(demux.streams()[0].timescale().ticks_per_second(), 1_000_000);
    let packets = read_all(&mut demux);
    assert_eq!(packets.len(), 4);
    let mut gathered = Vec::new();
    for (index, packet) in packets.iter().enumerate() {
        assert_eq!(packet.bytes().len(), 4);
        assert_eq!(
            packet.timestamp().presentation(),
            Some(i64::try_from(index).expect("i"))
        );
        assert_eq!(packet.timestamp().duration(), Some(1));
        assert_eq!(packet.timescale().ticks_per_second(), 44_100);
        assert!(packet.keyframe());
        gathered.extend_from_slice(packet.bytes());
    }
    assert_eq!(gathered, data);
    assert_eq!(demux.duration_samples(), Some(4));
}

#[test]
fn one_frame_and_many_frames_both_open() {
    let (short, _) = stereo_file(1);
    let (longer, _) = stereo_file(4);
    assert_eq!(read_all(&mut must_open(&short)).len(), 1);
    assert_eq!(read_all(&mut must_open(&longer)).len(), 4);
}

#[test]
fn an_extensible_mask_reaches_stream_extradata() {
    let mut fmt = pcm_fmt(2, 48_000, 16);
    fmt[0] = 0xFE;
    fmt[1] = 0xFF;
    fmt.extend_from_slice(&22u16.to_le_bytes());
    fmt.extend_from_slice(&16u16.to_le_bytes());
    let mask = 0x0000_0003u32;
    fmt.extend_from_slice(&mask.to_le_bytes());
    fmt.extend_from_slice(&1u32.to_le_bytes());
    fmt.extend_from_slice(&0x0010_0000u32.to_le_bytes());
    fmt.extend_from_slice(&0xAA00_0080u32.to_le_bytes());
    fmt.extend_from_slice(&0x719B_3800u32.to_le_bytes());
    let data = [1u8, 0, 2, 0];
    let file = riff(&[chunk(b"fmt ", &fmt), chunk(b"data", &data)].concat());
    let demux = must_open(&file);
    assert!(demux.format().extensible());
    assert_eq!(demux.format().channel_mask(), mask);
    let extra = demux.streams()[0].params().extradata();
    assert_eq!(CHANNEL_MASK_OFFSET, 8);
    assert_eq!(&extra[8..12], &mask.to_le_bytes());
}

#[test]
fn an_unknown_subformat_is_named_on_the_stream() {
    let mut fmt = pcm_fmt(1, 8_000, 8);
    fmt[0] = 0xFE;
    fmt[1] = 0xFF;
    fmt.extend_from_slice(&22u16.to_le_bytes());
    fmt.extend_from_slice(&8u16.to_le_bytes());
    fmt.extend_from_slice(&0u32.to_le_bytes());
    fmt.extend_from_slice(&[0xAB; 16]);
    let file = riff(&[chunk(b"fmt ", &fmt), chunk(b"data", &[0])].concat());
    let demux = must_open(&file);
    let StreamCodec::Unknown(tag) = demux.format().codec() else {
        panic!("known");
    };
    assert!(tag.tag().starts_with(b"wav-guid:"));
    assert_eq!(read_all(&mut must_open(&file)).len(), 1);
}

#[test]
fn float_pcm_uses_the_data_chunk_when_fact_disagrees() {
    let mut fmt = pcm_fmt(1, 48_000, 32);
    fmt[0] = 3;
    fmt.extend_from_slice(&0u16.to_le_bytes());
    let data = [0u8; 8];
    let fact = 1u32.to_le_bytes();
    let file = riff(
        &[
            chunk(b"fmt ", &fmt),
            chunk(b"fact", &fact),
            chunk(b"data", &data),
        ]
        .concat(),
    );
    let demux = must_open(&file);
    assert!(demux.format().is_float());
    assert_eq!(demux.format().codec(), &StreamCodec::Known(CodecId::Pcm));
    assert_eq!(demux.duration_samples(), Some(2));
}

#[test]
fn rf64_uses_ds64_and_rejects_a_disagreement() {
    let honored = rf64_pcm(&[0u8; 8], u32::MAX, 8, 2);
    let demux = must_open(&honored);
    assert_eq!(demux.brand(), crate::Brand::Rf64);
    assert_eq!(demux.duration_samples(), Some(2));

    let bad_size = rf64_pcm(&[0u8; 4], 4, 8, 1);
    let error = must_err(&bad_size);
    assert!(
        error
            .to_string()
            .contains("ds64 data size disagrees with the data chunk")
    );
    assert!(!error.streams_published());

    let bad_samples = rf64_pcm(&[0u8; 8], 8, 8, 9);
    let error = must_err(&bad_samples);
    assert!(
        error
            .to_string()
            .contains("ds64 sample count disagrees with the data chunk")
    );
}

fn rf64_pcm(data: &[u8], data_field: u32, ds_data: u64, ds_samples: u64) -> Vec<u8> {
    let mut ds = vec![0u8; 28];
    ds[8..16].copy_from_slice(&ds_data.to_le_bytes());
    ds[16..24].copy_from_slice(&ds_samples.to_le_bytes());
    let mut body = chunk(b"ds64", &ds);
    body.extend(chunk(b"fmt ", &pcm_fmt(2, 44_100, 16)));
    body.extend(chunk_sized(b"data", data_field, data));
    let riff_size = u64::try_from(4 + body.len()).expect("riff");
    body[8..16].copy_from_slice(&riff_size.to_le_bytes());
    let mut file = b"RF64".to_vec();
    file.extend_from_slice(&u32::MAX.to_le_bytes());
    file.extend_from_slice(b"WAVE");
    file.extend(body);
    file
}

#[test]
fn an_odd_chunk_pad_is_not_part_of_the_next_header() {
    let (bare, data) = stereo_file(1);
    let junk = chunk(b"JUNK", &[0xAB]);
    let mut body = junk;
    body.extend(&bare[12..]);
    let file = riff(&body);
    let demux = must_open(&file);
    assert_eq!(demux.kept()[0].payload(), &[0xAB]);
    assert_eq!(read_all(&mut must_open(&file))[0].bytes(), data.as_slice());
}

#[test]
fn a_small_unknown_chunk_is_kept_and_a_large_one_is_named() {
    let (bare, _) = stereo_file(1);
    let mut kept_body = chunk(b"JUNK", b"abcd");
    kept_body.extend(&bare[12..]);
    let kept = must_open(&riff(&kept_body));
    assert_eq!(kept.kept()[0].id(), *b"JUNK");
    assert_eq!(kept.kept()[0].payload(), b"abcd");
    assert!(kept.dropped().is_empty());

    let big = vec![0u8; usize::try_from(MAX_KEPT_CHUNK).expect("cap") + 1];
    let mut dropped_body = chunk(b"JUNK", &big);
    dropped_body.extend(&bare[12..]);
    let dropped = must_open(&riff(&dropped_body));
    assert!(dropped.kept().is_empty());
    assert_eq!(dropped.dropped()[0].id(), *b"JUNK");
    assert_eq!(dropped.dropped()[0].declared(), MAX_KEPT_CHUNK + 1);
}

#[test]
fn hostile_files_fail_closed() {
    let short = must_err(&[0; 8]);
    assert!(matches!(short, OpenError::Truncated { .. }));
    assert!(!short.streams_published());

    let mut body = chunk(b"fmt ", &pcm_fmt(1, 8_000, 8));
    body.extend(chunk_sized(b"data", 100, &[0; 4]));
    let overrun = must_err(&riff(&body));
    assert!(matches!(overrun, OpenError::HostileSize { .. }));

    let mut list = b"INFO".to_vec();
    list.extend(b"JUNK");
    list.extend_from_slice(&1_000u32.to_le_bytes());
    let nested = must_err(&riff(&chunk(b"LIST", &list)));
    assert!(matches!(nested, OpenError::HostileSize { .. }));

    let bomb = must_err(&riff(&nested_lists(12)));
    assert!(matches!(bomb, OpenError::HostileSize { .. }));
    assert!(!bomb.streams_published());
}

fn nested_lists(depth: usize) -> Vec<u8> {
    let mut inner = chunk(b"JUNK", &[]);
    for _ in 0..depth {
        let mut payload = b"INFO".to_vec();
        payload.extend(inner);
        inner = chunk(b"LIST", &payload);
    }
    inner
}

#[test]
fn pcm_seek_lands_on_the_frame_without_reading_the_payload() {
    let frames = 250_000usize;
    let data = vec![0u8; frames * 4];
    let file = riff(
        &[
            chunk(b"fmt ", &pcm_fmt(2, 16_000, 16)),
            chunk(b"data", &data),
        ]
        .concat(),
    );
    let mut demux = must_open(&file);
    let before = demux.consumed();
    assert!(before < 256, "{before}");
    let first = demux
        .seek(seek_at(-1, Some(i64::try_from(frames).expect("n"))))
        .expect("first");
    assert_eq!(packet_at(first.landing), (0, 0));
    assert!(first.index_used);
    assert_eq!(first.payload_scans, 0);
    let middle = frames / 2;
    let mid = demux
        .seek(seek_at(i64::try_from(middle).expect("mid"), None))
        .expect("mid");
    assert_eq!(
        packet_at(mid.landing),
        (
            u32::try_from(middle).expect("ord"),
            i64::try_from(middle).expect("t")
        )
    );
    assert_eq!(demux.consumed(), before);
    let last = frames - 1;
    let end = demux
        .seek(seek_at(i64::try_from(last).expect("last"), None))
        .expect("last");
    assert_eq!(packet_at(end.landing).0, u32::try_from(last).expect("ord"));
    let past = demux
        .seek(seek_at(i64::try_from(frames).expect("end"), None))
        .expect("past");
    assert!(matches!(past.landing, Landing::End));
    assert_eq!(past.payload_scans, 0);
    let mid_again = demux.seek(seek_at(2, None)).expect("back");
    let packet = demux.read_packet().expect("read").expect("frame");
    assert_eq!(packet.timestamp().presentation(), Some(2));
    assert_eq!(packet_at(mid_again.landing).0, 2);
}

#[test]
fn a_forward_reader_plays_and_refuses_seek() {
    let (file, _) = stereo_file(2);
    let mut demux = WavDemux::open_forward(Cursor::new(file)).expect("forward");
    assert_eq!(read_all(&mut demux).len(), 2);
    let error = demux.seek(seek_at(0, None)).expect_err("seek");
    assert!(matches!(error, ReadError::Seek(SeekError::ForwardOnly)));
}

#[test]
fn data_before_fmt_opens_on_random_access_only() {
    let data = [1u8, 0, 2, 0];
    let body = [
        chunk(b"data", &data),
        chunk(b"fmt ", &pcm_fmt(2, 44_100, 16)),
    ]
    .concat();
    let file = riff(&body);
    assert_eq!(read_all(&mut must_open(&file))[0].bytes(), &data);
    let error = match WavDemux::open_forward(Cursor::new(file)) {
        Ok(_) => panic!("forward opened data before fmt"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("missing fmt"));
}

#[test]
fn cancel_and_an_empty_data_chunk_do_not_invent_packets() {
    let (file, _) = stereo_file(1);
    let mut demux = must_open(&file);
    demux.cancel();
    assert!(matches!(demux.read_packet(), Err(ReadError::Cancelled)));
    assert!(matches!(
        demux.seek(seek_at(0, None)),
        Err(ReadError::Cancelled)
    ));

    let empty = riff(&[chunk(b"fmt ", &pcm_fmt(1, 8_000, 8)), chunk(b"data", &[])].concat());
    let mut demux = must_open(&empty);
    assert!(demux.read_packet().expect("read").is_none());
    let error = demux.seek(seek_at(0, None)).expect_err("empty");
    assert!(matches!(error, ReadError::Seek(SeekError::Empty)));
}

#[test]
fn mpeg_in_wav_is_one_frame_and_not_pcm() {
    let frame = mpeg_frame();
    let file = riff(&[chunk(b"fmt ", &mp3_fmt()), chunk(b"data", &frame)].concat());
    let mut demux = must_open(&file);
    assert_eq!(demux.format().codec(), &StreamCodec::Known(CodecId::Mp3));
    let packets = read_all(&mut demux);
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].bytes(), frame.as_slice());
    assert_eq!(packets[0].timestamp().duration(), Some(1152));
    assert_eq!(packets[0].timescale().ticks_per_second(), 44_100);
    let mut demux = must_open(&file);
    let landing = demux.seek(seek_at(0, None)).expect("seek");
    assert!(!landing.index_used);
    assert!(landing.payload_scans >= 1);
    let past = demux.seek(seek_at(1152, None)).expect("end");
    assert!(matches!(past.landing, Landing::End));
}

fn mpeg_frame() -> Vec<u8> {
    let mut bits = 0u32;
    bits |= 0x7FF << 21;
    bits |= 0b11 << 19;
    bits |= 0b01 << 17;
    bits |= 1 << 16;
    bits |= 9 << 12;
    let mut frame = bits.to_be_bytes().to_vec();
    frame.resize(417, 0);
    frame
}

fn mp3_fmt() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&0x0055u16.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&44_100u32.to_le_bytes());
    out.extend_from_slice(&16_000u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

#[test]
fn ima_adpcm_is_one_block_with_the_block_duration() {
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&0x0011u16.to_le_bytes());
    fmt.extend_from_slice(&1u16.to_le_bytes());
    fmt.extend_from_slice(&8_000u32.to_le_bytes());
    fmt.extend_from_slice(&4_000u32.to_le_bytes());
    fmt.extend_from_slice(&256u16.to_le_bytes());
    fmt.extend_from_slice(&4u16.to_le_bytes());
    fmt.extend_from_slice(&0u16.to_le_bytes());
    let file = riff(&[chunk(b"fmt ", &fmt), chunk(b"data", &[0u8; 256])].concat());
    let mut demux = must_open(&file);
    let StreamCodec::Unknown(tag) = demux.format().codec() else {
        panic!("pcm");
    };
    assert_eq!(tag.tag(), b"wav:0011");
    let packets = read_all(&mut demux);
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].bytes().len(), 256);
    assert_eq!(packets[0].timestamp().duration(), Some(505));
}

#[test]
fn rifx_is_the_family_and_is_not_opened() {
    let mut header = b"RIFX".to_vec();
    header.extend_from_slice(&4u32.to_le_bytes());
    header.extend_from_slice(b"WAVE");
    let error = must_err(&header);
    assert!(matches!(
        error,
        OpenError::UnsupportedBrand {
            family: ContainerId::Riff
        }
    ));
}

#[test]
fn a_named_unknown_tag_demuxes_and_will_not_mux() {
    let mut fmt = pcm_fmt(1, 8_000, 8);
    fmt[0] = 0x99;
    fmt[1] = 0x99;
    fmt.extend_from_slice(&0u16.to_le_bytes());
    let file = riff(
        &[
            chunk(b"fmt ", &fmt),
            chunk(b"fact", &4u32.to_le_bytes()),
            chunk(b"data", &[0; 4]),
        ]
        .concat(),
    );
    let demux = must_open(&file);
    let StreamCodec::Unknown(tag) = demux.streams()[0].params().codec() else {
        panic!("known");
    };
    assert_eq!(tag.tag(), b"wav:9999");
    let mut mux = WavMux::new(Cursor::new(Vec::new()));
    let error = mux.add_stream(NewStream {
        params: demux.streams()[0].params().clone(),
        timescale: Timescale::new(8_000),
        language: None,
    });
    assert!(error.is_err());
}

#[test]
fn fact_and_ds64_sample_counts_must_agree_for_compressed_audio() {
    let mut fmt = pcm_fmt(1, 8_000, 8);
    fmt[0] = 0xFF;
    fmt[1] = 0x00;
    fmt.extend_from_slice(&0u16.to_le_bytes());
    let mut ds = vec![0u8; 28];
    ds[8..16].copy_from_slice(&4u64.to_le_bytes());
    ds[16..24].copy_from_slice(&11u64.to_le_bytes());
    let mut body = chunk(b"ds64", &ds);
    body.extend(chunk(b"fmt ", &fmt));
    body.extend(chunk(b"fact", &10u32.to_le_bytes()));
    body.extend(chunk(b"data", &[0; 4]));
    let riff_size = u64::try_from(4 + body.len()).expect("riff");
    body[8..16].copy_from_slice(&riff_size.to_le_bytes());
    let mut file = b"RF64".to_vec();
    file.extend_from_slice(&u32::MAX.to_le_bytes());
    file.extend_from_slice(b"WAVE");
    file.extend(body);
    let error = must_err(&file);
    assert!(
        error
            .to_string()
            .contains("fact sample count disagrees with ds64")
    );
}

#[test]
fn mux_round_trips_pcm_without_fact_and_float_with_fact() {
    let (file, data) = stereo_file(4);
    let out = remux(&file, false);
    assert!(!out.windows(4).any(|window| window == b"fact"));
    let mut again = must_open(&out);
    let mut gathered = Vec::new();
    for packet in read_all(&mut again) {
        gathered.extend_from_slice(packet.bytes());
    }
    assert_eq!(gathered, data);

    let mut fmt = pcm_fmt(1, 48_000, 32);
    fmt[0] = 3;
    fmt.extend_from_slice(&0u16.to_le_bytes());
    let floats = [0u8; 8];
    let source = riff(&[chunk(b"fmt ", &fmt), chunk(b"data", &floats)].concat());
    let written = remux(&source, false);
    assert!(written.windows(4).any(|window| window == b"fact"));
    let demux = must_open(&written);
    assert!(demux.format().is_float());
    assert_eq!(demux.duration_samples(), Some(2));
}

fn remux(file: &[u8], rf64: bool) -> Vec<u8> {
    let demux = must_open(file);
    let params = demux.streams()[0].params().clone();
    let rate = demux.format().sample_rate();
    let mut source = must_open(file);
    let packets = read_all(&mut source);
    let ticks = packets
        .iter()
        .filter_map(|packet| packet.timestamp().duration())
        .sum();
    let mut mux = if rf64 {
        WavMux::rf64(Cursor::new(Vec::new()))
    } else {
        WavMux::new(Cursor::new(Vec::new()))
    };
    mux.add_stream(NewStream {
        params,
        timescale: Timescale::new(rate),
        language: None,
    })
    .expect("stream");
    for packet in &packets {
        mux.write_packet(packet).expect("write");
    }
    let done = mux.finish(&[track_duration(rate, ticks)]).expect("finish");
    assert!(done.is_complete());
    assert_eq!(done.index(), IndexPlace::Absent);
    assert_eq!(done.mode(), MuxMode::Progressive);
    assert!(done.interleaved());
    mux.into_inner().into_inner()
}

#[test]
fn high_rate_stereo_stays_classic_and_a_mask_becomes_extensible() {
    let params = StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 96_000, 2, Vec::new())
        .expect("pcm");
    let bytes = write_pcm(params, 96_000, vec![stamp(96_000, vec![0, 0, 0, 0], 0, 1)]);
    let demux = must_open(&bytes);
    assert_eq!(demux.format().tag(), 1);
    assert!(!demux.format().extensible());
    assert!(!bytes.windows(4).any(|window| window == b"fact"));

    let mut fmt = pcm_fmt(2, 48_000, 16);
    fmt[0] = 0xFE;
    fmt[1] = 0xFF;
    fmt.extend_from_slice(&22u16.to_le_bytes());
    fmt.extend_from_slice(&16u16.to_le_bytes());
    fmt.extend_from_slice(&3u32.to_le_bytes());
    fmt.extend_from_slice(&1u32.to_le_bytes());
    fmt.extend_from_slice(&0x0010_0000u32.to_le_bytes());
    fmt.extend_from_slice(&0xAA00_0080u32.to_le_bytes());
    fmt.extend_from_slice(&0x719B_3800u32.to_le_bytes());
    let source = riff(&[chunk(b"fmt ", &fmt), chunk(b"data", &[0, 0, 0, 0])].concat());
    let written = remux(&source, false);
    let demux = must_open(&written);
    assert!(demux.format().extensible());
    assert_eq!(demux.format().channel_mask(), 3);
    assert!(written.windows(4).any(|window| window == b"fact"));
}

fn write_pcm(params: StreamParams, rate: u32, packets: Vec<Packet>) -> Vec<u8> {
    let ticks = packets
        .iter()
        .filter_map(|packet| packet.timestamp().duration())
        .sum();
    let mut mux = WavMux::new(Cursor::new(Vec::new()));
    mux.add_stream(NewStream {
        params,
        timescale: Timescale::new(rate),
        language: None,
    })
    .expect("stream");
    for packet in &packets {
        mux.write_packet(packet).expect("write");
    }
    mux.finish(&[track_duration(rate, ticks)]).expect("finish");
    mux.into_inner().into_inner()
}

#[test]
fn a_kept_chunk_survives_mux_and_a_duration_mismatch_writes_nothing() {
    let (file, _) = stereo_file(1);
    let mut body = chunk(b"JUNK", &[0xAB]);
    body.extend(&file[12..]);
    let source = riff(&body);
    let demux = must_open(&source);
    let kept = demux.kept()[0].payload().to_vec();
    let params = demux.streams()[0].params().clone();
    let mut packets_src = must_open(&source);
    let packets = read_all(&mut packets_src);
    let mut mux = WavMux::new(Cursor::new(Vec::new()));
    mux.keep_chunk(*b"JUNK", kept).expect("keep");
    mux.add_stream(NewStream {
        params,
        timescale: Timescale::new(44_100),
        language: None,
    })
    .expect("stream");
    for packet in &packets {
        mux.write_packet(packet).expect("write");
    }
    mux.finish(&[track_duration(44_100, 1)]).expect("finish");
    let out = mux.into_inner().into_inner();
    let again = must_open(&out);
    assert_eq!(again.kept()[0].payload(), &[0xAB]);

    let mut mux = WavMux::new(Cursor::new(Vec::new()));
    let params = StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 44_100, 2, Vec::new())
        .expect("pcm");
    mux.add_stream(NewStream {
        params,
        timescale: Timescale::new(44_100),
        language: None,
    })
    .expect("stream");
    mux.write_packet(&stamp(44_100, vec![0, 0, 0, 0], 0, 1))
        .expect("packet");
    let failed: FailedMux = mux
        .finish(&[track_duration(44_100, 5)])
        .expect_err("mismatch");
    assert!(!failed.is_complete());
    assert!(failed.durations().is_none());
    assert!(mux.into_inner().into_inner().is_empty());
}

#[test]
fn rf64_mux_is_readable_and_rejected_streams_do_not_finish() {
    let (file, data) = stereo_file(2);
    let written = remux(&file, true);
    assert!(written.starts_with(b"RF64"));
    assert!(written.windows(4).any(|window| window == b"ds64"));
    let mut demux = must_open(&written);
    let mut gathered = Vec::new();
    for packet in read_all(&mut demux) {
        gathered.extend_from_slice(packet.bytes());
    }
    assert_eq!(gathered, data);

    let mut mux = WavMux::new(Cursor::new(Vec::new()));
    let video = StreamParams::try_audio(StreamCodec::Known(CodecId::H264), 44_100, 2, Vec::new())
        .expect("audio params");
    let error = mux.add_stream(NewStream {
        params: video,
        timescale: Timescale::new(44_100),
        language: None,
    });
    assert!(matches!(error, Err(MuxError::Rejected { .. })));

    let mut mux = WavMux::new(Cursor::new(Vec::new()));
    let pcm = StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 44_100, 1, Vec::new())
        .expect("pcm");
    mux.add_stream(NewStream {
        params: pcm.clone(),
        timescale: Timescale::new(44_100),
        language: None,
    })
    .expect("first");
    let error = mux.add_stream(NewStream {
        params: pcm,
        timescale: Timescale::new(44_100),
        language: None,
    });
    assert!(matches!(error, Err(MuxError::TooManyTracks { .. })));
}

#[test]
fn a_language_tag_is_not_written() {
    let params = StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 8_000, 1, Vec::new())
        .expect("pcm");
    let mut mux = WavMux::new(Cursor::new(Vec::new()));
    mux.add_stream(NewStream {
        params,
        timescale: Timescale::new(8_000),
        language: Some(cutlass_container::Language::new("en").expect("lang")),
    })
    .expect("stream");
    mux.write_packet(&stamp(8_000, vec![0, 0], 0, 1))
        .expect("packet");
    mux.finish(&[track_duration(8_000, 1)]).expect("finish");
    let bytes = mux.into_inner().into_inner();
    let demux = must_open(&bytes);
    assert!(demux.streams()[0].language().is_none());
}
