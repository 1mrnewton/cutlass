//! MP4 and MOV files built in the test, with no external tool.

use std::num::NonZeroU32;

use cutlass_core::{ColorTags, Timestamp, VideoCodec};
use cutlass_mp4::{AudioCodec, Error, Fourcc, Kind, Movie, Rotation};

fn scale(value: u32) -> NonZeroU32 {
    match NonZeroU32::new(value) {
        Some(scale) => scale,
        None => panic!("timescale is non-zero"),
    }
}

fn time(value: i64) -> Timestamp {
    Timestamp::new(value, scale(1000))
}

fn expect_err(result: Result<Movie, Error>, expected: Error) {
    match result {
        Err(error) if error == expected => {}
        Err(error) => panic!("expected {expected}, got {error}"),
        Ok(_) => panic!("expected {expected}"),
    }
}

#[test]
fn a_movie_reports_tracks_samples_and_media_time() {
    let movie = Movie::parse(&video_file(false)).expect("video file");
    assert_eq!(movie.timescale(), scale(600));
    assert_eq!(movie.duration(), 1000);
    assert_eq!(movie.tracks().len(), 2);

    let video = &movie.tracks()[0];
    assert_eq!(video.id(), 7);
    assert!(video.enabled());
    assert_eq!(video.timescale(), scale(1000));
    assert_eq!(video.duration(), 400);
    assert_eq!(video.rotation(), Rotation::Zero);
    assert_eq!(video.display_width(), 320);
    assert_eq!(video.display_height(), 180);
    assert_eq!(video.frame_duration(), None);
    let Kind::Video(config) = video.kind() else {
        panic!("video track");
    };
    assert_eq!(config.codec, VideoCodec::H264);
    assert_eq!(config.width, 320);
    assert_eq!(config.height, 180);
    assert_eq!(config.config_record, [1, 0x42, 0x00]);
    assert_eq!(video.color(), None);

    let samples = video.samples();
    assert_eq!(samples.len(), 3);
    assert_eq!(samples[0].data, [1, 2, 3]);
    assert_eq!(samples[1].data, [4, 5]);
    assert_eq!(samples[2].data, [6]);
    assert_eq!(samples[0].dts, Some(time(0)));
    assert_eq!(samples[1].dts, Some(time(100)));
    assert_eq!(samples[2].dts, Some(time(200)));
    assert_eq!(samples[0].pts, time(200));
    assert_eq!(samples[1].pts, time(100));
    assert_eq!(samples[2].pts, time(200));
    assert_eq!(samples[0].duration, Some(time(100)));
    assert_eq!(samples[2].duration, Some(time(200)));
    assert!(samples[0].keyframe);
    assert!(!samples[1].keyframe);
    assert!(!samples[2].keyframe);
    // The edit list is stored and does not move the samples.
    assert_eq!(video.edits().len(), 1);
    assert_eq!(video.edits()[0].segment_duration(), 50);
    assert_eq!(video.edits()[0].media_time(), 100);
    assert_eq!(video.edits()[0].media_rate(), 0x0001_0000);

    match movie.tracks()[1].kind() {
        Kind::Other(code) => assert_eq!(code.bytes(), *b"tmcd"),
        Kind::Video(_) | Kind::Audio(_) => panic!("timecode track"),
    }
    assert!(movie.tracks()[1].samples().is_empty());
}

#[test]
fn an_nclx_box_reports_pq_as_hdr() {
    let movie = Movie::parse(&hdr_file()).expect("hdr file");
    let color = movie.tracks()[0].color().expect("color tags");
    assert_eq!(
        color,
        ColorTags {
            primaries: 9,
            transfer: 16,
            matrix: 9,
            full_range: Some(false),
        }
    );
    assert!(color.is_hdr());
}

#[test]
fn a_clockwise_matrix_is_ninety_degrees() {
    let movie = Movie::parse(&video_file(true)).expect("rotated file");
    let video = &movie.tracks()[0];
    assert_eq!(video.rotation(), Rotation::Clockwise90);
    assert_eq!(video.display_width(), 180);
    assert_eq!(video.display_height(), 320);
    let Kind::Video(config) = video.kind() else {
        panic!("video track");
    };
    assert_eq!((config.width, config.height), (320, 180));
}

#[test]
fn a_quicktime_brand_uses_the_same_reader() {
    let mut file = video_file(false);
    file[8..12].copy_from_slice(b"qt  ");
    let movie = Movie::parse(&file).expect("mov file");
    assert_eq!(movie.tracks()[0].samples().len(), 3);
}

#[test]
fn version_one_headers_and_wide_offsets_round_trip() {
    let movie = Movie::parse(&wide_file()).expect("wide file");
    let video = &movie.tracks()[0];
    assert_eq!(video.duration(), 90_000);
    assert_eq!(video.frame_duration(), Some(512));
    assert_eq!(video.samples().len(), 1);
    assert_eq!(video.samples()[0].data, [9]);
    assert!(video.samples()[0].keyframe);
}

#[test]
fn audio_keeps_its_rate_channels_and_decoder_config() {
    let movie = Movie::parse(&audio_file(&mp4a(48_000, 2, &[0x12, 0x10]))).expect("audio file");
    let audio = &movie.tracks()[0];
    assert_eq!(audio.frame_duration(), Some(1024));
    let Kind::Audio(described) = audio.kind() else {
        panic!("audio track");
    };
    assert_eq!(described.codec(), AudioCodec::Aac);
    assert_eq!(described.sample_rate(), 48_000);
    assert_eq!(described.channels(), 2);
    assert_eq!(described.config_record(), [0x12, 0x10]);
    assert_eq!(audio.samples().len(), 2);
    assert!(audio.samples().iter().all(|sample| sample.keyframe));
}

#[test]
fn quicktime_version_two_audio_uses_the_float_rate() {
    let movie = Movie::parse(&audio_file(&mp4a_v2(48_000.0, 2, &[0x12, 0x10]))).expect("version 2");
    let Kind::Audio(described) = movie.tracks()[0].kind() else {
        panic!("audio track");
    };
    assert_eq!(described.sample_rate(), 48_000);
    assert_eq!(described.channels(), 2);
    assert_eq!(described.config_record(), [0x12, 0x10]);
}

#[test]
fn an_elementary_stream_descriptor_inside_wave_is_read() {
    let movie = Movie::parse(&audio_file(&mp4a_wave(48_000, 2, &[0x12, 0x10]))).expect("wave");
    let Kind::Audio(described) = movie.tracks()[0].kind() else {
        panic!("audio track");
    };
    assert_eq!(described.config_record(), [0x12, 0x10]);
}

#[test]
fn compact_sizes_pack_two_samples_in_a_byte() {
    let movie = Movie::parse(&nibble_file()).expect("compact sizes");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples[0].data, [9, 9]);
    assert_eq!(samples[1].data, [8]);
}

#[test]
fn a_file_without_a_movie_is_rejected() {
    expect_err(
        Movie::parse(&atom(b"ftyp", b"isom")),
        Error::MissingBox(Fourcc::new(*b"moov")),
    );
}

#[test]
fn a_short_box_is_rejected() {
    expect_err(
        Movie::parse(&[0, 0, 0, 2, b'f', b't', b'y', b'p']),
        Error::BadBoxSize {
            kind: Fourcc::new(*b"ftyp"),
            size: 2,
        },
    );
}

#[test]
fn a_fragment_is_rejected() {
    let mut file = video_file(false);
    file.extend(atom(b"moof", &[]));
    expect_err(Movie::parse(&file), Error::Fragmented);
}

#[test]
fn a_zero_timescale_is_rejected() {
    expect_err(Movie::parse(&movie_of(&bare_trak(0))), Error::ZeroTimescale);
}

#[test]
fn sample_tables_must_agree() {
    expect_err(
        Movie::parse(&mismatched_counts()),
        Error::Inconsistent("decoding time entries do not cover every sample"),
    );
}

#[test]
fn a_sample_past_the_end_is_rejected() {
    expect_err(
        Movie::parse(&sample_past_end()),
        Error::Inconsistent("a sample extends past the end of the file"),
    );
}

fn video_file(clockwise90: bool) -> Vec<u8> {
    let samples = [1u8, 2, 3, 4, 5, 6];
    let ftyp = atom(b"ftyp", b"isom\0\0\0\0isom");
    let mdat = atom(b"mdat", &samples);
    let payload_at = u32::try_from(ftyp.len() + 8).expect("offset fits");
    let mut file = ftyp;
    file.extend(mdat);
    file.extend(video_moov(payload_at, clockwise90));
    file
}

fn video_moov(payload_at: u32, clockwise90: bool) -> Vec<u8> {
    let (matrix, display) = if clockwise90 {
        (MATRIX_90, (180, 320))
    } else {
        (MATRIX_0, (320, 180))
    };
    let mut trak = tkhd(0, 7, 400, matrix, display.0, display.1);
    trak.extend(elst(50, 100));
    let mut mdia = mdhd(0, 1000, 400);
    mdia.extend(hdlr(b"vide"));
    let mut stbl = stsd(&avc1(320, 180, &[1, 0x42, 0x00]));
    stbl.extend(stts(&[(2, 100), (1, 200)]));
    stbl.extend(ctts(&[(1, 200), (2, 0)]));
    stbl.extend(stsc(&[(1, 2, 1), (2, 1, 1)]));
    stbl.extend(stsz(&[3, 2, 1]));
    stbl.extend(stco(&[payload_at, payload_at + 5]));
    stbl.extend(stss(&[1]));
    mdia.extend(atom(b"minf", &atom(b"stbl", &stbl)));
    trak.extend(atom(b"mdia", &mdia));
    let mut moov = mvhd(0, 600, 1000);
    moov.extend(atom(b"trak", &trak));
    moov.extend(timecode_trak());
    atom(b"moov", &moov)
}

fn timecode_trak() -> Vec<u8> {
    let mut trak = tkhd(0, 8, 0, MATRIX_0, 0, 0);
    let mut mdia = mdhd(0, 1000, 0);
    mdia.extend(hdlr(b"tmcd"));
    trak.extend(atom(b"mdia", &mdia));
    atom(b"trak", &trak)
}

fn hdr_file() -> Vec<u8> {
    let ftyp = atom(b"ftyp", b"isom\0\0\0\0isom");
    let mdat = atom(b"mdat", &[9]);
    let payload_at = u32::try_from(ftyp.len() + 8).expect("offset fits");
    let entry = avc1_boxes(16, 16, &[1], &colr_nclx(16, false));
    let mut trak = tkhd(0, 1, 1, MATRIX_0, 16, 16);
    let mut mdia = mdhd(0, 1000, 1);
    mdia.extend(hdlr(b"vide"));
    let mut stbl = stsd(&entry);
    stbl.extend(stts(&[(1, 1)]));
    stbl.extend(stsc(&[(1, 1, 1)]));
    stbl.extend(stsz(&[1]));
    stbl.extend(stco(&[payload_at]));
    mdia.extend(atom(b"minf", &atom(b"stbl", &stbl)));
    trak.extend(atom(b"mdia", &mdia));
    let mut moov = mvhd(0, 1000, 1);
    moov.extend(atom(b"trak", &trak));
    let mut file = ftyp;
    file.extend(mdat);
    file.extend(atom(b"moov", &moov));
    file
}

fn wide_file() -> Vec<u8> {
    let ftyp = atom(b"ftyp", b"qt  \0\0\0\0qt  ");
    let mdat = atom(b"mdat", &[9]);
    let payload_at = u64::try_from(ftyp.len() + 8).expect("offset fits");
    let mut trak = tkhd(1, 1, 90_000, MATRIX_0, 16, 16);
    let mut mdia = mdhd(1, 90_000, 90_000);
    mdia.extend(hdlr(b"vide"));
    let mut stbl = stsd(&avc1(16, 16, &[1]));
    stbl.extend(stts(&[(1, 512)]));
    stbl.extend(stsc(&[(1, 1, 1)]));
    stbl.extend(stsz(&[1]));
    stbl.extend(co64(&[payload_at]));
    mdia.extend(atom(b"minf", &atom(b"stbl", &stbl)));
    trak.extend(atom(b"mdia", &mdia));
    let mut moov = mvhd(1, 90_000, 90_000);
    moov.extend(atom(b"trak", &trak));
    let mut file = ftyp;
    file.extend(mdat);
    file.extend(atom(b"moov", &moov));
    file
}

fn audio_file(entry: &[u8]) -> Vec<u8> {
    let ftyp = atom(b"ftyp", b"M4A \0\0\0\0M4A ");
    let mdat = atom(b"mdat", &[1, 2]);
    let payload_at = u32::try_from(ftyp.len() + 8).expect("offset fits");
    let mut trak = tkhd(0, 1, 2048, MATRIX_0, 0, 0);
    let mut mdia = mdhd(0, 48_000, 2048);
    mdia.extend(hdlr(b"soun"));
    let mut stbl = stsd(entry);
    stbl.extend(stts(&[(2, 1024)]));
    stbl.extend(stsc(&[(1, 2, 1)]));
    stbl.extend(stsz(&[1, 1]));
    stbl.extend(stco(&[payload_at]));
    mdia.extend(atom(b"minf", &atom(b"stbl", &stbl)));
    trak.extend(atom(b"mdia", &mdia));
    let mut moov = mvhd(0, 1000, 1);
    moov.extend(atom(b"trak", &trak));
    let mut file = ftyp;
    file.extend(mdat);
    file.extend(atom(b"moov", &moov));
    file
}

fn nibble_file() -> Vec<u8> {
    let ftyp = atom(b"ftyp", b"isom\0\0\0\0isom");
    let mdat = atom(b"mdat", &[9, 9, 8]);
    let payload_at = u32::try_from(ftyp.len() + 8).expect("offset fits");
    let mut trak = tkhd(0, 1, 2, MATRIX_0, 16, 16);
    let mut mdia = mdhd(0, 1000, 2);
    mdia.extend(hdlr(b"vide"));
    let mut stbl = stsd(&avc1(16, 16, &[1]));
    stbl.extend(stts(&[(2, 1)]));
    stbl.extend(stsc(&[(1, 2, 1)]));
    stbl.extend(stz2(4, &[0x21]));
    stbl.extend(stco(&[payload_at]));
    mdia.extend(atom(b"minf", &atom(b"stbl", &stbl)));
    trak.extend(atom(b"mdia", &mdia));
    let mut moov = mvhd(0, 1000, 2);
    moov.extend(atom(b"trak", &trak));
    let mut file = ftyp;
    file.extend(mdat);
    file.extend(atom(b"moov", &moov));
    file
}

fn mismatched_counts() -> Vec<u8> {
    let ftyp = atom(b"ftyp", b"isom\0\0\0\0isom");
    let mdat = atom(b"mdat", &[1, 2]);
    let payload_at = u32::try_from(ftyp.len() + 8).expect("offset fits");
    let mut trak = tkhd(0, 1, 2, MATRIX_0, 16, 16);
    let mut mdia = mdhd(0, 1000, 2);
    mdia.extend(hdlr(b"vide"));
    let mut stbl = stsd(&avc1(16, 16, &[1]));
    stbl.extend(stts(&[(1, 1)]));
    stbl.extend(stsc(&[(1, 2, 1)]));
    stbl.extend(stsz(&[1, 1]));
    stbl.extend(stco(&[payload_at]));
    mdia.extend(atom(b"minf", &atom(b"stbl", &stbl)));
    trak.extend(atom(b"mdia", &mdia));
    let mut moov = mvhd(0, 1000, 2);
    moov.extend(atom(b"trak", &trak));
    let mut file = ftyp;
    file.extend(mdat);
    file.extend(atom(b"moov", &moov));
    file
}

fn sample_past_end() -> Vec<u8> {
    let mut trak = tkhd(0, 1, 1, MATRIX_0, 16, 16);
    let mut mdia = mdhd(0, 1000, 1);
    mdia.extend(hdlr(b"vide"));
    let mut stbl = stsd(&avc1(16, 16, &[1]));
    stbl.extend(stts(&[(1, 1)]));
    stbl.extend(stsc(&[(1, 1, 1)]));
    stbl.extend(stsz(&[1]));
    stbl.extend(stco(&[0x00FF_0000]));
    mdia.extend(atom(b"minf", &atom(b"stbl", &stbl)));
    trak.extend(atom(b"mdia", &mdia));
    movie_of(&atom(b"trak", &trak))
}

fn bare_trak(timescale: u32) -> Vec<u8> {
    let mut trak = tkhd(0, 1, 1, MATRIX_0, 16, 16);
    let mut mdia = mdhd(0, timescale, 1);
    mdia.extend(hdlr(b"vide"));
    trak.extend(atom(b"mdia", &mdia));
    atom(b"trak", &trak)
}

fn movie_of(trak: &[u8]) -> Vec<u8> {
    let mut moov = mvhd(0, 1000, 1);
    moov.extend_from_slice(trak);
    atom(b"moov", &moov)
}

fn atom(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = u32::try_from(8 + payload.len()).expect("box fits");
    let mut out = size.to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

fn full(kind: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let flag_bytes = flags.to_be_bytes();
    let mut body = vec![version, flag_bytes[1], flag_bytes[2], flag_bytes[3]];
    body.extend_from_slice(payload);
    atom(kind, &body)
}

fn mvhd(version: u8, timescale: u32, duration: u32) -> Vec<u8> {
    let mut body = Vec::new();
    if version == 0 {
        body.extend_from_slice(&[0; 8]);
        body.extend_from_slice(&timescale.to_be_bytes());
        body.extend_from_slice(&duration.to_be_bytes());
    } else {
        body.extend_from_slice(&[0; 16]);
        body.extend_from_slice(&timescale.to_be_bytes());
        body.extend_from_slice(&u64::from(duration).to_be_bytes());
    }
    full(b"mvhd", version, 0, &body)
}

fn tkhd(version: u8, id: u32, duration: u32, matrix: [u32; 9], width: u32, height: u32) -> Vec<u8> {
    let mut body = Vec::new();
    if version == 0 {
        body.extend_from_slice(&[0; 8]);
        body.extend_from_slice(&id.to_be_bytes());
        body.extend_from_slice(&[0; 4]);
        body.extend_from_slice(&duration.to_be_bytes());
        body.extend_from_slice(&[0; 16]);
    } else {
        body.extend_from_slice(&[0; 16]);
        body.extend_from_slice(&id.to_be_bytes());
        body.extend_from_slice(&[0; 4]);
        body.extend_from_slice(&u64::from(duration).to_be_bytes());
        body.extend_from_slice(&[0; 16]);
    }
    for value in matrix {
        body.extend_from_slice(&value.to_be_bytes());
    }
    body.extend_from_slice(&(width << 16).to_be_bytes());
    body.extend_from_slice(&(height << 16).to_be_bytes());
    full(b"tkhd", version, 1, &body)
}

fn mdhd(version: u8, timescale: u32, duration: u32) -> Vec<u8> {
    let mut body = Vec::new();
    if version == 0 {
        body.extend_from_slice(&[0; 8]);
        body.extend_from_slice(&timescale.to_be_bytes());
        body.extend_from_slice(&duration.to_be_bytes());
    } else {
        body.extend_from_slice(&[0; 16]);
        body.extend_from_slice(&timescale.to_be_bytes());
        body.extend_from_slice(&u64::from(duration).to_be_bytes());
    }
    full(b"mdhd", version, 0, &body)
}

fn hdlr(kind: &[u8; 4]) -> Vec<u8> {
    let mut body = vec![0; 8];
    body[4..8].copy_from_slice(kind);
    full(b"hdlr", 0, 0, &body)
}

fn elst(duration: u32, media_time: i32) -> Vec<u8> {
    let mut body = 1u32.to_be_bytes().to_vec();
    body.extend_from_slice(&duration.to_be_bytes());
    body.extend_from_slice(&media_time.to_be_bytes());
    body.extend_from_slice(&0x0001_0000u32.to_be_bytes());
    atom(b"edts", &full(b"elst", 0, 0, &body))
}

fn avc1(width: u16, height: u16, avcc: &[u8]) -> Vec<u8> {
    avc1_boxes(width, height, avcc, &[])
}

fn avc1_boxes(width: u16, height: u16, avcc: &[u8], extra: &[u8]) -> Vec<u8> {
    let mut payload = vec![0; 78];
    payload[7] = 1;
    payload[24..26].copy_from_slice(&width.to_be_bytes());
    payload[26..28].copy_from_slice(&height.to_be_bytes());
    payload.extend(atom(b"avcC", avcc));
    payload.extend_from_slice(extra);
    atom(b"avc1", &payload)
}

fn colr_nclx(transfer: u16, full_range: bool) -> Vec<u8> {
    let mut payload = b"nclx".to_vec();
    payload.extend_from_slice(&9u16.to_be_bytes());
    payload.extend_from_slice(&transfer.to_be_bytes());
    payload.extend_from_slice(&9u16.to_be_bytes());
    payload.push(if full_range { 0x80 } else { 0 });
    atom(b"colr", &payload)
}

fn mp4a(sample_rate: u32, channels: u16, config: &[u8]) -> Vec<u8> {
    let mut payload = vec![0; 28];
    payload[7] = 1;
    payload[16..18].copy_from_slice(&channels.to_be_bytes());
    payload[24..28].copy_from_slice(&(sample_rate << 16).to_be_bytes());
    payload.extend(full(b"esds", 0, 0, &esds(config)));
    atom(b"mp4a", &payload)
}

fn mp4a_v2(sample_rate: f64, channels: u32, config: &[u8]) -> Vec<u8> {
    let mut payload = vec![0; 64];
    payload[7] = 1;
    payload[8..10].copy_from_slice(&2u16.to_be_bytes());
    payload[32..40].copy_from_slice(&sample_rate.to_bits().to_be_bytes());
    payload[40..44].copy_from_slice(&channels.to_be_bytes());
    payload.extend(full(b"esds", 0, 0, &esds(config)));
    atom(b"mp4a", &payload)
}

fn mp4a_wave(sample_rate: u32, channels: u16, config: &[u8]) -> Vec<u8> {
    let mut payload = vec![0; 28];
    payload[7] = 1;
    payload[16..18].copy_from_slice(&channels.to_be_bytes());
    payload[24..28].copy_from_slice(&(sample_rate << 16).to_be_bytes());
    payload.extend(atom(b"wave", &full(b"esds", 0, 0, &esds(config))));
    atom(b"mp4a", &payload)
}

fn esds(config: &[u8]) -> Vec<u8> {
    let specific = descriptor(0x05, config);
    let mut decoder = vec![0x40, 0x15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    decoder.extend(specific);
    let mut elementary = vec![0x00, 0x01, 0x00];
    elementary.extend(descriptor(0x04, &decoder));
    descriptor(0x03, &elementary)
}

fn descriptor(tag: u8, body: &[u8]) -> Vec<u8> {
    assert!(body.len() < 128, "test descriptor fits in one size byte");
    let mut out = vec![tag, u8::try_from(body.len()).expect("size fits")];
    out.extend_from_slice(body);
    out
}

fn stsd(entry: &[u8]) -> Vec<u8> {
    let mut body = 1u32.to_be_bytes().to_vec();
    body.extend_from_slice(entry);
    full(b"stsd", 0, 0, &body)
}

fn stts(runs: &[(u32, u32)]) -> Vec<u8> {
    let mut body = u32::try_from(runs.len())
        .expect("count fits")
        .to_be_bytes()
        .to_vec();
    for (count, delta) in runs {
        body.extend_from_slice(&count.to_be_bytes());
        body.extend_from_slice(&delta.to_be_bytes());
    }
    full(b"stts", 0, 0, &body)
}

fn ctts(runs: &[(u32, u32)]) -> Vec<u8> {
    let mut body = u32::try_from(runs.len())
        .expect("count fits")
        .to_be_bytes()
        .to_vec();
    for (count, offset) in runs {
        body.extend_from_slice(&count.to_be_bytes());
        body.extend_from_slice(&offset.to_be_bytes());
    }
    full(b"ctts", 0, 0, &body)
}

fn stsc(entries: &[(u32, u32, u32)]) -> Vec<u8> {
    let mut body = u32::try_from(entries.len())
        .expect("count fits")
        .to_be_bytes()
        .to_vec();
    for (first, per_chunk, description) in entries {
        body.extend_from_slice(&first.to_be_bytes());
        body.extend_from_slice(&per_chunk.to_be_bytes());
        body.extend_from_slice(&description.to_be_bytes());
    }
    full(b"stsc", 0, 0, &body)
}

fn stsz(sizes: &[u32]) -> Vec<u8> {
    let mut body = 0u32.to_be_bytes().to_vec();
    body.extend_from_slice(
        &u32::try_from(sizes.len())
            .expect("count fits")
            .to_be_bytes(),
    );
    for size in sizes {
        body.extend_from_slice(&size.to_be_bytes());
    }
    full(b"stsz", 0, 0, &body)
}

fn stz2(field_size: u8, packed: &[u8]) -> Vec<u8> {
    let mut body = u32::from(field_size).to_be_bytes().to_vec();
    body.extend_from_slice(&2u32.to_be_bytes());
    body.extend_from_slice(packed);
    full(b"stz2", 0, 0, &body)
}

fn stco(offsets: &[u32]) -> Vec<u8> {
    let mut body = u32::try_from(offsets.len())
        .expect("count fits")
        .to_be_bytes()
        .to_vec();
    for offset in offsets {
        body.extend_from_slice(&offset.to_be_bytes());
    }
    full(b"stco", 0, 0, &body)
}

fn co64(offsets: &[u64]) -> Vec<u8> {
    let mut body = u32::try_from(offsets.len())
        .expect("count fits")
        .to_be_bytes()
        .to_vec();
    for offset in offsets {
        body.extend_from_slice(&offset.to_be_bytes());
    }
    full(b"co64", 0, 0, &body)
}

fn stss(samples: &[u32]) -> Vec<u8> {
    let mut body = u32::try_from(samples.len())
        .expect("count fits")
        .to_be_bytes()
        .to_vec();
    for sample in samples {
        body.extend_from_slice(&sample.to_be_bytes());
    }
    full(b"stss", 0, 0, &body)
}

const MATRIX_0: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];

const MATRIX_90: [u32; 9] = [0, 0x0001_0000, 0, 0xFFFF_0000, 0, 0, 0, 0, 0x4000_0000];
