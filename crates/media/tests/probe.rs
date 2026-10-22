//! Probe reads an MP4 and reports streams, timing, size, rotation, and HDR.

use cutlass_core::{AudioCodec, ColorTags, ContentLight, MasteringDisplay, VideoCodec};
use cutlass_media::{Content, Error, FrameTiming, Probe};
use cutlass_mp4::{Fourcc, Rotation};

#[test]
fn a_file_reports_streams_timing_size_rotation_and_hdr() {
    let probe = Probe::parse(&sample_file()).expect("sample file");
    assert_eq!(probe.timescale().get(), 600);
    assert_eq!(probe.duration(), 1000);

    let tracks = probe.tracks();
    assert_eq!(tracks.len(), 3);

    let video = &tracks[0];
    assert_eq!(video.id, 1);
    assert!(video.enabled);
    assert_eq!(video.timescale.get(), 1000);
    assert_eq!(video.duration, 300);
    assert_eq!(video.rotation, Rotation::Clockwise90);
    assert_eq!(video.display_width, 18);
    assert_eq!(video.display_height, 32);
    assert_eq!(video.timing, FrameTiming::Variable);
    assert_eq!(video.hdr(), Some(true));
    let Content::Video {
        config,
        color,
        mastering,
        content_light,
    } = &video.content
    else {
        panic!("track 1 is video");
    };
    assert_eq!(config.codec, VideoCodec::H264);
    assert_eq!((config.width, config.height), (32, 18));
    assert_eq!(
        color.expect("colour tags"),
        ColorTags {
            primaries: 9,
            transfer: 16,
            matrix: 9,
            full_range: Some(false),
        }
    );
    assert_eq!(
        *mastering,
        Some(MasteringDisplay {
            primaries_x: [1, 2, 3],
            primaries_y: [4, 5, 6],
            white_x: 7,
            white_y: 8,
            max_luminance: 10_000_000,
            min_luminance: 0,
        })
    );
    assert_eq!(
        *content_light,
        Some(ContentLight {
            max_content_light_level: 1_000,
            max_pic_average_light_level: 0,
        })
    );
    assert_eq!(probe.samples(0).expect("video samples").len(), 2);

    let audio = &tracks[1];
    assert_eq!(audio.timing, FrameTiming::Constant(1024));
    assert_eq!(audio.hdr(), None);
    let Content::Audio(config) = &audio.content else {
        panic!("track 2 is audio");
    };
    assert_eq!(config.codec, AudioCodec::Aac);
    assert_eq!(config.sample_rate, 48_000);
    assert_eq!(config.channels, 2);
    assert_eq!(probe.samples(1).expect("audio samples").len(), 2);

    assert_eq!(tracks[2].timing, FrameTiming::Empty);
    assert_eq!(tracks[2].content, Content::Other(Fourcc::new(*b"tmcd")));
    assert!(probe.samples(2).expect("timecode").is_empty());

    assert_eq!(probe.samples(3), None);
    assert!(matches!(probe.open_video(3), Err(Error::NoTrack(3))));
    assert!(matches!(probe.open_video(1), Err(Error::NotVideo(2))));
}

#[cfg(target_vendor = "apple")]
#[test]
fn an_h264_fixture_decodes_through_the_probe() {
    use cutlass_media::{DecoderKind, choose_video_decoder};

    let fixture = cutlass_test_fixtures::video("h264-high-bframes-128x72");
    assert_eq!(
        choose_video_decoder(fixture.config.codec),
        Ok(DecoderKind::System)
    );
    let probe = Probe::parse(&fixture_file(&fixture)).expect("fixture file");
    let mut session = probe.open_video(0).expect("open the system decoder");
    let mut frames = 0usize;
    for packet in probe.samples(0).expect("samples") {
        session.send_packet(packet).expect("send a packet");
        while session.receive_frame().expect("receive a frame").is_some() {
            frames += 1;
        }
    }
    session.flush().expect("flush the decoder");
    while session
        .receive_frame()
        .expect("receive a flushed frame")
        .is_some()
    {
        frames += 1;
    }
    assert_eq!(frames, fixture.frames.len());
}

#[cfg(target_vendor = "apple")]
fn fixture_file(fixture: &cutlass_test_fixtures::VideoFixture) -> Vec<u8> {
    let mut samples = Vec::new();
    let mut sizes = Vec::new();
    let mut runs: Vec<(u32, u32)> = Vec::new();
    let mut duration = 0u32;
    for packet in &fixture.packets {
        sizes.push(u32::try_from(packet.data.len()).expect("sample fits"));
        samples.extend_from_slice(&packet.data);
        let sample =
            u32::try_from(packet.duration.expect("duration").value).expect("duration fits");
        duration = duration.checked_add(sample).expect("track duration fits");
        match runs.last_mut() {
            Some((count, last)) if *last == sample => *count += 1,
            _ => runs.push((1, sample)),
        }
    }
    let count = u32::try_from(fixture.packets.len()).expect("count fits");
    let timescale = fixture
        .packets
        .first()
        .expect("fixture has a packet")
        .pts
        .timescale
        .get();
    let ftyp = atom(b"ftyp", b"isom\0\0\0\0isom");
    let payload_at = u32::try_from(ftyp.len() + 8).expect("offset fits");
    let entry = avc1(
        u16::try_from(fixture.config.width).expect("width fits"),
        u16::try_from(fixture.config.height).expect("height fits"),
        &fixture.config.config_record,
        &[],
    );
    let mut trak = tkhd(
        1,
        duration,
        MATRIX_0,
        fixture.config.width,
        fixture.config.height,
    );
    let mut mdia = mdhd(timescale, duration);
    mdia.extend(hdlr(b"vide"));
    let mut stbl = stsd(&entry);
    stbl.extend(stts(&runs));
    stbl.extend(stsc(&[(1, count, 1)]));
    stbl.extend(stsz(&sizes));
    stbl.extend(stco(&[payload_at]));
    mdia.extend(atom(b"minf", &atom(b"stbl", &stbl)));
    trak.extend(atom(b"mdia", &mdia));
    let mut moov = mvhd(timescale, duration);
    moov.extend(atom(b"trak", &trak));
    let mut file = ftyp;
    file.extend(atom(b"mdat", &samples));
    file.extend(atom(b"moov", &moov));
    file
}

fn sample_file() -> Vec<u8> {
    let samples = [1u8, 2, 3, 4, 7, 8];
    let ftyp = atom(b"ftyp", b"isom\0\0\0\0isom");
    let payload_at = u32::try_from(ftyp.len() + 8).expect("offset fits");
    let mut file = ftyp;
    file.extend(atom(b"mdat", &samples));
    file.extend(moov(payload_at));
    file
}

fn moov(payload_at: u32) -> Vec<u8> {
    let mut moov = mvhd(600, 1000);
    moov.extend(video_trak(payload_at));
    moov.extend(audio_trak(payload_at + 4));
    moov.extend(timecode_trak());
    atom(b"moov", &moov)
}

fn video_trak(offset: u32) -> Vec<u8> {
    let mut extra = colr_nclx(16);
    extra.extend(mdcv());
    extra.extend(clli());
    let entry = avc1(32, 18, &[1, 0x42, 0x00], &extra);
    let mut trak = tkhd(1, 300, MATRIX_90, 18, 32);
    let mut mdia = mdhd(1000, 300);
    mdia.extend(hdlr(b"vide"));
    let mut stbl = stsd(&entry);
    stbl.extend(stts(&[(1, 100), (1, 200)]));
    stbl.extend(stsc(&[(1, 2, 1)]));
    stbl.extend(stsz(&[3, 1]));
    stbl.extend(stco(&[offset]));
    mdia.extend(atom(b"minf", &atom(b"stbl", &stbl)));
    trak.extend(atom(b"mdia", &mdia));
    atom(b"trak", &trak)
}

fn audio_trak(offset: u32) -> Vec<u8> {
    let mut trak = tkhd(2, 2048, MATRIX_0, 0, 0);
    let mut mdia = mdhd(48_000, 2048);
    mdia.extend(hdlr(b"soun"));
    let mut stbl = stsd(&mp4a(48_000, 2, &[0x11, 0x90]));
    stbl.extend(stts(&[(2, 1024)]));
    stbl.extend(stsc(&[(1, 2, 1)]));
    stbl.extend(stsz(&[1, 1]));
    stbl.extend(stco(&[offset]));
    mdia.extend(atom(b"minf", &atom(b"stbl", &stbl)));
    trak.extend(atom(b"mdia", &mdia));
    atom(b"trak", &trak)
}

fn timecode_trak() -> Vec<u8> {
    let mut trak = tkhd(3, 0, MATRIX_0, 0, 0);
    let mut mdia = mdhd(1000, 0);
    mdia.extend(hdlr(b"tmcd"));
    trak.extend(atom(b"mdia", &mdia));
    atom(b"trak", &trak)
}

fn atom(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = u32::try_from(8 + payload.len()).expect("box fits");
    let mut out = size.to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

fn full(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut body = vec![0, 0, 0, 0];
    body.extend_from_slice(payload);
    atom(kind, &body)
}

fn mvhd(timescale: u32, duration: u32) -> Vec<u8> {
    let mut body = vec![0; 8];
    body.extend_from_slice(&timescale.to_be_bytes());
    body.extend_from_slice(&duration.to_be_bytes());
    full(b"mvhd", &body)
}

fn tkhd(id: u32, duration: u32, matrix: [u32; 9], width: u32, height: u32) -> Vec<u8> {
    let mut body = vec![0; 8];
    body.extend_from_slice(&id.to_be_bytes());
    body.extend_from_slice(&[0; 4]);
    body.extend_from_slice(&duration.to_be_bytes());
    body.extend_from_slice(&[0; 16]);
    for value in matrix {
        body.extend_from_slice(&value.to_be_bytes());
    }
    body.extend_from_slice(&(width << 16).to_be_bytes());
    body.extend_from_slice(&(height << 16).to_be_bytes());
    let flags = 1u32.to_be_bytes();
    let mut versioned = vec![0, flags[1], flags[2], flags[3]];
    versioned.extend(body);
    atom(b"tkhd", &versioned)
}

fn mdhd(timescale: u32, duration: u32) -> Vec<u8> {
    let mut body = vec![0; 8];
    body.extend_from_slice(&timescale.to_be_bytes());
    body.extend_from_slice(&duration.to_be_bytes());
    full(b"mdhd", &body)
}

fn hdlr(kind: &[u8; 4]) -> Vec<u8> {
    let mut body = vec![0; 8];
    body[4..8].copy_from_slice(kind);
    full(b"hdlr", &body)
}

fn avc1(width: u16, height: u16, avcc: &[u8], extra: &[u8]) -> Vec<u8> {
    let mut payload = vec![0; 78];
    payload[7] = 1;
    payload[24..26].copy_from_slice(&width.to_be_bytes());
    payload[26..28].copy_from_slice(&height.to_be_bytes());
    payload.extend(atom(b"avcC", avcc));
    payload.extend_from_slice(extra);
    atom(b"avc1", &payload)
}

fn mdcv() -> Vec<u8> {
    let mut payload = Vec::with_capacity(24);
    for (x, y) in [(1u16, 4u16), (2, 5), (3, 6)] {
        payload.extend(x.to_be_bytes());
        payload.extend(y.to_be_bytes());
    }
    payload.extend(7u16.to_be_bytes());
    payload.extend(8u16.to_be_bytes());
    payload.extend(10_000_000u32.to_be_bytes());
    payload.extend(0u32.to_be_bytes());
    atom(b"mdcv", &payload)
}

fn clli() -> Vec<u8> {
    let mut payload = Vec::with_capacity(4);
    payload.extend(1_000u16.to_be_bytes());
    payload.extend(0u16.to_be_bytes());
    atom(b"clli", &payload)
}

fn colr_nclx(transfer: u16) -> Vec<u8> {
    let mut payload = b"nclx".to_vec();
    payload.extend_from_slice(&9u16.to_be_bytes());
    payload.extend_from_slice(&transfer.to_be_bytes());
    payload.extend_from_slice(&9u16.to_be_bytes());
    payload.push(0);
    atom(b"colr", &payload)
}

fn mp4a(sample_rate: u32, channels: u16, config: &[u8]) -> Vec<u8> {
    let mut payload = vec![0; 28];
    payload[7] = 1;
    payload[16..18].copy_from_slice(&channels.to_be_bytes());
    payload[24..28].copy_from_slice(&(sample_rate << 16).to_be_bytes());
    payload.extend(full(b"esds", &esds(config)));
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
    full(b"stsd", &body)
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
    full(b"stts", &body)
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
    full(b"stsc", &body)
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
    full(b"stsz", &body)
}

fn stco(offsets: &[u32]) -> Vec<u8> {
    let mut body = u32::try_from(offsets.len())
        .expect("count fits")
        .to_be_bytes()
        .to_vec();
    for offset in offsets {
        body.extend_from_slice(&offset.to_be_bytes());
    }
    full(b"stco", &body)
}

const MATRIX_0: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];
const MATRIX_90: [u32; 9] = [0, 0x0001_0000, 0, 0xFFFF_0000, 0, 0, 0, 0, 0x4000_0000];
