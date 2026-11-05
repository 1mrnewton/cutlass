//! The reader opens fixture files. A generated solid is not a container.

use std::cell::Cell;
use std::fs::File;
use std::path::{Path, PathBuf};

use cutlass_codec::{
    CodecId, ContainerId, FrameSize, Packet as FilePacket, StreamCodec, StreamParams, Timescale,
    Timestamp,
};
use cutlass_container::{
    Demux, DisplayQuarter, Gap, Input, Landing, OpenError, Probe, ProbeResult, ReadError,
    SeekLanding, SeekTo, Stream, TrackId, Trim,
};
use cutlass_decode::{
    BitDepth, CodecError, CodecParams, ColorPrimaries, ColorType, Decoder, Output, Packet,
    PictureInfo, PictureOutput, PixelFormat, Transfer,
};
use cutlass_jpeg::JpegDecoder;
use cutlass_model::{
    Draft, MaterialAudio, MaterialId, MaterialVideo, Micros, Segment, SegmentId, TimeRange, Track,
    TrackAttribute, TrackType,
};
use cutlass_picture::{Picture, PictureSource, Sample, rgba_len};
use cutlass_png::PngDecoder;
use cutlass_still::{StillDemux, StillProbe, still_limits};

use crate::clock::ClockKind;
use crate::error::EngineError;
use crate::map::ticks_to_micros;
use crate::reader::{MAX_AUDIO_FRAMES, ReaderFactory, Steer};
use crate::source::{open_segment, peaks_from};
use crate::{MAX_BUCKETS, MAX_EDGE};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

thread_local! {
    static PLATFORM_OPENS: Cell<u32> = const { Cell::new(0) };
    static RUST_OPENS: Cell<u32> = const { Cell::new(0) };
    static PROBES: Cell<u32> = const { Cell::new(0) };
}

fn reset_counts() {
    PLATFORM_OPENS.with(|count| count.set(0));
    RUST_OPENS.with(|count| count.set(0));
    PROBES.with(|count| count.set(0));
}

fn count(slot: &Cell<u32>) -> u32 {
    slot.get()
}

fn open_still(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let file = File::open(path).map_err(|error| OpenError::Source {
        message: error.to_string(),
    })?;
    let len = file
        .metadata()
        .map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?
        .len();
    Ok(Box::new(StillDemux::open(Input::random(
        file,
        len,
        still_limits(),
    ))?))
}

fn probe_still(header: &[u8]) -> ProbeResult {
    PROBES.with(|count| count.set(count.get().saturating_add(1)));
    StillProbe.probe(header)
}

fn movie_probe(header: &[u8]) -> ProbeResult {
    if header.len() < 8 {
        return ProbeResult::NeedBytes {
            have: header.len(),
            need: 8,
        };
    }
    if header.get(4..8) == Some(b"ftyp".as_slice()) {
        ProbeResult::Accept {
            family: ContainerId::Isobmff,
            examined: 8,
        }
    } else {
        ProbeResult::Refuse { examined: 8 }
    }
}

fn movie_open(_path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    Err(OpenError::Unrecognized {
        family: ContainerId::Isobmff,
    })
}

fn open_platform(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    PLATFORM_OPENS.with(|count| count.set(count.get().saturating_add(1)));
    if params.extra.as_slice() != b"hw" {
        return Err(CodecError::Unsupported { what: "platform" });
    }
    Ok(Box::new(Token))
}

fn open_jpeg(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    RUST_OPENS.with(|count| count.set(count.get().saturating_add(1)));
    Ok(Box::new(JpegDecoder::open(params)?))
}

fn open_png(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(PngDecoder::open(params)?))
}

fn open_token(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    if params.codec != CodecId::Jpeg {
        return Err(CodecError::Unsupported { what: "token" });
    }
    Ok(Box::new(Token))
}

struct Planar;

impl Decoder for Planar {
    fn codec(&self) -> CodecId {
        CodecId::Jpeg
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        let mut bytes = vec![16; 4];
        bytes.push(128);
        bytes.push(128);
        let picture = PictureOutput::cpu(
            PictureInfo {
                width: 2,
                height: 2,
                pixel: PixelFormat::Yuv420p,
                primaries: ColorPrimaries::Bt709,
                transfer: Transfer::Bt709,
            },
            bytes,
            packet.pts,
            packet.timescale,
        )?;
        Ok(vec![Output::Picture(picture)])
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {}

    fn decode_picture(
        &mut self,
        _packet: &Packet,
    ) -> Result<Vec<cutlass_decode::StillOutput>, CodecError> {
        Err(CodecError::Unsupported {
            what: "picture output",
        })
    }
}

struct Token;

impl Decoder for Token {
    fn codec(&self) -> CodecId {
        CodecId::Jpeg
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        let picture = PictureOutput::platform(
            PictureInfo {
                width: 1,
                height: 1,
                pixel: PixelFormat::Rgba8,
                primaries: ColorPrimaries::Bt709,
                transfer: Transfer::Bt709,
            },
            7,
            packet.pts,
            packet.timescale,
        )?;
        Ok(vec![Output::Picture(picture)])
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {}

    fn decode_picture(
        &mut self,
        _packet: &Packet,
    ) -> Result<Vec<cutlass_decode::StillOutput>, CodecError> {
        Err(CodecError::Unsupported {
            what: "picture output",
        })
    }
}

fn factory() -> ReaderFactory {
    let mut factory = ReaderFactory::new();
    factory.register_container(movie_probe, movie_open);
    factory.register_container(probe_still, open_still);
    factory
        .decoders_mut()
        .register(CodecId::Jpeg, open_platform);
    factory.decoders_mut().register(CodecId::Jpeg, open_jpeg);
    factory.decoders_mut().register(CodecId::Png, open_png);
    factory
}

fn draft_for(path: &Path) -> (Draft, SegmentId) {
    let mut draft = Draft::empty("D");
    let mut video = MaterialVideo::new("v");
    video.path = path.display().to_string();
    draft.materials.videos.push(video);
    let mut segment = Segment::new("s", "v", TimeRange::new(0, 1_000_000));
    segment.source_range = Some(TimeRange::new(0, 1_000_000));
    draft.tracks[0].segments.push(segment);
    (draft, SegmentId::new("s"))
}

#[test]
fn the_engine_does_not_depend_on_a_codec_or_container_crate() {
    let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
        .expect("toml");
    let dependencies = text.split("[dev-dependencies]").next().expect("deps");
    for banned in [
        "cutlass-jpeg",
        "cutlass-png",
        "cutlass-still",
        "cutlass-gif",
        "cutlass-webp",
        "cutlass-h264",
        "cutlass-isobmff",
    ] {
        assert!(
            !dependencies.contains(banned),
            "{banned} is linked by the engine"
        );
    }
    assert!(dependencies.contains("cutlass-decode"));
    assert!(dependencies.contains("cutlass-container"));
    assert!(dependencies.contains("cutlass-color"));
    assert!(dependencies.contains("cutlass-audio"));
}

#[test]
fn a_fixture_opens_through_the_still_container_after_the_platform_decoder_refuses() {
    reset_counts();
    let factory = factory();
    let path = fixture("pattern.jpg");
    let info = factory.probe(&path).expect("probe");
    assert_eq!(PROBES.with(count), 1);
    assert_eq!(PLATFORM_OPENS.with(count), 0);
    assert_eq!(RUST_OPENS.with(count), 0);
    assert_eq!(info.family, ContainerId::Still);
    assert_eq!(info.streams[0].codec, "Jpeg");
    assert!(info.byte_size > 0);
    assert_eq!(info.width, Some(16));
    assert!(info.duration.is_none());

    let media = factory.open(&path).expect("open");
    assert_eq!(PLATFORM_OPENS.with(count), 1);
    assert_eq!(RUST_OPENS.with(count), 1);
    assert_eq!(media.info().family, ContainerId::Still);
}

#[test]
fn jpeg_png_webp_and_gif_open_as_stills_and_a_bad_file_is_an_error() {
    reset_counts();
    let factory = factory();
    for (name, codec, clock) in [
        ("pattern.jpg", "Jpeg", false),
        ("pattern.png", "Png", false),
        ("pattern.webp", "Webp", false),
        ("pattern.gif", "Gif", true),
    ] {
        let info = factory.probe(&fixture(name)).expect(name);
        assert_eq!(info.family, ContainerId::Still, "{name}");
        assert_eq!(info.streams[0].codec, codec, "{name}");
        assert_eq!(info.duration.is_some(), clock, "{name}");
        assert!(info.byte_size > 0, "{name}");
    }

    let png = factory.open(&fixture("pattern.png")).expect("png");
    let mut png = png;
    let frame = png
        .decode_at(Micros(0), &TimeRange::new(0, 1))
        .expect("png frame");
    assert_eq!(frame.frame.width(), 16);
    assert_eq!(frame.frame.height(), 16);

    let gif = match factory.open(&fixture("pattern.gif")) {
        Err(error) => error,
        Ok(_) => panic!("gif decoder"),
    };
    assert!(matches!(gif, EngineError::NoDecoder { .. }));
    let webp = match factory.open(&fixture("pattern.webp")) {
        Err(error) => error,
        Ok(_) => panic!("webp decoder"),
    };
    assert!(matches!(webp, EngineError::NoDecoder { .. }));

    let hostile = factory.probe(&fixture("hostile.jpg")).expect_err("hostile");
    assert!(matches!(hostile, EngineError::Hostile { .. }), "{hostile}");
    let truncated = factory
        .probe(&fixture("truncated.jpg"))
        .expect_err("truncated");
    assert!(
        matches!(truncated, EngineError::Truncated { .. }),
        "{truncated}"
    );
    let unknown = factory.probe(&fixture("unknown.bin")).expect_err("unknown");
    assert_eq!(unknown, EngineError::UnknownContainer);
}

#[test]
fn gif_duration_uses_the_file_timescale() {
    let factory = factory();
    let path = fixture("pattern.gif");
    let info = factory.probe(&path).expect("gif");
    let file = File::open(&path).expect("file");
    let len = file.metadata().expect("len").len();
    let mut demux = StillDemux::open(Input::random(file, len, still_limits())).expect("still");
    let scale = demux.streams()[0].timescale();
    let mut end_ticks = 0_i64;
    while let Some(packet) = demux.read_packet().expect("packet") {
        let start = packet.timestamp().presentation_or_decode().unwrap_or(0);
        let duration = packet.timestamp().duration().unwrap_or(0);
        end_ticks = end_ticks.max(start.saturating_add(duration));
    }
    let micros = ticks_to_micros(end_ticks, scale).expect("micros");
    assert_eq!(info.duration, Some(micros));
    assert!(micros.0 > 0);
}

#[test]
fn export_returns_the_still_and_playback_does_not_decode() {
    reset_counts();
    let factory = factory();
    let path = fixture("pattern.jpg");
    let (draft, id) = draft_for(&path);
    let before = draft.clone();
    let export = open_segment(&factory, &draft, &id, ClockKind::Export, 4).expect("export");
    assert_eq!(draft, before);
    let picture = export.picture_at(Micros(0)).expect("rgba");
    let Picture::Cpu(bytes) = picture else {
        panic!("cpu picture");
    };
    assert_eq!(bytes.rgba(), jpeg_rgba(&path));
    assert_eq!(export.decodes(), 1);
    assert_eq!(export.cached(), 1);
    let again = export.picture_at(Micros(100)).expect("cached");
    let Picture::Cpu(again) = again else {
        panic!("cpu picture");
    };
    assert_eq!(again.rgba(), bytes.rgba());
    assert_eq!(export.decodes(), 1);
    assert!(matches!(export.sample(Micros(0)), Sample::Ready(_)));

    let playback = open_segment(&factory, &draft, &id, ClockKind::Playback, 4).expect("play");
    assert!(matches!(playback.sample(Micros(0)), Sample::Miss));
    assert_eq!(playback.decodes(), 0);
    let audio = playback.advance_audio(48, 48_000).expect("clock");
    assert_eq!(audio, Micros(1_000));
    let late = playback.fill(Micros(0)).expect("late");
    assert_eq!(late, Sample::Miss);
    assert_eq!(playback.cached(), 0);
    assert_eq!(playback.decodes(), 1);

    let waited = export.advance_audio(48_000, 48_000).expect("ahead");
    assert!(waited.0 > 0);
    let filled = export.fill(Micros(0)).expect("fill");
    let Sample::Ready(Picture::Cpu(filled)) = filled else {
        panic!("filled picture");
    };
    assert_eq!(filled.rgba(), jpeg_rgba(&path));
    assert!(export.cached() >= 1);
}

fn jpeg_rgba(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).expect("jpeg");
    let params = CodecParams::audio(CodecId::Jpeg, 1, cutlass_decode::ChannelLayout::mono());
    let mut decoder = JpegDecoder::open(&params).expect("decoder");
    let image = decoder
        .decode_picture(&Packet::audio(bytes, 0, 1, 1))
        .expect("decode")
        .into_iter()
        .next()
        .expect("still")
        .image;
    assert_eq!(image.color, ColorType::Rgb);
    assert_eq!(image.depth, BitDepth::Eight);
    let mut rgba = Vec::new();
    for chunk in image.samples.as_chunks::<3>().0 {
        rgba.extend([
            u8::try_from(chunk[0]).expect("red"),
            u8::try_from(chunk[1]).expect("green"),
            u8::try_from(chunk[2]).expect("blue"),
            255,
        ]);
    }
    rgba
}

#[test]
fn a_planar_420_frame_uses_the_color_converter() {
    let mut factory = ReaderFactory::new();
    factory.register_container(probe_still, open_still);
    factory
        .decoders_mut()
        .register(CodecId::Jpeg, |_params| Ok(Box::new(Planar)));
    let path = fixture("pattern.jpg");
    let (draft, id) = draft_for(&path);
    let source = open_segment(&factory, &draft, &id, ClockKind::Export, 2).expect("source");
    let picture = source.picture_at(Micros(0)).expect("rgba");
    let Picture::Cpu(bytes) = picture else {
        panic!("cpu picture");
    };
    assert_eq!(bytes.width(), 2);
    assert_eq!(bytes.height(), 2);
    assert_eq!(
        bytes.rgba(),
        &[0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]
    );
}

/// Longer than a read used to wait for the clip's thread before it gave up.
const SLOW: std::time::Duration = std::time::Duration::from_millis(2_500);

/// Decodes after [`SLOW`], or fails after it when the flag is off.
struct Slow(bool);

impl Decoder for Slow {
    fn codec(&self) -> CodecId {
        CodecId::Jpeg
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        std::thread::sleep(SLOW);
        if !self.0 {
            return Err(CodecError::Unsupported { what: "slow" });
        }
        Planar.push(packet)
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {}

    fn decode_picture(
        &mut self,
        _packet: &Packet,
    ) -> Result<Vec<cutlass_decode::StillOutput>, CodecError> {
        Err(CodecError::Unsupported {
            what: "picture output",
        })
    }
}

#[test]
fn a_patient_read_waits_for_a_slow_decode_and_a_failed_one_is_its_error() {
    let path = fixture("pattern.jpg");
    let (draft, id) = draft_for(&path);

    let mut factory = ReaderFactory::new();
    factory.register_container(probe_still, open_still);
    factory
        .decoders_mut()
        .register(CodecId::Jpeg, |_params| Ok(Box::new(Slow(true))));
    let mut source = open_segment(&factory, &draft, &id, ClockKind::Export, 1).expect("source");
    source.read_ahead(&factory, 2).expect("thread");
    source.set_patient(true);
    let started = std::time::Instant::now();
    let picture = source.picture_at(Micros(0)).expect("the slow picture");
    assert!(
        started.elapsed() >= SLOW,
        "the picture came from the thread"
    );
    let Picture::Cpu(bytes) = picture else {
        panic!("cpu picture");
    };
    assert_eq!((bytes.width(), bytes.height()), (2, 2));
    assert!(!source.take_stale());

    let mut factory = ReaderFactory::new();
    factory.register_container(probe_still, open_still);
    factory
        .decoders_mut()
        .register(CodecId::Jpeg, |_params| Ok(Box::new(Slow(false))));
    let mut source = open_segment(&factory, &draft, &id, ClockKind::Export, 1).expect("source");
    source.read_ahead(&factory, 2).expect("thread");
    source.set_patient(true);
    let error = source.picture_at(Micros(0)).expect_err("no picture");
    assert_ne!(error, EngineError::NotReady, "the decode's own error");
}

#[test]
fn a_platform_buffer_is_the_picture_source_and_text_does_not_open_a_decoder() {
    reset_counts();
    let mut factory = ReaderFactory::new();
    factory.register_container(probe_still, open_still);
    factory.decoders_mut().register(CodecId::Jpeg, open_token);
    let path = fixture("pattern.jpg");
    let (draft, id) = draft_for(&path);
    let source = open_segment(&factory, &draft, &id, ClockKind::Export, 2).expect("source");
    let picture = source.picture_at(Micros(0)).expect("picture");
    assert!(matches!(
        picture,
        Picture::Platform(buffer) if buffer.token == 7
    ));
    assert!(matches!(
        source.sample(Micros(0)),
        Sample::Ready(Picture::Platform(_))
    ));

    let mut draft = draft;
    draft.tracks[0].kind = TrackType::Text;
    let before = draft.clone();
    let probes = PROBES.with(count);
    let text = match open_segment(&factory, &draft, &id, ClockKind::Playback, 2) {
        Err(error) => error,
        Ok(_) => panic!("text"),
    };
    assert_eq!(text, EngineError::NotMedia);
    assert_eq!(draft, before);
    assert_eq!(PROBES.with(count), probes);
}

#[test]
fn a_reverse_path_opens_that_file_and_a_missing_file_leaves_the_draft() {
    let factory = factory();
    let png = fixture("pattern.png");
    let jpg = fixture("pattern.jpg");
    let (mut draft, id) = draft_for(&png);
    draft.materials.videos[0].reverse_path = jpg.display().to_string();
    draft.tracks[0].segments[0].reverse = true;
    let source = open_segment(&factory, &draft, &id, ClockKind::Playback, 2).expect("reverse");
    assert_eq!(source.path(), jpg.as_path());
    assert_eq!(source.info().streams[0].codec, "Jpeg");

    let (draft, id) = draft_for(Path::new("/no/such/cutlass-segment-media"));
    let before = draft.clone();
    let missing = match open_segment(&factory, &draft, &id, ClockKind::Export, 2) {
        Err(error) => error,
        Ok(_) => panic!("missing"),
    };
    assert!(matches!(missing, EngineError::MissingFile { .. }));
    assert_eq!(draft, before);
}

#[test]
fn a_real_local_file_opens_when_the_folder_is_present() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
    if !root.is_dir() {
        eprintln!("skipped: ../local-files is missing");
        return;
    }
    let path = root.join("04-motion-graphics.png");
    if !path.is_file() {
        eprintln!("skipped: 04-motion-graphics.png is not in ../local-files");
        return;
    }
    let factory = factory();
    let info = factory.probe(&path).expect("probe");
    assert_eq!(info.family, ContainerId::Still);
    assert!(info.byte_size > 1_000);
    assert!(info.width.is_some_and(|width| width > 0));
    assert!(info.height.is_some_and(|height| height > 0));
    let mut media = factory.open(&path).expect("open");
    let frame = media
        .decode_at(Micros(0), &TimeRange::new(0, 1))
        .expect("one frame");
    assert!(frame.frame.width() > 0);
    assert!(path.is_file());
    assert!(
        !PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/04-motion-graphics.png")
            .exists()
    );
}

#[test]
fn a_leading_chunk_does_not_spend_the_picture_budget() {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_script, open_leading);
    readers
        .decoders_mut()
        .register(CodecId::Jpeg, open_one_picture);
    let mut media = readers.open(&fixture("pattern.jpg")).expect("open");
    let frame = media
        .decode_at(Micros(0), &TimeRange::new(0, 1))
        .expect("picture");
    assert_eq!(frame.frame.width(), 2);
    assert_eq!(frame.frame.height(), 2);
}

#[test]
fn sixty_four_silent_pictures_hit_the_decode_bound() {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_script, open_silent);
    readers
        .decoders_mut()
        .register(CodecId::Jpeg, open_silent_decoder);
    let mut media = readers.open(&fixture("pattern.jpg")).expect("open");
    let error = media
        .decode_at(Micros(0), &TimeRange::new(0, 1))
        .expect_err("bound");
    assert!(matches!(error, EngineError::DecodeLimit));
}

#[test]
fn decode_at_returns_the_picture_that_covers_the_time() {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_script, open_eight);
    readers
        .decoders_mut()
        .register(CodecId::Jpeg, open_one_picture);
    let mut media = readers.open(&fixture("pattern.jpg")).expect("open");
    let tick = |ticks: i64| ticks_to_micros(ticks, Timescale::new(30)).expect("time");
    let cover = TimeRange::new(0, 1_000_000);
    for wanted in [3, 3, 5, 6, 1, 7, 0] {
        let held = media
            .decode_at(Micros(tick(wanted).0 + 10), &cover)
            .expect("picture");
        assert_eq!(held.start, tick(wanted), "tick {wanted}");
    }
    let past = media.decode_at(tick(20), &cover).expect("last picture");
    assert_eq!(past.start, tick(7));
}

#[test]
fn a_steered_decode_moves_to_a_later_time_or_stops_where_it_got_to() {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_script, open_eight);
    readers
        .decoders_mut()
        .register(CodecId::Jpeg, open_one_picture);
    let mut media = readers.open(&fixture("pattern.jpg")).expect("open");
    let tick = |ticks: i64| ticks_to_micros(ticks, Timescale::new(30)).expect("time");
    let cover = TimeRange::new(0, 1_000_000);

    let moved = media
        .decode_steered(Micros(tick(2).0 + 10), &cover, &mut |held| {
            if held.start == tick(1) {
                Steer::Retarget(Micros(tick(5).0 + 10))
            } else {
                Steer::Go
            }
        })
        .expect("retargeted");
    assert_eq!(moved.start, tick(5));

    let mut seen = Vec::new();
    let stopped = media.decode_steered(Micros(tick(1).0 + 10), &cover, &mut |held| {
        seen.push(held.start);
        if held.start == tick(1) {
            Steer::Retarget(tick(0))
        } else {
            Steer::Go
        }
    });
    assert_eq!(stopped, Err(EngineError::Interrupted));
    let next = media
        .decode_steered(Micros(tick(3).0 + 10), &cover, &mut |held| {
            seen.push(held.start);
            Steer::Go
        })
        .expect("continued");
    assert_eq!(next.start, tick(3));
    // After stopping at tick 1 the stream went on from there, without a seek
    // back to tick 0.
    assert_eq!(seen, [tick(0), tick(1), tick(2), tick(3)]);
}

fn open_eight(_path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    Ok(Box::new(scripted(0, 8)))
}

#[test]
fn pictures_a_drain_releases_reach_the_scan_at_their_own_times() {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_script, open_four);
    readers
        .decoders_mut()
        .register(CodecId::Jpeg, |_params| Ok(Box::new(Held(Vec::new()))));
    let mut media = readers.open(&fixture("pattern.jpg")).expect("open");
    let frames = media
        .pictures_across(&TimeRange::new(0, 1_000_000), 0, 10)
        .expect("scan");
    let starts: Vec<i64> = frames.iter().map(|held| held.start.0).collect();
    let expected: Vec<i64> = (0..4)
        .map(|ticks| ticks_to_micros(ticks, Timescale::new(30)).expect("time").0)
        .collect();
    assert_eq!(starts, expected);
}

fn open_four(_path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    Ok(Box::new(scripted(0, 4)))
}

/// Holds every picture until `drain`, like a decoder waiting on reorder.
struct Held(Vec<(i64, u32)>);

impl Decoder for Held {
    fn codec(&self) -> CodecId {
        CodecId::Jpeg
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        self.0.push((packet.pts, packet.timescale));
        Ok(Vec::new())
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        let mut outputs = Vec::new();
        for (pts, timescale) in self.0.drain(..) {
            let picture = PictureOutput::cpu(
                PictureInfo {
                    width: 2,
                    height: 2,
                    pixel: PixelFormat::Yuv420p,
                    primaries: ColorPrimaries::Bt709,
                    transfer: Transfer::Bt709,
                },
                vec![16, 16, 16, 16, 128, 128],
                pts,
                timescale,
            )?;
            outputs.push(Output::Picture(picture));
        }
        Ok(outputs)
    }

    fn reset(&mut self) {
        self.0.clear();
    }
}

fn probe_script(header: &[u8]) -> ProbeResult {
    if header.is_empty() {
        return ProbeResult::NeedBytes { have: 0, need: 1 };
    }
    ProbeResult::Accept {
        family: ContainerId::Isobmff,
        examined: 1,
    }
}

fn open_leading(_path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    Ok(Box::new(scripted(80, 1)))
}

fn open_silent(_path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    Ok(Box::new(scripted(0, 64)))
}

fn open_turned(_path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let mut movie = scripted(0, 1);
    movie.streams[0] = movie.streams[0]
        .clone()
        .with_display(DisplayQuarter::Clockwise90);
    Ok(Box::new(movie))
}

#[test]
fn the_picture_stream_keeps_its_display_quarter() {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_script, open_turned);
    readers
        .decoders_mut()
        .register(CodecId::Jpeg, open_one_picture);
    let path = fixture("pattern.jpg");
    let info = readers.probe(&path).expect("probe");
    assert_eq!(info.display, DisplayQuarter::Clockwise90);
    let (draft, id) = draft_for(&path);
    let source = open_segment(&readers, &draft, &id, ClockKind::Export, 1).expect("source");
    assert_eq!(source.display(), DisplayQuarter::Clockwise90);
}

fn open_one_picture(_params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(OnePicture))
}

fn open_silent_decoder(_params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(Silent))
}

struct OnePicture;

impl Decoder for OnePicture {
    fn codec(&self) -> CodecId {
        CodecId::Jpeg
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        let mut bytes = vec![16; 4];
        bytes.push(128);
        bytes.push(128);
        let picture = PictureOutput::cpu(
            PictureInfo {
                width: 2,
                height: 2,
                pixel: PixelFormat::Yuv420p,
                primaries: ColorPrimaries::Bt709,
                transfer: Transfer::Bt709,
            },
            bytes,
            packet.pts,
            packet.timescale,
        )?;
        Ok(vec![Output::Picture(picture)])
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {}
}

struct Silent;

impl Decoder for Silent {
    fn codec(&self) -> CodecId {
        CodecId::Jpeg
    }

    fn push(&mut self, _packet: &Packet) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {}
}

struct Scripted {
    streams: Vec<Stream>,
    packets: Vec<FilePacket>,
    cursor: usize,
    cancelled: bool,
}

fn scripted(foreign: usize, pictures: usize) -> Scripted {
    let scale = Timescale::new(30);
    let size = FrameSize::try_new(2, 2).expect("size");
    let picture = Stream::new(
        TrackId::new(1),
        StreamParams::video(StreamCodec::Known(CodecId::Jpeg), size, Vec::new()),
        scale,
        None,
        Vec::new(),
        Trim::default(),
    );
    let other = Stream::new(
        TrackId::new(2),
        StreamParams::opaque(StreamCodec::Known(CodecId::Pcm), Vec::new()),
        scale,
        None,
        Vec::new(),
        Trim::default(),
    );
    let mut packets = Vec::new();
    for index in 0..foreign {
        packets.push(file_packet(2, index));
    }
    for index in 0..pictures {
        packets.push(file_packet(1, foreign.saturating_add(index)));
    }
    Scripted {
        streams: vec![picture, other],
        packets,
        cursor: 0,
        cancelled: false,
    }
}

fn file_packet(stream: u32, index: usize) -> FilePacket {
    let timestamp = Timestamp::try_new(
        Some(i64::try_from(index).unwrap_or(i64::MAX)),
        Some(i64::try_from(index).unwrap_or(i64::MAX)),
        Some(1),
    )
    .expect("time");
    FilePacket::try_new(
        stream,
        b"unit".to_vec(),
        timestamp,
        Timescale::new(30),
        stream == 1,
    )
    .expect("packet")
}

impl Demux for Scripted {
    fn family(&self) -> ContainerId {
        ContainerId::Isobmff
    }

    fn streams(&self) -> &[Stream] {
        &self.streams
    }

    fn gaps(&self) -> &[Gap] {
        &[]
    }

    fn read_packet(&mut self) -> Result<Option<FilePacket>, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        let Some(packet) = self.packets.get(self.cursor) else {
            return Ok(None);
        };
        self.cursor = self.cursor.saturating_add(1);
        Ok(Some(packet.clone()))
    }

    fn seek(&mut self, _to: SeekTo) -> Result<SeekLanding, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        self.cursor = 0;
        Ok(SeekLanding {
            landing: Landing::Packet {
                ordinal: 0,
                time: 0,
                keyframe: true,
                preroll: 0,
                byte_position: 0,
            },
            index_used: false,
            payload_scans: 0,
            entries_visited: 1,
        })
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }

    fn consumed(&self) -> u64 {
        0
    }
}

fn open_wav(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let file = File::open(path).map_err(|error| OpenError::Source {
        message: error.to_string(),
    })?;
    let length = file
        .metadata()
        .map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?
        .len();
    Ok(Box::new(cutlass_riff::WavDemux::open(file, length)?))
}

fn write_pcm_wav(path: &Path, rate: u32, bits: u16, samples: &[u8]) {
    let data_bytes = u32::try_from(samples.len()).expect("wav size");
    let channels = 1_u16;
    let align = channels * (bits / 8);
    let byte_rate = rate * u32::from(align);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&align.to_le_bytes());
    bytes.extend_from_slice(&bits.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_bytes.to_le_bytes());
    bytes.extend_from_slice(samples);
    std::fs::write(path, bytes).expect("wav");
}

#[test]
fn a_pcm_wav_seeks_and_returns_full_scale_samples() {
    let path = std::env::temp_dir().join(format!("cutlass-pcm-window-{}", std::process::id()));
    let mut payload = Vec::new();
    for sample in [0_i16, 32_767, -32_767] {
        payload.extend_from_slice(&sample.to_le_bytes());
    }
    write_pcm_wav(&path, 48_000, 16, &payload);
    let mut readers = ReaderFactory::new();
    readers.register_container(cutlass_riff::probe, open_wav);
    cutlass_pcm::register_decoders(readers.decoders_mut());
    let mut media = readers.open(&path).expect("open");
    let head = media.audio_from(Micros(0), 1).expect("head");
    assert_eq!(head.rate, 48_000);
    assert_eq!(head.as_f32(), Some(&[0.0][..]));
    let later = media.audio_from(Micros(21), 2).expect("seek");
    assert_eq!(later.as_f32(), Some(&[1.0, -1.0][..]));
    std::fs::remove_file(&path).expect("remove");
}

#[test]
fn audio_windows_join_without_a_seam_and_a_seek_lands_on_the_sample() {
    let path = std::env::temp_dir().join(format!("cutlass-pcm-ramp-{}", std::process::id()));
    let ramp: Vec<i16> = (0..9_600_i32)
        .map(|index| i16::try_from(index - 4_800).expect("sample"))
        .collect();
    let payload: Vec<u8> = ramp
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect();
    write_pcm_wav(&path, 48_000, 16, &payload);
    let mut readers = ReaderFactory::new();
    readers.register_container(cutlass_riff::probe, open_wav);
    cutlass_pcm::register_decoders(readers.decoders_mut());
    let expected = |from: usize, count: usize| -> Vec<f32> {
        ramp[from..from + count]
            .iter()
            .map(|sample| f32::from(*sample) / 32_767.0)
            .collect()
    };

    let mut media = readers.open(&path).expect("open");
    let mut joined = Vec::new();
    for window in 0..10_i64 {
        let pcm = media
            .audio_from(Micros(30_000 + window * 10_000), 480)
            .expect("window");
        joined.extend_from_slice(pcm.as_f32().expect("f32"));
    }
    assert_eq!(joined, expected(1_440, 4_800));

    let mut fresh = readers.open(&path).expect("open");
    let landed = fresh.audio_from(Micros(125_000), 96).expect("seek");
    assert_eq!(landed.as_f32().expect("f32"), &expected(6_000, 96)[..]);
    let back = fresh.audio_from(Micros(20_000), 48).expect("seek back");
    assert_eq!(back.as_f32().expect("f32"), &expected(960, 48)[..]);
    std::fs::remove_file(&path).expect("remove");
}

#[test]
fn an_eight_bit_wav_is_not_a_pcm_layout_this_decoder_opens() {
    let path = std::env::temp_dir().join(format!("cutlass-pcm-eight-{}", std::process::id()));
    write_pcm_wav(&path, 8_000, 8, &[0x80]);
    let mut readers = ReaderFactory::new();
    readers.register_container(cutlass_riff::probe, open_wav);
    cutlass_pcm::register_decoders(readers.decoders_mut());
    let Err(error) = readers.open(&path) else {
        panic!("8-bit opened");
    };
    assert_eq!(error, EngineError::Layout);
    std::fs::remove_file(&path).expect("remove");
}

#[test]
fn a_filmstrip_ask_does_not_decode_and_a_fill_stores_the_still() {
    let factory = factory();
    let path = fixture("pattern.png");
    let (draft, id) = draft_for(&path);
    let source = open_segment(&factory, &draft, &id, ClockKind::Export, 4).expect("export");
    assert_eq!(
        source.strip_at(Micros(5_000_000)).expect_err("outside"),
        EngineError::Outside
    );
    assert_eq!(
        source
            .fill_strip(Micros(5_000_000))
            .expect_err("outside fill"),
        EngineError::Outside
    );
    assert_eq!(source.decodes(), 0);
    assert_eq!(
        source.strip_at(Micros(0)).expect_err("miss"),
        EngineError::NotReady
    );
    assert_eq!(source.decodes(), 0);
    source.fill_strip(Micros(0)).expect("fill");
    assert_eq!(source.decodes(), 1);
    assert_eq!(source.cached(), 0);
    let cell = source.strip_at(Micros(100)).expect("cell");
    assert_eq!(source.decodes(), 1);
    assert_eq!(cell.width(), 16);
    assert_eq!(cell.height(), 16);
    assert!(cell.width().max(cell.height()) <= MAX_EDGE);
    assert_eq!(
        cell.rgba().len(),
        rgba_len(cell.width(), cell.height()).expect("bytes")
    );
    let picture = source.picture_at(Micros(0)).expect("picture");
    let Picture::Cpu(bytes) = picture else {
        panic!("cpu picture");
    };
    assert_eq!(cell.rgba(), bytes.rgba());
    assert_eq!(source.decodes(), 2);
    let again = source.strip_at(Micros(0)).expect("cached cell");
    assert_eq!(again.rgba(), cell.rgba());
    assert_eq!(source.decodes(), 2);
}

#[test]
fn playback_fills_a_filmstrip_without_filling_the_preview_window() {
    let factory = factory();
    let path = fixture("pattern.png");
    let (draft, id) = draft_for(&path);
    let source = open_segment(&factory, &draft, &id, ClockKind::Playback, 4).expect("play");
    assert_eq!(
        source.picture_at(Micros(0)).expect_err("playback"),
        EngineError::NotReady
    );
    assert_eq!(
        source.strip_at(Micros(0)).expect_err("strip"),
        EngineError::NotReady
    );
    assert_eq!(source.decodes(), 0);
    let audio = source.advance_audio(48, 48_000).expect("clock");
    assert_eq!(audio, Micros(1_000));
    source.fill_strip(Micros(0)).expect("strip fill");
    assert_eq!(source.decodes(), 1);
    assert_eq!(source.cached(), 0);
    let cell = source.strip_at(Micros(0)).expect("cell");
    assert_eq!(cell.width(), 16);
    assert_eq!(cell.height(), 16);
    assert_eq!(source.fill(Micros(0)).expect("late"), Sample::Miss);
    assert_eq!(source.cached(), 0);
    assert_eq!(
        source.strip_at(Micros(0)).expect("kept").rgba(),
        cell.rgba()
    );
    assert_eq!(source.decodes(), 2);
}

#[test]
fn a_platform_buffer_is_not_a_filmstrip_cell() {
    let mut factory = ReaderFactory::new();
    factory.register_container(probe_still, open_still);
    factory.decoders_mut().register(CodecId::Jpeg, open_token);
    let path = fixture("pattern.jpg");
    let (draft, id) = draft_for(&path);
    let source = open_segment(&factory, &draft, &id, ClockKind::Export, 2).expect("source");
    assert_eq!(
        source.fill_strip(Micros(0)).expect_err("platform"),
        EngineError::NotReady
    );
    assert_eq!(source.decodes(), 1);
    assert_eq!(
        source.strip_at(Micros(0)).expect_err("empty"),
        EngineError::NotReady
    );
    assert_eq!(source.decodes(), 1);
}

#[test]
fn a_real_still_fills_a_filmstrip_when_the_folder_is_present() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
    if !root.is_dir() {
        eprintln!("skipped: ../local-files is missing");
        return;
    }
    let path = root.join("04-motion-graphics.png");
    if !path.is_file() {
        eprintln!("skipped: 04-motion-graphics.png is not in ../local-files");
        return;
    }
    let factory = factory();
    let (draft, id) = draft_for(&path);
    let source = open_segment(&factory, &draft, &id, ClockKind::Playback, 2).expect("still");
    assert_eq!(
        source.strip_at(Micros(0)).expect_err("miss"),
        EngineError::NotReady
    );
    assert_eq!(source.decodes(), 0);
    source.fill_strip(Micros(0)).expect("fill");
    let cell = source.strip_at(Micros(0)).expect("cell");
    assert_eq!(source.decodes(), 1);
    assert!(cell.width() >= 1 && cell.height() >= 1);
    assert!(cell.width() <= MAX_EDGE && cell.height() <= MAX_EDGE);
    assert_eq!(
        cell.rgba().len(),
        rgba_len(cell.width(), cell.height()).expect("bytes")
    );
    assert_eq!(
        source.strip_at(Micros(0)).expect("again").rgba(),
        cell.rgba()
    );
    assert_eq!(source.decodes(), 1);
}

fn audio_readers() -> ReaderFactory {
    let mut readers = ReaderFactory::new();
    readers.register_container(cutlass_riff::probe, open_wav);
    cutlass_pcm::register_decoders(readers.decoders_mut());
    readers
}

fn audio_draft(path: &Path, mute: bool) -> (Draft, SegmentId) {
    let mut draft = Draft::empty("D");
    draft.materials.audios.push(MaterialAudio {
        id: MaterialId::new("tone"),
        path: path.display().to_string(),
        duration: Micros(1_000_000),
        ..MaterialAudio::default()
    });
    let mut track = Track::main("audio");
    track.kind = TrackType::Audio;
    if mute {
        track.attribute = TrackAttribute(TrackAttribute::MUTE);
    }
    let mut segment = Segment::new("clip", "tone", TimeRange::new(0, 1_000_000));
    segment.source_range = Some(TimeRange::new(0, 1_000_000));
    track.segments.push(segment);
    draft.tracks.push(track);
    (draft, SegmentId::new("clip"))
}

fn full_scale_wav(path: &Path) {
    let mut payload = Vec::new();
    for sample in [0_i16, 32_767, -32_767] {
        payload.extend_from_slice(&sample.to_le_bytes());
    }
    write_pcm_wav(path, 48_000, 16, &payload);
}

fn assert_full_scale(wave: &crate::Waveform) {
    assert_eq!(wave.frames(), 3);
    assert!(!wave.capped());
    assert_eq!(wave.peaks().len(), MAX_BUCKETS);
    let mid = MAX_BUCKETS / 3;
    let last = 2 * MAX_BUCKETS / 3;
    for (index, peak) in wave.peaks().iter().enumerate() {
        let (min, max): (f32, f32) = match index {
            0 => (0.0, 0.0),
            index if index == mid => (1.0, 1.0),
            index if index == last => (-1.0, -1.0),
            _ => (0.0, 0.0),
        };
        assert_eq!(peak.min.to_bits(), min.to_bits(), "{index}");
        assert_eq!(peak.max.to_bits(), max.to_bits(), "{index}");
    }
}

#[test]
fn a_waveform_ask_does_not_decode_and_a_fill_stores_full_scale_peaks() {
    let path = std::env::temp_dir().join(format!("cutlass-wave-peaks-{}", std::process::id()));
    full_scale_wav(&path);
    let readers = audio_readers();
    let (draft, id) = audio_draft(&path, false);
    let source = open_segment(&readers, &draft, &id, ClockKind::Playback, 2).expect("audio");
    assert_eq!(source.waveform().expect_err("miss"), EngineError::NotReady);
    assert_eq!(source.decodes(), 0);
    source.fill_waveform().expect("fill");
    assert_eq!(source.decodes(), 0);
    assert_full_scale(&source.waveform().expect("wave"));
    assert_eq!(source.decodes(), 0);
    assert_eq!(
        source.fill_strip(Micros(0)).expect_err("picture"),
        EngineError::NoDecoder {
            codec: "picture".to_owned(),
        }
    );
    assert_eq!(source.decodes(), 1);
    assert_eq!(source.waveform().expect("kept").frames(), 3);
    std::fs::remove_file(&path).expect("remove");
}

#[test]
fn a_muted_track_still_stores_the_file_waveform() {
    let path = std::env::temp_dir().join(format!("cutlass-wave-mute-{}", std::process::id()));
    full_scale_wav(&path);
    let readers = audio_readers();
    let (draft, id) = audio_draft(&path, true);
    let source = open_segment(&readers, &draft, &id, ClockKind::Playback, 2).expect("muted");
    source.fill_waveform().expect("fill");
    assert_eq!(source.decodes(), 0);
    assert_full_scale(&source.waveform().expect("wave"));
    std::fs::remove_file(&path).expect("remove");
}

#[test]
fn a_waveform_prefix_drops_the_unread_tail() {
    let longer = std::env::temp_dir().join(format!("cutlass-wave-prefix-{}", std::process::id()));
    let mut loud_tail = Vec::new();
    for sample in [0_i16, 0, 0, 0, 32_767] {
        loud_tail.extend_from_slice(&sample.to_le_bytes());
    }
    write_pcm_wav(&longer, 48_000, 16, &loud_tail);
    let readers = audio_readers();
    let mut media = readers.open(&longer).expect("longer");
    let prefix = peaks_from(&mut media, 4).expect("prefix");
    assert!(prefix.capped());
    assert_eq!(prefix.frames(), 4);
    assert_eq!(prefix.peaks().len(), MAX_BUCKETS);
    for peak in prefix.peaks() {
        assert_eq!(peak.max.to_bits(), 0.0_f32.to_bits());
        assert_eq!(peak.min.to_bits(), 0.0_f32.to_bits());
    }

    let exact = std::env::temp_dir().join(format!("cutlass-wave-exact-{}", std::process::id()));
    write_pcm_wav(&exact, 48_000, 16, &loud_tail[..8]);
    let mut media = readers.open(&exact).expect("exact");
    let whole = peaks_from(&mut media, 4).expect("whole");
    assert!(!whole.capped());
    assert_eq!(whole.frames(), 4);
    std::fs::remove_file(&longer).expect("remove longer");
    std::fs::remove_file(&exact).expect("remove exact");
}

#[test]
fn a_still_has_no_waveform() {
    let factory = factory();
    let path = fixture("pattern.png");
    let (draft, id) = draft_for(&path);
    let source = open_segment(&factory, &draft, &id, ClockKind::Export, 2).expect("still");
    assert_eq!(
        source.fill_waveform().expect_err("audio"),
        EngineError::NotMedia
    );
    assert_eq!(source.waveform().expect_err("empty"), EngineError::NotReady);
    assert_eq!(source.decodes(), 0);
}

#[test]
fn a_real_wav_fills_a_waveform_when_the_folder_is_present() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
    if !root.is_dir() {
        eprintln!("skipped: ../local-files is missing");
        return;
    }
    let Some(path) = smallest_wav(&root) else {
        eprintln!("skipped: ../local-files has no wav");
        return;
    };
    let readers = audio_readers();
    let (draft, id) = audio_draft(&path, false);
    let source = open_segment(&readers, &draft, &id, ClockKind::Playback, 2).expect("wav");
    assert_eq!(source.waveform().expect_err("miss"), EngineError::NotReady);
    source.fill_waveform().expect("fill");
    let wave = source.waveform().expect("wave");
    assert_eq!(source.decodes(), 0);
    assert_eq!(wave.peaks().len(), MAX_BUCKETS);
    assert!(wave.frames() > 0);
    assert!(wave.frames() <= MAX_AUDIO_FRAMES);
    if wave.frames() == MAX_AUDIO_FRAMES {
        assert!(wave.capped());
    } else {
        assert!(!wave.capped());
    }
    for peak in wave.peaks() {
        assert!(peak.min <= peak.max);
        assert!(peak.min >= -1.0 && peak.max <= 1.0);
    }
    assert_eq!(source.waveform().expect("again").frames(), wave.frames());
}

fn smallest_wav(root: &Path) -> Option<PathBuf> {
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("wav") {
            continue;
        }
        let Ok(len) = entry.metadata().map(|meta| meta.len()) else {
            continue;
        };
        if len == 0 {
            continue;
        }
        if best.as_ref().is_none_or(|(size, _)| len < *size) {
            best = Some((len, path));
        }
    }
    best.map(|(_, path)| path)
}

#[test]
fn a_stand_in_opens_in_place_of_its_source_and_probing_still_reads_the_source() {
    let factory = factory();
    let source = fixture("pattern.png");
    let proxy = fixture("pattern.jpg");
    let readers = factory.with_stand_ins(std::collections::HashMap::from([(
        source.canonicalize().expect("source"),
        proxy,
    )]));
    assert_eq!(
        readers.open(&source).expect("open").info().streams[0].codec,
        "Jpeg"
    );
    assert_eq!(
        readers.probe(&source).expect("probe").streams[0].codec,
        "Png"
    );
    // The readers it came from, as export keeps them, open the source.
    assert_eq!(
        factory.open(&source).expect("open").info().streams[0].codec,
        "Png"
    );
    // A stand-in no longer on disk leaves the source to open.
    let gone = factory.with_stand_ins(std::collections::HashMap::from([(
        source.clone(),
        fixture("no-such-proxy.mp4"),
    )]));
    assert_eq!(
        gone.open(&source).expect("open").info().streams[0].codec,
        "Png"
    );
}
