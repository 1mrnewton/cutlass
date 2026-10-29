//! The reader opens fixture files. A generated solid is not a container.

use std::cell::Cell;
use std::fs::File;
use std::path::{Path, PathBuf};

use cutlass_codec::{CodecId, ContainerId};
use cutlass_container::{Demux, Input, OpenError, Probe, ProbeResult};
use cutlass_decode::{
    CodecError, CodecParams, ColorPrimaries, Decoder, Output, Packet, PictureInfo, PictureOutput,
    PixelFormat, Transfer,
};
use cutlass_jpeg::JpegDecoder;
use cutlass_model::{Draft, MaterialVideo, Micros, Segment, SegmentId, TimeRange, TrackType};
use cutlass_picture::{Picture, PictureSource, Sample};
use cutlass_png::PngDecoder;
use cutlass_still::{StillDemux, StillProbe, still_limits};

use crate::clock::ClockKind;
use crate::error::EngineError;
use crate::map::ticks_to_micros;
use crate::reader::ReaderFactory;
use crate::source::open_segment;

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
fn export_decodes_a_fixture_and_names_the_color_miss_playback_does_not_decode() {
    reset_counts();
    let factory = factory();
    let path = fixture("pattern.jpg");
    let (draft, id) = draft_for(&path);
    let before = draft.clone();
    let export = open_segment(&factory, &draft, &id, ClockKind::Export, 4).expect("export");
    assert_eq!(draft, before);
    let err = export.picture_at(Micros(0)).expect_err("color");
    assert!(matches!(err, EngineError::ColorMiss { layout } if !layout.is_empty()));
    assert_eq!(export.decodes(), 1);
    assert_eq!(export.cached(), 1);
    let again = export.picture_at(Micros(100)).expect_err("cached miss");
    assert!(matches!(again, EngineError::ColorMiss { .. }));
    assert_eq!(export.decodes(), 1);
    assert!(matches!(export.sample(Micros(0)), Sample::Miss));

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
    let filled = export.fill(Micros(0));
    assert!(matches!(filled, Err(EngineError::ColorMiss { .. })));
    assert!(export.cached() >= 1);
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
