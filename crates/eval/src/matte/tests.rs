use std::fs::{self, File};
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_container::{Demux, Input, OpenError, Probe, ProbeResult};
use cutlass_effects::{MATTING, Pick, PickClick, Stroke, StrokeStyle};
use cutlass_engine::ReaderFactory;
use cutlass_model::{MaterialVideo, RawMaterial, Segment, TrackType};
use cutlass_renderer_software::SoftwareDevice;
use cutlass_still::{StillDemux, StillProbe, still_limits};
use cutlass_store::matte_file;

use super::*;
use crate::{OpenClips, Snapshot, frame_in};

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cutlass-eval-matte-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../engine/fixtures")
        .join(name)
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

fn stills() -> ReaderFactory {
    let mut factory = ReaderFactory::new();
    factory.register_container(
        |header: &[u8]| -> ProbeResult { StillProbe.probe(header) },
        open_still,
    );
    cutlass_png::register_decoders(factory.decoders_mut());
    factory
}

/// A grey PNG, `width` by `height`, one byte a pixel row by row, stored
/// without compression.
fn grey_png(width: u32, height: u32, grey: &[u8]) -> Vec<u8> {
    fn crc(bytes: &[u8]) -> u32 {
        let mut crc = 0xffff_ffff_u32;
        for byte in bytes {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xedb8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }
    fn chunk(out: &mut Vec<u8>, kind: &[u8], data: &[u8]) {
        out.extend_from_slice(&u32::try_from(data.len()).unwrap().to_be_bytes());
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc(&body).to_be_bytes());
    }
    let mut raw = Vec::new();
    for row in grey.chunks(width as usize) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let (mut a, mut b) = (1_u32, 0_u32);
    for byte in &raw {
        a = (a + u32::from(*byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    let mut zlib = vec![0x78, 0x01, 1];
    let len = u16::try_from(raw.len()).unwrap();
    zlib.extend_from_slice(&len.to_le_bytes());
    zlib.extend_from_slice(&(!len).to_le_bytes());
    zlib.extend_from_slice(&raw);
    zlib.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut header = Vec::new();
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, 0, 0, 0, 0]);
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    chunk(&mut out, b"IHDR", &header);
    chunk(&mut out, b"IDAT", &zlib);
    chunk(&mut out, b"IEND", &[]);
    out
}

/// A project holding `pattern.png` as `walk.png`, and a draft whose one
/// clip shows a second of it from 1 s on a 16 by 16 canvas, with
/// `matting` when there is some.
fn project(matting: Option<&Matting>) -> (Temp, Draft) {
    let temp = Temp::new();
    fs::copy(fixture("pattern.png"), temp.0.join("walk.png")).unwrap();
    let mut draft = Draft::empty("D");
    draft.fps = 10.0;
    draft.canvas_config.width = 16;
    draft.canvas_config.height = 16;
    draft.materials.videos.push(MaterialVideo {
        id: "m".into(),
        kind: "video".into(),
        path: "walk.png".into(),
        width: 16,
        height: 16,
        duration: Micros(5_000_000),
        ..MaterialVideo::default()
    });
    let mut segment = Segment::new("v", "m", TimeRange::new(2_000_000, 1_000_000));
    segment.source_range = Some(TimeRange::new(1_000_000, 1_000_000));
    if let Some(matting) = matting {
        draft.materials.common_mask.push(RawMaterial {
            id: "r".into(),
            kind: MATTING.into(),
            fields: matting.to_fields(),
        });
        segment.extra_material_refs.push("r".into());
    }
    draft.tracks[0].segments.push(segment);
    (temp, draft)
}

fn draw(draft: &Draft, root: &Path, at: i64) -> Vec<u8> {
    let mut clips = OpenClips::new();
    frame_in(
        &Snapshot::new(draft),
        Micros(at),
        &mut SoftwareDevice::new(),
        &stills(),
        &mut clips,
        0,
        Some(root),
    )
    .unwrap()
    .rgba
}

/// Writes the job's matte for `draft`'s clip: the right half kept, at
/// half the clip's size so it is stretched.
fn keep_matte(temp: &Temp, draft: &Draft) -> PathBuf {
    let ask = matte_ask(draft, Some(&temp.0), &"v".into()).unwrap();
    let file = matte_file(&temp.0, &ask.key(), &TimeRange::new(500_000, 2_500_000)).unwrap();
    let grey: Vec<u8> = (0..8 * 8)
        .map(|i| if i % 8 >= 4 { 255 } else { 0 })
        .collect();
    fs::write(&file, grey_png(8, 8, &grey)).unwrap();
    file
}

fn alpha_at(rgba: &[u8], x: usize, y: usize) -> u8 {
    rgba[(y * 16 + x) * 4 + 3]
}

#[test]
fn the_ask_names_the_model_and_the_picks() {
    let (temp, draft) = project(Some(&Matting::auto()));
    let ask = matte_ask(&draft, Some(&temp.0), &"v".into()).unwrap();
    assert_eq!(ask.source, temp.0.join("walk.png").canonicalize().unwrap());
    assert_eq!(ask.range, TimeRange::new(1_000_000, 1_000_000));
    assert_eq!(ask.rate, 10.0);
    let key = ask.key();
    assert_eq!(
        (key.model.as_str(), key.picks.as_str()),
        (MATTE_AUTO_MODEL, "")
    );
    assert_eq!(ask.frames().range, ask.range);

    let custom = Matting {
        mode: MattingMode::Custom,
        picks: vec![Pick {
            frame: Micros(1_200_000),
            clicks: vec![PickClick {
                x: 0.5,
                y: 0.5,
                keep: true,
            }],
            brushes: Vec::new(),
        }],
        ..Matting::auto()
    };
    let (temp, draft) = project(Some(&custom));
    let key = matte_ask(&draft, Some(&temp.0), &"v".into()).unwrap().key();
    assert_eq!(key.model, MATTE_FOLLOW_MODEL);
    assert_eq!(key.picks, custom.picks_key());
}

#[test]
fn no_matting_no_pick_a_graphic_or_a_file_outside_is_refused() {
    let (temp, plain) = project(None);
    let root = Some(temp.0.as_path());
    assert!(matte_ask(&plain, root, &"v".into()).is_err());
    let unpicked = Matting {
        mode: MattingMode::Custom,
        ..Matting::auto()
    };
    let (temp, draft) = project(Some(&unpicked));
    let root = Some(temp.0.as_path());
    let Err(EnhanceError::Clip { reason }) = matte_ask(&draft, root, &"v".into()) else {
        panic!("a Custom removal with no pick was asked for");
    };
    assert!(reason.contains("picked"), "{reason}");
    let (temp, draft) = project(Some(&Matting::auto()));
    let root = Some(temp.0.as_path());
    let mut text = draft.clone();
    text.tracks[0].kind = TrackType::Text;
    assert!(matte_ask(&text, root, &"v".into()).is_err());
    let mut escaping = draft;
    escaping.materials.videos[0].path = "../../etc/passwd".into();
    assert!(matte_ask(&escaping, root, &"v".into()).is_err());
}

#[test]
fn a_cut_out_clip_draws_uncut_until_its_matte_is_made_then_cut() {
    let (temp, draft) = project(Some(&Matting::auto()));
    let (_, plain) = project(None);
    let own = draw(&plain, &temp.0, 2_500_000);
    assert_eq!(draw(&draft, &temp.0, 2_500_000), own, "no matte yet: uncut");
    assert_eq!(matte_video(&draft, Some(&temp.0), &"v".into()), None);

    let file = keep_matte(&temp, &draft);
    assert_eq!(
        matte_video(&draft, Some(&temp.0), &"v".into()),
        Some((file, Micros(500_000)))
    );
    let cut = draw(&draft, &temp.0, 2_500_000);
    for y in [0, 8, 15] {
        assert_eq!(alpha_at(&cut, 1, y), 0, "the left half is removed");
        assert_eq!(alpha_at(&cut, 14, y), alpha_at(&own, 14, y));
        assert_eq!(
            &cut[(y * 16 + 14) * 4..(y * 16 + 15) * 4],
            &own[(y * 16 + 14) * 4..(y * 16 + 15) * 4],
            "the right half is the clip"
        );
    }
    // No project folder, or the clip's file changed: drawn uncut.
    assert_eq!(matte_video(&draft, None, &"v".into()), None);
    // A matte made by other picks is not this clip's.
    let other = Matting {
        mode: MattingMode::Custom,
        picks: vec![Pick {
            frame: Micros(1_000_000),
            clicks: vec![PickClick {
                x: 0.1,
                y: 0.1,
                keep: true,
            }],
            brushes: Vec::new(),
        }],
        ..Matting::auto()
    };
    let (_, picked) = project(Some(&other));
    assert_eq!(draw(&picked, &temp.0, 2_500_000), own);
}

#[test]
fn an_outline_is_drawn_under_the_kept_half() {
    let outlined = Matting {
        stroke: Some(Stroke {
            size: 1.0,
            ..Stroke::new(StrokeStyle::Solid)
        }),
        ..Matting::auto()
    };
    let (temp, draft) = project(Some(&outlined));
    keep_matte(&temp, &draft);
    let drawn = draw(&draft, &temp.0, 2_500_000);
    // The matte's soft edge is at x 7 and 8 once stretched. The band sits
    // under it and a pixel past it, and leaves the far left removed.
    assert_eq!(alpha_at(&drawn, 7, 8), 255);
    assert_eq!(alpha_at(&drawn, 8, 8), 255);
    assert_eq!(alpha_at(&drawn, 5, 8), 0);
    assert_eq!(alpha_at(&drawn, 1, 8), 0);
    // Solid's colour is #ffd23f: its red shows at full under the edge.
    assert!(
        drawn[(8 * 16 + 7) * 4] >= 191,
        "{}",
        drawn[(8 * 16 + 7) * 4]
    );
}

#[test]
fn cut_out_stretches_the_matte_and_feathers_it() {
    let picture = CpuBytes::new(8, 1, vec![255; 8 * 4]).unwrap();
    let mut matte_rgba = Vec::new();
    for value in [0_u8, 0, 255, 255] {
        matte_rgba.extend_from_slice(&[value, value, value, 255]);
    }
    let matte = CpuBytes::new(4, 1, matte_rgba).unwrap();
    let cut = cut_out(picture.clone(), &matte, &Matting::auto());
    let alpha: Vec<u8> = cut.rgba().chunks(4).map(|px| px[3]).collect();
    assert_eq!(alpha[0], 0);
    assert_eq!(alpha[7], 255);
    assert!(alpha.windows(2).all(|pair| pair[0] <= pair[1]));
    // A matte of no size leaves the picture uncut.
    let empty = CpuBytes::new(0, 0, Vec::new()).unwrap();
    assert_eq!(cut_out(picture.clone(), &empty, &Matting::auto()), picture);
}
