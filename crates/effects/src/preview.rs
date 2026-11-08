//! What an effect's card shows: the effect running over a small scene.
//!
//! The app ships a handful of scenes, each picked to show some effects at
//! their best: a face for colour and glow, a lit street at night for
//! glitches, a road for zooms. A recipe names its scene. A card shows one
//! still of it, and while the pointer rests on the card it plays the effect
//! in a loop at a low rate, as a short clip would.
//!
//! The pictures are drawn by the same GPU blocks as the timeline, so a
//! preview can never drift from what the effect does.

use std::collections::HashMap;
use std::sync::OnceLock;

use cutlass_decode::{CodecId, CodecParams, ColorType, Decoder, Packet};
use cutlass_jpeg::JpegDecoder;
use cutlass_model::Micros;

/// The scenes, by name, as the JPEG files the app ships.
const SCENE_FILES: [(&str, &[u8]); 15] = [
    ("portrait", include_bytes!("../previews/portrait.jpg")),
    ("city", include_bytes!("../previews/city.jpg")),
    ("beach", include_bytes!("../previews/beach.jpg")),
    ("dancer", include_bytes!("../previews/dancer.jpg")),
    ("road", include_bytes!("../previews/road.jpg")),
    ("flowers", include_bytes!("../previews/flowers.jpg")),
    ("skater", include_bytes!("../previews/skater.jpg")),
    ("lake", include_bytes!("../previews/lake.jpg")),
    ("cafe", include_bytes!("../previews/cafe.jpg")),
    ("concert", include_bytes!("../previews/concert.jpg")),
    ("food", include_bytes!("../previews/food.jpg")),
    ("street", include_bytes!("../previews/street.jpg")),
    ("park", include_bytes!("../previews/park.jpg")),
    ("bridge", include_bytes!("../previews/bridge.jpg")),
    ("picnic", include_bytes!("../previews/picnic.jpg")),
];

/// The scene a recipe that names none is shown over.
pub const DEFAULT_PREVIEW_SCENE: &str = "portrait";

/// Steps a second a playing preview shows: enough to read the motion,
/// few enough to stay a glance.
pub const PREVIEW_RATE: u32 = 12;

/// How long a preview plays before it starts again.
pub const PREVIEW_LOOP: Micros = Micros(3_000_000);

/// The moment a card's still is taken from, when the recipe names none.
pub const DEFAULT_POSTER: f32 = 0.5;

/// A scene's names, in the order they ship.
#[must_use]
pub fn preview_scenes() -> Vec<&'static str> {
    SCENE_FILES.iter().map(|(name, _)| *name).collect()
}

/// One scene as opaque RGBA, top row first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreviewScene {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// The scene called `name`, decoded once and kept.
#[must_use]
pub fn preview_scene(name: &str) -> Option<&'static PreviewScene> {
    static SCENES: OnceLock<HashMap<&'static str, PreviewScene>> = OnceLock::new();
    SCENES
        .get_or_init(|| {
            SCENE_FILES
                .iter()
                .filter_map(|(name, bytes)| Some((*name, decode(bytes)?)))
                .collect()
        })
        .get(name)
}

/// The effect's own time a preview playing for `elapsed` shows: held for
/// a step at [`PREVIEW_RATE`], and back to the start every
/// [`PREVIEW_LOOP`].
#[must_use]
pub fn preview_time(elapsed: Micros) -> Micros {
    let step = 1_000_000 / i64::from(PREVIEW_RATE);
    let looped = elapsed.0.max(0) % PREVIEW_LOOP.0;
    Micros(looped - looped % step)
}

fn decode(bytes: &[u8]) -> Option<PreviewScene> {
    let mut decoder = JpegDecoder::open(&CodecParams::video(CodecId::Jpeg, 0, 0)).ok()?;
    let picture = decoder
        .decode_picture(&Packet::audio(bytes.to_vec(), 0, 0, 0))
        .ok()?
        .into_iter()
        .next()?
        .image;
    let rgba = match picture.color {
        ColorType::Rgb => picture
            .samples
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|[r, g, b]| [*r as u8, *g as u8, *b as u8, 255])
            .collect(),
        ColorType::Gray => picture
            .samples
            .iter()
            .flat_map(|value| [*value as u8, *value as u8, *value as u8, 255])
            .collect(),
        _ => return None,
    };
    Some(PreviewScene {
        width: picture.width,
        height: picture.height,
        rgba,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scene_decodes_at_the_card_size() {
        for name in preview_scenes() {
            let scene = preview_scene(name).unwrap_or_else(|| panic!("{name} decodes"));
            assert_eq!((scene.width, scene.height), (288, 192), "{name}");
            assert_eq!(scene.rgba.len(), 288 * 192 * 4);
        }
        assert!(preview_scene(DEFAULT_PREVIEW_SCENE).is_some());
        assert!(preview_scene("none").is_none());
    }

    #[test]
    fn a_preview_steps_at_its_rate_and_loops() {
        let step = 1_000_000 / i64::from(PREVIEW_RATE);
        assert_eq!(preview_time(Micros(0)), Micros(0));
        assert_eq!(preview_time(Micros(step - 1)), Micros(0));
        assert_eq!(preview_time(Micros(step + 5)), Micros(step));
        assert_eq!(preview_time(PREVIEW_LOOP), Micros(0));
        assert_eq!(preview_time(Micros(-4)), Micros(0));
    }
}
