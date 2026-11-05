//! The sticker catalog: where each sticker comes from and its license.
//!
//! Cutlass's own pack is drawn here from signed distance shapes, as a
//! die-cut sticker: the colored shape, a white edge round it, and a soft
//! shadow under both. Each is a square transparent PNG that any app writes
//! into the project and places like a picture, so a sticker moves, keys
//! and animates like any other clip. Packs from elsewhere join the catalog
//! with their own source and license.

#![forbid(unsafe_code)]

mod draw;

/// The edge of a drawn sticker, in pixels.
pub const STICKER_SIDE: u32 = 512;

/// One sticker in the catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sticker {
    /// Stable id, kept in a project's file name.
    pub id: &'static str,
    /// The name the shelf shows.
    pub name: &'static str,
    /// Words the shelf's search also matches.
    pub tags: &'static str,
    /// Where the sticker comes from.
    pub source: &'static str,
    /// The license it is used under.
    pub license: &'static str,
    shape: draw::Shape,
}

const OWN: &str = "Cutlass";
const OWN_LICENSE: &str = "Drawn by Cutlass; free to use in anything you make";

const fn own(
    id: &'static str,
    name: &'static str,
    tags: &'static str,
    shape: draw::Shape,
) -> Sticker {
    Sticker {
        id,
        name,
        tags,
        source: OWN,
        license: OWN_LICENSE,
        shape,
    }
}

/// Every sticker the shelf offers, in shelf order.
pub const STICKERS: &[Sticker] = &[
    own("heart", "Heart", "love like", draw::Shape::Heart),
    own("star", "Star", "favourite gold", draw::Shape::Star),
    own(
        "sparkle",
        "Sparkle",
        "shine magic new",
        draw::Shape::Sparkle,
    ),
    own("arrow", "Arrow", "point look here", draw::Shape::Arrow),
    own("burst", "Burst", "wow pop boom", draw::Shape::Burst),
    own("check", "Check", "tick yes done", draw::Shape::Check),
    own("smile", "Smile", "happy face emoji", draw::Shape::Smile),
    own("bolt", "Bolt", "lightning flash energy", draw::Shape::Bolt),
];

/// The sticker with `id`.
#[must_use]
pub fn sticker(id: &str) -> Option<&'static Sticker> {
    STICKERS.iter().find(|sticker| sticker.id == id)
}

/// Stickers whose name or tags hold `query`, ignoring case. An empty
/// query is every sticker.
#[must_use]
pub fn search(query: &str) -> Vec<&'static Sticker> {
    let query = query.trim().to_lowercase();
    STICKERS
        .iter()
        .filter(|sticker| {
            query.is_empty()
                || sticker.name.to_lowercase().contains(&query)
                || sticker.tags.contains(&query)
        })
        .collect()
}

impl Sticker {
    /// The file name a project keeps this sticker under.
    #[must_use]
    pub fn file_name(&self) -> String {
        format!("sticker-{}.png", self.id)
    }

    /// The sticker drawn at `side` pixels square: straight RGBA.
    #[must_use]
    pub fn rgba(&self, side: u32) -> Vec<u8> {
        draw::rgba(self.shape, side)
    }

    /// The sticker as a PNG file, [`STICKER_SIDE`] pixels square.
    pub fn png(&self) -> Result<Vec<u8>, String> {
        use cutlass_encode::{
            CodecId, CodecParams, CpuFrame, Encoder, FrameInput, FrameTime, PixelFormat,
        };

        let side = STICKER_SIDE;
        let mut encoder =
            cutlass_png::PngEncoder::open(&CodecParams::video(CodecId::Png, side, side))
                .map_err(|error| error.to_string())?;
        let time = FrameTime::new(0, 1, 1).map_err(|error| error.to_string())?;
        let frame = CpuFrame::new(side, side, PixelFormat::Rgba8, self.rgba(side), time)
            .map_err(|error| error.to_string())?;
        let packets = encoder
            .push_frame(&FrameInput::Cpu(frame))
            .map_err(|error| error.to_string())?;
        Ok(packets.into_iter().flat_map(|packet| packet.data).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_sticker_has_an_id_a_source_and_a_license() {
        let mut ids: Vec<_> = STICKERS.iter().map(|sticker| sticker.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), STICKERS.len());
        for item in STICKERS {
            assert!(!item.source.is_empty() && !item.license.is_empty());
            assert_eq!(sticker(item.id), Some(item));
            assert!(item.file_name().ends_with(".png"));
        }
        assert_eq!(sticker("gone"), None);
    }

    #[test]
    fn search_matches_names_and_tags() {
        assert_eq!(search("").len(), STICKERS.len());
        assert_eq!(search("HEART")[0].id, "heart");
        assert_eq!(search("wow")[0].id, "burst");
        assert!(search("nothing like this").is_empty());
    }

    #[test]
    fn the_png_is_a_square_picture() {
        let png = STICKERS[0].png().unwrap();
        assert_eq!(&png[1..4], b"PNG");
        let width = u32::from_be_bytes([png[16], png[17], png[18], png[19]]);
        let height = u32::from_be_bytes([png[20], png[21], png[22], png[23]]);
        assert_eq!((width, height), (STICKER_SIDE, STICKER_SIDE));
    }
}
