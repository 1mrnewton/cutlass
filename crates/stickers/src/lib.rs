//! The sticker catalog: where each sticker comes from and its license.
//!
//! Cutlass's own pack is drawn here from signed distance shapes, as a
//! die-cut sticker: the colored shape, a white edge round it, and a soft
//! shadow under both. Open sets ship as their own files inside the app:
//! Noto Emoji pictures (PNG) and Noto Animated Emoji (Lottie). Any app
//! writes a sticker's file into the project and places it like a picture,
//! so a sticker moves, keys and animates like any other clip, and an
//! animated one plays round for as long as its clip lasts. [`CREDITS`]
//! names each open set for the app's licenses screen.

#![forbid(unsafe_code)]

mod bundled;
mod draw;

use std::collections::HashMap;
use std::sync::OnceLock;

/// The edge of a drawn sticker, in pixels.
pub const STICKER_SIDE: u32 = 512;

/// What a sticker's file is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Drawn here, written as a PNG.
    Drawn,
    /// A bundled PNG picture.
    Picture,
    /// A bundled Lottie animation, written as `.json`.
    Animation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Art {
    Drawn(draw::Shape),
    Png(&'static [u8]),
    Lottie(&'static [u8]),
}

/// One sticker in the catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sticker {
    /// Stable id, kept in a project's file name.
    pub id: &'static str,
    /// The name the shelf shows.
    pub name: &'static str,
    /// Words the shelf's search also matches.
    pub tags: &'static str,
    /// The set it belongs to; an open set has its line in [`CREDITS`].
    pub set: &'static str,
    /// Where the sticker comes from: for an open set, the file's address.
    pub source: &'static str,
    /// The license it is used under.
    pub license: &'static str,
    art: Art,
}

/// One open set's credit, for the licenses screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Credit {
    pub set: &'static str,
    pub author: &'static str,
    /// The license's SPDX id.
    pub license: &'static str,
    /// The set's home.
    pub link: &'static str,
    /// The license text, word for word.
    pub text: &'static str,
}

const OWN: &str = "Cutlass";
const OWN_LICENSE: &str = "Drawn by Cutlass; free to use in anything you make";
const NOTO: &str = "Noto Emoji";
const NOTO_LICENSE: &str = "Apache-2.0";
const NOTO_ANIMATED: &str = "Noto Animated Emoji";
const NOTO_ANIMATED_LICENSE: &str = "CC-BY-4.0";

/// The open sets the catalog ships, with their credit.
pub const CREDITS: &[Credit] = &[
    Credit {
        set: NOTO,
        author: "Google",
        license: NOTO_LICENSE,
        link: "https://github.com/googlefonts/noto-emoji",
        text: include_str!("../assets/noto-emoji/LICENSE"),
    },
    Credit {
        set: NOTO_ANIMATED,
        author: "Google",
        license: NOTO_ANIMATED_LICENSE,
        link: "https://googlefonts.github.io/noto-emoji-animation/",
        text: include_str!("../assets/noto-emoji-animation/LICENSE"),
    },
];

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
        set: OWN,
        source: OWN,
        license: OWN_LICENSE,
        art: Art::Drawn(shape),
    }
}

/// The Noto Emoji commit the pictures come from, when the repository's
/// images were under Apache 2.0.
macro_rules! noto_png_folder {
    () => {
        "https://raw.githubusercontent.com/googlefonts/noto-emoji/41e31b110b4eb929dffb410264694a06205b7ad7/png/512/"
    };
}

/// A Noto Emoji picture by its upstream codepoint.
macro_rules! noto {
    ($code:literal, $name:literal, $tags:literal) => {
        Sticker {
            id: concat!("noto-", $code),
            name: $name,
            tags: concat!($tags, " emoji"),
            set: NOTO,
            source: concat!(noto_png_folder!(), "emoji_u", $code, ".png"),
            license: NOTO_LICENSE,
            art: Art::Png(include_bytes!(concat!(
                "../assets/noto-emoji/emoji_u",
                $code,
                ".png"
            ))),
        }
    };
}

/// A Noto Animated Emoji by its upstream codepoint.
macro_rules! noto_animated {
    ($code:literal, $name:literal, $tags:literal) => {
        Sticker {
            id: concat!("noto-anim-", $code),
            name: $name,
            tags: concat!($tags, " emoji animated moving"),
            set: NOTO_ANIMATED,
            source: concat!(
                "https://fonts.gstatic.com/s/e/notoemoji/latest/",
                $code,
                "/lottie.json"
            ),
            license: NOTO_ANIMATED_LICENSE,
            art: Art::Lottie(include_bytes!(concat!(
                "../assets/noto-emoji-animation/emoji_u",
                $code,
                ".json"
            ))),
        }
    };
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
    noto_animated!("1f525", "Animated fire", "lit hot flame"),
    noto_animated!("2764_fe0f", "Animated heart", "love like red"),
    noto_animated!("2728", "Animated sparkles", "shine magic new"),
    noto_animated!("1f4af", "Animated 100", "hundred points perfect score"),
    noto_animated!("1f44d", "Animated thumbs up", "like yes good ok"),
    noto_animated!("1f44f", "Animated clap", "applause bravo hands"),
    noto_animated!("1f389", "Animated party popper", "celebrate congrats tada"),
    noto_animated!("1f602", "Animated tears of joy", "laugh lol funny face"),
    noto_animated!("1f60d", "Animated heart eyes", "love crush adore face"),
    noto_animated!("1f440", "Animated eyes", "look see watch"),
    noto_animated!("1f914", "Animated thinking", "hmm think wonder face"),
    noto_animated!("1f680", "Animated rocket", "launch fast space"),
    noto_animated!("1f973", "Animated party face", "celebrate birthday"),
    noto_animated!("1f4a5", "Animated boom", "collision bang pop"),
    noto_animated!("1f44b", "Animated wave", "hello hi bye hand"),
    noto_animated!("1f929", "Animated star struck", "wow amazed face"),
    noto!("2764", "Red heart", "love like"),
    noto!("1f525", "Fire", "lit hot flame"),
    noto!("2b50", "Gold star", "favourite"),
    noto!("2728", "Sparkles", "shine magic new"),
    noto!("1f4af", "100", "hundred points perfect score"),
    noto!("1f44d", "Thumbs up", "like yes good ok"),
    noto!("1f44f", "Clap", "applause bravo hands"),
    noto!("1f389", "Party popper", "celebrate congrats tada"),
    noto!("1f602", "Tears of joy", "laugh lol funny face"),
    noto!("1f60d", "Heart eyes", "love crush adore face"),
    noto!("1f440", "Eyes", "look see watch"),
    noto!("1f914", "Thinking", "hmm think wonder face"),
    noto!("2705", "Check mark", "tick yes done"),
    noto!("274c", "Cross mark", "no wrong x"),
    noto!("27a1", "Right arrow", "point next"),
    noto!("2b05", "Left arrow", "point back"),
    noto!("2b06", "Up arrow", "point"),
    noto!("2b07", "Down arrow", "point"),
    noto!("26a0", "Warning", "caution alert sign"),
    noto!("1f4a1", "Light bulb", "idea tip"),
    noto!("1f680", "Rocket", "launch fast space"),
    noto!("1f4b0", "Money bag", "cash rich dollar"),
    noto!("1f381", "Gift", "present giveaway"),
    noto!("1f3b5", "Music note", "song sound"),
    noto!("1f4f7", "Camera", "photo picture"),
    noto!("1f451", "Crown", "king queen royal"),
    noto!("1f3c6", "Trophy", "win winner award"),
    noto!("1f60e", "Sunglasses", "cool face"),
    noto!("1f4a5", "Boom", "collision bang pop"),
    noto!("1f44b", "Wave", "hello hi bye hand"),
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
    /// What the sticker's file is.
    #[must_use]
    pub fn kind(&self) -> Kind {
        match self.art {
            Art::Drawn(_) => Kind::Drawn,
            Art::Png(_) => Kind::Picture,
            Art::Lottie(_) => Kind::Animation,
        }
    }

    /// The file name a project keeps this sticker under: `.json` for an
    /// animation, `.png` for the rest.
    #[must_use]
    pub fn file_name(&self) -> String {
        let extension = match self.art {
            Art::Lottie(_) => "json",
            Art::Drawn(_) | Art::Png(_) => "png",
        };
        format!("sticker-{}.{extension}", self.id)
    }

    /// The file a project keeps: a drawn sticker as a PNG [`STICKER_SIDE`]
    /// pixels square, a bundled one as shipped.
    pub fn bytes(&self) -> Result<Vec<u8>, String> {
        match self.art {
            Art::Drawn(shape) => drawn_png(shape),
            Art::Png(bytes) | Art::Lottie(bytes) => Ok(bytes.to_vec()),
        }
    }

    /// The picture's width and height, in its own pixels.
    pub fn size(&self) -> Result<(u32, u32), String> {
        match self.art {
            Art::Drawn(_) => Ok((STICKER_SIDE, STICKER_SIDE)),
            Art::Png(png) => bundled::png_size(png),
            Art::Lottie(json) => {
                let animation = bundled::animation(json)?;
                let edge = |value: f64| value.round().clamp(1.0, f64::from(u32::MAX)) as u32;
                Ok((edge(animation.width()), edge(animation.height())))
            }
        }
    }

    /// One play-through of an animation, in seconds. `None` for a still
    /// or an animation that does not read. Each file is read once, the
    /// first time any is asked for, so the shelf can ask every frame.
    #[must_use]
    pub fn seconds(&self) -> Option<f64> {
        static SECONDS: OnceLock<HashMap<&'static str, f64>> = OnceLock::new();
        if !matches!(self.art, Art::Lottie(_)) {
            return None;
        }
        SECONDS
            .get_or_init(|| {
                STICKERS
                    .iter()
                    .filter_map(|sticker| match sticker.art {
                        Art::Lottie(json) => bundled::animation(json)
                            .ok()
                            .map(|animation| (sticker.id, animation.duration())),
                        Art::Drawn(_) | Art::Png(_) => None,
                    })
                    .collect()
            })
            .get(self.id)
            .copied()
    }

    /// The sticker `side` pixels square for the shelf, straight RGBA: an
    /// animation shows its middle frame.
    pub fn rgba(&self, side: u32) -> Result<Vec<u8>, String> {
        match self.art {
            Art::Drawn(shape) => Ok(draw::rgba(shape, side)),
            Art::Png(png) => {
                let (width, height, rgba) = bundled::png_rgba(png)?;
                if width != height {
                    return Err("the sticker is not square".to_owned());
                }
                Ok(bundled::resample(&rgba, width, side))
            }
            Art::Lottie(json) => bundled::lottie_poster(json, side),
        }
    }
}

/// A drawn sticker as a PNG file, [`STICKER_SIDE`] pixels square.
fn drawn_png(shape: draw::Shape) -> Result<Vec<u8>, String> {
    use cutlass_encode::{
        CodecId, CodecParams, CpuFrame, Encoder, FrameInput, FrameTime, PixelFormat,
    };

    let side = STICKER_SIDE;
    let mut encoder = cutlass_png::PngEncoder::open(&CodecParams::video(CodecId::Png, side, side))
        .map_err(|error| error.to_string())?;
    let time = FrameTime::new(0, 1, 1).map_err(|error| error.to_string())?;
    let frame = CpuFrame::new(
        side,
        side,
        PixelFormat::Rgba8,
        draw::rgba(shape, side),
        time,
    )
    .map_err(|error| error.to_string())?;
    let packets = encoder
        .push_frame(&FrameInput::Cpu(frame))
        .map_err(|error| error.to_string())?;
    Ok(packets.into_iter().flat_map(|packet| packet.data).collect())
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
            let extension = if item.kind() == Kind::Animation {
                ".json"
            } else {
                ".png"
            };
            assert!(item.file_name().ends_with(extension), "{}", item.id);
            assert!(
                item.set == OWN || CREDITS.iter().any(|credit| credit.set == item.set),
                "{} has no credit",
                item.id
            );
        }
        assert_eq!(sticker("gone"), None);
    }

    #[test]
    fn the_open_sets_are_all_there_with_their_source_and_license() {
        let of = |kind| STICKERS.iter().filter(move |item| item.kind() == kind);
        assert_eq!(of(Kind::Drawn).count(), 8);
        assert_eq!(of(Kind::Picture).count(), 30);
        assert_eq!(of(Kind::Animation).count(), 16);
        for item in of(Kind::Picture) {
            let code = item.id.strip_prefix("noto-").unwrap();
            assert_eq!(item.set, "Noto Emoji");
            assert_eq!(item.license, "Apache-2.0");
            assert!(
                item.source.starts_with(
                    "https://raw.githubusercontent.com/googlefonts/noto-emoji/41e31b11"
                )
            );
            assert!(
                item.source
                    .ends_with(&format!("/png/512/emoji_u{code}.png"))
            );
        }
        for item in of(Kind::Animation) {
            let code = item.id.strip_prefix("noto-anim-").unwrap();
            assert_eq!(item.set, "Noto Animated Emoji");
            assert_eq!(item.license, "CC-BY-4.0");
            assert_eq!(
                item.source,
                format!("https://fonts.gstatic.com/s/e/notoemoji/latest/{code}/lottie.json")
            );
        }
    }

    #[test]
    fn every_credit_carries_its_license_text() {
        assert_eq!(CREDITS.len(), 2);
        let apache = &CREDITS[0];
        assert!(apache.text.contains("Apache License"));
        assert!(apache.text.contains("Version 2.0"));
        let by = &CREDITS[1];
        assert!(by.text.contains("Attribution 4.0 International"));
        for credit in CREDITS {
            assert!(credit.link.starts_with("https://"));
            assert!(!credit.author.is_empty());
        }
    }

    #[test]
    fn search_matches_names_and_tags() {
        assert_eq!(search("").len(), STICKERS.len());
        assert_eq!(search("HEART")[0].id, "heart");
        assert_eq!(search("wow")[0].id, "burst");
        assert!(search("nothing like this").is_empty());
        let fire: Vec<_> = search("fire").iter().map(|item| item.id).collect();
        assert_eq!(fire, ["noto-anim-1f525", "noto-1f525"]);
        assert!(
            search("animated")
                .iter()
                .all(|item| item.kind() == Kind::Animation)
        );
        assert_eq!(search("Animated").len(), 16);
        assert_eq!(search("rocket").len(), 2);
    }

    #[test]
    fn the_png_is_a_square_picture() {
        let png = STICKERS[0].bytes().unwrap();
        assert_eq!(&png[1..4], b"PNG");
        assert_eq!(STICKERS[0].size(), Ok((STICKER_SIDE, STICKER_SIDE)));
        assert_eq!(bundled::png_size(&png), Ok((STICKER_SIDE, STICKER_SIDE)));
        assert_eq!(STICKERS[0].seconds(), None);
    }

    #[test]
    fn every_bundled_picture_is_a_square_png_that_draws() {
        for item in STICKERS.iter().filter(|item| item.kind() == Kind::Picture) {
            let bytes = item.bytes().unwrap();
            assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "{}", item.id);
            assert_eq!(item.size(), Ok((512, 512)), "{}", item.id);
            assert_eq!(item.seconds(), None);
            let poster = item.rgba(32).unwrap();
            assert_eq!(poster.len(), 32 * 32 * 4);
            let seen = poster
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|pixel| pixel[3] > 200)
                .count();
            assert!(seen > 32 * 32 / 8, "{} is nearly clear: {seen}", item.id);
            // Its corner is clear, as an emoji's is.
            assert_eq!(poster[3], 0, "{}", item.id);
        }
    }

    #[test]
    fn every_bundled_animation_reads_draws_and_moves() {
        for item in STICKERS
            .iter()
            .filter(|item| item.kind() == Kind::Animation)
        {
            let bytes = item.bytes().unwrap();
            let animation = cutlass_lottie::Animation::parse(&bytes).unwrap();
            let seconds = item.seconds().unwrap();
            assert!((0.3..=5.0).contains(&seconds), "{} {seconds}", item.id);
            let (width, height) = item.size().unwrap();
            assert!(width >= 512 && width == height, "{}", item.id);
            let poster = item.rgba(48).unwrap();
            let seen = poster
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|pixel| pixel[3] > 200)
                .count();
            assert!(seen > 48 * 48 / 10, "{} is nearly clear: {seen}", item.id);
            // Two moments a quarter of a round apart are not the same picture.
            let at = |share: f64| {
                let frame = animation.frame_at(seconds * share);
                bundled::frame_rgba(&animation, frame, 48).unwrap()
            };
            assert_ne!(at(0.25), at(0.5), "{} does not move", item.id);
        }
    }

    #[test]
    fn the_bundled_files_stay_small() {
        let total: usize = STICKERS
            .iter()
            .filter(|item| item.kind() != Kind::Drawn)
            .map(|item| item.bytes().unwrap().len())
            .sum();
        assert!(total < 3 * 1024 * 1024, "{total} bytes");
    }
}
