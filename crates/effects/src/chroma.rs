//! Chroma key: a colour cut out of a clip, so what is below shows through.
//!
//! The key is a `chroma` record in the draft's `chromas` pool that the clip
//! lists among its extra materials, with `color` (`#rrggbb`), `strength`
//! and `shadow`, each from 0 to 1. Strength is how far from the key colour
//! a pixel may be and still go; shadow keeps the darker parts of the key
//! colour, such as a person's shadow on a green wall. The colour left on
//! an edge has the key colour pulled out of it.

use cutlass_model::{Draft, MaterialRef, Segment};

/// The material `type` of a chroma key.
pub const CHROMA: &str = "chroma";

/// One clip's chroma key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chroma {
    /// The key colour, 0 to 1 a channel.
    pub color: [f32; 3],
    pub strength: f32,
    pub shadow: f32,
}

impl Chroma {
    /// The key colour as `#rrggbb`.
    #[must_use]
    pub fn hex(&self) -> String {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let byte = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        format!(
            "#{:02x}{:02x}{:02x}",
            byte(self.color[0]),
            byte(self.color[1]),
            byte(self.color[2])
        )
    }

    /// Cut the key out of premultiplied RGBA.
    pub fn apply(&self, rgba: &mut [u8]) {
        if self.strength <= 0.0 {
            return;
        }
        let key = chroma_of(self.color);
        let key_luma = luma(self.color);
        // How far from the key, in chroma for its brightness, a pixel is
        // cut out entirely, and how wide the soft edge past that is.
        let reach = 0.05 + self.strength.clamp(0.0, 1.0) * 0.6;
        let soft = 0.15;
        for pixel in rgba.as_chunks_mut::<4>().0 {
            let alpha = f32::from(pixel[3]) / 255.0;
            if alpha <= 0.0 {
                continue;
            }
            let colour = [0, 1, 2].map(|index| (f32::from(pixel[index]) / 255.0 / alpha).min(1.0));
            let here = chroma_of(colour);
            let distance = ((here[0] - key[0]).powi(2) + (here[1] - key[1]).powi(2)).sqrt();
            let mut keep = smoothstep(reach, reach + soft, distance);
            if self.shadow > 0.0 && keep < 1.0 {
                let darker = ((key_luma - luma(colour)) / key_luma.max(1e-3)).clamp(0.0, 1.0);
                keep = keep.max(darker * self.shadow);
            }
            if keep >= 1.0 {
                continue;
            }
            // Pull the key colour out of what stays.
            let spill = 1.0 - keep;
            let grey = luma(colour);
            let out_alpha = alpha * keep;
            for index in 0..3 {
                let excess = (colour[index] - grey) * (self.color[index] - luma(self.color));
                let cleaned = if excess > 0.0 {
                    grey + (colour[index] - grey) * (1.0 - spill)
                } else {
                    colour[index]
                };
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let byte = (cleaned.clamp(0.0, 1.0) * out_alpha * 255.0).round() as u8;
                pixel[index] = byte;
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let byte = (out_alpha * 255.0).round() as u8;
            pixel[3] = byte;
        }
    }
}

/// The chroma key `segment` lists, when it has one with a readable colour.
#[must_use]
pub fn segment_chroma(draft: &Draft, segment: &Segment) -> Option<Chroma> {
    segment.extra_material_refs.iter().find_map(|id| {
        let Some(MaterialRef::Other(raw)) = draft.materials.get(id.as_str()) else {
            return None;
        };
        if raw.kind != CHROMA {
            return None;
        }
        let color = parse_hex(raw.fields.get("color")?.as_str()?)?;
        #[allow(clippy::cast_possible_truncation)]
        let unit = |name: &str| {
            raw.fields
                .get(name)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or_default()
                .clamp(0.0, 1.0) as f32
        };
        Some(Chroma {
            color,
            strength: unit("strength"),
            shadow: unit("shadow"),
        })
    })
}

/// `#rrggbb` as 0 to 1 a channel.
#[must_use]
pub fn parse_hex(text: &str) -> Option<[f32; 3]> {
    let digits = text.strip_prefix('#')?;
    if digits.len() != 6 || !digits.is_ascii() {
        return None;
    }
    let channel = |at: usize| {
        u8::from_str_radix(&digits[at..at + 2], 16)
            .ok()
            .map(|byte| f32::from(byte) / 255.0)
    };
    Some([channel(0)?, channel(2)?, channel(4)?])
}

fn luma([r, g, b]: [f32; 3]) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// The colour without its brightness: blue and red difference, over the
/// brightness, so a dark green is as green as a bright one. Near black the
/// brightness is held, so noise there does not look like colour.
fn chroma_of(colour: [f32; 3]) -> [f32; 2] {
    let y = luma(colour);
    let scale = y.max(0.1);
    [
        (colour[2] - y) * 0.539 / scale,
        (colour[0] - y) * 0.635 / scale,
    ]
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialId, Micros, RawMaterial, Segment, TimeRange};
    use serde_json::json;

    use super::{CHROMA, Chroma, parse_hex, segment_chroma};

    const GREEN: Chroma = Chroma {
        color: [0.0, 1.0, 0.0],
        strength: 0.5,
        shadow: 0.0,
    };

    fn run(chroma: &Chroma, pixel: [u8; 4]) -> [u8; 4] {
        let mut bytes = pixel.to_vec();
        chroma.apply(&mut bytes);
        [bytes[0], bytes[1], bytes[2], bytes[3]]
    }

    #[test]
    fn the_key_colour_goes_and_other_colours_stay() {
        assert_eq!(run(&GREEN, [0, 255, 0, 255])[3], 0);
        assert_eq!(run(&GREEN, [20, 220, 30, 255])[3], 0);
        assert_eq!(run(&GREEN, [200, 40, 40, 255]), [200, 40, 40, 255]);
        assert_eq!(run(&GREEN, [128, 128, 128, 255]), [128, 128, 128, 255]);
        let none = Chroma {
            strength: 0.0,
            ..GREEN
        };
        assert_eq!(run(&none, [0, 255, 0, 255]), [0, 255, 0, 255]);
    }

    #[test]
    fn shadow_keeps_the_darker_key_and_an_edge_loses_its_green() {
        let shadowed = Chroma {
            shadow: 1.0,
            ..GREEN
        };
        let dark = run(&shadowed, [0, 90, 0, 255]);
        assert!(dark[3] > 100, "{dark:?}");
        assert_eq!(run(&GREEN, [0, 90, 0, 255])[3], 0);
        // A greenish grey is half kept, with its green pulled down.
        let edge = run(&GREEN, [70, 180, 70, 255]);
        assert!(edge[3] < 255, "{edge:?}");
        assert!(edge[3] > 0, "{edge:?}");
        assert!(
            i32::from(edge[1]) - i32::from(edge[0]) < 110 * i32::from(edge[3]) / 255,
            "{edge:?}"
        );
    }

    #[test]
    fn the_draft_holds_the_key() {
        assert_eq!(parse_hex("#00ff00"), Some([0.0, 1.0, 0.0]));
        assert_eq!(parse_hex("00ff00"), None);
        assert_eq!(parse_hex("#00ff0"), None);
        let mut draft = Draft::empty("D");
        let mut raw = RawMaterial {
            id: MaterialId::new("k"),
            kind: CHROMA.to_owned(),
            ..RawMaterial::default()
        };
        raw.fields.insert("color".to_owned(), json!("#0000ff"));
        raw.fields.insert("strength".to_owned(), json!(0.4));
        raw.fields.insert("shadow".to_owned(), json!(3.0));
        draft.materials.chromas.push(raw);
        let mut segment = Segment::new("s", "m", TimeRange::new(0, 1));
        segment.extra_material_refs.push(MaterialId::new("k"));
        let chroma = segment_chroma(&draft, &segment).unwrap();
        assert_eq!(chroma.hex(), "#0000ff");
        assert!((chroma.strength - 0.4).abs() < 1e-6);
        assert_eq!(chroma.shadow, 1.0);

        // Keys move the intensity and the shadow apart, held to 0 to 1.
        segment.target = TimeRange::new(0, 2_000);
        cutlass_model::put_key(&mut segment, "KFTypeChromaIntensity", Micros(0), 0.0);
        cutlass_model::put_key(&mut segment, "KFTypeChromaIntensity", Micros(2_000), 1.0);
        cutlass_model::put_key(&mut segment, "KFTypeChromaShadow", Micros(0), 5.0);
        let keyed = crate::segment_colour_at(&draft, &segment, Micros(1_000))
            .chroma
            .unwrap();
        assert!((keyed.strength - 0.5).abs() < 1e-6);
        assert_eq!(keyed.shadow, 1.0);
        assert_eq!(keyed.hex(), "#0000ff");
        // Without a key the stored setting holds.
        let plain = crate::segment_colour(&draft, &segment).chroma.unwrap();
        assert!((plain.strength - 0.4).abs() < 1e-6);
    }
}
