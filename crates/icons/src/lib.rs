//! Icons for every shell, from one table.
//!
//! `icons.txt` names each icon and the free Hugeicons glyph it comes from.
//! `tools/icongen` reads that list, takes the path data from the
//! `@hugeicons/core-free-icons` package, and writes [`hugeicons`], the iOS
//! image sets, and the Android drawables. Desktop strokes the marks itself.
//! The phones draw the files: `Image("icon_play")` and
//! `painterResource(R.drawable.icon_play)`.
//!
//! Every mark lives in a 24 viewBox. Paint is black; each shell tints it.

#![forbid(unsafe_code)]

mod custom;
mod hugeicons;

use std::fmt::Write as _;

pub use custom::{BLADE, CUSTOM};
pub use hugeicons::ICONS;

/// The viewBox edge every mark is drawn in.
pub const VIEWBOX: f32 = 24.0;

/// The icon list, from the repository root.
pub const MANIFEST: &str = "crates/icons/icons.txt";

/// The asset catalog folder that holds one image set per icon, from the repository root.
pub const IOS_ICONS: &str = "apps/ios/ios/Assets.xcassets/Icons";

/// The folder that holds one VectorDrawable per icon, from the repository root.
pub const ANDROID_DRAWABLES: &str = "apps/android/app/src/main/res/drawable";

/// The `Contents.json` of the `Icons` folder in the asset catalog.
pub const IOS_FOLDER_CONTENTS: &str =
    "{\n  \"info\" : {\n    \"author\" : \"xcode\",\n    \"version\" : 1\n  }\n}\n";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IconData {
    /// The name the shells ask for, such as `textVideo`.
    pub name: &'static str,
    /// The Hugeicons glyph, such as `ImageToVideoIcon`. `None` for a custom mark.
    pub source: Option<&'static str>,
    pub marks: &'static [Mark],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mark {
    pub shape: Shape,
    /// Fill the shape with the tint.
    pub fill: bool,
    /// Fill by the even-odd rule instead of nonzero.
    pub even_odd: bool,
    pub stroke: Option<Stroke>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Circle {
        cx: f32,
        cy: f32,
        r: f32,
    },
    /// Absolute `M`, `L`, `H`, `V`, `C`, and `Z` only.
    Path(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stroke {
    pub width: f32,
    pub cap: Cap,
    pub join: Join,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cap {
    Butt,
    Round,
    Square,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Join {
    Miter,
    Round,
    Bevel,
}

/// The icon with this name.
pub fn get(name: &str) -> Option<&'static IconData> {
    ICONS.iter().find(|icon| icon.name == name)
}

/// The file name both phones use: `textVideo` becomes `icon_text_video`.
pub fn asset_name(name: &str) -> String {
    let mut asset = String::from("icon_");
    for character in name.chars() {
        if character.is_ascii_uppercase() {
            asset.push('_');
            asset.push(character.to_ascii_lowercase());
        } else if character == '-' {
            asset.push('_');
        } else {
            asset.push(character);
        }
    }
    asset
}

/// The SVG an iOS image set holds.
pub fn svg(icon: &IconData) -> String {
    let mut out = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"24\" height=\"24\" viewBox=\"0 0 24 24\" fill=\"none\">\n",
    );
    for mark in icon.marks {
        let _ = match mark.shape {
            Shape::Circle { cx, cy, r } => {
                write!(out, "  <circle cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\"")
            }
            Shape::Path(data) => write!(out, "  <path d=\"{data}\""),
        };
        if mark.fill {
            out.push_str(" fill=\"#000000\"");
            if mark.even_odd {
                out.push_str(" fill-rule=\"evenodd\"");
            }
        }
        if let Some(stroke) = mark.stroke {
            let _ = write!(out, " stroke=\"#000000\" stroke-width=\"{}\"", stroke.width);
            match stroke.cap {
                Cap::Butt => {}
                Cap::Round => out.push_str(" stroke-linecap=\"round\""),
                Cap::Square => out.push_str(" stroke-linecap=\"square\""),
            }
            match stroke.join {
                Join::Miter => {}
                Join::Round => out.push_str(" stroke-linejoin=\"round\""),
                Join::Bevel => out.push_str(" stroke-linejoin=\"bevel\""),
            }
        }
        out.push_str("/>\n");
    }
    out.push_str("</svg>\n");
    out
}

/// The `Contents.json` beside the SVG. The image keeps its vector and takes the tint.
pub fn imageset_contents(asset: &str) -> String {
    format!(
        "{{\n  \"images\" : [\n    {{\n      \"filename\" : \"{asset}.svg\",\n      \"idiom\" : \"universal\"\n    }}\n  ],\n  \"info\" : {{\n    \"author\" : \"xcode\",\n    \"version\" : 1\n  }},\n  \"properties\" : {{\n    \"preserves-vector-representation\" : true,\n    \"template-rendering-intent\" : \"template\"\n  }}\n}}\n"
    )
}

/// The Android VectorDrawable. A circle becomes two arcs.
pub fn vector_drawable(icon: &IconData) -> String {
    let mut out = String::from(
        "<vector xmlns:android=\"http://schemas.android.com/apk/res/android\"\n    android:width=\"24dp\"\n    android:height=\"24dp\"\n    android:viewportWidth=\"24\"\n    android:viewportHeight=\"24\">\n",
    );
    for mark in icon.marks {
        let data = match mark.shape {
            Shape::Circle { cx, cy, r } => format!(
                "M{} {}A{r} {r} 0 1 0 {} {}A{r} {r} 0 1 0 {} {}Z",
                cx - r,
                cy,
                cx + r,
                cy,
                cx - r,
                cy
            ),
            Shape::Path(data) => data.to_owned(),
        };
        let _ = write!(out, "  <path\n      android:pathData=\"{data}\"");
        if mark.fill {
            out.push_str("\n      android:fillColor=\"#FF000000\"");
            if mark.even_odd {
                out.push_str("\n      android:fillType=\"evenOdd\"");
            }
        }
        if let Some(stroke) = mark.stroke {
            let _ = write!(
                out,
                "\n      android:strokeColor=\"#FF000000\"\n      android:strokeWidth=\"{}\"",
                stroke.width
            );
            match stroke.cap {
                Cap::Butt => {}
                Cap::Round => out.push_str("\n      android:strokeLineCap=\"round\""),
                Cap::Square => out.push_str("\n      android:strokeLineCap=\"square\""),
            }
            match stroke.join {
                Join::Miter => {}
                Join::Round => out.push_str("\n      android:strokeLineJoin=\"round\""),
                Join::Bevel => out.push_str("\n      android:strokeLineJoin=\"bevel\""),
            }
        }
        out.push_str(" />\n");
    }
    out.push_str("</vector>\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_names_are_lower_snake_case() {
        assert_eq!(asset_name("play"), "icon_play");
        assert_eq!(asset_name("textVideo"), "icon_text_video");
        assert_eq!(asset_name("zoomFit"), "icon_zoom_fit");
    }

    #[test]
    fn the_blade_fills_and_a_missing_name_is_none() {
        let blade = get("blade").expect("blade");
        assert!(blade.marks.iter().all(|mark| mark.fill));
        assert!(get("not-an-icon").is_none());
    }

    #[test]
    fn a_circle_and_a_stroke_reach_both_files() {
        let icon = IconData {
            name: "dot",
            source: None,
            marks: &[Mark {
                shape: Shape::Circle {
                    cx: 7.5,
                    cy: 7.5,
                    r: 1.5,
                },
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Bevel,
                }),
            }],
        };
        let svg = svg(&icon);
        assert!(svg.contains("<circle cx=\"7.5\" cy=\"7.5\" r=\"1.5\" stroke=\"#000000\" stroke-width=\"1.5\" stroke-linecap=\"round\" stroke-linejoin=\"bevel\"/>"));
        let drawable = vector_drawable(&icon);
        assert!(
            drawable
                .contains("android:pathData=\"M6 7.5A1.5 1.5 0 1 0 9 7.5A1.5 1.5 0 1 0 6 7.5Z\"")
        );
        assert!(drawable.contains("android:strokeLineCap=\"round\""));
        assert!(!drawable.contains("fillColor"));
    }
}
