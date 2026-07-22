#![allow(unused_imports)]

use std::cell::Cell;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use cutlass_engine::EngineConfig;
use slint::ComponentHandle;
use slint::Global;
use slint::Model;
use slint::ModelRc;
use slint::SharedString;
use slint::VecModel;

/// Library palette key → engine generator, for drag-drops from the Library's
/// Titles/Shapes tabs. Content and size are placeholders the user edits next
/// (via the inspector); `None` for unknown keys.
pub(crate) fn generator_from_key(key: &str) -> Option<cutlass_models::Generator> {
    use cutlass_models::{Generator, Shape};
    if let Some(asset) = key.strip_prefix("sticker:") {
        // Only catalog ids drop; a stale key would place an invisible clip.
        cutlass_models::sticker_spec(asset)?;
        return Some(Generator::sticker(asset));
    }
    // A standalone effect-lane segment; the id after the prefix seeds the
    // clip's effect chain (validated by the engine's AddEffect).
    if key.strip_prefix("effect:").is_some() {
        return Some(Generator::Effect);
    }
    Some(match key {
        "text" => Generator::text("Title"),
        "solid" => Generator::SolidColor {
            rgba: [30, 30, 30, 255],
        },
        "rect" => Generator::shape(Shape::Rectangle, [255, 255, 255, 255]),
        "ellipse" => Generator::shape(Shape::Ellipse, [255, 255, 255, 255]),
        _ => return None,
    })
}

/// A bundled sticker's first frame as a Slint image — the Library tile
/// thumbnail. Blank when decode fails (the tile shows its label only).
pub(crate) fn sticker_thumbnail(spec: &cutlass_models::StickerSpec) -> slint::Image {
    let Ok(frames) = cutlass_decoder::decode_animation(spec.bytes) else {
        return slint::Image::default();
    };
    let first = &frames[0].image;
    let buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(
        &first.pixels,
        first.width,
        first.height,
    );
    slint::Image::from_rgba8(buffer)
}

/// Map an inspector param key to the engine's `ClipParam` plus the matching
/// `ParamValue` shape (position is the one vec2; scalars ride `value_x`).
/// `None` for an unknown key.
pub(crate) fn clip_param_value(
    param: &str,
    value_x: f32,
    value_y: f32,
) -> Option<(cutlass_models::ClipParam, cutlass_models::ParamValue)> {
    use cutlass_models::{ClipParam, LookParam, ParamValue, StyleParam, TextParam};
    // Color commands use two exact u16 lanes: RG and BA. Scalar rows always
    // leave `value_y` at zero, so this encoding cannot be confused with one.
    let color = || {
        let rg = value_x.round().clamp(0.0, u16::MAX as f32) as u16;
        let ba = value_y.round().clamp(0.0, u16::MAX as f32) as u16;
        ParamValue::Color([(rg >> 8) as u8, rg as u8, (ba >> 8) as u8, ba as u8])
    };
    Some(match param {
        "position" => (ClipParam::Position, ParamValue::Vec2([value_x, value_y])),
        "anchor" => (ClipParam::AnchorPoint, ParamValue::Vec2([value_x, value_y])),
        "scale" => (ClipParam::Scale, ParamValue::Scalar(value_x)),
        "rotation" => (ClipParam::Rotation, ParamValue::Scalar(value_x)),
        "opacity" => (ClipParam::Opacity, ParamValue::Scalar(value_x)),
        "volume" => (ClipParam::Volume, ParamValue::Scalar(value_x)),
        // Crop keyframes carry `[x,y,w,h]` via `ParamValue::Rect`. The Slint
        // two-float channel cannot express a full rect — this arm resolves the
        // param for remove/diamond tooling; commits that need a real rect use
        // `ParamValue::Rect` directly (crop tool / AI wire).
        "crop" => (
            ClipParam::Crop,
            ParamValue::Rect([value_x, value_y, 1.0, 1.0]),
        ),
        "text_size" => (
            ClipParam::Text {
                param: TextParam::Size,
            },
            ParamValue::Scalar(value_x),
        ),
        "text_fill" => (
            ClipParam::Text {
                param: TextParam::Fill,
            },
            color(),
        ),
        "text_letter_spacing" => (
            ClipParam::Text {
                param: TextParam::LetterSpacing,
            },
            ParamValue::Scalar(value_x),
        ),
        "text_line_spacing" => (
            ClipParam::Text {
                param: TextParam::LineSpacing,
            },
            ParamValue::Scalar(value_x),
        ),
        "text_stroke_width" => (
            ClipParam::Text {
                param: TextParam::StrokeWidth,
            },
            ParamValue::Scalar(value_x),
        ),
        "text_stroke_color" => (
            ClipParam::Text {
                param: TextParam::StrokeColor,
            },
            color(),
        ),
        "text_shadow_blur" => (
            ClipParam::Text {
                param: TextParam::ShadowBlur,
            },
            ParamValue::Scalar(value_x),
        ),
        "text_shadow_distance" => (
            ClipParam::Text {
                param: TextParam::ShadowDistance,
            },
            ParamValue::Scalar(value_x),
        ),
        "text_shadow_color" => (
            ClipParam::Text {
                param: TextParam::ShadowColor,
            },
            color(),
        ),
        "look_filter_intensity" => (
            ClipParam::Look {
                param: LookParam::FilterIntensity,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_lut_intensity" => (
            ClipParam::Look {
                param: LookParam::LutIntensity,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_adjust_brightness" => (
            ClipParam::Look {
                param: LookParam::AdjustBrightness,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_adjust_contrast" => (
            ClipParam::Look {
                param: LookParam::AdjustContrast,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_adjust_saturation" => (
            ClipParam::Look {
                param: LookParam::AdjustSaturation,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_adjust_exposure" => (
            ClipParam::Look {
                param: LookParam::AdjustExposure,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_adjust_temperature" => (
            ClipParam::Look {
                param: LookParam::AdjustTemperature,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_adjust_tint" => (
            ClipParam::Look {
                param: LookParam::AdjustTint,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_adjust_hue" => (
            ClipParam::Look {
                param: LookParam::AdjustHue,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_adjust_highlights" => (
            ClipParam::Look {
                param: LookParam::AdjustHighlights,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_adjust_shadows" => (
            ClipParam::Look {
                param: LookParam::AdjustShadows,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_adjust_sharpness" => (
            ClipParam::Look {
                param: LookParam::AdjustSharpness,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_adjust_vignette" => (
            ClipParam::Look {
                param: LookParam::AdjustVignette,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_mask_feather" => (
            ClipParam::Look {
                param: LookParam::MaskFeather,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_mask_center" => (
            ClipParam::Look {
                param: LookParam::MaskCenter,
            },
            ParamValue::Vec2([value_x, value_y]),
        ),
        "look_mask_size" => (
            ClipParam::Look {
                param: LookParam::MaskSize,
            },
            ParamValue::Vec2([value_x, value_y]),
        ),
        "look_mask_rotation" => (
            ClipParam::Look {
                param: LookParam::MaskRotation,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_mask_roundness" => (
            ClipParam::Look {
                param: LookParam::MaskRoundness,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_chroma_strength" => (
            ClipParam::Look {
                param: LookParam::ChromaStrength,
            },
            ParamValue::Scalar(value_x),
        ),
        "look_chroma_shadow" => (
            ClipParam::Look {
                param: LookParam::ChromaShadow,
            },
            ParamValue::Scalar(value_x),
        ),
        "style_shadow_color" => (
            ClipParam::Style {
                param: StyleParam::ShadowColor,
            },
            color(),
        ),
        "style_shadow_offset" => (
            ClipParam::Style {
                param: StyleParam::ShadowOffset,
            },
            ParamValue::Vec2([value_x, value_y]),
        ),
        "style_shadow_blur" => (
            ClipParam::Style {
                param: StyleParam::ShadowBlur,
            },
            ParamValue::Scalar(value_x),
        ),
        "style_glow_color" => (
            ClipParam::Style {
                param: StyleParam::GlowColor,
            },
            color(),
        ),
        "style_glow_radius" => (
            ClipParam::Style {
                param: StyleParam::GlowRadius,
            },
            ParamValue::Scalar(value_x),
        ),
        "style_glow_intensity" => (
            ClipParam::Style {
                param: StyleParam::GlowIntensity,
            },
            ParamValue::Scalar(value_x),
        ),
        "style_outline_color" => (
            ClipParam::Style {
                param: StyleParam::OutlineColor,
            },
            color(),
        ),
        "style_outline_width" => (
            ClipParam::Style {
                param: StyleParam::OutlineWidth,
            },
            ParamValue::Scalar(value_x),
        ),
        "style_background_color" => (
            ClipParam::Style {
                param: StyleParam::BackgroundColor,
            },
            color(),
        ),
        "style_background_padding" => (
            ClipParam::Style {
                param: StyleParam::BackgroundPadding,
            },
            ParamValue::Scalar(value_x),
        ),
        "style_background_radius" => (
            ClipParam::Style {
                param: StyleParam::BackgroundRadius,
            },
            ParamValue::Scalar(value_x),
        ),
        _ => return None,
    })
}

/// Apply one layer-style field as a [`Param::Constant`] onto a cloned
/// [`LayerStyles`] for live preview. Ensures the owning block exists (model
/// default) then overwrites the named field. Supports axis keys
/// `style_shadow_offset_x` / `_y` (keeps the other axis from the offset
/// sampled at the clip-relative `tick`, so a keyframed offset previews the
/// playhead's composite value). Returns `false` for unknown keys.
pub(crate) fn apply_style_preview_constant(
    styles: &mut cutlass_models::LayerStyles,
    key: &str,
    value_x: f32,
    value_y: f32,
    tick: i64,
) -> bool {
    use cutlass_models::{
        ClipParam, LayerBackground, LayerGlow, LayerOutline, LayerShadow, Param, ParamValue,
        StyleParam,
    };

    let (key, value_x, value_y) = match key {
        "style_shadow_offset_x" => {
            let y = styles
                .shadow
                .as_ref()
                .map(|s| s.offset.sample(tick)[1])
                .unwrap_or(4.0);
            ("style_shadow_offset", value_x, y)
        }
        "style_shadow_offset_y" => {
            let x = styles
                .shadow
                .as_ref()
                .map(|s| s.offset.sample(tick)[0])
                .unwrap_or(4.0);
            ("style_shadow_offset", x, value_x)
        }
        other => (other, value_x, value_y),
    };

    let Some((ClipParam::Style { param }, value)) = clip_param_value(key, value_x, value_y) else {
        return false;
    };

    match param {
        StyleParam::ShadowColor | StyleParam::ShadowOffset | StyleParam::ShadowBlur => {
            styles.shadow.get_or_insert_with(LayerShadow::default);
        }
        StyleParam::GlowColor | StyleParam::GlowRadius | StyleParam::GlowIntensity => {
            styles.glow.get_or_insert_with(LayerGlow::default);
        }
        StyleParam::OutlineColor | StyleParam::OutlineWidth => {
            styles.outline.get_or_insert_with(LayerOutline::default);
        }
        StyleParam::BackgroundColor
        | StyleParam::BackgroundPadding
        | StyleParam::BackgroundRadius => {
            styles
                .background
                .get_or_insert_with(LayerBackground::default);
        }
    }

    match (param, value) {
        (StyleParam::ShadowBlur, ParamValue::Scalar(v)) => {
            styles.shadow.as_mut().unwrap().blur = Param::Constant(v);
        }
        (StyleParam::ShadowOffset, ParamValue::Vec2(v)) => {
            styles.shadow.as_mut().unwrap().offset = Param::Constant(v);
        }
        (StyleParam::ShadowColor, ParamValue::Color(v)) => {
            styles.shadow.as_mut().unwrap().rgba = Param::Constant(v);
        }
        (StyleParam::GlowRadius, ParamValue::Scalar(v)) => {
            styles.glow.as_mut().unwrap().radius = Param::Constant(v);
        }
        (StyleParam::GlowIntensity, ParamValue::Scalar(v)) => {
            styles.glow.as_mut().unwrap().intensity = Param::Constant(v);
        }
        (StyleParam::GlowColor, ParamValue::Color(v)) => {
            styles.glow.as_mut().unwrap().rgba = Param::Constant(v);
        }
        (StyleParam::OutlineWidth, ParamValue::Scalar(v)) => {
            styles.outline.as_mut().unwrap().width = Param::Constant(v);
        }
        (StyleParam::OutlineColor, ParamValue::Color(v)) => {
            styles.outline.as_mut().unwrap().rgba = Param::Constant(v);
        }
        (StyleParam::BackgroundPadding, ParamValue::Scalar(v)) => {
            styles.background.as_mut().unwrap().padding = Param::Constant(v);
        }
        (StyleParam::BackgroundRadius, ParamValue::Scalar(v)) => {
            styles.background.as_mut().unwrap().radius = Param::Constant(v);
        }
        (StyleParam::BackgroundColor, ParamValue::Color(v)) => {
            styles.background.as_mut().unwrap().rgba = Param::Constant(v);
        }
        _ => return false,
    }
    true
}

/// Run `f` on the next event-loop turn, outside whatever callback is
/// currently executing. Used to flip Timer-bound state (see `request-stop`)
/// without re-entering Slint's timer machinery. Must never run anything that
/// blocks on a nested run loop (e.g. a modal `rfd::FileDialog`): the closure
/// executes inside Slint's timer activation, and the display link re-entering
/// it aborts with "Recursion in timer code".
pub(crate) fn defer_main_thread(f: impl FnOnce() + Send + 'static) {
    slint::Timer::single_shot(std::time::Duration::ZERO, f);
}

// File dialogs use `rfd::AsyncFileDialog`: on macOS it presents a sheet via
// `beginSheetModalForWindow:completionHandler:` and never blocks the main
// thread. The blocking `rfd::FileDialog` spins a nested `runModal` run loop,
// during which Slint's display-link tick re-enters timer processing and
// aborts with "Recursion in timer code".

// Extensions accepted by the import dialog and OS file-drop handler.
pub(crate) const MEDIA_IMPORT_EXTENSIONS: &[&str] = &[
    "mp4", "mov", "mkv", "webm", "m4v", "mp3", "wav", "m4a", "aac", "flac", "ogg", "png", "jpg",
    "jpeg", "webp",
];
pub(crate) const VIDEO_IMPORT_EXTENSIONS: &[&str] = &["mp4", "mov", "mkv", "webm", "m4v"];
pub(crate) const AUDIO_IMPORT_EXTENSIONS: &[&str] = &["mp3", "wav", "m4a", "aac", "flac", "ogg"];
pub(crate) const IMAGE_IMPORT_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];

pub(crate) fn media_extension_supported(path: &std::path::Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    let ext = ext.to_ascii_lowercase();
    MEDIA_IMPORT_EXTENSIONS
        .iter()
        .any(|&supported| supported == ext)
}

/// Extension-only audio check for files still owned by the OS drag (the
/// probe runs after the drop): drives the hover preview's lane kind.
pub(crate) fn audio_extension(path: &std::path::Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    let ext = ext.to_ascii_lowercase();
    AUDIO_IMPORT_EXTENSIONS
        .iter()
        .any(|&supported| supported == ext)
}

/// Where an OS file drop at window-space `cursor` (logical points) lands on
/// the timeline: `Some((lane row, sequence tick))` when the cursor is over
/// the timeline panel, `None` anywhere else — the caller falls back to the
/// pool-only import. Geometry comes from the crate::AppState mirror TimelinePanel
/// maintains (`sync-os-drop-geometry`); the launch-screen and maximized-
/// preview guards cover the states where that mirror is stale because the
/// panel is unmounted.
pub(crate) fn os_drop_timeline_target(
    app: &crate::AppWindow,
    cursor: (f32, f32),
) -> Option<(i64, i64)> {
    let state = app.global::<crate::AppState>();
    if state.get_launch_visible() || state.get_preview_maximized() {
        return None;
    }
    let (cx, cy) = cursor;
    let (px, py) = (state.get_timeline_panel_x(), state.get_timeline_panel_y());
    let (pw, ph) = (state.get_timeline_panel_w(), state.get_timeline_panel_h());
    if pw <= 0.0 || ph <= 0.0 || cx < px || cx >= px + pw || cy < py || cy >= py + ph {
        return None;
    }
    let pitch = state.get_timeline_row_pitch();
    let zoom = app.global::<crate::TimelineStore>().get_zoom();
    if pitch <= 0.0 || zoom <= 0.0 {
        return None;
    }
    let view = app.global::<crate::TimelineViewState>();
    // Same math as the library drag targeting in timeline.slint: cursor →
    // lane-content space → tick / row (row 0 is the head spacer, lanes
    // start at row 1).
    let tick = ((cx - state.get_timeline_lanes_x() - view.get_scroll_x()) / zoom)
        .round()
        .max(0.0) as i64;
    let row =
        ((cy - state.get_timeline_lanes_y() - view.get_scroll_y()) / pitch).floor() as i64 - 1;
    Some((row, tick))
}

pub(crate) async fn pick_import_paths() -> Vec<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter("Media", MEDIA_IMPORT_EXTENSIONS)
        .add_filter("Video", VIDEO_IMPORT_EXTENSIONS)
        .add_filter("Audio", AUDIO_IMPORT_EXTENSIONS)
        .add_filter("Images", IMAGE_IMPORT_EXTENSIONS)
        .pick_files()
        .await
        .map(|files| files.into_iter().map(|f| f.path().to_path_buf()).collect())
        .unwrap_or_default()
}

/// File picker for Open file… — choose an external `.cutlass` to import into
/// a new draft (the app-owned store; see [`drafts`]).
pub(crate) async fn pick_open_path() -> Option<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter("Cutlass project", &["cutlass"])
        .pick_file()
        .await
        .map(|file| file.path().to_path_buf())
}

pub(crate) async fn pick_relink_path() -> Option<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter(
            "Media",
            &[
                "mp4", "mov", "mkv", "webm", "m4v", "mp3", "wav", "m4a", "aac", "flac", "ogg",
                "png", "jpg", "jpeg", "webp",
            ],
        )
        .add_filter("Video", &["mp4", "mov", "mkv", "webm", "m4v"])
        .add_filter("Audio", &["mp3", "wav", "m4a", "aac", "flac", "ogg"])
        .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
        .pick_file()
        .await
        .map(|file| file.path().to_path_buf())
}

/// File picker for a user-supplied `.cube` 3D LUT (look inspector Browse…).
pub(crate) async fn pick_lut_path() -> Option<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter("LUT", &["cube"])
        .pick_file()
        .await
        .map(|file| file.path().to_path_buf())
}

pub(crate) async fn pick_relink_folder() -> Option<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .pick_folder()
        .await
        .map(|file| file.path().to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::{ClipParam, ParamValue, StyleParam};

    #[test]
    fn style_param_keys_round_trip_through_clip_param_value() {
        let (param, value) = clip_param_value("style_shadow_blur", 12.5, 0.0).expect("blur");
        assert_eq!(
            param,
            ClipParam::Style {
                param: StyleParam::ShadowBlur
            }
        );
        assert_eq!(value, ParamValue::Scalar(12.5));

        let (param, value) = clip_param_value("style_shadow_offset", 4.0, -2.0).expect("offset");
        assert_eq!(
            param,
            ClipParam::Style {
                param: StyleParam::ShadowOffset
            }
        );
        assert_eq!(value, ParamValue::Vec2([4.0, -2.0]));

        // Color packing: RG and BA as u16 lanes (same as text color keys).
        let rg = ((10u16) << 8) | 20;
        let ba = ((30u16) << 8) | 40;
        let (param, value) =
            clip_param_value("style_glow_color", rg as f32, ba as f32).expect("color");
        assert_eq!(
            param,
            ClipParam::Style {
                param: StyleParam::GlowColor
            }
        );
        assert_eq!(value, ParamValue::Color([10, 20, 30, 40]));
    }
}
