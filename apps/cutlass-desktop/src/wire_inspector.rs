//! Callback wiring extracted from `main` — structural split only.
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
use slint::winit_030::EventResult;
use slint::winit_030::WinitWindowAccessor;
use slint::winit_030::winit::event::WindowEvent;

use crate::bootstrap::*;
use crate::cache_ui::*;
use crate::library_helpers::*;
use crate::session::*;
use crate::*;

pub(crate) fn wire_inspector(
    app: &AppWindow,
    preview_worker: &crate::preview_worker::PreviewWorker,
) {
    // --- inspector: selection resolve, playhead sampling, commits ---------

    app.global::<InspectorBackend>()
        .on_resolve_selection(|sequence, track_id, clip_id| {
            inspector::resolve_selection(sequence, track_id.as_str(), clip_id.as_str())
        });

    app.global::<InspectorBackend>()
        .on_sample_transform(|clip, playhead| inspector::sample_transform(&clip, playhead));
    app.global::<InspectorBackend>()
        .on_compensate_anchor_position(
            |clip, sequence, playhead, anchor_x, anchor_y, scale_x, scale_y, rotation| {
                inspector::compensate_anchor_position(
                    &clip, sequence, playhead, anchor_x, anchor_y, scale_x, scale_y, rotation,
                )
            },
        );

    app.global::<InspectorBackend>()
        .on_sample_audio(|clip, playhead| inspector::sample_audio(&clip, playhead));
    app.global::<InspectorBackend>()
        .on_sample_scalar_param(|clip, param, playhead| {
            inspector::sample_scalar_param(&clip, param.as_str(), playhead)
        });

    let kf_set_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_set_param_keyframe(
        move |clip_id, param, tick, value_x, value_y, easing| {
            let Some((param, value)) = clip_param_value(param.as_str(), value_x, value_y) else {
                tracing::error!(param = param.as_str(), "ignoring keyframe on unknown param");
                return;
            };
            kf_set_handle.set_param_keyframe(
                clip_id.to_string(),
                param,
                i64::from(tick),
                value,
                params::easing_from_ui(easing, [0.0; 4]),
            );
        },
    );

    let kf_remove_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_remove_param_keyframe(move |clip_id, param, tick| {
            let Some((param, _)) = clip_param_value(param.as_str(), 0.0, 0.0) else {
                tracing::error!(
                    param = param.as_str(),
                    "ignoring keyframe removal on unknown param"
                );
                return;
            };
            kf_remove_handle.remove_param_keyframe(clip_id.to_string(), param, i64::from(tick));
        });

    let param_constant_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_set_param_constant(
        move |clip_id, param, value_x, value_y| {
            let Some((param, value)) = clip_param_value(param.as_str(), value_x, value_y) else {
                tracing::error!(param = param.as_str(), "ignoring constant on unknown param");
                return;
            };
            param_constant_handle.set_param_constant(clip_id.to_string(), param, value);
        },
    );

    // Timeline keyframe diamonds: merged tick model for the selected clip,
    // drag-retime, right-click delete.
    app.global::<KeyframeBackend>()
        .on_ticks(|clip| params::merged_keyframe_ticks(&clip));
    let kf_retime_handle = preview_worker.handle();
    app.global::<KeyframeBackend>()
        .on_retime(move |clip_id, from_tick, to_tick| {
            kf_retime_handle.retime_keyframes(
                clip_id.to_string(),
                i64::from(from_tick),
                i64::from(to_tick),
            );
        });
    let kf_remove_at_handle = preview_worker.handle();
    app.global::<KeyframeBackend>()
        .on_remove_at(move |clip_id, tick| {
            kf_remove_at_handle.remove_keyframes_at(clip_id.to_string(), i64::from(tick));
        });
    let set_speed_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_speed(move |clip_id, num, den, reversed| {
            set_speed_handle.set_clip_speed(clip_id.to_string(), num, den, reversed);
        });
    let set_pitch_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_pitch(move |clip_id, preserve| {
            set_pitch_handle.set_clip_pitch(clip_id.to_string(), preserve);
        });
    let set_denoise_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_denoise(move |clip_id, denoise| {
            set_denoise_handle.set_denoise(clip_id.to_string(), denoise);
        });
    let set_curve_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_speed_curve(move |clip_id, preset| {
            set_curve_handle.set_speed_curve(clip_id.to_string(), preset.to_string());
        });
    let set_curve_point_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_speed_curve_point(move |clip_id, index, value| {
            set_curve_point_handle.set_speed_curve_point(clip_id.to_string(), index, value);
        });
    let set_audio_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_set_clip_audio(
        move |clip_id, volume, fade_in_s, fade_out_s| {
            set_audio_handle.set_clip_audio(clip_id.to_string(), volume, fade_in_s, fade_out_s);
        },
    );
    let set_fades_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_fades(move |clip_id, fade_in_s, fade_out_s| {
            set_fades_handle.set_clip_fades(clip_id.to_string(), fade_in_s, fade_out_s);
        });
    app.global::<InspectorBackend>()
        .on_can_duck_under_voice(|sequence, track_id| {
            inspector::can_duck_under_voice(sequence, track_id.as_str())
        });
    let duck_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_duck_under_voice(move |clip_id| {
            duck_handle.duck_under_voice(clip_id.to_string());
        });
    let detect_beats_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_detect_beats(move |clip_id| {
            detect_beats_handle.detect_beats(clip_id.to_string());
        });
    let clear_beats_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_clear_beats(move |clip_id| {
            clear_beats_handle.clear_beats(clip_id.to_string());
        });
    let set_crop_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_set_clip_crop(
        move |clip_id, left, top, right, bottom, flip_h, flip_v, playhead_tick| {
            // Insets (UI/agent shape) → kept-region rect (model shape). The
            // sliders cap each inset at 49%, so the window stays valid; the
            // floor only guards float dust against the engine's minimum.
            let crop = cutlass_models::CropRect {
                x: left,
                y: top,
                w: (1.0 - left - right).max(cutlass_models::MIN_CROP_FRACTION),
                h: (1.0 - top - bottom).max(cutlass_models::MIN_CROP_FRACTION),
            };
            set_crop_handle.set_clip_crop(
                clip_id.to_string(),
                crop,
                flip_h,
                flip_v,
                i64::from(playhead_tick),
            );
        },
    );

    let set_blend_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_blend_mode(move |clip_id, mode| {
            set_blend_handle.set_blend_mode(clip_id.to_string(), mode.to_string());
        });

    let set_mask_kind_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_mask_kind(move |clip_id, kind| {
            let Some(project) = set_mask_kind_handle.snapshot_project() else {
                return;
            };
            let Some(raw) = clip_id.as_str().parse::<u64>().ok() else {
                tracing::error!(clip = %clip_id, "set-clip-mask-kind ignored: unparsable clip id");
                return;
            };
            let clip_key = cutlass_models::ClipId::from_raw(raw);
            let Some(clip) = project.clip(clip_key) else {
                tracing::error!(clip = %clip_id, "set-clip-mask-kind ignored: unknown clip");
                return;
            };
            let mask = if kind.is_empty() {
                None
            } else {
                let Some(spec) = cutlass_models::mask_catalog()
                    .iter()
                    .find(|s| s.kind.id() == kind.as_str())
                else {
                    tracing::error!(
                        kind = kind.as_str(),
                        "set-clip-mask-kind ignored: unknown kind"
                    );
                    return;
                };
                // Preserve feather / invert / geometry when switching kind;
                // enable-from-none uses Mask::new defaults.
                let mut mask = clip
                    .mask
                    .clone()
                    .unwrap_or_else(|| cutlass_models::Mask::new(spec.kind));
                mask.kind = spec.kind;
                Some(mask)
            };
            set_mask_kind_handle.set_mask(clip_id.to_string(), mask);
        });

    let set_mask_invert_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_mask_invert(move |clip_id, invert| {
            let Some(project) = set_mask_invert_handle.snapshot_project() else {
                return;
            };
            let Some(raw) = clip_id.as_str().parse::<u64>().ok() else {
                tracing::error!(clip = %clip_id, "set-clip-mask-invert ignored: unparsable clip id");
                return;
            };
            let clip_key = cutlass_models::ClipId::from_raw(raw);
            let Some(clip) = project.clip(clip_key) else {
                tracing::error!(clip = %clip_id, "set-clip-mask-invert ignored: unknown clip");
                return;
            };
            let Some(mut mask) = clip.mask.clone() else {
                tracing::error!(clip = %clip_id, "set-clip-mask-invert ignored: clip has no mask");
                return;
            };
            mask.invert = invert;
            set_mask_invert_handle.set_mask(clip_id.to_string(), Some(mask));
        });

    let toggle_chroma_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_toggle_clip_chroma(move |clip_id, enabled| {
            let chroma = enabled.then_some(cutlass_models::ChromaKey {
                rgb: [0, 255, 0],
                strength: cutlass_models::Param::Constant(0.5),
                shadow: cutlass_models::Param::Constant(0.0),
            });
            toggle_chroma_handle.set_chroma(clip_id.to_string(), chroma);
        });

    let set_chroma_color_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_chroma_color(move |clip_id, r, g, b| {
            let Some(project) = set_chroma_color_handle.snapshot_project() else {
                return;
            };
            let Some(raw) = clip_id.as_str().parse::<u64>().ok() else {
                tracing::error!(clip = %clip_id, "set-clip-chroma-color ignored: unparsable clip id");
                return;
            };
            let clip_key = cutlass_models::ClipId::from_raw(raw);
            let Some(clip) = project.clip(clip_key) else {
                tracing::error!(clip = %clip_id, "set-clip-chroma-color ignored: unknown clip");
                return;
            };
            let Some(mut chroma) = clip.chroma_key.clone() else {
                tracing::error!(clip = %clip_id, "set-clip-chroma-color ignored: chroma off");
                return;
            };
            chroma.rgb = [
                r.clamp(0, 255) as u8,
                g.clamp(0, 255) as u8,
                b.clamp(0, 255) as u8,
            ];
            set_chroma_color_handle.set_chroma(clip_id.to_string(), Some(chroma));
        });

    let toggle_style_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_toggle_clip_style(move |clip_id, block, enabled| {
            let Some(project) = toggle_style_handle.snapshot_project() else {
                return;
            };
            let Some(raw) = clip_id.as_str().parse::<u64>().ok() else {
                tracing::error!(clip = %clip_id, "toggle-clip-style ignored: unparsable clip id");
                return;
            };
            let clip_key = cutlass_models::ClipId::from_raw(raw);
            let Some(clip) = project.clip(clip_key) else {
                tracing::error!(clip = %clip_id, "toggle-clip-style ignored: unknown clip");
                return;
            };
            let mut styles = clip.styles.clone();
            match block.as_str() {
                "shadow" => {
                    styles.shadow = enabled.then(cutlass_models::LayerShadow::default);
                }
                "glow" => {
                    styles.glow = enabled.then(cutlass_models::LayerGlow::default);
                }
                "outline" => {
                    styles.outline = enabled.then(cutlass_models::LayerOutline::default);
                }
                "background" => {
                    styles.background = enabled.then(cutlass_models::LayerBackground::default);
                }
                other => {
                    tracing::error!(block = other, "toggle-clip-style ignored: unknown block");
                    return;
                }
            }
            toggle_style_handle.set_layer_styles(clip_id.to_string(), styles);
        });

    let style_color_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_style_color(move |clip_id, key, r, g, b, a| {
            let r = r.clamp(0, 255) as u16;
            let g = g.clamp(0, 255) as u16;
            let b = b.clamp(0, 255) as u16;
            let a = a.clamp(0, 255) as u16;
            let value_x = ((r << 8) | g) as f32;
            let value_y = ((b << 8) | a) as f32;
            let Some((param, value)) = clip_param_value(key.as_str(), value_x, value_y) else {
                tracing::error!(
                    key = key.as_str(),
                    "set-clip-style-color ignored: unknown key"
                );
                return;
            };
            style_color_handle.set_param_constant(clip_id.to_string(), param, value);
        });

    let preview_style_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_preview_clip_style(move |clip_id, key, value, tick| {
            let Some(project) = preview_style_handle.snapshot_project() else {
                return;
            };
            let Some(raw) = clip_id.as_str().parse::<u64>().ok() else {
                tracing::error!(clip = %clip_id, "preview-clip-style ignored: unparsable clip id");
                return;
            };
            let clip_key = cutlass_models::ClipId::from_raw(raw);
            let Some(clip) = project.clip(clip_key) else {
                tracing::error!(clip = %clip_id, "preview-clip-style ignored: unknown clip");
                return;
            };
            let mut styles = clip.styles.clone();
            let local_tick = clip.animation_tick(i64::from(tick));
            if !apply_style_preview_constant(&mut styles, key.as_str(), value, 0.0, local_tick) {
                tracing::error!(
                    key = key.as_str(),
                    "preview-clip-style ignored: unknown key"
                );
                return;
            }
            preview_style_handle.preview_clip_styles(clip_id.to_string(), styles, i64::from(tick));
        });

    let preview_style_color_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_preview_clip_style_color(move |clip_id, key, r, g, b, a, tick| {
            let Some(project) = preview_style_color_handle.snapshot_project() else {
                return;
            };
            let Some(raw) = clip_id.as_str().parse::<u64>().ok() else {
                tracing::error!(
                    clip = %clip_id,
                    "preview-clip-style-color ignored: unparsable clip id"
                );
                return;
            };
            let clip_key = cutlass_models::ClipId::from_raw(raw);
            let Some(clip) = project.clip(clip_key) else {
                tracing::error!(clip = %clip_id, "preview-clip-style-color ignored: unknown clip");
                return;
            };
            let r = r.clamp(0, 255) as u16;
            let g = g.clamp(0, 255) as u16;
            let b = b.clamp(0, 255) as u16;
            let a = a.clamp(0, 255) as u16;
            let value_x = ((r << 8) | g) as f32;
            let value_y = ((b << 8) | a) as f32;
            let mut styles = clip.styles.clone();
            let local_tick = clip.animation_tick(i64::from(tick));
            if !apply_style_preview_constant(
                &mut styles,
                key.as_str(),
                value_x,
                value_y,
                local_tick,
            ) {
                tracing::error!(
                    key = key.as_str(),
                    "preview-clip-style-color ignored: unknown key"
                );
                return;
            }
            preview_style_color_handle.preview_clip_styles(
                clip_id.to_string(),
                styles,
                i64::from(tick),
            );
        });

    let clear_styles_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_clear_clip_styles(move |tick| {
            clear_styles_handle.clear_styles_override(i64::from(tick));
        });

    let set_filter_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_filter(move |clip_id, filter_id, intensity| {
            set_filter_handle.set_clip_filter(
                clip_id.to_string(),
                filter_id.to_string(),
                intensity,
            );
        });

    let set_animation_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_set_clip_animation(
        move |clip_id, slot, animation_id, speed, intensity, stagger| {
            set_animation_handle.set_clip_animation(
                clip_id.to_string(),
                slot.to_string(),
                animation_id.to_string(),
                speed,
                intensity,
                stagger,
            );
        },
    );

    let set_adjust_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_adjust(move |clip_id, adjust| {
            set_adjust_handle
                .set_clip_adjust(clip_id.to_string(), inspector::adjust_from_ui(&adjust));
        });

    let preview_look_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_preview_clip_look(
        move |clip_id, filter_id, intensity, adjust, tick| {
            preview_look_handle.preview_clip_look(
                clip_id.to_string(),
                filter_id.to_string(),
                intensity,
                inspector::adjust_from_ui(&adjust),
                i64::from(tick),
            );
        },
    );

    let fit_clip_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_fit_clip(move |clip_id, fill, tick| {
            fit_clip_handle.fit_clip(clip_id.to_string(), fill, i64::from(tick));
        });

    let set_text_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_set_text_generator(
        move |_track_id, clip_id, content, style| {
            // Route the edit through the engine (undoable) rather than mutating
            // the Slint model, which the next projection republish would revert.
            // The inspector sends the full style each time, so one committed
            // edit == one coherent `Generator::Text`.
            set_text_handle.set_generator(
                clip_id.to_string(),
                cutlass_models::Generator::Text {
                    content: content.to_string(),
                    style: inspector::text_style_from_ui(&style),
                },
            );
        },
    );

    let preview_text_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_preview_text_generator(
        move |clip_id, content, style, tick| {
            // Live, uncommitted preview (e.g. font-size drag): render the clip
            // from this generator without touching history. Release commits.
            preview_text_handle.generator_override(
                clip_id.to_string(),
                cutlass_models::Generator::Text {
                    content: content.to_string(),
                    style: inspector::text_style_from_ui(&style),
                },
                i64::from(tick),
            );
        },
    );

    let clear_text_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_clear_text_generator(move |tick| {
            clear_text_handle.clear_generator_override(i64::from(tick));
        });

    let set_shape_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_shape_generator(move |clip_id, width, height| {
            set_shape_handle.set_shape_size(clip_id.to_string(), width, height);
        });

    let preview_shape_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_preview_shape_generator(
        move |clip_id, width, height, tick| {
            preview_shape_handle.preview_shape_size(
                clip_id.to_string(),
                width,
                height,
                i64::from(tick),
            );
        },
    );

    let clear_shape_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_clear_shape_generator(move |tick| {
            clear_shape_handle.clear_generator_override(i64::from(tick));
        });

    app.global::<InspectorBackend>()
        .on_filter_fonts(|query, items| {
            let needle = query.to_lowercase();
            let filtered: Vec<SharedString> = items
                .iter()
                .filter(|family| {
                    needle.is_empty() || family.as_str().to_lowercase().contains(&needle)
                })
                .collect();
            ModelRc::new(VecModel::from(filtered))
        });

    // Blend, filter, effect & transition catalogs are filled once from the
    // model catalogs, then inspector/timeline edits route to undoable commands.
    {
        let inspector = app.global::<InspectorBackend>();
        let blend_rows: Vec<CatalogEntry> = cutlass_models::BlendMode::ALL
            .iter()
            .map(|mode| CatalogEntry {
                id: mode.id().into(),
                label: mode.label().into(),
                has_speed: false,
                has_intensity: false,
                has_stagger: false,
            })
            .collect();
        inspector.set_blend_catalog(ModelRc::new(VecModel::from(blend_rows)));

        let mask_rows: Vec<CatalogEntry> = cutlass_models::mask_catalog()
            .iter()
            .map(|s| CatalogEntry {
                id: s.kind.id().into(),
                label: s.label.into(),
                has_speed: false,
                has_intensity: false,
                has_stagger: false,
            })
            .collect();
        inspector.set_mask_catalog(ModelRc::new(VecModel::from(mask_rows)));

        let filter_rows: Vec<CatalogEntry> = cutlass_models::filter_catalog()
            .iter()
            .map(|s| CatalogEntry {
                id: s.id.into(),
                label: s.label.into(),
                has_speed: false,
                has_intensity: false,
                has_stagger: false,
            })
            .collect();
        inspector.set_filter_catalog(ModelRc::new(VecModel::from(filter_rows)));

        let entry = |s: &cutlass_models::AnimationSpec| CatalogEntry {
            id: s.id.into(),
            label: s.label.into(),
            has_speed: s.knobs.speed,
            has_intensity: s.knobs.intensity,
            has_stagger: s.knobs.stagger,
        };
        // Media / general clips: whole-layer presets only.
        let animation_in: Vec<CatalogEntry> = cutlass_models::animation_catalog()
            .iter()
            .filter(|s| s.slot == cutlass_models::AnimationSlot::In && !s.text_only)
            .map(entry)
            .collect();
        inspector.set_animation_in_catalog(ModelRc::new(VecModel::from(animation_in)));

        let animation_out: Vec<CatalogEntry> = cutlass_models::animation_catalog()
            .iter()
            .filter(|s| s.slot == cutlass_models::AnimationSlot::Out && !s.text_only)
            .map(entry)
            .collect();
        inspector.set_animation_out_catalog(ModelRc::new(VecModel::from(animation_out)));

        let animation_combo: Vec<CatalogEntry> = cutlass_models::animation_catalog()
            .iter()
            .filter(|s| s.slot == cutlass_models::AnimationSlot::Combo && !s.text_only)
            .map(entry)
            .collect();
        inspector.set_animation_combo_catalog(ModelRc::new(VecModel::from(animation_combo)));

        // Text panel: whole-layer + per-character presets in every slot.
        let animation_text_in: Vec<CatalogEntry> = cutlass_models::animation_catalog()
            .iter()
            .filter(|s| s.slot == cutlass_models::AnimationSlot::In)
            .map(entry)
            .collect();
        inspector.set_animation_text_in_catalog(ModelRc::new(VecModel::from(animation_text_in)));

        let animation_text_out: Vec<CatalogEntry> = cutlass_models::animation_catalog()
            .iter()
            .filter(|s| s.slot == cutlass_models::AnimationSlot::Out)
            .map(entry)
            .collect();
        inspector.set_animation_text_out_catalog(ModelRc::new(VecModel::from(animation_text_out)));

        // Text combo tab: per-character loops plus whole-layer presence presets.
        let animation_text_combo: Vec<CatalogEntry> = cutlass_models::animation_catalog()
            .iter()
            .filter(|s| s.slot == cutlass_models::AnimationSlot::Combo)
            .map(entry)
            .collect();
        inspector
            .set_animation_text_combo_catalog(ModelRc::new(VecModel::from(animation_text_combo)));
    }
    {
        let effects = app.global::<EffectsBackend>();
        let effect_rows: Vec<CatalogEntry> = cutlass_models::effect_catalog()
            .iter()
            .map(|s| CatalogEntry {
                id: s.id.into(),
                label: s.label.into(),
                has_speed: false,
                has_intensity: false,
                has_stagger: false,
            })
            .collect();
        effects.set_effect_catalog(ModelRc::new(VecModel::from(effect_rows)));
        let transition_rows: Vec<CatalogEntry> = cutlass_models::transition_catalog()
            .iter()
            .map(|s| CatalogEntry {
                id: s.id.into(),
                label: s.label.into(),
                has_speed: false,
                has_intensity: false,
                has_stagger: false,
            })
            .collect();
        effects.set_transition_catalog(ModelRc::new(VecModel::from(transition_rows)));
        let sticker_rows: Vec<StickerTile> = cutlass_models::sticker_catalog()
            .iter()
            .map(|s| StickerTile {
                id: s.id.into(),
                label: s.label.into(),
                icon: sticker_thumbnail(s),
            })
            .collect();
        effects.set_sticker_catalog(ModelRc::new(VecModel::from(sticker_rows)));
    }
    let add_effect_handle = preview_worker.handle();
    app.global::<EffectsBackend>()
        .on_add_effect(move |clip_id, effect_id| {
            add_effect_handle.add_effect(clip_id.to_string(), effect_id.to_string());
        });
    let remove_effect_handle = preview_worker.handle();
    app.global::<EffectsBackend>()
        .on_remove_effect(move |clip_id, index| {
            remove_effect_handle.remove_effect(clip_id.to_string(), index.max(0) as u32);
        });
    let set_effect_param_handle = preview_worker.handle();
    app.global::<EffectsBackend>()
        .on_set_effect_param(move |clip_id, index, param, value| {
            set_effect_param_handle.set_effect_param(
                clip_id.to_string(),
                index.max(0) as u32,
                param.to_string(),
                value,
            );
        });
    let set_effect_param_color_handle = preview_worker.handle();
    app.global::<EffectsBackend>().on_set_effect_param_color(
        move |clip_id, index, param, r, g, b, a| {
            let rgba = [
                r.clamp(0, 255) as u8,
                g.clamp(0, 255) as u8,
                b.clamp(0, 255) as u8,
                a.clamp(0, 255) as u8,
            ];
            set_effect_param_color_handle.set_effect_param_value(
                clip_id.to_string(),
                index.max(0) as u32,
                param.to_string(),
                cutlass_models::ParamValue::Color(rgba),
            );
        },
    );
    let set_effect_param_vec2_handle = preview_worker.handle();
    app.global::<EffectsBackend>()
        .on_set_effect_param_vec2(move |clip_id, index, param, x, y| {
            set_effect_param_vec2_handle.set_effect_param_value(
                clip_id.to_string(),
                index.max(0) as u32,
                param.to_string(),
                cutlass_models::ParamValue::Vec2([x, y]),
            );
        });
    let add_transition_handle = preview_worker.handle();
    app.global::<EffectsBackend>()
        .on_add_transition(move |clip_id, transition_id| {
            add_transition_handle.add_transition(clip_id.to_string(), transition_id.to_string());
        });
    let remove_transition_handle = preview_worker.handle();
    app.global::<EffectsBackend>()
        .on_remove_transition(move |clip_id| {
            remove_transition_handle.remove_transition(clip_id.to_string());
        });
    let set_transition_handle = preview_worker.handle();
    app.global::<EffectsBackend>()
        .on_set_transition(move |clip_id, duration| {
            set_transition_handle.set_transition(clip_id.to_string(), i64::from(duration));
        });

    // Enumerate system fonts off the UI thread (the scan is slow) and feed
    // the font picker once ready.
    let font_app = app.as_weak();
    std::thread::spawn(move || {
        let families = cutlass_text::system_font_families();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(app) = font_app.upgrade() {
                let model: Vec<SharedString> = families.into_iter().map(Into::into).collect();
                app.global::<InspectorBackend>()
                    .set_font_families(ModelRc::new(VecModel::from(model)));
            }
        });
    });
}
