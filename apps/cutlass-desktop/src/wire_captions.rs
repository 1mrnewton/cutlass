//! `CaptionBackend` wiring: caption lookups for the cue list, and the caption
//! commands the caption inspector and library entry points fire.

use slint::{ComponentHandle, Global, ModelRc, VecModel};

use crate::auto_captions::AutoCaptionService;
use crate::cache_registry::CacheRegistry;
use crate::library_helpers::defer_main_thread;
use crate::preview_worker::{CaptionOp, PreviewWorker};
use crate::{AppWindow, CaptionBackend, CatalogEntry, captions, inspector};

/// Subtitle files the caption importer reads.
const SUBTITLE_EXTENSIONS: &[&str] = &["srt", "vtt", "webvtt"];

pub(crate) fn wire_captions(
    app: &AppWindow,
    preview_worker: &PreviewWorker,
    jobs: &cutlass_jobs::JobManager,
    caches: &CacheRegistry,
) {
    let backend = app.global::<CaptionBackend>();

    // Caption template catalog, once at startup — the model owns the list, so
    // the picker can't drift from what `SetCaptionGroupTemplate` accepts.
    backend.set_templates(ModelRc::from(std::rc::Rc::new(VecModel::from(
        cutlass_models::caption_template_catalog()
            .iter()
            .map(|spec| CatalogEntry {
                id: spec.id.into(),
                label: spec.label.into(),
                has_speed: false,
                has_intensity: false,
                has_stagger: false,
            })
            .collect::<Vec<_>>(),
    ))));

    backend.on_group(|sequence, group_id| captions::group(sequence, group_id.as_str()));
    backend.on_cues(|sequence, group_id| captions::cues(sequence, group_id.as_str()));

    let handle = preview_worker.handle();
    backend.on_set_cue_text(move |clip_id, text| {
        handle.caption(CaptionOp::SetCueText {
            clip: clip_id.to_string(),
            text: text.to_string(),
        });
    });

    let handle = preview_worker.handle();
    backend.on_apply_style_to_all(move |group_id, style, keep_overrides| {
        handle.caption(CaptionOp::ApplyStyle {
            group: group_id.to_string(),
            style: Box::new(inspector::text_style_from_ui(&style)),
            keep_overrides,
        });
    });

    let handle = preview_worker.handle();
    backend.on_set_template(move |group_id, template| {
        handle.caption(CaptionOp::SetTemplate {
            group: group_id.to_string(),
            template: template.to_string(),
        });
    });

    let handle = preview_worker.handle();
    backend.on_set_label(move |group_id, label| {
        handle.caption(CaptionOp::SetLabel {
            group: group_id.to_string(),
            label: label.to_string(),
        });
    });

    let handle = preview_worker.handle();
    backend.on_set_layout(
        move |group_id, max_chars_per_line, max_lines, safe_area_bottom, reflow| {
            handle.caption(CaptionOp::SetLayout {
                group: group_id.to_string(),
                // The engine validates the real bounds; these casts only keep
                // a wild UI value from wrapping around.
                max_chars_per_line: max_chars_per_line.clamp(0, i32::from(u16::MAX)) as u16,
                max_lines: max_lines.clamp(0, i32::from(u8::MAX)) as u8,
                safe_area_bottom,
                reflow,
            });
        },
    );

    let handle = preview_worker.handle();
    backend.on_set_highlight(
        move |group_id,
              mode,
              fill_r,
              fill_g,
              fill_b,
              plate_enabled,
              plate_r,
              plate_g,
              plate_b,
              plate_a,
              scale| {
            let Some(mode) = highlight_mode(mode) else {
                tracing::error!(mode, "ignoring caption highlight with unknown mode");
                return;
            };
            handle.caption(CaptionOp::SetHighlight {
                group: group_id.to_string(),
                mode,
                fill: [channel(fill_r), channel(fill_g), channel(fill_b), 255],
                plate: plate_enabled.then(|| {
                    [
                        channel(plate_r),
                        channel(plate_g),
                        channel(plate_b),
                        channel(plate_a),
                    ]
                }),
                scale,
            });
        },
    );

    let handle = preview_worker.handle();
    backend.on_split_cue(move |clip_id, tick| {
        handle.caption(CaptionOp::SplitCue {
            clip: clip_id.to_string(),
            tick: i64::from(tick),
        });
    });

    let handle = preview_worker.handle();
    backend.on_merge_with_next(move |clip_id| {
        handle.caption(CaptionOp::MergeWithNext {
            clip: clip_id.to_string(),
        });
    });

    let handle = preview_worker.handle();
    backend.on_remove_group(move |group_id| {
        handle.caption(CaptionOp::RemoveGroup {
            group: group_id.to_string(),
        });
    });

    let handle = preview_worker.handle();
    backend.on_ungroup(move |group_id| {
        handle.caption(CaptionOp::Ungroup {
            group: group_id.to_string(),
        });
    });

    let handle = preview_worker.handle();
    backend.on_add_caption(move |track_id, tick, text, template| {
        handle.caption(CaptionOp::AddGroup {
            track: track_id.to_string(),
            tick: i64::from(tick),
            text: text.to_string(),
            template: template.to_string(),
        });
    });

    let handle = preview_worker.handle();
    backend.on_import_subtitles(move |tick, template| {
        let handle = handle.clone();
        let template = template.to_string();
        let tick = i64::from(tick);
        // Defer past popup teardown so the macOS sheet can present (same
        // reasoning as the LUT picker).
        defer_main_thread(move || {
            let task = slint::spawn_local(async move {
                if let Some(path) = pick_subtitle_path().await {
                    handle.caption(CaptionOp::ImportFile {
                        path,
                        tick,
                        template,
                    });
                }
            });
            if let Err(e) = task {
                tracing::error!("failed to open subtitle file dialog: {e}");
            }
        });
    });

    wire_auto_captions(&backend, preview_worker, jobs, caches);
}

/// Auto captions: the dialog's source lookup, and the transcription job it
/// starts. The job owns no UI or engine state — it publishes progress into the
/// properties above and sends its cues through the worker like any other edit.
fn wire_auto_captions(
    backend: &CaptionBackend<'_>,
    preview_worker: &PreviewWorker,
    jobs: &cutlass_jobs::JobManager,
    caches: &CacheRegistry,
) {
    backend.on_auto_source(|project, selected_clip, playhead_tick| {
        captions::auto_source(project, selected_clip.as_str(), playhead_tick)
    });

    let service = AutoCaptionService::new(
        jobs.clone(),
        caches.clone(),
        preview_worker.handle(),
        backend.as_weak(),
    );
    let starter = service.clone();
    backend.on_start_auto(move |project, clip_id, max_chars_per_line, template| {
        match captions::auto_request(
            &project,
            clip_id.as_str(),
            max_chars_per_line,
            template.as_str(),
        ) {
            Some(request) => starter.start(request),
            None => tracing::error!(
                clip = clip_id.as_str(),
                "auto captions ignored: the clip or its media is gone"
            ),
        }
    });
    backend.on_cancel_auto(move || service.cancel());
}

async fn pick_subtitle_path() -> Option<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter("Subtitles", SUBTITLE_EXTENSIONS)
        .pick_file()
        .await
        .map(|file| file.path().to_path_buf())
}

/// `CaptionGroupView.highlight-mode` back to the model enum.
fn highlight_mode(mode: i32) -> Option<cutlass_models::CaptionHighlightMode> {
    match mode {
        0 => Some(cutlass_models::CaptionHighlightMode::Off),
        1 => Some(cutlass_models::CaptionHighlightMode::Word),
        2 => Some(cutlass_models::CaptionHighlightMode::Line),
        _ => None,
    }
}

fn channel(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}
