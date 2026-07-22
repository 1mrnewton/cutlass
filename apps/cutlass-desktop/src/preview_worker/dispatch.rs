use super::*;

/// Route one [`WorkerMsg`] to its handler and publish the resulting engine
/// state to the UI. Split out of the worker loop (which owns `fit`, `cache`,
/// `ui`, and the rest of this context) so the routing table — unavoidably
/// long, one arm per message — doesn't drown the loop's scheduling logic.
#[allow(clippy::too_many_arguments)] // one entry point for the whole message set
pub(super) fn dispatch(
    engine: &mut Engine,
    clipboard: &mut Option<Vec<ClipboardClip>>,
    main_magnet: &mut bool,
    linkage: &mut bool,
    msg: WorkerMsg,
    tl_rate: Rational,
    preview_weak: &slint::Weak<PreviewStore<'static>>,
    fit: &FrameFit,
    cache: &FrameCache,
    sprite_mode: &Cell<bool>,
    export_state: &ExportJobState,
    ui: &UiSink,
) {
    match msg {
        // Reached only when the viewport report drains behind a frame or
        // mutation; the follow-up render picks the new bound up.
        WorkerMsg::Viewport { width, height } => {
            fit.set_viewport(width, height);
        }
        WorkerMsg::Import(path) => import_and_publish(engine, &path, ui),
        WorkerMsg::ImportMediaRpc {
            path,
            reply,
            operation,
        } => serve_worker_rpc(reply, operation, || {
            import_media_rpc_and_publish(engine, &path, Some(ui))
        }),
        WorkerMsg::DropFiles { paths, target } => {
            drop_files_and_publish(engine, &paths, target, *main_magnet, ui);
        }
        WorkerMsg::ProxyReady {
            media_id,
            source,
            proxy,
        } => bind_media_proxy(engine, media_id, &source, proxy, cache, ui),
        WorkerMsg::AddClip {
            media,
            track,
            start_tick,
            drop_row,
            insert,
        } => add_clip_and_publish(engine, &media, &track, start_tick, drop_row, insert, ui),
        WorkerMsg::AddGenerated {
            generator,
            track,
            start_tick,
            duration_ticks,
            drop_row,
            effect,
            animations,
        } => add_generated_and_publish(
            engine,
            generator,
            &track,
            start_tick,
            duration_ticks,
            drop_row,
            effect.as_deref(),
            &animations,
            ui,
        ),
        WorkerMsg::MoveClip {
            clip,
            track,
            insert_row,
            start_tick,
            insert,
        } => move_clip_and_publish(
            engine,
            &clip,
            &track,
            insert_row,
            start_tick,
            insert,
            *main_magnet,
            ui,
        ),
        WorkerMsg::MoveGroup { moves } => move_group_and_publish(engine, &moves, ui),
        WorkerMsg::TrimClip {
            clip,
            start_tick,
            duration_ticks,
        } => trim_clip_and_publish(
            engine,
            &clip,
            start_tick,
            duration_ticks,
            *linkage,
            *main_magnet,
            ui,
        ),
        WorkerMsg::RemoveClips { clips } => {
            remove_clips_and_publish(engine, &clips, *main_magnet, ui)
        }
        WorkerMsg::RippleDeleteClips { clips } => {
            ripple_delete_clips_and_publish(engine, &clips, ui)
        }
        WorkerMsg::ReverseClip { clip } => reverse_clip_and_publish(engine, &clip, *linkage, ui),
        WorkerMsg::ExtractAudio { clip } => extract_audio_and_publish(engine, &clip, ui),
        WorkerMsg::SetGenerator { clip, generator } => {
            set_generator_and_publish(engine, &clip, generator, ui)
        }
        WorkerMsg::SetShapeSize {
            clip,
            width,
            height,
        } => {
            if let Some(generator) = shape_size_from_engine(engine, &clip, width, height) {
                set_generator_and_publish(engine, &clip, generator, ui);
            }
        }
        // Only reached when a shape-resize burst interleaves with another
        // coalesced gesture's drain (practically impossible — one slider at
        // a time). The dedicated loop arm coalesces the common case.
        WorkerMsg::PreviewShapeSize {
            clip,
            width,
            height,
            tick,
        } => {
            if let Some(generator) = shape_size_from_engine(engine, &clip, width, height) {
                apply_generator_override(engine, &clip, generator);
                render_frame(
                    engine,
                    tl_rate,
                    preview_weak,
                    tick,
                    fit,
                    cache,
                    SeekPolicy::Exact,
                );
            }
        }
        WorkerMsg::SetClipSpeed {
            clip,
            num,
            den,
            reversed,
        } => set_clip_speed_and_publish(engine, &clip, num, den, reversed, *linkage, ui),
        WorkerMsg::SetClipPitch { clip, preserve } => {
            set_clip_pitch_and_publish(engine, &clip, preserve, *linkage, ui)
        }
        WorkerMsg::SetDenoise { clip, denoise } => {
            set_denoise_and_publish(engine, &clip, denoise, *linkage, ui)
        }
        WorkerMsg::SetSpeedCurve { clip, curve } => {
            set_speed_curve_and_publish(engine, &clip, &curve, *linkage, ui)
        }
        WorkerMsg::SetSpeedCurvePoint { clip, index, value } => {
            set_speed_curve_point_and_publish(engine, &clip, index, value, *linkage, ui)
        }
        WorkerMsg::SetClipAudio {
            clip,
            volume,
            fade_in_s,
            fade_out_s,
        } => set_clip_audio_and_publish(engine, &clip, volume, fade_in_s, fade_out_s, ui),
        WorkerMsg::DuckUnderVoice { clip } => duck_under_voice_and_publish(engine, &clip, ui),
        WorkerMsg::DetectBeats { clip } => detect_beats_and_publish(engine, &clip, ui),
        WorkerMsg::ClearBeats { clip } => clear_beats_and_publish(engine, &clip, ui),
        WorkerMsg::SetClipCrop {
            clip,
            crop,
            flip_h,
            flip_v,
        } => set_clip_crop_and_publish(engine, &clip, crop, flip_h, flip_v, ui),
        WorkerMsg::SetBlendMode { clip, mode } => {
            set_blend_mode_and_publish(engine, &clip, &mode, ui)
        }
        WorkerMsg::SetLayerStyles { clip, styles } => {
            set_layer_styles_and_publish(engine, &clip, styles, ui)
        }
        WorkerMsg::SetClipFilter {
            clip,
            filter_id,
            intensity,
        } => set_clip_filter_and_publish(engine, &clip, &filter_id, intensity, ui),
        WorkerMsg::SetClipLut {
            clip,
            path,
            intensity,
        } => set_clip_lut_and_publish(engine, &clip, &path, intensity, ui),
        WorkerMsg::SetClipAdjust { clip, adjust } => {
            set_clip_adjust_and_publish(engine, &clip, adjust, ui)
        }
        WorkerMsg::SetAgentRules { rules } => {
            engine.set_agent_rules(rules);
            publish_projection(engine, ui);
        }
        WorkerMsg::SetClipAnimation {
            clip,
            slot,
            animation_id,
            speed,
            intensity,
            stagger,
        } => set_clip_animation_and_publish(
            engine,
            &clip,
            &slot,
            &animation_id,
            speed,
            intensity,
            stagger,
            ui,
        ),
        // Only reached when a look-preview burst interleaves with another
        // coalesced gesture's drain. The dedicated loop arm coalesces the
        // common case.
        WorkerMsg::PreviewClipLook {
            clip,
            filter_id,
            intensity,
            adjust,
            tick,
        } => {
            apply_look_override(engine, &clip, &filter_id, intensity, &adjust);
            render_frame(
                engine,
                tl_rate,
                preview_weak,
                tick,
                fit,
                cache,
                SeekPolicy::Exact,
            );
        }
        // Same interleave fallback for styles preview bursts.
        WorkerMsg::PreviewClipStyles { clip, styles, tick } => {
            apply_styles_override(engine, &clip, styles);
            render_frame(
                engine,
                tl_rate,
                preview_weak,
                tick,
                fit,
                cache,
                SeekPolicy::Exact,
            );
        }
        WorkerMsg::ClearStylesOverride { tick } => {
            engine.set_styles_override(None);
            render_frame(
                engine,
                tl_rate,
                preview_weak,
                tick,
                fit,
                cache,
                SeekPolicy::Exact,
            );
        }
        WorkerMsg::AddEffect { clip, effect_id } => {
            add_effect_and_publish(engine, &clip, &effect_id, ui)
        }
        WorkerMsg::RemoveEffect { clip, index } => {
            remove_effect_and_publish(engine, &clip, index, ui)
        }
        WorkerMsg::SetEffectParam {
            clip,
            index,
            param,
            value,
        } => set_effect_param_and_publish(engine, &clip, index, &param, value, ui),
        WorkerMsg::AddTransition {
            clip,
            transition_id,
        } => add_transition_and_publish(engine, &clip, &transition_id, ui),
        WorkerMsg::RemoveTransition { clip } => remove_transition_and_publish(engine, &clip, ui),
        WorkerMsg::SetTransition { clip, duration } => {
            set_transition_and_publish(engine, &clip, duration, ui)
        }
        WorkerMsg::SetCanvas {
            aspect_index,
            background,
        } => set_canvas_and_publish(engine, aspect_index, background, ui),
        WorkerMsg::ClearTransformOverride { tick } => {
            engine.set_transform_override(None);
            render_frame_exit_sprite(engine, tl_rate, preview_weak, tick, fit, sprite_mode);
        }
        WorkerMsg::ClearGeneratorOverride { tick } => {
            engine.set_generator_override(None);
            render_frame(
                engine,
                tl_rate,
                preview_weak,
                tick,
                fit,
                cache,
                SeekPolicy::Exact,
            );
        }
        // Only reached if a generator-override burst interleaves with
        // another coalesced gesture's drain (practically impossible — you
        // can't drag two controls at once). The dedicated loop arm handles
        // the common case with coalescing.
        WorkerMsg::GeneratorOverride {
            clip,
            generator,
            tick,
        } => {
            apply_generator_override(engine, &clip, generator);
            render_frame(
                engine,
                tl_rate,
                preview_weak,
                tick,
                fit,
                cache,
                SeekPolicy::Exact,
            );
        }
        WorkerMsg::SetTransform {
            clip,
            transform,
            tick,
        } => {
            // The override previewed this exact transform; clearing it as
            // the command lands means the next render is identical — no
            // flicker between gesture end and commit.
            engine.set_transform_override(None);
            // The gesture happened at the visible frame: pass the playhead
            // so animated properties get a keyframe there instead of being
            // flattened (M2 compose semantics).
            let at = RationalTime::new(tick, tl_rate);
            set_transform_and_publish(engine, &clip, transform, at, ui);
            render_frame_exit_sprite(engine, tl_rate, preview_weak, tick, fit, sprite_mode);
        }
        WorkerMsg::FitClip { clip, fill, tick } => {
            fit_clip_and_publish(engine, &clip, fill, tick, tl_rate, ui);
            render_frame(
                engine,
                tl_rate,
                preview_weak,
                tick,
                fit,
                cache,
                SeekPolicy::Exact,
            );
        }
        WorkerMsg::SetParamKeyframe {
            clip,
            param,
            tick,
            value,
            easing,
        } => set_param_keyframe_and_publish(
            engine,
            &clip,
            param,
            RationalTime::new(tick, tl_rate),
            value,
            easing,
            ui,
        ),
        WorkerMsg::SetParamConstant { clip, param, value } => {
            set_param_constant_and_publish(engine, &clip, param, value, ui)
        }
        WorkerMsg::RemoveParamKeyframe { clip, param, tick } => remove_param_keyframe_and_publish(
            engine,
            &clip,
            param,
            RationalTime::new(tick, tl_rate),
            ui,
        ),
        WorkerMsg::RetimeKeyframes {
            clip,
            from_tick,
            to_tick,
        } => retime_keyframes_and_publish(engine, &clip, from_tick, to_tick, tl_rate, ui),
        WorkerMsg::RemoveKeyframesAt { clip, tick } => {
            remove_keyframes_at_and_publish(engine, &clip, tick, tl_rate, ui)
        }
        WorkerMsg::SplitClip { clip, at_tick } => {
            split_clip_and_publish(engine, &clip, at_tick, *linkage, ui)
        }
        WorkerMsg::AddMarker {
            at_tick,
            name,
            color,
        } => add_marker_and_publish(engine, at_tick, &name, &color, tl_rate, ui),
        WorkerMsg::RemoveMarker { marker } => remove_marker_and_publish(engine, &marker, ui),
        WorkerMsg::SetMarker {
            marker,
            at_tick,
            name,
            color,
        } => set_marker_and_publish(engine, &marker, at_tick, &name, &color, tl_rate, ui),
        WorkerMsg::RemoveTrackManual { track } => {
            remove_track_manual_and_publish(engine, &track, ui)
        }
        WorkerMsg::MoveTrackManual { track, index } => {
            move_track_manual_and_publish(engine, &track, index, ui)
        }
        WorkerMsg::SetTrackName { track, name } => {
            set_track_name_and_publish(engine, &track, &name, ui)
        }
        WorkerMsg::Undo => history_step_and_publish(engine, false, ui),
        WorkerMsg::Redo => history_step_and_publish(engine, true, ui),
        WorkerMsg::CopyClips { clips } => {
            // The block origin only matters to duplicate; paste re-bases
            // on the playhead tick.
            if let Some((_, block)) = snapshot_block(engine, &clips) {
                info!(count = block.len(), "copied clips to clipboard");
                *clipboard = Some(block);
            }
        }
        WorkerMsg::PasteAt { tick } => match clipboard {
            Some(block) => paste_and_publish(engine, block, tick, *main_magnet, ui),
            None => info!("paste ignored: clipboard empty"),
        },
        WorkerMsg::DuplicateClips { clips } => {
            duplicate_clips_and_publish(engine, &clips, *main_magnet, ui)
        }
        WorkerMsg::UnlinkClips { clips } => unlink_clips_and_publish(engine, &clips, ui),
        WorkerMsg::SetMainMagnet(enabled) => {
            *main_magnet = enabled;
            info!(enabled, "main-track magnet toggled");
            if enabled {
                pack_main_track_and_publish(engine, ui);
            }
        }
        WorkerMsg::SetLinkage(enabled) => {
            *linkage = enabled;
            info!(enabled, "linkage toggled");
        }
        WorkerMsg::SetTrackFlag { track, flag, value } => {
            set_track_flag_and_publish(engine, &track, flag, value, ui)
        }
        WorkerMsg::Export(request) => start_export(engine, ui, export_state, request),
        WorkerMsg::CancelExport => {
            info!("export cancel requested");
            export_state.cancel.store(true, Ordering::Relaxed);
        }
        WorkerMsg::SaveProject { path } => save_project_and_publish(engine, path, ui),
        WorkerMsg::SaveProjectRpc {
            path,
            reply,
            operation,
        } => serve_worker_rpc(reply, operation, || {
            save_project_rpc_and_publish(engine, path, Some(ui))
        }),
        WorkerMsg::OpenProject { path } => open_project_and_publish(engine, path, ui),
        WorkerMsg::OpenProjectRpc {
            path,
            reply,
            operation,
        } => serve_worker_rpc(reply, operation, || {
            open_project_rpc_and_publish(engine, path, Some(ui))
        }),
        WorkerMsg::RelinkMedia { media, path } => {
            relink_media_and_publish(engine, &media, &path, ui)
        }
        WorkerMsg::RelinkMediaRpc {
            media,
            path,
            reply,
            operation,
        } => serve_worker_rpc(reply, operation, || {
            relink_media_rpc_and_publish(engine, &media, &path, Some(ui))
        }),
        WorkerMsg::RelinkFolder { folder } => relink_folder_and_publish(engine, folder, ui),
        WorkerMsg::RelinkFolderRpc {
            folder,
            reply,
            operation,
        } => serve_worker_rpc(reply, operation, || {
            relink_folder_rpc_and_publish(engine, folder, Some(ui))
        }),
        WorkerMsg::RemoveMedia { media, force } => {
            remove_media_and_publish(engine, &media, force, ui)
        }
        WorkerMsg::NewProject => new_project_and_publish(engine, ui),
        WorkerMsg::NewProjectRpc { reply, operation } => serve_worker_rpc(reply, operation, || {
            new_project_rpc_and_publish(engine, Some(ui))
        }),
        WorkerMsg::ApplyTemplate { path, picks } => {
            apply_template_and_publish(engine, path, picks, ui)
        }
        WorkerMsg::ApplyTemplateRpc {
            path,
            picks,
            reply,
            operation,
        } => serve_worker_rpc(reply, operation, || {
            apply_template_rpc_and_publish(engine, path, picks, Some(ui))
        }),
        WorkerMsg::RenameProject { name } => rename_project_and_publish(engine, name, ui),
        WorkerMsg::GetPreviewCacheStats { reply, operation } => {
            if operation.claim() {
                let _ = reply.send(cache.stats());
            }
        }
        WorkerMsg::ClearPreviewCache { reply, operation } => {
            if operation.claim() {
                let _ = reply.send(cache.clear());
            }
        }
        WorkerMsg::BeginProjectMaintenance {
            reply,
            resume,
            operation,
        } => {
            let action = serve_project_maintenance(engine.project(), reply, resume, operation);
            if action == ProjectMaintenanceResumeAction::RefreshProxies {
                refresh_proxies_after_maintenance(engine, cache, ui);
            }
        }
        WorkerMsg::SnapshotProject { reply } => {
            let _ = reply.send(engine.project().clone());
        }
        WorkerMsg::AgentApplyPlan { phases, reply } => {
            let _ = reply.send(agent_apply_and_publish(engine, phases, ui));
        }
        WorkerMsg::Frame(_) => unreachable!("frames are handled by the drain below"),
        WorkerMsg::TransformOverride { .. } => {
            unreachable!("overrides are handled by the drain below")
        }
        WorkerMsg::BeginTransformGesture { clip, tick } => {
            begin_transform_gesture(engine, &clip, tick, tl_rate, preview_weak, fit, sprite_mode)
        }
        WorkerMsg::EndTransformGesture => {
            if sprite_mode.get() {
                sprite_mode.set(false);
                clear_gesture_sprite_ready(preview_weak);
            }
        }
    }
}
