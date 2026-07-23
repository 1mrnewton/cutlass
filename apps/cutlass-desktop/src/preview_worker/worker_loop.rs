use super::*;

pub struct PreviewWorker {
    handle: WorkerHandle,
    _join: JoinHandle<()>,
}

impl PreviewWorker {
    /// Spawns a dedicated thread that owns the [`Engine`] (required: decoders are not `Send`).
    #[allow(clippy::too_many_arguments)] // one-shot bring-up call from main()
    pub fn spawn(
        config: EngineConfig,
        preview_weak: slint::Weak<PreviewStore<'static>>,
        editor_weak: slint::Weak<EditorStore<'static>>,
        export_weak: slint::Weak<ExportBackend<'static>>,
        audio: crate::audio::AudioHandle,
        thumbs: ThumbnailHandle,
        strips: StripHandle,
        proxy: ProxyHandle,
    ) -> Result<(Self, PreviewSession), String> {
        let (ready_tx, ready_rx) = bounded(1);
        let (req_tx, req_rx) = unbounded();

        let join = std::thread::Builder::new()
            .name("cutlass-preview".into())
            .spawn(move || {
                if let Err(e) = worker_main(
                    config,
                    preview_weak,
                    editor_weak,
                    export_weak,
                    audio,
                    thumbs,
                    strips,
                    proxy,
                    req_rx,
                    ready_tx,
                ) {
                    error!("preview worker exited: {e}");
                }
            })
            .map_err(|e| e.to_string())?;

        let session = ready_rx
            .recv()
            .map_err(|e| e.to_string())?
            .map_err(|e: String| e)?;

        Ok((
            Self {
                handle: WorkerHandle { tx: req_tx },
                _join: join,
            },
            session,
        ))
    }

    /// Clone a sender for a UI callback.
    pub fn handle(&self) -> WorkerHandle {
        self.handle.clone()
    }
}

#[allow(clippy::too_many_arguments)] // one-shot thread entry, mirrors spawn
fn worker_main(
    config: EngineConfig,
    preview_weak: slint::Weak<PreviewStore<'static>>,
    editor_weak: slint::Weak<EditorStore<'static>>,
    export_weak: slint::Weak<ExportBackend<'static>>,
    audio: crate::audio::AudioHandle,
    thumbs: ThumbnailHandle,
    strips: StripHandle,
    proxy: ProxyHandle,
    req_rx: Receiver<WorkerMsg>,
    ready_tx: Sender<Result<PreviewSession, String>>,
) -> Result<(), String> {
    // Start from an empty project: media arrives via user-driven imports
    // (Library → engine), not a hardcoded bootstrap asset.
    let mut engine = Engine::new(config).map_err(|e| e.to_string())?;
    let timeline = engine.project().timeline();
    let session = PreviewSession {
        duration_ticks: timeline.duration().value,
        tl_rate: timeline.frame_rate,
    };
    // Debug, not info: the worker boots an empty engine so it's ready behind
    // the launch screen, but no project exists yet — the user-facing project
    // lifecycle ("new session" / "opened project") logs at info once they
    // actually create or open one.
    debug!(
        duration_ticks = session.duration_ticks,
        tl_rate = ?session.tl_rate,
        "preview worker ready (empty project)"
    );
    let tl_rate = session.tl_rate;
    ready_tx.send(Ok(session)).map_err(|e| e.to_string())?;

    // Seed the UI with the engine's project so the editor reads from the engine
    // from the first frame (rather than any Slint-side placeholder).
    let ui = UiSink {
        editor: editor_weak,
        export: export_weak,
        audio,
        thumbs,
        strips,
        proxy,
    };
    publish_projection(&mut engine, &ui);

    worker_loop(&mut engine, tl_rate, preview_weak, ui, req_rx);
    Ok(())
}

/// Single consumer for the engine thread. Scrub frames coalesce to the latest
/// pending tick, but every mutation (import, add-clip, move-clip, …) is
/// executed in order — it must never be discarded by the coalescing drain.
fn worker_loop(
    engine: &mut Engine,
    tl_rate: Rational,
    preview_weak: slint::Weak<PreviewStore<'static>>,
    ui: UiSink,
    req_rx: Receiver<WorkerMsg>,
) {
    // Clipboard lives with the loop: it's edit-session state, not project
    // state — copies survive any number of edits/undos and die with the app.
    // One block per copy (the whole selection); never empty when `Some`.
    let mut clipboard: Option<Vec<ClipboardClip>> = None;
    // Mirror of TimelineStore.main-magnet-enabled (must match its default).
    // Drag gestures carry their resolved insert flag; this drives the ops
    // without a drag resolution (delete/paste/duplicate) and pack-on-enable.
    let mut main_magnet = true;
    // Mirror of TimelineStore.link-enabled (must match its default): drops
    // of media with audio create linked pairs, trims/splits follow links.
    let mut linkage = true;
    // Fit bound + quality ladder for every preview render (the PreviewFeed
    // model). `Cell`s because the `mutate` closure repaints too.
    let fit = FrameFit::default();
    // Composited frames already delivered this session, keyed by
    // (tick, revision, fit bound) — re-visited scrub positions and hover
    // jitter become buffer clones instead of decode + composite + readback.
    // Interior mutability for the same reason as `fit`.
    let cache = FrameCache::default();
    // Debounced auto-save: an edit arms a deadline; once the worker has been
    // idle of further work for `PERSIST_DEBOUNCE` the dirty draft is written
    // to its `project.cutlass`. Scrub/seek `Frame`s deliberately don't push
    // the deadline out (they're reads), so playback over a pending edit still
    // flushes between frames; a session swap / close forces an immediate
    // write through `SaveProject`.
    let mut persist_deadline: Option<Instant> = None;
    // Last tick the preview rendered (the playhead). Scrub/seek `Frame`s keep
    // it current; edits re-render here so the composite reflects a delete,
    // generator change, etc. without waiting for the user to move the playhead.
    let mut last_tick: i64 = 0;
    // Zero-drift transform gesture: partitioned frames are on the UI thread;
    // per-move worker renders are suppressed until commit/cancel.
    let sprite_mode = Cell::new(false);
    // A post-edit repaint owed but deferred: with more requests already
    // queued, rendering between messages would serialize one slow composite
    // per edit behind a rapid burst (multi-second stalls on weak iGPUs).
    let mut pending_redraw = false;
    // An exact render of `last_tick` owed after a keyframe-snapped scrub frame.
    let mut settle_deadline: Option<Instant> = None;
    // One export job at a time. `active` outlives jobs (the export thread
    // clears it when it exits); `cancel` flags the running job to stop.
    let export_state = ExportJobState::default();

    loop {
        // Settle a deferred post-edit repaint once the burst has drained —
        // exactly one composite per burst instead of one per edit. The
        // render is exact at `last_tick`, so it also pays off any pending
        // snap-settle debt.
        if pending_redraw && req_rx.is_empty() {
            render_frame(
                engine,
                tl_rate,
                &preview_weak,
                last_tick,
                &fit,
                &cache,
                SeekPolicy::Exact,
            );
            pending_redraw = false;
            settle_deadline = None;
        }
        let msg = match next_message(&req_rx, persist_deadline, settle_deadline) {
            Wake::Stop => break,
            // The debounce elapsed: write the dirty draft to its project file.
            Wake::Persist => {
                save_project_and_publish(engine, None, &ui);
                persist_deadline = None;
                continue;
            }
            // The drag paused with a snapped (approximate) frame on screen:
            // replace it with the exact frame. A read, like `Frame` — the
            // persist debounce is deliberately not reset.
            Wake::Settle => {
                settle_deadline = None;
                render_frame(
                    engine,
                    tl_rate,
                    &preview_weak,
                    last_tick,
                    &fit,
                    &cache,
                    SeekPolicy::Exact,
                );
                pending_redraw = false;
                continue;
            }
            Wake::Message(msg) => msg,
        };
        // Scrub/seek frames are reads — they must not push the auto-save
        // deadline out, or playback over a pending edit would never flush.
        let resets_deadline = !matches!(
            msg,
            WorkerMsg::Frame(_)
                | WorkerMsg::GetPreviewCacheStats { .. }
                | WorkerMsg::ClearPreviewCache { .. }
                | WorkerMsg::BeginProjectMaintenance { .. }
        );
        match msg {
            WorkerMsg::Frame(mut tick) => {
                // Frame requests coalesced away right here: when > 0 the UI
                // is producing playhead moves faster than we can render —
                // the signature of an active scrub drag on slow media.
                let mut coalesced = 0usize;
                while let Ok(next) = req_rx.try_recv() {
                    match next {
                        WorkerMsg::Frame(latest) => {
                            tick = latest;
                            coalesced += 1;
                        }
                        WorkerMsg::TransformOverride {
                            clip,
                            transform,
                            tick: at,
                        } => {
                            apply_transform_override(engine, &clip, transform);
                            tick = at;
                        }
                        WorkerMsg::ParamOverride {
                            clip,
                            param,
                            value,
                            tick: at,
                        } => {
                            apply_param_override(engine, &clip, param, value, Some(&ui.audio));
                            tick = at;
                        }
                        other => dispatch(
                            engine,
                            &mut clipboard,
                            &mut main_magnet,
                            &mut linkage,
                            other,
                            tl_rate,
                            &preview_weak,
                            &fit,
                            &cache,
                            &sprite_mode,
                            &export_state,
                            &ui,
                        ),
                    }
                }
                let prev_tick = std::mem::replace(&mut last_tick, tick);
                // Mid-drag (requests outpacing renders): snap to the nearest
                // sync frame — one decode instead of a GOP-prefix walk, so
                // the preview tracks the drag instead of freezing. The exact
                // frame lands via the settle pass once the queue goes idle.
                let dragging = coalesced > 0 || !req_rx.is_empty();
                let policy = if dragging {
                    SeekPolicy::NearestSync
                } else {
                    SeekPolicy::Exact
                };
                render_frame(engine, tl_rate, &preview_weak, tick, &fit, &cache, policy);
                // This render displayed post-edit state at the current tick,
                // covering any repaint a drained mutation deferred.
                pending_redraw = false;
                if policy == SeekPolicy::NearestSync {
                    // Owe an exact render of this tick once the drag pauses.
                    settle_deadline = Some(Instant::now() + SNAP_SETTLE_DELAY);
                } else {
                    settle_deadline = None;
                    // Steady forward motion (playback, frame-stepping, a slow
                    // forward drag) requests evenly spaced ticks: with nothing
                    // else queued, render the predicted next tick into the cache
                    // now. Its decode overlaps the UI's display time of `tick`,
                    // so the request that follows is a hit; if playback skips
                    // ahead instead, the work still advanced the decoder's
                    // roll-forward cursor, so nothing is lost. Gestures render
                    // uncacheable override state — never speculate under one.
                    let delta = tick - prev_tick;
                    if (1..=MAX_SPECULATIVE_STEP).contains(&delta)
                        && req_rx.is_empty()
                        && !engine.has_live_overrides()
                    {
                        let _ = render_frame_buffer(
                            engine,
                            tl_rate,
                            tick + delta,
                            &fit,
                            &cache,
                            SeekPolicy::Exact,
                        );
                    }
                }
            }
            WorkerMsg::BeginTransformGesture { clip, tick } => {
                last_tick = tick;
                begin_transform_gesture(
                    engine,
                    &clip,
                    tick,
                    tl_rate,
                    &preview_weak,
                    &fit,
                    &sprite_mode,
                );
            }
            WorkerMsg::EndTransformGesture => {
                if sprite_mode.get() {
                    sprite_mode.set(false);
                    clear_gesture_sprite_ready(&preview_weak);
                }
            }
            // Drag-gesture overrides arrive at pointer-move rate; render only
            // the newest one (same coalescing as scrub frames) so a fast drag
            // can't back the queue up behind stale composites.
            WorkerMsg::TransformOverride {
                mut clip,
                mut transform,
                mut tick,
            } => {
                // Queue order must hold against drained mutations: the
                // release's SetTransform often lands right behind the last
                // pointer-move, and it commits + clears the override. Apply
                // the coalesced override *before* such a mutation and never
                // after it, or the stale gesture override outlives the commit
                // and pins the clip's transform on every later frame
                // (keyframed animation freezes in preview until re-cleared).
                let mut pending = true;
                while let Ok(next) = req_rx.try_recv() {
                    match next {
                        WorkerMsg::Frame(latest) => tick = latest,
                        WorkerMsg::TransformOverride {
                            clip: c,
                            transform: t,
                            tick: at,
                        } => {
                            clip = c;
                            transform = t;
                            tick = at;
                            pending = true;
                        }
                        other => {
                            if std::mem::take(&mut pending) {
                                apply_transform_override(engine, &clip, transform);
                            }
                            dispatch(
                                engine,
                                &mut clipboard,
                                &mut main_magnet,
                                &mut linkage,
                                other,
                                tl_rate,
                                &preview_weak,
                                &fit,
                                &cache,
                                &sprite_mode,
                                &export_state,
                                &ui,
                            )
                        }
                    }
                }
                last_tick = tick;
                if pending {
                    apply_transform_override(engine, &clip, transform);
                }
                if sprite_mode.get() {
                    // UI-side sprite compositing owns mid-gesture pixels.
                } else {
                    render_frame(
                        engine,
                        tl_rate,
                        &preview_weak,
                        tick,
                        &fit,
                        &cache,
                        SeekPolicy::Exact,
                    );
                    pending_redraw = false;
                    settle_deadline = None;
                }
            }
            // Live inspector edits (font-size drag) arrive at pointer-move
            // rate; coalesce to the newest like transform overrides do.
            WorkerMsg::GeneratorOverride {
                mut clip,
                mut generator,
                mut tick,
            } => {
                // Same ordering rule as TransformOverride above: a drained
                // mutation (the release's SetGenerator / ClearGeneratorOverride)
                // must not be followed by a re-apply of the override it ended.
                let mut pending = true;
                while let Ok(next) = req_rx.try_recv() {
                    match next {
                        WorkerMsg::Frame(latest) => tick = latest,
                        WorkerMsg::GeneratorOverride {
                            clip: c,
                            generator: g,
                            tick: at,
                        } => {
                            clip = c;
                            generator = g;
                            tick = at;
                            pending = true;
                        }
                        other => {
                            if std::mem::take(&mut pending) {
                                apply_generator_override(engine, &clip, generator.clone());
                            }
                            dispatch(
                                engine,
                                &mut clipboard,
                                &mut main_magnet,
                                &mut linkage,
                                other,
                                tl_rate,
                                &preview_weak,
                                &fit,
                                &cache,
                                &sprite_mode,
                                &export_state,
                                &ui,
                            )
                        }
                    }
                }
                last_tick = tick;
                if pending {
                    apply_generator_override(engine, &clip, generator);
                }
                render_frame(
                    engine,
                    tl_rate,
                    &preview_weak,
                    tick,
                    &fit,
                    &cache,
                    SeekPolicy::Exact,
                );
                pending_redraw = false;
                settle_deadline = None;
            }
            // Shape resize drags (width/height sliders) arrive at pointer-move
            // rate; coalesce to the newest like the generator/transform
            // overrides so a fast drag can't back the render queue up. The
            // override generator is rebuilt from committed engine state, so the
            // drained-mutation ordering rule (above) applies unchanged.
            WorkerMsg::PreviewShapeSize {
                mut clip,
                mut width,
                mut height,
                mut tick,
            } => {
                let mut pending = true;
                while let Ok(next) = req_rx.try_recv() {
                    match next {
                        WorkerMsg::Frame(latest) => tick = latest,
                        WorkerMsg::PreviewShapeSize {
                            clip: c,
                            width: w,
                            height: h,
                            tick: at,
                        } => {
                            clip = c;
                            width = w;
                            height = h;
                            tick = at;
                            pending = true;
                        }
                        other => {
                            if std::mem::take(&mut pending)
                                && let Some(generator) =
                                    shape_size_from_engine(engine, &clip, width, height)
                            {
                                apply_generator_override(engine, &clip, generator);
                            }
                            dispatch(
                                engine,
                                &mut clipboard,
                                &mut main_magnet,
                                &mut linkage,
                                other,
                                tl_rate,
                                &preview_weak,
                                &fit,
                                &cache,
                                &sprite_mode,
                                &export_state,
                                &ui,
                            )
                        }
                    }
                }
                last_tick = tick;
                if pending
                    && let Some(generator) = shape_size_from_engine(engine, &clip, width, height)
                {
                    apply_generator_override(engine, &clip, generator);
                }
                render_frame(
                    engine,
                    tl_rate,
                    &preview_weak,
                    tick,
                    &fit,
                    &cache,
                    SeekPolicy::Exact,
                );
                pending_redraw = false;
                settle_deadline = None;
            }
            // Look drags (filter intensity / adjust sliders) carry the whole
            // grade so preview frames never mix a new adjustment with a stale
            // filter or vice versa. Coalesce to the newest value like other
            // inspector preview overrides.
            WorkerMsg::PreviewClipLook {
                mut clip,
                mut filter_id,
                mut intensity,
                mut adjust,
                mut tick,
            } => {
                let mut pending = true;
                while let Ok(next) = req_rx.try_recv() {
                    match next {
                        WorkerMsg::Frame(latest) => tick = latest,
                        WorkerMsg::PreviewClipLook {
                            clip: c,
                            filter_id: f,
                            intensity: i,
                            adjust: a,
                            tick: at,
                        } => {
                            clip = c;
                            filter_id = f;
                            intensity = i;
                            adjust = a;
                            tick = at;
                            pending = true;
                        }
                        other => {
                            if std::mem::take(&mut pending) {
                                apply_look_override(engine, &clip, &filter_id, intensity, &adjust);
                            }
                            dispatch(
                                engine,
                                &mut clipboard,
                                &mut main_magnet,
                                &mut linkage,
                                other,
                                tl_rate,
                                &preview_weak,
                                &fit,
                                &cache,
                                &sprite_mode,
                                &export_state,
                                &ui,
                            )
                        }
                    }
                }
                last_tick = tick;
                if pending {
                    apply_look_override(engine, &clip, &filter_id, intensity, &adjust);
                }
                render_frame(
                    engine,
                    tl_rate,
                    &preview_weak,
                    tick,
                    &fit,
                    &cache,
                    SeekPolicy::Exact,
                );
            }
            // Layer-style slider drags carry a single param delta; the worker
            // rebuilds the full styles override from committed engine state +
            // that delta. Coalesce to the newest delta (latest wins per clip)
            // like look/transform/generator overrides.
            WorkerMsg::PreviewClipStyleDelta {
                mut clip,
                mut key,
                mut value_x,
                mut value_y,
                mut tick,
            } => {
                let mut pending = true;
                while let Ok(next) = req_rx.try_recv() {
                    match next {
                        WorkerMsg::Frame(latest) => tick = latest,
                        WorkerMsg::PreviewClipStyleDelta {
                            clip: c,
                            key: k,
                            value_x: x,
                            value_y: y,
                            tick: at,
                        } => {
                            clip = c;
                            key = k;
                            value_x = x;
                            value_y = y;
                            tick = at;
                            pending = true;
                        }
                        other => {
                            if std::mem::take(&mut pending) {
                                apply_styles_preview_delta(
                                    engine, &clip, &key, value_x, value_y, tick,
                                );
                            }
                            dispatch(
                                engine,
                                &mut clipboard,
                                &mut main_magnet,
                                &mut linkage,
                                other,
                                tl_rate,
                                &preview_weak,
                                &fit,
                                &cache,
                                &sprite_mode,
                                &export_state,
                                &ui,
                            )
                        }
                    }
                }
                last_tick = tick;
                if pending {
                    apply_styles_preview_delta(engine, &clip, &key, value_x, value_y, tick);
                }
                render_frame(
                    engine,
                    tl_rate,
                    &preview_weak,
                    tick,
                    &fit,
                    &cache,
                    SeekPolicy::Exact,
                );
            }
            // Motion-blur shutter/samples drags: coalesce latest field delta
            // per clip (worker merges against committed blur).
            WorkerMsg::PreviewMotionBlurDelta {
                clip,
                key,
                value,
                tick,
            } => {
                let tick = coalesce_motion_blur_deltas(
                    engine,
                    &mut clipboard,
                    &mut main_magnet,
                    &mut linkage,
                    clip,
                    key,
                    value,
                    tick,
                    &req_rx,
                    tl_rate,
                    &preview_weak,
                    &fit,
                    &cache,
                    &sprite_mode,
                    &export_state,
                    &ui,
                );
                last_tick = tick;
                render_frame(
                    engine,
                    tl_rate,
                    &preview_weak,
                    tick,
                    &fit,
                    &cache,
                    SeekPolicy::Exact,
                );
            }
            // Animation speed/intensity/stagger drags: same coalesce rule.
            WorkerMsg::PreviewClipAnimationDelta {
                clip,
                slot,
                key,
                value,
                tick,
            } => {
                let tick = coalesce_animation_deltas(
                    engine,
                    &mut clipboard,
                    &mut main_magnet,
                    &mut linkage,
                    clip,
                    slot,
                    key,
                    value,
                    tick,
                    &req_rx,
                    tl_rate,
                    &preview_weak,
                    &fit,
                    &cache,
                    &sprite_mode,
                    &export_state,
                    &ui,
                );
                last_tick = tick;
                render_frame(
                    engine,
                    tl_rate,
                    &preview_weak,
                    tick,
                    &fit,
                    &cache,
                    SeekPolicy::Exact,
                );
            }
            // Generic inspector param drags (crop, chroma, effect knobs, …)
            // arrive at pointer-move rate; coalesce to the newest value per
            // (clip, param) and render once — same queue-pressure rule as
            // TransformOverride / look / styles.
            WorkerMsg::ParamOverride {
                clip,
                param,
                value,
                tick,
            } => {
                let tick = coalesce_param_overrides(
                    engine,
                    &mut clipboard,
                    &mut main_magnet,
                    &mut linkage,
                    clip,
                    param,
                    value,
                    tick,
                    &req_rx,
                    tl_rate,
                    &preview_weak,
                    &fit,
                    &cache,
                    &sprite_mode,
                    &export_state,
                    &ui,
                );
                last_tick = tick;
                render_frame(
                    engine,
                    tl_rate,
                    &preview_weak,
                    tick,
                    &fit,
                    &cache,
                    SeekPolicy::Exact,
                );
                pending_redraw = false;
                settle_deadline = None;
            }
            // The preview panel resized (or first laid out): renders now fit
            // the new bound. Repaint the current frame only when the bucketed
            // size actually changed — live window resizes report every frame.
            WorkerMsg::Viewport { width, height } => {
                if fit.set_viewport(width, height) {
                    render_frame(
                        engine,
                        tl_rate,
                        &preview_weak,
                        last_tick,
                        &fit,
                        &cache,
                        SeekPolicy::Exact,
                    );
                    pending_redraw = false;
                    settle_deadline = None;
                }
            }
            other => {
                let redraw = message_invalidates_preview(&other);
                dispatch(
                    engine,
                    &mut clipboard,
                    &mut main_magnet,
                    &mut linkage,
                    other,
                    tl_rate,
                    &preview_weak,
                    &fit,
                    &cache,
                    &sprite_mode,
                    &export_state,
                    &ui,
                );
                // Edits otherwise only repaint when the playhead moves; owe a
                // repaint so the change becomes visible. The loop-top flush
                // renders it as soon as the queue is idle — immediately for a
                // lone edit, once per burst for rapid ones instead of one
                // composite per edit (each is a multi-second stall on weak
                // iGPUs).
                if redraw {
                    pending_redraw = true;
                }
            }
        }
        // After any non-scrub work, (re)arm the debounce while the draft has
        // unsaved edits, or clear it once a save/swap left the draft clean.
        if resets_deadline {
            persist_deadline = engine.is_dirty().then(|| Instant::now() + PERSIST_DEBOUNCE);
        }
    }
}

/// Idle gap after the last edit before the dirty draft is written. Long
/// enough to coalesce a burst of rapid edits (typing a title, dragging a
/// slider) into one write, short enough that work is never far from disk.
const PERSIST_DEBOUNCE: Duration = Duration::from_millis(300);

/// Idle gap after a keyframe-snapped scrub frame before the exact frame is
/// rendered in its place. Longer than the lull between pointer-move events
/// in an ongoing drag (so a slow drag doesn't stall on mid-drag exact
/// renders), short enough that the sharp frame feels immediate on release.
const SNAP_SETTLE_DELAY: Duration = Duration::from_millis(250);

/// What the worker should do next (see [`next_message`]).
///
/// `Message` dwarfs the marker variants (a `WorkerMsg` carries whole
/// generators), but a `Wake` only ever lives on the stack between
/// `next_message` and the match on it — boxing would add a heap round-trip
/// per message on the edit path to save nothing.
#[allow(clippy::large_enum_variant)]
enum Wake {
    /// A request arrived; process it.
    Message(WorkerMsg),
    /// The debounce elapsed with the draft dirty; write it.
    Persist,
    /// The snap-settle delay elapsed with an approximate frame on screen;
    /// render the exact frame.
    Settle,
    /// The request channel closed; the loop should exit.
    Stop,
}

/// Block for the next request, waking for the earlier of the auto-save and
/// snap-settle deadlines when one passes. With neither pending it's a plain
/// blocking receive. A settle/persist tie goes to settle — it repaints what
/// the user is looking at, and the persist deadline fires on the next turn.
fn next_message(
    req_rx: &Receiver<WorkerMsg>,
    persist: Option<Instant>,
    settle: Option<Instant>,
) -> Wake {
    let deadline = match (persist, settle) {
        (None, None) => None,
        (Some(p), None) => Some((p, Wake::Persist)),
        (None, Some(s)) => Some((s, Wake::Settle)),
        (Some(p), Some(s)) => Some(if s <= p {
            (s, Wake::Settle)
        } else {
            (p, Wake::Persist)
        }),
    };
    match deadline {
        None => match req_rx.recv() {
            Ok(msg) => Wake::Message(msg),
            Err(_) => Wake::Stop,
        },
        Some((deadline, wake)) => {
            let timeout = deadline.saturating_duration_since(Instant::now());
            match req_rx.recv_timeout(timeout) {
                Ok(msg) => Wake::Message(msg),
                Err(RecvTimeoutError::Timeout) => wake,
                Err(RecvTimeoutError::Disconnected) => Wake::Stop,
            }
        }
    }
}
