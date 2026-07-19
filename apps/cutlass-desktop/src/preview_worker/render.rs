use super::*;

/// Text uses the live full-frame override path during transform gestures.
/// Its transparent full-canvas sprite is valid at the renderer boundary, but
/// the transformed embedded image is not reliably painted by Slint on macOS
/// (the selection overlay moves while the title disappears). Other clip kinds
/// retain the cheaper partitioned-sprite path.
fn supports_ui_gesture_sprite(project: &Project, clip_id: ClipId) -> bool {
    !project
        .clip(clip_id)
        .is_some_and(|clip| matches!(&clip.content, ClipSource::Generated(Generator::Text { .. })))
}

pub(super) fn clear_gesture_sprite_ready(preview_weak: &slint::Weak<PreviewStore<'static>>) {
    let weak = preview_weak.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        if let Some(store) = weak.upgrade() {
            store.set_gesture_sprite_ready(false);
        }
    }) {
        error!("failed to clear gesture sprite mode on UI: {e}");
    }
}

pub(super) fn begin_transform_gesture(
    engine: &mut Engine,
    clip: &str,
    tick: i64,
    tl_rate: Rational,
    preview_weak: &slint::Weak<PreviewStore<'static>>,
    fit: &FrameFit,
    sprite_mode: &Cell<bool>,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "begin transform gesture ignored: unparsable clip id");
        return;
    };
    if !supports_ui_gesture_sprite(engine.project(), clip_id) {
        // TransformOverride will render the complete frame on each move while
        // sprite_mode is false, keeping text pixels and controls in lockstep.
        sprite_mode.set(false);
        clear_gesture_sprite_ready(preview_weak);
        debug!(%clip_id, tick, "text gesture using live frame preview");
        return;
    }
    let at = RationalTime::new(tick, tl_rate);
    let started = Instant::now();
    let result = match fit.fit_bound() {
        Some((max_w, max_h)) => engine.get_gesture_frames(at, clip_id, max_w, max_h),
        None => Ok(None),
    };
    match result {
        Ok(Some(frames)) => {
            fit.note_render_cost(started.elapsed().as_secs_f64() * 1000.0);
            sprite_mode.set(true);
            let weak = preview_weak.clone();
            if let Err(e) = slint::invoke_from_event_loop(move || {
                if let Some(store) = weak.upgrade() {
                    store.set_gesture_frame_below(slint_image_from_rgba(frames.below));
                    store.set_gesture_frame_sprite(slint_image_from_rgba(frames.sprite));
                    if let Some(above) = frames.above {
                        store.set_gesture_frame_above(slint_image_from_rgba(above));
                        store.set_gesture_has_above(true);
                    } else {
                        store.set_gesture_has_above(false);
                    }
                    store.set_gesture_sprite_ready(true);
                }
            }) {
                error!("failed to deliver gesture frames to UI: {e}");
                sprite_mode.set(false);
            }
        }
        Ok(None) => {
            debug!(%clip_id, tick, "gesture sprite unavailable; using override path");
            sprite_mode.set(false);
        }
        Err(e) => {
            error!(%clip_id, tick, "gesture frame render failed: {e}");
            sprite_mode.set(false);
        }
    }
}

pub(super) fn render_frame_exit_sprite(
    engine: &mut Engine,
    tl_rate: Rational,
    preview_weak: &slint::Weak<PreviewStore<'static>>,
    tick: i64,
    fit: &FrameFit,
    sprite_mode: &Cell<bool>,
) {
    let clear_sprite = sprite_mode.get();
    sprite_mode.set(false);
    let at = RationalTime::new(tick, tl_rate);
    let started = Instant::now();
    let result = match fit.fit_bound() {
        Some((max_w, max_h)) => engine.get_frame_fit(at, max_w, max_h),
        None => engine.get_frame(at),
    };
    match result {
        Ok(frame) => {
            fit.note_render_cost(started.elapsed().as_secs_f64() * 1000.0);
            let weak = preview_weak.clone();
            if let Err(e) = slint::invoke_from_event_loop(move || {
                if let Some(store) = weak.upgrade() {
                    if clear_sprite {
                        store.set_gesture_sprite_ready(false);
                    }
                    store.set_frame(slint_image_from_rgba(frame));
                }
            }) {
                error!("failed to deliver preview frame to UI: {e}");
            }
        }
        Err(e) => {
            if clear_sprite {
                clear_gesture_sprite_ready(preview_weak);
            }
            error!(tick, "preview frame failed: {e}");
        }
    }
}

fn slint_image_from_rgba(frame: cutlass_render::RgbaImage) -> slint::Image {
    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(frame.width, frame.height);
    buffer.make_mut_bytes().copy_from_slice(&frame.pixels);
    slint::Image::from_rgba8(buffer)
}

pub(super) fn render_frame(
    engine: &mut Engine,
    tl_rate: Rational,
    preview_weak: &slint::Weak<PreviewStore<'static>>,
    tick: i64,
    fit: &FrameFit,
    cache: &FrameCache,
    policy: SeekPolicy,
) {
    if let Some(buffer) = render_frame_buffer(engine, tl_rate, tick, fit, cache, policy) {
        deliver_frame(preview_weak, buffer);
    }
}

/// Allocates the UI pixel buffer at the size the renderer reports and lets
/// the compositor's readback write de-padded rows straight into it — the
/// single CPU copy between the mapped GPU buffer and the pixels Slint
/// displays. A fresh buffer per rendered frame is deliberate: the cache and
/// the UI keep refcounted clones of it, so reusing one would silently
/// copy-on-write.
#[derive(Default)]
struct SlintFrameSink {
    buffer: Option<SharedPixelBuffer<Rgba8Pixel>>,
}

impl cutlass_render::FrameSink for SlintFrameSink {
    fn pixels(&mut self, width: u32, height: u32) -> &mut [u8] {
        self.buffer
            .insert(SharedPixelBuffer::new(width, height))
            .make_mut_bytes()
    }
}

/// Produce the composited frame at `tick` — from the cache when the same
/// (tick, revision, fit) was already rendered, otherwise by rendering and
/// caching. Cache hits don't feed the quality ladder (nothing was rendered).
/// Frames rendered under a live gesture override belong to no revision, and
/// frames rendered under [`SeekPolicy::NearestSync`] are approximations
/// (the keyframe near `tick`, not the frame *at* it) — both bypass the
/// cache in both directions so wrong pixels are never stored under an
/// exact key.
pub(super) fn render_frame_buffer(
    engine: &mut Engine,
    tl_rate: Rational,
    tick: i64,
    fit: &FrameFit,
    cache: &FrameCache,
    policy: SeekPolicy,
) -> Option<SharedPixelBuffer<Rgba8Pixel>> {
    let bound = fit.fit_bound();
    let cacheable = !engine.has_live_overrides() && policy == SeekPolicy::Exact;
    let key = FrameKey {
        tick,
        revision: engine.revision(),
        bound,
    };
    if cacheable && let Some(buffer) = cache.get(&key) {
        return Some(buffer);
    }

    let at = RationalTime::new(tick, tl_rate);
    let started = Instant::now();
    let mut sink = SlintFrameSink::default();
    let result = match bound {
        Some((max_w, max_h)) => engine.get_frame_fit_into(at, max_w, max_h, policy, &mut sink),
        None => engine.get_frame_into(at, policy, &mut sink),
    };
    match result {
        Ok(()) => {
            let elapsed = started.elapsed();
            // The ladder sees only the resolution-dependent share of the
            // cost (raster + composite + readback): decode is native-size
            // work no tier change can reduce, so it must not drop quality.
            fit.note_render_cost(engine.last_frame_stats().scaled_cost_ms());
            // Total preview latency for this frame (decode + composite +
            // readback + copy). The renderer logs the per-stage split; this
            // warn is the default-visible "the preview is why edits feel
            // slow" breadcrumb, with the fit state that produced it.
            let ms = elapsed.as_secs_f64() * 1000.0;
            if ms > SLOW_PREVIEW_WARN_MS {
                warn!(
                    tick,
                    ?bound,
                    tier = fit.tier.get(),
                    "slow preview render: {ms:.0} ms"
                );
            } else {
                debug!(tick, ?bound, "preview render: {ms:.1} ms");
            }
            let buffer = sink.buffer.expect("successful render fills the sink");
            if cacheable {
                cache.insert(key, buffer.clone());
            }
            Some(buffer)
        }
        Err(e) => {
            error!(tick, "preview frame failed: {e}");
            None
        }
    }
}

/// Hand `buffer` to the preview panel. The buffer is refcounted, so the
/// event-loop hop and the `Image` wrapper share the worker's allocation.
fn deliver_frame(
    preview_weak: &slint::Weak<PreviewStore<'static>>,
    buffer: SharedPixelBuffer<Rgba8Pixel>,
) {
    let weak = preview_weak.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        if let Some(store) = weak.upgrade() {
            store.set_frame(slint::Image::from_rgba8(buffer));
        }
    }) {
        error!("failed to deliver preview frame to UI: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_transform_gestures_use_live_frames_instead_of_ui_sprites() {
        let mut project = Project::new("p", Rational::FPS_24);
        let text_track = project.add_track(TrackKind::Text, "T1");
        let text = project
            .add_generated(
                text_track,
                Generator::Text {
                    content: "hello".into(),
                    style: Default::default(),
                },
                TimeRange::at_rate(0, 24, Rational::FPS_24),
            )
            .unwrap();
        let shape_track = project.add_track(TrackKind::Sticker, "S1");
        let shape = project
            .add_generated(
                shape_track,
                Generator::SolidColor {
                    rgba: [255, 0, 0, 255],
                },
                TimeRange::at_rate(0, 24, Rational::FPS_24),
            )
            .unwrap();

        assert!(!supports_ui_gesture_sprite(&project, text));
        assert!(supports_ui_gesture_sprite(&project, shape));
    }
}
