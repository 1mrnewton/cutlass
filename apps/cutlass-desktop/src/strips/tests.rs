use super::*;

struct StripCacheCleanup;

impl Drop for StripCacheCleanup {
    fn drop(&mut self) {
        clear_all_caches();
    }
}

fn test_image(width: u32, height: u32) -> Image {
    let rgba = vec![0; width as usize * height as usize * 4];
    Image::from_rgba8(SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(
        &rgba, width, height,
    ))
}

#[test]
fn strip_cache_stats_and_clear_are_exact_and_idempotent() {
    clear_all_caches();
    let _cleanup = StripCacheCleanup;

    FILMSTRIPS.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.insert(
            (1, 0),
            CacheEntry {
                image: test_image(2, 3),
                last_used: 1,
            },
        );
        cache.insert(
            (1, 1_000_000),
            CacheEntry {
                image: test_image(4, 5),
                last_used: 2,
            },
        );
    });
    WAVES.with(|cache| {
        cache.borrow_mut().insert(
            (2, 0, -1),
            CacheEntry {
                image: test_image(7, 2),
                last_used: 3,
            },
        );
    });

    let expected = StripCacheStats {
        filmstrips: StripImageCacheStats {
            entry_count: 2,
            estimated_rgba_bytes: 2 * 3 * 4 + 4 * 5 * 4,
        },
        waves: StripImageCacheStats {
            entry_count: 1,
            estimated_rgba_bytes: 7 * 2 * 4,
        },
    };
    assert_eq!(cache_stats(), expected);
    assert_eq!(clear_all_caches(), expected);
    assert_eq!(cache_stats(), StripCacheStats::default());
    assert_eq!(clear_all_caches(), StripCacheStats::default());
}

#[test]
fn strip_proxy_clear_is_idempotent_and_preserves_original_and_waveform_state() {
    let media_id = 17;
    let original = PathBuf::from("/media/original.mov");
    let mut state = WorkerState::default();
    state.paths.insert(media_id, original.clone());
    state
        .proxies
        .insert(media_id, PathBuf::from("/old-proxies/proxy.mp4"));
    state.scratch.insert(
        media_id,
        ScratchClip {
            project: Project::new("scratch", Rational::FPS_30),
            rate: Rational::FPS_30,
            duration_frames: 30,
        },
    );
    state.frames_failed.insert(media_id);
    state.peaks.insert(
        media_id,
        AudioPeaks {
            per_second: 100.0,
            peaks: vec![0.25, 0.5],
        },
    );
    state.peaks_failed.insert(99);

    let (tx, rx) = unbounded();
    let handle = StripHandle { tx };
    let mut backlog = Vec::new();
    handle.clear_proxies();
    triage(rx.recv().unwrap(), &mut state, &mut backlog);

    assert_eq!(state.paths.get(&media_id), Some(&original));
    assert!(state.proxies.is_empty());
    assert!(state.scratch.is_empty());
    assert!(state.frames_failed.is_empty());
    assert!(state.renderer.is_none());
    assert_eq!(state.peaks[&media_id].peaks, vec![0.25, 0.5]);
    assert!(state.peaks_failed.contains(&99));
    assert!(backlog.is_empty());

    handle.clear_proxies();
    triage(rx.recv().unwrap(), &mut state, &mut backlog);
    assert_eq!(state.paths.get(&media_id), Some(&original));
    assert!(state.proxies.is_empty());
    assert!(state.scratch.is_empty());
    assert_eq!(state.peaks[&media_id].peaks, vec![0.25, 0.5]);
    assert!(state.peaks_failed.contains(&99));
    assert!(backlog.is_empty());
}

#[test]
fn clearing_strip_caches_resets_pending_request_suppression() {
    clear_all_caches();
    let _cleanup = StripCacheCleanup;
    let frame_key = (41, 2_000_000);
    let wave_key = (42, 3_000_000, 1);

    PENDING_FRAMES.with(|pending| {
        let mut pending = pending.borrow_mut();
        assert!(pending.insert(frame_key));
        assert!(!pending.insert(frame_key), "duplicate frame is suppressed");
    });
    PENDING_WAVES.with(|pending| {
        let mut pending = pending.borrow_mut();
        assert!(pending.insert(wave_key));
        assert!(!pending.insert(wave_key), "duplicate wave is suppressed");
    });
    USE_TICK.with(|tick| tick.set(99));

    assert_eq!(clear_all_caches(), StripCacheStats::default());
    assert_eq!(USE_TICK.with(Cell::get), 0);
    PENDING_FRAMES.with(|pending| {
        assert!(
            pending.borrow_mut().insert(frame_key),
            "frame can be requested again after clear"
        );
    });
    PENDING_WAVES.with(|pending| {
        assert!(
            pending.borrow_mut().insert(wave_key),
            "wave can be requested again after clear"
        );
    });
}

#[test]
fn strip_cache_kinds_can_be_cleared_independently() {
    clear_all_caches();
    let _cleanup = StripCacheCleanup;
    FILMSTRIPS.with(|cache| {
        cache.borrow_mut().insert(
            (1, 0),
            CacheEntry {
                image: test_image(2, 2),
                last_used: 1,
            },
        );
    });
    WAVES.with(|cache| {
        cache.borrow_mut().insert(
            (2, 0, 0),
            CacheEntry {
                image: test_image(3, 2),
                last_used: 2,
            },
        );
    });
    PENDING_FRAMES.with(|pending| {
        pending.borrow_mut().insert((1, 0));
    });
    PENDING_WAVES.with(|pending| {
        pending.borrow_mut().insert((2, 0, 0));
    });
    USE_TICK.with(|tick| tick.set(7));

    assert_eq!(
        clear_filmstrips(),
        StripImageCacheStats {
            entry_count: 1,
            estimated_rgba_bytes: 16,
        }
    );
    assert_eq!(cache_stats().filmstrips, StripImageCacheStats::default());
    assert_eq!(cache_stats().waves.entry_count, 1);
    assert!(PENDING_FRAMES.with(|pending| pending.borrow().is_empty()));
    assert_eq!(PENDING_WAVES.with(|pending| pending.borrow().len()), 1);
    assert_eq!(USE_TICK.with(Cell::get), 7);

    assert_eq!(clear_waveforms().entry_count, 1);
    assert!(PENDING_WAVES.with(|pending| pending.borrow().is_empty()));
    assert_eq!(USE_TICK.with(Cell::get), 0);
}

#[test]
fn choose_k_picks_smallest_interval_at_least_min_px() {
    // 100 px/s: 1s tile = 100px ≥ 64, 0.5s = 50px < 64 → k = 0.
    assert_eq!(choose_k(100.0, 64.0), 0);
    // 600 px/s: 0.125s = 75px ≥ 64, 0.0625s = 37.5px → k = -3.
    assert_eq!(choose_k(600.0, 64.0), -3);
    // 1 px/s: need 64s → next power of two is 64 → k = 6.
    assert_eq!(choose_k(1.0, 64.0), 6);
    // Degenerate zooms clamp instead of exploding.
    assert_eq!(choose_k(1e-12, 64.0), MAX_K);
    assert_eq!(choose_k(1e12, 64.0), MIN_K);
}

#[test]
fn interval_us_is_exact_for_all_allowed_k() {
    assert_eq!(interval_us_for(0), 1_000_000);
    assert_eq!(interval_us_for(2), 4_000_000);
    assert_eq!(interval_us_for(-2), 250_000);
    assert_eq!(interval_us_for(MIN_K), 15_625);
    // Exactness: interval reconstructs 2^k seconds with no remainder.
    for k in MIN_K..=6 {
        let us = interval_us_for(k);
        assert_eq!(us as f64 / 1e6, 2.0_f64.powi(k), "k={k}");
    }
}

#[test]
fn coarser_grids_nest_inside_finer_ones() {
    // Every key on the k+1 grid is also on the k grid: zooming in reuses
    // every frame already decoded while zoomed out.
    for k in MIN_K..10 {
        assert_eq!(interval_us_for(k + 1) % interval_us_for(k), 0);
    }
}

#[test]
fn plan_covers_the_visible_window() {
    // 30fps, zoom 2px/tick → 60 px/s → 1s tiles are 60px < 64 → k=1:
    // 2s tiles, 120px each.
    let grid = plan_tiles(0.0, 3000, 30, 1, 1.0, 2.0, 0, 3, 64.0).expect("grid");
    assert_eq!(grid.k, 1);
    assert_eq!(grid.i0, 0);
    // Window: 0..1024px (+1 tile pad) of a 6000px clip → ~9 tiles.
    assert!(grid.i1 >= 8 && grid.i1 <= 10, "i1 = {}", grid.i1);
    assert_eq!(grid.x_px(0), 0.0);
    assert!((grid.width_px() - 120.0).abs() < 1e-9);
}

#[test]
fn plan_clamps_to_the_clip_extent() {
    // Visible window far past the clip's end → nothing.
    assert_eq!(plan_tiles(0.0, 100, 30, 1, 1.0, 1.0, 10, 20, 64.0), None);
    // Window before the clip (negative buckets) → nothing.
    assert_eq!(plan_tiles(0.0, 100, 30, 1, 1.0, 1.0, -20, -10, 64.0), None);
}

#[test]
fn plan_anchors_tiles_to_media_time() {
    // A clip trimmed 3.5s into its source at k=1 (2s tiles): the first
    // visible tile is the grid tile containing 3.5s (i=1 → 2s), drawn
    // 1.5s-worth of px to the left of the clip edge.
    let grid = plan_tiles(3.5, 3000, 30, 1, 1.0, 2.0, 0, 3, 64.0).expect("grid");
    assert_eq!(grid.k, 1);
    assert_eq!(grid.i0, 1);
    assert_eq!(grid.key_us(grid.i0), 2_000_000);
    assert!((grid.x_px(1) - (2.0 - 3.5) * 60.0).abs() < 1e-9);
}

#[test]
fn plan_rejects_degenerate_inputs() {
    assert_eq!(plan_tiles(0.0, 0, 30, 1, 1.0, 1.0, 0, 3, 64.0), None);
    assert_eq!(plan_tiles(0.0, 100, 0, 1, 1.0, 1.0, 0, 3, 64.0), None);
    assert_eq!(plan_tiles(0.0, 100, 30, 1, 1.0, 0.0, 0, 3, 64.0), None);
}

#[test]
fn plan_scales_px_per_source_second_by_speed() {
    // Same zoom, 2× speed: each source second occupies half the px, so
    // the source window covered by the same clip width doubles.
    let normal = plan_tiles(0.0, 3000, 30, 1, 1.0, 2.0, 0, 3, 64.0).expect("grid");
    let double = plan_tiles(0.0, 3000, 30, 1, 2.0, 2.0, 0, 3, 64.0).expect("grid");
    assert!((normal.px_per_s - 60.0).abs() < 1e-9);
    assert!((double.px_per_s - 30.0).abs() < 1e-9);
    // Degenerate speeds fall back to 1×.
    let fallback = plan_tiles(0.0, 3000, 30, 1, 0.0, 2.0, 0, 3, 64.0).expect("grid");
    assert!((fallback.px_per_s - 60.0).abs() < 1e-9);
}

#[test]
fn plan_caps_the_tile_count() {
    // Absurd window: tile count stays bounded.
    let grid =
        plan_tiles(0.0, i32::MAX, 1000, 1, 1.0, 100.0, 0, i32::MAX / 256, 64.0).expect("grid");
    assert!(grid.i1 - grid.i0 < MAX_TILES);
}

#[test]
fn wave_tile_paints_mirrored_bars_on_transparent_ground() {
    let peaks = vec![1.0f32; 200]; // 2s of loud audio at 100/s
    let rgba = render_wave_tile(&peaks, 100.0, 0.0, 2.0, 64, 32);
    assert_eq!(rgba.len(), 64 * 32 * 4);

    let px = |x: usize, y: usize| {
        let i = (y * 64 + x) * 4;
        [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
    };
    // Bar column at center rows is painted; the gap column is not.
    assert_eq!(px(0, 16), WAVE_BAR_RGBA);
    assert_eq!(px(3, 16), [0, 0, 0, 0], "gap between bars stays clear");
    // Mirrored: symmetric rows above/below the midline both painted.
    assert_eq!(px(0, 4), px(0, 27));
    // Top margin rows stay clear (max_half leaves 2px headroom).
    assert_eq!(px(0, 0), [0, 0, 0, 0]);
}

#[test]
fn wave_tile_silence_keeps_a_center_line() {
    let peaks = vec![0.0f32; 100];
    let rgba = render_wave_tile(&peaks, 100.0, 0.0, 1.0, 16, 32);
    let center = (16 * 16) * 4;
    assert_eq!(&rgba[center..center + 4], &WAVE_BAR_RGBA);
    let top = 16 * 4_usize;
    assert_eq!(&rgba[top..top + 4], &[0, 0, 0, 0]);
}

#[test]
fn wave_tile_past_the_end_of_audio_does_not_panic() {
    // 115 peaks ≈ 1.15s of audio at 100/s, but the tile spans [1.0, 2.0)s:
    // later bars start past the last peak (a clip whose tick duration
    // outlives its decoded audio). Used to panic in clamp() with
    // `min > max. min = 119, max = 115`.
    let peaks = vec![0.5f32; 115];
    let rgba = render_wave_tile(&peaks, 100.0, 1.0, 1.0, 64, 32);
    assert_eq!(rgba.len(), 64 * 32 * 4);
    // Bars fully past the audio stay transparent.
    let last_bar_x = 60; // bar 15 of 16 covers [1.9375, 2.0)s
    let center = (16 * 64 + last_bar_x) * 4;
    assert_eq!(&rgba[center..center + 4], &[0, 0, 0, 0]);
}

#[test]
fn wave_tile_with_no_peaks_is_fully_transparent() {
    let rgba = render_wave_tile(&[], 100.0, 0.0, 1.0, 16, 16);
    assert!(rgba.iter().all(|&b| b == 0));
}

#[test]
fn scratch_clip_renders_fixture_frames() {
    // End-to-end decode spine over the checked-in fixture: probe →
    // scratch project → headless fit-render, exactly what the strip
    // worker does per tile. Skips when the fixture or a GPU is missing.
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../apps/cutlass-ios-macos/cutlass-ios-macos/Fixtures/demo1.mp4");
    if !fixture.exists() {
        return;
    }
    // The fixture is checked in, so existence alone isn't enough: platforms
    // with no native video decoder (e.g. Linux CI) can't open it, so skip
    // rather than unwrap an Unsupported error.
    let Ok(scratch) = scratch_clip(&fixture) else {
        return;
    };
    assert!(scratch.duration_frames > 0);
    let Ok(mut renderer) = Renderer::new_headless() else {
        return;
    };
    let image = renderer
        .render_frame_fit(
            &scratch.project,
            RationalTime::new(0, scratch.rate),
            FRAME_MAX_W,
            FRAME_MAX_H,
        )
        .expect("fit render");
    assert!(image.width > 0 && image.width <= FRAME_MAX_W);
    assert!(image.height > 0 && image.height <= FRAME_MAX_H);
    assert_eq!(
        image.pixels.len(),
        (image.width * image.height * 4) as usize
    );
}

#[test]
fn lru_eviction_keeps_recently_used_entries() {
    let mut map: HashMap<u32, CacheEntry> = HashMap::new();
    for i in 0..(CACHE_CAP as u32 + 100) {
        map.insert(
            i,
            CacheEntry {
                image: Image::default(),
                last_used: u64::from(i),
            },
        );
    }
    evict_lru(&mut map);
    assert!(map.len() <= CACHE_CAP);
    // The newest entries survive, the oldest are gone.
    assert!(map.contains_key(&(CACHE_CAP as u32 + 99)));
    assert!(!map.contains_key(&0));
}
