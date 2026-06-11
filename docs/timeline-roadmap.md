# Timeline Roadmap — CapCut-style, end to end

Policy: **we follow CapCut.** When a UX question comes up, the answer is
"what does CapCut desktop do?" — magnet snapping with guide lines, kind-segregated
lanes, auto-created/auto-removed tracks, a magnetic main track, ghost previews
that never lie.

This doc tracks the path from today's timeline to that target. Phases are
ordered so each ships something usable on its own.

## Architecture invariants (apply to every phase)

These patterns are already established — new timeline work should follow them
rather than invent parallel mechanisms:

- **The engine is the single source of truth.** UI gestures end in a
  `cutlass_commands::EditCommand` applied on the worker thread
  (`crates/cutlass-ui/src/preview_worker.rs`); the UI re-renders from the
  republished projection (`crates/cutlass-ui/src/projection.rs`). No Slint-side
  mutation of project state, ever.
- **One resolver per gesture, shared by preview and commit.** Placement logic
  lives in a Rust pure callback (`crates/cutlass-ui/src/snap.rs`, exposed via
  `ui/lib/drag-backend.slint`). The ghost, the guides, and the release commit
  all read the *same* resolution, so the preview is exactly what a release
  does. Trim, ripple, etc. get the same treatment.
- **Gesture state is recorded by the grabbed element, resolved by the panel.**
  `ClipView` only snapshots the press + cursor deltas into `TimelineViewState`;
  `TimelinePanel` owns resolution, visuals, and teardown
  (`ui/panels/timeline/timeline.slint`).
- **Lane list is stack top-first.** Top lane = front compositing layer
  (CapCut/Premiere convention). UI row `r` ↔ engine order index
  `track_count − 1 − r`; inserting so a lane appears at row `r` means engine
  index `(len − r).clamp(0, len)`.
- **Every command is undoable.** New `EditCommand`s need an inverse action
  (`crates/cutlass-engine/src/action/edit/`).
- **Perf:** drag-frame resolvers are hot paths — keep them allocation-light and
  O(total clips) or better; decode/thumbnail work never blocks the UI thread.

## Status legend

- [x] shipped
- [ ] not started / in progress

---

## Phase 0 — Foundation (done)

- [x] Engine command surface: `AddTrack` (with stack index), `AddClip`,
      `AddGenerated`, `SplitClip`, `TrimClip`, `MoveClip`, `RemoveClip`,
      `RemoveTrack`, `RippleDelete` — all with undo inverses.
- [x] Worker thread owns the engine; scrub frames coalesce, mutations never
      dropped; projection republished after every edit.
- [x] Lane list renders stack top-first; per-kind lane colors.

## Phase 1 — Media drops (done)

- [x] Drag library tile → timeline with window-level ghost (real duration width).
- [x] Drop on a video lane lands there; occupied spans slide right into the
      first gap that fits (`first_fit_start`).
- [x] Drop on empty space / foreign lane creates a video track at the drop row
      (`create_track`), named per kind (V1, V2, …).
- [x] Drop position snaps (clip edges, playhead, tick 0).

## Phase 2 — Clip move, snapping, guides (done)

- [x] Free x/y drag: floating copy follows the cursor, original dims in place.
- [x] Magnet snapping to clip edges on all lanes, the playhead, and tick 0;
      vertical guide line at the snap tick; toolbar **Snap** toggle
      (`TimelineStore.snap-enabled`).
- [x] Landing ghost shows the exact release position; conflicts and
      foreign-kind lanes resolve to a **new lane** with a horizontal insertion
      line at the hovered row (kinds never mix).
- [x] Cross-lane moves that empty their source lane remove it
      (CapCut deletes emptied overlay tracks).
- [x] No-op drags (click without move) commit nothing.

## Phase 3 — Trim (edge drag) ✅

The next gesture CapCut users reach for.

- [x] Trim handles on the clip's left/right edges (`ew-resize` cursor, ~6px
      hit zones capped at ⅓ of the clip width in `clip.slint`; bracket bars
      on selection/hover).
- [x] `resolve_clip_trim` pure callback in `snap.rs`: clamps to source media
      bounds (the projection computes per-edge `head/tail-room-ticks`,
      rate-converted conservatively so the engine can never reject a
      UI-offered extension), neighbor edges on the lane, tick 0, and a 1-tick
      minimum; magnets the dragged edge to the same snap candidates and
      reuses the vertical guide line (snaps the clamp rejects are dropped).
- [x] Live preview: opaque stretch rect over the dimmed original during the
      gesture; commit `EditCommand::TrimClip` on release (engine still
      validates source-out-of-bounds and overlap atomically).
- [x] CapCut detail: duration + signed-delta tooltip above the dragged edge.

## Phase 4 — Playhead, ruler, scrubbing ✅

- [x] CapCut-style ruler, rebuilt from scratch (`src/ruler.rs` +
      `ruler.slint`): compact `MM:SS` labels centered on their position
      (no tick line — the text is the marker), `Nf` frame labels between
      second boundaries at deep zoom, dot subdivisions instead of hash
      marks, pin-shaped playhead head. Adaptive ladder runs on integer
      frames against the *nominal* fps (frame steps must divide it, so
      second boundaries always stay labeled); marks are virtualized to
      the viewport and capped.
- [x] Click/drag on the ruler moves the playhead (replaced the temporary
      toolbar slider as the scrub control; playhead changes funnel
      through one `TimelinePanel` watcher into coalesced frame requests).
- [x] Scrubbing snaps the playhead to clip edges / tick 0 when the magnet
      is on (same resolver as clip drags, zero-width span).
- [x] Keyboard: ←/→ frame step, Home/End.
- [x] Toolbar zoom slider (log scale, anchored on the playhead / viewport
      center) so the adaptive ruler is reachable; Ctrl+scroll zoom stays
      in Phase 9.
- [x] Preview frame requests keep coalescing through the worker
      (`WorkerMsg::Frame`).

## Phase 5 — Selection ops & shortcuts ✅

Commands existed in the engine; this was UI wiring. All shortcuts accept
Ctrl and Cmd (macOS); they live in a window-level `FocusScope`
(`app.slint`) and route through `TimelineActions`, the same functions
the toolbar buttons call, so gating can never diverge. Timeline
interactions bump a refocus nonce so shortcuts reclaim the keyboard
after a text input had it.

- [x] Delete/Backspace → `RemoveClip` (+ auto-remove emptied lane, same
      helper as moves); toolbar **Delete** button.
- [x] Split at playhead: toolbar button + Ctrl/Cmd+B → `SplitClip`, gated
      on the playhead being strictly inside the selected clip.
- [x] Undo/redo: Ctrl/Cmd+Z / +Shift+Z → engine history; toolbar buttons
      driven by `can-undo`/`can-redo` republished with every projection.
- [x] Copy/paste/duplicate: Ctrl/Cmd+C/V/D. Copy snapshots the clip's
      *content* on the worker (survives deleting the original); paste
      lands at the playhead on the source lane, first-fit sliding right
      (same policy as drops; recreates the lane if it's gone); duplicate
      places the copy right after the original.
- [x] Esc clears selection (empty-lane click already did).

## Phase 6 — Compound undo (one gesture = one history entry) ✅

A new-lane move used to record up to three entries (`AddTrack` + `MoveClip`
+ `RemoveTrack`), and a delete that emptied its lane two (`RemoveClip` +
`RemoveTrack`); one Ctrl+Z now reverts the whole gesture.

- [x] Engine: history groups (`Engine::begin_group` / `commit_group`) collect
      every inverse a dispatched batch records into one compound entry; undo
      applies them in reverse order, and the entry oscillates like any single
      action. Empty groups record nothing; single-command groups collapse to
      a plain entry.
- [x] `Engine::rollback_group` aborts a failed gesture: the collected
      inverses are applied in reverse on the spot, restoring the pre-gesture
      state and leaving history untouched — including the redo stack, so a
      failed gesture is a complete no-op. (Replaces the worker's hand-rolled
      "remove the lane we just created" compensation; failed drops now clean
      up their lane too.)
- [x] Worker: new-lane moves, drops that create a lane, deletes that empty
      their lane, and pastes that recreate a lane each commit as one group;
      future ripple ops should use the same wrapper.

## Phase 7 — Main-track magnet (ripple) ✅

CapCut's signature behavior, behind its own toolbar **Magnet** toggle
(separate from Snap, as in CapCut; on by default). Engine stays mechanism,
the magnet policy lives UI/worker-side.

- [x] Main track designation: the **bottom video lane** (engine: first video
      track in stack order; resolver: last video row). Computed, not stored —
      it follows lane creation/removal automatically.
- [x] Engine: `ShiftClips { track, from, delta }` ripple primitive (shift
      every clip starting ≥ `from`; validated atomically, exact-set inverse)
      and `RippleInsert { track, media, source, at }` (shift right + place,
      atomic with a compound inverse — built on Phase 6's `CompoundAction`).
- [x] With the magnet on, the main lane stays gapless: library drops
      `RippleInsert`; cross-lane moves in open a hole (`ShiftClips` + 
      `MoveClip`); reorders park-close-open-land as one group; moves *off*
      and `RippleDelete`s close the gap behind them; paste/duplicate insert
      at the nearest clip boundary / right after the original. Every gesture
      is one history entry (Phase 6 groups), rollback on failure.
- [x] Enabling the magnet packs the main lane (leading gap included) as one
      undoable entry — CapCut's lane is gapless the moment the toggle is on.
      The worker mirrors the flag (`SetMainMagnet`) for the ops that have no
      drag resolution (delete/paste/duplicate/pack).
- [x] Drag UX on the main lane: insertion caret between clips (slot picked
      by the dragged left edge vs clip midpoints) instead of free
      positioning, for clip drags and library drags alike. Reorders commit
      in post-close space; releasing on the clip's own slot is a no-op.
- [x] Off state = freeform behavior, unchanged everywhere else.

Deliberate gap: **trims don't ripple yet.** CapCut ripple-trims the main
track (later clips follow the dragged edge); here a magnet-on trim can still
leave/eat a gap. Needs a resolver mode (no neighbor clamp) plus a
`TrimClip`+`ShiftClips` composition with order depending on grow vs shrink —
tracked as the first item of future ripple work.

## Phase 8 — Clip content rendering ✅

Perf-sensitive; everything decodes off the UI thread (`src/strips.rs` +
a dedicated `cutlass-strips` worker, newest request first) and lands in
UI-thread caches via a `StripBackend.generation` bump — the tile models
are pure callbacks that take it as an argument, so delivery re-evaluates
them automatically (same reactivity pattern as the ruler).

- [x] Video clips: filmstrip thumbnails. Tiles sit on a power-of-two grid
      of *media-time* seconds picked from the zoom (tile width ∈ [64, 128)
      px); powers of two nest, so zooming reuses every cached frame, and
      trims/moves slide the strip under the clip window (grid keys are
      media-anchored) instead of resampling. `cutlass_decoder::video_strip`
      decodes N targets in one demuxer pass, rolling forward between nearby
      targets and re-seeking past gaps; frames are decoded *to* the target
      pts so tiles inside one GOP differ. Resolvers are viewport-virtualized
      (clip-local 256px buckets from `ClipView`) and LRU-capped; misses
      render the lane-colored card until the frame lands.
- [x] Audio clips: waveform strips. A peak file (~100 peaks/s,
      `audio_peaks_per_second`) is computed once per media on first demand;
      tiles are rasterized per power-of-two zoom bucket on the worker
      (mirrored bars on a transparent ground over the lane color) and
      stretched ≤ 2× between buckets.
- [x] Text clips: `text-content` rendered inline (centered, elided;
      falls back to the lane label while empty).
- [x] Clip badges: name + duration (`3.4s` / `M:SS`, computed rate-exactly
      in the projection) on a thin top scrim, hidden on sliver-thin clips;
      the selection outline moved above the content tiles. Speed/volume
      markers join when those land in the model.

Deliberate gap: **no speed/volume badges yet** (no model fields). The drag
floating copy and trim stretch preview stay flat color — content in those
gestures is a polish item for later.

## Phase 9 — Drag & viewport polish

- [ ] Auto-scroll when dragging near the viewport edges (CapCut scrolls the
      timeline under the drag; applies to clip moves, trims, and library
      drops).
- [ ] Snap guides for library drags (the window-level ghost currently doesn't
      show the vertical guide the resolver already computes).
- [ ] Zoom-to-fit button + Ctrl+scroll zoom centered on the cursor (the
      toolbar slider with playhead/center anchoring landed in Phase 4).
- [ ] Timecode tooltip while dragging/trimming.
- [ ] Track headers: mute/lock/hide toggles (engine `Track.enabled` exists).

## Phase 10 — Multi-clip & linking

- [ ] Multi-select: shift-click, marquee; group move with one resolution
      (collision policy: reject or new-lane the whole set, CapCut-style).
- [ ] Linked video+audio from the same media (CapCut "linkage" toggle):
      import drops create linked pairs once audio tracks land; linked clips
      move/trim together.
- [ ] Compound clips (select N clips → one nested clip) — far future.

## Phase 11 — Transitions & effects on the timeline

- [ ] Transition drop targets at clip junctions (only when clips abut).
- [ ] Effect/filter/adjustment lanes already exist as kinds; drag-drop from a
      future effects panel follows the Phase 1/2 resolver pattern.

---

## Known gaps / tech debt

- `changed` callbacks defer one event-loop iteration — drop commits read
  final-position state, but keep this in mind for new gesture wiring.
- Slint tick model is `i32` (projection clamps engine `i64`); fine for
  realistic timelines, revisit if hour-scale 120fps projects appear.
- Selection is keyed `(track-id, clip-id)`; engine clip ids are globally
  unique, so this can simplify to clip id alone.
- Selection can go stale after undo/redo (the projection republish doesn't
  touch `TimelineStore`); stale ids resolve to "nothing selected"
  everywhere, but clearing selection on history steps would be cleaner.
- The clipboard lives on the worker thread (content snapshot, not a
  reference) — fine for clips, revisit when multi-select copy lands.
