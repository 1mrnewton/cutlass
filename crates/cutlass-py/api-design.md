# cutlass-py API design (v2)

Reference for the shipped v2 `cutlass` Python package: a track-first object
model that matches how the Rust engine works, with code samples for every
surface. Every call maps onto an existing `cutlass-models` / `cutlass-render` /
`cutlass-decoder` API (see the mapping table at the end).

## Motivation

Problems with the retired flat v1 API:

- **Tracks exist but do nothing.** `add_track()` returns an id that no other
  method accepts. `add_solid()` / `add_text()` silently create a *new* track
  per call, so two clips can never share a lane, and the stacking order is an
  accident of call order.
- **Raw integer ids.** `split(clip_id, at)` takes a bare `u64`. There is no
  way to ask a clip where it is, how long it is, or what track it's on.
- **No media.** The engine imports/probes video and audio, renders them, and
  mixes audio into exports — none of that is reachable from Python.
- **No editing depth.** Transform, crop, speed, volume/fades, keyframes,
  effects, and transitions are all in the model and renderer today, but not
  exposed.

## Design principles

1. **Objects, not ids.** `Media`, `Track`, `Clip`, and `Effect` are live
   handles bound to their `Project`. Methods live on the object they act on.
2. **Tracks are explicit.** You create a track, then add clips *to it*. A
   track holds many clips; same-track overlap is an error (as in the engine).
   Nothing auto-creates tracks.
3. **Content is separate from placement.** *What* (a `Media` slice, `Text`,
   `Solid`, a shape) is a value you construct freely; *where/when* is decided
   by `track.add(content, start=…)`, which returns the placed `Clip`. This
   mirrors `ClipSource` vs `TimeRange` in the Rust model, and is also why
   there is no bare `Clip(...)` constructor: a clip *is* a placement.
4. **Seconds everywhere.** Floats in the public API, exact rational ticks
   internally (existing behavior). Timeline time for placement (`add`,
   `move`, `split`, `trim`); clip-relative time for animation (`animate`).
5. **Properties for constants, `animate()` for keyframes.**
   `clip.opacity = 0.5` sets a constant; `clip.animate(opacity=1.0, at=0.4)`
   writes a keyframe.
6. **Errors are Python exceptions** with precise types (`OverlapError`,
   `TrackKindError`, …), raised eagerly by the same validation the Rust
   model already performs.

## The object model

```
Project
├── media pool          p.import_media(path) -> Media   (probed asset, no placement)
├── tracks (bottom→top) p.add_track(kind)    -> Track   (ordered stack, kind-typed)
│   └── clips           track.add(content)   -> Clip    (non-overlapping placements)
│       ├── transform / crop / speed / volume  (constants or keyframed)
│       ├── effects     clip.add_effect(id)  -> Effect
│       └── transition  clip.transition(id)            (junction with next clip)
└── render/export       p.get_frame(t), p.export(path), p.save(path)
```

## A full example

```python
import cutlass
from cutlass import Project, Text, Solid

p = Project("trailer", fps=30, canvas="16:9", background="#101018")

# -- media pool (probed, not yet on the timeline) -------------------------
beach = p.import_media("footage/beach.mp4")     # video (+ its audio)
drone = p.import_media("footage/drone.mp4")
music = p.import_media("audio/theme.mp3")       # audio-only

print(beach)          # Media(video 12.4s 1920x1080 @29.97 'beach.mp4')

# -- tracks: an explicit, ordered stack (later tracks draw on top) --------
main    = p.add_track("video",   name="Main")
titles  = p.add_track("text",    name="Titles")
score   = p.add_track("audio",   name="Music")

# -- multiple clips on one track ------------------------------------------
a = main.add(beach.subclip(3.0, 8.0), start=0.0)   # timeline 0..5
b = main.append(drone.subclip(10.0, 14.0))         # butts at 5..9
a.transition("crossfade", duration=0.8)            # at the a|b junction

# -- overlay + titles ------------------------------------------------------
stickers = p.add_track("sticker", name="Badge")
badge = stickers.add(Solid("#202840"), start=0.5, duration=3.0)
badge.scale = 0.25
badge.position = (0.35, -0.35)                     # toward top-right
badge.animate(opacity=[(0.0, 0.0), (0.4, 1.0)], easing="ease_out")

title = titles.add(
    Text("BIG WAVES", size=140, color="#ffffff", bold=True),
    start=1.0, duration=3.0,
)

# -- audio -----------------------------------------------------------------
bed = score.add(music.subclip(0.0, 9.0), start=0.0)
bed.volume = 0.6
bed.fade_out = 1.5

# -- inspect ----------------------------------------------------------------
for track in p.tracks:                       # bottom → top
    print(track.name, [c.start for c in track])

# -- render / export ---------------------------------------------------------
frame = p.get_frame(2.0)                     # numpy uint8, (H, W, 4) RGBA
p.export("trailer.mp4")                      # H.264/mp4 + mixed AAC audio
p.save("trailer.cutlass")
```

---

## `Project`

```python
Project(name, fps=30, canvas="auto", background=(0, 0, 0))
Project.load(path) -> Project
```

| Member | Description |
|---|---|
| `p.import_media(path) -> Media` | Probe a video, audio, or still-image file and register it in the pool. Stills get `kind == "image"` and a 5 s default duration. |
| `p.remove_media(media)` | Drop a pool entry; errors if any clip references it. |
| `p.media -> list[Media]` | The pool. |
| `p.add_track(kind, name="", index=None) -> Track` | New track. `kind` is `"video" \| "audio" \| "text" \| "sticker" \| "effect" \| "filter" \| "adjustment"`. `index=None` stacks on top; `0` is the bottom. |
| `p.tracks -> list[Track]` | Bottom → top. Later tracks composite on top. |
| `p.track(name) -> Track` | Lookup by name (errors if missing/ambiguous). |
| `p.canvas` | Aspect preset, get/set: `"auto"`, `"16:9"`, `"9:16"`, `"1:1"`, `"4:5"`, `"21:9"`. |
| `p.background` | Canvas color, get/set. |
| `p.get_frame(t) -> numpy.ndarray` | Composite frame at timeline `t`, `(H, W, 4)` RGBA uint8. |
| `p.export(path) -> int` | Encode the whole timeline (video + mixed audio; native encoder on Apple and Windows); returns frames written. |
| `p.save(path)` / `Project.load(path)` | Cutlass JSON project documents. |
| `p.load_font(path)` | Register a TTF/OTF for deterministic text. |
| `p.duration`, `p.size`, `p.fps` | Read-only, as today. |

## `Media` — a pool asset

A probed source file. Assets are *referenced* by clips; one asset can appear
on the timeline many times.

```python
m = p.import_media("beach.mp4")
m.path        # "/abs/path/beach.mp4"
m.kind        # "video" | "audio" | "image"
m.duration    # seconds
m.size        # (w, h)   — (0, 0) for audio-only
m.fps         # native frame rate
m.has_audio   # bool

m.subclip(3.0, 8.0)    # -> a trimmed reference (source window), not a copy
m[3.0:8.0]             # same thing, slice sugar
```

`subclip` produces a lightweight *content descriptor* — nothing happens until
a track places it. Out-of-range windows error at `add` time.

> **v1 render coverage:** `TextStyle` fields beyond font, size, fill color,
> and horizontal alignment (bold, italic, underline, stroke, background, shadow,
> vertical alignment, wrap toggle) are stored by the model but not yet rendered
> by the compositor.

## `Track`

```python
track = p.add_track("video", name="Main")
```

### Placing content

```python
track.add(content, start, duration=None) -> Clip
track.append(content, duration=None)     -> Clip   # at track.end (butt-joined)
```

- `content` is a `Media`, a `media.subclip(...)`, or a generator descriptor
  (`Text`, `Solid`, shapes — see below).
- `duration=None` means: full (remaining) source for media. Generated content
  has no intrinsic length, so `duration` is **required**.
- The track's kind must accept the content (`video`/`audio` ⇢ media, `text` ⇢
  `Text`, `sticker` ⇢ `Solid`/shapes) — else `TrackKindError`.
- Placing over an occupied span raises `OverlapError` (nothing is nudged
  implicitly; use `append`, or move/ripple ops to make room).

```python
v = p.add_track("video")
v.add(beach, start=0)                     # whole file
v.add(beach.subclip(10, 13), start=20)    # 3s window placed at t=20
v.add(beach, start=40, duration=2)        # shorthand for subclip(0, 2)
v.append(drone)                           # butts against the last clip

t = p.add_track("text")
t.add(Text("Hello"), start=1, duration=3)

s = p.add_track("sticker")
s.add(Solid("#202840"), start=0, duration=10)
```

### Introspection & flags

```python
track.clips            # list[Clip], ordered by start
track.clip_at(4.2)     # Clip | None
len(track)             # clip count
for clip in track: …   # iterates in start order
track.end              # content end in seconds (0.0 when empty)

track.name             # get/set
track.kind             # "video" | "audio" | …  (read-only)
track.enabled = False  # video: excluded from the composite
track.muted   = True   # audio: silenced
track.locked  = True   # editing lock (engine semantics)
track.remove()         # delete the track and its clips
```

## `Clip`

A placement on a track. All timing is in timeline seconds.

### Timing & structure

```python
clip.start, clip.end, clip.duration   # read-only floats
clip.track                            # Track handle
clip.media                            # Media | None (generated clips)
clip.source_start, clip.source_duration   # media clips: window into the source

right = clip.split(at=7.0)     # timeline position inside the clip -> right half
clip.trim(start=5.0, end=9.0)  # new placement; source window follows (speed-aware)
clip.move(start=2.0)           # reposition on its track
clip.move(start=2.0, track=v2) # move across tracks (kind-checked)
clip.delete()                  # remove, leaving a gap
clip.ripple_delete()           # remove and slide later clips left
```

Handles stay valid across edits (ids are stable); using a deleted clip's
handle raises `CutlassError`.

### Transform (visual clips)

Canvas-normalized, engine semantics: `position` is the anchor's offset from
canvas center (+x right, +y down), `scale` 1.0 aspect-fits the canvas,
`rotation` is degrees clockwise, `opacity` 0..1.

```python
clip.position = (0.25, -0.1)
clip.anchor   = (0.5, 0.5)
clip.scale    = 0.5
clip.rotation = 15.0
clip.opacity  = 0.85
```

Setting a property writes a constant (flattening any keyframes on it).
Reading returns the constant, or the value at the clip start if animated;
`clip.transform_at(t)` samples the animated transform at clip-relative `t`.

### Animation

Keyframes on any animatable property, in **clip-relative** seconds (a
keyframe belongs to the clip and survives moves). Two forms:

```python
# One keyframe; several properties may share it. `at` is required.
clip.animate(opacity=0.5, at=1.0)
clip.animate(scale=1.2, position=(0.0, 0.0), at=2.0)

# A whole curve per property: a list of (time, value[, easing]) pairs.
clip.animate(opacity=[(0.0, 0.0), (0.6, 0.8), (1.2, 1.0)])
clip.animate(
    scale=[(0.0, 1.0), (2.0, 1.3)],
    position=[(0.0, (0.0, 0.0)), (2.0, (0.25, -0.1))],
    easing="ease_in_out",          # default for pairs without their own
)

clip.remove_keyframe("opacity", at=0.5)
clip.clear_animation("opacity", "scale")   # flatten back to constants
```

The rule: **tuples are values, lists are curves.** A plain value (or an
`(x, y)` tuple for vector properties) is a single keyframe and needs `at=`;
a list of `(time, value)` / `(time, value, easing)` pairs is a whole curve
and takes no `at=`. `animate` is additive — it inserts (or replaces, at an
existing time) keyframes on the property's curve — so curves can be built
incrementally; `clear_animation` flattens back to a constant. Batch pairs
may be listed in any order (sorted internally); duplicate times within one
call raise `ValueError`.

`easing` names the curve of the segment *leaving* a keyframe (engine
semantics — the last keyframe's easing is inert): `"linear"`, `"ease_in"`,
`"ease_out"`, `"ease_in_out"`, or a CSS-style cubic bezier tuple
`(x1, y1, x2, y2)`. In batch form, call-level `easing=` is the default for
every pair without its own third element.

A parallel-lists spelling (`opacity=[0.0, 1.0], times=[0.0, 0.6]`) was
considered and rejected: two lists synced by position are easy to get
wrong, and pairs keep each keyframe's time and value together.

Animatable: `position`, `anchor`, `scale`, `rotation`, `opacity`, `volume`
(media clips), effect parameters (via `Effect.animate`), and shape
properties (below).

### Audio (media-backed clips)

```python
clip.volume   = 0.8     # flat gain, 0 mutes, up to engine max boost
clip.fade_in  = 0.5     # linear fade lengths in seconds
clip.fade_out = 1.5
clip.animate(volume=[(2.5, 1.0), (3.0, 0.2), (6.0, 0.2), (6.5, 1.0)])  # duck under a voiceover
```

A video clip on a video track carries its own audio into preview/export —
no separate audio lane needed (CapCut semantics, already how the mixer
works).

### Speed & crop

```python
clip.set_speed(2.0)                  # 2x faster; timeline duration re-derives
clip.set_speed(0.5, reverse=True)    # slow-mo, played backward
clip.speed, clip.reversed            # read-only

clip.crop(x=0.1, y=0.0, w=0.8, h=1.0, flip_h=False, flip_v=False)
```

### Effects

```python
fx = clip.add_effect("gaussian_blur", radius=8.0)   # -> Effect
fx["radius"] = 16.0                                 # constant param
fx.animate(radius=[(0.0, 16.0), (2.0, 0.0)])        # keyframed (same forms as clip.animate)
clip.effects                                        # list[Effect]
clip.remove_effect(fx)
```

Effect ids and parameter ranges come from the engine catalog
(`cutlass.effects()`): `gaussian_blur`, `vignette`, `sharpen`, `pixelate`,
`glitch`, `chromatic_aberration`, `grain`, `glow`, `zoom_blur`, `mirror`.

### Transitions

A transition lives at the junction between a clip and the next clip it
*abuts* on the same track (`track.append` gives you abutting clips for
free).

```python
a = main.add(beach.subclip(0, 5), start=0)
b = main.append(drone.subclip(0, 4))
a.transition("crossfade", duration=0.8)   # at a's right edge, into b
a.remove_transition()
```

Catalog (`cutlass.transitions()`): `crossfade`, `dip_to_black`,
`dip_to_white`, `wipe_left`, `wipe_right`, `wipe_up`, `wipe_down`, `slide`.
Editing that breaks the abutment prunes the transition (engine behavior).

### Text & shape editing

```python
title.text = "NEW TITLE"                      # content, keeps style
title.set_style(size=160, color="#ffcc00", italic=True)

box = s.add(cutlass.Rect(width=400, height=300, color="#ff0055",
                         corner_radius=24), start=0, duration=5)
box.set_style(color="#00ffaa")
box.animate(width=800.0, at=1.0)              # shape geometry is animatable
```

## Content descriptors

Plain values; construct them anywhere, place them with `track.add`.

```python
Text(content, font="", size=90.0, color="white", bold=False, italic=False,
     underline=False, case="normal", letter_spacing=0.0, line_spacing=1.2,
     align=("center", "middle"), wrap=True,
     stroke=None,        # TextStroke(color, width)
     background=None,    # TextBackground(color, radius)
     shadow=None)        # TextShadow(color, blur, distance)

Solid(color)

# Shapes (sticker tracks). Sizes in reference pixels @ 1080p canvas height.
Rect(width=200, height=200, color="white", corner_radius=0.0, stroke=None)
Ellipse(width=200, height=200, color="white", stroke=None)
Polygon(sides, width=200, height=200, color="white", corner_radius=0.0, stroke=None)
Star(points, inner_ratio=0.5, width=200, height=200, color="white", stroke=None)
Line(length=200, thickness=8, color="white")
Arrow(width=200, height=200, color="white")
Heart(width=200, height=200, color="white")
# stroke = ShapeStroke(color, width)
```

## Colors, easing, catalogs, errors

```python
# Colors: everywhere a color is accepted —
(255, 128, 0)          # RGB
(255, 128, 0, 200)     # RGBA
"#ff8000" / "#ff8000c8" / "white"   # hex or a small named set

# Easing: "linear" | "ease_in" | "ease_out" | "ease_in_out" | (x1, y1, x2, y2)

# Catalogs (from the engine, for discovery/validation):
cutlass.effects()      # [EffectSpec(id, label, params=[ParamSpec(name, default, min, max)])]
cutlass.transitions()  # [TransitionSpec(id, label)]

# Exceptions:
cutlass.CutlassError           # base (also raised on stale handles)
├── cutlass.OverlapError       # placement collides with a clip on the track
├── cutlass.TrackKindError     # content not allowed on that track kind
├── cutlass.MediaError         # probe/decode failures, missing files, bad source windows
└── cutlass.RenderError        # GPU / encode failures
# ValueError for bad values (negative durations, unknown ids, bad colors)
```

## How it maps to the Rust model

No new engine features are needed; the binding stays a thin wrapper.

| Python | Rust |
|---|---|
| `Project(...)` / `save` / `load` | `Project::new` / `save_to_file` / `load_from_file` |
| `import_media` | `cutlass_decoder::probe` → `MediaSource::new` → `add_media` (video/audio only in v1) |
| `add_track(kind, index=)` | `add_track` / `insert_track` |
| `track.add(media…)` | `Project::add_clip` |
| `track.add(Text/Solid/shape)` | `Project::add_generated` |
| `clip.split / trim / move` | `split_clip` / `trim_clip` / `move_clip` |
| `clip.delete / ripple_delete` | `Timeline::remove_clip` / `ripple_delete` |
| `clip.set_speed` | `set_clip_speed` |
| `clip.volume / fade_*` | `set_clip_audio` |
| `clip.crop` | `set_clip_crop` |
| transform properties | `set_transform` / `set_param_constant` |
| `clip.animate(...)` | `set_param_keyframe` (`ClipParam::{Position,Scale,…,Volume,Shape,Effect}`) |
| `add_effect` / `fx[...]` | `add_effect` / `set_effect_param` |
| `clip.transition` | `add_transition` + `set_transition_duration` |
| `title.text` / `set_style` | `set_generator` |
| `track.enabled/muted/locked` | `Track` fields |
| `get_frame` / `export` | `Renderer::render_frame` / `export_to_file` |
| `cutlass.effects()/transitions()` | `effect_catalog` / `transition_catalog` |

Binding architecture: `Project` owns the model + lazy renderer (as today).
`Media`/`Track`/`Clip`/`Effect` are `#[pyclass]` handles holding
`(Py<Project>, id)`; every method borrows the project for the duration of
the call, so handles never cache state and stay coherent across edits. Ids
are engine ids (stable, never reused), so staleness is detected by lookup
failure.

## Out of scope for v1 (deliberately)

- **Speed curves** (`set_clip_speed_curve`) — the normalized-ramp domain
  needs its own design; flat speed + reverse covers the common cases.
- **Markers, templates, linked audio detach** — engine features that can
  layer on later without changing this core.
- **MoviePy-style clip composition** (`concatenate`, `CompositeVideoClip`) —
  the track model covers these; sugar can come later if scripts want it.

## Open questions

1. **Descriptors vs per-kind methods.** This design uses
   `track.add(Text(...))`. The alternative is `track.add_text("Hi", ...)`
   with one method per content kind. Recommendation: descriptors — one
   placement signature, reusable/styleable values, and it mirrors the model;
   `add_text`-style sugar can be added on top later if wanted.
2. **`media.subclip(a, b)` + slicing vs `track.add(media, offset=, duration=)`.**
   Recommendation: `subclip` (MoviePy heritage, "what" carries its trim);
   `duration=` stays as a shorthand for `subclip(0, d)`.
3. **Required `duration` for generated content** vs a CapCut-style 3 s
   default. Recommendation: required — scripts should be explicit.
4. **Project-level convenience adds** (auto-pick/create a compatible track).
   Recommendation: omit in v1; it's the behavior that made the current API
   confusing.

## Implementation plan (small, reviewable commits)

1. **Core object model** — `Media`/`Track`/`Clip` handles, `import_media`
   (video/audio), `add_track`/`tracks`, `track.add/append` for media +
   `Text` + `Solid`, `split/trim/move/delete/ripple_delete`, introspection,
   exception types; rewrite README. Replaces the current flat API outright
   (the package is unpublished — no compatibility shim).
2. **Transform + animation** — transform properties, `animate` /
   `remove_keyframe` / `clear_animation`, easing.
3. **Audio, speed, crop** — volume/fade properties, volume envelopes,
   `set_speed`, `crop`.
4. **Effects + transitions** — `Effect` handle, catalogs.
5. **Shapes** — descriptors, `set_style`, shape animation.
6. **Polish** — `media[a:b]` slicing, `__repr__`s, `.pyi` type stubs for IDE
   completion, example scripts.

Each step lands with `maturin develop` + a Python smoke test and the crate's
`cargo check`/`cargo test`.
