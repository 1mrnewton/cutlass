# cutlass (Python)

MoviePy-style Python bindings for the [Cutlass](https://github.com/1mrnewton/cutlass)
video engine. Build a project with explicit tracks, place clips, animate properties,
and export — all through a track-first object model over the pure-Rust timeline,
GPU compositor, and platform-native exporter.

See [api-design.md](./api-design.md) for the full v2 API reference.

## Install

```bash
pip install --pre cutlass-py   # imports as `cutlass`
```

Wheels ship for macOS (Apple Silicon) and Windows (x64), Python 3.9+. The
distribution is named `cutlass-py` (the bare `cutlass` name on PyPI belongs to
an unrelated project); the module you import is `cutlass`. `--pre` is needed
while releases carry a pre-release version (e.g. `0.5.3a0`).

## Install (from source)

Requires the Rust toolchain and [maturin](https://www.maturin.rs/).

```bash
cd crates/cutlass-py
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release      # builds + installs `cutlass` into the venv
```

## Quick start

```python
import cutlass
from cutlass import Project, Text, Solid

p = Project("demo", fps=30, canvas="16:9", background=(20, 20, 30))

bg = p.add_track("sticker", name="Background")
overlay = p.add_track("text", name="Titles")

bg.add(Solid((38, 42, 64, 255)), start=0.0, duration=2.0)
title = overlay.add(
    Text("Cutlass", size=220, color="#f0f0ff"),
    start=0.0,
    duration=2.0,
)
title.animate(opacity=[(0.0, 0.0), (0.5, 1.0)], easing="ease_out")

print(p)                      # Project(size=(1920, 1080), fps=30.000, duration=2.000s)
frame = p.get_frame(0.5)      # numpy uint8 array, shape (height, width, 4), RGBA
p.export("out.mp4")           # H.264/mp4 (+ AAC when audio clips exist)
p.save("demo.cutlass")
```

## Object model

```
Project
├── media pool          p.import_media(path) -> Media
├── tracks              p.add_track(kind) -> Track
│   └── clips           track.add(content, start=...) -> Clip
│       ├── transform / crop / speed / volume
│       ├── effects     clip.add_effect(id) -> Effect
│       └── transition  clip.transition(id, duration=...)
└── render              p.get_frame(t), p.export(path)
```

**Tracks are explicit.** Create a track, then add clips to it. Content descriptors
(`Text`, `Solid`, shapes, `media.subclip(...)`) describe *what*; `track.add` decides
*where* and *when*.

**Animation:** set constants with properties (`clip.opacity = 0.5`) or keyframes with
`clip.animate(opacity=1.0, at=0.4)` / batch curves
`clip.animate(opacity=[(0.0, 0.0), (0.4, 1.0)])`.

## Media import

`import_media` accepts probed **video and audio** files. Still images are deferred
(the renderer has no still decoder yet); PNG/JPEG/etc. raise `MediaError`.

```python
clip = p.import_media("footage.mp4")
main = p.add_track("video")
main.add(clip.subclip(3.0, 8.0), start=0.0)
main.add(clip[3:8], start=0.0)   # slicing sugar
```

## Track kinds

`video`, `audio`, `text`, `sticker`, `effect`, `filter`, `adjustment` — the kind
must match the clip content (`TrackKindError` on mismatch): media goes on
`video`/`audio`, `Text` on `text`, `Solid` and shapes on `sticker`.

## Errors

| Exception | When |
|-----------|------|
| `OverlapError` | Clips overlap on the same track |
| `TrackKindError` | Wrong track kind for content |
| `MediaError` | Import/probe failure, still images |
| `RenderError` | GPU frame or export failure |

## Development

```bash
maturin develop                     # debug build into the active venv
pip install pytest
python -m pytest tests/ -q          # full API suite (bootstraps its own mp4 via export)
python tests/smoke.py               # quick GPU frame smoke test
python examples/tour.py             # optional; needs sample media paths
```

Rust-side unit tests cover the pure conversion helpers. The test binary links
libpython, so on macOS point dyld at the Python framework:

```bash
DYLD_FRAMEWORK_PATH="$(python3 -c 'import sys; print(sys.base_prefix.rsplit("/Python3.framework", 1)[0])')" \
    cargo test --no-default-features
```

> **Export:** uses the platform-native encoder — Apple (VideoToolbox) and Windows
> (Media Foundation), both H.264 + AAC → mp4. Other platforms raise until their
> backends land.
