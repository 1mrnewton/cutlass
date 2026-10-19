# cutlass (Python)

MoviePy-style Python bindings for the [Cutlass](https://github.com/1mrnewton/cutlass)
video engine. A thin, Pythonic wrapper over the pure-Rust timeline, GPU
compositor, and platform-native exporter — build a project, pull frames as
NumPy arrays, or export straight to an `.mp4`.

The editor, AI agent, and engine stay pure Rust; this package only wraps them.

## Install (from source)

Requires the Rust toolchain and [maturin](https://www.maturin.rs/).

```bash
cd crates/cutlass-py
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release      # builds + installs `cutlass` into the venv
```

## Usage

```python
import cutlass

p = cutlass.Project("demo", fps=30)
p.set_canvas("16:9", background=(20, 20, 30))   # auto / 16:9 / 9:16 / 1:1 / 4:5 / 21:9

p.add_solid((38, 42, 64, 255), start=0.0, duration=2.0)
p.add_text("Cutlass", start=0.0, duration=2.0, size=220.0, color=(240, 240, 255, 255))

print(p)                      # Project(size=(1920, 1080), fps=30.000, duration=2.000s)
print(p.size, p.fps, p.duration)

frame = p.get_frame(0.5)      # numpy uint8 array, shape (height, width, 4), RGBA
n = p.export("out.mp4")       # native H.264/mp4 (Apple today); returns frame count

p.save("demo.cutlass")
p2 = cutlass.Project.load("demo.cutlass")
```

## API

`Project(name, fps=30)`

- `set_canvas(aspect, background=(r, g, b))` — aspect preset + opaque background.
- `add_solid(color, start=0.0, duration=1.0) -> clip_id` — solid `(r,g,b,a)` lane.
- `add_text(text, start=0.0, duration=1.0, size=96.0, color=(255,255,255,255)) -> clip_id`.
- `add_track(kind, name="") -> track_id` — low-level lane (`video`/`audio`/`text`/…).
- `split(clip_id, at) -> new_clip_id` — split a clip at time `at` (seconds).
- `load_font(path)` — register a TTF/OTF for deterministic text.
- `get_frame(t) -> numpy.ndarray` — composite the frame at `t` (seconds).
- `export(path) -> int` — encode the whole timeline; returns frames written.
- `save(path)` / `Project.load(path)` — Cutlass JSON project documents.
- Properties: `size` `(w, h)`, `fps`, `duration` (seconds).

Generated lanes stack in insertion order (later calls draw on top).

> Note: `export` uses the platform-native encoder. Apple (VideoToolbox H.264 →
> mp4) is implemented today; other platforms raise until their backends land.
