# Cutlass

Cutlass is a free, open-source video editor with an AI assistant built in.
Edit the normal way on a timeline — cut, trim, arrange, add effects and audio —
or just tell the assistant what you want and watch it do the edit for you.

It's still alpha and moving fast, but it's a real editor now, not a toy.

## What you can do

**Edit a timeline**

- Import video, audio, and images onto a multi-lane timeline.
- Cut, trim, split, move, duplicate, link/unlink, ripple-delete, multi-select.
- Change speed and reverse, crop and flip, move/scale/rotate, set opacity.
- Add styled text, solid colors, and shapes.
- Pick a canvas shape (16:9, 9:16, 1:1, 4:5, 21:9) and a background color.
- Keyframe almost anything — animate transforms and effect settings over time.

**Effects & transitions**

- A GPU effect engine with a starter pack: blur, vignette, sharpen, pixelate,
  glitch, chromatic aberration, grain, glow, zoom-blur, mirror.
- Transitions between clips: crossfade, dip to black/white, wipes, slide, zoom,
  blur.
- Adjustment layers that affect everything beneath them.

**Audio that doesn't need a separate app**

- Volume envelopes and draggable fade handles.
- Change speed without the chipmunk effect — pitch stays put, ramps included.
- Auto-duck music under a voice track.
- One-click noise reduction.
- Beat detection you can snap your cuts to.

**Preview & export**

- Live GPU preview with scrubbing and real-time playback.
- Export to H.264/AAC MP4.

**The AI assistant**

Describe an edit in plain language and the assistant makes it on your timeline.
It uses the same actions you would, so every change stays visible, undoable, and
reviewable — nothing happens behind your back, and you can preview the plan
before it runs. It's optional; the editor works fine without it.

## Install

Download a build from the [releases page](https://github.com/1Mr-Newton/cutlass/releases).

- **Windows** (x64 / Arm64) — run the `Setup.exe` installer, or use the
  portable `.zip`.
- **macOS** (Apple Silicon) — unzip and drag `Cutlass.app` to Applications.
  On first launch, right-click the app and choose **Open** (builds aren't
  notarized yet).
- **Linux** (x86_64) — extract the `.tar.gz` and run `./cutlass-ui`.

## Setting up the AI assistant

Cutlass doesn't ship a model. Point it at any OpenAI-compatible endpoint — a
local one like [Ollama](https://ollama.com), or a cloud provider.

Create `~/.cutlass/config.toml`:

```toml
[ai]
base_url = "http://localhost:11434/v1"   # e.g. Ollama
model = "qwen3:14b"
# api_key = "sk-..."                      # for cloud endpoints, or:
# api_key_env = "OPENAI_API_KEY"          # read the key from an env var
```

Your key stays in that file or your environment — it's never written into
project files. Smaller local models work but tool-call less reliably; the
assistant's dry-run mode lets you preview a plan before it touches anything.

## Projects

Cutlass owns your projects, CapCut-style — there's no file to save by hand.
Every edit auto-saves continuously, so a clean exit never loses work, and the
launch screen lists your projects to reopen or delete. Rename a project inline
from the title bar.

Use **Open file…** to import an external `.cutlass` into your projects, and
**Export** to render an `.mp4`. Media is referenced from where it lives on
disk, so importing a project from another machine may ask you to relink media.

## Build from source

You need a recent stable Rust toolchain (edition 2024, MSRV 1.85).

```bash
cargo build --workspace
cargo test --workspace
```

The codecs live under `crates/codecs/`: Rust codecs in `software/`, and the
operating-system codec APIs in `native/`. Which codec each platform covers is
in [docs/codecs.md](docs/codecs.md).

## Roadmap

Codec coverage for decode and encode is in [docs/codecs.md](docs/codecs.md).

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the
project layout, and the commit/PR style.

A lot of Cutlass is written with AI coding tools and then reviewed by
maintainers. Contributions are welcome on the same footing whether they come
from a person or an assistant, as long as they're solid and meet the bar.

## License

Dual-licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT),
at your option.
