# Cutlass

Cutlass is a free, open-source video editor with an AI assistant built in.
Edit on the timeline the normal way, or describe the edit in plain language
and the assistant runs it as regular timeline commands you can review and
undo.

It's early alpha. The core editing loop works, but expect rough edges and a
project format that hasn't settled yet.

## What works today

**Timeline editing**

- Import video, audio, and images onto a multi-lane timeline.
- Cut, trim, split, move, duplicate, link/unlink, ripple-delete, multi-select.
- Speed changes (flat or ramped), reverse, crop, flip, move/scale/rotate
  (per-axis scale), opacity.
- Styled text, solid colors, shapes, and bundled stickers, static and animated.
- Entrance/exit/combo animations from a catalog.
- Keyframes on transforms and effect settings, with a graph editor and easing
  presets; motion paths support bezier tangents.
- Blend modes, layer styles (shadow / glow / outline / background), animatable
  crop, and per-clip transform motion blur.
- Canvas presets (16:9, 9:16, 1:1, 4:5, 21:9) and a background color.

**Effects and color**

- GPU effect passes: gaussian blur, vignette, pixelate, plus typed params
  (including duotone). The rest of the catalog (sharpen, glitch, grain, glow,
  and so on) is selectable but renders as a no-op until its shader lands.
- Per-clip mask geometry (linear, mirror, circle, rectangle, heart, star) and
  chroma key.
- Filter presets and expanded color adjust (11 sliders) per clip, plus
  lane-wide adjustment / effect / filter passes that grade everything beneath
  them.
- Transitions: crossfade and wipe-left are implemented; the other catalog
  entries currently play as a crossfade.

**Audio**

- Volume envelopes, stereo pan, and draggable fade handles.
- Speed changes resample the audio, ramps included. Pitch follows the rate
  for now; pitch-preserving stretch is planned but not built.
- Noise reduction per clip (RNNoise).

**Preview and export**

- Live GPU preview with scrubbing and playback.
- Export to H.264/AAC MP4.

**The AI assistant**

Describe an edit and the assistant applies it through the same commands the
UI uses, so its work shows up on the timeline like yours would and undoes in
one step. Dry-run preview is on by default: you see the plan before anything
changes. The assistant is optional and the editor works fine without it.

## Install

Download a build from the [releases page](https://github.com/1Mr-Newton/cutlass/releases).

- **macOS** (Apple Silicon): unzip and drag `Cutlass.app` to Applications.
  On first launch, right-click the app and choose **Open**; builds aren't
  notarized yet. Media decode/encode uses the system's AVFoundation, so
  there's nothing else to install.
- **Windows** (x64): unzip and run, or use the Setup.exe installer. Media
  decode/encode uses Media Foundation, so nothing else to install. Builds are
  unsigned for now; SmartScreen will warn on first run.
- **Linux**: preview builds only. The UI runs, but the Linux media backend
  isn't implemented yet, so imported media won't play.

## Setting up the AI assistant

Cutlass doesn't ship a model. Use **Local** (Ollama / LM Studio), **OpenRouter**
for cloud, or **Advanced** for any OpenAI-compatible endpoint. The Settings
dialog is the usual path; or create `~/.cutlass/config.toml`:

```toml
# Local (Ollama / LM Studio) — curated models only
[ai]
source = "local"
base_url = "http://localhost:11434/v1"
model = "qwen3:14b"

# Or OpenRouter cloud — one key, curated slugs
# [ai]
# source = "openrouter"
# model = "openai/gpt-5.6-sol"
# api_key = "sk-or-…"
# # api_key_env = "OPENROUTER_API_KEY"
```

The key stays in that file or your environment; it's never written into
project files. Small local models work but their tool calling is less
reliable, which is part of why dry-run is the default.

## Projects

Cutlass owns your projects, CapCut-style. There's no save button: every edit
auto-saves, and the launch screen lists your projects to reopen or delete.
Rename a project from the title bar.

**Open file…** imports an external `.cutlass` into your projects, and
**Export** renders an `.mp4`. Media is referenced from wherever it lives on
disk, so a project from another machine may ask you to relink its media.

## Build from source

You need a recent stable Rust toolchain. There are no third-party media
libraries to install; decode/encode is platform-native (AVFoundation and
VideoToolbox on Apple platforms, Media Foundation on Windows).

```bash
cargo run -p cutlass-desktop
# or open straight into a file:
cargo run -p cutlass-desktop -- path/to/video.mp4
```

Build and test everything:

```bash
cargo build --workspace
cargo test --workspace
```

The iOS/macOS SwiftUI app lives in `apps/cutlass-ios-macos` (built with Xcode
on the same engine through `cutlass-mobile`), and the Android test app in
`apps/cutlass-android`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, project layout, and the
commit/PR style. Each crate has its own README under `crates/`, and packaging
notes live in [packaging/](packaging/README.md).

A lot of Cutlass is written with AI coding tools and reviewed by maintainers.
Contributions are judged on what they do, not how they were made.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at
your option.
