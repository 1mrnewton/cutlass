# Cutlass

**Cutlass** is an open-source video editor where you edit by describing what you want. Tell it to trim the intro, cut a section, or tighten a clip — and it does the work on your timeline.

It's built for everyday editing: cuts, trims, and the basics you actually use. Think of the speed and simplicity of apps like CapCut, with an assistant that understands plain language instead of making you dig through menus for every change.

Cutlass is still in early development. The sections below describe what runs **today** versus where it's headed, so there are no surprises.

## Status

This is an early-stage project. The headless editing core is real and tested, and a basic desktop UI now drives it; the natural-language agent is not built yet.

**Works today**

- A Rust workspace with a tested project/timeline model and a headless editing engine.
- Media decode via FFmpeg, with hardware-accelerated decode where available.
- A closed set of deterministic, undo/redo-able edit commands: add clip, add generated clip, split, trim, move, remove, and ripple-delete.
- Frame resolution through the engine: timeline frame → ordered layers → composited image.
- A WGPU compositor and an on-disk proxy/transcode cache to keep cold seeks fast.
- A desktop editor shell (`cutlass-ui`, built on [Slint](https://slint.dev/)): import a video, scrub and play back a live preview, drag clips, split/delete/ripple-delete, undo/redo, with background proxy progress.
- A small end-to-end CLI (`cutlass-app`) that decodes a clip and renders one timeline frame to a PNG — a smoke test for the whole pipeline.

**Not built yet (the goal)**

- The natural-language agent that turns a prompt into edit commands. The command layer it will drive already exists.
- GPU compositing via [wgpu](https://wgpu.rs/) for preview (export path next).

## Architecture

The codebase is a Cargo workspace split into focused crates:

| Crate | Responsibility |
| --- | --- |
| `cutlass-models` | Project, timeline, track, and clip data model with edit invariants. |
| `cutlass-decoder` | FFmpeg demux + decode, hardware acceleration, keyframe indexing, proxy encode. |
| `cutlass-compositor` | WGPU frame compositor (multi-layer alpha-over, RGBA readback). |
| `cutlass-engine` | Headless editing engine: edit commands + undo/redo, frame resolution, frame cache, proxy/media pool. |
| `cutlass-ui` | Slint desktop shell: preview, scrub/playback, timeline editing, undo/redo, proxy progress. |
| `cutlass-app` | End-to-end render CLI that exercises the full decode → resolve → composite pipeline. |

## Prerequisites

- A recent stable **Rust** toolchain (edition 2024; Rust 1.85 or newer).
- **FFmpeg** development libraries, required by the `ffmpeg-next` bindings.

Install FFmpeg:

```bash
# macOS (Homebrew)
brew install ffmpeg pkg-config

# Debian / Ubuntu
sudo apt-get install -y pkg-config clang \
  libavcodec-dev libavformat-dev libavutil-dev \
  libavfilter-dev libavdevice-dev libswscale-dev libswresample-dev
```

## Build & run

```bash
# Build everything
cargo build --workspace

# Run the tests
cargo test --workspace
```

### Desktop editor

The `cutlass-ui` shell opens a window where you can import a video, scrub and play a live preview, drag clips on the timeline, split/delete/ripple-delete, and undo/redo:

```bash
# Open the editor (use the Import button to add a video)
cargo run -p cutlass-ui

# …or open with a video already loaded
cargo run -p cutlass-ui -- path/to/video.mp4
```

### Render CLI

The `cutlass-app` CLI decodes a video, builds a one-clip project, composites a single timeline frame, and writes it to a PNG:

```bash
# cutlass-app <video> [frame_index] [output.png]
cargo run -p cutlass-app -- path/to/video.mp4 1000 frame.png
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

### Third-party dependencies

The MIT/Apache-2.0 dual license above covers Cutlass's own source. Cutlass builds on third-party components that are distributed under their **own** licenses, and those terms continue to apply to the parts they cover:

- **FFmpeg**, used via the [`ffmpeg-next`](https://crates.io/crates/ffmpeg-next) bindings, is licensed under the **LGPL-2.1-or-later** by default, and can fall under the **GPL** depending on how the FFmpeg libraries you link against were configured (e.g. with GPL-only components enabled). If you distribute builds that link FFmpeg, you are responsible for complying with its license — review the licensing terms of the specific FFmpeg build you ship. See the [FFmpeg legal page](https://www.ffmpeg.org/legal.html).
- The Rust crate dependencies (such as `ffmpeg-next`, `rustc-hash`, `thiserror`, `tracing`, `png`, and others) are each distributed under their own licenses (commonly MIT and/or Apache-2.0). Run `cargo tree` to see the full dependency graph, and consult each crate for its exact terms.

Cutlass does not bundle FFmpeg; it links against the FFmpeg development libraries you install separately (see [Prerequisites](#prerequisites)).
