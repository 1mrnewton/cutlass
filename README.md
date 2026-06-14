# Cutlass

Cutlass is an open-source video editor built around a simple idea: describe the edit you want, then keep refining it on a real timeline.

It is early alpha software. The editor already supports practical timeline work, live preview, export, and an AI assistant panel, but the experience is still changing quickly.

## What Cutlass Does

Cutlass is aimed at fast everyday editing:

- Import videos, audio, and still images.
- Arrange clips on a multi-lane timeline.
- Trim, split, move, duplicate, link, unlink, and ripple-delete clips.
- Adjust speed, reverse playback, crop, flip, transform, fade, and volume.
- Add styled text, solid color clips, and simple shapes.
- Preview edits live with GPU rendering and audio playback.
- Export H.264/AAC MP4 files.
- Ask the AI assistant to make timeline edits from plain-language prompts.

The AI assistant does not replace the editor. It drives the same timeline actions you can perform by hand, so prompted edits remain visible, undoable, and reviewable.

## Current Status

Cutlass is under active development. It is useful for testing the core editing workflow, experimenting with prompt-to-edit interactions, and contributing to the project.

Expect rough edges. Alpha builds are not notarized on macOS, Windows builds are not published yet, and advanced editing features such as effects stacks, transitions, captions, masks, and full audio mixing are still evolving.

## Roadmap

Visit the [Cutlass v1 roadmap](docs/v1-roadmap.md) to see what is planned, what is in progress, and what has already landed.

## Install

Prebuilt alpha releases are published on [GitHub Releases](https://github.com/1Mr-Newton/cutlass/releases) when tagged.

For macOS Apple Silicon, download `Cutlass-*-macos-arm64.zip`, unzip it, and drag `Cutlass.app` to Applications. On first launch, right-click `Cutlass.app` and choose **Open** because alpha builds are not notarized.

For Linux x86_64, download `Cutlass-*-linux-x86_64.tar.gz`, extract it, and run `./cutlass-ui`. Install FFmpeg runtime libraries first if your distribution does not already provide them.

## Build From Source

You need a recent stable Rust toolchain and FFmpeg development libraries.

```bash
# macOS
brew install ffmpeg pkg-config

# Debian / Ubuntu
sudo apt-get install -y pkg-config clang \
  libavcodec-dev libavformat-dev libavutil-dev \
  libavfilter-dev libavdevice-dev libswscale-dev libswresample-dev
```

Build and test the workspace:

```bash
cargo build --workspace
cargo test --workspace
```

Run the desktop editor:

```bash
cargo run -p cutlass-ui
```

You can also start with a media file:

```bash
cargo run -p cutlass-ui -- path/to/video.mp4
```

## AI Assistant

The assistant works with OpenAI-compatible chat endpoints, including local servers such as Ollama and cloud providers that expose the same API shape.

Configure it in `~/.cutlass/config.toml`:

```toml
[ai]
base_url = "http://localhost:11434/v1"
model = "qwen3:14b"
# api_key_env = "OPENAI_API_KEY"
```

API keys should stay in your user config or environment. They are not stored in Cutlass project files.

## Project Files

Cutlass projects are saved as `.cutlass` files. Media is referenced from its location on disk, so moving a project between machines may require relinking missing media when it is opened.

## Developing

This repository is a Rust workspace. The root commands above build and test everything, while individual crates can be run or tested with Cargo's `-p` flag.

Each crate has its own README under `crates/` with the details for that part of the system. Packaging notes for maintainers are in [packaging/README.md](packaging/README.md).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, project layout, coding conventions, commit style, and how to open a pull request.

## License

Cutlass is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in Cutlass, as defined in the Apache-2.0 license, shall be dual licensed as above without additional terms or conditions.

## Third-Party Licenses

Cutlass builds on third-party components that keep their own licenses.

FFmpeg, used through [`ffmpeg-next`](https://crates.io/crates/ffmpeg-next), is licensed under LGPL-2.1-or-later by default and may fall under GPL depending on how the linked FFmpeg libraries were configured. If you distribute Cutlass builds that link FFmpeg, review the licensing terms of the FFmpeg build you ship.
