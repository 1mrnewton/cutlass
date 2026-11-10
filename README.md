<p align="center">
  <img src="assets/app-icon.PNG" alt="Cutlass" width="112">
</p>

<h1 align="center">Cutlass</h1>

<p align="center">
  <b>A free, open-source video editor with an assistant you can talk to.</b><br>
  For macOS, Windows and Linux.
</p>

<p align="center">
  <a href="https://github.com/1mrnewton/cutlass/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/1mrnewton/cutlass?style=flat-square&label=release"></a>
  <a href="https://github.com/1mrnewton/cutlass/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/1mrnewton/cutlass/ci.yml?branch=main&style=flat-square&label=build"></a>
  <a href="#license"><img alt="License: MIT or Apache-2.0" src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="#download">Download</a> ·
  <a href="#what-you-can-do">Features</a> ·
  <a href="#the-assistant">Assistant</a> ·
  <a href="#formats">Formats</a> ·
  <a href="#build-from-source">Build</a> ·
  <a href="#not-here-yet">Not here yet</a> ·
  <a href="https://cutlass.sh">Website</a>
</p>

**English** · [简体中文](docs/readme/README.zh-CN.md) · [繁體中文](docs/readme/README.zh-TW.md) · [日本語](docs/readme/README.ja.md) · [한국어](docs/readme/README.ko.md) · [Español](docs/readme/README.es.md) · [Português](docs/readme/README.pt-BR.md) · [Français](docs/readme/README.fr.md) · [Deutsch](docs/readme/README.de.md) · [Italiano](docs/readme/README.it.md) · [Русский](docs/readme/README.ru.md) · [العربية](docs/readme/README.ar.md) · [हिन्दी](docs/readme/README.hi.md) · [Bahasa Indonesia](docs/readme/README.id.md)

Cutlass is a timeline video editor. Bring in footage, sound and pictures, cut them on tracks, add text, effects and color, and export a finished video. You can do every step by hand, or describe the edit in plain language and let the assistant make it on the timeline, where you can change it or undo it.

It is free. There is no account, no watermark and no upload. Your footage stays on your disk, and the tools that use machine learning run on your own computer.

Under the editor, the codecs are our own. H.264, HEVC, AV1, ProRes, AAC, Opus and the rest are written in Rust in this repository. Cutlass does not bundle FFmpeg.

## Download

| System | File |
| --- | --- |
| macOS, Apple silicon | [Cutlass.dmg](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass.dmg) |
| Windows, 64-bit | [Cutlass-setup.exe](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-setup.exe) |
| Linux, x86_64 | [Cutlass-x86_64.AppImage](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-x86_64.AppImage) |

Every file has a `.sha256` checksum beside it on the [releases](https://github.com/1mrnewton/cutlass/releases) page. Once installed, Cutlass checks for a newer release itself: **Check for updates…** in the menu.

The builds are not signed with a developer certificate yet, so the first launch takes one more step:

- **macOS:** open the app once, then allow it under System Settings › Privacy & Security › **Open Anyway**.
- **Windows:** if SmartScreen stops the installer, choose **More info** › **Run anyway**.
- **Linux:** make the file executable with `chmod +x Cutlass-x86_64.AppImage`, then run it.

## What you can do

**Cut on a timeline.** Video, audio and text tracks. Split, trim, close gaps, freeze a frame, reverse, mirror, rotate and crop. Group clips, or fold them into a compound clip and open it later. Replace a clip and keep its edits, or copy one clip's attributes onto another. Heavy footage can be cut through a proxy.

**Speed and motion.** Set a constant speed, or pick a speed curve and reshape it by dragging its points. Animate a clip with keyframes, and move or delete several keyframes at once. In, out and combo animations are ready to apply.

**Color.** Basic adjustments, HSL, curves and color wheels. Load a `.cube` LUT and keep skin tones natural under it. 200 filters, and adjustment layers on the timeline.

**Effects and transitions.** 200 effects, 180 transitions and 13 blend modes. Mask a clip with a line, a mirror, a circle, a rectangle, a heart or a star. Key out a green screen, or click an object to cut it out.

**Text, captions and stickers.** Text with fonts, stroke, shadow, background, bubbles and effects, animated in, out or in a loop. Captions are heard from the clip on your machine (English for now), then edited as a list: retype a line, find and replace, pick a template, style single words, and export the result as an SRT file. Stickers and shapes sit on the timeline like any other clip.

**Picture tools that run on your computer.** Stabilize, enhance quality, reduce image noise, optical flow for smooth slow motion, relight, auto reframe and motion blur. Nothing is sent anywhere.

**Sound.** Volume and fades, extract the audio from a video, sync video and audio by their sound, and mark the beat.

**Export.** MP4 or MOV, the whole timeline, a range, or the selected clips. Sound on its own, a GIF, or an animated WebP. On macOS and Windows the system's hardware codecs do the work when they can, and a switch in Settings turns them off.

## The assistant

Say what you want, and the assistant edits the open project with the same requests the timeline sends. Each change it makes is one undo step.

- **It runs on your machine.** The assistant talks to a model runner on your own computer, such as [Ollama](https://ollama.com), the llama.cpp server or LM Studio. Set the address and the model in Settings › Assistant. The prompt does not leave your computer on this path.
- **It asks first.** Anything that would send data off the machine waits for you to allow it.
- **It has skills.** Captions, removing pauses, motion graphics, a product demo and project setup ship with the app, and it can call tools from the MCP servers you add.
- **Other agents can drive Cutlass too.** `cutlass mcp /path/to/project` serves the same tools over stdio with no window, so Claude or any MCP client can read and edit a project.

## Formats

The decoders and encoders below are Rust code in [`crates/codecs`](crates/codecs) and [`crates/containers`](crates/containers), tested against FFmpeg as an outside reference. Most crates keep a record of what they support beside their source.

| | Open | Export |
| --- | :---: | :---: |
| **Files** | MP4, MOV, M4V, M4A, WAV, AIFF, FLAC, Ogg, MP3 | MP4, MOV, and sound on its own |
| **Video** | H.264, HEVC, AV1, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG | H.264, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG. HEVC through the system encoder on macOS and Windows |
| **Sound** | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM |
| **Pictures** | JPEG, PNG, WebP, GIF, animated or still | GIF, animated WebP |

## Build from source

You need the stable [Rust](https://rustup.rs) toolchain and a C compiler. On Linux, also the GTK and ALSA headers:

```sh
sudo apt-get install libgtk-3-dev libasound2-dev
```

Then:

```sh
git clone https://github.com/1mrnewton/cutlass
cd cutlass
cargo run --release -p cutlass
```

The checks CI runs (the last one needs [cargo-nextest](https://nexte.st)):

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
```

## Not here yet

Cutlass is young. In the app, a feature that is drawn but not built is dimmed and marked **Later**, so you always know what works. The larger gaps today:

- **More containers.** MKV, WebM, AVI and MPEG-TS files do not open yet.
- **Voice tools.** Noise reduction, voice enhance, loudness normalization, voice isolation and text to speech.
- **Generative tools.** AI remove, expand and stylize.
- **Tracking.** Text, stickers and masks that follow a subject.
- **Captions in more languages.**
- **A hosted model for the assistant.** It uses a local model today.
- **Phones.** The iOS and Android apps are in [`apps`](apps) and share the same Rust core, but the desktop comes first.

A bug report with the file that caused it is the most useful thing you can send. [Open an issue](https://github.com/1mrnewton/cutlass/issues).

## How the code is laid out

| Folder | What is in it |
| --- | --- |
| [`apps/desktop`](apps/desktop) | The desktop editor, an egui window. It builds the `cutlass` binary |
| [`apps/ios`](apps/ios), [`apps/android`](apps/android) | The phone apps, in SwiftUI and Kotlin Compose. They reach the Rust core through `crates/ffi` |
| [`crates/model`](crates/model), [`crates/session`](crates/session), [`crates/editor`](crates/editor) | The project, the edit requests with undo, and editor behavior shared by every app |
| [`crates/eval`](crates/eval), [`crates/engine`](crates/engine), [`crates/render`](crates/render), [`crates/playback`](crates/playback), [`crates/export`](crates/export) | Turning a project and a time into a frame and its sound, for preview and for export |
| [`crates/codecs`](crates/codecs), [`crates/containers`](crates/containers) | One crate per codec and per file format |
| [`crates/platform-apple`](crates/platform-apple), [`crates/platform-windows`](crates/platform-windows) | The system's hardware codecs |
| [`crates/effects`](crates/effects), [`crates/text`](crates/text), [`crates/captions`](crates/captions) | Filters, effects, transitions, masks, color, text and captions |
| [`crates/vision`](crates/vision), [`crates/transcription`](crates/transcription) | The on-device models for pictures and for speech |
| [`crates/agent`](crates/agent) | The assistant: its tools, skills and MCP |

## Contributing

Use Cutlass on real footage and tell us where it breaks. The [issue forms](https://github.com/1mrnewton/cutlass/issues/new/choose) cover bugs, features, performance, and files that will not open.

Sending code? Run the three checks above first. Shared crates stay free of any one platform, so the same edit works on the desktop and on phones.

## License

Cutlass is free under either license. Pick one.

* [Apache License 2.0](LICENSE-APACHE)
* [MIT License](LICENSE-MIT)

Copyright (c) 2026 1mrnewton
