# Contributing to Cutlass

Thanks for your interest in Cutlass. It is early alpha software and there is
plenty of room to help: fixing bugs, filling in editing features, improving
docs, sharpening the AI assistant, or just reporting what breaks.

This guide explains how the repository is organized, how to build and test it,
and what we expect from a change before it lands.

## Human and AI Contributions Both Welcome

We judge a contribution by what it does, not by how it was made. Whether you
wrote it by hand, with an AI assistant, or it was largely AI-generated, we
accept it as long as it adds something useful — a bug fix, a new feature, a
performance win, tests, or docs — and meets the bar in this guide.

That bar is the same for everyone: the change has to build, pass the checks
below, be reasonably scoped and reviewable, and actually improve the project.
You are responsible for any code you submit, so understand it, test it, and be
ready to explain and revise it in review. Don't open low-effort or untested
PRs (AI-generated or otherwise) just to make noise — those waste reviewer time
and will be closed.

It's nice (and appreciated) to mention in the PR description when a change was
AI-generated or AI-assisted. It's not required and won't count against you — it
just gives reviewers helpful context.

## Before You Start

- Read the [README](README.md) to understand what Cutlass is.
- Search [open issues](https://github.com/1Mr-Newton/cutlass/issues) before
  filing a new one or starting work, so effort is not duplicated.
- For anything larger than a small fix, open an issue first to discuss the
  approach. It saves everyone a wasted PR.

## Ways to Contribute

- **Report bugs** using the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).
  Include your OS, how you built or installed Cutlass, exact steps to reproduce,
  what you expected, and what happened. Screenshots and a sample `.cutlass`
  project help a lot.
- **Request features** using the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).
  Describe the problem first, then the change you have in mind.
- **Send code** for bug fixes, editing features, performance work, tests, or
  docs. See the workflow below.

## Development Setup

Cutlass is a Rust workspace. You need a recent stable Rust toolchain (the repo
pins `stable` via `rust-toolchain.toml`, edition 2024, MSRV 1.85).

Build and test the whole workspace:

```bash
cargo build --workspace
cargo test --workspace
```

## Project Layout

- `crates/core` (`cutlass-core`) holds the types codecs share: timestamps,
  packets, frames, pixel formats.
- `crates/project` (`cutlass-project`) holds the project, the timeline, tracks,
  and segments. It does no I/O.
- `crates/codecs/software/` holds the Rust codecs, which run on any device.
- `crates/codecs/native/` holds one crate per operating-system codec API.
- `fixtures/` holds the small media files the tests decode, and
  `crates/testing/fixtures` loads them. `scripts/make-video-fixtures.py`
  regenerates the video ones.

Rust codecs, under `crates/codecs/software/`:

- `cutlass-h264` — H.264 decode and encode.
- `cutlass-hevc` — HEVC decode. Encode stays on the hardware backends.
- `cutlass-vp9`, `cutlass-vp8`, `cutlass-av1` — decode.
- `cutlass-mpeg4` — MPEG-4 Part 2 decode.
- `cutlass-mpeg2`, `cutlass-mjpeg` — decode.
- `cutlass-prores` — decode. Encode stays on Apple silicon.
- `cutlass-aac` — AAC decode and encode.
- `cutlass-mp3`, `cutlass-flac`, `cutlass-opus`, `cutlass-vorbis`, `cutlass-ac3` — decode.
- `cutlass-pcm` — PCM decode and encode (WAV and AIFF).
- `cutlass-alac` — decode. Encode stays on Apple.

Operating-system codecs, under `crates/codecs/native/`:

- `cutlass-videotoolbox` — video on macOS and iOS.
- `cutlass-audiotoolbox` — audio on macOS and iOS.
- `cutlass-mediacodec` — video and audio on Android.
- `cutlass-mediafoundation` — video and audio on Windows.
- `cutlass-vaapi` — video on Intel and AMD GPUs on Linux.
- `cutlass-nvcodec` — video on NVIDIA GPUs (NVDEC and NVENC).

One boundary matters: a crate in `software/` never links an operating-system
SDK. Anything that calls the OS lives in `native/`.

## Rules

Tools enforce the mechanical parts:

- The lints in the root `Cargo.toml` and `clippy.toml`. For example: no
  `unwrap()` outside tests, no `#[allow]` without a reason, and a `// SAFETY:`
  comment on every `unsafe` block.
- `deny.toml`, which allows only permissive licenses and bans the FFmpeg crates.
- CI, which runs clippy on Linux, macOS, iOS, Windows, and Android. It also
  builds the software codecs for `wasm32-unknown-unknown` to prove they use no
  OS APIs.

## Commits

Commit history is a deliverable. Optimize for small, real commits rather than
one giant change.

- **One logical change per commit.** Land a feature as a series (model/schema,
  then engine/actions, then UI wiring, then tests/docs) instead of a single
  mega-commit. Bug fixes and refactors are their own commits.
- **Every commit must build and pass its targeted tests.** No empty, padding,
  or fabricated commits.
- **Message style** matches existing history (`git log --oneline`): an
  imperative, descriptive subject ending with a period that says what shipped
  and why it matters, not which files changed. Feature milestones use the
  `Ship <milestone> <name>: <summary>.` shape. Use the body to explain intent,
  trade-offs, and deliberate gaps.
- Use an author identity that matches your GitHub account so your contributions
  are attributed correctly.
- Don't commit build output (`/target`, `/dist`), media scratch, `.env`, or
  large media files.

## Submitting a Pull Request

1. Fork the repo and create a topic branch from `master`.
2. Make your change as a clean series of commits (see above).
3. Run the same checks CI runs, and make sure they pass:

   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   cargo deny check
   ```

   Install `cargo-deny` once with `cargo install cargo-deny --locked`.
4. Add or update tests. Run a single crate's tests with `cargo test -p <crate>`.
5. Open the PR with a clear description of what changed and why, and link any
   related issue. Keep PRs focused; unrelated drive-by changes should land
   separately.

A maintainer will review and may ask for changes. Address feedback as new
commits rather than rewriting published history.

## Code of Conduct

Be respectful and constructive. Assume good faith, keep discussion focused on
the work, and help make this a welcoming project to contribute to.

## License

By contributing, you agree that your contributions are dual licensed under the
[Apache License 2.0](LICENSE-APACHE) and the [MIT license](LICENSE-MIT),
matching the rest of Cutlass, without additional terms or conditions.
