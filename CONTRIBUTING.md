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

- Read the [README](README.md) to understand what Cutlass is and how to run it.
- Check the [Cutlass v1 roadmap](docs/v1-roadmap.md) and the topic roadmaps in
  [`docs/`](docs/) to see what is planned, in progress, or already done.
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
pins `stable` via `rust-toolchain.toml`, edition 2024, MSRV 1.85) and FFmpeg
development libraries.

```bash
# macOS
brew install ffmpeg pkg-config

# Debian / Ubuntu
sudo apt-get install -y pkg-config clang \
  libavcodec-dev libavformat-dev libavutil-dev \
  libavfilter-dev libavdevice-dev libswscale-dev libswresample-dev
```

Build and test the whole workspace:

```bash
cargo build --workspace
cargo test --workspace
```

Run the desktop editor (optionally with a media file):

```bash
cargo run -p cutlass-ui
cargo run -p cutlass-ui -- path/to/video.mp4
```

`cutlass-app` is a headless smoke-test CLI that drives the engine end to end
(import, edit, preview, save, export) without the UI — handy for quick checks:

```bash
cargo run -p cutlass-app
```

## Project Layout

The workspace is split into focused crates, layered from data up to UI:

- `cutlass-models` — shared project/timeline data model, time math, and the
  `.cutlass` file schema. UI-, decode-, and render-agnostic.
- `cutlass-commands` — the structured command vocabulary every edit goes through.
- `cutlass-probe` — reads media metadata before import.
- `cutlass-decoder` — FFmpeg-backed video/image decode, thumbnails, waveforms,
  and audio playback.
- `cutlass-cache` — on-disk decoded-frame cache for preview.
- `cutlass-compositor` — WGPU GPU compositor that combines layers into RGBA frames.
- `cutlass-encoder` — FFmpeg encode/mux for exported MP4 and proxy media.
- `cutlass-engine` — the headless editing engine: applies commands, records
  undo/redo, produces preview frames, and exports timelines. Main integration
  point for non-UI callers.
- `cutlass-ai` — prompt-to-edit layer that turns model responses into validated
  commands applied through the normal engine path.
- `cutlass-ui` — the Slint desktop editor (timeline, preview, inspector,
  library, export dialog, AI panel).
- `cutlass-app` — end-to-end smoke-test CLI for the engine.

Each crate has its own `README.md` describing its responsibilities and what
does and does not belong there. Read the relevant one before adding code.

Two boundaries matter:

- **Keep the engine UI-agnostic.** Slint models, widgets, file dialogs, and
  interaction state belong in `cutlass-ui`, not `cutlass-engine`.
- **Keep `cutlass-models` pure data.** No file I/O, FFmpeg, GPU work, Slint
  bindings, LLM prompts, or undo/redo dispatch — those live in higher crates.

## Coding Conventions

- **Performance is a correctness concern** on interactive paths (timeline
  scrubbing, preview, per-frame and per-sample work, export). Know the
  asymptotic cost of hot loops, avoid accidental O(n²), and profile or
  benchmark before rewriting hot code. Prioritize clarity on cold paths. See
  [`.cursor/rules/perf.mdc`](.cursor/rules/perf.mdc).
- **Comments explain intent, not mechanics.** Skip comments that just narrate
  what the code does; document non-obvious trade-offs and constraints.
- **Icons in the Slint UI**: never block a feature on missing art. Ship a short
  text/char placeholder, register it in [`docs/icons.md`](docs/icons.md), and
  move on. See [`.cursor/rules/icons.mdc`](.cursor/rules/icons.mdc).
- **Do not commit ignored paths**: build artifacts (`/target`, `/dist`), media
  scratch dirs (`frames/`, `proxy/`, `assets/`), `.env`, or `.cutlass/`.

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

## Submitting a Pull Request

1. Fork the repo and create a topic branch from `master`.
2. Make your change as a clean series of commits (see above).
3. Run the same checks CI runs, and make sure they pass:

   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets -- -D warnings
   cargo build --workspace --all-targets
   cargo test --workspace
   ```

   CI enforces clippy with `-D warnings`, a workspace build, and the test
   suite. Running `cargo fmt` keeps formatting consistent.
4. Add or update tests. Engine and model changes should cover invariants and
   behavior; for undoable edits, test both the forward edit and its inverse.
   Run a single crate's tests with `cargo test -p <crate>` and benchmarks with
   `cargo bench -p cutlass-engine --bench preview` (or `--bench export`).
5. Update docs (crate `README.md`, `docs/`, the icon registry) when behavior or
   structure changes.
6. Open the PR with a clear description of what changed and why, and link any
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
