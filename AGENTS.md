# AGENTS.md

## Cursor Cloud specific instructions

Cutlass is a Rust workspace (video editor). Standard build/lint/test/run commands
live in `README.md` and `CONTRIBUTING.md`; the notes below only cover
non-obvious, durable caveats for this cloud environment.

### Environment

- The startup update script runs `cargo fetch` to warm dependencies. System
  libraries (FFmpeg dev libs, `libfontconfig1-dev`, `libasound2-dev`, Vulkan,
  and `libxkbcommon-x11-0`) are baked into the VM snapshot — the same set the CI
  workflow (`.github/workflows/ci.yml`) installs, plus `libxkbcommon-x11-0`
  which winit/Slint needs at runtime for the desktop app.
- There is **no real GPU**. The compositor/renderer use the Mesa `llvmpipe`
  software Vulkan rasterizer (`lvp_icd.json`), so `GpuContext::new_headless_blocking()`
  succeeds and all GPU-backed tests run (just slower). Export `XDG_RUNTIME_DIR`
  (e.g. `/tmp/xdg-runtime`) to silence a Vulkan/xdg warning; it is non-fatal.

### Runnable binaries

- `cargo run -p cutlass-cli [out.png]` — headless smoke demo: composites a
  solid + text layer on the GPU and writes a PNG. Best fully-headless E2E check.
- `cargo run -p cutlass-desktop` — the Slint GUI editor (a scrubbable preview
  demo). It opens a real winit window, so it needs a display (`DISPLAY=:1` is
  available here); it cannot run truly headless.
- The root `README.md` mentions `cargo run -p cutlass-ui` / `cutlass-app`; those
  crate names are stale on this branch — the GUI binary is `cutlass-desktop` and
  the smoke CLI is `cutlass-cli`.

### Testing caveat

- `cargo test --workspace` has **one expected failure on Linux**:
  `cutlass-mobile`'s `decodes_exported_clip_through_ffi` fails with
  `no native video encoder for this platform yet`. On this branch the
  `cutlass-encoder` backend is only implemented for Apple/Android, so MP4 export
  (and any test that round-trips through it) is unsupported on Linux/Windows.
  This is a known branch-state gap, not an environment problem. To get a fully
  green run, use `cargo test --workspace --exclude cutlass-mobile`.

### Mobile apps

- `apps/cutlass-android` (Android Studio) and `apps/cutlass-ios-macos` (Xcode)
  are **not testable** on this Linux VM. Only the shared `cutlass-mobile` Rust
  library and its host-side tests are reachable here.
