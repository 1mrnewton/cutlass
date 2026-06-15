# cutlass-ml

`cutlass-ml` is the local-first, provider-abstracted media inference layer for Cutlass. It is where model-backed capabilities live — transcription first, then matting and text-to-speech — each behind a trait so a local runtime and a cloud adapter are interchangeable.

The crate knows nothing about projects, timelines, or the compositor. Inference is sample/pixel-domain in and plain data out, mirroring the audio DSP seam in `cutlass-decoder`. The engine owns decode and the seconds → tick mapping; the worker owns the thread.

This crate is a workspace member but **not** a default member (like the planned `cutlass-py`), so the editor build stays lean. Heavy native backends (e.g. whisper.cpp) sit behind opt-in features.

## Responsibilities

- Define the inference traits that form the provider seam (`Transcribe` today).
- Provide the plain data types capabilities produce (`Transcript`, `Segment`, `Word`).
- Host concrete backends — local runtimes first, cloud adapters additive.
- Resolve and cache model weights under `~/.cutlass/models/`, fetched on demand with a checksum, never bundled.

## Main APIs

- `Transcribe`: speech audio (`&[f32]` mono) → word-timed `Transcript`.
- `Transcript`, `Segment`, `Word`: the transcription result types.
- `TranscribeOptions`: language hint and translate task.
- `TranscribeError`: distinct failure kinds (not configured / model unavailable / backend / cancelled).
- `StubTranscriber`: a deterministic backend for tests and headless development.
- `WhisperTranscriber` (feature `whisper`): local transcription via whisper.cpp.
- `ModelCache` / `ModelSpec` / `whisper_model`: on-demand, checksummed model weights and the whisper model registry.

## Features

- `whisper` (off by default): builds `WhisperTranscriber` over `whisper-rs`. This pulls a C/C++ + cmake build toolchain (and `whisper-rs-sys`, which needs Rust 1.88+), so it is opt-in and kept out of the default build and CI.

```bash
cargo build -p cutlass-ml --features whisper
```

## Architecture invariants

- **Provider-abstracted, local-first, never local-only.** No feature hard-codes a runtime.
- **Models are data, downloaded on demand** to `~/.cutlass/models/` with a checksum.
- **Off the UI thread.** Backends are blocking and run on a worker.

## Testing

```bash
cargo test -p cutlass-ml
```
