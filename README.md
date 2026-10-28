# Cutlass

**English** · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass is a timeline video editor. Desktop, iOS, and Android share one draft and one session. A gesture on any shell becomes the same request. Playback and export see the same frame for the same draft and the same time.

Time in the document is a signed count of microseconds. A segment stores two ranges in that unit. The source range is the part of the media it uses. The target range is where it sits on the timeline.

This file is the reference for the translations.

## Status

The workspace is early.

| Piece | State |
| --- | --- |
| `cutlass-model` | The draft: tracks, segments, material pools, and microsecond time |
| `cutlass-session` | Place, move, and trim as one undo step. A failed request leaves the draft as it was. Undo and redo advance the revision |
| Other crates | Empty members. They reserve the engine, containers, codecs, playback, export, and the agent |
| `apps/desktop` | Prints `cutlass` and returns. The window comes later |
| `apps/ios`, `apps/android` | Arrive in this tree later |

Containers and codecs are native Rust, one crate per family or codec. On Apple, Windows, and Android the platform media frameworks are the fast path. Linux uses the Rust codecs and a software preview.

## Layout

```text
apps/desktop/          egui and eframe
crates/                one component per crate
crates/containers/     one container family per crate
crates/codecs/         one codec per crate
```

Crate names use the `cutlass-` prefix. The desktop binary links the crates directly. The phone shells link `cutlass-ffi` only.

## Build

Install stable Rust. `rust-toolchain.toml` selects the channel, Clippy, and rustfmt.

```sh
cargo test --workspace
cargo fmt --all -- --check
cargo lint
```

`cargo lint` runs Clippy on the workspace and denies warnings. Ubuntu CI runs these checks and `cargo deny`.

## License

Cutlass is available under either of these licenses, at your option:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

Copyright (c) 2026 Emmanuel Aidoo
