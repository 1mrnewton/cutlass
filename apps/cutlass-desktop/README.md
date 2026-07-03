# cutlass-desktop

The Cutlass desktop editor — a native Rust + [Slint](https://slint.dev)
frontend. Unlike the mobile apps, it links `cutlass-engine` **directly** (no
C-ABI/JNI bridge): `Engine::apply` for edits, `Engine::get_frame` for preview.

```bash
cargo run -p cutlass-desktop
```

Slice 1 shows a scrubbable demo preview; real media, the timeline, and editing
tools land in later slices.
