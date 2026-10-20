---
name: cutlass-prototype
description: Work on the Cutlass desktop editor prototype. Use when editing the Cutlass UI, layout, icons, timeline, or docs, or when the user mentions Cutlass, the editor, Hugeicons, or the desktop prototype.
---

# Cutlass prototype

This repository is the desktop picture of the editor. Read `docs/product.md` and `docs/desktop-layout.md` before changing it. The topic notes are `docs/header.md`, `docs/export.md`, `docs/library.md`, `docs/preview.md`, `docs/inspector.md`, `docs/timeline.md`, `docs/ruler.md`, `docs/playhead.md`, and `docs/time.md`.

## Layout

One column: head, then a row of library / preview / inspector, then the timeline. Details are in `docs/desktop-layout.md`.

## Icons

Use the free Hugeicons set only, through `Icon` in `src/editor/icons.tsx`. See `docs/icons.md`. If a needed glyph is not exported from `@hugeicons/core-free-icons`, pick another free icon. Do not add a Pro package.

## After a decision

Update the matching file in `docs/` before finishing. Add a line to `docs/decisions.md` when the choice is new.

## Leave out

Do not add a phone layout to this app. Do not treat this React UI as the shipping desktop app. That app will be Rust, eframe, and egui.
