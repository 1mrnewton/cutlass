# Desktop layout

The editor is a column of three regions. Each region has its own note.

1. **Head.** `docs/header.md`. Export is `docs/export.md`.
2. **Workspace.** A row of three regions:
   - Library: `docs/library.md`
   - Preview: `docs/preview.md`. Time is `docs/time.md`.
   - Inspector: `docs/inspector.md`
3. **Timeline.** `docs/timeline.md`. The ruler is `docs/ruler.md`. The playhead is `docs/playhead.md`. The workspace takes the remaining height.

A dark gap sits between the library, the preview, the inspector, and the timeline. That gap is the drag handle. Drag the gap above the timeline up to give it more height. It stops while the preview and the transport are still fully visible. Drag a side gap to widen the library or the inspector. The preview keeps whatever room is left.

Chrome is gray. Panels separate by that gap, not by rules. `#141414` is the gap. `#1c1c1c` is every panel: library, preview, inspector, and timeline. `#262626` is the project bar. Export is the cyan action. A selected clip gets a thin white edge. Editor text does not select. Fields still do.

Hovering a tool, a clip, a marker, a library card, or an icon button shows its name. A clip also shows its time span. The tip sits above the control.

Tailwind is optional and not installed. Layout is flex in `src/index.css`.
