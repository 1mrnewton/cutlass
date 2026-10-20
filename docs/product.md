# Product

Cutlass is a free, open-source editor in the spirit of CapCut. AI edits the timeline: remove silences, insert a generated shot at a time, or assemble a film from a bin of footage.

This repository is the desktop prototype. It is a React picture of the editor, on the private `prototype` branch of `cutlass-rewrite`.

The shipping desktop app will be Rust, eframe, and egui. iOS and Android will be native shells. The phone UI will be designed in its own React app, not in this one.

The prototype shares one mock project across the head, the workspace, and the timeline. The timeline starts empty. Library cards are placeholders you drag onto it. Commands mutate that mock. Nothing here renders video. Export opens a dialog and does not write a file.
