# Ruler

The ruler uses the CapCut 9.6.0 step table at the project frame rate. It starts at 30 fps. The research note is `docs/capcut-desktop-re/ruler.md`.

Clip times stay in seconds. Pixels per second is CapCut's pixels per microsecond times 1,000,000.

Under an hour the label is `MM:SS`. At an hour it is `HH:MM:SS`. Inside a second, a frame step shows the frame offset, such as `15f`. Drop frame, on 29.97 and 59.94, replaces those labels with `HH:MM:SS;FF`. 29.97 skips 2 frame numbers a minute and 59.94 skips 4, except every tenth minute.

A major tick is twice a minor tick. The major tick is 12px, the minor tick is 6px, and both sit on the bottom of the ruler. The label starts 4px to the right of the tick and 2px above the major tick.

Zooming in walks the step table up to the two-frame step. The ruler fills the lanes at every zoom. Past the end of the cut, the ticks keep going.
