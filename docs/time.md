# Time

Sequence time is an integer number of microseconds. `docs/project.md` defines that clock. The ruler, the playhead, and the transport draw it at the project frame rate, which starts at 30 fps.

The transport and the preview label show `HH:MM:SS:FF`. Drop frame uses `HH:MM:SS;FF`. The frame is `floor(seconds × nominal fps)`. Nominal fps is 24 for 23.976, 30 for 29.97, and 60 for 59.94. The playhead is drawn on that same frame.

Drop frame is a project setting for 29.97 and 59.94. Choosing either rate turns it on. 29.97 skips frame numbers 0 and 1 at the start of each minute. 59.94 skips 0 through 3. Every tenth minute skips none. Other rates stay on non-drop timecode, and the ruler stays on clock labels.

Inspector clip times, the export duration, library card times, and the still toast stay `M:SS.t`.
