# Playhead

The playhead is a white mark on the tick row and a 1px line through the tracks.

The mark is a rounded rectangle, 11px wide and as tall as a major tick. It sits in that tick band, centered on the line. The line starts at the bottom of the mark and runs to the bottom of the tracks. It does not enter the toolbar, the track head, or the cover column. Scrolling keeps it clipped to the ruler and the lanes.

The mark is the drag handle. Dragging it seeks. The drawn time is the current frame, the same frame the transport shows, so the line lands on a tick whenever that frame is a tick.

With no clips, the playhead is dimmed and cannot be dragged.

This follows the CapCut 9.6.0 head: a rounded mark on the ruler, with the line leaving the center of its bottom. CapCut draws that mark at the top of the ruler because its ticks start there. Here the ticks sit at the bottom of the ruler, so the mark sits on them.
