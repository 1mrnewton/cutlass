# Timeline

The timeline sits at the bottom, with no title bar. It is a column: toolbar, ruler, then the tracks. A hairline splits the toolbar from the ruler. It is not a drag handle.

The toolbar holds undo, redo, select, split, delete, freeze, reverse, mirror, rotate, crop, transcribe, split scenes, marker, beat, snap, main-track magnet, linkage, reference lines, reference-line lock, zoom to fit, and zoom. Undo and redo step through edits to the mock timeline. Command-Z undoes. Command-Shift-Z, and Control-Y, redo. Split is the blade. Hovering a toolbar button shows its name. Delete removes the selection, on every clip. Freeze, reverse, mirror, rotate, crop, transcribe, and split scenes follow the selection. A video clip turns all seven on. Audio also turns on reverse and transcribe. Text, stickers, and compositions also turn on mirror and rotate. A composition also turns on crop. Freeze and split scenes stay on video. Those seven are a picture. They do not change the mock. The tracks are a row. The head is 130px wide. A 72px column sits between the head and the lanes, reserved for the cover. The lanes take the rest, and each lane stays that width at every zoom.

Each lane is a CapCut track kind, named and colored as that kind. Main is always there. It sits in the vertical center. Other lanes spread up and down from it, with the same gap between each. An empty Main lane says to drag a clip there. From the top, the bands are plugin effect, then text and stickers together, effect, filter, adjust, overlay video, the main lane, then audio. Only audio can sit below the main lane.

The timeline starts empty. Library cards are placeholders. Drag one onto a lane to place it. The plus on a library card adds that clip at the end of the matching lane. A drop lands on the lane under the pointer when that time is free, including Main. Empty space, or another kind, opens a new lane in that kind's band. An occupied time is refused. With the magnet on, a drop on Main packs into that lane. A drop off Main stays where it is placed. The magnet can be turned off. A locked main lane refuses the magnet.

A dragged clip follows the pointer until it is released. A dim copy stays in the old place. Hovering a clip shows its name and time span. Shift-click adds or removes a clip from the selection. A click on empty space clears the selection. Dragging across empty space draws a box and selects every clip it touches. Shift keeps the clips already selected. Near either side of the sheet, the drag scrolls the timeline. Delete removes the selection. The Main lane stays.

Snap pulls a move or a trim to the playhead, another clip, a marker, a beat, a transition cut, a reference line, or either end of the timeline. A vertical line shows that snap while the pointer is down. Drag down from the ruler to leave a vertical reference line. Drag down from the corner above the track names to leave a horizontal one. Those lines stay. The toolbar hides or locks them, and Delete removes the selected one.

Linkage is its own switch. A clip that sits on only one main clip follows that clip when it moves, trims, or is deleted. Dragging the edge of a selected clip trims the media in or out. Split mode cuts a clip at the pointer, refuses a cut too close to an edge, and refuses a cut inside a transition. A transition dropped on a main-track cut sits on that cut. It does not get a lane. Marker mode drops a colored mark on the ruler or on a clip. Beat mode drops a mark on audio.

Zoom is pixels per second. Zooming in doubles the scale, up to the two-frame step. Zooming out halves it, until the cut fills a quarter of the lanes. That limit moves with the length of the cut. An empty timeline opens with its length across a third of the lanes. The first clip, and zoom to fit, put the cut across three quarters of the lanes. After the first clip, adding another keeps the current scale. A trackpad pinch zooms, and the time under the pointer stays put.

With no clips, split, marker, beat, play, the still, zoom, and the playhead are off. Each track label can lock that lane. A locked lane refuses edits. Empty lanes other than Main are removed. Right-click a clip to open its menu. A video clip lists the picture actions. Other kinds list copy, cut, delete, and a few actions for that kind. The prototype does not write a file.

A white pill drops a short way from the toolbar when a lane is locked or unlocked, a still is marked, or an edit is refused. It fades out where it lands.

The ruler is `docs/ruler.md`. The playhead is `docs/playhead.md`. Time is `docs/time.md`.
