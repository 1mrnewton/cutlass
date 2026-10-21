# CapCut desktop

Notes from the installed CapCut desktop app, version 9.6.0 (`com.lemon.lvoverseas`). The prototype timeline uses this list.

`none` is not a lane. A track is drawn only when it holds a clip. A new track of a kind sits above the other tracks of that kind. One video track is the main layer. It is the last picture lane. Only audio sits below it. Overlay video sits just above it. The prototype opens with only the main lane. Other lanes appear when a clip of that kind is dropped. From the top, those bands are plugin effect, text and stickers, effect, filter, adjust, overlay video, main video, then audio. Text and stickers share one band, so either can sit above the other. A plugin effect sits above that band. Templates are presets on a text track or inside a composition. Transitions are not a track kind.

The track-kind list comes from `LVVETrackType` in `Contents/Frameworks/libvideoeditor.dylib`. The function is `lvve::DraftEnumTransfer::TransferLVVETrackTypeToString`. Values above 13 become `none`.

| Value | String |
| --- | --- |
| 0 | `none` |
| 1 | `video` |
| 2 | `audio` |
| 3 | `sticker` |
| 4 | `text` |
| 5 | `effect` |
| 6 | `filter` |
| 7 | `adjust` |
| 8 | `text_to_video_text` |
| 9 | `text_to_video_text_audio` |
| 10 | `composition` |
| 11 | `plugin_effect` |
| 12 | `adcube` |
| 13 | `mixed` |

Adjust is its own track kind. There is no `template` value. Text templates, caption templates, and cover templates are presets that land on a text track or inside a composition.

The desktop UI groups clips more loosely than the enum. The linkage settings name Adjust, Effects, Filters, Audio, Sound effects, Stickers, Text, Text to speech, and Overlay. Overlay is a video track above the main video. Sound effects and text to speech are audio. Captions are text.

Meta type names such as `Meta_video`, `Meta_subtitle`, and `Meta_texttemplate` describe a segment’s material. They are not track kinds.

Editing, from `TimelineEditOperator` and `convert_metatype_to_tracktype` in the same library:

- A track holds one kind. Dragging a clip onto a different kind inserts a new track in that kind's band, not under the pointer. The main lane stays the bottom picture lane. Only audio can be inserted below it.
- Clips on a track cannot overlap. A conflict inserts another track of that kind.
- Empty tracks are removed. The main track stays.
- The main track is magnetic. Its clips pack from the start, with no gaps. The lane stays when its last clip is removed.
- Video, photo, gif, image, and combination land on video. Text, subtitles, lyrics, and text templates land on text. A sticker lands on sticker. An adjustment lands on adjust.
- `music`, `sound`, and `record` are audio materials. `filter` is a filter material. `video_effect` is an effect material. A transition is not a track.

Moves come in two forms. `rippleMoveSegments` is the main-track magnet: later clips close up. `multiTrackModeMoveSegments` is a free move on the other lanes. A move carries `need_insert_target_track` when the clip cannot stay on the lane under the pointer. Trim is separate from a move. Dragging an edge sets `clip_trim_in` or `clip_trim_out`. Split uses `splitSegment`. It fails too close to an edge, and a transition cannot be split.

Linkage is its own switch. Clips that sit over a main-track clip can follow it when that clip moves, trims, changes speed, or is removed. The linked kinds are the ones in the linkage settings: overlay, text, stickers, effects, filters, adjustments, audio, sound effects, and text to speech. The timeline also zooms (`Timeline zoom in` / `Timeline zoom out`) and holds beat marks and colored markers on clips.

Timeline zoom is a pixel size. Zoom in doubles it and zoom out halves it. The smallest size is a quarter of the lane divided by the length of the cut, so the floor moves as the cut changes. The largest size is the two-frame step. An empty timeline uses a third of the lane. The first clip, and zoom to fit, use three quarters of the lane. A later change in length updates that floor and leaves the current pixel size where it is.

The prototype applies the editing model above. Magnet and linkage are toolbar switches. A clip edge trims. Split refuses an edge and a transition. Zoom follows that pixel size. Marker and beat are tools. Speed stays at 1×.

## The rest of the timeline

REA opened `libvideoeditor.dylib` (arm64, sha256 `e2cdfb6e28aa17104b3cc210dbf6836e86fa42f212e5df653653625e16b70a30`) and inspected the Mach-O: 33,013 exports, four segments. No decompiler is installed, so this section is the exported C++ API, the embedded `qrc:/TimeLine/` QML, Qt method names in `libVECreator.dylib`, and the English UI strings. Snap distances in pixels are still unknown. The static track-order arrays `LVVETrackTypesAll`, `LVVETrackTypesAboveMain`, `LVVETrackTypesAboveMainVideo`, `LVVETrackTypesAboveMainEffect`, and `LVVETrackTypesUnderMain` live in zero-filled data and are filled at startup.

A project can hold more than one timeline. The engine can add, open, focus, close, name, and release one, and one of them is the main timeline. Mouse edits hit the focused timeline. A nested draft is its own timeline inside a segment: it can stay nested or be expanded into top-level tracks. Time in the draft CLI is microseconds. The mouse editor uses the same `long long` ranges.

A track has a kind and a flag. The flag is what marks the main video track. `find_target_track_index` picks a lane from the kind, the flag, a time range, a preferred index, and the ids to ignore. `get_insert_track` and `add_empty_track_if_needed` create the lane. `need_insert_target_track` is set when the lane under the pointer cannot keep the segment. Empty lanes are removed. The main lane is recreated if it is missing. It can hold no clips. A locked main lane refuses both turning the magnet on and turning it off.

A segment has a timeline range and a source range. `clip_segment` changes one edge and takes a clip type. Speed converts a trim delta into a timeline delta through `mapTrimDeltaToSeqDelta`, and the reverse through `mapSeqDeltaToTrimDelta`. Keyframes are stored as an offset on the segment and converted to timeline time. Two segments on one track cannot overlap: `is_segment_overlap_in_target_track`, then `move_segments_if_conflict` or `fixSegmentOverlapInTrack`. A failed insert is logged as `insert track failed`.

Dropping from the library is `DragToTimeline` and `DragResourceDrop`. Dropping files is `DragFilesDrop`. Moving a segment is `DragSegment`, then `DraggingSegmentsDropped`. A drag must move a Manhattan distance before it counts. While the pointer is near the edge, `DragScrollTimer` scrolls the sheet. `DragFollow` draws the clips that come along. A segment can be dropped onto another timeline in the same project.

The move that is committed is one of two APIs. `rippleMoveSegments` is the main-track magnet: later segments close up. `multiTrackModeMoveSegments` is the free move. `rippleDeleteSegments` closes the gap after a delete. Paste uses `multiTrackModePasteSegments` and checks `getPasteSegsConflict`. A separate draft command can also place a nested or expanded draft by appending, by rippling at a microsecond, or by overwriting a range.

Auto snapping and the main-track magnet are different switches. Snapping is `turn_on_auto-snap` / `turn_off_automatic_snapping`, with toolbar icons `auto_snapping_on` and `auto_snapping_off`. The magnet is `maintrack_adsorb`. The snap engine is `MainTimeLineAdsorbModel`. It can snap to the playhead (`adsorbCursor`), a segment edge (`adsorbSegment`, `adsorbShiftSegment`), every edge (`adsorbAll`), a keyframe, a time mark, an audio beat, the middle of a transition, or the timeline ends. The line drawn during that drag is `MainTimeLineHighlightTipsLine.qml`. Its height, offset, and distance are `AdsorbLineHeight`, `AdsorbOffset`, and `AdsorbedDistance`. Audio has its own `MainTimeLineTrackAudioAdsorb.qml`.

Reference lines are the guides pulled off the ruler (`ReferenceLineFromRulerFinish`). They are stored on the timeline as horizontal and vertical values, can be locked or hidden, and can be shown in pixels or percent. They stay after the drag. The adsorb line does not.

Trim is an edge drag, including several clips at once. The toolbar also has quick cut left, quick cut right, delete left, and delete right. A trim commit carries `sourceStart`, `sourceDuration`, `targetStart`, and `targetDuration`, and the seek after a trim can snap. Linkage follows with `HandleOwnerUpdateClip`.

Linkage is per kind: overlay, text, stickers, effects, filters, adjustments, audio, sound effects, and text to speech. Each dependent has an owner (`linkageGetOwnerByDependent` / `linkageGetDependents`). A meta type can be added or ignored, and split can be told to keep or break the link. The owner handlers are move out, move inside, update move, update clip, remove, and update speed. Group linkage and multi-cut linkage are separate builders. If a linked clip has no room, the log says to insert a track on the source track. Script-to-video projects start with linkage off.

Split is `splitSegment`. It fails too close to an edge, inside a transition, or on a patch conflict. Smart split and split by scene are separate tools. A transition is `MainTimeLineSegmentTransition` and has `transition.overlap`. It sits on a cut. `TransitionConflictRepairTask` repairs a bad one.

The timeline toolbar also has zoom in, zoom out, zoom to fit, markers, AI markers, beat matching, a preview axis that plays audio while the playhead is dragged, track lock, segment lock, undo, redo, freeze, reverse, mirror, extract audio, record, and cutoff. Segment shade lines, filmstrip frames, and keyframes are drawn on the clip. Highlight tips also exist for a segment, a track, replace, and match-beats.

The prototype draws the snap line while a drag or trim is aligned, keeps reference lines pulled off the ruler, scrolls when a drag nears the side, lets the playhead be dragged, fits the zoom, and locks a track. It does not play audio while seeking, and it does not build freeze, reverse, record, or a second timeline.

## Where a lane can sit

The desktop timeline keeps five type lists in `editor::utils`: `aboveMainTrackTypes`, `aboveVideoTrackTypes`, `videoTrackTypes`, `underMainTrackTypes`, and `noVideoTypes`. They are filled at startup from constants in `libVECreator.dylib`. The engine keeps the same split in `LVVETrackTypesAboveMain`, `LVVETrackTypesAboveMainVideo`, `LVVETrackTypesAboveMainEffect`, and `LVVETrackTypesUnderMain`.

Above the main lane: video, sticker, text, effect, filter, adjust, and plugin effect. The engine list names `mixed` where the UI list names plugin effect. Under the main lane: audio only. `aboveVideoTrackTypes` is that above-main list without video, so sticker, text, effect, filter, adjust, and plugin effect stay above overlay video too.

Picture order comes from the render-index floors in `RenderIndexUtils`. A missing index is raised to the floor. Low to high, which is bottom to top on the timeline: video, then adjust at 7000, filter at 10000, effect at 11000, then text, stickers, shapes, and text templates together at 14000, then plugin effect at 20000. Text and stickers share a floor, so one can sit above the other. A filter cannot move up into the effect band, and nothing in those bands can move below the main lane.

While a drag is over the wrong kind, `doTrackModePlacement` tries the track above, then creates a lane. The log lines are `cannot place on ref track, try upper track` and `cannot place on upper track, create new track`. The hot zones on a lane are the upper and lower edges. The prototype inserts a new lane in that band. Text and stickers can still be ordered against each other. Audio is the only lane below main.

The timeline ruler, its pixel size, and its tick steps are in `ruler.md`.
