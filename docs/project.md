# Project

The edit is a project: settings, assets, and one timeline. Commands change that
value. Undo is the command history.

Sequence time is an integer number of microseconds, a `Timestamp` in
`cutlass-core` with timescale `1_000_000`. `docs/time.md` is how that time is
drawn. This file is the document.

The types go in `crates/project` (`cutlass-project`). It depends only on
`cutlass-core`. It does no I/O. The shell loads and saves. The file format is
not this model; it comes later.

## Time

A time range is a start and a duration, each a `Timestamp`.

The timeline range uses the sequence clock. The source range keeps the asset's
own timescale, the one the container counted in. Speed maps one onto the other.
The timeline side rounds to the nearest microsecond, halves away from zero.

The frame rate is a project setting. It starts at 30. It is a rational, used
for the ruler, snap, and export.

| Rate | Stored as | Nominal frames | Drop frame |
| --- | --- | --- | --- |
| 23.976 | 24000/1001 | 24 | Off |
| 24 | 24/1 | 24 | Off |
| 25 | 25/1 | 25 | Off |
| 29.97 | 30000/1001 | 30 | On when chosen |
| 30 | 30/1 | 30 | Off |
| 50 | 50/1 | 50 | Off |
| 59.94 | 60000/1001 | 60 | On when chosen |
| 60 | 60/1 | 60 | Off |

Choosing 29.97 or 59.94 turns drop frame on. It can be turned off without
changing the rate. Any other rate forces it off. Drop frame changes labels
only. The stored time stays microseconds. Nominal frames and the skip pattern
are in `docs/time.md`.

At 29.97 and 59.94 a frame is not a whole number of microseconds. Snap and the
playhead use the nominal frame in `docs/time.md`, then the sequence clock
again.

## Project

| Field | What it is |
| --- | --- |
| Id | UUID |
| Name | The cut's name |
| Canvas | Width, height, and a ratio: original, 16:9, 9:16, 1:1, 4:5, or 4:3. Original takes the first video's size |
| Frame rate | A row of the table above |
| Drop frame | The bool next to that rate |
| Color space | Rec. 709, Rec. 2020 HLG, or Rec. 2100 PQ. Rec. 709 is the start |
| Background | A canvas color. Black is the start |
| Magnet | Main-track clips pack from the start, with no gaps |
| Linkage | A segment that sits on exactly one main segment follows that segment when it moves, trims, or is deleted |

Duration is the end of the last segment. It is not stored.

## Asset

One media file is one asset: id, kind, the shell's source string, and the
media duration. A video asset may carry audio. An image has no media duration.
Width and height sit on visual assets.

The source string is a path or a content URI. The project crate does not open
it.

| Kind | Holds |
| --- | --- |
| Video | Picture, and audio when the file has it |
| Audio | Sound only |
| Image | A still |

## Timeline

One timeline: id and an ordered list of tracks. The list is the paint order,
bottom to top. Nested timelines wait.

## Track

A track has an id, a kind, and its segments. One video track is the main
track. It is always there, including when it holds nothing. Any other track
is removed when its last segment goes.

Segments on one track do not overlap. A command that would overlap inserts
another track of that kind.

| Kind | Where it sits |
| --- | --- |
| Video, main | The bottom picture lane. Magnet applies here |
| Video | Overlay, just above main |
| Adjust | Above overlay |
| Filter | Above adjust |
| Effect | Above filter |
| Text | Shares a band with stickers, above effect |
| Sticker | Shares a band with text |
| Audio | The only kind below main |
| Plugin effect | The top picture lane |

Text and stickers interleave by list order. A new track is inserted in its
band, which is the rule in `docs/timeline.md`.

## Segment

A segment is one use of an asset.

| Field | What it is |
| --- | --- |
| Id | UUID |
| Asset | The file |
| Source range | Where in the asset, in the asset's timescale. Unused for an image |
| Timeline range | Where on the sequence, in microseconds |
| Channels | Picture, sound, or both. Video assets default to both. Audio assets are sound. Images are picture |
| Speed | A rational. `1/1` is the start. Timeline duration is the source duration divided by speed, then rounded onto the sequence clock |
| Visible | Shown when on |
| Volume | Linear gain. `1` is the start |
| Transform | Position, scale, rotation, opacity, and flips. Identity is the start |
| Keyframes | Samples of those properties. The offset is from the segment's timeline start |

A keyframe property is one of position, scale, rotation, opacity, or volume.

## Left behind

These are CapCut's, looked up in `libvideoeditor.dylib` and in desktop drafts.
They stay out on purpose.

**Their draft file.** `draft_info.json` is schema `360000`, a bag of maps, and
a platform block that stores a device id, a disk id, and a MAC address. Cutlass
does not write that file and does not store a machine id. An importer can read
a draft later. The Cutlass file format is a separate decision.

**Frame rate as a float.** Drafts store `fps` as a JSON number. `29.97` is not
a frame rate. The table above is the closed set.

**Both ranges in microseconds.** Drafts convert the source range into
microseconds too. That rounds a trim off the container's own ticks. The source
range keeps the asset timescale so a seek still names the same packet.

**A stored duration.** The draft repeats the timeline length in `duration`.
The length is the end of the last segment, so the two cannot drift.

**The track flag integer.** Main is flag `0` on a video track, or on a mixed
track. The converter keeps `1` through `6` and turns anything else into `0`.
Those six values have no names in the library. Main is a role on a video
track. One kind per track covers the rest.

**Mixed, script-to-video, ad, and composition tracks.** The type enum also has
`none`, `text_to_video_text`, `text_to_video_text_audio`, `composition`,
`adcube`, and `mixed`. `none` is not a lane. Script-to-video and ad tracks are
product modes. Mixed exists so one lane can hold several kinds. A composition
is a nested draft. One kind per track, and one timeline, cover the editor.
A nested cut can later be a timeline id on a segment.

**Render-index floors.** Picture order is a set of floors: video, then adjust
at 7000, filter at 10000, effect at 11000, text and stickers at 14000, plugin
effect at 20000. Track order is that stack. The floors do not need a second
number.

**The materials dictionary.** A draft has a map per effect kind, and a segment
points at those rows with `extra_material_refs`. One asset per file. Segment
fields hold transform, volume, and speed. A later effect is an attachment by
id. A missing attachment is off, which replaces the `enable_lut` style flags
on every segment.

**Speed as a material and a field.** Each segment carries `speed` and also
points at a speed material. One rational on the segment is the constant.
A curve, when it exists, replaces that rational. It is not a second object.

**`render_timerange`.** The third range on a segment is `0` in real drafts.
Source and timeline are the two ranges.

**A three-hour still.** A photo's media duration is `10800` seconds times
`1_000_000`. An image has no media duration. How long it sits is the segment's
timeline duration.

**A second asset for the file's audio.** `video_original_sound` is another
material with the same path. The video asset already has that audio. Channels
on the segment play the picture, the sound, or both. Detach audio adds a sound
segment of the same asset.

**Reverse as a flag.** `reverse` asks the engine to play a file backwards.
That needs new frames. When it exists, it is a new asset, and the segment
points at it.

**View state in the document.** Drafts store zoom. Zoom, scroll, selection,
and the playhead are the session. They are not commands, and they are not
undo.

**Seconds as the stored clock.** The prototype UI kept clip times in seconds.
Seconds remain a way to draw a time, in `docs/time.md`. The document stores
ticks.
