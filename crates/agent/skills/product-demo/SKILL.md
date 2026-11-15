title: Product demo
description: Cut a product demo with the named media, titles, callouts, zooms, and captions.
tools: media.probe, draft.query, draft.apply, catalog.search, speech.start, speech.poll
keywords: walkthrough, screen recording, tutorial, showcase, promo

Place the named media, then titles, callouts, zooms, and captions. Probe the media with media.probe, query the draft, search the catalog, and run speech.start and speech.poll when the captions need word times. Apply those placements in one combo with draft.apply.

1. media.probe each file: it answers the type to import it as, its duration, size and streams.
2. One draft.apply: import each file with those numbers (a new id each), then place them on main one after another. A placed segment has id, material_id, target_timerange {start, duration} on the timeline and source_timerange for the part of the file it plays.
3. Trim and split to the moments that matter (trim, split, remove).
4. Titles: add_text at the start and before each part. Callouts: add_shape (arrow or rectangle) and add_text over the stretch they point at, each on its own lane.
5. Zooms: on the clip, add_keyframe at the offset the zoom starts, key_pose a little later with scale 1.3 and x, y moved toward the detail, then key_pose back to scale 1 and 0, 0.
6. Joins: set_transition on the outgoing clip (effect_id "cutlass-dissolve", duration 500000).
7. Captions: speech.start and speech.poll for the word times, then add_subtitles, as the Captions skill says.
8. draft.query "segment <id>" reads back what a clip now stores.

draft.query "ops import place add_text key_pose set_transition" shows the exact JSON.
