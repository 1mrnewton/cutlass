title: Motion graphics
description: Build motion graphics with shapes, text, keyframes, and catalog effects.
tools: catalog.search, package.ensure, draft.query, draft.apply, sandbox.run
keywords: animate, animation, title card, lower third, logo, intro, outro

Place shapes, text, keyframes, and catalog effects. Call catalog.search, package.ensure, draft.query, and draft.apply for those placements in one combo. Use sandbox.run when a number has to be computed.

- Shapes: add_shape (rectangle, circle, triangle, line, arrow) or draw_shape on a sticker lane, then set_shape_size, set_shape_fill, set_shape_stroke, set_shape_corners.
- Text: add_text on a text lane (face a font file such as Inter[opsz,wght].ttf, never empty), then set_text_size, set_text_color, set_text_background, set_text_outline.
- Place: set_position (x and y are fractions of half the canvas from the middle, y up), set_scale, set_rotation, set_opacity.
- Move: set_animation gives an in, out or loop animation (catalog.search "animation" lists the ids). For a move of your own, add_keyframe at the first offset keys the pose as it is, then key_pose at a later offset writes the new pose; key_pose only keys what already has a key, so add_keyframe comes first. set_keyframe_graph eases the span ending at a key ("Ease Out").
- Effects: catalog.search finds ids. add_effect takes "recipe/<name>", add_filter "look/<name>", set_transition effect_id "cutlass-<key>" on the outgoing clip.
- Two clips may not overlap on one lane: give each layer its own track_id (a new id makes the lane).
- New ids for every segment_id and material_id. One draft.apply for each finished idea.
- To see a frame, player.seek to its time and export.start "new <path>.png".

draft.query "ops <kind> ..." shows the exact JSON of any request.
