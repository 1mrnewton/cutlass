title: Project setup
description: Set the canvas, frame rate, and background for a new project.
tools: draft.query, draft.apply
keywords: vertical, portrait, landscape, square, aspect ratio, resolution, shorts, reels

Query the project with draft.query. Set the canvas, the frame rate, and the background through draft requests in one combo with draft.apply.

These are requests on the project, sent with draft.apply. settings.set is for the app's own settings and does not change a project.

1. draft.query "project" shows the canvas, fps and background as they are.
2. One draft.apply with what should change:
   - set_canvas_size width, height and ratio: vertical 1080 by 1920 "9:16", square 1080 by 1080 "1:1", wide 1920 by 1080 "16:9", 1080 by 1350 "4:5". Give a size that has that ratio.
   - set_frame_rate fps 24, 25, 30 or 60 (or 23.976, 29.97, 59.94), with drop_frame false.
   - set_canvas_background color "#rrggbb".
   - set_project_name name.
3. A clip that no longer fills the new canvas: set_letterbox_blur fills behind it with a blurred copy, set_scale enlarges it, set_reframe follows its subject.
4. draft.query "project" again to confirm.
