# Persistent state

This table is the target lifetime model from the specification. The current
decoder does not store reference images, saved CDFs, segmentation, loop-filter
deltas, global motion, grain parameters, or motion fields. `reset` clears the
sequence header, in-progress assembly, the output queue, and the needs-reset
flag. A shown lossless intra frame is the only image that is published.

Fields below are the ones later frames can load. "Intra" means
`FrameIsIntra || error_resilient_mode` (primary ref none), not every intra
block.

| Field | Init | Load | Mutate | Save | Show-existing |
| --- | --- | --- | --- | --- | --- |
| Sequence header | First sequence OBU | — | Replaced only when a new CVS starts | — | Unchanged |
| SeenFrameHeader | 0 | — | Frame header / temporal delimiter / finished tile group | — | Cleared after wrapup |
| RefValid, RefOrderHint | 0 on shown keyframe | Error-resilient order-hint check may invalidate | Refresh mask | Slot | Shown keyframe sets refresh to all slots via the loaded frame |
| Reference pixels | Empty | Inter prediction reads | Wrapup copies `Arc` into refreshed slots | Slot | Not filtered in place; presentation clones the `Arc` |
| Saved CDFs | `init_*` when primary ref is none | `load` from primary ref | Tile adaptation; context-update tile copied at exit if enabled | Slot at wrapup | Loaded-keyframe path copies the shown frame's state when the spec says so |
| Segmentation features | Header, or previous when update flags are clear | Primary ref via `load_previous` | Header and map | Slot | Loaded with the shown keyframe state |
| Loop-filter deltas | Defaults, then header | Primary ref when delta update is 0 | Header | Slot | Same as segmentation |
| Global motion | Identity, then header | `PrevGmParams` from primary ref | Header | Slot | Loaded with keyframe state |
| Film grain params | Reset when absent | `load_grain_params` for reuse or show-existing | Header | Slot | Loaded, then applied only on the output copy |
| Motion field | Estimated at header when `use_ref_frame_mvs` | From reference `mfmv` | Stored at wrapup | Slot | Not recomputed from a display-only event |
| Current Q / delta LF | `base_q_idx` / 0 at each tile | — | Superblock deltas | Not a reference field | — |
| Order hint, frame id | Header | Checked against slots | Refresh | Slot | Display id checked; PTS comes from the current packet |

`reset` clears every row except immutable tables. A repeated identical sequence
header does not drop references. A new coded video sequence (temporal unit with
a sequence header and a shown keyframe at temporal id 0) replaces sequence
state and follows the keyframe reference reset.
