#ifndef CUTLASS_MOBILE_H
#define CUTLASS_MOBILE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * C ABI for the `cutlass-mobile` Rust library.
 *
 * Two data shapes cross this boundary:
 *  - pixels: `CutlassImage` RGBA8 buffers, released with `cutlass_image_free`;
 *  - everything else: UTF-8 JSON strings (commands, intents, UI state,
 *    verdicts), released with `cutlass_string_free`.
 *
 * String-returning calls answer with a response envelope:
 *   {"ok": <payload>, "revision": <n>}   on success
 *   {"err": {"kind": "...", "message": "..."}} on failure
 * `kind` is stable ("model" | "time" | "render" | "decode" | "io" |
 * "import" | "export" | "missing_media" | "unsupported" | "protocol" |
 * "cancelled"); `message` is human-readable.
 */

typedef struct CutlassImage {
    /* RGBA8 pixels (`len` bytes), or NULL if rendering failed. */
    uint8_t *data;
    /* Length of `data` in bytes (== width * height * 4). */
    size_t len;
    uint32_t width;
    uint32_t height;
} CutlassImage;

/* Render the built-in demo scene at `width` x `height`. `data` is NULL on failure. */
CutlassImage cutlass_render_demo(uint32_t width, uint32_t height);

/*
 * Decode + composite the first frame of the video at `path_utf8` (a UTF-8 path
 * of `path_len` bytes, no NUL terminator required), scaled to fit
 * `max_width` x `max_height`. `data` is NULL on failure.
 */
CutlassImage cutlass_render_file_frame(const uint8_t *path_utf8, size_t path_len,
                                       uint32_t max_width, uint32_t max_height);

/* Release a buffer returned by `cutlass_render_demo`. NULL/empty is a no-op. */
void cutlass_image_free(CutlassImage img);

/*
 * Interactive preview.
 *
 * A `CutlassPreview` holds a persistent GPU device + decoder cache bound to a
 * project, so scrubbing only pays for the frame at a given time. Open a session,
 * call `cutlass_preview_render` per slider tick, and free it with
 * `cutlass_preview_close`. Not thread-safe: serialize calls on one handle.
 */
typedef struct CutlassPreview CutlassPreview;

/* Open the synthetic scrub demo (no assets). NULL on failure. */
CutlassPreview *cutlass_preview_open_demo(void);

/*
 * Open a preview that scrubs the video at `path_utf8` (`path_len` UTF-8 bytes).
 * NULL if the file can't be probed or the GPU is unavailable.
 */
CutlassPreview *cutlass_preview_open_video(const uint8_t *path_utf8, size_t path_len);

/* Total scrub length in seconds. 0.0 for a NULL handle. */
double cutlass_preview_duration_seconds(const CutlassPreview *handle);

/*
 * Render the preview frame at `seconds` (clamped to range). `data` is NULL on
 * failure; release every non-null result once with `cutlass_image_free`.
 */
CutlassImage cutlass_preview_render(CutlassPreview *handle, double seconds);

/* Release a handle from `cutlass_preview_open_*`. NULL is a no-op. */
void cutlass_preview_close(CutlassPreview *handle);

/*
 * Editing session.
 *
 * A `CutlassSession` owns the full editing engine for one project: state,
 * command dispatch, undo/redo, dirty tracking, save/load, and a persistent
 * renderer. Commands/intents go in as JSON, UI state comes out as JSON,
 * preview frames come out as `CutlassImage`.
 *
 * Not thread-safe: serialize all calls on one handle (wrap it in an actor).
 * Different handles are independent.
 */
typedef struct CutlassSession CutlassSession;

/* Free a string returned by any `cutlass_session_*`/`cutlass_export_*` call.
 * NULL is a no-op. */
void cutlass_string_free(char *ptr);

/*
 * Open a fresh session: an empty project at `fps_num/fps_den` (falls back to
 * 30 fps when non-positive) with a main video track. NULL when the GPU
 * renderer can't be brought up. Free with `cutlass_session_close`.
 */
CutlassSession *cutlass_session_new(int32_t fps_num, int32_t fps_den);

/*
 * Open a session from a `.cutlass` project file (`path_len` UTF-8 bytes).
 * Missing media paths are tolerated (clips relink later). NULL on failure.
 */
CutlassSession *cutlass_session_open(const uint8_t *path_utf8, size_t path_len);

/* Release a session handle. NULL is a no-op. */
void cutlass_session_close(CutlassSession *handle);

/*
 * Apply one wire command (`json_len` UTF-8 bytes of the flat {"type": ...}
 * command object; Save/Load/Import are commands too). Returns a response
 * envelope; free it with `cutlass_string_free`.
 */
char *cutlass_session_apply(CutlassSession *handle, const uint8_t *json_utf8, size_t json_len);

/*
 * Run one gesture-level intent ({"intent": ...} JSON). Multi-command intents
 * are grouped into a single undo step and roll back atomically on failure.
 * Returns a response envelope; free with `cutlass_string_free`.
 */
char *cutlass_session_intent(CutlassSession *handle, const uint8_t *json_utf8, size_t json_len);

/*
 * The full UI presentation state (ordered lanes, clips, canvas, durations,
 * undo/redo/dirty flags) as JSON. Free with `cutlass_string_free`.
 */
char *cutlass_session_ui_state(CutlassSession *handle);

/* Undo/redo the most recent edit (or group). Returns whether anything changed. */
bool cutlass_session_undo(CutlassSession *handle);
bool cutlass_session_redo(CutlassSession *handle);
bool cutlass_session_can_undo(const CutlassSession *handle);
bool cutlass_session_can_redo(const CutlassSession *handle);

/* Monotonic revision; bumps on every successful mutation (cache keys). */
uint64_t cutlass_session_revision(const CutlassSession *handle);

/* Whether the session has edits not yet saved to its project file. */
bool cutlass_session_is_dirty(const CutlassSession *handle);

/*
 * History grouping: every command between begin and commit folds into one
 * undo step (property-panel sessions). Rollback aborts the open group,
 * reverting its commands.
 */
void cutlass_session_begin_group(CutlassSession *handle);
void cutlass_session_commit_group(CutlassSession *handle);
void cutlass_session_rollback_group(CutlassSession *handle);

/* End of the timeline in seconds (0 for an empty project). */
double cutlass_session_duration_seconds(const CutlassSession *handle);

/*
 * Render the timeline frame nearest `seconds` (snapped to the project frame
 * grid, clamped to the timeline), scaled to fit `max_width` x `max_height`
 * (aspect preserved, never upscaled). `data` is NULL on failure; release
 * every non-null result once with `cutlass_image_free`.
 */
CutlassImage cutlass_session_render_fit(CutlassSession *handle, double seconds,
                                        uint32_t max_width, uint32_t max_height);

#endif /* CUTLASS_MOBILE_H */
