#ifndef CUTLASS_MOBILE_H
#define CUTLASS_MOBILE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * C ABI for the `cutlass-mobile` Rust library.
 *
 * The engine composites a frame on a headless wgpu (Metal) device and returns
 * the pixels as RGBA8. Every non-null buffer must be released exactly once with
 * `cutlass_image_free`.
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
 * Editing session: the JSON command/state bridge.
 *
 * A `CutlassSession` owns the editor engine. The app drives it the same way the
 * AI agent will: send a `Command` as JSON (`cutlass_session_apply`), read a
 * curated project read-model back as JSON (`cutlass_session_state_json`), and
 * render preview frames at a time (`cutlass_session_render`). Edits flow in as
 * commands only; the engine owns undo/redo and persistence.
 *
 * JSON-returning calls hand back a heap C string the caller must release once
 * with `cutlass_string_free`. Not thread-safe: serialize calls on one handle.
 */
typedef struct CutlassSession CutlassSession;

/*
 * Create a new, empty session at `fps_num`/`fps_den` (e.g. 30/1) with one video
 * track, caching under `cache_dir_utf8` (`cache_dir_len` UTF-8 bytes; pass NULL
 * for a temp dir). NULL on invalid fps or engine/GPU failure.
 */
CutlassSession *cutlass_session_new(int32_t fps_num, int32_t fps_den,
                                    const uint8_t *cache_dir_utf8, size_t cache_dir_len);

/*
 * Open a `.cutlass` project file into a new session (every referenced media path
 * must exist), caching under `cache_dir_utf8`. NULL on failure.
 */
CutlassSession *cutlass_session_open(const uint8_t *path_utf8, size_t path_len,
                                     const uint8_t *cache_dir_utf8, size_t cache_dir_len);

/* Save the session to a `.cutlass` file. Returns true on success. */
bool cutlass_session_save(CutlassSession *handle, const uint8_t *path_utf8, size_t path_len);

/*
 * Apply a JSON-encoded `Command`. Returns a JSON C string:
 * `{ok, error?, created_clip?, created_track?, media?, revision, dirty,
 *   can_undo, can_redo}`. Free with `cutlass_string_free`.
 */
char *cutlass_session_apply(CutlassSession *handle, const uint8_t *json_utf8, size_t json_len);

/* Undo / redo the last edit. Returns the same result JSON as `apply`. */
char *cutlass_session_undo(CutlassSession *handle);
char *cutlass_session_redo(CutlassSession *handle);

/*
 * Import a media file into the pool. Returns a JSON C string with the new media
 * id and metadata for building an `AddClip`:
 * `{ok, media, fps:{num,den}, duration_ticks, width, height, is_image,
 *   has_audio, ...}`. Free with `cutlass_string_free`.
 */
char *cutlass_session_import(CutlassSession *handle, const uint8_t *path_utf8, size_t path_len);

/*
 * Serialize the curated project read-model the timeline renders from:
 * `{name, fps, duration_ticks, revision, dirty, can_undo, can_redo, canvas,
 *   media:[...], tracks:[{id, kind, name, ..., clips:[{id, kind, start,
 *   duration, media_id?, label}]}]}`. Free with `cutlass_string_free`.
 */
char *cutlass_session_state_json(const CutlassSession *handle);

/*
 * Render the session preview frame at `seconds` (clamped to the timeline).
 * `data` is NULL on failure; release every non-null result with
 * `cutlass_image_free`.
 */
CutlassImage cutlass_session_render(CutlassSession *handle, double seconds);

/* Release a session handle from `cutlass_session_new`/`_open`. NULL is a no-op. */
void cutlass_session_close(CutlassSession *handle);

/* Free a JSON C string returned by a `cutlass_session_*` call. NULL is a no-op. */
void cutlass_string_free(char *ptr);

#endif /* CUTLASS_MOBILE_H */
