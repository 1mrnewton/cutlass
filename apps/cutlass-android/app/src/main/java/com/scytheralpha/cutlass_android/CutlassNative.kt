package com.scytheralpha.cutlass_android

/**
 * JNI bridge to the `cutlass-mobile` Rust library (`libcutlass_mobile.so`).
 *
 * The native side brings up a headless wgpu device, composites a demo scene on
 * the GPU, and returns the pixels as ARGB_8888 ints — exactly the layout
 * `Bitmap.createBitmap(int[], w, h, ARGB_8888)` expects.
 */
object CutlassNative {
    init {
        System.loadLibrary("cutlass_mobile")
    }

    /**
     * Render the built-in demo scene at [width]×[height].
     *
     * @return `[width, height, argb…]` (ARGB_8888 pixels prefixed by their
     *   dimensions), or an empty array if the GPU/compositor was unavailable.
     */
    external fun renderDemo(width: Int, height: Int): IntArray

    /**
     * Decode + composite the first frame of the video at [path], scaled to fit
     * [maxWidth]×[maxHeight].
     *
     * @return `[width, height, argb…]` (the actual fitted size), or an empty
     *   array on failure.
     */
    external fun renderFileFrame(path: String, maxWidth: Int, maxHeight: Int): IntArray

    /**
     * Open the synthetic scrub demo (a hue sweep over ~6s, no assets).
     *
     * @return an opaque session handle, or `0` if the GPU/renderer was
     *   unavailable. Pass it to the other `preview*` calls and release it with
     *   [previewClose].
     */
    external fun previewOpenDemo(): Long

    /**
     * Open a preview that scrubs the video at [path] (hardware decode).
     *
     * @return a session handle, or `0` on failure.
     */
    external fun previewOpenVideo(path: String): Long

    /** Total scrub length in seconds for [handle] (`0.0` if invalid). */
    external fun previewDurationSeconds(handle: Long): Double

    /**
     * Render the frame at [seconds] for [handle] (clamped to range).
     *
     * Not thread-safe: callers must serialize renders on a single handle.
     *
     * @return `[width, height, argb…]`, or an empty array on failure.
     */
    external fun previewRender(handle: Long, seconds: Double): IntArray

    /** Release a handle from `previewOpen*`. A `0` handle is a no-op. */
    external fun previewClose(handle: Long)
}
