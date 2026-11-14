package com.scytheralpha.cutlass_android

import android.graphics.Bitmap
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import java.nio.ByteBuffer
import java.util.UUID
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import uniffi.cutlass_ffi_bindings.AnimationPreset
import uniffi.cutlass_ffi_bindings.Background
import uniffi.cutlass_ffi_bindings.CanvasRatio
import uniffi.cutlass_ffi_bindings.Clip
import uniffi.cutlass_ffi_bindings.ClipAnimations
import uniffi.cutlass_ffi_bindings.ClipState
import uniffi.cutlass_ffi_bindings.EffectChoice
import uniffi.cutlass_ffi_bindings.Editor
import uniffi.cutlass_ffi_bindings.Frame
import uniffi.cutlass_ffi_bindings.Keyframes
import uniffi.cutlass_ffi_bindings.Lane
import uniffi.cutlass_ffi_bindings.LaneKind
import uniffi.cutlass_ffi_bindings.Stage
import uniffi.cutlass_ffi_bindings.TextInfo
import uniffi.cutlass_ffi_bindings.TextStyleChange
import uniffi.cutlass_ffi_bindings.Tick
import uniffi.cutlass_ffi_bindings.Tile
import uniffi.cutlass_ffi_bindings.ToolBar
import uniffi.cutlass_ffi_bindings.Wave
import uniffi.cutlass_ffi_bindings.animationPresets
import uniffi.cutlass_ffi_bindings.cropRatios
import uniffi.cutlass_ffi_bindings.effectChoices
import uniffi.cutlass_ffi_bindings.flingStart
import uniffi.cutlass_ffi_bindings.flingStep
import uniffi.cutlass_ffi_bindings.Font as FaceInfo

/**
 * The open project as the screens see it.
 *
 * Every edit goes through the Rust editor, which owns the draft and its
 * history. This model keeps what the screens draw: the rows, the length,
 * the playhead, and the picture at the playhead. A picture is drawn off the
 * main thread; while one draws, only the newest time asked for waits.
 *
 * Every call here is made on the main thread, as on iOS.
 */
class EditorModel(project: String, private val scope: CoroutineScope) {
    var lanes by mutableStateOf(emptyList<Lane>())
        private set
    var durationUs by mutableLongStateOf(0L)
        private set
    private var playheadValue by mutableLongStateOf(0L)
    var playheadUs: Long
        get() = playheadValue
        private set(value) {
            if (value != playheadValue) {
                playheadValue = value
                if (selectedClip != null) refreshStage()
            }
        }
    var picture by mutableStateOf<ImageBitmap?>(null)
        private set
    var canvasAspect by mutableStateOf(16f / 9f)
        private set

    /** The canvas ratios, Original first, as Rust sizes them for this project. */
    var ratios by mutableStateOf(emptyList<CanvasRatio>())
        private set

    /** What fills the canvas around the main clips. */
    // Named apart from setBackground(), which edits the draft.
    @set:JvmName("keepBackground")
    var background by mutableStateOf<Background>(Background.None)
        private set
    var isPlaying by mutableStateOf(false)
        private set
    var isOpening by mutableStateOf(true)
        private set
    var isImporting by mutableStateOf(false)
        private set
    var notice by mutableStateOf<Notice?>(null)
        private set

    private var selectedValue by mutableStateOf<String?>(null)
    var selectedClip: String?
        get() = selectedValue
        set(value) {
            if (value != selectedValue) {
                selectedValue = value
                refreshStage(now = true)
            }
        }

    /**
     * The selected picture clip's box on the canvas, as the picture on
     * screen shows it. Text and sound have none.
     */
    var stage by mutableStateOf<Stage?>(null)
        private set

    /**
     * The box for the draft as it is now. After an edit it waits for the
     * picture drawn from that draft, so the box never runs ahead of it.
     */
    private var placed: Pair<String?, Stage?> = null to null

    /** The selected clip's keyframes; clips that take none have null. */
    var keys by mutableStateOf<Keyframes?>(null)
        private set

    /** The selected clip's in and out animations; clips that take none have null. */
    var animations by mutableStateOf<ClipAnimations?>(null)
        private set

    /** The Animations panel's presets; they never change. */
    val animationList: List<AnimationPreset> = animationPresets()

    /** dp per microsecond on the timeline, as iOS uses points. Rust holds it inside what the cut allows. */
    var pixelSize by mutableStateOf(60.0 / 1_000_000)
        private set

    /** Ruler ticks and filmstrip tiles for the window around the playhead. */
    var ticks by mutableStateOf(emptyList<Tick>())
        private set
    var films by mutableStateOf(emptyMap<String, List<Tile>>())
        private set

    /** Each clip's sound over the window read, by clip id. */
    var waves by mutableStateOf(emptyMap<String, Wave>())
        private set

    /** Filmstrip pictures by tile key. */
    val thumbs = mutableStateMapOf<String, ImageBitmap>()

    data class Notice(val text: String, val id: String = UUID.randomUUID().toString())

    /** The faces text can be set in, as Rust found them. */
    var fonts by mutableStateOf(emptyList<FaceInfo>())
        private set

    /**
     * The Rust editor once open. Hidden while an export holds its thread,
     * so nothing here waits on it.
     */
    private val editor: Editor? get() = if (exporting) null else opened
    private var opened: Editor? = null
    var exporting by mutableStateOf(false)
        private set
    private var drawing = false
    private var waiting: Long? = null
    private var playback: Job? = null

    /** The strip sliding on after a swipe. */
    private var coasting: Job? = null

    /** The window the ticks and tiles cover, and the edit they were read for. */
    private data class StripWindow(val from: Long, val to: Long, val pixelSize: Double, val edit: Int)

    private var stripWindow: StripWindow? = null

    /** Counts the project's changes, so a panel showing a clip rereads it. */
    var edits by mutableIntStateOf(0)
        private set
    private var stripRetry: Job? = null
    private val fetching = mutableSetOf<String>()

    /** The half width the strip last drew at, for a retry. */
    var lastHalfWidth = 0.0

    init {
        val root = Places.root
        val bundle = Places.bundle
        scope.launch {
            val result = withContext(Dispatchers.Default) {
                runCatching {
                    val editor = Editor.open(root, bundle, project)
                    editor.setFontFolders(Places.fontFolders)
                    editor
                }
            }
            result.onSuccess { editor ->
                opened = editor
                fonts = runCatching { editor.fonts() }.getOrDefault(emptyList())
                reload()
            }.onFailure { show(it) }
            isOpening = false
        }
    }

    val hasClips: Boolean get() = lanes.any { it.clips.isNotEmpty() }

    /** The clip a split cuts: the selection when it covers the playhead, else the main-lane clip under it. */
    val splitTarget: Clip?
        get() = selection?.takeIf { it.covers(playheadUs) } ?: clipUnderPlayhead

    /** The clip a delete removes: the selection wherever it is, else the main-lane clip under the playhead. */
    val deleteTarget: Clip? get() = selection ?: clipUnderPlayhead

    /** The selected clip as the project holds it now. */
    val selection: Clip?
        get() {
            val id = selectedClip ?: return null
            return lanes.asSequence().flatMap { it.clips }.firstOrNull { it.id == id }
        }

    /** The tool bar for the selected clip, as Rust picks it for its row. */
    val selectionBar: ToolBar?
        get() {
            val editor = editor ?: return null
            val id = selectedClip ?: return null
            return runCatching { editor.clipBar(id) }.getOrNull()
        }

    private val clipUnderPlayhead: Clip?
        get() = lanes.firstOrNull { it.main }?.clips?.firstOrNull { it.covers(playheadUs) }

    // Playhead

    fun seek(time: Long) {
        playheadUs = time.coerceIn(0, durationUs)
        runCatching { editor?.seek(playheadUs) }
        draw()
    }

    /**
     * Let the strip slide on after a swipe let go at `velocity` dp a
     * second, finger right being back in time. Rust steps the slowing.
     */
    fun coast(velocity: Double) {
        stopCoast()
        var speed = flingStart(velocity)
        if (speed == 0.0) return
        coasting = scope.launch {
            var last = System.nanoTime()
            while (speed != 0.0 && isActive) {
                delay(16)
                val now = System.nanoTime()
                val step = flingStep(speed, (now - last) / 1e9)
                last = now
                speed = step.velocity
                val target = playheadUs - (step.distance / pixelSize).toLong()
                seek(target)
                if (target <= 0 || target >= durationUs) return@launch
            }
        }
    }

    /** Stop a coast where it is, as a finger landing on the strip does. */
    fun stopCoast() {
        coasting?.cancel()
        coasting = null
    }

    fun togglePlay() = if (isPlaying) pause() else play()

    /**
     * Play with sound. Rust moves the playhead with the heard audio through
     * AAudio; each screen frame asks it where the playhead is, off the main
     * thread, and draws that picture.
     */
    fun play() {
        val editor = editor ?: return
        if (durationUs <= 0) return
        stopCoast()
        isPlaying = true
        playback = scope.launch {
            val started = withContext(Dispatchers.Default) { runCatching { editor.play() } }
            started.onFailure {
                show(it)
                stopped()
                return@launch
            }
            var last = System.nanoTime()
            while (isActive) {
                delay(16)
                val now = System.nanoTime()
                val elapsed = (now - last) / 1_000
                last = now
                val step = withContext(Dispatchers.Default) {
                    runCatching { editor.tick(elapsed) to editor.isPlaying() }
                }.getOrNull()
                if (step == null || !isPlaying) {
                    stopped()
                    return@launch
                }
                playheadUs = step.first
                draw()
                if (!step.second) {
                    stopped()
                    return@launch
                }
            }
        }
    }

    fun pause() {
        runCatching { editor?.pause() }
        stopped()
    }

    private fun stopped() {
        playback?.cancel()
        playback = null
        isPlaying = false
    }

    // Export

    /** Stop playing and hand the editor to an export. Every other call waits until [endExport]. */
    fun beginExport(): Editor? {
        val editor = editor
        if (editor == null || durationUs <= 0) {
            say("Add a photo or video first")
            return null
        }
        stopCoast()
        pause()
        exporting = true
        return editor
    }

    fun endExport() {
        exporting = false
        reload()
    }

    // Edits

    fun importFile(path: String) {
        edit {
            val material = it.import(path)
            selectedClip = it.append(material)
        }
    }

    fun beginImport() {
        isImporting = true
    }

    fun endImport() {
        isImporting = false
    }

    /** Select the main clip under the playhead, for its tools. */
    fun editUnderPlayhead() {
        val clip = clipUnderPlayhead ?: lanes.firstOrNull { it.main }?.clips?.lastOrNull()
        if (clip == null) {
            say("Add a photo or video first")
            return
        }
        selectedClip = clip.id
    }

    fun split() {
        val clip = splitTarget
        if (clip == null) {
            say("Move the playhead over a clip to split it")
            return
        }
        edit { it.split(clip.id, playheadUs) }
    }

    // Effects

    /** The effects that ship in the app, in picker order. */
    val effects: List<EffectChoice> = effectChoices()

    /** Add `effect` at the playhead and select it. */
    fun addEffect(effect: String) {
        var added: String? = null
        edit { added = it.addEffect(effect, playheadUs) }
        added?.let { selectedClip = it }
    }

    // Text

    /**
     * Type into `clip` as the words change, or add a title at the playhead
     * in `face` once the words are not blank, and select it. The whole
     * typing is one undo step. Returns the clip, or null while none exists.
     */
    fun typeText(clip: String?, content: String, face: String): String? {
        var typed = clip
        edit { typed = it.typeText(clip, content, face, playheadUs) }
        typed?.let { selectedClip = it }
        return typed
    }

    fun setTextFont(clip: String, face: String) = edit { it.setTextFont(clip, face) }

    /** Change how `clip` looks. Changes while `live` join one undo step and are saved when the last one, not live, arrives. */
    fun styleText(clip: String, change: TextStyleChange, live: Boolean) =
        edit(save = !live) { it.styleText(clip, change, live) }

    /** What `clip` says, or null when it is not a text clip. */
    fun text(clip: String): TextInfo? = runCatching { editor?.text(clip) }.getOrNull()

    /** The faces whose names have every word of `query`, as Rust matches them; all of them for no words. */
    fun fontsMatching(query: String): List<FaceInfo> =
        runCatching { editor?.fontsMatching(query) }.getOrNull() ?: fonts

    fun deleteClip() {
        val clip = deleteTarget
        if (clip == null) {
            say("Select a clip to delete")
            return
        }
        edit { it.remove(clip.id) }
        selectedClip = null
    }

    /** Drop `clip` on `row` near `startUs`. It may snap within `snapUs` and packs in on a magnetic row. */
    fun moveClip(clip: String, row: Int, startUs: Long, snapUs: Long) = edit {
        it.moveClip(clip, row.coerceAtLeast(0).toUInt(), startUs.coerceAtLeast(0), snapUs, playheadUs)
    }

    /** Drag the start or end edge of `clip` to `timeUs`. */
    fun trim(clip: String, startEdge: Boolean, timeUs: Long, snapUs: Long) = edit {
        it.trim(clip, startEdge, timeUs, snapUs, playheadUs)
    }

    /** Set the clip's linear gain: 1 is unity and 0 is silence. */
    fun setVolume(clip: String, volume: Double) = edit { it.setVolume(clip, volume) }

    /** Give the canvas a ratio. The clips keep their place and fit the new frame. */
    fun setRatio(ratio: String) = edit { it.setRatio(ratio) }

    /** Fill the canvas around the main clips. */
    fun setBackground(background: Background) = edit { it.setBackground(background) }

    /** Play the clip at a constant speed. Its length follows. */
    fun setSpeed(clip: String, speed: Double) = edit { it.setSpeed(clip, speed) }

    // Clip bar edits. Each is one undo step; a clip it does not suit shows
    // Rust's reason as a notice.

    /** How `clip` stands at the playhead, for its clip tools, or null when it is gone. */
    fun clipState(clip: String): ClipState? = runCatching { editor?.clipState(clip, playheadUs) }.getOrNull()

    fun toggleMirror(clip: String) = edit { it.toggleMirror(clip) }

    fun turn(clip: String) = edit { it.turn(clip) }

    fun toggleReverse(clip: String) = edit { it.toggleReverse(clip) }

    /** Crop to the first ratio Rust offers, or show the whole frame again. */
    fun toggleCrop(clip: String) {
        val cropped = clipState(clip)?.crop?.isNotEmpty() ?: return
        val ratio = if (cropped) "" else cropRatios().firstOrNull() ?: return
        edit { it.setCrop(clip, ratio) }
    }

    /** Hold the frame under the playhead as a still, and select the still. */
    fun freeze(clip: String) {
        var still: String? = null
        edit { still = it.freeze(clip, playheadUs) }
        still?.let {
            selectedClip = it
            say("Frame held")
        }
    }

    /** Put the video's sound on its own clip on an audio row, and select that. */
    fun extractAudio(clip: String) {
        val before = soundClips()
        var done = false
        edit {
            it.extractAudio(clip)
            done = true
        }
        if (!done) return
        (soundClips() - before).firstOrNull()?.let { selectedClip = it }
        say("Sound extracted")
    }

    private fun soundClips(): Set<String> =
        lanes.filter { it.kind == LaneKind.AUDIO }.flatMap { lane -> lane.clips.map { it.id } }.toSet()

    /** Drop a beat on the sound at the playhead, or take away the one there. */
    fun toggleBeat(clip: String) {
        var done = false
        edit {
            it.toggleBeat(clip, playheadUs)
            done = true
        }
        if (!done) return
        clipState(clip)?.let { say(if (it.beatHere) "Beat added" else "Beat removed") }
    }

    /** The clip's fade in and fade out, or none. */
    fun fade(clip: String): Fade = clipState(clip)?.let { Fade(it.fadeInUs, it.fadeOutUs) } ?: Fade(0, 0)

    fun setFade(clip: String, inUs: Long, outUs: Long) = edit { it.setFade(clip, inUs, outUs) }

    /** Move an overlay picture onto the main row where it starts. */
    fun toMain(clip: String) = edit { it.toMain(clip) }

    /** Import a picture and put it on an overlay row at the playhead, selected. */
    fun importOverlay(path: String) {
        edit {
            val material = it.import(path)
            selectedClip = it.addOverlay(material, playheadUs)
        }
    }

    /** The beats on each sound clip, as timeline times, read again after each edit. */
    var beats by mutableStateOf<Map<String, List<Long>>>(emptyMap())
        private set

    // Stage

    /** The canvas size in pixels, from the chosen ratio. */
    val canvasSize: Pair<Double, Double>?
        get() {
            val ratio = ratios.firstOrNull { it.chosen } ?: return null
            if (ratio.width <= 0 || ratio.height <= 0) return null
            return ratio.width.toDouble() to ratio.height.toDouble()
        }

    /** Select the topmost picture under the canvas point, or clear the selection when there is none. */
    fun pick(point: Offset) {
        val editor = editor ?: return
        selectedClip = runCatching {
            editor.stagePick(playheadUs, point.x.toDouble(), point.y.toDouble())
        }.getOrNull()
    }

    /** Start moving `clip` on the preview, selecting it. False when it cannot move there. */
    fun beginStage(clip: String): Boolean {
        val editor = editor ?: return false
        pause()
        if (runCatching { editor.stageBegin(clip, playheadUs) }.isFailure) return false
        selectedClip = clip
        return true
    }

    /** Place the clip being moved. Rust snaps it and keeps one undo step. */
    fun moveStage(shiftX: Double, shiftY: Double, pinch: Double, turn: Double) {
        val editor = editor ?: return
        try {
            placed = selectedClip to editor.stageMove(shiftX, shiftY, pinch, turn)
        } catch (error: Exception) {
            show(error)
            return
        }
        draw()
    }

    /** End the preview gesture and save. */
    fun endStage() = edit { it.stageEnd() }

    /** Read the selected box again. `now` shows it at once, for a new selection; otherwise it waits for the next picture. */
    private fun refreshStage(now: Boolean = false) {
        val at = playheadUs
        val clip = selectedClip
        val editor = editor
        val box = clip?.let { runCatching { editor?.stage(it, at) }.getOrNull() }
        placed = clip to box
        if (now) stage = box
        keys = clip?.let { runCatching { editor?.keyframes(it, at) }.getOrNull() }
        animations = clip?.let { runCatching { editor?.animations(it) }.getOrNull() }
    }

    /** The keyframe button: remove the keyframe under the playhead, else key the selected clip there. */
    fun toggleKeyframe() {
        val clip = selectedClip ?: return
        pause()
        val at = playheadUs
        edit { it.toggleKeyframe(clip, at) }
    }

    /** Ease the span the playhead is in along the named curve; an empty name makes it straight. */
    fun easeKeyframes(graph: String) {
        val clip = selectedClip ?: return
        pause()
        val at = playheadUs
        edit { it.easeKeyframes(clip, at, graph) }
    }

    /** Give the selected clip the in (`enter`) or out animation `id`, or clear that side with an empty id. */
    fun setAnimation(enter: Boolean, id: String, durationUs: Long) {
        val clip = selectedClip ?: return
        pause()
        edit { it.setAnimation(clip, enter, id, durationUs) }
    }

    /** Put a main clip at `place` in the main row's order. */
    fun reorder(clip: String, place: Int) = edit { it.reorder(clip, place.coerceAtLeast(0).toUInt()) }

    fun undo() = edit { it.undo() }

    fun redo() = edit { it.redo() }

    fun clearNotice() {
        notice = null
    }

    fun say(text: String) {
        Log.i("cutlass", text)
        notice = Notice(text)
    }

    fun show(error: Throwable) = say(error.message ?: error.toString())

    /** Run one edit, save the project, and redraw. A refusal shows its reason and leaves the project as it was. */
    private fun edit(save: Boolean = true, change: (Editor) -> Unit) {
        val editor = editor ?: return
        try {
            change(editor)
            if (save) editor.save()
        } catch (error: Exception) {
            show(error)
        }
        reload()
    }

    private fun reload() {
        val editor = editor ?: return
        edits += 1
        try {
            lanes = editor.lanes()
            durationUs = editor.durationUs()
            ratios = editor.ratios()
            background = editor.background()
            beats = lanes.filter { it.kind == LaneKind.AUDIO }
                .flatMap { lane -> lane.clips }
                .associate { clip -> clip.id to clip.beatsUs }
                .filterValues { it.isNotEmpty() }
        } catch (error: Exception) {
            show(error)
        }
        val id = selectedClip
        if (id != null && lanes.none { lane -> lane.clips.any { it.id == id } }) {
            selectedClip = null
        }
        playheadUs = playheadUs.coerceAtMost(durationUs)
        refreshStage()
        draw()
    }

    // Strip

    /** Zoom by `factor` for a strip `width` dp wide. */
    fun zoom(start: Double, factor: Double, width: Double) {
        val editor = editor ?: return
        runCatching { editor.zoom(start, factor, width) }.onSuccess { pixelSize = it }
    }

    /**
     * Read the ticks and tiles when the strip, `halfWidth` dp each side
     * of the playhead, leaves the window last read, the zoom changes, or the
     * cut changes. The window reaches a screen past each side.
     */
    fun refreshStrip(halfWidth: Double) {
        val editor = editor ?: return
        if (halfWidth <= 0) return
        lastHalfWidth = halfWidth
        val reach = (halfWidth / pixelSize).toLong()
        val from = playheadUs - reach
        val to = playheadUs + reach
        stripWindow?.let { window ->
            if (window.pixelSize == pixelSize && window.edit == edits && window.from <= from && window.to >= to) return
        }
        val wideFrom = (playheadUs - 3 * reach).coerceAtLeast(0)
        val wideTo = playheadUs + 3 * reach
        stripWindow = StripWindow(wideFrom, wideTo, pixelSize, edits)
        try {
            ticks = editor.ruler(pixelSize, wideFrom, wideTo)
            films = editor.film(wideFrom, wideTo, pixelSize, FILM_HEIGHT)
                .associate { it.clipId to it.tiles }
            waves = editor.waves(wideFrom, wideTo, pixelSize).associateBy { it.clipId }
        } catch (error: Exception) {
            show(error)
            return
        }
        fetchThumbs()
    }

    /** Fetch the stored pictures not yet held. While some are still being decoded, read the tiles again shortly. */
    private fun fetchThumbs() {
        val editor = editor ?: return
        val tiles = films.values.flatten()
        val wanted = tiles.filter { it.ready && thumbs[it.key] == null && it.key !in fetching }
        for (tile in wanted) {
            val key = tile.key
            fetching += key
            scope.launch {
                val image = withContext(Dispatchers.Default) {
                    runCatching { editor.thumb(key)?.bitmap() }.getOrNull()
                }
                fetching -= key
                if (image != null) thumbs[key] = image
            }
        }
        // About 180 KB a tile at the worker's height.
        if (thumbs.size > 300) thumbs.clear()
        stripRetry?.cancel()
        if (tiles.any { !it.ready } || waves.values.any { !it.ready }) {
            stripRetry = scope.launch {
                delay(250)
                stripWindow = null
                refreshStrip(lastHalfWidth)
            }
        }
    }

    // Picture

    private fun draw() {
        waiting = playheadUs
        val editor = editor ?: return
        if (drawing) return
        val time = waiting ?: return
        waiting = null
        drawing = true
        val box = placed
        scope.launch {
            val result = withContext(Dispatchers.Default) {
                runCatching {
                    val frame = editor.frame(time)
                    Triple(frame.bitmap(), frame.width.toFloat(), frame.height.toFloat())
                }
            }
            result.onSuccess { (image, width, height) ->
                if (image != null) {
                    picture = image
                    canvasAspect = width / height.coerceAtLeast(1f)
                }
            }.onFailure { show(it) }
            if (box.first == selectedClip) stage = box.second
            drawing = false
            if (waiting != null) draw()
        }
    }

    companion object {
        /** Tile height the filmstrip asks for, in dp. */
        const val FILM_HEIGHT = 52.0
    }
}

/** A clip's fade in and fade out. Zero is no fade. */
data class Fade(val inUs: Long, val outUs: Long)

val Clip.endUs: Long get() = startUs + durationUs

fun Clip.covers(time: Long): Boolean = time in startUs until endUs

/** The premultiplied RGBA bytes as an image, or null when the size and bytes disagree. */
fun Frame.bitmap(): ImageBitmap? {
    val width = width.toInt()
    val height = height.toInt()
    if (width <= 0 || height <= 0 || rgba.size != width * height * 4) return null
    // ARGB_8888 keeps its bytes in R, G, B, A order in memory.
    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(rgba))
    return bitmap.asImageBitmap()
}
