package com.scytheralpha.cutlass_android

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.EaseOut
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.detectDragGesturesAfterLongPress
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.PointerInputScope
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChange
import androidx.compose.ui.input.pointer.util.VelocityTracker
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.Layout
import androidx.compose.ui.platform.LocalViewConfiguration
import androidx.compose.ui.platform.ViewConfiguration
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.zIndex
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import uniffi.cutlass_ffi_bindings.Clip
import uniffi.cutlass_ffi_bindings.Lane
import uniffi.cutlass_ffi_bindings.LaneKind
import uniffi.cutlass_ffi_bindings.Tile
import uniffi.cutlass_ffi_bindings.Wave

// Sizes are in dp, as the iOS strip's points. The model's pixelSize is dp
// per microsecond here, as it is points per microsecond on iOS, so the
// zoom, the ticks, the tiles and the coast match the iPhone's.

private const val RULER_HEIGHT = 26f
private const val LANE_GAP = 8f
private const val SQUARE = 52f

/** The band a video's sound draws in over the foot of its picture. */
private const val BAND = 16f

/** How far a snap reaches, in dp at this zoom. */
private const val SNAP = 10f

/** A keyframe mark's side. */
private const val DIAMOND = 11f

/**
 * How far past each side of the strip a clip is still drawn. A long clip
 * zoomed in is cut to the strip and this much more, so its cut ends never
 * show and its size stays inside what a layout can hold.
 */
private const val REACH = 40f

/** How long a press must last before a clip lifts, as on iOS. */
private const val HOLD_MILLIS = 300L

private val VIDEO_FILL = Color(30, 76, 81)

/** A clip off main being moved, and how far the finger has gone, in dp. */
private data class StripMove(val id: String, val offset: Offset)

/** The edge being trimmed and how far it has gone, in dp. */
private data class StripTrim(val id: String, val startEdge: Boolean, val dx: Float)

/** A main clip being reordered: where the finger went down and where it is, in strip dp. */
private data class StripHold(val id: String, val pressX: Float, val fingerX: Float)

/** The squares as the finger left them, kept while they fade out. */
private data class StripFold(val held: StripHold, val lane: Lane)

/** What the fingers are doing to the clips. */
private class StripState {
    var moving by mutableStateOf<StripMove?>(null)
    var trimming by mutableStateOf<StripTrim?>(null)
    var reordering by mutableStateOf<StripHold?>(null)
    var parting by mutableStateOf<StripFold?>(null)

    /** A grip has the finger, so the clip under it does not lift. */
    var gripHeld = false
}

/**
 * The rows under a fixed playhead in the middle of the strip.
 *
 * Dragging sideways moves time under the playhead, and a swipe let go
 * slides on and slows to a stop until a finger lands; a pinch zooms. Picture
 * clips show a filmstrip. A tap selects a clip and a tap on empty space
 * clears it. A selected clip shows its length and a grip at each end to
 * trim it. A long press on a main clip folds the main row into squares to
 * reorder; on any other row it moves the clip in time or to another row.
 * Rust places, snaps and packs when the finger lifts.
 */
@Composable
fun TimelineStrip(model: EditorModel, onAdd: () -> Unit, modifier: Modifier = Modifier) {
    val state = remember { StripState() }
    val base = LocalViewConfiguration.current
    val configuration = remember(base) {
        object : ViewConfiguration by base {
            override val longPressTimeoutMillis: Long get() = HOLD_MILLIS
        }
    }
    CompositionLocalProvider(LocalViewConfiguration provides configuration) {
        BoxWithConstraints(
            modifier
                .clipToBounds()
                .pointerInput(model) { scrubAndZoom(model) },
        ) {
            val width = maxWidth.value
            val middle = width / 2
            val origin = middle - model.stripX(model.playheadUs)
            LaunchedEffect(model.playheadUs, model.pixelSize, model.lanes, middle) {
                model.refreshStrip(middle.toDouble())
            }
            Ruler(model, origin, Modifier.fillMaxWidth().height(RULER_HEIGHT.dp))
            Column(
                Modifier
                    .fillMaxWidth()
                    .wrapContentHeight(Alignment.Top, unbounded = true)
                    .padding(top = (RULER_HEIGHT + 6).dp),
                verticalArrangement = Arrangement.spacedBy(LANE_GAP.dp),
            ) {
                model.lanes.forEachIndexed { index, lane ->
                    key(index) { LaneRow(model, state, lane, index, origin, width) }
                }
            }
            Box(
                Modifier
                    .offset(x = (middle - 1).dp)
                    .width(2.dp)
                    .fillMaxHeight()
                    .background(Color.White),
            )
            AddButton(
                enabled = !model.isOpening && !model.isImporting,
                onAdd = onAdd,
                modifier = Modifier.offset(x = (width - 52).dp, y = (RULER_HEIGHT + 12).dp),
            )
        }
    }
}

// Time and dp

private fun EditorModel.stripX(time: Long): Float = (time * pixelSize).toFloat()

private fun EditorModel.micros(points: Float): Long = (points / pixelSize).toLong()

private fun rowHeight(lane: Lane): Float = if (lane.main) 56f else 40f

// Ruler

/**
 * A label on each major tick and a dot on each minor one, as Rust laid
 * them out for this zoom.
 */
@Composable
private fun Ruler(model: EditorModel, origin: Float, modifier: Modifier) {
    val measurer = rememberTextMeasurer()
    val style = remember {
        TextStyle(color = Color.White.copy(alpha = 0.55f), fontSize = 10.sp, fontFeatureSettings = "tnum")
    }
    val ticks = model.ticks
    val pixelSize = model.pixelSize
    Canvas(modifier) {
        val unit = 1.dp.toPx()
        val width = size.width / unit
        val middle = size.height / 2
        // A dot is left out within 22 dp of a label, or 14 of the dot before it.
        val labels = ticks.filter { it.major }.map { origin + (it.timeUs * pixelSize).toFloat() }
        var last = Float.NEGATIVE_INFINITY
        for (tick in ticks) {
            val position = origin + (tick.timeUs * pixelSize).toFloat()
            if (position <= -40f || position >= width + 40f) continue
            if (!tick.major) {
                if (position - last < 14f) continue
                if (labels.any { abs(it - position) < 22f }) continue
                last = position
            }
            val label = tick.label
            if (tick.major && label != null) {
                val text = measurer.measure(label, style)
                drawText(
                    text,
                    topLeft = Offset(
                        position * unit - text.size.width / 2f,
                        middle - text.size.height / 2f,
                    ),
                )
            } else {
                drawCircle(
                    Color.White.copy(alpha = 0.3f),
                    radius = 1.5f * unit,
                    center = Offset(position * unit, middle),
                )
            }
        }
    }
}

// Rows

@Composable
private fun LaneRow(
    model: EditorModel,
    state: StripState,
    lane: Lane,
    index: Int,
    origin: Float,
    stripWidth: Float,
) {
    val height = rowHeight(lane)
    val held = state.reordering
    // The held clip's row stays under the squares: it owns the gesture,
    // which ends only while it is there.
    val folded = lane.main && held != null
    val shown by animateFloatAsState(
        if (folded) 0f else 1f,
        tween(180, easing = EaseOut),
        label = "fold",
    )
    val movingIn = state.moving?.let { moving -> lane.clips.any { it.id == moving.id } } == true
    Box(
        Modifier
            .zIndex(if (movingIn) 1f else 0f)
            .fillMaxWidth()
            .height(height.dp),
    ) {
        FreeRow(Modifier.fillMaxSize().alpha(shown)) {
            for (clip in lane.clips) {
                key(clip.id) { ClipView(model, state, clip, lane, index, origin, stripWidth) }
            }
            if (lane.main && lane.clips.isEmpty()) {
                Text(
                    "Tap + to add a photo or video",
                    color = Color.White.copy(alpha = 0.45f),
                    fontSize = 13.sp,
                    modifier = Modifier
                        .offset(x = (origin + 12).dp)
                        .height(height.dp)
                        .wrapContentHeight(),
                )
            }
        }
        if (lane.main) {
            AnimatedVisibility(
                visible = folded,
                enter = fadeIn(tween(180, easing = EaseOut)),
                exit = fadeOut(tween(180, easing = EaseOut)),
            ) {
                val fold = state.reordering?.let { StripFold(it, lane) } ?: state.parting
                if (fold != null) Squares(model, fold, Modifier.fillMaxSize())
            }
        }
    }
}

/**
 * A row whose children keep the size they ask for and sit at their own
 * offset, even past its edges, as a SwiftUI ZStack lets them.
 */
@Composable
private fun FreeRow(modifier: Modifier, content: @Composable () -> Unit) {
    Layout(content, modifier) { measurables, constraints ->
        val placeables = measurables.map { it.measure(Constraints()) }
        layout(constraints.maxWidth, constraints.maxHeight) {
            placeables.forEach { it.place(0, 0) }
        }
    }
}

@Composable
private fun ClipView(
    model: EditorModel,
    state: StripState,
    clip: Clip,
    lane: Lane,
    index: Int,
    origin: Float,
    stripWidth: Float,
) {
    val height = rowHeight(lane)
    val selected = model.selectedClip == clip.id
    var left = origin + model.stripX(clip.startUs)
    var width = model.stripX(clip.durationUs)
    var shownUs = clip.durationUs
    var trimmedLeft = 0f
    val trim = state.trimming?.takeIf { it.id == clip.id }
    if (trim != null) {
        if (trim.startEdge) {
            val dx = min(trim.dx, width - 4f)
            left += dx
            width -= dx
            trimmedLeft = dx
        } else {
            width = max(width + trim.dx, 4f)
        }
        shownUs = model.micros(width)
    }
    val moving = state.moving?.takeIf { it.id == clip.id }
    val moved = moving?.offset ?: Offset.Zero
    // The whole clip, then the part of it drawn: the strip and a little more.
    val clipLeft = left + 1 + moved.x
    val clipWidth = max(width - 2, 2f)
    val from = max(clipLeft, -REACH)
    val to = min(clipLeft + clipWidth, stripWidth + REACH)
    val cut = from - clipLeft
    val drawn = max(to - from, 0f)
    val drawnLeft = rememberUpdatedState(from)

    val gestures = when {
        lane.locked -> Modifier
        lane.main -> Modifier.pointerInput(clip) { reorder(model, state, clip) { drawnLeft.value } }
        else -> Modifier.pointerInput(clip, index) { move(model, state, clip, index) }
    }
    Box(
        Modifier
            .offset { IntOffset(from.dp.roundToPx(), moved.y.dp.roundToPx()) }
            .size(drawn.dp, height.dp)
            .graphicsLayer {
                if (moving != null) {
                    scaleX = 1.04f
                    scaleY = 1.04f
                    shadowElevation = 6.dp.toPx()
                    shape = RoundedCornerShape(6.dp)
                }
            }
            .pointerInput(clip.id) {
                detectTapGestures(
                    onTap = {
                        model.stopCoast()
                        model.selectedClip = clip.id
                    },
                )
            }
            .then(gestures),
    ) {
        ClipBlock(
            clip = clip,
            kind = lane.kind,
            selected = selected,
            height = height,
            tiles = model.films[clip.id].orEmpty(),
            thumbs = model.thumbs,
            wave = model.waves[clip.id],
            shift = trimmedLeft + cut,
            nameLeft = -cut,
            nameWidth = clipWidth,
            modifier = Modifier.fillMaxSize(),
        )
        if (selected) {
            Text(
                String.format(Locale.ROOT, "%.1fs", shownUs / 1_000_000.0),
                color = Color.White,
                fontSize = 10.sp,
                fontWeight = FontWeight.SemiBold,
                style = TextStyle(fontFeatureSettings = "tnum"),
                maxLines = 1,
                softWrap = false,
                modifier = Modifier
                    .offset(x = (-cut + if (lane.locked) 4f else 16f).dp, y = 4.dp)
                    .background(Color.Black.copy(alpha = 0.55f), RoundedCornerShape(3.dp))
                    .padding(horizontal = 4.dp, vertical = 1.dp),
            )
        }
        if (selected && !lane.locked) {
            Grip(model, state, clip, startEdge = true, Modifier.offset(x = (-cut).dp))
            Grip(model, state, clip, startEdge = false, Modifier.offset(x = (-cut + clipWidth - 20).dp))
        }
        val keys = model.keys
        if (selected && keys != null) {
            for (time in keys.timesUs) {
                KeyDiamond(
                    Modifier
                        .align(Alignment.CenterStart)
                        .offset(x = (model.stripX(time - clip.startUs) - 1 - DIAMOND / 2 - cut).dp),
                )
            }
        }
    }
}

/**
 * A white tab on one end of the selected clip. Dragging it trims that end.
 * It takes the finger at once, before the strip or the clip can.
 */
@Composable
private fun Grip(model: EditorModel, state: StripState, clip: Clip, startEdge: Boolean, modifier: Modifier) {
    val shape = if (startEdge) {
        RoundedCornerShape(topStart = 6.dp, bottomStart = 6.dp)
    } else {
        RoundedCornerShape(topEnd = 6.dp, bottomEnd = 6.dp)
    }
    // The touch area reaches 8 dp into the clip past the 12 dp tab.
    Box(
        modifier
            .width(20.dp)
            .fillMaxHeight()
            .pointerInput(clip, startEdge) { trimEdge(model, state, clip, startEdge) }
            .semantics { contentDescription = if (startEdge) "Trim start" else "Trim end" },
        contentAlignment = if (startEdge) Alignment.CenterStart else Alignment.CenterEnd,
    ) {
        Box(
            Modifier
                .width(12.dp)
                .fillMaxHeight()
                .background(Color.White, shape),
            contentAlignment = Alignment.Center,
        ) {
            Box(
                Modifier
                    .size(width = 2.dp, height = 16.dp)
                    .background(Color.Black.copy(alpha = 0.45f), CircleShape),
            )
        }
    }
}

/** The pinned add button over the main row. */
@Composable
private fun AddButton(enabled: Boolean, onAdd: () -> Unit, modifier: Modifier) {
    val shape = RoundedCornerShape(8.dp)
    Box(
        modifier
            .size(40.dp)
            .shadow(
                4.dp,
                shape,
                ambientColor = Color.Black.copy(alpha = 0.4f),
                spotColor = Color.Black.copy(alpha = 0.4f),
            )
            .background(Color.White, shape)
            .clip(shape)
            .clickable(enabled = enabled, onClick = onAdd)
            .semantics { contentDescription = "Add media" },
        contentAlignment = Alignment.Center,
    ) {
        HugeIcon("plus", size = 22.dp, tint = Color.Black)
    }
}

// Trim

private suspend fun PointerInputScope.trimEdge(
    model: EditorModel,
    state: StripState,
    clip: Clip,
    startEdge: Boolean,
) {
    awaitEachGesture {
        val down = awaitFirstDown(requireUnconsumed = false)
        down.consume()
        state.gripHeld = true
        var dx = 0f
        var dragged = false
        try {
            while (true) {
                val change = awaitPointerEvent().changes.firstOrNull { it.id == down.id } ?: break
                if (!change.pressed) {
                    change.consume()
                    break
                }
                val step = change.positionChange().x
                change.consume()
                if (!dragged && step == 0f) continue
                if (!dragged) {
                    dragged = true
                    model.pause()
                }
                dx += step / density
                state.trimming = StripTrim(clip.id, startEdge, dx)
            }
        } finally {
            state.gripHeld = false
            if (dragged) state.trimming = null
        }
        if (dragged) {
            val edge = if (startEdge) clip.startUs else clip.endUs
            model.trim(clip.id, startEdge, edge + model.micros(dx), model.micros(SNAP))
        }
    }
}

// Move off main

/** Hold, then drag a clip to another time or row. */
private suspend fun PointerInputScope.move(model: EditorModel, state: StripState, clip: Clip, row: Int) {
    detectDragGesturesAfterLongPress(
        onDragStart = {
            if (!state.gripHeld) {
                model.pause()
                model.selectedClip = clip.id
                state.moving = StripMove(clip.id, Offset.Zero)
            }
        },
        onDragEnd = {
            val moving = state.moving?.takeIf { it.id == clip.id }
            if (moving != null) {
                state.moving = null
                model.moveClip(
                    clip.id,
                    row = targetRow(model.lanes, row, moving.offset.y),
                    startUs = clip.startUs + model.micros(moving.offset.x),
                    snapUs = model.micros(SNAP),
                )
            }
        },
        onDragCancel = {
            if (state.moving?.id == clip.id) state.moving = null
        },
        onDrag = { change, amount ->
            val moving = state.moving?.takeIf { it.id == clip.id }
            if (moving != null) {
                change.consume()
                state.moving = moving.copy(offset = moving.offset + amount / density)
            }
        },
    )
}

/** The row a vertical drag of `dy` dp from `row` lands on. */
private fun targetRow(lanes: List<Lane>, row: Int, dy: Float): Int {
    if (row !in lanes.indices) return row
    var top = 0f
    for (index in 0 until row) top += rowHeight(lanes[index]) + LANE_GAP
    val finger = top + rowHeight(lanes[row]) / 2 + dy
    var edge = 0f
    for ((index, lane) in lanes.withIndex()) {
        edge += rowHeight(lane) + LANE_GAP
        if (finger < edge) return index
    }
    return lanes.size
}

// Reorder main

/**
 * Hold a main clip: the row folds into squares, the held one follows the
 * finger, and the others make room where it would land. `drawnLeft` is
 * where the clip's drawn box starts in the strip, in dp.
 */
private suspend fun PointerInputScope.reorder(
    model: EditorModel,
    state: StripState,
    clip: Clip,
    drawnLeft: () -> Float,
) {
    detectDragGesturesAfterLongPress(
        onDragStart = { at ->
            if (!state.gripHeld) {
                model.pause()
                model.selectedClip = clip.id
                val x = drawnLeft() + at.x / density
                state.reordering = StripHold(clip.id, x, x)
            }
        },
        onDragEnd = { landReorder(model, state, clip.id) },
        onDragCancel = {
            if (state.reordering?.id == clip.id) state.reordering = null
        },
        onDrag = { change, amount ->
            val held = state.reordering?.takeIf { it.id == clip.id }
            if (held != null) {
                change.consume()
                state.reordering = held.copy(fingerX = held.fingerX + amount.x / density)
            }
        },
    )
}

private fun landReorder(model: EditorModel, state: StripState, id: String) {
    val held = state.reordering?.takeIf { it.id == id } ?: return
    val main = model.lanes.firstOrNull { it.main }
    if (main == null) {
        state.reordering = null
        return
    }
    val place = landing(held, main)
    state.parting = StripFold(held, main)
    state.reordering = null
    if (place != main.clips.indexOfFirst { it.id == held.id }) model.reorder(held.id, place)
}

/** The place the held clip lands in the order without it. */
private fun landing(held: StripHold, lane: Lane): Int {
    val start = lane.clips.indexOfFirst { it.id == held.id }.coerceAtLeast(0)
    val steps = ((held.fingerX - held.pressX) / (SQUARE + 6)).roundToInt()
    return (start + steps).coerceIn(0, max(lane.clips.size - 1, 0))
}

@Composable
private fun Squares(model: EditorModel, fold: StripFold, modifier: Modifier) {
    val held = fold.held
    val lane = fold.lane
    val start = lane.clips.indexOfFirst { it.id == held.id }.coerceAtLeast(0)
    val place = landing(held, lane)
    val others = lane.clips.filter { it.id != held.id }
    val shape = RoundedCornerShape(6.dp)
    FreeRow(modifier) {
        others.forEachIndexed { index, clip ->
            key(clip.id) {
                val slot = if (index < place) index else index + 1
                val x by animateDpAsState(
                    (held.pressX + (slot - start) * (SQUARE + 6) - SQUARE / 2).dp,
                    tween(150, easing = EaseOut),
                    label = "slot",
                )
                SquareClip(
                    clip,
                    model.films[clip.id]?.firstOrNull(),
                    model.thumbs,
                    Modifier
                        .offset(x = x, y = 2.dp)
                        .size(SQUARE.dp),
                )
            }
        }
        val clip = lane.clips.firstOrNull { it.id == held.id }
        if (clip != null) {
            SquareClip(
                clip,
                model.films[clip.id]?.firstOrNull(),
                model.thumbs,
                Modifier
                    .zIndex(1f)
                    .offset(x = (held.fingerX - SQUARE / 2).dp, y = 2.dp)
                    .graphicsLayer {
                        scaleX = 1.08f
                        scaleY = 1.08f
                        shadowElevation = 6.dp.toPx()
                        this.shape = shape
                    }
                    .size(SQUARE.dp)
                    .border(2.dp, Color.White, shape),
            )
        }
    }
}

// Scrub and zoom

/**
 * One finger scrubs and coasts when let go; two pinch. A finger landing
 * stops a coast. A tap nothing else took clears the selection. A clip, a
 * grip or the add button that took the finger keeps it.
 */
private suspend fun PointerInputScope.scrubAndZoom(model: EditorModel) {
    awaitEachGesture {
        val down = awaitFirstDown(requireUnconsumed = false)
        model.stopCoast()
        val tracker = VelocityTracker()
        tracker.addPosition(down.uptimeMillis, down.position)
        var scrubFrom: Long? = null
        var pinchFrom: Double? = null
        var pinchSpan = 0f
        var taken = false
        while (true) {
            val event = awaitPointerEvent()
            val pressed = event.changes.filter { it.pressed }
            if (pressed.isEmpty()) {
                if (scrubFrom != null) {
                    event.changes.firstOrNull { it.id == down.id }?.let {
                        tracker.addPosition(it.uptimeMillis, it.position)
                    }
                    model.coast((tracker.calculateVelocity().x / density).toDouble())
                } else if (!taken && pinchFrom == null && event.changes.none { it.isConsumed }) {
                    model.selectedClip = null
                }
                break
            }
            if (scrubFrom == null && event.changes.any { it.isConsumed }) taken = true
            if (taken) continue
            if (pressed.size >= 2) {
                val span = (pressed[0].position - pressed[1].position).getDistance()
                val start = pinchFrom ?: model.pixelSize.also {
                    pinchFrom = it
                    pinchSpan = span
                    scrubFrom = null
                }
                if (pinchSpan > 0f && span > 0f) {
                    model.zoom(start, (span / pinchSpan).toDouble(), (size.width / density).toDouble())
                }
                event.changes.forEach { it.consume() }
                continue
            }
            if (pinchFrom != null) continue
            val change = pressed.firstOrNull { it.id == down.id } ?: continue
            tracker.addPosition(change.uptimeMillis, change.position)
            if (scrubFrom == null) {
                if ((change.position - down.position).getDistance() < viewConfiguration.touchSlop) continue
                model.pause()
                scrubFrom = model.playheadUs
            }
            val from = scrubFrom ?: continue
            model.seek(from - model.micros((change.position.x - down.position.x) / density))
            change.consume()
        }
    }
}

// Clips

/**
 * One clip: a filmstrip for a picture, else its colour by kind and its
 * name, with a white frame when selected. `shift` is the dp cut off its
 * left, by a start trim and by the strip's edge; `nameLeft` and
 * `nameWidth` place the name on the whole clip.
 */
@Composable
private fun ClipBlock(
    clip: Clip,
    kind: LaneKind,
    selected: Boolean,
    height: Float,
    tiles: List<Tile>,
    thumbs: Map<String, ImageBitmap>,
    wave: Wave?,
    shift: Float,
    nameLeft: Float,
    nameWidth: Float,
    modifier: Modifier,
) {
    val shape = RoundedCornerShape(6.dp)
    Box(
        modifier
            .clip(shape)
            .background(fill(kind))
            .then(if (selected) Modifier.border(2.dp, Color.White, shape) else Modifier),
    ) {
        if (tiles.isEmpty()) {
            Text(
                clip.name,
                color = Color.White.copy(alpha = 0.9f),
                fontSize = 11.sp,
                fontWeight = FontWeight.Medium,
                maxLines = 1,
                softWrap = false,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier
                    .offset(x = (nameLeft + 6).dp, y = 4.dp)
                    .wrapContentWidth(Alignment.Start, unbounded = true)
                    .widthIn(max = (nameWidth - 12).coerceIn(0f, 4_000f).dp),
            )
        } else {
            Film(tiles, thumbs, shift, Modifier.fillMaxSize())
        }
        Sound(wave, height, shift, Modifier.align(Alignment.BottomStart))
    }
}

/** The tiles as Rust laid them out, each picture filling its tile. */
@Composable
private fun Film(tiles: List<Tile>, thumbs: Map<String, ImageBitmap>, shift: Float, modifier: Modifier) {
    Canvas(modifier) {
        val empty = Palette.white(0.16)
        for (tile in tiles) {
            val left = (tile.left.toFloat() - shift).dp.toPx()
            val width = tile.width.toFloat().dp.toPx()
            if (left + width < 0f || left > size.width) continue
            val image = thumbs[tile.key]
            if (image == null || image.width <= 0 || image.height <= 0) {
                drawRect(empty, Offset(left, 0f), Size(width, size.height))
                continue
            }
            val x = left.roundToInt()
            val drawnWidth = (left + width).roundToInt() - x
            val drawnHeight = size.height.roundToInt()
            if (drawnWidth <= 0 || drawnHeight <= 0) continue
            val fit = max(drawnWidth.toFloat() / image.width, drawnHeight.toFloat() / image.height)
            val sourceWidth = (drawnWidth / fit).roundToInt().coerceIn(1, image.width)
            val sourceHeight = (drawnHeight / fit).roundToInt().coerceIn(1, image.height)
            drawImage(
                image,
                srcOffset = IntOffset((image.width - sourceWidth) / 2, (image.height - sourceHeight) / 2),
                srcSize = IntSize(sourceWidth, sourceHeight),
                dstOffset = IntOffset(x, 0),
                dstSize = IntSize(drawnWidth, drawnHeight),
            )
        }
    }
}

/**
 * The clip's waveform: bars from the middle of an audio clip, from the
 * foot of a video's band. Dim when the row is muted.
 */
@Composable
private fun Sound(wave: Wave?, height: Float, shift: Float, modifier: Modifier) {
    if (wave == null) return
    val under = wave.underPicture
    val band = Color.Black.copy(alpha = 0.45f)
    if (wave.levels.isEmpty()) {
        if (under) Box(modifier.fillMaxWidth().height(BAND.dp).background(band))
        return
    }
    val tall = if (under) BAND else height
    Canvas(
        modifier
            .fillMaxWidth()
            .height(tall.dp)
            .then(if (under) Modifier.background(band) else Modifier),
    ) {
        val unit = 1.dp.toPx()
        val path = Path()
        val left = wave.first.toLong().toFloat() - shift
        val reach = if (under) size.height - 2 * unit else size.height * 0.8f
        wave.levels.forEachIndexed { index, level ->
            val x = left + index
            if (x >= -1f && x * unit <= size.width) {
                val bar = max(level * reach, unit)
                val top = if (under) size.height - bar else (size.height - bar) / 2
                path.addRect(Rect(x * unit, top, x * unit + unit, top + bar))
            }
        }
        drawPath(path, Color.White.copy(alpha = if (wave.muted) 0.25f else 0.7f))
    }
}

private fun fill(kind: LaneKind): Color = when (kind) {
    LaneKind.VIDEO -> VIDEO_FILL
    LaneKind.AUDIO -> Color(29, 54, 94)
    LaneKind.TEXT -> Color(120, 84, 40)
    else -> Palette.white(0.3)
}

/** A main clip folded to a square while the row is reordered. */
@Composable
private fun SquareClip(clip: Clip, tile: Tile?, thumbs: Map<String, ImageBitmap>, modifier: Modifier) {
    val image = tile?.let { thumbs[it.key] }
    Box(modifier.clip(RoundedCornerShape(6.dp))) {
        if (image != null) {
            Image(
                image,
                contentDescription = null,
                contentScale = ContentScale.Crop,
                modifier = Modifier.fillMaxSize(),
            )
        } else {
            Box(
                Modifier
                    .fillMaxSize()
                    .background(VIDEO_FILL)
                    .padding(3.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    clip.name,
                    color = Color.White.copy(alpha = 0.8f),
                    fontSize = 9.sp,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}

/** A keyframe's mark on the selected clip. */
@Composable
private fun KeyDiamond(modifier: Modifier) {
    Box(modifier.size(DIAMOND.dp), contentAlignment = Alignment.Center) {
        Box(
            Modifier
                .size((DIAMOND / 1.414f).dp)
                .rotate(45f)
                .background(Color(1f, 0.3f, 0.45f))
                .border(1.5.dp, Color.White),
        )
    }
}
