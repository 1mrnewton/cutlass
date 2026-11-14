package com.scytheralpha.cutlass_android

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateRotation
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.input.pointer.AwaitPointerEventScope
import androidx.compose.ui.input.pointer.PointerInputChange
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import uniffi.cutlass_ffi_bindings.Stage

/** The touch area of the delete button and the corner handle, in dp. */
private const val CONTROL = 32f

/** The canvas fitted in the overlay as the picture is drawn, and view pixels per canvas pixel. */
private data class StageFit(val rect: Rect, val scale: Float) {
    fun center(stage: Stage): Offset = Offset(
        rect.left + stage.centerX.toFloat() * scale,
        rect.top + stage.centerY.toFloat() * scale,
    )

    fun canvasPoint(point: Offset): Offset =
        Offset((point.x - rect.left) / scale, (point.y - rect.top) / scale)
}

/**
 * The selected picture or title's box on the preview, and the fingers
 * that move, pinch, and turn it. A tap selects the clip under it; a tap on
 * the selected title opens its words. Rust does the geometry and the
 * snapping; this only turns view pixels into canvas pixels.
 *
 * It sits in a box that is the canvas, so the canvas fits it exactly; the
 * fit is still worked out, as on iOS, in case the two ever differ.
 */
@Composable
fun StageOverlay(model: EditorModel, editText: (String) -> Unit) {
    var live by remember { mutableStateOf(false) }
    val density = LocalDensity.current
    val haptics = LocalHapticFeedback.current
    val edit = rememberUpdatedState(editText)
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val canvas = model.canvasSize ?: (16.0 to 9.0)
        val rect = fitted(canvas, constraints.maxWidth.toFloat(), constraints.maxHeight.toFloat())
        val fit = StageFit(rect, (rect.width / canvas.first).toFloat())
        val current = rememberUpdatedState(fit)
        val stage = shownStage(model)

        // Both centre lines together, so a haptic marks each new snap.
        val snapped = live && stage != null && (stage.verticalGuide || stage.horizontalGuide)
        LaunchedEffect(snapped) {
            if (snapped) haptics.performHapticFeedback(HapticFeedbackType.SegmentTick)
        }

        Box(
            Modifier
                .fillMaxSize()
                .pointerInput(model) {
                    awaitEachGesture {
                        val down = awaitFirstDown(requireUnconsumed = false)
                        val at = current.value
                        val shown = shownStage(model)
                        if (shown != null) {
                            val center = at.center(shown)
                            val box = boxSize(shown, at.scale)
                            val half = (CONTROL / 2).dp.toPx()
                            // The delete button has this finger.
                            if (near(down.position, corner(center, box, shown.turn, -1f), half)) {
                                return@awaitEachGesture
                            }
                            if (near(down.position, corner(center, box, shown.turn, 1f), half)) {
                                scaleAndTurn(model, down, center) { live = it }
                                return@awaitEachGesture
                            }
                        }
                        moveAndTwist(model, down, { current.value }, { live = it }) { point ->
                            tapAt(model, current.value, point) { edit.value(it) }
                        }
                    }
                },
        ) {
            if (stage != null) {
                val boxCenter = fit.center(stage)
                val box = with(density) { boxSize(stage, fit.scale) }
                val guides = live
                Canvas(Modifier.fillMaxSize()) {
                    val line = 1.dp.toPx()
                    if (guides && stage.verticalGuide) {
                        drawRect(
                            Palette.accent,
                            Offset(rect.center.x - line / 2, rect.top),
                            Size(line, rect.height),
                        )
                    }
                    if (guides && stage.horizontalGuide) {
                        drawRect(
                            Palette.accent,
                            Offset(rect.left, rect.center.y - line / 2),
                            Size(rect.width, line),
                        )
                    }
                    rotate(stage.turn.toFloat(), pivot = boxCenter) {
                        val topLeft = Offset(boxCenter.x - box.width / 2, boxCenter.y - box.height / 2)
                        drawRect(
                            Color.Black.copy(alpha = 0.3f),
                            topLeft,
                            box,
                            style = Stroke(3.5.dp.toPx()),
                        )
                        drawRect(Color.White, topLeft, box, style = Stroke(1.5.dp.toPx()))
                    }
                }
                StageControl(
                    icon = "close",
                    label = "Delete clip",
                    at = corner(boxCenter, box, stage.turn, -1f),
                    turn = stage.turn,
                    modifier = Modifier.clickable(interactionSource = null, indication = null) {
                        model.deleteClip()
                    },
                )
                StageControl(
                    icon = "rotate",
                    label = "Scale and turn",
                    at = corner(boxCenter, box, stage.turn, 1f),
                    turn = stage.turn,
                    modifier = Modifier,
                )
            }
        }
    }
}

/**
 * A white disc on a corner of the box, turned with it, in a touch square
 * that stays upright. The corner handle's fingers are read by the overlay.
 */
@Composable
private fun StageControl(icon: String, label: String, at: Offset, turn: Double, modifier: Modifier) {
    Box(
        Modifier
            .offset {
                val half = (CONTROL / 2).dp.toPx()
                IntOffset((at.x - half).roundToInt(), (at.y - half).roundToInt())
            }
            .size(CONTROL.dp)
            .then(modifier)
            .semantics { contentDescription = label },
        contentAlignment = Alignment.Center,
    ) {
        Box(
            Modifier
                .size(20.dp)
                .graphicsLayer { rotationZ = turn.toFloat() }
                .background(Color.White, CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            HugeIcon(icon, size = 12.dp, tint = Color.Black)
        }
    }
}

/** The selected clip's box, while the clip is on screen at the playhead. */
private fun shownStage(model: EditorModel): Stage? =
    if (model.selection?.covers(model.playheadUs) == true) model.stage else null

// Gestures

/**
 * A tap opens the selected title's words when it lands in its box,
 * padding too; anything else selects the clip under it.
 */
private fun Density.tapAt(model: EditorModel, fit: StageFit, position: Offset, editText: (String) -> Unit) {
    val point = fit.canvasPoint(position)
    val stage = shownStage(model)
    val clip = model.selectedClip
    if (stage != null && stage.text && clip != null &&
        contains(stage, point, boxSize(stage, fit.scale) / fit.scale)
    ) {
        editText(clip)
        return
    }
    model.pick(point)
}

/**
 * One finger moves the selected picture, or the one it starts on; two
 * pinch and turn the selected picture. As on iOS, a second finger ends the
 * move, so the fingers that pinch do not also drag. A finger that never
 * travels is a tap. Everything ends once every finger is up.
 */
private suspend fun AwaitPointerEventScope.moveAndTwist(
    model: EditorModel,
    down: PointerInputChange,
    fit: () -> StageFit,
    setLive: (Boolean) -> Unit,
    tap: (Offset) -> Unit,
) {
    val slop = 6.dp.toPx()
    var dragging = false
    var twisting = false
    var started = false
    var shift = Offset.Zero
    var pinch = 1.0
    var turn = 0.0
    try {
        while (true) {
            val event = awaitPointerEvent()
            val pressed = event.changes.filter { it.pressed }
            if (pressed.isEmpty()) {
                if (!dragging && !twisting && event.changes.none { it.isConsumed }) tap(down.position)
                break
            }
            if (pressed.size >= 2) {
                if (!twisting) {
                    twisting = true
                    val clip = model.selectedClip
                    if (!started && shownStage(model) != null && clip != null) {
                        started = model.beginStage(clip)
                        setLive(started)
                    }
                }
                if (started) {
                    pinch *= event.calculateZoom()
                    turn += event.calculateRotation()
                    model.moveStage(shift.x.toDouble(), shift.y.toDouble(), pinch, turn)
                }
                event.changes.forEach { it.consume() }
                continue
            }
            if (twisting) continue
            val change = pressed.firstOrNull { it.id == down.id } ?: continue
            val moved = change.position - down.position
            if (!dragging) {
                if (moved.getDistance() < slop) continue
                dragging = true
                if (!started) {
                    started = start(model, fit(), down.position)
                    setLive(started)
                }
            }
            if (started) {
                shift = moved / fit().scale
                model.moveStage(shift.x.toDouble(), shift.y.toDouble(), pinch, turn)
                change.consume()
            }
        }
    } finally {
        if (started) model.endStage()
        setLive(false)
    }
}

/** The corner handle scales and turns about the box centre with one finger. */
private suspend fun AwaitPointerEventScope.scaleAndTurn(
    model: EditorModel,
    down: PointerInputChange,
    center: Offset,
    setLive: (Boolean) -> Unit,
) {
    val clip = model.selectedClip ?: return
    down.consume()
    if (!model.beginStage(clip)) return
    setLive(true)
    // Where the drag started, from the box centre, in view pixels.
    val from = down.position - center
    try {
        while (true) {
            val change = awaitPointerEvent().changes.firstOrNull { it.id == down.id } ?: break
            if (!change.pressed) break
            change.consume()
            if (from.getDistance() <= 1f) continue
            val to = change.position - center
            val pinch = (to.getDistance() / from.getDistance()).toDouble()
            val turn = (atan2(to.y, to.x) - atan2(from.y, from.x)) * 180.0 / PI
            model.moveStage(0.0, 0.0, pinch, turn)
        }
    } finally {
        model.endStage()
        setLive(false)
    }
}

/**
 * Begin on the selected picture when the finger starts in its box, else
 * on the picture under the finger.
 */
private fun Density.start(model: EditorModel, fit: StageFit, position: Offset): Boolean {
    val point = fit.canvasPoint(position)
    val stage = shownStage(model)
    val clip = if (stage != null && contains(stage, point, boxSize(stage, fit.scale) / fit.scale)) {
        model.selectedClip
    } else {
        model.pick(point)
        if (shownStage(model) == null) null else model.selectedClip
    }
    return clip != null && model.beginStage(clip)
}

// Geometry

/**
 * The box in view pixels. A title's picture is cut tight to its letters,
 * so it gets room around them, and at least enough that a finger on the
 * words misses the controls at its corners.
 */
private fun Density.boxSize(stage: Stage, scale: Float): Size {
    val width = stage.width.toFloat() * scale
    val height = stage.height.toFloat() * scale
    if (!stage.text) return Size(width, height)
    return Size(max(width + 16.dp.toPx(), 72.dp.toPx()), max(height + 16.dp.toPx(), 52.dp.toPx()))
}

/** The canvas fitted in the overlay, as the picture is drawn. */
private fun fitted(canvas: Pair<Double, Double>, width: Float, height: Float): Rect {
    val scale = min(width / canvas.first, height / canvas.second)
    val fittedWidth = (canvas.first * scale).toFloat()
    val fittedHeight = (canvas.second * scale).toFloat()
    return Rect(
        Offset((width - fittedWidth) / 2, (height - fittedHeight) / 2),
        Size(fittedWidth, fittedHeight),
    )
}

/** A corner of the turned box: `sign` -1 is the top left, 1 the bottom right. */
private fun corner(center: Offset, box: Size, turn: Double, sign: Float): Offset {
    val radians = turn * PI / 180
    val x = sign * box.width / 2
    val y = sign * box.height / 2
    return Offset(
        center.x + (x * cos(radians) - y * sin(radians)).toFloat(),
        center.y + (x * sin(radians) + y * cos(radians)).toFloat(),
    )
}

/** Whether `point` is in the upright square `half` pixels each side of `at`. */
private fun near(point: Offset, at: Offset, half: Float): Boolean =
    abs(point.x - at.x) <= half && abs(point.y - at.y) <= half

/** Whether `point` is in the box `size` canvas pixels about the clip's centre. */
private fun contains(stage: Stage, point: Offset, size: Size): Boolean {
    val radians = stage.turn * PI / 180
    val dx = point.x - stage.centerX
    val dy = point.y - stage.centerY
    val x = dx * cos(radians) + dy * sin(radians)
    val y = -dx * sin(radians) + dy * cos(radians)
    return abs(x) <= size.width / 2 && abs(y) <= size.height / 2
}
