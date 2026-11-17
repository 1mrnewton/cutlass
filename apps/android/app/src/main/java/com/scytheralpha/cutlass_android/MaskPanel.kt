package com.scytheralpha.cutlass_android

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import uniffi.cutlass_ffi_bindings.ClipMask
import uniffi.cutlass_ffi_bindings.MaskHandles

/** What a finger on the preview does while the Mask panel is open. */
enum class MaskTool { HANDLES, PAINT, ERASE, PEN }

/**
 * The selected clip's masks: a tile per shape adds one, a chip per mask
 * takes it in hand, and the rows under them change the mask in hand. Rust
 * keeps the list; the masks combine in order, each by its own mode.
 */
@Composable
fun MaskPanel(model: EditorModel, done: () -> Unit) {
    PanelFrame(title = "Mask", close = done) {
        Row(
            Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            for (shape in model.maskShapeList) {
                Column(
                    Modifier
                        .size(width = 64.dp, height = 56.dp)
                        .clickable { model.addMask(shape.key) }
                        .semantics { contentDescription = "Add ${shape.label} mask" },
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(4.dp, Alignment.CenterVertically),
                ) {
                    MaskGlyph(shape.key, 24.dp.value)
                    Text(shape.label, fontSize = 11.sp, maxLines = 1, color = Color.White)
                }
            }
        }
        if (model.masks.isNotEmpty()) {
            MaskList(model)
        }
        model.maskInHand?.let { MaskSettings(model, it) }
    }
}

@Composable
private fun MaskList(model: EditorModel) {
    Row(
        Modifier.fillMaxWidth().height(36.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(
            Modifier
                .weight(1f)
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            model.masks.forEachIndexed { index, mask ->
                val chosen = index == model.maskIndex
                Row(
                    Modifier
                        .height(30.dp)
                        .background(
                            if (chosen) Palette.accent.copy(alpha = 0.25f) else Palette.white(0.08),
                            CircleShape,
                        )
                        .border(1.dp, if (chosen) Palette.accent else Color.Transparent, CircleShape)
                        .clickable { model.chooseMask(index) }
                        .padding(horizontal = 10.dp)
                        .semantics {
                            contentDescription = "Mask ${index + 1}, ${mask.shape}"
                            selected = chosen
                        },
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    MaskGlyph(mask.shape, 14.dp.value)
                    Text("${index + 1}", fontSize = 12.sp, color = Color.White)
                }
            }
        }
        Box(Modifier.alpha(if (model.maskIndex > 0) 1f else 0.3f)) {
            IconButton("chevronLeft", "Move up", enabled = model.maskIndex > 0) { model.moveMask(up = true) }
        }
        val last = model.maskIndex + 1 >= model.masks.size
        Box(Modifier.alpha(if (last) 0.3f else 1f)) {
            IconButton("chevron", "Move down", enabled = !last) { model.moveMask(up = false) }
        }
        IconButton("delete", "Delete mask") { model.removeMask() }
    }
}

@Composable
private fun MaskSettings(model: EditorModel, mask: ClipMask) {
    Row(
        Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Choices(
            listOf("add" to "Add", "subtract" to "Subtract", "intersect" to "Intersect"),
            chosen = mask.mode,
            label = "Combine",
            modifier = Modifier.weight(1f),
        ) { model.editMask(mask.copy(mode = it), live = false) }
        Chip("Invert", mask.invert) { model.editMask(mask.copy(invert = !mask.invert), live = false) }
    }
    MaskSlider("Feather", mask.feather, 0.0..1.0, { "${(it * 100).roundToInt()}" }) { value, live ->
        model.editMask(mask.copy(feather = value), live)
    }
    if (mask.shape == "rectangle") {
        MaskSlider("Round", mask.round, 0.0..1.0, { "${(it * 100).roundToInt()}" }) { value, live ->
            model.editMask(mask.copy(round = value), live)
        }
    }
    MaskSlider("Rotate", mask.rotation, -180.0..180.0, { "${it.roundToInt()}°" }) { value, live ->
        model.editMask(mask.copy(rotation = value), live)
    }
    when (mask.shape) {
        "text" -> TextRow(model, mask)
        "brush" -> BrushRow(model)
        "pen" -> PenRow(model)
    }
}

@Composable
private fun TextRow(model: EditorModel, mask: ClipMask) {
    var words by remember(model.maskIndex) { mutableStateOf(mask.text ?: "") }
    Row(
        Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        OutlinedTextField(
            value = words,
            onValueChange = { words = it },
            singleLine = true,
            placeholder = { Text("Text") },
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
            keyboardActions = KeyboardActions(onDone = { if (words.isNotEmpty()) model.setMaskText(words) }),
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor = Color.White,
                unfocusedTextColor = Color.White,
                focusedBorderColor = Palette.accent,
                cursorColor = Palette.accent,
            ),
            modifier = Modifier
                .weight(1f)
                .semantics { contentDescription = "Mask text" },
        )
        Chip("Set", false, enabled = words.isNotEmpty()) { model.setMaskText(words) }
    }
}

@Composable
private fun BrushRow(model: EditorModel) {
    Choices(
        listOf(MaskTool.HANDLES to "Move", MaskTool.PAINT to "Paint", MaskTool.ERASE to "Erase"),
        chosen = model.maskTool,
        label = "Brush",
        modifier = Modifier.fillMaxWidth(),
    ) { model.maskTool = it }
    MaskSlider("Size", model.maskBrushSize, 4.0..200.0, { "${it.roundToInt()}" }) { value, _ ->
        model.maskBrushSize = value
    }
}

@Composable
private fun PenRow(model: EditorModel) {
    Row(
        Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        if (model.maskTool == MaskTool.PEN) {
            Text(
                "${model.penPoints.size} points",
                fontSize = 13.sp,
                color = Color.White.copy(alpha = 0.7f),
                modifier = Modifier.weight(1f),
            )
            Chip("Cancel", false) {
                model.penPoints = emptyList()
                model.maskTool = MaskTool.HANDLES
            }
            Chip("Done", false, enabled = model.penPoints.size >= 3) { model.finishOutline() }
        } else {
            Chip("Draw outline", false) {
                model.penPoints = emptyList()
                model.maskTool = MaskTool.PEN
            }
        }
    }
}

/** A row of choices, one of them chosen, like a segmented control. */
@Composable
private fun <T> Choices(
    choices: List<Pair<T, String>>,
    chosen: T,
    label: String,
    modifier: Modifier,
    choose: (T) -> Unit,
) {
    Row(
        modifier
            .height(34.dp)
            .background(Palette.white(0.08), RoundedCornerShape(8.dp))
            .padding(2.dp)
            .semantics { contentDescription = label },
    ) {
        for ((value, name) in choices) {
            val on = value == chosen
            Box(
                Modifier
                    .weight(1f)
                    .fillMaxSize()
                    .background(if (on) Palette.white(0.22) else Color.Transparent, RoundedCornerShape(6.dp))
                    .clickable { choose(value) }
                    .semantics {
                        contentDescription = name
                        selected = on
                    },
                contentAlignment = Alignment.Center,
            ) { Text(name, fontSize = 13.sp, color = Color.White) }
        }
    }
}

@Composable
private fun Chip(name: String, on: Boolean, enabled: Boolean = true, tap: () -> Unit) {
    Box(
        Modifier
            .height(34.dp)
            .alpha(if (enabled) 1f else 0.4f)
            .background(if (on) Palette.accent.copy(alpha = 0.3f) else Palette.white(0.08), RoundedCornerShape(8.dp))
            .clickable(enabled = enabled, onClick = tap)
            .padding(horizontal = 12.dp)
            .semantics {
                contentDescription = name
                selected = on
            },
        contentAlignment = Alignment.Center,
    ) { Text(name, fontSize = 13.sp, color = Color.White) }
}

/** A labelled slider; `change` hears each move as live and the release as not. */
@Composable
private fun MaskSlider(
    name: String,
    stored: Double,
    range: ClosedFloatingPointRange<Double>,
    shown: (Double) -> String,
    change: (Double, Boolean) -> Unit,
) {
    var value by remember { mutableDoubleStateOf(stored) }
    var held by remember { mutableStateOf(false) }
    LaunchedEffect(stored) { if (!held) value = stored }
    Row(
        Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(name, fontSize = 13.sp, color = Color.White.copy(alpha = 0.7f), modifier = Modifier.width(64.dp))
        Slider(
            value = value.toFloat(),
            onValueChange = {
                held = true
                value = it.toDouble()
                change(value, true)
            },
            onValueChangeFinished = {
                held = false
                change(value, false)
            },
            valueRange = range.start.toFloat()..range.endInclusive.toFloat(),
            colors = SliderDefaults.colors(
                thumbColor = Palette.accent,
                activeTrackColor = Palette.accent,
                inactiveTrackColor = Palette.white(0.3),
            ),
            modifier = Modifier
                .weight(1f)
                .semantics {
                    contentDescription = name
                    stateDescription = shown(value)
                },
        )
        Text(shown(value), fontSize = 12.sp, color = Color.White.copy(alpha = 0.5f), modifier = Modifier.width(40.dp))
    }
}

/** A small drawing of a mask shape, as the tiles and chips show it. */
@Composable
private fun MaskGlyph(shape: String, size: Float) {
    when (shape) {
        "text" -> HugeIcon("text", size = size.dp)
        "brush" -> HugeIcon("draw", size = size.dp)
        "pen" -> HugeIcon("pen", size = size.dp)
        "star" -> HugeIcon("star", size = size.dp)
        else -> Canvas(Modifier.size(size.dp)) { drawShape(shape) }
    }
}

private fun DrawScope.drawShape(shape: String) {
    val line = Stroke(width = 1.6.dp.toPx(), cap = StrokeCap.Round)
    val w = this.size.width
    val h = this.size.height
    val ink = Color.White
    when (shape) {
        "linear" -> drawLine(ink, Offset(w * 0.15f, h * 0.85f), Offset(w * 0.85f, h * 0.15f), line.width, StrokeCap.Round)
        "mirror" -> {
            drawLine(ink, Offset(w * 0.15f, h * 0.38f), Offset(w * 0.85f, h * 0.38f), line.width, StrokeCap.Round)
            drawLine(ink, Offset(w * 0.15f, h * 0.62f), Offset(w * 0.85f, h * 0.62f), line.width, StrokeCap.Round)
        }
        "circle" -> drawCircle(ink, w * 0.4f, style = line)
        "rectangle" -> drawRect(ink, Offset(w * 0.1f, h * 0.22f), Size(w * 0.8f, h * 0.56f), style = line)
        "heart" -> {
            val path = Path().apply {
                moveTo(w * 0.5f, h * 0.85f)
                cubicTo(w * 0.05f, h * 0.55f, w * 0.1f, h * 0.12f, w * 0.5f, h * 0.3f)
                cubicTo(w * 0.9f, h * 0.12f, w * 0.95f, h * 0.55f, w * 0.5f, h * 0.85f)
                close()
            }
            drawPath(path, ink, style = line)
        }
        else -> drawCircle(ink, w * 0.15f)
    }
}

/**
 * The mask in hand on the preview: its box and knobs to drag, or a brush
 * or pen drawing on it. Rust does the geometry; this only turns view
 * pixels into canvas pixels.
 */
@Composable
fun MaskOverlay(model: EditorModel) {
    var stroke by remember { mutableStateOf(emptyList<Offset>()) }
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val canvas = model.canvasSize ?: (16.0 to 9.0)
        val rect = fitted(canvas, constraints.maxWidth.toFloat(), constraints.maxHeight.toFloat())
        val scale = (rect.width / canvas.first).toFloat()
        val fit = rememberUpdatedState(rect to scale)
        val toCanvas = { point: Offset ->
            val (area, by) = fit.value
            Offset((point.x - area.left) / by, (point.y - area.top) / by)
        }
        Box(
            Modifier
                .fillMaxSize()
                .semantics { contentDescription = "Mask on the preview" }
                .pointerInput(model) {
                    awaitEachGesture {
                        val down = awaitFirstDown(requireUnconsumed = false)
                        when (model.maskTool) {
                            MaskTool.PEN -> {
                                model.penPoints = model.penPoints + toCanvas(down.position)
                                down.consume()
                            }
                            MaskTool.PAINT, MaskTool.ERASE -> {
                                var points = listOf(down.position)
                                stroke = points
                                down.consume()
                                while (true) {
                                    val event = awaitPointerEvent()
                                    val change = event.changes.firstOrNull { it.id == down.id } ?: break
                                    if (!change.pressed) break
                                    points = points + change.position
                                    stroke = points
                                    change.consume()
                                }
                                stroke = emptyList()
                                model.paintMask(points.map(toCanvas), erase = model.maskTool == MaskTool.ERASE)
                            }
                            MaskTool.HANDLES -> {
                                if (!model.beginMaskDrag(toCanvas(down.position))) return@awaitEachGesture
                                down.consume()
                                while (true) {
                                    val event = awaitPointerEvent()
                                    val change = event.changes.firstOrNull { it.id == down.id } ?: break
                                    if (!change.pressed) break
                                    if (change.position != change.previousPosition) {
                                        model.dragMask(toCanvas(change.position))
                                    }
                                    change.consume()
                                }
                                model.endMaskDrag()
                            }
                        }
                    }
                },
        ) {
            val handles = model.maskHandles
            val pen = model.penPoints
            val trail = stroke
            val brush = model.maskBrushSize
            val tool = model.maskTool
            Canvas(Modifier.fillMaxSize()) {
                val view = { x: Double, y: Double ->
                    Offset(rect.left + x.toFloat() * scale, rect.top + y.toFloat() * scale)
                }
                if (handles != null && tool != MaskTool.PEN) drawHandles(handles, view)
                if (trail.size > 1) {
                    val path = Path().apply {
                        moveTo(trail[0].x, trail[0].y)
                        for (point in trail.drop(1)) lineTo(point.x, point.y)
                    }
                    drawPath(
                        path,
                        if (tool == MaskTool.ERASE) Color.Red.copy(alpha = 0.5f) else Palette.accent.copy(alpha = 0.5f),
                        style = Stroke(width = (brush * scale).toFloat(), cap = StrokeCap.Round),
                    )
                }
                if (pen.isNotEmpty()) {
                    val points = pen.map { view(it.x.toDouble(), it.y.toDouble()) }
                    val path = Path().apply {
                        moveTo(points[0].x, points[0].y)
                        for (point in points.drop(1)) lineTo(point.x, point.y)
                    }
                    drawPath(path, Palette.accent, style = Stroke(width = 1.5.dp.toPx()))
                    for (point in points) drawCircle(Palette.accent, 4.dp.toPx(), point)
                }
            }
        }
    }
}

private fun DrawScope.drawHandles(handles: MaskHandles, view: (Double, Double) -> Offset) {
    val knob = 6.dp.toPx()
    val dashed = Stroke(width = 1.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(8f, 6f)))
    val corners = handles.corners.chunked(2).filter { it.size == 2 }.map { view(it[0], it[1]) }
    if (handles.line.size >= 4) {
        val ends = handles.line.chunked(2).filter { it.size == 2 }.map { view(it[0], it[1]) }
        for (pair in ends.chunked(2).filter { it.size == 2 }) {
            drawLine(Color.White, pair[0], pair[1], 1.5.dp.toPx(), pathEffect = dashed.pathEffect)
        }
    } else if (corners.size == 4) {
        val path = Path().apply {
            moveTo(corners[0].x, corners[0].y)
            for (point in corners.drop(1)) lineTo(point.x, point.y)
            close()
        }
        drawPath(path, Color.White, style = dashed)
        for (point in corners) drawCircle(Color.White, knob, point)
    }
    drawCircle(Palette.white(0.6), knob, view(handles.middleX, handles.middleY))
    drawCircle(Palette.accent, knob, view(handles.turnX, handles.turnY))
    drawCircle(Color(0xFFFFCC00), knob, view(handles.featherX, handles.featherY))
}
