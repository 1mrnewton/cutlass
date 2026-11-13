package com.scytheralpha.cutlass_android

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.setProgress
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.math.BigDecimal
import java.math.MathContext
import java.util.Locale
import kotlin.math.abs
import kotlin.math.roundToLong
import uniffi.cutlass_ffi_bindings.Clip
import uniffi.cutlass_ffi_bindings.speedAt
import uniffi.cutlass_ffi_bindings.speedPosition
import uniffi.cutlass_ffi_bindings.volumeAt
import uniffi.cutlass_ffi_bindings.volumePosition

/**
 * A tool panel over the tool bar: a title, the control, and a check that
 * closes it. A value is sent once the finger lifts, so one drag is one
 * undo step.
 */
@Composable
internal fun PanelFrame(title: String, close: () -> Unit, content: @Composable ColumnScope.() -> Unit) {
    CompositionLocalProvider(LocalContentColor provides Color.White) {
        Column(
            Modifier
                .fillMaxWidth()
                .background(Palette.tools)
                .padding(start = 16.dp, end = 16.dp, top = 10.dp, bottom = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                Text(title, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
                Box(
                    Modifier
                        .align(Alignment.CenterEnd)
                        .size(width = 40.dp, height = 32.dp)
                        .clickable(onClick = close)
                        .semantics { contentDescription = "Done" },
                    contentAlignment = Alignment.Center,
                ) { HugeIcon("done", size = 20.dp) }
            }
            content()
        }
    }
}

/** Clip gain from silence to 1000%, unity in the middle of the slider. */
@Composable
fun VolumePanel(model: EditorModel, clip: Clip, done: () -> Unit) {
    var position by remember { mutableDoubleStateOf(volumePosition(clip.volume)) }
    LaunchedEffect(clip.volume) { position = volumePosition(clip.volume) }
    val percent = "${(volumeAt(position) * 100).roundToLong()}%"

    PanelFrame(title = "Volume", close = done) {
        Row(
            Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text("0", style = clipPanelDigits(12), color = Color.White.copy(alpha = 0.5f))
            Slider(
                value = position.toFloat(),
                onValueChange = { position = it.toDouble() },
                onValueChangeFinished = { model.setVolume(clip.id, volumeAt(position)) },
                valueRange = 0f..1f,
                colors = SliderDefaults.colors(
                    thumbColor = Palette.accent,
                    activeTrackColor = Palette.accent,
                    inactiveTrackColor = Palette.white(0.3),
                ),
                modifier = Modifier
                    .weight(1f)
                    .semantics {
                        contentDescription = "Volume"
                        stateDescription = percent
                    },
            )
            Text("1000", style = clipPanelDigits(12), color = Color.White.copy(alpha = 0.5f))
        }
        Text(percent, style = clipPanelDigits(13), color = Color.White.copy(alpha = 0.75f))
    }
}

/** The speeds labelled under the scale, the same distance apart. */
private val speedMarks = listOf(0.1, 1.0, 2.0, 5.0, 10.0, 100.0)

/** Space at each end of the scale, so its end labels fit. */
private val speedInset = 20.dp

/**
 * Constant speed from 0.1x to 100x on a scale whose marks sit the same
 * distance apart. The clip's new length shows while dragging.
 */
@Composable
fun SpeedPanel(model: EditorModel, clip: Clip, done: () -> Unit) {
    var position by remember { mutableDoubleStateOf(speedPosition(clip.speed)) }
    // The clip's length when the panel opened.
    val before = remember { clip.durationUs }
    LaunchedEffect(clip.speed) { position = speedPosition(clip.speed) }

    // The source length stays; the timeline length follows the speed.
    val source = clip.durationUs.toDouble() * clip.speed
    val newLength = (source / maxOf(speedAt(position), 0.01)).toLong()

    PanelFrame(title = "Speed", close = done) {
        Row(
            Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            val dim = Color.White.copy(alpha = 0.55f)
            Text("Duration", style = clipPanelDigits(13), color = dim)
            Text(speedSeconds(before), style = clipPanelDigits(13), color = dim)
            HugeIcon("arrowRight", size = 14.dp, tint = Palette.accent)
            Text(speedSeconds(newLength), style = clipPanelDigits(13))
            Spacer(Modifier.weight(1f))
            Text(
                speedLabel(speedAt(position)),
                style = clipPanelDigits(13, FontWeight.SemiBold),
            )
        }
        SpeedScale(
            position = position,
            move = { position = it },
            commit = { model.setSpeed(clip.id, speedAt(position)) },
        )
    }
}

/**
 * The scale: minor ticks, a major one at each mark with its label under
 * it, and a ringed thumb. A press anywhere moves the thumb there.
 */
@Composable
private fun SpeedScale(position: Double, move: (Double) -> Unit, commit: () -> Unit) {
    val measurer = rememberTextMeasurer()
    val labelStyle = clipPanelDigits(12).copy(color = Color.White.copy(alpha = 0.75f))
    val label = speedLabel(speedAt(position))
    val latestMove by rememberUpdatedState(move)
    val latestCommit by rememberUpdatedState(commit)
    Canvas(
        Modifier
            .fillMaxWidth()
            .height(58.dp)
            .semantics {
                contentDescription = "Speed"
                stateDescription = label
                progressBarRangeInfo = ProgressBarRangeInfo(position.toFloat(), 0f..1f)
                setProgress { target ->
                    val step = 0.2 / 7
                    val next = position + if (target > position) step else -step
                    move(next.coerceIn(0.0, 1.0))
                    commit()
                    true
                }
            }
            .pointerInput(Unit) {
                awaitEachGesture {
                    val down = awaitFirstDown(requireUnconsumed = false)
                    val inset = speedInset.toPx()
                    val width = size.width - 2 * inset
                    val snap = 8.dp.toPx()
                    fun follow(x: Float) {
                        if (width <= 0f) return
                        val fraction = ((x - inset) / width).coerceIn(0f, 1f).toDouble()
                        latestMove(speedSnapped(fraction, width, snap))
                    }
                    follow(down.position.x)
                    down.consume()
                    while (true) {
                        val change = awaitPointerEvent().changes.firstOrNull { it.id == down.id } ?: break
                        change.consume()
                        if (!change.pressed) break
                        follow(change.position.x)
                    }
                    latestCommit()
                }
            },
    ) {
        val inset = speedInset.toPx()
        val width = size.width - 2 * inset
        val minors = 35
        for (index in 0..minors) {
            val major = index % 7 == 0
            val x = inset + width * index / minors
            drawLine(
                color = Color.White.copy(alpha = if (major) 0.8f else 0.35f),
                start = Offset(x, (if (major) 6 else 10).dp.toPx()),
                end = Offset(x, (if (major) 26 else 22).dp.toPx()),
                strokeWidth = 1.5.dp.toPx(),
            )
        }
        for ((index, mark) in speedMarks.withIndex()) {
            val x = inset + width * index / (speedMarks.size - 1)
            val text = measurer.measure(speedLabel(mark), labelStyle)
            drawText(
                text,
                topLeft = Offset(
                    x - text.size.width / 2f,
                    size.height - 8.dp.toPx() - text.size.height / 2f,
                ),
            )
        }
        val thumb = Offset(inset + width * position.toFloat(), 4.dp.toPx() + 12.dp.toPx())
        drawCircle(Palette.tools, radius = 12.dp.toPx(), center = thumb)
        drawCircle(
            Palette.accent,
            radius = 12.dp.toPx() - 1.5.dp.toPx(),
            center = thumb,
            style = Stroke(width = 3.dp.toPx()),
        )
    }
}

/** A position within 8 points of a mark lands on the mark. */
private fun speedSnapped(position: Double, width: Float, snap: Float): Double {
    val gaps = (speedMarks.size - 1).toDouble()
    val nearest = Math.round(position * gaps) / gaps
    return if (abs(nearest - position) * width < snap) nearest else position
}

private fun speedSeconds(micros: Long): String =
    String.format(Locale.ROOT, "%.1fs", micros / 1_000_000.0)

/** `2×`, or `0.5×`: whole speeds without a point, others as C's `%g` writes them. */
private fun speedLabel(speed: Double): String {
    val text = if (speed == Math.rint(speed)) {
        speed.toLong().toString()
    } else {
        BigDecimal(speed).round(MathContext(6)).stripTrailingZeros().toPlainString()
    }
    return "$text×"
}

/** A size in points with figures of one width, as `monospacedDigit()` gives. */
private fun clipPanelDigits(size: Int, weight: FontWeight? = null) =
    TextStyle(fontSize = size.sp, fontWeight = weight, fontFeatureSettings = "tnum")

/** The longest fade the panel offers: half the clip, and never past ten seconds. */
private fun fadeLimit(clip: Clip): Long = minOf(clip.durationUs / 2, 10_000_000L).coerceAtLeast(0)

/** "0.5s", to a tenth of a second. */
private fun tenths(micros: Long): String = "${micros / 1_000_000}.${micros % 1_000_000 / 100_000}s"

/**
 * Fade in and fade out, each from none to half the clip. Each slider is
 * one undo step when it is let go.
 */
@Composable
fun FadePanel(model: EditorModel, clip: Clip, done: () -> Unit) {
    val limit = fadeLimit(clip)
    val stored = remember(clip.id, model.edits) { model.fade(clip.id) }
    var fadeIn by remember { mutableDoubleStateOf(stored.inUs.toDouble()) }
    var fadeOut by remember { mutableDoubleStateOf(stored.outUs.toDouble()) }
    LaunchedEffect(stored) {
        fadeIn = stored.inUs.toDouble()
        fadeOut = stored.outUs.toDouble()
    }
    val commit = { model.setFade(clip.id, fadeIn.toLong(), fadeOut.toLong()) }

    PanelFrame(title = "Fade", close = done) {
        for ((label, value, set) in listOf(
            Triple("Fade in", fadeIn) { v: Double -> fadeIn = v },
            Triple("Fade out", fadeOut) { v: Double -> fadeOut = v },
        )) {
            Row(
                Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text(label, style = clipPanelDigits(13), color = Color.White.copy(alpha = 0.75f))
                Slider(
                    value = value.toFloat(),
                    onValueChange = { set(it.toDouble()) },
                    onValueChangeFinished = commit,
                    valueRange = 0f..limit.toFloat().coerceAtLeast(1f),
                    enabled = limit > 0,
                    colors = SliderDefaults.colors(
                        thumbColor = Palette.accent,
                        activeTrackColor = Palette.accent,
                        inactiveTrackColor = Palette.white(0.3),
                    ),
                    modifier = Modifier
                        .weight(1f)
                        .semantics {
                            contentDescription = label
                            stateDescription = tenths(value.toLong())
                        },
                )
                Text(tenths(value.toLong()), style = clipPanelDigits(12), color = Color.White.copy(alpha = 0.5f))
            }
        }
    }
}
