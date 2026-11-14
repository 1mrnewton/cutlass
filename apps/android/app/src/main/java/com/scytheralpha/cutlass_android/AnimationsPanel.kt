package com.scytheralpha.cutlass_android

import androidx.compose.animation.core.animateDpAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.TextAutoSize
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.inset
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.util.Locale
import kotlinx.coroutines.delay
import uniffi.cutlass_ffi_bindings.AnimationPreset
import uniffi.cutlass_ffi_bindings.ClipAnimations

/**
 * The selected clip's In and Out animations: a tab for each side, None and
 * that side's presets in a row, and a length slider for the chosen one. A
 * tap sets the side at once, and letting go of the slider sets its length,
 * each as one undo step.
 */
@Composable
fun AnimationsPanel(model: EditorModel, done: () -> Unit) {
    var enter by remember { mutableStateOf(true) }
    // The slider's length while a finger is on it.
    val dragging = remember { mutableStateOf<Float?>(null) }

    AnimationsPanelFrame(title = "Animations", close = done) {
        Column(
            Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            AnimationsPanelSides(enter = enter) {
                if (it != enter) {
                    enter = it
                    dragging.value = null
                }
            }
            val animations = model.animations
            if (animations != null) {
                val side = animationsPanelSide(animations, enter)
                AnimationsPanelTiles(
                    model = model,
                    enter = enter,
                    chosen = side.first,
                    onTap = { dragging.value = null },
                )
                AnimationsPanelLength(
                    model = model,
                    animations = animations,
                    enter = enter,
                    dragging = dragging,
                )
            } else {
                Box(Modifier.height(64.dp), contentAlignment = Alignment.Center) {
                    Text(
                        "Select a picture or title to animate it",
                        fontSize = 13.sp,
                        color = Color.White.copy(alpha = 0.6f),
                    )
                }
            }
        }
    }
}

/** The chosen side's id and length; an empty id is None. */
private fun animationsPanelSide(animations: ClipAnimations, enter: Boolean): Pair<String, Long> =
    if (enter) animations.inId to animations.inUs else animations.outId to animations.outUs

/** A panel over the tool bar: a title, the control, and a check that closes it. */
@Composable
private fun AnimationsPanelFrame(title: String, close: () -> Unit, content: @Composable () -> Unit) {
    CompositionLocalProvider(LocalContentColor provides Color.White) {
        Column(
            Modifier
                .fillMaxWidth()
                .background(Palette.tools)
                .padding(start = 16.dp, end = 16.dp, top = 10.dp, bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                Text(title, fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = Color.White)
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

/** In and Out as a segmented control, the chosen side raised. */
@Composable
private fun AnimationsPanelSides(enter: Boolean, choose: (Boolean) -> Unit) {
    BoxWithConstraints(
        Modifier
            .fillMaxWidth()
            .height(32.dp)
            .background(Color(0x3D767680), RoundedCornerShape(8.dp))
            .padding(2.dp),
    ) {
        val half = maxWidth / 2
        val knob by animateDpAsState(if (enter) 0.dp else half, label = "animationSide")
        Box(
            Modifier
                .offset(x = knob)
                .width(half)
                .fillMaxHeight()
                .background(Color(0xFF636366), RoundedCornerShape(7.dp)),
        )
        Row(Modifier.fillMaxSize()) {
            for ((label, side) in listOf("In" to true, "Out" to false)) {
                Box(
                    Modifier
                        .weight(1f)
                        .fillMaxHeight()
                        .clickable { choose(side) }
                        .semantics {
                            role = Role.Tab
                            selected = enter == side
                        },
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        label,
                        fontSize = 13.sp,
                        fontWeight = if (enter == side) FontWeight.SemiBold else FontWeight.Normal,
                        color = Color.White,
                    )
                }
            }
        }
    }
}

/** None, then the presets for the side, scrolling sideways. */
@Composable
private fun AnimationsPanelTiles(model: EditorModel, enter: Boolean, chosen: String, onTap: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        AnimationsPanelTile(name = "None", preset = null, chosen = chosen.isEmpty()) {
            onTap()
            model.setAnimation(enter, "", 0)
        }
        for (preset in model.animationList.filter { it.enter == enter }) {
            AnimationsPanelTile(name = preset.name, preset = preset, chosen = chosen == preset.id) {
                onTap()
                model.setAnimation(enter, preset.id, preset.durationUs)
            }
        }
    }
}

/**
 * The chosen animation's length, from a tenth of a second to what the clip
 * has left beside its other side.
 */
@Composable
private fun AnimationsPanelLength(
    model: EditorModel,
    animations: ClipAnimations,
    enter: Boolean,
    dragging: MutableState<Float?>,
) {
    val side = animationsPanelSide(animations, enter)
    val other = if (enter) animations.outUs else animations.inUs
    val longest = (maxOf(animations.lengthUs - other, 100_000L) / 1e6).toFloat()
    // The Material slider is taller than the iOS one; the empty row matches it so the panel keeps its height.
    Box(Modifier.fillMaxWidth().height(48.dp), contentAlignment = Alignment.CenterStart) {
        if (side.first.isNotEmpty() && longest > 0.1f) {
            val held = dragging.value
            val shown = (held ?: (side.second / 1e6).toFloat()).coerceIn(0.1f, longest)
            val commit = { seconds: Float ->
                model.setAnimation(enter, side.first, Math.round(seconds.toDouble() * 10) * 100_000L)
                dragging.value = null
            }
            Row(
                Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text(
                    String.format(Locale.ROOT, "%.1fs", shown),
                    fontSize = 13.sp,
                    style = TextStyle(fontFeatureSettings = "tnum"),
                    modifier = Modifier.width(40.dp),
                )
                Slider(
                    value = shown,
                    onValueChange = { dragging.value = it },
                    valueRange = 0.1f..longest,
                    onValueChangeFinished = { dragging.value?.let(commit) },
                    colors = SliderDefaults.colors(
                        thumbColor = Palette.accent,
                        activeTrackColor = Palette.accent,
                        inactiveTrackColor = Palette.white(0.3),
                    ),
                    modifier = Modifier.weight(1f),
                )
            }
            // As on iOS, a drag also lands once the finger rests.
            LaunchedEffect(held) {
                val seconds = held ?: return@LaunchedEffect
                delay(400)
                commit(seconds)
            }
        }
    }
}

/** One animation: a small drawing of its motion, and its name under it. */
@Composable
private fun AnimationsPanelTile(name: String, preset: AnimationPreset?, chosen: Boolean, onClick: () -> Unit) {
    val ink = if (chosen) Palette.accent else Palette.text
    Column(
        Modifier
            .size(64.dp)
            .clip(RoundedCornerShape(6.dp))
            .background(Palette.white(if (chosen) 0.2 else 0.15))
            .clickable(role = Role.Button, onClick = onClick)
            .semantics {
                contentDescription = name
                selected = chosen
            },
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(6.dp, Alignment.CenterVertically),
    ) {
        Box(Modifier.size(34.dp), contentAlignment = Alignment.Center) {
            AnimationsPanelGlyph(name = preset?.name ?: "", ink = ink)
        }
        Text(
            name,
            color = ink,
            maxLines = 1,
            autoSize = TextAutoSize.StepBased(minFontSize = 7.7.sp, maxFontSize = 11.sp, stepSize = 0.1.sp),
            modifier = Modifier
                .padding(horizontal = 2.dp)
                .clearAndSetSemantics {},
        )
    }
}

/**
 * A fade is a square going from clear to solid, a slide an arrow its way,
 * a zoom two nested squares, and None a cross.
 */
@Composable
private fun AnimationsPanelGlyph(name: String, ink: Color) {
    when {
        name.startsWith("Fade") -> Canvas(Modifier.fillMaxSize()) {
            val entering = name.endsWith("In")
            inset(5.dp.toPx()) {
                drawRoundRect(
                    brush = Brush.horizontalGradient(
                        listOf(
                            Color.White.copy(alpha = if (entering) 0.05f else 0.9f),
                            Color.White.copy(alpha = if (entering) 0.9f else 0.05f),
                        ),
                    ),
                    cornerRadius = CornerRadius(3.dp.toPx()),
                )
            }
        }
        name.startsWith("Slide") -> Canvas(Modifier.fillMaxSize()) {
            inset(6.dp.toPx()) {
                rotate(animationsPanelTurn(name)) { drawAnimationsPanelArrow(ink) }
            }
        }
        name.startsWith("Zoom") -> Canvas(Modifier.fillMaxSize()) {
            val line = Stroke(width = 1.5.dp.toPx())
            val corner = CornerRadius(2.dp.toPx())
            inset(4.dp.toPx()) {
                drawRoundRect(ink, cornerRadius = corner, style = line)
                inset(8.dp.toPx()) { drawRoundRect(ink, cornerRadius = corner, style = line) }
            }
        }
        else -> HugeIcon("close", size = 20.dp, tint = ink)
    }
}

/** The way a slide's arrow turns from pointing right, in degrees clockwise. */
private fun animationsPanelTurn(name: String): Float = when (name) {
    "Slide Left" -> 180f
    "Slide Up" -> -90f
    "Slide Down" -> 90f
    else -> 0f
}

/** An arrow pointing right across the drawing area. */
private fun DrawScope.drawAnimationsPanelArrow(ink: Color) {
    val width = size.width
    val height = size.height
    val path = Path().apply {
        moveTo(0f, height / 2)
        lineTo(width, height / 2)
        moveTo(width - width * 0.35f, height * 0.15f)
        lineTo(width, height / 2)
        lineTo(width - width * 0.35f, height - height * 0.15f)
    }
    drawPath(
        path,
        ink,
        style = Stroke(width = 1.8.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round),
    )
}
