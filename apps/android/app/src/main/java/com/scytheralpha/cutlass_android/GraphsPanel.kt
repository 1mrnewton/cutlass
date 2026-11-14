package com.scytheralpha.cutlass_android

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.TextAutoSize
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import uniffi.cutlass_ffi_bindings.GraphCurve
import uniffi.cutlass_ffi_bindings.graphCurves

/** Rust samples every curve once; they never change. */
private val graphsPanelCurves: List<GraphCurve> by lazy { graphCurves() }

/**
 * The keyframe curves in a row, each drawn as its shape. A tap eases the
 * span the playhead is in, as one undo step; None makes it straight again.
 */
@Composable
fun GraphsPanel(model: EditorModel, done: () -> Unit) {
    GraphsPanelFrame(title = "Graphs", close = done) {
        val keys = model.keys
        if (keys != null && keys.inSpan) {
            Row(
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                GraphsPanelTile(name = "None", curve = null, chosen = keys.graph.isEmpty()) {
                    model.easeKeyframes("")
                }
                for (curve in graphsPanelCurves) {
                    GraphsPanelTile(name = curve.name, curve = curve, chosen = keys.graph == curve.name) {
                        model.easeKeyframes(curve.name)
                    }
                }
            }
        } else {
            Box(Modifier.height(64.dp), contentAlignment = Alignment.Center) {
                Text(
                    "Put the playhead between two keyframes",
                    fontSize = 13.sp,
                    color = Color.White.copy(alpha = 0.6f),
                )
            }
        }
    }
}

/** A panel over the tool bar: a title, the control, and a check that closes it. */
@Composable
private fun GraphsPanelFrame(title: String, close: () -> Unit, content: @Composable () -> Unit) {
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

/** One curve in a square, its name under it. No curve draws a cross. */
@Composable
private fun GraphsPanelTile(name: String, curve: GraphCurve?, chosen: Boolean, onClick: () -> Unit) {
    val ink = if (chosen) Palette.accent else Palette.text
    Column(
        Modifier
            .size(width = 60.dp, height = 64.dp)
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
            if (curve == null) {
                HugeIcon("close", size = 20.dp, tint = ink)
            } else {
                GraphsPanelCurve(curve, ink)
            }
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
 * The curve in the box, y up. Rebounds leave the unit square, so the box
 * fits the curve's own height.
 */
@Composable
private fun GraphsPanelCurve(curve: GraphCurve, ink: Color) {
    Canvas(Modifier.fillMaxSize()) {
        val count = minOf(curve.xs.size, curve.ys.size)
        if (count == 0) return@Canvas
        val low = minOf(curve.ys.minOrNull() ?: 0.0, 0.0)
        val high = maxOf(curve.ys.maxOrNull() ?: 1.0, 1.0)
        val side = size.width
        fun x(index: Int) = (curve.xs[index] * side).toFloat()
        fun y(index: Int) = (side - (curve.ys[index] - low) / (high - low) * side).toFloat()
        val path = Path().apply {
            moveTo(x(0), y(0))
            for (index in 1 until count) lineTo(x(index), y(index))
        }
        drawPath(
            path,
            ink,
            style = Stroke(width = 1.5.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round),
        )
    }
}
