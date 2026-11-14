package com.scytheralpha.cutlass_android

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.TextAutoSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import uniffi.cutlass_ffi_bindings.CanvasRatio

/** A ratio tile's width, and the gap between tiles. */
private val ratioTileWidth = 60.dp
private val ratioTileGap = 6.dp

/** The square each ratio's frame is fitted in. */
private const val RATIO_BOX = 34f

/**
 * The canvas ratios in a row, each drawn at its own shape. A tap sets the
 * ratio at once, as one undo step, and the preview takes the new frame.
 */
@Composable
fun RatioPanel(model: EditorModel, done: () -> Unit) {
    val scroll = rememberScrollState()
    val density = LocalDensity.current
    var viewport by remember { mutableIntStateOf(0) }
    var centered by remember { mutableStateOf(false) }

    PanelFrame(title = "Ratio", close = done) {
        Row(
            Modifier
                .fillMaxWidth()
                .onSizeChanged { viewport = it.width }
                .horizontalScroll(scroll),
            horizontalArrangement = Arrangement.spacedBy(ratioTileGap),
        ) {
            for (ratio in model.ratios) {
                RatioTile(ratio) { model.setRatio(ratio.ratio) }
            }
        }
    }

    // Open with the chosen ratio in the middle.
    LaunchedEffect(viewport, model.ratios) {
        if (centered || viewport <= 0) return@LaunchedEffect
        val index = model.ratios.indexOfFirst { it.chosen }
        if (index < 0) return@LaunchedEffect
        centered = true
        val middle = with(density) {
            (ratioTileWidth + ratioTileGap).toPx() * index + ratioTileWidth.toPx() / 2
        }
        scroll.scrollTo((middle - viewport / 2f).toInt().coerceAtLeast(0))
    }
}

/** One ratio: its frame outlined inside a square, and its name under it. */
@Composable
private fun RatioTile(ratio: CanvasRatio, tap: () -> Unit) {
    val ink = if (ratio.chosen) Palette.accent else Palette.text
    // The ratio's frame fitted in the box, never thinner than a few points.
    val width = maxOf(ratio.width, 1).toFloat()
    val height = maxOf(ratio.height, 1).toFloat()
    val scale = RATIO_BOX / maxOf(width, height)

    Column(
        Modifier
            .size(width = ratioTileWidth, height = 64.dp)
            .background(Palette.white(if (ratio.chosen) 0.2 else 0.15), RoundedCornerShape(6.dp))
            .clickable(onClick = tap)
            .semantics {
                contentDescription = ratio.label
                selected = ratio.chosen
            },
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(6.dp, Alignment.CenterVertically),
    ) {
        Box(Modifier.size(RATIO_BOX.dp), contentAlignment = Alignment.Center) {
            Box(
                Modifier
                    .size(
                        width = maxOf(width * scale, 6f).dp,
                        height = maxOf(height * scale, 6f).dp,
                    )
                    .border(1.5.dp, ink, RoundedCornerShape(3.dp)),
            )
        }
        Text(
            ratio.label,
            color = ink,
            maxLines = 1,
            autoSize = TextAutoSize.StepBased(minFontSize = 7.7.sp, maxFontSize = 11.sp),
        )
    }
}
