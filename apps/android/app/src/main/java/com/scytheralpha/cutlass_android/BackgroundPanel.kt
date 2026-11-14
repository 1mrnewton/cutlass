package com.scytheralpha.cutlass_android

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.BlurredEdgeTreatment
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import uniffi.cutlass_ffi_bindings.Background
import uniffi.cutlass_ffi_bindings.backgroundBlurs
import uniffi.cutlass_ffi_bindings.backgroundColors

/** The two kinds of fill the panel offers. */
private enum class BackgroundFillTab(val title: String) {
    COLOR("Color"),
    BLUR("Blur"),
}

private val backgroundSwatches: List<String> by lazy { backgroundColors() }
private val backgroundStrengths: List<Double> by lazy { backgroundBlurs() }

/** The side of every tile in the panel. */
private val backgroundTile = 52.dp

private val backgroundTileShape = RoundedCornerShape(8.dp)

/**
 * What fills the canvas around the main clips. Color sets the project
 * color; Blur puts every main clip over a blurred copy of itself. A tap
 * applies at once, as one undo step.
 */
@Composable
fun BackgroundPanel(model: EditorModel, done: () -> Unit) {
    var tab by remember {
        mutableStateOf(if (model.background is Background.Blur) BackgroundFillTab.BLUR else BackgroundFillTab.COLOR)
    }

    PanelFrame(title = "Background", close = done) {
        BackgroundTabs(tab) { tab = it }
        Row(
            Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState())
                // Room for the ring around the chosen tile.
                .padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            BackgroundNoneTile(chosen = model.background == Background.None) {
                model.setBackground(Background.None)
            }
            when (tab) {
                BackgroundFillTab.COLOR -> {
                    for (color in backgroundSwatches) {
                        val fill = Background.Color(color)
                        BackgroundSwatchTile(color = color, chosen = model.background == fill) {
                            model.setBackground(fill)
                        }
                    }
                }
                BackgroundFillTab.BLUR -> {
                    backgroundStrengths.forEachIndexed { index, blur ->
                        val fill = Background.Blur(blur)
                        BackgroundBlurTile(
                            picture = model.picture,
                            level = index + 1,
                            strength = blur,
                            chosen = model.background == fill,
                        ) { model.setBackground(fill) }
                    }
                }
            }
        }
    }
}

/** Color and Blur side by side, the chosen one raised, as iOS's segmented picker draws them. */
@Composable
private fun BackgroundTabs(tab: BackgroundFillTab, choose: (BackgroundFillTab) -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .height(32.dp)
            .background(Palette.white(0.19), RoundedCornerShape(8.dp))
            .padding(2.dp),
    ) {
        for (option in BackgroundFillTab.entries) {
            val chosen = option == tab
            Box(
                Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .background(
                        if (chosen) Palette.white(0.39) else Color.Transparent,
                        RoundedCornerShape(7.dp),
                    )
                    .clickable { choose(option) }
                    .semantics { selected = chosen },
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    option.title,
                    color = Color.White,
                    fontSize = 13.sp,
                    fontWeight = if (chosen) FontWeight.SemiBold else FontWeight.Normal,
                )
            }
        }
    }
}

/** A tile that takes a color or blur off. */
@Composable
private fun BackgroundNoneTile(chosen: Boolean, tap: () -> Unit) {
    Box(
        Modifier
            .size(backgroundTile)
            .backgroundChosenRing(chosen)
            .background(Palette.raised, backgroundTileShape)
            .clickable(onClick = tap)
            .semantics {
                contentDescription = "None"
                selected = chosen
            },
        contentAlignment = Alignment.Center,
    ) { HugeIcon("unavailable", size = 22.dp, tint = Palette.muted) }
}

/** A tile of one color, ringed when chosen. */
@Composable
private fun BackgroundSwatchTile(color: String, chosen: Boolean, tap: () -> Unit) {
    Box(
        Modifier
            .size(backgroundTile)
            .backgroundChosenRing(chosen)
            .background(backgroundHexColor(color), backgroundTileShape)
            .border(1.dp, Color.White.copy(alpha = 0.15f), backgroundTileShape)
            .clickable(onClick = tap)
            .semantics {
                contentDescription = color
                selected = chosen
            },
    )
}

/** The picture at the playhead, blurred more at each level. */
@Composable
private fun BackgroundBlurTile(
    picture: ImageBitmap?,
    level: Int,
    strength: Double,
    chosen: Boolean,
    tap: () -> Unit,
) {
    Box(
        Modifier
            .size(backgroundTile)
            .backgroundChosenRing(chosen)
            .clip(backgroundTileShape)
            .background(Palette.raised)
            .clickable(onClick = tap)
            .semantics {
                contentDescription = "Blur $level"
                selected = chosen
            },
    ) {
        if (picture != null) {
            Image(
                picture,
                contentDescription = null,
                contentScale = ContentScale.Crop,
                modifier = Modifier
                    .fillMaxSize()
                    .blur((strength * 60).dp, BlurredEdgeTreatment.Unbounded),
            )
        }
    }
}

/** An accent ring 3 points outside the tile when it is chosen, drawn over it and never clipped by it. */
private fun Modifier.backgroundChosenRing(chosen: Boolean, outset: Dp = 3.dp): Modifier =
    drawWithContent {
        drawContent()
        if (chosen) {
            val out = outset.toPx()
            drawRoundRect(
                color = Palette.accent,
                topLeft = Offset(-out, -out),
                size = Size(size.width + 2 * out, size.height + 2 * out),
                cornerRadius = CornerRadius(8.dp.toPx()),
                style = Stroke(width = 2.dp.toPx()),
            )
        }
    }

/** `#RRGGBB`. Anything else is black. */
private fun backgroundHexColor(hex: String): Color {
    val digits = hex.removePrefix("#")
    val valid = digits.length == 6 && digits.all { it in '0'..'9' || it in 'a'..'f' || it in 'A'..'F' }
    val value = if (valid) digits.toLong(16) else 0L
    return Color(0xFF000000L or value)
}
