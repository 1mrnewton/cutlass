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
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.roundToInt
import uniffi.cutlass_ffi_bindings.TextStyleChange
import uniffi.cutlass_ffi_bindings.TextStyleRanges
import uniffi.cutlass_ffi_bindings.textColors
import uniffi.cutlass_ffi_bindings.textStyleRanges
import uniffi.cutlass_ffi_bindings.TextStyle as TitleStyle

/** The panel's two tabs. */
private enum class TextTab(val title: String) {
    Fonts("Fonts"),
    Styles("Styles"),
}

/**
 * Add text, or change a text clip: the words on top with a check that
 * closes, then the faces Rust found or the title's styles. The words show
 * on the preview as they are typed: the first letter adds the clip at the
 * playhead, and the whole typing is one undo step. Picking a face or a
 * color applies it at once, as an undo step of its own, and each slider
 * drag is one step.
 *
 * `clip` is the clip being changed, or null to add one.
 */
@Composable
fun TextPanel(model: EditorModel, clip: String?, done: () -> Unit) {
    // The clip the words go into: `clip`, or the one the first letter added.
    var typed by remember { mutableStateOf(clip) }
    val start = remember { clip?.let { model.text(it) } }
    var words by remember { mutableStateOf(start?.content ?: "") }
    var face by remember { mutableStateOf(start?.face ?: model.fonts.firstOrNull()?.name ?: "") }
    var tab by remember { mutableStateOf(TextTab.Fonts) }
    val typing = remember { FocusRequester() }
    val focus = LocalFocusManager.current
    val keyboard = LocalSoftwareKeyboardController.current

    fun stopTyping() {
        focus.clearFocus()
        keyboard?.hide()
    }

    fun pick(name: String) {
        face = name
        typed?.let { model.setTextFont(it, name) }
    }

    LaunchedEffect(Unit) {
        if (clip == null) {
            typing.requestFocus()
            keyboard?.show()
        }
    }

    CompositionLocalProvider(LocalContentColor provides Color.White) {
        Column(
            Modifier
                .fillMaxWidth()
                .background(Palette.tools)
                .imePadding()
                .padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                BasicTextField(
                    value = words,
                    onValueChange = { changed ->
                        if (changed != words) {
                            words = changed
                            typed = model.typeText(typed, changed, face)
                        }
                    },
                    modifier = Modifier
                        .weight(1f)
                        .focusRequester(typing),
                    textStyle = TextStyle(color = Color.White, fontSize = 17.sp),
                    keyboardOptions = KeyboardOptions(
                        capitalization = KeyboardCapitalization.Sentences,
                        imeAction = ImeAction.Done,
                    ),
                    keyboardActions = KeyboardActions(onDone = { stopTyping() }),
                    singleLine = false,
                    minLines = 1,
                    maxLines = 3,
                    cursorBrush = SolidColor(Palette.accent),
                    decorationBox = { field ->
                        Box(
                            Modifier
                                .fillMaxWidth()
                                .background(Palette.white(0.18), RoundedCornerShape(8.dp))
                                .padding(horizontal = 12.dp, vertical = 9.dp),
                        ) {
                            if (words.isEmpty()) {
                                Text("Enter text", color = Color.White.copy(alpha = 0.4f), fontSize = 17.sp)
                            }
                            field()
                        }
                    },
                )
                Box(
                    Modifier
                        .size(width = 40.dp, height = 36.dp)
                        .clickable {
                            stopTyping()
                            done()
                        }
                        .semantics { contentDescription = "Done" },
                    contentAlignment = Alignment.Center,
                ) { HugeIcon("done", size = 20.dp) }
            }
            Segmented(
                choices = TextTab.entries,
                chosen = tab,
                title = { it.title },
            ) {
                tab = it
                if (it == TextTab.Styles) stopTyping()
            }
            when (tab) {
                TextTab.Fonts -> FontsTab(model, face, ::pick)
                TextTab.Styles -> {
                    val into = typed
                    if (into != null) {
                        StylesTab(model, into)
                    } else {
                        Text(
                            "Type some words to style them.",
                            fontSize = 13.sp,
                            color = Color.White.copy(alpha = 0.6f),
                        )
                    }
                }
            }
        }
    }
}

/** The faces Rust found, each a capsule, the chosen one in the accent. */
@Composable
private fun FontsTab(model: EditorModel, face: String, pick: (String) -> Unit) {
    if (model.fonts.isEmpty()) {
        Text(
            "No fonts were found on this device.",
            fontSize = 13.sp,
            color = Color.White.copy(alpha = 0.6f),
        )
        return
    }
    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        items(model.fonts, key = { it.name }) { font ->
            val chosen = font.name == face
            Text(
                font.label,
                fontSize = 13.sp,
                maxLines = 1,
                color = if (chosen) Color.Black else Color.White,
                modifier = Modifier
                    .clip(CircleShape)
                    .background(if (chosen) Palette.accent else Palette.white(0.18))
                    .selectable(selected = chosen) { pick(font.name) }
                    .padding(horizontal = 12.dp, vertical = 8.dp),
            )
        }
    }
}

/**
 * A row of choices in one rounded track, the chosen one on a lighter
 * thumb, as a segmented picker draws.
 */
@Composable
private fun <T> Segmented(
    choices: List<T>,
    chosen: T,
    title: (T) -> String,
    choose: (T) -> Unit,
) {
    Row(
        Modifier
            .fillMaxWidth()
            .height(32.dp)
            .background(Palette.white(0.18), RoundedCornerShape(8.dp))
            .padding(2.dp),
    ) {
        for (choice in choices) {
            val on = choice == chosen
            Box(
                Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .height(28.dp)
                    .clip(RoundedCornerShape(7.dp))
                    .background(if (on) Palette.white(0.38) else Color.Transparent)
                    .selectable(selected = on) { choose(choice) },
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    title(choice),
                    fontSize = 13.sp,
                    fontWeight = if (on) FontWeight.SemiBold else FontWeight.Medium,
                    color = Color.White,
                    maxLines = 1,
                )
            }
        }
    }
}

/** The parts of a title the Styles tab changes. */
private enum class StylePart(val title: String) {
    Text("Text"),
    Outline("Outline"),
    Background("Background"),
    Shadow("Shadow"),
    Glow("Glow"),
}

/** The colors the Styles tab offers, from Rust. */
private val styleColors: List<String> by lazy { textColors() }

/** The Styles tab's slider ends and starting values, from Rust. */
private val styleRanges: TextStyleRanges by lazy { textStyleRanges() }

/**
 * Two Material sliders, 44 high each, and the gap between them: room for
 * two sliders on every part, so the panel and the preview above it keep
 * their size.
 */
private val SLIDERS_HEIGHT = 94.dp

/**
 * A title's color and size, outline, background, shadow, and glow. Colors,
 * ranges, and the values a part starts at come from Rust.
 */
@Composable
private fun StylesTab(model: EditorModel, clip: String) {
    var part by remember { mutableStateOf(StylePart.Text) }
    val style = remember(clip, model.edits) { model.text(clip)?.style }

    fun send(change: TextStyleChange, live: Boolean) = model.styleText(clip, change, live)

    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            for (each in StylePart.entries) {
                val on = each == part
                Text(
                    each.title,
                    fontSize = 13.sp,
                    fontWeight = if (on) FontWeight.SemiBold else FontWeight.Normal,
                    color = if (on) Color.White else Color.White.copy(alpha = 0.55f),
                    modifier = Modifier.selectable(selected = on) { part = each },
                )
            }
        }
        if (style != null) {
            Swatches(part, style, ::send)
            Column(
                Modifier.height(SLIDERS_HEIGHT),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                key(part) { Sliders(part, style, ::send) }
            }
        }
    }
}

/** A part's colors, with a tile that turns the part off on every part but the text. */
@Composable
private fun Swatches(
    part: StylePart,
    style: TitleStyle,
    send: (TextStyleChange, Boolean) -> Unit,
) {
    val chosen = chosenColor(part, style)
    Row(
        Modifier
            .horizontalScroll(rememberScrollState())
            // Room for the ring around the chosen tile.
            .padding(4.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        if (part != StylePart.Text) {
            NoColorTile(chosen = chosen.isEmpty()) { setColor(part, "", style, send) }
        }
        for (color in styleColors) {
            ColorTile(color = color, chosen = color.equals(chosen, ignoreCase = true)) {
                setColor(part, color, style, send)
            }
        }
    }
}

@Composable
private fun Sliders(
    part: StylePart,
    style: TitleStyle,
    send: (TextStyleChange, Boolean) -> Unit,
) {
    val ranges = styleRanges
    when (part) {
        StylePart.Text -> StyleSlider(
            label = "Size",
            current = style.size,
            range = ranges.sizeMin..ranges.sizeMax,
            shown = { it.roundToInt().toString() },
        ) { size, live -> send(TextStyleChange.Size(size), live) }
        StylePart.Outline -> if (style.outlineColor.isNotEmpty()) {
            StyleSlider(
                label = "Width",
                current = style.outlineWidth,
                range = 0.0..ranges.outlineMax,
                shown = { percent(it / ranges.outlineMax) },
            ) { width, live -> send(TextStyleChange.Outline(style.outlineColor, width), live) }
        }
        StylePart.Background -> if (style.backgroundColor.isNotEmpty()) {
            StyleSlider(
                label = "Opacity",
                current = style.backgroundAlpha,
                range = 0.0..1.0,
                shown = ::percent,
            ) { alpha, live -> send(background(style, alpha = alpha), live) }
            StyleSlider(
                label = "Corners",
                current = style.backgroundRound,
                range = 0.0..1.0,
                shown = ::percent,
            ) { round, live -> send(background(style, round = round), live) }
        }
        StylePart.Shadow -> if (style.shadowColor.isNotEmpty()) {
            StyleSlider(
                label = "Distance",
                current = style.shadowDistance,
                range = 0.0..ranges.shadowDistanceMax,
                shown = { percent(it / ranges.shadowDistanceMax) },
            ) { distance, live -> send(shadow(style, distance = distance), live) }
            StyleSlider(
                label = "Blur",
                current = style.shadowBlur,
                range = 0.0..1.0,
                shown = ::percent,
            ) { blur, live -> send(shadow(style, blur = blur), live) }
        }
        StylePart.Glow -> if (style.glowColor.isNotEmpty()) {
            StyleSlider(
                label = "Strength",
                current = style.glowStrength,
                range = 0.0..1.0,
                shown = ::percent,
            ) { strength, live -> send(glow(style, strength = strength), live) }
            StyleSlider(
                label = "Spread",
                current = style.glowRange,
                range = 0.0..1.0,
                shown = ::percent,
            ) { range, live -> send(glow(style, range = range), live) }
        }
    }
}

private fun chosenColor(part: StylePart, style: TitleStyle): String = when (part) {
    StylePart.Text -> style.color.ifEmpty { "#FFFFFF" }
    StylePart.Outline -> style.outlineColor
    StylePart.Background -> style.backgroundColor
    StylePart.Shadow -> style.shadowColor
    StylePart.Glow -> style.glowColor
}

/** Pick a color; a part turned on starts at Rust's values. */
private fun setColor(
    part: StylePart,
    color: String,
    style: TitleStyle,
    send: (TextStyleChange, Boolean) -> Unit,
) {
    val ranges = styleRanges
    when (part) {
        StylePart.Text -> send(TextStyleChange.Color(color), false)
        StylePart.Outline -> {
            val width = if (style.outlineWidth > 0) style.outlineWidth else ranges.outlineStart
            send(TextStyleChange.Outline(color, width), false)
        }
        StylePart.Background -> {
            val fresh = style.backgroundColor.isEmpty()
            send(
                TextStyleChange.Background(
                    color = color,
                    alpha = if (fresh) ranges.backgroundAlphaStart else style.backgroundAlpha,
                    round = if (fresh) ranges.backgroundRoundStart else style.backgroundRound,
                ),
                false,
            )
        }
        StylePart.Shadow -> {
            val fresh = style.shadowColor.isEmpty()
            send(
                TextStyleChange.Shadow(
                    color = color,
                    distance = if (fresh) ranges.shadowDistanceStart else style.shadowDistance,
                    blur = if (fresh) ranges.shadowBlurStart else style.shadowBlur,
                ),
                false,
            )
        }
        StylePart.Glow -> {
            val fresh = style.glowColor.isEmpty()
            send(
                TextStyleChange.Glow(
                    color = color,
                    strength = if (fresh) ranges.glowStrengthStart else style.glowStrength,
                    range = if (fresh) ranges.glowRangeStart else style.glowRange,
                ),
                false,
            )
        }
    }
}

private fun background(
    style: TitleStyle,
    alpha: Double = style.backgroundAlpha,
    round: Double = style.backgroundRound,
): TextStyleChange = TextStyleChange.Background(style.backgroundColor, alpha, round)

private fun shadow(
    style: TitleStyle,
    distance: Double = style.shadowDistance,
    blur: Double = style.shadowBlur,
): TextStyleChange = TextStyleChange.Shadow(style.shadowColor, distance, blur)

private fun glow(
    style: TitleStyle,
    strength: Double = style.glowStrength,
    range: Double = style.glowRange,
): TextStyleChange = TextStyleChange.Glow(style.glowColor, strength, range)

private fun percent(fraction: Double): String = (fraction * 100).roundToInt().toString()

/**
 * A labelled slider that reports each move as live and the release as
 * final, so a drag is one undo step. Between drags it follows the title,
 * so an undo moves it back.
 */
@Composable
private fun StyleSlider(
    label: String,
    current: Double,
    range: ClosedFloatingPointRange<Double>,
    shown: (Double) -> String,
    change: (Double, Boolean) -> Unit,
) {
    fun clamped(value: Double) = value.coerceIn(range.start, range.endInclusive)
    var value by remember { mutableDoubleStateOf(clamped(current)) }
    var dragging by remember { mutableStateOf(false) }

    LaunchedEffect(current) {
        if (!dragging) value = clamped(current)
    }

    Row(
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            label,
            fontSize = 13.sp,
            color = Color.White.copy(alpha = 0.75f),
            maxLines = 1,
            modifier = Modifier.width(72.dp),
        )
        Slider(
            value = value.toFloat(),
            onValueChange = {
                dragging = true
                value = it.toDouble()
                change(value, true)
            },
            onValueChangeFinished = {
                dragging = false
                change(value, false)
            },
            valueRange = range.start.toFloat()..range.endInclusive.toFloat(),
            colors = SliderDefaults.colors(
                thumbColor = Palette.accent,
                activeTrackColor = Palette.accent,
                inactiveTrackColor = Color.White.copy(alpha = 0.2f),
            ),
            modifier = Modifier
                .weight(1f)
                .semantics {
                    contentDescription = label
                    stateDescription = shown(value)
                },
        )
        Text(
            shown(value),
            fontSize = 12.sp,
            color = Color.White.copy(alpha = 0.6f),
            textAlign = TextAlign.End,
            maxLines = 1,
            style = TextStyle(fontFeatureSettings = "tnum"),
            modifier = Modifier.width(36.dp),
        )
    }
}

/** A tile that takes a part's color off. */
@Composable
private fun NoColorTile(chosen: Boolean, tap: () -> Unit) {
    Box(
        Modifier
            .size(36.dp)
            .chosenRing(chosen)
            .clip(RoundedCornerShape(8.dp))
            .background(Palette.raised)
            .selectable(selected = chosen, onClick = tap)
            .semantics { contentDescription = "None" },
        contentAlignment = Alignment.Center,
    ) { HugeIcon("unavailable", size = 22.dp, tint = Palette.muted) }
}

/** A tile of one color, ringed when chosen. */
@Composable
private fun ColorTile(color: String, chosen: Boolean, tap: () -> Unit) {
    val shape = RoundedCornerShape(8.dp)
    Box(
        Modifier
            .size(36.dp)
            .chosenRing(chosen)
            .clip(shape)
            .background(hexColor(color))
            .border(1.dp, Color.White.copy(alpha = 0.15f), shape)
            .selectable(selected = chosen, onClick = tap)
            .semantics { contentDescription = color },
    )
}

/** The accent ring 3 outside a chosen tile. */
private fun Modifier.chosenRing(chosen: Boolean): Modifier = drawWithContent {
    drawContent()
    if (chosen) {
        val out = 3.dp.toPx()
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
private fun hexColor(hex: String): Color {
    val digits = hex.removePrefix("#")
    val value = if (digits.length == 6) digits.toIntOrNull(16) ?: 0 else 0
    return Color(red = (value shr 16) and 0xFF, green = (value shr 8) and 0xFF, blue = value and 0xFF)
}
