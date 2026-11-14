package com.scytheralpha.cutlass_android

import android.annotation.SuppressLint
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.painter.ColorPainter
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.material3.LocalContentColor

/**
 * A free Hugeicons glyph from `crates/icons`. The name is the one in
 * `icons.txt`; `cargo run -p icongen` writes the drawables. The content
 * color tints it. The button or the text beside it carries the label.
 */
@SuppressLint("DiscouragedApi", "LocalContextGetResourceValueCall")
@Composable
fun HugeIcon(
    name: String,
    modifier: Modifier = Modifier,
    size: Dp = 22.dp,
    tint: Color = LocalContentColor.current,
) {
    val context = LocalContext.current
    val id = remember(name) {
        context.resources.getIdentifier(assetName(name), "drawable", context.packageName)
    }
    val painter = if (id != 0) painterResource(id) else ColorPainter(Color.Transparent)
    Icon(painter, contentDescription = null, modifier = modifier.size(size), tint = tint)
}

/** `textVideo` becomes `icon_text_video`, as in `cutlass_icons::asset_name`. */
fun assetName(name: String): String = buildString {
    append("icon_")
    for (character in name) {
        if (character.isUpperCase()) {
            append('_')
            append(character.lowercaseChar())
        } else {
            append(character)
        }
    }
}
