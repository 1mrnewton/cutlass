package com.scytheralpha.cutlass_android

import androidx.compose.ui.graphics.Color

/** The desktop's colors, as the iOS app has them. */
object Palette {
    val panel = Color(0xFF1C1C1C)
    val bar = Color(0xFF262626)
    val export = Color(0xFF00C8D7)
    val exportInk = Color(0xFF062226)
    val text = Color(0xFFF3F3F3)
    val muted = Color(0xFFA3A3A3)

    /** Sliders, the selected control, and other live accents. */
    val accent = export

    /** The editor's own greys, from SwiftUI's `Color(white:)`. */
    val editor = white(0.06)
    val strip = white(0.09)
    val tools = white(0.11)
    val raised = white(0.17)

    fun white(level: Double): Color = Color(level.toFloat(), level.toFloat(), level.toFloat())
}
