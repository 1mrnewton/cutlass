package com.scytheralpha.cutlass_android

import org.junit.Assert.assertEquals
import org.junit.Test

class NamesTest {
    @Test
    fun an_icon_name_becomes_its_drawable_name() {
        assertEquals("icon_play", assetName("play"))
        assertEquals("icon_text_video", assetName("textVideo"))
        assertEquals("icon_chevron_left", assetName("chevronLeft"))
    }

    @Test
    fun a_length_reads_in_seconds_then_minutes() {
        assertEquals("4.2s", length(4_200_000))
        assertEquals("0.0s", length(0))
        assertEquals("1:05.3", length(65_300_000))
    }
}
