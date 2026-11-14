package com.scytheralpha.cutlass_android

import androidx.compose.ui.text.TextRange
import org.junit.Assert.assertEquals
import org.junit.Test

class LettersTest {
    @Test
    fun a_caret_stands_for_the_whole_of_the_words() {
        assertEquals(0 until 11, letters("Hello world", TextRange(4)))
        assertEquals(0 until 0, letters("", TextRange(0)))
    }

    @Test
    fun a_selection_is_its_letters_whichever_way_it_was_dragged() {
        assertEquals(0 until 5, letters("Hello world", TextRange(0, 5)))
        assertEquals(6 until 11, letters("Hello world", TextRange(11, 6)))
    }

    @Test
    fun a_letter_outside_the_basic_plane_counts_once() {
        // The field counts the clapper board as two UTF-16 units.
        val words = "a🎬b"
        assertEquals(0 until 3, letters(words, TextRange(0)))
        assertEquals(2 until 3, letters(words, TextRange(3, 4)))
        assertEquals(1 until 2, letters(words, TextRange(1, 3)))
    }

    @Test
    fun a_selection_past_the_end_stops_at_the_end() {
        assertEquals(3 until 5, letters("Hello", TextRange(3, 40)))
    }
}
