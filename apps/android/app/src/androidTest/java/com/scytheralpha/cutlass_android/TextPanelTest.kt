package com.scytheralpha.cutlass_android

import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsOff
import androidx.compose.ui.test.assertIsOn
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithContentDescription
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onFirst
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performSemanticsAction
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.test.performTextInputSelection
import androidx.compose.ui.text.TextRange
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import uniffi.cutlass_ffi_bindings.createProject

/**
 * The text panel on the phone, tapped and typed into as a person would:
 * the search narrows the faces to the ones Rust matches, bold, italic and
 * underline go on the letters selected in the words, and a face that
 * varies gets a slider for each of its axes.
 */
@RunWith(AndroidJUnit4::class)
class TextPanelTest {
    @get:Rule
    val compose = createComposeRule()

    private val target = InstrumentationRegistry.getInstrumentation().targetContext
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private lateinit var model: EditorModel
    private lateinit var title: String

    @Before
    fun open() {
        Places.set(target)
        compose.runOnUiThread { model = EditorModel(createProject(Places.root), scope) }
        compose.waitUntil(30_000) { !model.isOpening }
        compose.runOnUiThread {
            val added = model.typeText(null, "Hello world", INTER)
            assertNotNull(model.notice?.text ?: "no title was added", added)
            title = added!!
        }
    }

    @After
    fun close() {
        scope.cancel()
    }

    private fun show() {
        compose.setContent { TextPanel(model, title) {} }
        compose.waitForIdle()
    }

    @Test
    fun the_shipped_families_are_listed_and_a_search_narrows_them() {
        assertTrue("${model.fonts.size} faces", model.fonts.size > 100)
        assertTrue(model.fonts.any { it.name == INTER })
        val first = model.fonts.first()
        val last = model.fonts.last()
        show()
        compose.onAllNodesWithText(first.label).onFirst().assertIsDisplayed()
        // The row draws only the faces on screen, and the last is far off it.
        compose.onAllNodesWithText(last.label).assertCountEquals(0)

        compose.onNodeWithContentDescription("Search fonts").performTextInput(last.label.uppercase())
        compose.waitForIdle()
        val found = model.fontsMatching(last.label)
        assertTrue("${found.size} of ${model.fonts.size}", found.isNotEmpty() && found.size < model.fonts.size)
        compose.onAllNodesWithText(found.first().label).onFirst().assertIsDisplayed()
        if (found.none { it.label == first.label }) {
            compose.onAllNodesWithText(first.label).assertCountEquals(0)
        }

        compose.onNodeWithContentDescription("Search fonts").performTextInput(" qqzzxx")
        compose.waitForIdle()
        compose.onNodeWithText("No fonts match.").assertIsDisplayed()

        compose.onNodeWithContentDescription("Clear search").performClick()
        compose.waitForIdle()
        compose.onAllNodesWithText(first.label).onFirst().assertIsDisplayed()
        assertEquals(model.fonts, model.fontsMatching(""))
    }

    @Test
    fun bold_italic_and_underline_go_on_the_selected_letters() {
        show()
        // Nothing selected is the whole title, which starts plain.
        compose.onNodeWithContentDescription("Bold").assertIsOff()
        compose.onNodeWithContentDescription("Words").performTextInputSelection(TextRange(0, 5))
        compose.waitForIdle()
        compose.onNodeWithContentDescription("Bold").performClick()
        compose.waitForIdle()

        compose.runOnUiThread {
            assertTrue(model.notice?.text ?: "", model.textPattern(title, 0, 5)!!.bold)
            assertFalse("the second word stays plain", model.textPattern(title, 6, 11)!!.bold)
            assertFalse("not every letter is bold", model.textPattern(title, 0, 11)!!.bold)
        }
        compose.onNodeWithContentDescription("Bold").assertIsOn()
        compose.onNodeWithContentDescription("Italic").assertIsOff()

        // The other word shows its own pattern, and takes an underline.
        compose.onNodeWithContentDescription("Words").performTextInputSelection(TextRange(11, 6))
        compose.waitForIdle()
        compose.onNodeWithContentDescription("Bold").assertIsOff()
        compose.onNodeWithContentDescription("Underline").performClick()
        compose.waitForIdle()
        compose.onNodeWithContentDescription("Underline").assertIsOn()
        compose.runOnUiThread {
            assertTrue(model.textPattern(title, 6, 11)!!.underline)
            assertFalse(model.textPattern(title, 0, 5)!!.underline)
        }

        // A caret alone is every letter again: italic goes on them all.
        compose.onNodeWithContentDescription("Words").performTextInputSelection(TextRange(3))
        compose.waitForIdle()
        compose.onNodeWithContentDescription("Bold").assertIsOff()
        compose.onNodeWithContentDescription("Italic").performClick()
        compose.waitForIdle()
        compose.onNodeWithContentDescription("Italic").assertIsOn()
        compose.runOnUiThread {
            assertTrue(model.textPattern(title, 0, 11)!!.italic)
            // Each tap was one undo step.
            model.undo()
            assertFalse(model.textPattern(title, 0, 11)!!.italic)
            assertTrue(model.textPattern(title, 6, 11)!!.underline)
            model.undo()
            assertFalse(model.textPattern(title, 6, 11)!!.underline)
            assertTrue(model.textPattern(title, 0, 5)!!.bold)
        }
    }

    @Test
    fun a_face_that_varies_has_a_slider_for_each_axis_and_another_has_none() {
        show()
        val axes = compose.runOnUiThread<List<String>> { model.textFontAxes(title).map { it.name } }
        assertEquals(listOf("Optical size", "Weight"), axes)
        compose.onNodeWithContentDescription("Optical size").assertIsDisplayed()
        compose.onNodeWithContentDescription("Weight").assertIsDisplayed()
        fun weight() = compose.runOnUiThread<Double> {
            model.textFontAxes(title).first { it.tag == "wght" }.value
        }
        assertEquals(400.0, weight(), 0.0)

        // Moving the thumb and letting go is one undo step.
        compose.onNodeWithContentDescription("Weight")
            .performSemanticsAction(SemanticsActions.SetProgress) { it(700f) }
        compose.waitForIdle()
        assertEquals(model.notice?.text ?: "", 700.0, weight(), 0.5)
        compose.runOnUiThread { model.undo() }
        assertEquals(400.0, weight(), 0.0)

        // A face with one fixed design has no axes, so no sliders.
        val fixed = compose.runOnUiThread<String?> {
            model.fonts.map { it.name }.filter { '[' !in it && '@' !in it }.firstOrNull { face ->
                model.setTextFont(title, face)
                model.textFontAxes(title).isEmpty()
            }
        }
        assertNotNull("no face without axes was found", fixed)
        compose.waitForIdle()
        compose.onAllNodesWithContentDescription("Weight").assertCountEquals(0)
        compose.onAllNodesWithContentDescription("Optical size").assertCountEquals(0)
    }

    private companion object {
        /** A shipped face that varies in weight and optical size. */
        const val INTER = "Inter[opsz,wght].ttf"
    }
}
