package com.scytheralpha.cutlass_android

import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onFirst
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import uniffi.cutlass_ffi_bindings.createProject

/**
 * The text panel on the phone, tapped and typed into as a person would:
 * the search narrows the faces to the ones Rust matches.
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

    private companion object {
        /** A shipped face that varies in weight and optical size. */
        const val INTER = "Inter[opsz,wght].ttf"
    }
}
