package com.scytheralpha.cutlass_android

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.test.pinch
import androidx.compose.ui.unit.dp
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import uniffi.cutlass_ffi_bindings.createProject

/**
 * The timeline and preview gestures with real touches: pinch zooms the
 * strip, a long press then a drag reorders main, and a drag and a pinch on
 * the preview move and grow the picture.
 */
@RunWith(AndroidJUnit4::class)
class GesturesTest {
    @get:Rule
    val compose = createComposeRule()

    private val target = InstrumentationRegistry.getInstrumentation().targetContext
    private val tests = InstrumentationRegistry.getInstrumentation().context
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private lateinit var scratch: File
    private lateinit var model: EditorModel

    @Before
    fun open() {
        Places.set(target)
        scratch = File(target.cacheDir, "gestures-${System.nanoTime()}").apply { mkdirs() }
        val still = File(scratch, "pattern.png")
        tests.assets.open("pattern.png").use { input -> still.outputStream().use { input.copyTo(it) } }
        compose.runOnUiThread { model = EditorModel(createProject(Places.root), scope) }
        compose.waitUntil(30_000) { !model.isOpening }
        compose.runOnUiThread {
            model.importFile(still.path)
            model.importFile(still.path)
            model.selectedClip = null
            model.seek(0)
        }
    }

    @After
    fun close() {
        scope.cancel()
        scratch.deleteRecursively()
    }

    private fun mainClips() = model.lanes.first { it.main }.clips.map { it.id }

    @Test
    fun a_pinch_zooms_the_strip_and_a_held_drag_reorders_main() {
        compose.setContent {
            Column {
                TimelineStrip(model, onAdd = {}, modifier = Modifier.fillMaxWidth().height(300.dp).testTag("strip"))
            }
        }
        compose.waitForIdle()
        val before = model.pixelSize
        compose.onNodeWithTag("strip").performTouchInput {
            val y = 200.dp.toPx()
            pinch(
                start0 = Offset(centerX - 40.dp.toPx(), y),
                end0 = Offset(centerX - 120.dp.toPx(), y),
                start1 = Offset(centerX + 40.dp.toPx(), y),
                end1 = Offset(centerX + 120.dp.toPx(), y),
            )
        }
        compose.waitForIdle()
        assertTrue("pinching out zooms in: ${model.pixelSize} after $before", model.pixelSize > before * 1.5)

        // Narrow the clips to about 20 dp a second, inside what the cut allows.
        compose.runOnUiThread { model.zoom(model.pixelSize, 20.0 / 1_000_000 / model.pixelSize, 400.0) }
        compose.waitForIdle()
        val order = mainClips()
        assertEquals(2, order.size)
        val firstWidth = (model.lanes.first { it.main }.clips[0].durationUs * model.pixelSize).toFloat()
        compose.onNodeWithTag("strip").performTouchInput {
            val row = (26f + 6f + 28f).dp.toPx()
            val start = Offset(centerX + (firstWidth / 2).dp.toPx(), row)
            down(start)
            advanceEventTime(600)
            for (step in 1..20) {
                moveTo(start + Offset((firstWidth * 1.6f * step / 20).dp.toPx(), 0f))
                advanceEventTime(16)
            }
            up()
        }
        compose.waitForIdle()
        compose.waitUntil(5_000) { mainClips() != order }
        assertEquals(order.reversed(), mainClips())
    }

    @Test
    fun a_drag_moves_and_a_pinch_grows_the_picture_on_the_preview() {
        compose.setContent {
            Box(Modifier.size(320.dp).testTag("stage")) {
                Box(Modifier.size(320.dp)) { StageOverlay(model) {} }
            }
        }
        compose.runOnUiThread { model.selectedClip = mainClips().first() }
        compose.waitUntil(10_000) { model.stage != null }
        val placed = model.stage!!
        compose.onNodeWithTag("stage").performTouchInput {
            down(center)
            for (step in 1..10) {
                moveTo(center + Offset((6 * step).dp.toPx(), (3 * step).dp.toPx()))
                advanceEventTime(16)
            }
            up()
        }
        compose.waitForIdle()
        compose.waitUntil(5_000) { model.stage?.centerX != placed.centerX }
        val moved = model.stage!!
        assertNotEquals(placed.centerX, moved.centerX)
        assertNotEquals(placed.centerY, moved.centerY)

        compose.onNodeWithTag("stage").performTouchInput {
            pinch(
                start0 = center - Offset(30.dp.toPx(), 0f),
                end0 = center - Offset(90.dp.toPx(), 0f),
                start1 = center + Offset(30.dp.toPx(), 0f),
                end1 = center + Offset(90.dp.toPx(), 0f),
            )
        }
        compose.waitForIdle()
        compose.waitUntil(5_000) { (model.stage?.width ?: 0.0) > moved.width * 1.3 }
        assertTrue(model.stage!!.width > moved.width * 1.3)
    }
}
