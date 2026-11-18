package com.scytheralpha.cutlass_android

import androidx.compose.ui.geometry.Offset
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import uniffi.cutlass_ffi_bindings.createProject

/**
 * The Mask panel's edits on the phone, through the model the screens use:
 * masks are added and taken in hand, combine by their own modes, move by
 * a knob as one undo step, and text, brush and pen masks take a drawing.
 */
@RunWith(AndroidJUnit4::class)
class MasksTest {
    private val target = InstrumentationRegistry.getInstrumentation().targetContext
    private val tests = InstrumentationRegistry.getInstrumentation().context

    private suspend fun until(what: String, ready: () -> Boolean) {
        withTimeout(60_000) {
            while (!ready()) delay(50)
        }
        assertTrue(what, ready())
    }

    @Test
    fun a_clip_takes_several_masks_and_their_handles() = runBlocking {
        Places.set(target)
        val scratch = File(target.cacheDir, "masks-${System.nanoTime()}").apply { mkdirs() }
        val still = File(scratch, "pattern.png")
        tests.assets.open("pattern.png").use { input -> still.outputStream().use { input.copyTo(it) } }
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
        try {
            withContext(Dispatchers.Main) {
                val model = EditorModel(createProject(Places.root), scope)
                until("the project opens") { !model.isOpening }
                model.importFile(still.path)
                assertNotNull(model.notice?.text ?: "", model.selectedClip)
                assertTrue(model.maskShapeList.any { it.key == "brush" })

                model.addMask("circle")
                model.addMask("rectangle")
                assertEquals(listOf("circle", "rectangle"), model.masks.map { it.shape })
                assertEquals("the new mask is in hand", 1, model.maskIndex)
                assertNotNull(model.maskHandles)

                model.editMask(model.maskInHand!!.copy(mode = "subtract"), live = false)
                assertEquals("subtract", model.masks[1].mode)
                assertEquals("add", model.masks[0].mode)

                // A slider drag: live steps, then the release, are one undo step.
                val before = model.maskInHand!!
                model.editMask(before.copy(feather = 0.2), live = true)
                model.editMask(before.copy(feather = 0.4), live = true)
                model.editMask(before.copy(feather = 0.5), live = false)
                assertEquals(0.5, model.maskInHand!!.feather, 1e-6)
                model.undo()
                assertEquals(0.0, model.maskInHand!!.feather, 1e-6)

                // The middle knob moves the mask; the drag is one undo step.
                val handles = model.maskHandles!!
                val middle = Offset(handles.middleX.toFloat(), handles.middleY.toFloat())
                assertTrue(model.beginMaskDrag(middle))
                model.dragMask(middle + Offset(10f, 0f))
                model.dragMask(middle + Offset(40f, 0f))
                model.endMaskDrag()
                assertTrue(model.maskInHand!!.x > before.x)
                model.undo()
                assertEquals(before.x, model.maskInHand!!.x, 1e-6)

                model.moveMask(up = true)
                assertEquals(listOf("rectangle", "circle"), model.masks.map { it.shape })
                assertEquals(0, model.maskIndex)
                model.undo()

                model.addMask("text")
                model.setMaskText("Hi")
                assertEquals("Hi", model.maskInHand!!.text)

                val (width, height) = model.canvasSize!!
                model.addMask("brush")
                model.maskBrushSize = width / 10
                model.paintMask(
                    listOf(
                        Offset((width * 0.3).toFloat(), (height * 0.5).toFloat()),
                        Offset((width * 0.7).toFloat(), (height * 0.5).toFloat()),
                    ),
                    erase = false,
                )
                val brushed = model.maskInHand!!
                assertEquals("brush", brushed.shape)

                model.addMask("pen")
                val pen = model.maskIndex
                model.penPoints = listOf(
                    Offset((width * 0.5).toFloat(), (height * 0.2).toFloat()),
                    Offset((width * 0.8).toFloat(), (height * 0.8).toFloat()),
                    Offset((width * 0.2).toFloat(), (height * 0.8).toFloat()),
                )
                model.finishOutline()
                assertEquals(MaskTool.HANDLES, model.maskTool)
                assertTrue(model.penPoints.isEmpty())
                assertNotEquals(0.0, model.masks[pen].width)

                val count = model.masks.size
                model.removeMask()
                assertEquals(count - 1, model.masks.size)
            }
        } finally {
            scope.cancel()
            scratch.deleteRecursively()
        }
    }
}
