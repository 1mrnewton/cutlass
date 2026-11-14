package com.scytheralpha.cutlass_android

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
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import uniffi.cutlass_ffi_bindings.ExportProgress
import uniffi.cutlass_ffi_bindings.createProject

/**
 * The editor end to end on the phone, through the same model the screens
 * use: a picture goes in, draws, exports as an MP4 through MediaCodec, and
 * that movie goes into a second project and draws again.
 */
@RunWith(AndroidJUnit4::class)
class EditorFlowTest {
    private val target = InstrumentationRegistry.getInstrumentation().targetContext
    private val tests = InstrumentationRegistry.getInstrumentation().context

    private suspend fun until(what: String, ready: () -> Boolean) {
        withTimeout(60_000) {
            while (!ready()) delay(50)
        }
        assertTrue(what, ready())
    }

    private suspend fun opened(scope: CoroutineScope): EditorModel {
        val model = EditorModel(createProject(Places.root), scope)
        until("the project opens") { !model.isOpening }
        return model
    }

    @Test
    fun a_picture_draws_exports_and_the_movie_comes_back() = runBlocking {
        Places.set(target)
        val scratch = File(target.cacheDir, "flow-${System.nanoTime()}").apply { mkdirs() }
        val still = File(scratch, "pattern.png")
        tests.assets.open("pattern.png").use { input -> still.outputStream().use { input.copyTo(it) } }
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
        try {
            withContext(Dispatchers.Main) {
                val model = opened(scope)
                model.importFile(still.path)
                assertTrue(model.notice?.text ?: "", model.hasClips)
                assertTrue("a still lasts", model.durationUs > 0)
                until("the picture draws") { model.picture != null }
                model.seek(model.durationUs / 2)
                model.split()
                assertEquals(2, model.lanes.first { it.main }.clips.size)
                model.undo()
                assertEquals(1, model.lanes.first { it.main }.clips.size)

                val editor = model.beginExport()
                assertNotNull("the editor is handed to the export", editor)
                val movie = File(scratch, "flow.mp4")
                var seen = 0UL
                val done = withContext(Dispatchers.Default) {
                    editor!!.export(
                        movie.path,
                        720u,
                        object : ExportProgress {
                            override fun progress(done: ULong, total: ULong): Boolean {
                                seen = done
                                return true
                            }
                        },
                    )
                }
                model.endExport()
                assertTrue("frames were written", done.frames > 0UL && seen > 0UL)
                assertTrue(movie.length() > 1_000)

                val again = opened(scope)
                again.importFile(movie.path)
                assertTrue(again.notice?.text ?: "", again.hasClips)
                until("the movie draws") { again.picture != null }
                assertTrue(again.durationUs > 0)
            }
        } finally {
            scope.cancel()
            scratch.deleteRecursively()
        }
    }
}
