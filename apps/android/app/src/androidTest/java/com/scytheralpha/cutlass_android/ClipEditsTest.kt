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
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import uniffi.cutlass_ffi_bindings.ExportProgress
import uniffi.cutlass_ffi_bindings.LaneKind
import uniffi.cutlass_ffi_bindings.createProject

/**
 * The clip bar's one-tap edits on the phone, through the model the screens
 * use: a still mirrors, turns, crops and fades; a movie MediaCodec wrote
 * gives up its sound, takes a beat and freezes a frame; a picture goes on
 * an overlay row and back to main.
 */
@RunWith(AndroidJUnit4::class)
class ClipEditsTest {
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

    private fun EditorModel.clipsOn(kind: LaneKind, main: Boolean) =
        lanes.filter { it.kind == kind && it.main == main }.flatMap { it.clips }

    @Test
    fun the_clip_bar_edits_a_still_a_movie_and_an_overlay() = runBlocking {
        Places.set(target)
        val scratch = File(target.cacheDir, "edits-${System.nanoTime()}").apply { mkdirs() }
        val still = File(scratch, "pattern.png")
        tests.assets.open("pattern.png").use { input -> still.outputStream().use { input.copyTo(it) } }
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
        try {
            withContext(Dispatchers.Main) {
                val model = opened(scope)
                model.importFile(still.path)
                val picture = model.selectedClip
                assertNotNull(model.notice?.text ?: "", picture)
                picture!!
                val can = model.clipState(picture)!!
                assertTrue(can.picture && can.canCrop && can.plays)
                assertFalse("a still is not moving", can.canFreeze || can.canExtract)

                model.toggleMirror(picture)
                assertTrue(model.clipState(picture)!!.mirrored)
                model.turn(picture)
                model.toggleCrop(picture)
                assertTrue(model.clipState(picture)!!.crop.isNotEmpty())
                model.setFade(picture, 300_000, 500_000)
                assertEquals(300_000L, model.fade(picture).inUs)
                assertEquals(500_000L, model.fade(picture).outUs)
                repeat(4) { model.undo() }
                assertFalse(model.clipState(picture)!!.mirrored)
                assertFalse(model.clipState(picture)!!.crop.isNotEmpty())
                assertEquals(0L, model.fade(picture).inUs)

                model.clearNotice()
                model.extractAudio(picture)
                assertNotNull("a still has no sound to extract", model.notice)

                // A movie with a sound track, written by the phone's encoder.
                val editor = model.beginExport()!!
                val movie = File(scratch, "edits.mp4")
                withContext(Dispatchers.Default) {
                    editor.export(
                        movie.path,
                        720u,
                        object : ExportProgress {
                            override fun progress(done: ULong, total: ULong) = true
                        },
                    )
                }
                model.endExport()

                val again = opened(scope)
                again.importFile(movie.path)
                val video = again.selectedClip!!
                assertTrue(again.clipState(video)!!.canFreeze)

                again.extractAudio(video)
                val sound = again.clipsOn(LaneKind.AUDIO, main = false).singleOrNull()
                assertNotNull(again.notice?.text ?: "", sound)
                assertEquals(sound!!.id, again.selectedClip)
                assertTrue(again.clipState(sound.id)!!.canBeat)

                again.seek(again.durationUs / 2)
                again.toggleBeat(sound.id)
                assertEquals(1, again.beats[sound.id]?.size)
                again.toggleBeat(sound.id)
                assertNull(again.beats[sound.id])

                val mainBefore = again.clipsOn(LaneKind.VIDEO, main = true).size
                again.freeze(video)
                assertEquals("Frame held", again.notice?.text)
                assertTrue(again.clipsOn(LaneKind.VIDEO, main = true).size > mainBefore)
                again.undo()
                assertEquals(mainBefore, again.clipsOn(LaneKind.VIDEO, main = true).size)

                again.seek(0)
                again.importOverlay(still.path)
                val overlay = again.selectedClip!!
                assertTrue(again.clipsOn(LaneKind.VIDEO, main = false).any { it.id == overlay })
                again.toMain(overlay)
                assertTrue(again.clipsOn(LaneKind.VIDEO, main = true).any { it.id == overlay })
            }
        } finally {
            scope.cancel()
            scratch.deleteRecursively()
        }
    }
}
