package com.scytheralpha.cutlass_android

import android.media.AudioManager
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
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import uniffi.cutlass_ffi_bindings.ExportProgress
import uniffi.cutlass_ffi_bindings.createProject

/**
 * Playing a movie with a sound track opens an AAudio stream Android counts
 * as music, the playhead follows it, and pausing closes it again.
 */
@RunWith(AndroidJUnit4::class)
class PlaybackTest {
    private val target = InstrumentationRegistry.getInstrumentation().targetContext
    private val tests = InstrumentationRegistry.getInstrumentation().context

    private suspend fun until(what: String, ready: () -> Boolean) {
        withTimeout(60_000) {
            while (!ready()) delay(50)
        }
        assertTrue(what, ready())
    }

    @Test
    fun playing_a_movie_is_heard_and_pausing_stops_it() = runBlocking {
        Places.set(target)
        val audio = target.getSystemService(AudioManager::class.java)
        val scratch = File(target.cacheDir, "playback-${System.nanoTime()}").apply { mkdirs() }
        val still = File(scratch, "pattern.png")
        tests.assets.open("pattern.png").use { input -> still.outputStream().use { input.copyTo(it) } }
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
        try {
            withContext(Dispatchers.Main) {
                val first = EditorModel(createProject(Places.root), scope)
                until("the project opens") { !first.isOpening }
                first.importFile(still.path)
                val editor = first.beginExport()!!
                val movie = File(scratch, "playback.mp4")
                withContext(Dispatchers.Default) {
                    editor.export(
                        movie.path,
                        720u,
                        object : ExportProgress {
                            override fun progress(done: ULong, total: ULong) = true
                        },
                    )
                }
                first.endExport()

                val model = EditorModel(createProject(Places.root), scope)
                until("the second project opens") { !model.isOpening }
                model.importFile(movie.path)
                model.seek(0)
                until("nothing plays before play") { !audio.isMusicActive }

                model.play()
                until("the playhead moves") { model.playheadUs > 300_000 }
                until("Android hears the stream") { audio.isMusicActive }
                assertTrue(model.isPlaying)

                model.pause()
                until("the stream closes after pause") { !audio.isMusicActive }
                val held = model.playheadUs
                delay(300)
                assertTrue("the playhead holds when paused", model.playheadUs == held)
            }
        } finally {
            scope.cancel()
            scratch.deleteRecursively()
        }
    }
}
