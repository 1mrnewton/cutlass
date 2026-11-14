package com.scytheralpha.cutlass_android

import android.annotation.SuppressLint
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import uniffi.cutlass_ffi_bindings.Editor
import uniffi.cutlass_ffi_bindings.EditorException
import uniffi.cutlass_ffi_bindings.ToolBar
import uniffi.cutlass_ffi_bindings.ToolKind
import uniffi.cutlass_ffi_bindings.createProject
import uniffi.cutlass_ffi_bindings.listProjects
import uniffi.cutlass_ffi_bindings.toolBar

/** The app's calls into the Rust library, on the phone. */
@RunWith(AndroidJUnit4::class)
class RustBridgeTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    private fun scratch(): File =
        File(context.cacheDir, "bridge-${System.nanoTime()}").apply { mkdirs() }

    @Test
    fun a_new_project_is_listed_and_opens_empty() {
        val root = scratch()
        val bundle = File(root, "bundle").apply { mkdirs() }
        val path = createProject(root.path)
        val listed = listProjects(root.path)
        assertTrue(listed.any { it.path == path })
        val editor = Editor.open(root.path, bundle.path, path)
        assertEquals(0L, editor.durationUs())
        assertTrue(editor.lanes().all { it.clips.isEmpty() })
        root.deleteRecursively()
    }

    @Test
    fun a_file_that_is_not_media_is_refused() {
        val root = scratch()
        val bundle = File(root, "bundle").apply { mkdirs() }
        val editor = Editor.open(root.path, bundle.path, createProject(root.path))
        val junk = File(root, "junk.mp4").apply { writeText("not a movie") }
        var refused = false
        try {
            editor.import(junk.path)
        } catch (_: EditorException) {
            refused = true
        }
        assertTrue(refused)
        assertEquals(0L, editor.durationUs())
        root.deleteRecursively()
    }

    @SuppressLint("DiscouragedApi")
    @Test
    fun every_tool_on_every_bar_has_its_hugeicon() {
        val resources = context.resources
        for (bar in ToolBar.entries) {
            for (tool in toolBar(bar)) {
                val id = resources.getIdentifier(assetName(tool.kind.icon), "drawable", context.packageName)
                assertNotEquals("${tool.kind} has no drawable", 0, id)
            }
        }
        for (kind in ToolKind.entries) {
            val id = resources.getIdentifier(assetName(kind.icon), "drawable", context.packageName)
            assertNotEquals("$kind has no drawable", 0, id)
        }
    }
}
