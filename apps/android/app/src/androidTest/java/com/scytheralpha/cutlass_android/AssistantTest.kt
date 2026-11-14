package com.scytheralpha.cutlass_android

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
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
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import uniffi.cutlass_ffi_bindings.LineKind
import uniffi.cutlass_ffi_bindings.createProject

/**
 * The assistant on the phone: off until the cloud model has a key, the key
 * kept apart from the settings, and a turn that reaches the real service
 * over HTTPS from the device. The key here is not a real one, so the
 * service refuses it, and that refusal shows as the turn's notice.
 */
@RunWith(AndroidJUnit4::class)
class AssistantTest {
    private val target = InstrumentationRegistry.getInstrumentation().targetContext

    private suspend fun until(what: String, ready: () -> Boolean) {
        withTimeout(60_000) {
            while (!ready()) delay(50)
        }
        assertTrue(what, ready())
    }

    @Test
    fun the_assistant_asks_for_a_key_then_reaches_the_service() = runBlocking {
        Places.set(target)
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
        try {
            withContext(Dispatchers.Main) {
                val model = EditorModel(createProject(Places.root), scope)
                until("the project opens") { !model.isOpening }
                val cloud = model.assistantCloud()
                assertNotNull("the assistant opens with the project", cloud)
                // Start from no key, whatever an earlier run left.
                model.setAssistantCloud(cloud!!.copy(on = false, provider = "claude"), "")
                assertFalse(model.assistantCloud()!!.hasKey)

                model.ask("make it tall")
                until("the turn ends") { !model.assistantBusy && model.assistantLines.size >= 2 }
                assertEquals(LineKind.NOTICE, model.assistantLines.last().kind)
                assertTrue(model.assistantLines.last().text.contains("cloud model"))

                model.setAssistantCloud(
                    cloud.copy(on = true, provider = "claude", name = "claude-opus-5-5", url = ""),
                    "sk-ant-not-a-key",
                )
                assertTrue(model.assistantCloud()!!.hasKey)
                val settings = java.io.File(Places.root, "settings.json").readText()
                assertFalse("the key is not a setting", settings.contains("sk-ant-not-a-key"))

                model.clearAssistant()
                model.ask("make it tall")
                until("the service answers") {
                    !model.assistantBusy && model.assistantLines.lastOrNull()?.kind == LineKind.NOTICE
                }
                val notice = model.assistantLines.last().text
                assertTrue(notice, notice.contains("invalid x-api-key"))

                model.setAssistantCloud(cloud.copy(on = false), "")
            }
        } finally {
            scope.cancel()
        }
    }
}
