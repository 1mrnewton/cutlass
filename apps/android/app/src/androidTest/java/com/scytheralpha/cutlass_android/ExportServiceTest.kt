package com.scytheralpha.cutlass_android

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The export's foreground service: it holds the app up while an export
 * runs, its notification's Stop reaches the export, and it goes away when
 * the export ends.
 */
@RunWith(AndroidJUnit4::class)
class ExportServiceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Suppress("DEPRECATION")
    private fun foreground(): Boolean =
        context.getSystemService(ActivityManager::class.java)
            .getRunningServices(Int.MAX_VALUE)
            .any { it.service.className == ExportService::class.java.name && it.foreground }

    private fun waitFor(what: String, check: () -> Boolean) {
        val deadline = System.currentTimeMillis() + 10_000
        while (!check() && System.currentTimeMillis() < deadline) Thread.sleep(50)
        assertTrue(what, check())
    }

    @Test
    fun stop_in_the_notification_reaches_the_export_and_the_end_lets_go() {
        val stopped = CountDownLatch(1)
        ExportService.begin(context, "Test") { stopped.countDown() }
        waitFor("the service runs in the foreground") { foreground() }
        ExportService.progress(context, 0.5)

        context.startService(Intent(context, ExportService::class.java).setAction(ExportService.STOP))
        assertTrue("Stop reached the export", stopped.await(10, TimeUnit.SECONDS))

        ExportService.end(context)
        waitFor("the service is gone") { !foreground() }
        assertFalse(foreground())
    }
}
