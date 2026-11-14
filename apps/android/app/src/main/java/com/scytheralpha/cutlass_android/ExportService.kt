package com.scytheralpha.cutlass_android

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat

/**
 * Keeps the app in the foreground while a movie exports, so leaving the
 * app does not let Android stop it, and shows its progress with a Stop
 * button in a notification.
 *
 * The export itself runs where the sheet started it. This service only
 * holds the process up and draws the notification; the sheet tells it the
 * progress and stops it when the export ends, however it ends.
 */
class ExportService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == STOP) {
            onStop?.invoke()
            return START_NOT_STICKY
        }
        channel(this)
        val type = when {
            Build.VERSION.SDK_INT >= 35 -> ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q -> ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            else -> 0
        }
        ServiceCompat.startForeground(this, NOTIFICATION, notification(this, share), type)
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        running = false
        super.onDestroy()
    }

    companion object {
        private const val CHANNEL = "export"
        private const val NOTIFICATION = 1
        internal const val STOP = "com.scytheralpha.cutlass_android.STOP_EXPORT"

        @Volatile private var running = false
        @Volatile private var share = 0.0
        @Volatile private var title = ""
        @Volatile private var onStop: (() -> Unit)? = null

        /** Start holding the app up for an export of `name`. `stop` ends it early. */
        fun begin(context: Context, name: String, stop: () -> Unit) {
            title = name
            share = 0.0
            onStop = stop
            running = true
            ContextCompat.startForegroundService(context, Intent(context, ExportService::class.java))
        }

        /** Show `fraction` of the movie written. */
        fun progress(context: Context, fraction: Double) {
            share = fraction
            if (!running) return
            val manager = NotificationManagerCompat.from(context)
            if (manager.areNotificationsEnabled()) {
                runCatching { manager.notify(NOTIFICATION, notification(context, fraction)) }
            }
        }

        /** The export ended, written, stopped or failed: let the app go. */
        fun end(context: Context) {
            onStop = null
            running = false
            context.stopService(Intent(context, ExportService::class.java))
        }

        private fun channel(context: Context) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
            val manager = context.getSystemService(NotificationManager::class.java) ?: return
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL, "Export", NotificationManager.IMPORTANCE_LOW).apply {
                    description = "Progress while a movie exports"
                },
            )
        }

        private fun notification(context: Context, fraction: Double) =
            NotificationCompat.Builder(context, CHANNEL)
                .setSmallIcon(R.drawable.icon_export)
                .setContentTitle("Exporting ${title.ifEmpty { "movie" }}")
                .setContentText("${(fraction * 100).toInt()}%")
                .setProgress(100, (fraction * 100).toInt(), false)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setSilent(true)
                .setContentIntent(
                    PendingIntent.getActivity(
                        context,
                        0,
                        Intent(context, MainActivity::class.java)
                            .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT),
                        PendingIntent.FLAG_IMMUTABLE,
                    ),
                )
                .addAction(
                    0,
                    "Stop",
                    PendingIntent.getService(
                        context,
                        1,
                        Intent(context, ExportService::class.java).setAction(STOP),
                        PendingIntent.FLAG_IMMUTABLE,
                    ),
                )
                .build()
    }
}
