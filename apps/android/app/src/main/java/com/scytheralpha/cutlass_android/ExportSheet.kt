package com.scytheralpha.cutlass_android

import android.Manifest
import android.content.ClipData
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaScannerConnection
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.SheetValue
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import uniffi.cutlass_ffi_bindings.ExportProgress
import uniffi.cutlass_ffi_bindings.exportSides

/**
 * Export the whole cut as an MP4 and save it to the shared Movies folder.
 *
 * Rust draws, encodes and writes the file; this sheet picks the size,
 * shows the frames done, stops it on request, and hands the finished file
 * to Movies or the share sheet. The file is written in the cache folder
 * and removed when the sheet closes.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ExportSheet(model: EditorModel, name: String, onDismiss: () -> Unit) {
    // The activity's context, for the share chooser and the permission prompt.
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val sides = remember { exportSides() }
    val folder = remember { File(context.cacheDir, "exports") }
    val exportRun = remember { ExportSheetRun() }
    var side by remember { mutableStateOf(1080u) }
    var phase by remember { mutableStateOf<ExportSheetPhase>(ExportSheetPhase.Choosing) }
    val busy = phase is ExportSheetPhase.Writing || phase is ExportSheetPhase.Saving
    val busyNow = rememberUpdatedState(busy)
    val sheetState = rememberModalBottomSheetState(
        skipPartiallyExpanded = true,
        confirmValueChange = remember<(SheetValue) -> Boolean> {
            { it != SheetValue.Hidden || !busyNow.value }
        },
    )
    val permission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        exportRun.permission?.complete(granted)
        exportRun.permission = null
    }

    DisposableEffect(Unit) {
        onDispose {
            exportRun.closed = true
            val listener = exportRun.listener
            if (listener != null) listener.stop() else removeExportFolder(folder)
        }
    }

    val close: () -> Unit = {
        scope.launch { sheetState.hide() }.invokeOnCompletion {
            if (!sheetState.isVisible) onDismiss()
        }
    }

    /** Ask for storage access on Android 9, where Movies is a plain folder. */
    suspend fun askStorage(): Boolean {
        val waiter = CompletableDeferred<Boolean>()
        exportRun.permission = waiter
        permission.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        return waiter.await()
    }

    /** Write the movie off the main thread, then save it to Movies. */
    fun start() {
        val editor = model.beginExport()
        if (editor == null) {
            close()
            return
        }
        removeExportFolder(folder)
        val file = exportFileNamed(folder, name)
        val chosen = side
        phase = ExportSheetPhase.Writing(0.0)
        val listener = ExportSheetListener { share ->
            scope.launch {
                if (phase is ExportSheetPhase.Writing) phase = ExportSheetPhase.Writing(share)
            }
        }
        exportRun.listener = listener
        scope.launch {
            // The export holds the editor until it returns, even when the
            // sheet goes away; only then is the editor handed back.
            val written = try {
                withContext(Dispatchers.Default) {
                    runCatching { editor.export(file.path, chosen, listener) }
                }
            } finally {
                model.endExport()
                exportRun.listener = null
                if (exportRun.closed) removeExportFolder(folder)
            }
            val error = written.exceptionOrNull()
            phase = if (error == null) {
                phase = ExportSheetPhase.Saving
                val saved = saveExportToMovies(context, file) { askStorage() }
                ExportSheetPhase.Finished(file, saved)
            } else if (listener.stopped) {
                ExportSheetPhase.Choosing
            } else {
                ExportSheetPhase.Failed(error.message ?: error.toString())
            }
        }
    }

    ModalBottomSheet(
        onDismissRequest = { if (!busyNow.value) onDismiss() },
        sheetState = sheetState,
        sheetGesturesEnabled = !busy,
        containerColor = Palette.panel,
        contentColor = Palette.text,
        dragHandle = null,
    ) {
        // The sheet's own back handling hides it; while busy, back does nothing.
        BackHandler(enabled = busy) {}
        Box(
            Modifier
                .fillMaxWidth()
                .height(280.dp),
        ) {
            if (!busy) {
                Box(
                    Modifier
                        .align(Alignment.TopCenter)
                        .padding(top = 5.dp)
                        .size(width = 36.dp, height = 5.dp)
                        .background(Palette.white(0.35), CircleShape),
                )
            }
            Column(
                Modifier
                    .fillMaxWidth()
                    .padding(start = 20.dp, end = 20.dp, top = 16.dp, bottom = 24.dp),
                verticalArrangement = Arrangement.spacedBy(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                ExportSheetHeader(busy = busy, close = close)
                when (val now = phase) {
                    ExportSheetPhase.Choosing -> ExportSheetChoosing(
                        sides = sides,
                        side = side,
                        choose = { side = it },
                        start = { start() },
                    )
                    is ExportSheetPhase.Writing -> ExportSheetWriting(
                        share = now.share,
                        saving = false,
                        stop = { exportRun.listener?.stop() },
                    )
                    ExportSheetPhase.Saving -> ExportSheetWriting(share = 1.0, saving = true, stop = {})
                    is ExportSheetPhase.Finished -> ExportSheetFinished(
                        saved = now.saved,
                        share = {
                            runCatching { shareExportedMovie(context, now.file) }
                                .onFailure { model.show(it) }
                        },
                        done = close,
                    )
                    is ExportSheetPhase.Failed -> ExportSheetFailed(
                        message = now.message,
                        retry = { phase = ExportSheetPhase.Choosing },
                    )
                }
            }
        }
    }
}

private sealed interface ExportSheetPhase {
    data object Choosing : ExportSheetPhase

    /** The share of the frames written, 0 to 1. */
    data class Writing(val share: Double) : ExportSheetPhase

    data object Saving : ExportSheetPhase

    /** The file, and whether Movies took it. */
    data class Finished(val file: File, val saved: ExportSheetSave) : ExportSheetPhase

    data class Failed(val message: String) : ExportSheetPhase
}

/** How saving to Movies went. */
private enum class ExportSheetSave { Saved, Denied, Failed }

/** What the running export and the permission prompt leave behind for the sheet. */
private class ExportSheetRun {
    var listener: ExportSheetListener? = null

    /** The sheet has left the screen. */
    var closed = false
    var permission: CompletableDeferred<Boolean>? = null
}

@Composable
private fun ExportSheetHeader(busy: Boolean, close: () -> Unit) {
    Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
        Text("Export", color = Palette.text, fontWeight = FontWeight.SemiBold, fontSize = 17.sp)
        if (!busy) {
            Box(
                Modifier
                    .align(Alignment.CenterEnd)
                    .size(width = 40.dp, height = 32.dp)
                    .clickable(onClick = close)
                    .semantics {
                        contentDescription = "Close"
                        role = Role.Button
                    },
                contentAlignment = Alignment.Center,
            ) { HugeIcon("close", size = 20.dp, tint = Palette.text) }
        }
    }
}

@Composable
private fun ExportSheetChoosing(
    sides: List<UInt>,
    side: UInt,
    choose: (UInt) -> Unit,
    start: () -> Unit,
) {
    Column(
        Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(20.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        ExportSidePicker(sides = sides, chosen = side, choose = choose)
        Text("H.264 and AAC, saved to Movies", color = Palette.muted, fontSize = 13.sp)
        ExportWideButton("Export", start)
    }
}

/** The resolutions side by side, the chosen one raised, like a segmented picker. */
@Composable
private fun ExportSidePicker(sides: List<UInt>, chosen: UInt, choose: (UInt) -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .height(32.dp)
            .background(Palette.raised, CircleShape)
            .padding(2.dp)
            .selectableGroup(),
    ) {
        for (side in sides) {
            val on = side == chosen
            Box(
                Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .clip(CircleShape)
                    .background(if (on) Palette.white(0.36) else Palette.raised)
                    .selectable(selected = on, role = Role.Tab, onClick = { choose(side) }),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    exportSideLabel(side),
                    color = Palette.text,
                    fontSize = 13.sp,
                    fontWeight = if (on) FontWeight.SemiBold else FontWeight.Medium,
                )
            }
        }
    }
}

@Composable
private fun ExportSheetWriting(share: Double, saving: Boolean, stop: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .padding(top = 12.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(
            Modifier
                .fillMaxWidth()
                .height(4.dp)
                .clip(CircleShape)
                .background(Palette.raised),
        ) {
            Box(
                Modifier
                    .fillMaxWidth(share.toFloat().coerceIn(0f, 1f))
                    .fillMaxHeight()
                    .background(Palette.accent),
            )
        }
        Text(
            if (saving) "Saving to Movies" else "${(share * 100).toInt()}%",
            color = Palette.muted,
            fontSize = 13.sp,
            style = TextStyle(fontFeatureSettings = "tnum"),
        )
        if (!saving) {
            Text(
                "Stop",
                color = Palette.text,
                fontSize = 15.sp,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier
                    .padding(top = 8.dp)
                    .clickable(onClick = stop)
                    .semantics { role = Role.Button },
            )
        }
    }
}

@Composable
private fun ExportSheetFinished(saved: ExportSheetSave, share: () -> Unit, done: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .padding(top = 8.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            HugeIcon(
                "saved",
                size = 20.dp,
                tint = if (saved == ExportSheetSave.Saved) Palette.accent else Palette.muted,
            )
            Text(
                when (saved) {
                    ExportSheetSave.Saved -> "Saved to Movies"
                    ExportSheetSave.Denied -> "Storage access is off. Share the movie instead."
                    ExportSheetSave.Failed -> "The movie could not be saved. Share it instead."
                },
                color = Palette.text,
                fontSize = 15.sp,
                textAlign = TextAlign.Center,
            )
        }
        Row(
            Modifier
                .clickable(onClick = share)
                .semantics { role = Role.Button },
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            HugeIcon("share", size = 18.dp, tint = Palette.text)
            Text("Share", color = Palette.text, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
        }
        ExportWideButton("Done", done)
    }
}

@Composable
private fun ExportSheetFailed(message: String, retry: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .padding(top = 8.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(message, color = Palette.muted, fontSize = 15.sp, textAlign = TextAlign.Center)
        ExportWideButton("Try again", retry)
    }
}

@Composable
private fun ExportWideButton(title: String, action: () -> Unit) {
    Box(
        Modifier
            .fillMaxWidth()
            .height(46.dp)
            .clip(RoundedCornerShape(10.dp))
            .background(Palette.export)
            .clickable(onClick = action)
            .semantics { role = Role.Button },
        contentAlignment = Alignment.Center,
    ) {
        Text(title, color = Palette.exportInk, fontWeight = FontWeight.SemiBold, fontSize = 17.sp)
    }
}

private fun exportSideLabel(side: UInt): String = if (side >= 2160u) "4K" else "${side}p"

private fun removeExportFolder(folder: File) {
    folder.deleteRecursively()
}

/** A file in `folder` named after the project. */
private fun exportFileNamed(folder: File, name: String): File {
    val kept = name.filter { it.isLetterOrDigit() || it == ' ' || it == '-' }.trim()
    folder.mkdirs()
    return File(folder, (kept.ifEmpty { "Cutlass" }) + ".mp4")
}

/**
 * Hears Rust's progress on the editor's thread. Reports only whole
 * percents, so the main thread is not woken for every frame.
 */
private class ExportSheetListener(private val report: (Double) -> Unit) : ExportProgress {
    private val stopping = AtomicBoolean(false)
    private val percent = AtomicInteger(-1)

    val stopped: Boolean get() = stopping.get()

    fun stop() {
        stopping.set(true)
    }

    override fun progress(done: ULong, total: ULong): Boolean {
        val share = if (total == 0uL) 1.0 else done.toDouble() / total.toDouble()
        val whole = (share * 100).toInt()
        if (percent.getAndSet(whole) != whole) report(share)
        return !stopping.get()
    }
}

/**
 * Add the movie to Movies/Cutlass. Android 10 and later take it through
 * MediaStore without asking; Android 9 asks for storage access first and
 * writes the folder itself.
 */
private suspend fun saveExportToMovies(
    context: Context,
    file: File,
    askStorage: suspend () -> Boolean,
): ExportSheetSave {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        return withContext(Dispatchers.IO) { saveExportToMediaStore(context, file) }
    }
    val granted = ContextCompat.checkSelfPermission(
        context,
        Manifest.permission.WRITE_EXTERNAL_STORAGE,
    ) == PackageManager.PERMISSION_GRANTED || askStorage()
    if (!granted) return ExportSheetSave.Denied
    return withContext(Dispatchers.IO) { saveExportToMoviesFolder(context, file) }
}

/** Copy the movie into a pending MediaStore entry, then publish it. */
@RequiresApi(Build.VERSION_CODES.Q)
private fun saveExportToMediaStore(context: Context, file: File): ExportSheetSave {
    val resolver = context.contentResolver
    val values = ContentValues().apply {
        put(MediaStore.Video.Media.DISPLAY_NAME, file.name)
        put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
        put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/Cutlass")
        put(MediaStore.Video.Media.IS_PENDING, 1)
    }
    val collection = MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
    val uri = try {
        resolver.insert(collection, values)
    } catch (error: Exception) {
        Log.e("cutlass", "movies refused the movie: ${error.message}")
        null
    } ?: return ExportSheetSave.Failed
    return try {
        val sink = resolver.openOutputStream(uri) ?: error("no stream for $uri")
        sink.use { out -> file.inputStream().use { it.copyTo(out) } }
        val published = ContentValues().apply { put(MediaStore.Video.Media.IS_PENDING, 0) }
        resolver.update(uri, published, null, null)
        ExportSheetSave.Saved
    } catch (error: Exception) {
        Log.e("cutlass", "movies refused the movie: ${error.message}")
        runCatching { resolver.delete(uri, null, null) }
        ExportSheetSave.Failed
    }
}

/** Copy the movie into the Movies/Cutlass folder under a free name and have it scanned. */
private fun saveExportToMoviesFolder(context: Context, file: File): ExportSheetSave = try {
    @Suppress("DEPRECATION")
    val movies = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES)
    val folder = File(movies, "Cutlass").apply { mkdirs() }
    var target = File(folder, file.name)
    var count = 2
    while (target.exists()) {
        target = File(folder, "${file.nameWithoutExtension} $count.${file.extension}")
        count += 1
    }
    file.copyTo(target)
    MediaScannerConnection.scanFile(context, arrayOf(target.path), arrayOf("video/mp4"), null)
    ExportSheetSave.Saved
} catch (error: Exception) {
    Log.e("cutlass", "movies refused the movie: ${error.message}")
    ExportSheetSave.Failed
}

/** Offer the movie to other apps through the system share sheet. */
private fun shareExportedMovie(context: Context, file: File) {
    val uri = FileProvider.getUriForFile(context, context.packageName + ".files", file)
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "video/mp4"
        putExtra(Intent.EXTRA_STREAM, uri)
        clipData = ClipData.newRawUri(file.name, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(send, null))
}
