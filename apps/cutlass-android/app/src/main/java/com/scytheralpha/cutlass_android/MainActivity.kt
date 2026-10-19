package com.scytheralpha.cutlass_android

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.scytheralpha.cutlass_android.ui.theme.CutlassandroidTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            CutlassandroidTheme {
                Scaffold(modifier = Modifier.fillMaxWidth()) { innerPadding ->
                    Column(
                        modifier = Modifier
                            .padding(innerPadding)
                            .verticalScroll(rememberScrollState())
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(24.dp)
                    ) {
                        val context = LocalContext.current
                        VideoPickerPanel(context)
                        ScrubPanel("Preview — scrub (synthetic)") {
                            CutlassNative.previewOpenDemo()
                        }
                        ScrubPanel("Preview — scrub sample.mp4 (hardware decode)") {
                            val file = assetToCache(context, "sample.mp4")
                            CutlassNative.previewOpenVideo(file.absolutePath)
                        }
                        RenderPanel("Compositor demo (GPU)") {
                            bitmapFrom(CutlassNative.renderDemo(540, 960))
                        }
                    }
                }
            }
        }
    }
}

/** Build a [Bitmap] from the engine's `[width, height, argb…]` int array. */
private fun bitmapFrom(packed: IntArray): Bitmap? {
    if (packed.size < 2) return null
    val width = packed[0]
    val height = packed[1]
    if (width <= 0 || height <= 0 || packed.size != width * height + 2) return null
    return Bitmap.createBitmap(packed, 2, width, width, height, Bitmap.Config.ARGB_8888)
}

/** Copy a bundled asset to the cache dir (decoders need a real file path). */
private fun assetToCache(context: Context, name: String): File {
    val out = File(context.cacheDir, name)
    if (!out.exists() || out.length() == 0L) {
        context.assets.open(name).use { input ->
            out.outputStream().use { output -> input.copyTo(output) }
        }
    }
    return out
}

/**
 * Lets the user pick a video from the device (Android Photo Picker — no storage
 * permission needed) and scrubs it. The picked `content://` Uri is copied into
 * the cache so the native decoder can open it by path.
 */
@Composable
fun VideoPickerPanel(context: Context) {
    var pickedUri by remember { mutableStateOf<Uri?>(null) }
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri -> if (uri != null) pickedUri = uri }

    Column(modifier = Modifier.fillMaxWidth()) {
        Text(
            text = "Your video — scrub (hardware decode)",
            style = MaterialTheme.typography.titleMedium
        )
        Button(onClick = {
            launcher.launch(
                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)
            )
        }) {
            Text(if (pickedUri == null) "Pick a video" else "Pick a different video")
        }
        val uri = pickedUri
        if (uri != null) {
            ScrubPanel(title = "Picked video", key = uri) {
                val file = withContext(Dispatchers.IO) { uriToCache(context, uri) }
                if (file != null) CutlassNative.previewOpenVideo(file.absolutePath) else 0L
            }
        }
    }
}

/** Copy a `content://` video into a cache file the native decoder can open. */
private fun uriToCache(context: Context, uri: Uri): File? {
    return try {
        val out = File(context.cacheDir, "picked-${uri.hashCode()}.mp4")
        if (!out.exists() || out.length() == 0L) {
            context.contentResolver.openInputStream(uri)?.use { input ->
                out.outputStream().use { output -> input.copyTo(output) }
            } ?: return null
        }
        out
    } catch (t: Throwable) {
        null
    }
}

/**
 * Opens a native preview session and scrubs it with a slider.
 *
 * Renders run off the main thread through a single consumer loop fed by a
 * conflated channel: that inherently serializes renders (one in flight at a
 * time, latest-wins on rapid slider moves), which is exactly what the native
 * handle requires. The handle is closed in `finally` after the last render.
 */
@Composable
fun ScrubPanel(title: String, key: Any = title, openHandle: suspend () -> Long) {
    var duration by remember { mutableStateOf(0.0) }
    var time by remember { mutableStateOf(0f) }
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var failed by remember { mutableStateOf(false) }
    val requests = remember { Channel<Double>(Channel.CONFLATED) }

    LaunchedEffect(key) {
        // Re-keying (e.g. a newly picked video) reopens a fresh session, so
        // clear the previous frame/state first.
        duration = 0.0
        time = 0f
        bitmap = null
        failed = false
        val handle = try {
            withContext(Dispatchers.Default) { openHandle() }
        } catch (t: Throwable) {
            0L
        }
        if (handle == 0L) {
            failed = true
            return@LaunchedEffect
        }
        try {
            duration = withContext(Dispatchers.Default) {
                CutlassNative.previewDurationSeconds(handle)
            }
            requests.trySend(0.0)
            for (seconds in requests) {
                val packed = withContext(Dispatchers.Default) {
                    CutlassNative.previewRender(handle, seconds)
                }
                bitmapFrom(packed)?.let { bitmap = it }
            }
        } finally {
            // Runs after the in-flight render returns (single-threaded loop), so
            // there's no concurrent use of the handle when it's freed.
            withContext(NonCancellable + Dispatchers.Default) {
                CutlassNative.previewClose(handle)
            }
        }
    }

    Column(modifier = Modifier.fillMaxWidth()) {
        Text(text = title, style = MaterialTheme.typography.titleMedium)
        val current = bitmap
        when {
            current != null -> Image(
                bitmap = current.asImageBitmap(),
                contentDescription = title,
                contentScale = ContentScale.Fit,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 120.dp)
            )

            failed -> Text("Preview unavailable", color = MaterialTheme.colorScheme.error)
            else -> CircularProgressIndicator()
        }
        if (duration > 0.0) {
            Slider(
                value = time,
                valueRange = 0f..duration.toFloat(),
                onValueChange = {
                    time = it
                    requests.trySend(it.toDouble())
                }
            )
            Text(
                text = String.format("%.2f s / %.2f s", time, duration),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

/** Renders one frame off the main thread and shows it, with loading/error states. */
@Composable
fun RenderPanel(title: String, render: suspend () -> Bitmap?) {
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var failed by remember { mutableStateOf(false) }

    LaunchedEffect(title) {
        val rendered = try {
            withContext(Dispatchers.Default) { render() }
        } catch (t: Throwable) {
            null
        }
        if (rendered != null) bitmap = rendered else failed = true
    }

    Column(modifier = Modifier.fillMaxWidth()) {
        Text(text = title, style = MaterialTheme.typography.titleMedium)
        val current = bitmap
        when {
            current != null -> Image(
                bitmap = current.asImageBitmap(),
                contentDescription = title,
                contentScale = ContentScale.Fit,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 120.dp)
            )

            failed -> Text("Render failed", color = MaterialTheme.colorScheme.error)
            else -> CircularProgressIndicator()
        }
    }
}
