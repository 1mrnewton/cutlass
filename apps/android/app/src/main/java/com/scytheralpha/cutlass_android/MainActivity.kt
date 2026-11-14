package com.scytheralpha.cutlass_android

import android.content.Context
import android.os.Bundle
import android.util.Log
import android.graphics.Color as AndroidColor
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import java.io.File
import uniffi.cutlass_ffi_bindings.ProjectSummary
import uniffi.cutlass_ffi_bindings.listProjects
import uniffi.cutlass_ffi_bindings.start

/**
 * The two paths the app hands to Rust. Rust lays out everything under the
 * root: the projects, the cache, the downloads, the proxies, the logs.
 */
object Places {
    /** The one Cutlass folder, private to the app. */
    lateinit var root: String
        private set

    /**
     * The app's read-only resources, as a folder. An APK's assets are not
     * one, so this is an empty folder beside the root.
     */
    lateinit var bundle: String
        private set

    /** Where the system keeps the faces text can be set in. */
    val fontFolders = listOf("/system/fonts")

    fun set(context: Context) {
        val root = File(context.filesDir, "Cutlass").apply { mkdirs() }
        val bundle = File(context.noBackupFilesDir, "bundle").apply { mkdirs() }
        this.root = root.path
        this.bundle = bundle.path
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Places.set(this)
        try {
            start(Places.root)
        } catch (error: Exception) {
            Log.e("cutlass", error.message ?: "start failed")
        }
        // The app is always dark, so the bars draw light icons on it.
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(AndroidColor.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.dark(AndroidColor.TRANSPARENT),
        )
        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    primary = Palette.accent,
                    background = Palette.editor,
                    surface = Palette.panel,
                ),
            ) {
                AppScreen()
            }
        }
    }
}

/**
 * The whole app: the project list, or one open project in the editor.
 * There is no stack to go back through; the editor's home button and the
 * system back gesture return here, and the list is read again.
 */
@Composable
fun AppScreen() {
    var openPath by rememberSaveable { mutableStateOf<String?>(null) }
    AnimatedContent(
        targetState = openPath,
        transitionSpec = { fadeIn(tween(200)) togetherWith fadeOut(tween(200)) },
        label = "screen",
    ) { path ->
        val project = path?.let { wanted ->
            listProjects(Places.root).firstOrNull { it.path == wanted }
        }
        if (project != null) {
            BackHandler { openPath = null }
            EditorScreen(project = project, home = { openPath = null })
        } else {
            ProjectsScreen(onOpen = { summary: ProjectSummary -> openPath = summary.path })
        }
    }
}
