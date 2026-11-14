package com.scytheralpha.cutlass_android

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.io.File
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import uniffi.cutlass_ffi_bindings.ProjectSummary
import uniffi.cutlass_ffi_bindings.ToolBar
import uniffi.cutlass_ffi_bindings.ToolItem
import uniffi.cutlass_ffi_bindings.ToolKind
import uniffi.cutlass_ffi_bindings.clock
import uniffi.cutlass_ffi_bindings.toolBar

/** The tool panel open over the tool bar. */
sealed interface Panel {
    data object Volume : Panel
    data object Speed : Panel
    data object Effects : Panel
    data object Ratio : Panel
    data object Background : Panel
    data object Graphs : Panel
    data object Animations : Panel
    data object Fade : Panel

    /** Add text, or change the text clip with this id. */
    data class Text(val clip: String?) : Panel
}

/** What a tool asks the system to pick. */
enum class Pick { OVERLAY, SOUND }

/** The editor: preview on top, the transport, the timeline, then the tools. */
@Composable
fun EditorScreen(project: ProjectSummary, home: () -> Unit) {
    val scope = rememberCoroutineScope()
    val model = remember(project.path) { EditorModel(project.path, scope) }
    // Where the playhead was, kept across Android ending the process in the
    // background. The project reopens there with nothing selected.
    var keptPlayhead by rememberSaveable(project.path) { mutableLongStateOf(0L) }
    var restored by remember(project.path) { mutableStateOf(false) }
    LaunchedEffect(model.isOpening) {
        if (!model.isOpening && !restored) {
            restored = true
            if (keptPlayhead > 0) model.seek(keptPlayhead)
        }
    }
    LaunchedEffect(model.playheadUs) {
        if (restored) keptPlayhead = model.playheadUs
    }
    var panel by remember { mutableStateOf<Panel?>(null) }
    // The bar shown while nothing is selected.
    var openBar by remember { mutableStateOf(ToolBar.ROOT) }
    var exporting by remember { mutableStateOf(false) }
    val context = LocalContext.current

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickMultipleVisualMedia(),
    ) { uris -> importPicked(context, model, scope, uris) }
    val overlayPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia(),
    ) { uri -> importPicked(context, model, scope, listOfNotNull(uri), overlay = true) }
    val soundPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent(),
    ) { uri -> importPicked(context, model, scope, listOfNotNull(uri)) }
    val pick = { what: Pick ->
        when (what) {
            Pick.OVERLAY -> overlayPicker.launch(
                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageAndVideo),
            )
            Pick.SOUND -> soundPicker.launch("audio/*")
        }
    }

    DisposableEffect(model) { onDispose { model.pause() } }
    // System back steps out one layer at a time: the open panel, then the
    // selection, then a sub bar. Only then does it leave for the projects.
    BackHandler(enabled = panel != null || model.selectedClip != null || openBar != ToolBar.ROOT) {
        when {
            panel != null -> panel = null
            model.selectedClip != null -> model.selectedClip = null
            else -> openBar = ToolBar.ROOT
        }
    }
    LaunchedEffect(model.selectedClip) {
        if (model.selectedClip == null && panel !is Panel.Text && panel != Panel.Effects &&
            panel != Panel.Ratio && panel != Panel.Background
        ) {
            panel = null
        }
    }

    Box(
        Modifier
            .fillMaxSize()
            .background(Palette.editor)
            .windowInsetsPadding(WindowInsets.safeDrawing),
    ) {
        Column(Modifier.fillMaxSize()) {
            TitleBar(
                title = project.name,
                background = Palette.editor,
                leading = {
                    Box(
                        Modifier
                            .size(44.dp)
                            .clickable {
                                model.pause()
                                home()
                            }
                            .semantics { contentDescription = "Home" },
                        contentAlignment = Alignment.Center,
                    ) { HugeIcon("home", tint = Palette.text) }
                },
                trailing = {
                    val enabled = model.hasClips && !model.isOpening
                    Text(
                        "Export",
                        color = Palette.exportInk,
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 15.sp,
                        modifier = Modifier
                            .alpha(if (model.hasClips) 1f else 0.4f)
                            .background(Palette.export, CircleShape)
                            .clickable(enabled = enabled) {
                                model.pause()
                                exporting = true
                            }
                            .padding(horizontal = 12.dp, vertical = 5.dp),
                    )
                },
            )
            // The transport, timeline and tool bar under the preview, or
            // beside it on a phone held sideways, where a preview on top
            // would leave the timeline one row tall.
            val controls: @Composable ColumnScope.() -> Unit = {
                Transport(model) { panel = Panel.Graphs }
                TimelineStrip(
                    model = model,
                    onAdd = {
                        picker.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageAndVideo),
                        )
                    },
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .background(Palette.strip),
                )
                Bottom(
                    model = model,
                    panel = panel,
                    setPanel = { panel = it },
                    openBar = openBar,
                    setOpenBar = { openBar = it },
                    pick = pick,
                )
            }
            val configuration = LocalConfiguration.current
            if (configuration.screenWidthDp > configuration.screenHeightDp) {
                Row(Modifier.weight(1f).fillMaxWidth()) {
                    Preview(model, Modifier.weight(0.42f).fillMaxHeight()) { panel = Panel.Text(it) }
                    Column(Modifier.weight(0.58f).fillMaxHeight(), content = controls)
                }
            } else {
                Preview(model) { panel = Panel.Text(it) }
                controls()
            }
        }
        NoticeBanner(model, Modifier.align(Alignment.TopCenter))
    }
    if (exporting) {
        ExportSheet(model = model, name = project.name, onDismiss = { exporting = false })
    }
}

/**
 * The preview. Given no `area`, it fits 16:10 into the width but never past
 * this share of the screen's height; given one, as beside the timeline on a
 * phone held sideways, it fills it. The canvas inside keeps its own ratio
 * either way.
 */
@Composable
private fun Preview(model: EditorModel, area: Modifier? = null, editText: (String) -> Unit) {
    val tallest = (LocalConfiguration.current.screenHeightDp * 0.45f).dp
    BoxWithConstraints(area ?: Modifier.fillMaxWidth()) {
        val canvasFirst = area == null && model.canvasAspect < 1.6f ||
            area != null && model.canvasAspect < maxWidth / maxHeight
        Box(
            Modifier
                .fillMaxWidth()
                .then(if (area == null) Modifier.height(minOf(maxWidth / 1.6f, tallest)) else Modifier.fillMaxHeight())
                // A box larger than the preview stays inside it.
                .clipToBounds()
                .background(Palette.editor),
            contentAlignment = Alignment.Center,
        ) {
            // Lighter than the canvas, so the canvas shows its ratio.
            Box(
                Modifier
                    .aspectRatio(model.canvasAspect, matchHeightConstraintsFirst = canvasFirst)
                    .background(Color.Black),
            ) {
                model.picture?.let {
                    Image(
                        it,
                        contentDescription = null,
                        contentScale = ContentScale.Fit,
                        modifier = Modifier.fillMaxSize(),
                    )
                }
                if (!model.isPlaying) {
                    StageOverlay(model = model, editText = editText)
                }
            }
            if (model.isOpening || model.isImporting) {
                CircularProgressIndicator(color = Color.White, modifier = Modifier.size(32.dp))
            }
        }
    }
}

/**
 * The clock on the left, play pinned to the middle so it never moves with
 * the clock, undo and redo on the right. A selected picture or title adds
 * the keyframe button before undo, and the Graphs button when the playhead
 * is between two of its keyframes.
 */
@Composable
private fun Transport(model: EditorModel, openGraphs: () -> Unit) {
    CompositionLocalProvider(LocalContentColor provides Color.White) {
        Box(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 10.dp),
        ) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    clock(model.playheadUs) + " / " + clock(model.durationUs),
                    color = Color.White.copy(alpha = 0.7f),
                    fontSize = 13.sp,
                    style = androidx.compose.ui.text.TextStyle(
                        fontFeatureSettings = "tnum",
                    ),
                )
                Spacer(Modifier.weight(1f))
                val keys = model.keys
                if (keys != null && keys.inSpan) {
                    IconButton("graphs", "Graphs", onClick = openGraphs)
                    Spacer(Modifier.width(16.dp))
                }
                if (keys != null) {
                    Box(contentAlignment = Alignment.Center) {
                        IconButton(
                            "keyframe",
                            if (keys.here) "Remove keyframe" else "Add keyframe",
                        ) { model.toggleKeyframe() }
                        Text(
                            if (keys.here) "−" else "+",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White,
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .offset(x = 5.dp, y = (-5).dp),
                        )
                    }
                    Spacer(Modifier.width(16.dp))
                }
                IconButton("undo", "Undo") { model.undo() }
                Spacer(Modifier.width(16.dp))
                IconButton("redo", "Redo") { model.redo() }
            }
            Box(Modifier.align(Alignment.Center).alpha(if (model.hasClips) 1f else 0.4f)) {
                IconButton(
                    if (model.isPlaying) "pause" else "play",
                    if (model.isPlaying) "Pause" else "Play",
                    size = 24,
                    enabled = model.hasClips,
                ) { model.togglePlay() }
            }
        }
    }
}

/** A bare glyph that answers a tap, labelled for TalkBack. */
@Composable
fun IconButton(
    icon: String,
    label: String,
    size: Int = 22,
    enabled: Boolean = true,
    onClick: () -> Unit,
) {
    Box(
        Modifier
            .clickable(enabled = enabled, onClick = onClick)
            .padding(2.dp)
            .semantics { contentDescription = label },
        contentAlignment = Alignment.Center,
    ) { HugeIcon(icon, size = size.dp) }
}

/**
 * A panel when one is open, the selected clip's bar, else the bar the root
 * tools opened. Rust lists every bar.
 */
@Composable
private fun Bottom(
    model: EditorModel,
    panel: Panel?,
    setPanel: (Panel?) -> Unit,
    openBar: ToolBar,
    setOpenBar: (ToolBar) -> Unit,
    pick: (Pick) -> Unit,
) {
    val done = { setPanel(null) }
    val selectionBar = model.selectionBar
    val selection = model.selection
    // One key per thing the bottom can show, so a change slides.
    val key: Any = when {
        panel is Panel.Text -> "text:${panel.clip}"
        panel != null && panel !is Panel.Volume && panel !is Panel.Speed -> panel
        panel != null && selection != null -> panel
        selectionBar != null -> selectionBar
        else -> "open:$openBar"
    }
    AnimatedContent(
        targetState = key,
        transitionSpec = {
            (slideInVertically(tween(200)) { it / 3 } + fadeIn(tween(200))) togetherWith
                fadeOut(tween(120))
        },
        label = "bottom",
    ) { _ ->
        val use: (ToolItem) -> Unit = { tool ->
            useTool(tool, model, setPanel, setOpenBar, pick)
        }
        when {
            panel == Panel.Effects -> EffectsPanel(model, done)
            panel == Panel.Ratio -> RatioPanel(model, done)
            panel == Panel.Background -> BackgroundPanel(model, done)
            panel == Panel.Animations -> AnimationsPanel(model, done)
            panel == Panel.Graphs -> GraphsPanel(model, done)
            panel is Panel.Text -> TextPanel(model, panel.clip, done)
            panel == Panel.Volume && selection != null -> VolumePanel(model, selection, done)
            panel == Panel.Speed && selection != null -> SpeedPanel(model, selection, done)
            panel == Panel.Fade && selection != null -> FadePanel(model, selection, done)
            selectionBar != null -> ToolBarView(
                tools = toolBar(selectionBar),
                back = { model.selectedClip = null },
                tap = use,
            )
            else -> ToolBarView(
                tools = toolBar(openBar),
                back = if (openBar == ToolBar.ROOT) null else ({ setOpenBar(ToolBar.ROOT) }),
                tap = use,
            )
        }
    }
}

/** Open a bar, act, or say the tool is not ready yet. */
private fun useTool(
    tool: ToolItem,
    model: EditorModel,
    setPanel: (Panel?) -> Unit,
    setOpenBar: (ToolBar) -> Unit,
    pick: (Pick) -> Unit,
) {
    tool.opens?.let {
        setOpenBar(it)
        return
    }
    if (!tool.ready) {
        model.say("${tool.label} is coming soon")
        return
    }
    when (tool.kind) {
        ToolKind.EDIT -> model.editUnderPlayhead()
        ToolKind.SPLIT -> model.split()
        ToolKind.VOLUME -> setPanel(Panel.Volume)
        ToolKind.SPEED -> setPanel(Panel.Speed)
        ToolKind.ANIMATIONS -> setPanel(Panel.Animations)
        ToolKind.DELETE -> model.deleteClip()
        ToolKind.ADD_TEXT -> setPanel(Panel.Text(null))
        ToolKind.EDIT_TEXT -> setPanel(Panel.Text(model.selectedClip))
        ToolKind.MIRROR -> model.selectedClip?.let(model::toggleMirror)
        ToolKind.ROTATE -> model.selectedClip?.let(model::turn)
        ToolKind.REVERSE -> model.selectedClip?.let(model::toggleReverse)
        ToolKind.CROP -> model.selectedClip?.let(model::toggleCrop)
        ToolKind.FREEZE -> model.selectedClip?.let(model::freeze)
        ToolKind.EXTRACT_AUDIO -> model.selectedClip?.let(model::extractAudio)
        ToolKind.BEATS -> model.selectedClip?.let(model::toggleBeat)
        ToolKind.SPLICE -> model.selectedClip?.let(model::toMain)
        ToolKind.FADE -> setPanel(Panel.Fade)
        ToolKind.ADD_OVERLAY -> pick(Pick.OVERLAY)
        ToolKind.IMPORT_SOUND -> pick(Pick.SOUND)
        ToolKind.VIDEO_EFFECTS -> setPanel(Panel.Effects)
        ToolKind.ASPECT_RATIO -> {
            model.selectedClip = null
            setPanel(Panel.Ratio)
        }
        ToolKind.BACKGROUND -> {
            model.selectedClip = null
            setPanel(Panel.Background)
        }
        else -> model.say("${tool.label} is coming soon")
    }
}

/**
 * One tool bar as Rust lists it: a back chevron when there is somewhere to
 * go back to, then the tools, scrolling sideways. A tool that does not work
 * yet is dimmed but still answers a tap.
 */
@Composable
fun ToolBarView(tools: List<ToolItem>, back: (() -> Unit)?, tap: (ToolItem) -> Unit) {
    CompositionLocalProvider(LocalContentColor provides Color.White) {
        Row(
            Modifier
                .fillMaxWidth()
                .background(Palette.tools)
                .padding(top = 10.dp, bottom = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            if (back != null) {
                Box(
                    Modifier
                        .padding(start = 8.dp)
                        .size(width = 40.dp, height = 52.dp)
                        .background(Palette.raised, RoundedCornerShape(6.dp))
                        .clickable(onClick = back)
                        .semantics { contentDescription = "Back" },
                    contentAlignment = Alignment.Center,
                ) { HugeIcon("chevronLeft", size = 20.dp) }
            }
            Row(
                Modifier
                    .horizontalScroll(rememberScrollState())
                    .padding(horizontal = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                for (tool in tools) {
                    Column(
                        Modifier
                            .size(width = 68.dp, height = 52.dp)
                            .alpha(if (tool.ready) 1f else 0.4f)
                            .clickable { tap(tool) },
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(4.dp, Alignment.CenterVertically),
                    ) {
                        HugeIcon(tool.kind.icon)
                        Text(
                            tool.label,
                            fontSize = 11.sp,
                            maxLines = 1,
                            textAlign = TextAlign.Center,
                            color = Color.White,
                        )
                    }
                }
            }
        }
    }
}

/** The name in `crates/icons/icons.txt`. */
val ToolKind.icon: String
    get() = when (this) {
        ToolKind.EDIT -> "scissors"
        ToolKind.AUDIO -> "audio"
        ToolKind.TEXT -> "text"
        ToolKind.EFFECTS -> "effects"
        ToolKind.OVERLAY -> "overlay"
        ToolKind.CAPTIONS -> "captions"
        ToolKind.FILTERS -> "filters"
        ToolKind.ADJUST -> "adjustments"
        ToolKind.STICKERS -> "stickers"
        ToolKind.ASPECT_RATIO -> "aspectRatio"
        ToolKind.BACKGROUND -> "background"
        ToolKind.SPLIT -> "split"
        ToolKind.VOLUME -> "volume"
        ToolKind.SPEED -> "speed"
        ToolKind.ANIMATIONS -> "animations"
        ToolKind.DELETE -> "delete"
        ToolKind.DUPLICATE -> "duplicate"
        ToolKind.REPLACE -> "replace"
        ToolKind.CROP -> "crop"
        ToolKind.MIRROR -> "mirror"
        ToolKind.ROTATE -> "rotate"
        ToolKind.REVERSE -> "reverse"
        ToolKind.FREEZE -> "freeze"
        ToolKind.EXTRACT_AUDIO -> "extractAudio"
        ToolKind.FADE -> "fade"
        ToolKind.OPACITY -> "opacity"
        ToolKind.BEATS -> "beat"
        ToolKind.UNLINK -> "unlink"
        ToolKind.SPLICE -> "splice"
        ToolKind.ADD_OVERLAY -> "addOverlay"
        ToolKind.IMPORT_SOUND -> "importSound"
        ToolKind.RECORD -> "record"
        ToolKind.ADD_TEXT -> "addText"
        ToolKind.EDIT_TEXT -> "editText"
        ToolKind.DRAW -> "draw"
        ToolKind.VIDEO_EFFECTS -> "plugin"
        ToolKind.SHAPE_STYLE -> "shapes"
    }

@Composable
private fun NoticeBanner(model: EditorModel, modifier: Modifier) {
    val notice = model.notice
    AnimatedVisibility(
        visible = notice != null,
        modifier = modifier,
        enter = slideInVertically { -it } + fadeIn(),
        exit = slideOutVertically { -it } + fadeOut(),
    ) {
        val shown = notice ?: return@AnimatedVisibility
        Text(
            shown.text,
            color = Color.Black,
            fontSize = 13.sp,
            fontWeight = FontWeight.Medium,
            maxLines = 2,
            modifier = Modifier
                .padding(top = 8.dp)
                .background(Color.White, CircleShape)
                .padding(horizontal = 14.dp, vertical = 8.dp),
        )
    }
    LaunchedEffect(notice?.id) {
        if (notice == null) return@LaunchedEffect
        delay(2_500)
        if (model.notice == notice) model.clearNotice()
    }
}

/**
 * Copy each picked item out of the photo picker into a fresh scratch folder
 * that keeps its own name, hand it to Rust, then remove the copy. Rust reads
 * the file: what it is, its upright size, and its length.
 */
private fun importPicked(
    context: Context,
    model: EditorModel,
    scope: kotlinx.coroutines.CoroutineScope,
    uris: List<Uri>,
    overlay: Boolean = false,
) {
    if (uris.isEmpty()) return
    model.beginImport()
    scope.launch {
        for (uri in uris) {
            val copy = withContext(Dispatchers.IO) { runCatching { copyOut(context, uri) } }
            copy.onSuccess { file ->
                if (overlay) model.importOverlay(file.path) else model.importFile(file.path)
                file.parentFile?.deleteRecursively()
            }.onFailure { model.show(it) }
        }
        model.endImport()
    }
}

private fun copyOut(context: Context, uri: Uri): File {
    val name = context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
        ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
        ?.replace('/', '_')
        ?: "picked"
    val folder = File(context.cacheDir, UUID.randomUUID().toString()).apply { mkdirs() }
    val file = File(folder, name)
    val input = context.contentResolver.openInputStream(uri)
        ?: throw IllegalStateException("The item could not be read")
    input.use { source -> file.outputStream().use { source.copyTo(it) } }
    return file
}
