package com.scytheralpha.cutlass_android

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import uniffi.cutlass_ffi_bindings.EditorException
import uniffi.cutlass_ffi_bindings.ProjectSummary
import uniffi.cutlass_ffi_bindings.createProject
import uniffi.cutlass_ffi_bindings.listProjects

/**
 * The launch screen: a new-project button and the saved projects, newest
 * first, as on the desktop. A tap opens the project.
 */
@Composable
fun ProjectsScreen(onOpen: (ProjectSummary) -> Unit) {
    var projects by remember { mutableStateOf(emptyList<ProjectSummary>()) }
    var failure by remember { mutableStateOf<String?>(null) }
    fun reload() {
        projects = listProjects(Places.root)
    }
    LaunchedEffect(Unit) { reload() }

    Column(
        Modifier
            .fillMaxSize()
            .background(Palette.panel)
            .windowInsetsPadding(WindowInsets.safeDrawing),
    ) {
        TitleBar(title = "Projects")
        LazyVerticalGrid(
            columns = GridCells.Adaptive(160.dp),
            modifier = Modifier.fillMaxSize(),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item(span = { GridItemSpan(maxLineSpan) }) {
                Column(verticalArrangement = Arrangement.spacedBy(20.dp)) {
                    Box(
                        Modifier
                            .fillMaxWidth()
                            .height(48.dp)
                            .background(Palette.export, RoundedCornerShape(8.dp))
                            .clickable {
                                try {
                                    val path = createProject(Places.root)
                                    failure = null
                                    reload()
                                    projects.firstOrNull { it.path == path }?.let(onOpen)
                                } catch (error: EditorException) {
                                    failure = error.message
                                }
                            },
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            "New project",
                            color = Palette.exportInk,
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 17.sp,
                        )
                    }
                    failure?.let { Text(it, color = Color.Red, fontSize = 13.sp) }
                    if (projects.isEmpty()) {
                        Text("No projects yet.", color = Palette.muted, fontSize = 15.sp)
                    }
                }
            }
            items(projects, key = { it.path }) { project ->
                ProjectCard(project, Modifier.clickable { onOpen(project) })
            }
        }
    }
}

/** A slim bar with a centered title, and optional buttons on each side. */
@Composable
fun TitleBar(
    title: String,
    background: Color = Palette.bar,
    leading: @Composable RowScope.() -> Unit = {},
    trailing: @Composable RowScope.() -> Unit = {},
) {
    Box(
        Modifier
            .fillMaxWidth()
            .background(background)
            .height(48.dp),
    ) {
        Text(
            title,
            color = Palette.text,
            fontWeight = FontWeight.SemiBold,
            fontSize = 17.sp,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier
                .align(Alignment.Center)
                .padding(horizontal = 80.dp),
        )
        Row(
            Modifier
                .align(Alignment.CenterStart)
                .padding(horizontal = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            content = leading,
        )
        Row(
            Modifier
                .align(Alignment.CenterEnd)
                .padding(horizontal = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            content = trailing,
        )
    }
}

/** One saved project: its name, then its length and the day it changed. */
@Composable
fun ProjectCard(project: ProjectSummary, modifier: Modifier = Modifier) {
    val shape = RoundedCornerShape(12.dp)
    Column(
        modifier
            .fillMaxWidth()
            .background(Palette.bar, shape)
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            project.name,
            color = Palette.text,
            fontWeight = FontWeight.Medium,
            fontSize = 15.sp,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
        val day = dayFormat().format(Date(project.modifiedMs))
        val meta = if (project.durationUs > 0) "${length(project.durationUs)}  $day" else day
        Text(meta, color = Palette.muted, fontSize = 12.sp)
    }
}

private fun dayFormat() = SimpleDateFormat("yyyy-MM-dd", Locale.US).apply {
    timeZone = TimeZone.getTimeZone("UTC")
}

/** "4.2s" under a minute, "1:05.3" from a minute. */
fun length(micros: Long): String {
    val millis = micros / 1_000
    val minutes = millis / 60_000
    val whole = millis % 60_000 / 1_000
    val tenth = millis % 1_000 / 100
    return if (minutes > 0) {
        String.format(Locale.US, "%d:%02d.%d", minutes, whole, tenth)
    } else {
        "$whole.${tenth}s"
    }
}
