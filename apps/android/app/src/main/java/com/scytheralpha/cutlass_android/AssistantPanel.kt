package com.scytheralpha.cutlass_android

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import uniffi.cutlass_ffi_bindings.AssistantLine
import uniffi.cutlass_ffi_bindings.CloudChoice
import uniffi.cutlass_ffi_bindings.LineKind

/**
 * The assistant: the conversation, a box to ask in, and Stop while it
 * works. Until the cloud model is on with a key, it shows its settings
 * instead. Rust runs the model and the tools; this only shows the lines.
 */
@Composable
fun AssistantPanel(model: EditorModel, done: () -> Unit) {
    var cloud by remember { mutableStateOf(model.assistantCloud()) }
    var setting by remember { mutableStateOf(cloud?.let { !it.on || !it.hasKey } ?: true) }
    PanelFrame(title = "Assistant", close = done) {
        val current = cloud
        if (current == null) {
            Text("The assistant opens with the project.", color = Palette.text)
        } else if (setting) {
            CloudSettings(current) { choice, key ->
                model.setAssistantCloud(choice, key)
                cloud = model.assistantCloud()
                setting = cloud?.let { !it.on || !it.hasKey } ?: true
            }
        } else {
            Conversation(model) { setting = true }
        }
    }
}

@Composable
private fun Conversation(model: EditorModel, settings: () -> Unit) {
    var ask by remember { mutableStateOf("") }
    val list = rememberLazyListState()
    val lines = model.assistantLines
    LaunchedEffect(lines.size) {
        if (lines.isNotEmpty()) list.animateScrollToItem(lines.size - 1)
    }
    LazyColumn(
        state = list,
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 80.dp, max = 260.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        items(lines) { line -> Line(line) }
    }
    Row(
        Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        OutlinedTextField(
            value = ask,
            onValueChange = { ask = it },
            placeholder = { Text("Ask for an edit") },
            modifier = Modifier.weight(1f),
            maxLines = 4,
        )
        if (model.assistantBusy) {
            Pill("Stop") { model.stopAssistant() }
        } else {
            Pill("Send", enabled = ask.isNotBlank()) {
                model.ask(ask)
                ask = ""
            }
        }
    }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        TextButton(onClick = { model.clearAssistant() }, enabled = !model.assistantBusy) {
            Text("Clear", color = Palette.text)
        }
        TextButton(onClick = settings, enabled = !model.assistantBusy) {
            Text("Model and key", color = Palette.text)
        }
    }
}

/** One line: what was asked, said, a step and how it went, or a notice. */
@Composable
private fun Line(line: AssistantLine) {
    when (line.kind) {
        LineKind.ASKED -> Text(
            line.text,
            color = Palette.text,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.White.copy(alpha = 0.08f), RoundedCornerShape(10.dp))
                .padding(10.dp),
        )
        LineKind.SAID -> Text(line.text, color = Palette.text, fontSize = 14.sp)
        LineKind.STEP -> {
            val state = when {
                line.running -> "…"
                line.failed -> "failed: ${line.result.take(160)}"
                else -> "done"
            }
            Text(
                "${line.tool}  $state",
                color = if (line.failed) Color(0xFFFF8A80) else Palette.text.copy(alpha = 0.6f),
                fontSize = 12.sp,
            )
        }
        LineKind.NOTICE -> Text(line.text, color = Color(0xFFFFD180), fontSize = 13.sp)
    }
}

/**
 * The cloud model's service, name and key. Turning it on sends what is
 * asked off the phone, so it asks first.
 */
@Composable
private fun CloudSettings(current: CloudChoice, save: (CloudChoice, String?) -> Unit) {
    var provider by remember { mutableStateOf(current.provider) }
    var name by remember { mutableStateOf(current.name) }
    var url by remember { mutableStateOf(current.url) }
    var key by remember { mutableStateOf("") }
    var asking by remember { mutableStateOf(false) }
    Text(
        "The assistant uses a cloud model with your own key. What you ask and the project's " +
            "outline are sent to that service. The key stays on this phone, apart from the settings.",
        color = Palette.text.copy(alpha = 0.75f),
        fontSize = 13.sp,
    )
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Choice("Claude", provider == "claude") {
            if (provider != "claude") {
                provider = "claude"
                name = "claude-opus-5-5"
            }
        }
        Choice("Chat completions", provider == "openai") {
            if (provider != "openai") {
                provider = "openai"
                name = ""
            }
        }
    }
    if (provider == "openai") {
        OutlinedTextField(
            value = url,
            onValueChange = { url = it },
            label = { Text("Address") },
            placeholder = { Text("https://api.openai.com/v1/chat/completions") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
    }
    OutlinedTextField(
        value = name,
        onValueChange = { name = it },
        label = { Text("Model") },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
    OutlinedTextField(
        value = key,
        onValueChange = { key = it },
        label = { Text(if (current.hasKey && provider == current.provider) "Key (kept; type to replace)" else "Key") },
        singleLine = true,
        visualTransformation = PasswordVisualTransformation(),
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
        modifier = Modifier.fillMaxWidth(),
    )
    val hasKey = key.isNotBlank() || (current.hasKey && provider == current.provider)
    Pill("Use the cloud model", enabled = hasKey && name.isNotBlank()) { asking = true }
    if (asking) {
        AlertDialog(
            onDismissRequest = { asking = false },
            title = { Text("Send to the cloud?") },
            text = {
                Text(
                    "What you ask the assistant, and the project's outline it reads, go to " +
                        (if (provider == "claude") "Anthropic" else "the chat completions service") +
                        ". Your media files stay on this phone.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    asking = false
                    save(
                        CloudChoice(on = true, provider = provider, name = name, url = url, hasKey = hasKey),
                        key.ifBlank { null },
                    )
                }) { Text("Allow") }
            },
            dismissButton = { TextButton(onClick = { asking = false }) { Text("Don't allow") } },
        )
    }
}

@Composable
private fun Choice(label: String, chosen: Boolean, pick: () -> Unit) {
    Text(
        label,
        color = if (chosen) Palette.exportInk else Palette.text,
        fontSize = 14.sp,
        modifier = Modifier
            .background(if (chosen) Palette.export else Color.White.copy(alpha = 0.08f), CircleShape)
            .clickable(onClick = pick)
            .padding(horizontal = 12.dp, vertical = 6.dp),
    )
}

@Composable
private fun Pill(label: String, enabled: Boolean = true, tap: () -> Unit) {
    Text(
        label,
        color = Palette.exportInk,
        fontWeight = FontWeight.SemiBold,
        fontSize = 15.sp,
        modifier = Modifier
            .alpha(if (enabled) 1f else 0.4f)
            .background(Palette.export, CircleShape)
            .clickable(enabled = enabled, onClick = tap)
            .padding(horizontal = 14.dp, vertical = 8.dp)
            .semantics { contentDescription = label },
    )
}
