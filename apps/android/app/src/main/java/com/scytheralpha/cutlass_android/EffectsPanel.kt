package com.scytheralpha.cutlass_android

import android.content.Context
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.view.Surface
import android.view.TextureView
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.key
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView

/**
 * Video effects: a grid of the effects Rust ships, each tile looping its
 * short preview video. Tapping one adds it at the playhead, over every
 * clip below it.
 */
@Composable
fun EffectsPanel(model: EditorModel, done: () -> Unit) {
    CompositionLocalProvider(LocalContentColor provides Color.White) {
        Column(
            Modifier
                .fillMaxWidth()
                .background(Palette.tools)
                .padding(start = 16.dp, end = 16.dp, top = 10.dp, bottom = 12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text("Video effects", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = Color.White)
                Spacer(Modifier.weight(1f))
                Box(
                    Modifier
                        .size(width = 40.dp, height = 32.dp)
                        .clickable(onClick = done)
                        .semantics { contentDescription = "Done" },
                    contentAlignment = Alignment.Center,
                ) { HugeIcon("done", size = 20.dp) }
            }
            LazyVerticalGrid(
                columns = GridCells.Fixed(4),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(230.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                items(model.effects, key = { it.id }) { effect ->
                    Column(
                        Modifier
                            .clickable(role = Role.Button) {
                                model.addEffect(effect.id)
                                done()
                            }
                            .semantics { contentDescription = effect.name },
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(5.dp),
                    ) {
                        EffectsPanelVideo(
                            name = effect.preview,
                            modifier = Modifier
                                .fillMaxWidth()
                                .aspectRatio(1f)
                                .clip(RoundedCornerShape(8.dp)),
                        )
                        Text(
                            effect.name,
                            fontSize = 11.sp,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            color = Color.White.copy(alpha = 0.85f),
                            modifier = Modifier.clearAndSetSemantics {},
                        )
                    }
                }
            }
        }
    }
}

/**
 * A muted preview from the app's assets that plays over and over. Only the
 * tiles on screen hold a player: the grid releases the view of a tile that
 * scrolls away.
 */
@Composable
private fun EffectsPanelVideo(name: String, modifier: Modifier) {
    key(name) {
        AndroidView(
            factory = { context -> EffectsPanelVideoView(context, name) },
            modifier = modifier.background(Palette.white(0.18)),
            onRelease = { it.release() },
        )
    }
}

/**
 * Plays `EffectPreviews/<name>` from the APK's assets, filling its square
 * as the iOS player layer does. The preview is decoration: it never asks
 * for audio focus and plays at no volume, so it stops no other sound.
 */
private class EffectsPanelVideoView(context: Context, private val name: String) :
    TextureView(context),
    TextureView.SurfaceTextureListener {
    private var player: MediaPlayer? = null
    private var surface: Surface? = null
    private var videoWidth = 0
    private var videoHeight = 0

    init {
        // Clear until the first frame, so the grey behind shows.
        isOpaque = false
        surfaceTextureListener = this
    }

    override fun onSurfaceTextureAvailable(texture: SurfaceTexture, width: Int, height: Int) = play(texture)

    override fun onSurfaceTextureSizeChanged(texture: SurfaceTexture, width: Int, height: Int) = fill()

    override fun onSurfaceTextureDestroyed(texture: SurfaceTexture): Boolean {
        release()
        return true
    }

    override fun onSurfaceTextureUpdated(texture: SurfaceTexture) = Unit

    /** A missing or unreadable preview leaves the tile grey, as on iOS. */
    private fun play(texture: SurfaceTexture) {
        release()
        val output = Surface(texture)
        val media = MediaPlayer()
        try {
            context.assets.openFd("EffectPreviews/$name").use { file ->
                media.setDataSource(file.fileDescriptor, file.startOffset, file.length)
            }
            media.setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MOVIE)
                    .build(),
            )
            media.setVolume(0f, 0f)
            media.isLooping = true
            media.setSurface(output)
            media.setOnVideoSizeChangedListener { _, wide, tall ->
                videoWidth = wide
                videoHeight = tall
                fill()
            }
            media.setOnPreparedListener { it.start() }
            media.prepareAsync()
        } catch (error: Exception) {
            media.release()
            output.release()
            return
        }
        player = media
        surface = output
    }

    /** Scale the picture to cover the view, cropping the overflow evenly. */
    private fun fill() {
        if (videoWidth <= 0 || videoHeight <= 0 || width <= 0 || height <= 0) return
        val scale = maxOf(width.toFloat() / videoWidth, height.toFloat() / videoHeight)
        val matrix = Matrix()
        matrix.setScale(
            videoWidth * scale / width,
            videoHeight * scale / height,
            width / 2f,
            height / 2f,
        )
        setTransform(matrix)
    }

    fun release() {
        player?.release()
        player = null
        surface?.release()
        surface = null
    }
}
