//! Draw eval's frame layers with the wgpu compositor.
//!
//! Preview draws on the device eframe opened and shows the canvas as an
//! egui native texture, so a frame never comes back to the CPU.

use std::sync::Arc;

use cutlass_effects::PreviewScene;
use cutlass_eval::FrameLayers;
use cutlass_model::Micros;
use cutlass_render::{Canvas, Layer, Placement, Recipe, RenderError, Renderer, Rgba};
use eframe::{egui_wgpu, wgpu};
use egui::mutex::RwLock;

/// A renderer that samples this platform's decoded surfaces in place.
pub(crate) fn platform_renderer(device: wgpu::Device, queue: wgpu::Queue) -> Renderer {
    #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
    let mut renderer = Renderer::new(device, queue);
    #[cfg(target_os = "macos")]
    renderer.set_surface_import(Arc::new(cutlass_platform_apple::import_surface));
    renderer
}

/// What the preview canvas last showed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Drawn {
    pub(crate) draft: String,
    pub(crate) revision: u64,
    pub(crate) time: i64,
    /// The preview quality it was drawn at.
    pub(crate) quality: cutlass_settings::PreviewQuality,
}

/// The preview canvas on eframe's device, shown in egui as a native texture.
pub(crate) struct Preview {
    renderer: Renderer,
    egui: Arc<RwLock<egui_wgpu::Renderer>>,
    canvas: Option<(Canvas, egui::TextureId)>,
    drawn: Option<Drawn>,
    /// The layers on the canvas. Their pictures are shared with the clip
    /// caches, so a frame that brings the same ones is not drawn again.
    frame: Option<FrameLayers>,
    /// The effect card playing under the pointer, shown in egui as is.
    effect_live: Option<(Canvas, egui::TextureId)>,
    /// Where a card's still is drawn before it is read back.
    effect_still: Option<Canvas>,
}

impl Preview {
    pub(crate) fn new(state: &egui_wgpu::RenderState) -> Self {
        Self {
            renderer: platform_renderer(state.device.clone(), state.queue.clone()),
            egui: Arc::clone(&state.renderer),
            canvas: None,
            drawn: None,
            frame: None,
            effect_live: None,
            effect_still: None,
        }
    }

    /// Draw `recipe` at `time` over `scene` on the card canvas egui shows.
    pub(crate) fn effect_frame(
        &mut self,
        recipe: &Recipe,
        scene: &PreviewScene,
        time: Micros,
    ) -> Result<(egui::TextureId, [u32; 2]), RenderError> {
        let size = (scene.width, scene.height);
        if self
            .effect_live
            .as_ref()
            .is_none_or(|(canvas, _)| canvas.size() != size)
        {
            let canvas = self.renderer.canvas(size.0, size.1)?;
            let mut egui = self.egui.write();
            let id = match self.effect_live.take() {
                Some((_, id)) => {
                    egui.update_egui_texture_from_wgpu_texture(
                        self.renderer.device(),
                        canvas.view(),
                        wgpu::FilterMode::Linear,
                        id,
                    );
                    id
                }
                None => egui.register_native_texture(
                    self.renderer.device(),
                    canvas.view(),
                    wgpu::FilterMode::Linear,
                ),
            };
            self.effect_live = Some((canvas, id));
        }
        let Some((canvas, id)) = self.effect_live.as_ref() else {
            return Err(RenderError::Size {
                width: size.0,
                height: size.1,
            });
        };
        draw_effect(&self.renderer, canvas, recipe, scene, time)?;
        Ok((*id, [size.0, size.1]))
    }

    /// `recipe` at its poster moment over `scene`, as straight RGBA.
    pub(crate) fn effect_still(
        &mut self,
        recipe: &Recipe,
        scene: &PreviewScene,
    ) -> Result<Vec<u8>, RenderError> {
        let size = (scene.width, scene.height);
        if self
            .effect_still
            .as_ref()
            .is_none_or(|canvas| canvas.size() != size)
        {
            self.effect_still = Some(self.renderer.canvas(size.0, size.1)?);
        }
        let Some(canvas) = self.effect_still.as_ref() else {
            return Err(RenderError::Size {
                width: size.0,
                height: size.1,
            });
        };
        draw_effect(&self.renderer, canvas, recipe, scene, recipe.poster_time())?;
        // The scene is opaque, so premultiplied is the same as straight.
        self.renderer.read(canvas)
    }

    /// The compositor on eframe's device, which export draws through too.
    pub(crate) fn renderer(&self) -> &Renderer {
        &self.renderer
    }

    /// The texture and size when `key` is what the canvas already shows.
    pub(crate) fn shown(&self, key: &Drawn) -> Option<(egui::TextureId, [u32; 2])> {
        if self.drawn.as_ref() != Some(key) {
            return None;
        }
        let (canvas, id) = self.canvas.as_ref()?;
        let (width, height) = canvas.size();
        Some((*id, [width, height]))
    }

    /// The texture and size of what the canvas shows now, whatever its key.
    pub(crate) fn current(&self) -> Option<(egui::TextureId, [u32; 2])> {
        let (canvas, id) = self.canvas.as_ref()?;
        let (width, height) = canvas.size();
        Some((*id, [width, height]))
    }

    /// Draw the next frame even when its key matches, because what is on
    /// the canvas is not the picture that key asks for yet.
    pub(crate) fn forget(&mut self) {
        self.drawn = None;
    }

    /// The canvas as premultiplied RGBA, for tests.
    #[cfg(test)]
    pub(crate) fn read(&self) -> Option<Vec<u8>> {
        let (canvas, _) = self.canvas.as_ref()?;
        self.renderer.read(canvas).ok()
    }

    /// Draw `frame` and remember it as `key`. A frame equal to the one on
    /// the canvas is not drawn again: a repaint between two source pictures
    /// only takes the new key. Below full quality the canvas is that part
    /// of the frame's size each way, and the frame is shrunk onto it.
    pub(crate) fn show(
        &mut self,
        frame: FrameLayers,
        key: Drawn,
    ) -> Result<(egui::TextureId, [u32; 2]), RenderError> {
        let size = cutlass_render::preview_size((frame.width, frame.height), key.quality.scale());
        let resized = self
            .canvas
            .as_ref()
            .is_none_or(|(canvas, _)| canvas.size() != size);
        if resized {
            let canvas = self.renderer.canvas(size.0, size.1)?;
            let mut egui = self.egui.write();
            let id = match self.canvas.take() {
                Some((_, id)) => {
                    egui.update_egui_texture_from_wgpu_texture(
                        self.renderer.device(),
                        canvas.view(),
                        wgpu::FilterMode::Linear,
                        id,
                    );
                    id
                }
                None => egui.register_native_texture(
                    self.renderer.device(),
                    canvas.view(),
                    wgpu::FilterMode::Linear,
                ),
            };
            self.canvas = Some((canvas, id));
            self.frame = None;
        }
        self.drawn = None;
        let Some((canvas, id)) = self.canvas.as_ref() else {
            return Err(RenderError::Size {
                width: size.0,
                height: size.1,
            });
        };
        if self.frame.as_ref() != Some(&frame) {
            self.frame = None;
            self.renderer.draw_frame_scaled(canvas, &frame)?;
            self.frame = Some(frame);
        }
        self.drawn = Some(key);
        Ok((*id, [size.0, size.1]))
    }
}

/// `recipe` at `time` over the whole of `scene`.
fn draw_effect(
    renderer: &Renderer,
    canvas: &Canvas,
    recipe: &Recipe,
    scene: &PreviewScene,
    time: Micros,
) -> Result<(), RenderError> {
    let steps = recipe.steps_at(time);
    let picture = Rgba {
        width: scene.width,
        height: scene.height,
        bytes: &scene.rgba,
    };
    renderer.draw(
        canvas,
        &[
            Layer::Rgba(picture, Placement::canvas(scene.width, scene.height)),
            Layer::Effect(&steps, None),
        ],
    )
}
