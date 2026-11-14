//! Draw eval's frame layers with the wgpu compositor.
//!
//! Preview draws on the device eframe opened and shows the canvas as an
//! egui native texture, so a frame never comes back to the CPU.

use std::sync::Arc;

use cutlass_effects::{PreviewScene, TransitionKind};
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
    /// The effect studio's preview, shown in egui as is.
    studio_live: Option<(Canvas, egui::TextureId)>,
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
            studio_live: None,
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
        let (canvas, id) = live_card(&self.renderer, &self.egui, &mut self.effect_live, size)?;
        draw_effect(&self.renderer, canvas, recipe, scene, time)?;
        Ok((id, [size.0, size.1]))
    }

    /// Run `steps` over `picture` on the studio's canvas, which egui
    /// shows. With no steps the picture shows as it is.
    pub(crate) fn studio_effect(
        &mut self,
        steps: &[cutlass_render::Step],
        picture: &PreviewScene,
    ) -> Result<(egui::TextureId, [u32; 2]), RenderError> {
        let size = (picture.width, picture.height);
        let (canvas, id) = live_card(&self.renderer, &self.egui, &mut self.studio_live, size)?;
        let mut layers = vec![Layer::Rgba(
            scene_picture(picture),
            Placement::canvas(size.0, size.1),
        )];
        if !steps.is_empty() {
            layers.push(Layer::Effect(steps, None));
        }
        self.renderer.draw(canvas, &layers)?;
        Ok((id, [size.0, size.1]))
    }

    /// Draw one moment of a transition recipe from `from` to `to`, both
    /// one size, on the studio's canvas.
    pub(crate) fn studio_crossing(
        &mut self,
        crossing: &cutlass_effects::Crossing,
        from: &PreviewScene,
        to: &PreviewScene,
    ) -> Result<(egui::TextureId, [u32; 2]), RenderError> {
        let size = (from.width, from.height);
        let (canvas, id) = live_card(&self.renderer, &self.egui, &mut self.studio_live, size)?;
        self.renderer.draw(
            canvas,
            &[Layer::Crossing {
                crossing,
                outgoing: scene_picture(from),
                incoming: scene_picture(to),
            }],
        )?;
        Ok((id, [size.0, size.1]))
    }

    /// Draw `kind` at `t` from `from` to `to` on the card canvas egui shows.
    pub(crate) fn transition_frame(
        &mut self,
        kind: TransitionKind,
        from: &PreviewScene,
        to: &PreviewScene,
        t: f32,
    ) -> Result<(egui::TextureId, [u32; 2]), RenderError> {
        let size = (from.width, from.height);
        let (canvas, id) = live_card(&self.renderer, &self.egui, &mut self.effect_live, size)?;
        draw_transition(&self.renderer, canvas, kind, from, to, t)?;
        Ok((id, [size.0, size.1]))
    }

    /// `kind` halfway from `from` to `to`, as straight RGBA.
    pub(crate) fn transition_still(
        &mut self,
        kind: TransitionKind,
        from: &PreviewScene,
        to: &PreviewScene,
    ) -> Result<Vec<u8>, RenderError> {
        let canvas = still_card(
            &self.renderer,
            &mut self.effect_still,
            (from.width, from.height),
        )?;
        draw_transition(&self.renderer, canvas, kind, from, to, 0.5)?;
        // Both scenes are opaque, so what shows is opaque too.
        self.renderer.read(canvas)
    }

    /// `recipe` at its poster moment over `scene`, as straight RGBA.
    pub(crate) fn effect_still(
        &mut self,
        recipe: &Recipe,
        scene: &PreviewScene,
    ) -> Result<Vec<u8>, RenderError> {
        let canvas = still_card(
            &self.renderer,
            &mut self.effect_still,
            (scene.width, scene.height),
        )?;
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
/// The card canvas egui shows, `size`, made or resized as needed.
fn live_card<'a>(
    renderer: &Renderer,
    egui: &RwLock<egui_wgpu::Renderer>,
    slot: &'a mut Option<(Canvas, egui::TextureId)>,
    size: (u32, u32),
) -> Result<(&'a Canvas, egui::TextureId), RenderError> {
    if slot
        .as_ref()
        .is_none_or(|(canvas, _)| canvas.size() != size)
    {
        let canvas = renderer.canvas(size.0, size.1)?;
        let mut egui = egui.write();
        let id = match slot.take() {
            Some((_, id)) => {
                egui.update_egui_texture_from_wgpu_texture(
                    renderer.device(),
                    canvas.view(),
                    wgpu::FilterMode::Linear,
                    id,
                );
                id
            }
            None => egui.register_native_texture(
                renderer.device(),
                canvas.view(),
                wgpu::FilterMode::Linear,
            ),
        };
        *slot = Some((canvas, id));
    }
    let Some((canvas, id)) = slot.as_ref() else {
        return Err(RenderError::Size {
            width: size.0,
            height: size.1,
        });
    };
    Ok((canvas, *id))
}

/// The canvas a card's still is drawn on before it is read back.
fn still_card<'a>(
    renderer: &Renderer,
    slot: &'a mut Option<Canvas>,
    size: (u32, u32),
) -> Result<&'a Canvas, RenderError> {
    if slot.as_ref().is_none_or(|canvas| canvas.size() != size) {
        *slot = Some(renderer.canvas(size.0, size.1)?);
    }
    slot.as_ref().ok_or(RenderError::Size {
        width: size.0,
        height: size.1,
    })
}

/// `kind` at `t` from one card scene to another, both the canvas's size.
fn draw_transition(
    renderer: &Renderer,
    canvas: &Canvas,
    kind: TransitionKind,
    from: &PreviewScene,
    to: &PreviewScene,
    t: f32,
) -> Result<(), RenderError> {
    renderer.draw(
        canvas,
        &[Layer::Transition {
            kind,
            t,
            outgoing: scene_picture(from),
            incoming: scene_picture(to),
        }],
    )
}

/// A card scene as the compositor takes it.
fn scene_picture(scene: &PreviewScene) -> Rgba<'_> {
    Rgba {
        width: scene.width,
        height: scene.height,
        bytes: &scene.rgba,
    }
}

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
