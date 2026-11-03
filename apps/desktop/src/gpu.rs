//! Draw eval's frame layers with the wgpu compositor.
//!
//! Preview draws on the device eframe opened and shows the canvas as an
//! egui native texture, so a frame never comes back to the CPU.

use std::sync::Arc;

use cutlass_eval::FrameLayers;
use cutlass_render::{Canvas, RenderError, Renderer};
use eframe::{egui_wgpu, wgpu};
use egui::mutex::RwLock;

/// What the preview canvas last showed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Drawn {
    pub(crate) draft: String,
    pub(crate) revision: u64,
    pub(crate) time: i64,
}

/// The preview canvas on eframe's device, shown in egui as a native texture.
pub(crate) struct Preview {
    renderer: Renderer,
    egui: Arc<RwLock<egui_wgpu::Renderer>>,
    canvas: Option<(Canvas, egui::TextureId)>,
    drawn: Option<Drawn>,
}

impl Preview {
    pub(crate) fn new(state: &egui_wgpu::RenderState) -> Self {
        Self {
            renderer: Renderer::new(state.device.clone(), state.queue.clone()),
            egui: Arc::clone(&state.renderer),
            canvas: None,
            drawn: None,
        }
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

    /// Draw `frame` and remember it as `key`.
    pub(crate) fn show(
        &mut self,
        frame: &FrameLayers,
        key: Drawn,
    ) -> Result<(egui::TextureId, [u32; 2]), RenderError> {
        let size = (frame.width, frame.height);
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
        }
        self.drawn = None;
        let Some((canvas, id)) = self.canvas.as_ref() else {
            return Err(RenderError::Size {
                width: size.0,
                height: size.1,
            });
        };
        self.renderer.draw_frame(canvas, frame)?;
        self.drawn = Some(key);
        Ok((*id, [size.0, size.1]))
    }
}
