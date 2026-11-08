//! An effect card's picture: the effect over its scene, a still on the
//! card and playing while the pointer rests on it.

use cutlass_effects::{PreviewScene, bundled_recipe, preview_scene};
use cutlass_model::Micros;

use super::Shell;

/// The recipe and scene an effect card shows.
fn card_effect(card_id: &str) -> Option<(&'static cutlass_effects::Recipe, &'static PreviewScene)> {
    let effect_id = cutlass_editor::effect_of_card(card_id)?;
    let recipe = bundled_recipe(&effect_id)?;
    Some((recipe, preview_scene(recipe.preview_scene())?))
}

impl Shell {
    /// The still on an effect card, as straight RGBA, and its size. Without
    /// a GPU it is the scene alone.
    #[must_use]
    pub fn effect_still(&mut self, card_id: &str) -> Option<(Vec<u8>, [u32; 2])> {
        let (recipe, scene) = card_effect(card_id)?;
        let size = [scene.width, scene.height];
        let drawn = self
            .gpu
            .as_mut()
            .and_then(|gpu| gpu.effect_still(recipe, scene).ok());
        Some((drawn.unwrap_or_else(|| scene.rgba.clone()), size))
    }

    /// The effect card playing for `elapsed`, drawn on the GPU and shown in
    /// egui as is; none without a GPU.
    pub fn effect_frame(
        &mut self,
        card_id: &str,
        elapsed: Micros,
    ) -> Option<(egui::TextureId, [u32; 2])> {
        let (recipe, scene) = card_effect(card_id)?;
        let time = cutlass_effects::preview_time(elapsed);
        self.gpu.as_mut()?.effect_frame(recipe, scene, time).ok()
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::Micros;

    use crate::shell::tests::shell;

    #[test]
    fn an_effect_card_shows_its_effect_over_its_scene() {
        let mut desktop = shell();
        let card = format!("{}recipe/black-white", cutlass_editor::EFFECT_CARD_PREFIX);
        // Without a GPU the card shows the scene alone, and nothing plays.
        let (plain, size) = desktop.effect_still(&card).unwrap();
        assert_eq!(size, [288, 192]);
        assert!(desktop.effect_frame(&card, Micros(0)).is_none());
        assert!(desktop.effect_still("effect:recipe/none").is_none());

        let instance = eframe::wgpu::Instance::new(
            eframe::wgpu::InstanceDescriptor::new_without_display_handle(),
        );
        let Ok(state) = pollster::block_on(eframe::egui_wgpu::RenderState::create(
            &eframe::egui_wgpu::WgpuConfiguration::default(),
            &instance,
            None,
            eframe::egui_wgpu::RendererOptions::default(),
        )) else {
            eprintln!("skipped: no GPU");
            return;
        };
        desktop.attach_gpu(&state);
        let (grey, _) = desktop.effect_still(&card).unwrap();
        assert_ne!(grey, plain);
        assert!(
            grey.chunks(4)
                .all(|pixel| pixel[0].abs_diff(pixel[1]) <= 2 && pixel[1].abs_diff(pixel[2]) <= 2),
            "black and white has no colour left"
        );
        let (_, frame) = desktop.effect_frame(&card, Micros(400_000)).unwrap();
        assert_eq!(frame, [288, 192]);
    }
}
