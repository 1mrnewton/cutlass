//! The eframe window. The renderer here is the widget pass.

use crate::shell::Shell;
use crate::tokens;
use crate::view::{ViewState, show};

pub fn run() -> eframe::Result {
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("Cutlass")
        .with_inner_size([1280.0, 800.0])
        .with_min_inner_size([tokens::MIN_WINDOW_WIDTH, tokens::MIN_WINDOW_HEIGHT]);
    #[cfg(target_os = "macos")]
    {
        viewport = viewport
            .with_fullsize_content_view(true)
            .with_titlebar_shown(false)
            .with_title_shown(false);
    }
    #[cfg(not(target_os = "macos"))]
    {
        viewport = viewport.with_decorations(false);
    }
    eframe::run_native(
        "Cutlass",
        eframe::NativeOptions {
            viewport,
            ..Default::default()
        },
        Box::new(|_creation| {
            let (decoders, encoders) = crate::register_codecs();
            let mut shell = Shell::new(decoders, encoders, cutlass_settings::Settings::default());
            shell.open_preview_audio(cutlass_playback::AudioHost::current());
            Ok(Box::new(DesktopWindow {
                shell,
                view: ViewState::new(),
                last_time: None,
            }))
        }),
    )
}

struct DesktopWindow {
    shell: Shell,
    view: ViewState,
    last_time: Option<f64>,
}

impl eframe::App for DesktopWindow {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let now = ui.input(|input| input.time);
        if self.shell.playing()
            && let Some(last) = self.last_time
        {
            let elapsed = ((now - last) * 1_000_000.0).round();
            if elapsed.is_finite() {
                self.shell.advance(elapsed as i64);
            }
            ui.ctx().request_repaint();
        }
        self.last_time = Some(now);
        show(&mut self.shell, ui, &mut self.view, Some(frame));
    }
}
