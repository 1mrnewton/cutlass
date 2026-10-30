//! The eframe window. The renderer here is the widget pass.

use crate::shell::Shell;
use crate::tokens;
use crate::view::{ViewState, show};

pub fn run() -> eframe::Result {
    let viewport = egui::ViewportBuilder::default()
        .with_inner_size([1280.0, 800.0])
        .with_min_inner_size([tokens::MIN_WINDOW_WIDTH, tokens::MIN_WINDOW_HEIGHT])
        .with_decorations(false);
    eframe::run_native(
        "Cutlass",
        eframe::NativeOptions {
            viewport,
            ..Default::default()
        },
        Box::new(|_creation| {
            let (decoders, encoders) = crate::register_codecs();
            Ok(Box::new(DesktopWindow {
                shell: Shell::new(decoders, encoders, cutlass_settings::Settings::default()),
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
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
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
        show(&mut self.shell, ui, &mut self.view);
    }
}
