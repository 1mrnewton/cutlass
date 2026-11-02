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
            let settings = crate::load_settings(&Shell::settings_path());
            let (decoders, encoders) = crate::register_codecs_with(&settings.hardware);
            let mut shell = Shell::new(decoders, encoders, settings);
            shell.open_preview_audio(cutlass_playback::AudioHost::current());
            Ok(Box::new(DesktopWindow {
                shell,
                view: ViewState::new(),
                last_time: None,
            }))
        }),
    )
}

/// Move the playhead by the time since the previous frame.
///
/// The clock is sampled before the widgets run, so the frame that starts
/// playback does not add the idle gap from before the click. `show` asks
/// for the next frame once playback is running.
pub(crate) fn advance_if_playing(shell: &mut Shell, last_time: &mut Option<f64>, now: f64) {
    if shell.playing()
        && let Some(last) = *last_time
    {
        let elapsed = ((now - last) * 1_000_000.0).round();
        if elapsed.is_finite() && elapsed > 0.0 {
            shell.advance(elapsed as i64);
        }
    }
    *last_time = Some(now);
}

struct DesktopWindow {
    shell: Shell,
    view: ViewState,
    last_time: Option<f64>,
}

impl eframe::App for DesktopWindow {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let now = ui.input(|input| input.time);
        advance_if_playing(&mut self.shell, &mut self.last_time, now);
        show(&mut self.shell, ui, &mut self.view, Some(frame));
    }
}
