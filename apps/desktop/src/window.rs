//! The eframe window. The renderer here is the widget pass.

use crate::shell::Shell;
use crate::tokens;
use crate::view::{ViewState, show};

pub fn run() -> eframe::Result {
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("Cutlass")
        .with_inner_size([1280.0, 800.0])
        .with_min_inner_size([tokens::MIN_WINDOW_WIDTH, tokens::MIN_WINDOW_HEIGHT]);
    match icon(ICON) {
        Some(icon) => viewport = viewport.with_icon(icon),
        None => cutlass_logger::error!("the window icon did not decode"),
    }
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
        Box::new(|creation| {
            let settings = crate::load_settings(&Shell::settings_path());
            let (decoders, encoders) = crate::register_codecs_with(&settings.hardware);
            let mut shell = Shell::new(decoders, encoders, settings);
            if let Some(state) = creation.wgpu_render_state.as_ref() {
                shell.attach_gpu(state);
            }
            shell.open_preview_audio(cutlass_playback::AudioHost::current());
            let ctx = creation.egui_ctx.clone();
            shell.start_thumbnails(move || ctx.request_repaint());
            let ctx = creation.egui_ctx.clone();
            shell.start_waveforms(move || ctx.request_repaint());
            Ok(Box::new(DesktopWindow {
                shell,
                view: ViewState::new(),
                last_time: None,
            }))
        }),
    )
}

/// The app icon. macOS draws it as given, so its file is the rounded tile
/// with the margin the Dock expects; elsewhere it is the full square.
#[cfg(target_os = "macos")]
const ICON: &[u8] = include_bytes!("../icons/macos.png");
#[cfg(not(target_os = "macos"))]
const ICON: &[u8] = include_bytes!("../icons/square.png");

/// Decode an 8-bit RGBA PNG into the window's icon.
fn icon(png: &[u8]) -> Option<egui::IconData> {
    use cutlass_decode::{BitDepth, CodecId, CodecParams, ColorType, Decoder, Packet};

    let mut decoder =
        cutlass_png::PngDecoder::open(&CodecParams::video(CodecId::Png, 1, 1)).ok()?;
    let picture = decoder
        .decode_picture(&Packet::video(png.to_vec(), 0, 1, 1, true))
        .ok()?;
    let image = &picture.first()?.image;
    if image.color != ColorType::Rgba || image.depth != BitDepth::Eight {
        return None;
    }
    let rgba = image
        .samples
        .iter()
        .map(|&sample| u8::try_from(sample).ok())
        .collect::<Option<Vec<u8>>>()?;
    Some(egui::IconData {
        rgba,
        width: image.width,
        height: image.height,
    })
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
        if let Some(wait) = self.shell.autosave() {
            ui.ctx().request_repaint_after(wait);
        }
    }

    fn on_exit(&mut self) {
        if self.shell.unsaved()
            && let Err(error) = self.shell.save()
        {
            cutlass_logger::error!("save on quit: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::icon;

    #[test]
    fn both_icons_decode_square_and_opaque_in_the_middle() {
        for (png, side) in [
            (&include_bytes!("../icons/macos.png")[..], 512),
            (&include_bytes!("../icons/square.png")[..], 256),
        ] {
            let icon = icon(png).expect("icon");
            assert_eq!((icon.width, icon.height), (side, side));
            assert_eq!(icon.rgba.len(), (side * side * 4) as usize);
            let middle = ((side / 2 * side + side / 2) * 4 + 3) as usize;
            assert_eq!(icon.rgba[middle], 255);
        }
    }

    #[test]
    fn a_file_that_is_not_a_png_gives_no_icon() {
        assert!(icon(b"not a png").is_none());
    }
}
