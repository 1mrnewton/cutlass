//! The eframe window. The renderer here is the widget pass.

use crate::shell::Shell;
use crate::tokens;
use crate::view::{ViewState, show};

/// Open the window on the projects, or on what the command line names.
pub fn run() -> eframe::Result {
    run_on(false)
}

/// Open the window on the effect studio, as `cutlass studio` does. A word
/// after it is the id of an effect or a transition to open there.
pub fn run_studio() -> eframe::Result {
    run_on(true)
}

fn run_on(studio: bool) -> eframe::Result {
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("Cutlass")
        .with_inner_size([1280.0, 800.0])
        .with_min_inner_size([tokens::MIN_WINDOW_WIDTH, tokens::MIN_WINDOW_HEIGHT])
        // eframe takes one transparency setting from the main window for
        // every window it opens. Menus need it for their rounded corners and
        // shadow; the main window still looks solid, since the view paints
        // its whole area each frame.
        .with_transparent(true);
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
    // Finder sends a bundled app its documents as an Apple Event; listen
    // before the event loop so the ones that launch the app arrive too.
    #[cfg(target_os = "macos")]
    cutlass_platform_apple::listen_for_opened_files();
    eframe::run_native(
        "Cutlass",
        eframe::NativeOptions {
            viewport,
            ..Default::default()
        },
        Box::new(move |creation| {
            let file = Shell::settings_path();
            let loaded = crate::load_settings_file(&file);
            let (decoders, encoders) = crate::register_codecs_with(&loaded.settings.hardware);
            let mut shell = Shell::new(decoders, encoders, loaded.settings);
            shell.keep_settings_file(file, loaded.generation);
            shell.keep_presets_file(Shell::home().text_presets_file());
            shell.keep_studio_folder(Shell::home().studio());
            // At most once a day, and never with the switch off.
            shell.check_updates_at_launch();
            // A project folder or files named on the command line, as a
            // file association passes them.
            // After `cutlass studio` there are none: what follows is the
            // command and the recipe to open, not files.
            let named: Vec<std::path::PathBuf> = std::env::args_os()
                .skip(1)
                .filter(|_| !studio)
                .map(std::path::PathBuf::from)
                .collect();
            if studio
                && let Some(id) = std::env::args().nth(2)
                && !shell.studio_mut().open_effect(&id)
                && !shell.studio_mut().open_transition(&id)
            {
                cutlass_logger::warn!("cutlass studio: no effect or transition {id}");
            }
            if let Err(error) = shell.drop_files(&named) {
                shell.notice_error(&error);
            }
            if let Some(state) = creation.wgpu_render_state.as_ref() {
                shell.attach_gpu(state);
            }
            shell.open_preview_audio(cutlass_playback::AudioHost::current());
            let ctx = creation.egui_ctx.clone();
            shell.start_thumbnails(move || ctx.request_repaint());
            let ctx = creation.egui_ctx.clone();
            shell.start_waveforms(move || ctx.request_repaint());
            // Other hosts, such as `cutlass mcp`, reach the open project here.
            let ctx = creation.egui_ctx.clone();
            shell.listen(std::sync::Arc::new(move || ctx.request_repaint()));
            #[cfg(target_os = "macos")]
            {
                let ctx = creation.egui_ctx.clone();
                cutlass_platform_apple::wake_on_opened_files(move || ctx.request_repaint());
            }
            let mut view = ViewState::new();
            view.studio.open = studio;
            Ok(Box::new(DesktopWindow {
                shell,
                view,
                last_time: None,
                #[cfg(target_os = "macos")]
                menu_bar: None,
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

/// Move the playhead and the level meter by the time since the previous
/// frame.
///
/// The clock is sampled before the widgets run, so the frame that starts
/// playback does not add the idle gap from before the click. `show` asks
/// for the next frame once playback is running.
pub(crate) fn advance_if_playing(shell: &mut Shell, last_time: &mut Option<f64>, now: f64) {
    if let Some(last) = *last_time {
        let elapsed = ((now - last) * 1_000_000.0).round();
        if elapsed.is_finite() && elapsed > 0.0 {
            if shell.playing() {
                shell.advance(elapsed as i64);
            }
            // The meter moves after the playhead, so it reads the frame heard now.
            shell.update_meter(elapsed as i64);
        }
    }
    *last_time = Some(now);
}

struct DesktopWindow {
    shell: Shell,
    view: ViewState,
    last_time: Option<f64>,
    /// The system menu bar, kept for as long as the window is up.
    #[cfg(target_os = "macos")]
    menu_bar: Option<crate::menubar::Native>,
}

impl eframe::App for DesktopWindow {
    /// Clear to nothing: the view paints the whole main window, and a menu's
    /// window shows only its panel and shadow.
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0; 4]
    }

    /// Runs before every pass, the window shown or not, so the close
    /// button is caught even while the window is minimised.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        crate::view::close_to_projects(&mut self.shell, &mut self.view, ctx);
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let now = ui.input(|input| input.time);
        advance_if_playing(&mut self.shell, &mut self.last_time, now);
        // Documents Finder opened with the app: a project folder opens and
        // files import, as when they are dropped on the window.
        #[cfg(target_os = "macos")]
        {
            let opened = cutlass_platform_apple::take_opened_files();
            if !opened.is_empty() {
                cutlass_logger::info!("finder opened {} item(s)", opened.len());
                if let Err(error) = self.shell.drop_files(&opened) {
                    self.shell.notice_error(&error);
                }
            }
        }
        // Put up on the first frame, once the system's own launch menu is.
        #[cfg(target_os = "macos")]
        if self.menu_bar.is_none() {
            let ctx = ui.ctx().clone();
            self.menu_bar = crate::menubar::Native::install(move || ctx.request_repaint());
            if self.menu_bar.is_none() {
                cutlass_logger::warn!("the system menu bar could not be built");
            }
        }
        #[cfg(target_os = "macos")]
        {
            self.view.system_menus = self.menu_bar.is_some();
        }
        #[cfg(target_os = "macos")]
        if let Some(bar) = self.menu_bar.as_mut() {
            let settings = self.shell.settings();
            bar.show_keys(|action| settings.keys_for(action));
            for id in bar.take() {
                crate::view::run_menu(&mut self.shell, &mut self.view, Some(frame), &id);
            }
        }
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
