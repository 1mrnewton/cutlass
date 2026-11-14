//! The project's cover: the box beside the main lane, the dialog that
//! picks it, and the picture on a project's card at home.
//!
//! The dialog has two tabs. Select from video shows a strip of the whole
//! timeline with a handle to drag; the preview is the timeline's own
//! picture at the handle, so the playhead follows it while the dialog is
//! open and goes back when it closes. Local takes a picture from disk, cut
//! to the canvas.

use std::path::{Path, PathBuf};

use cutlass_engine::Thumb;
use cutlass_model::{COVER_FILE, Cover, Micros};
use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Ui, Vec2};

use crate::shell::Shell;
use crate::tokens;
use crate::widgets::{self, SheetEnd};

/// The sheet's width; the preview and strip fill it inside the padding.
const WIDTH: f32 = 760.0;
/// The tallest the preview grows.
const PREVIEW_HEIGHT: f32 = 360.0;
/// The strip's height.
const STRIP_HEIGHT: f32 = 56.0;
/// The most cells the strip draws.
const STRIP_CELLS: usize = 24;
/// Rows a local picture is shown at in the dialog.
const LOCAL_ROWS: u32 = 720;
/// Rows a cover is read at for the box and a project's card.
const CARD_ROWS: u32 = 240;

/// The cover dialog while it is open.
pub(super) struct CoverDialog {
    /// The Local tab is showing, not Select from video.
    local: bool,
    /// The frame picked from the video.
    time: Micros,
    /// Where the playhead was before the dialog moved it.
    return_to: Micros,
    /// The picture picked from disk, and how it shows.
    image: Option<PathBuf>,
    image_texture: Option<egui::TextureHandle>,
    /// The strip's pictures, uploaded once each lands.
    strip: Vec<Option<egui::TextureHandle>>,
    strip_version: u64,
}

impl CoverDialog {
    /// Open on the project's cover: its frame or its picture. Without one,
    /// the frame at the playhead.
    pub(super) fn open(shell: &Shell) -> Self {
        let playhead = shell.playhead();
        let mut dialog = Self {
            local: false,
            time: playhead,
            return_to: playhead,
            image: None,
            image_texture: None,
            strip: Vec::new(),
            strip_version: u64::MAX,
        };
        match shell.cover() {
            Some(Cover::Frame { time }) => dialog.time = *time,
            Some(Cover::Image { path }) => {
                dialog.local = true;
                dialog.image = Some(PathBuf::from(path));
            }
            None => {}
        }
        dialog
    }

    /// Take `path` from the picker; it is read on the dialog's next frame.
    pub(super) fn picked(&mut self, path: PathBuf) {
        self.image = Some(path);
        self.image_texture = None;
        self.local = true;
    }

    /// Show `path` in the Local tab, cut as the cover would be.
    pub(super) fn choose_image(&mut self, shell: &mut Shell, ctx: &egui::Context, path: PathBuf) {
        match shell.cover_image_preview(&path, LOCAL_ROWS) {
            Ok(picture) => {
                self.image_texture = Some(upload(ctx, "cover local", &picture));
                self.image = Some(path);
                self.local = true;
            }
            Err(error) => shell.notice_error(&error),
        }
    }
}

/// The id under which the box asks for the dialog to open.
fn open_id() -> egui::Id {
    egui::Id::unique("cover dialog asked")
}

/// Whether the Cover box was clicked since the last ask.
pub(super) fn take_open_request(ctx: &egui::Context) -> bool {
    ctx.data_mut(|data| data.remove_temp::<bool>(open_id()))
        .unwrap_or(false)
}

/// Draw the dialog; it closes itself into `slot` when it ends.
pub(super) fn dialog(
    shell: &mut Shell,
    ui: &mut Ui,
    slot: &mut Option<CoverDialog>,
    pick_image: &mut bool,
) {
    let Some(dialog) = slot.as_mut() else {
        return;
    };
    if shell.draft().is_none() {
        *slot = None;
        return;
    }
    if dialog.local && dialog.image.is_some() && dialog.image_texture.is_none() {
        let path = dialog.image.clone().unwrap_or_default();
        dialog.choose_image(shell, ui.ctx(), path);
    }
    let ((), end) = widgets::sheet(ui, "Select a cover", WIDTH, "Save", |ui| {
        body(shell, ui, dialog, pick_image);
    });
    match end {
        SheetEnd::Open => {}
        SheetEnd::Cancel => close(shell, slot),
        SheetEnd::Confirm => {
            let result = if dialog.local {
                match &dialog.image {
                    Some(path) => shell.set_cover_image(path),
                    None => {
                        shell.set_notice("Choose a picture first".to_owned());
                        return;
                    }
                }
            } else {
                shell.set_cover_frame(dialog.time)
            };
            match result {
                Ok(()) => close(shell, slot),
                Err(error) => shell.notice_error(&error),
            }
        }
    }
}

fn close(shell: &mut Shell, slot: &mut Option<CoverDialog>) {
    if let Some(dialog) = slot.take() {
        shell.set_scrubbing(false);
        shell.seek(dialog.return_to);
    }
    shell.forget_cover_strip();
}

fn body(shell: &mut Shell, ui: &mut Ui, dialog: &mut CoverDialog, pick_image: &mut bool) {
    let width = ui.available_width();
    let (canvas_w, canvas_h) = shell.draft().map_or((16.0, 9.0), |draft| {
        (
            draft.canvas_config.width.max(1) as f32,
            draft.canvas_config.height.max(1) as f32,
        )
    });
    let (stage, stage_response) = ui.allocate_exact_size(
        Vec2::new(width, PREVIEW_HEIGHT.min(width * 9.0 / 16.0)),
        Sense::click(),
    );
    ui.painter()
        .rect_filled(stage, CornerRadius::same(6), Color32::BLACK);
    let fit = fit_rect(stage, canvas_w / canvas_h);
    if dialog.local {
        match &dialog.image_texture {
            Some(texture) => {
                ui.painter().image(
                    texture.id(),
                    fit,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }
            None => {
                ui.painter().text(
                    stage.center(),
                    Align2::CENTER_CENTER,
                    "Choose a picture",
                    FontId::proportional(14.0),
                    if stage_response.hovered() {
                        tokens::TEXT
                    } else {
                        tokens::MUTED
                    },
                );
            }
        }
        if stage_response.clicked() {
            *pick_image = true;
        }
    } else {
        if shell.playhead() != dialog.time {
            shell.seek(dialog.time);
        }
        match shell.preview_texture() {
            Ok(Some((texture, _))) => {
                ui.painter().image(
                    texture,
                    fit,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }
            Ok(None) => {}
            Err(error) => shell.note_preview(Some(&error)),
        }
        if shell.preview_pending() {
            ui.ctx().request_repaint();
        }
    }
    ui.add_space(16.0);
    ui.vertical_centered(|ui| {
        crate::ui::Segmented::show(ui, |ui| {
            for (local, name) in [(false, "Select from video"), (true, "Local")] {
                if ui
                    .add(crate::ui::Choice::new(dialog.local == local, name))
                    .clicked()
                {
                    dialog.local = local;
                    if local && dialog.image.is_none() {
                        *pick_image = true;
                    }
                }
            }
        });
    });
    ui.add_space(12.0);
    if dialog.local {
        local_row(ui, dialog, pick_image);
    } else {
        strip(shell, ui, dialog, canvas_w / canvas_h);
    }
}

/// The picked picture's name and a button to choose another.
fn local_row(ui: &mut Ui, dialog: &CoverDialog, pick_image: &mut bool) {
    ui.horizontal(|ui| {
        ui.set_height(STRIP_HEIGHT);
        let name = dialog
            .image
            .as_deref()
            .and_then(Path::file_name)
            .map_or_else(
                || "No picture chosen".to_owned(),
                |name| name.to_string_lossy().into_owned(),
            );
        ui.label(egui::RichText::new(name).size(13.0).color(tokens::MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.add(crate::ui::Button::new("Choose picture")).clicked() {
                *pick_image = true;
            }
        });
    });
}

/// The whole timeline as a strip, with a handle on the picked frame.
/// Pressing or dragging on it moves the frame.
fn strip(shell: &mut Shell, ui: &mut Ui, dialog: &mut CoverDialog, aspect: f32) {
    let width = ui.available_width();
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(width, STRIP_HEIGHT), Sense::click_and_drag());
    let cell_width = (STRIP_HEIGHT * aspect).max(8.0);
    let cells = ((width / cell_width).ceil() as usize).clamp(1, STRIP_CELLS);
    let rows = (STRIP_HEIGHT * ui.ctx().pixels_per_point()).round() as u32;
    let (pictures, version) = shell.cover_strip(cells, rows);
    if version != dialog.strip_version || dialog.strip.len() != pictures.len() {
        dialog.strip = pictures
            .iter()
            .map(|picture| {
                picture
                    .as_ref()
                    .map(|picture| upload(ui.ctx(), "cover strip", picture))
            })
            .collect();
        dialog.strip_version = version;
    }
    if dialog.strip.iter().any(Option::is_none) {
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(50));
    }
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, CornerRadius::same(6), tokens::LANE);
    let step = rect.width() / cells as f32;
    for (index, cell) in dialog.strip.iter().enumerate() {
        let Some(texture) = cell else {
            continue;
        };
        let cell = Rect::from_min_size(
            Pos2::new(rect.left() + step * index as f32, rect.top()),
            Vec2::new(step, rect.height()),
        );
        let [w, h] = texture.size();
        painter.image(
            texture.id(),
            cell,
            super::cover_uv(cell.size(), w as f32, h as f32),
            Color32::WHITE,
        );
    }
    let end = shell.draft().map_or(0, |draft| draft.duration.0).max(1);
    if let Some(pointer) = response.interact_pointer_pos()
        && (response.clicked() || response.dragged() || response.drag_started())
    {
        let share = ((pointer.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
        dialog.time = Micros(((f64::from(share) * end as f64) as i64).clamp(0, end - 1));
    }
    shell.set_scrubbing(response.dragged());
    let x = rect.left() + rect.width() * (dialog.time.0 as f32 / end as f32).clamp(0.0, 1.0);
    let line = egui::Stroke::new(2.0, tokens::PLAYHEAD);
    ui.painter().line_segment(
        [Pos2::new(x, rect.top() - 6.0), Pos2::new(x, rect.bottom())],
        line,
    );
    let grip = Rect::from_center_size(Pos2::new(x, rect.top() - 10.0), Vec2::new(12.0, 14.0));
    ui.painter().rect(
        grip,
        CornerRadius {
            nw: 2,
            ne: 2,
            sw: 6,
            se: 6,
        },
        tokens::LANE,
        line,
        egui::StrokeKind::Inside,
    );
    response.on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
}

/// The largest rect of `aspect` (width over height) centred in `outer`.
fn fit_rect(outer: Rect, aspect: f32) -> Rect {
    let size = if outer.width() / outer.height() > aspect {
        Vec2::new(outer.height() * aspect, outer.height())
    } else {
        Vec2::new(outer.width(), outer.width() / aspect)
    };
    Rect::from_center_size(outer.center(), size)
}

fn upload(ctx: &egui::Context, name: &str, picture: &Thumb) -> egui::TextureHandle {
    let image = egui::ColorImage::from_rgba_premultiplied(
        [picture.width as usize, picture.height as usize],
        &picture.rgba,
    );
    ctx.load_texture(name, image, egui::TextureOptions::LINEAR)
}

/// The picture of the cover file at `file`, read once for each `stamp`
/// and kept in the window's memory.
fn cover_texture(
    shell: &Shell,
    ctx: &egui::Context,
    file: &Path,
    stamp: u64,
) -> Option<egui::TextureHandle> {
    let id = egui::Id::unique(("cover texture", file));
    if let Some((kept, texture)) =
        ctx.data(|data| data.get_temp::<(u64, Option<egui::TextureHandle>)>(id))
        && kept == stamp
    {
        return texture;
    }
    let texture = shell
        .picture_thumb(file, CARD_ROWS)
        .map(|picture| upload(ctx, "cover", &picture));
    ctx.data_mut(|data| data.insert_temp(id, (stamp, texture.clone())));
    texture
}

/// The Cover box in the bare column beside the main lane. Without a
/// cover it is a pencil and the word Cover in a dashed square; with one,
/// the cover fills it under the word. A click opens the dialog.
pub(super) fn paint_box(shell: &Shell, ui: &Ui, column: Rect) {
    const SIDE: f32 = 52.0;
    let side = SIDE.min(column.height() - 12.0).max(28.0);
    let card = Rect::from_center_size(column.center(), Vec2::splat(side));
    let response = ui.interact(card, ui.scope_id().with("cover"), Sense::click());
    let painter = ui.painter();
    let radius = CornerRadius::same(6);
    let picture = shell
        .cover_file()
        .and_then(|file| cover_texture(shell, ui.ctx(), &file, shell.cover_version()));
    match &picture {
        Some(texture) => {
            let [w, h] = texture.size();
            painter.add(
                egui::epaint::RectShape::filled(card, radius, Color32::WHITE).with_texture(
                    texture.id(),
                    super::cover_uv(card.size(), w as f32, h as f32),
                ),
            );
            // A shade under the word, so it reads on any picture.
            let foot = Rect::from_min_max(
                Pos2::new(card.left(), card.bottom() - 16.0),
                card.right_bottom(),
            );
            painter.rect_filled(
                foot,
                CornerRadius {
                    nw: 0,
                    ne: 0,
                    sw: 6,
                    se: 6,
                },
                Color32::from_black_alpha(150),
            );
            painter.text(
                foot.center(),
                Align2::CENTER_CENTER,
                "Cover",
                FontId::proportional(10.0),
                Color32::WHITE,
            );
            if response.hovered() {
                painter.rect_stroke(
                    card,
                    radius,
                    egui::Stroke::new(1.0, tokens::TEXT),
                    egui::StrokeKind::Inside,
                );
            }
        }
        None => {
            let fill = if response.hovered() {
                tokens::DIALOG_HEAD
            } else {
                tokens::LANE_LINE
            };
            painter.rect_filled(card, radius, fill);
            let dash = egui::Stroke::new(1.0, tokens::FAINT);
            let inner = card.shrink(6.0);
            for (from, to) in [
                (
                    Pos2::new(inner.left(), card.top()),
                    Pos2::new(inner.right(), card.top()),
                ),
                (
                    Pos2::new(inner.left(), card.bottom()),
                    Pos2::new(inner.right(), card.bottom()),
                ),
                (
                    Pos2::new(card.left(), inner.top()),
                    Pos2::new(card.left(), inner.bottom()),
                ),
                (
                    Pos2::new(card.right(), inner.top()),
                    Pos2::new(card.right(), inner.bottom()),
                ),
            ] {
                painter.extend(egui::Shape::dashed_line(&[from, to], dash, 3.0, 2.0));
            }
            let icon = Rect::from_center_size(
                Pos2::new(card.center().x, card.top() + side * 0.36),
                Vec2::splat(14.0),
            );
            let _ = crate::icons::paint(painter, icon, "editText", tokens::TEXT);
            painter.text(
                Pos2::new(card.center().x, card.top() + side * 0.72),
                Align2::CENTER_CENTER,
                "Cover",
                FontId::proportional(11.0),
                tokens::TEXT,
            );
        }
    }
    if response.clicked() {
        ui.ctx().data_mut(|data| data.insert_temp(open_id(), true));
    }
    response.on_hover_text(if picture.is_some() {
        "Change the project's cover"
    } else {
        "Pick the project's cover"
    });
}

/// A project card's picture: its cover, read once for each time the file
/// changes. At most one cover is read each frame, so a long list of
/// projects fills in over a few frames rather than holding one up.
pub(super) fn card_picture(
    shell: &Shell,
    ctx: &egui::Context,
    project: &Path,
) -> Option<egui::TextureHandle> {
    let file = project.join(COVER_FILE);
    let modified = std::fs::metadata(&file)
        .and_then(|metadata| metadata.modified())
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos() as u64);
    let id = egui::Id::unique(("cover texture", &file));
    if let Some((kept, texture)) =
        ctx.data(|data| data.get_temp::<(u64, Option<egui::TextureHandle>)>(id))
        && kept == modified
    {
        return texture;
    }
    let pass = ctx.cumulative_pass_nr();
    let read_this_pass = egui::Id::unique("cover read this pass");
    if ctx.data(|data| data.get_temp::<u64>(read_this_pass)) == Some(pass) {
        ctx.request_repaint();
        return None;
    }
    ctx.data_mut(|data| data.insert_temp(read_this_pass, pass));
    cover_texture(shell, ctx, &file, modified)
}

#[cfg(test)]
mod tests {
    use super::fit_rect;
    use egui::{Pos2, Rect};

    #[test]
    fn a_fit_keeps_the_shape_inside_and_centred() {
        let outer = Rect::from_min_max(Pos2::ZERO, Pos2::new(200.0, 100.0));
        let square = fit_rect(outer, 1.0);
        assert_eq!(
            square,
            Rect::from_min_max(Pos2::new(50.0, 0.0), Pos2::new(150.0, 100.0))
        );
        let wide = fit_rect(outer, 4.0);
        assert_eq!(
            wide,
            Rect::from_min_max(Pos2::new(0.0, 25.0), Pos2::new(200.0, 75.0))
        );
    }
}
