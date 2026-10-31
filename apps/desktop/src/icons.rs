//! Free Hugeicons, stroked in the widget pass.
//!
//! Path data lives in [`icon_data`]. A name that is not in that map is an
//! error. The button shows the error instead of an empty mark. The blade
//! path is the only mark that is not from the free set.

mod icon_data;

use std::fmt;

pub use icon_data::ICONS;

/// One mark in the 24 viewBox.
#[derive(Clone, Copy, Debug)]
pub enum Mark {
    Circle { cx: f32, cy: f32, r: f32 },
    Path(&'static str),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IconError {
    pub name: String,
}

impl fmt::Display for IconError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "icon {} is not in the map", self.name)
    }
}

impl std::error::Error for IconError {}

struct Outline {
    points: Vec<(f32, f32)>,
    closed: bool,
}

/// The icon, or an error when the name is absent.
pub fn require(name: &str) -> Result<&'static icon_data::IconData, IconError> {
    ICONS
        .iter()
        .find(|icon| icon.name == name)
        .ok_or_else(|| IconError {
            name: name.to_owned(),
        })
}

/// Flatten every path in the named icon. A path the map cannot stroke is an error.
pub fn outlines(name: &str) -> Result<Vec<Vec<(f32, f32)>>, IconError> {
    let icon = require(name)?;
    let mut outlines = Vec::new();
    for mark in icon.marks {
        if let Mark::Path(data) = mark {
            let parsed = parse_path(data).map_err(|_| IconError {
                name: name.to_owned(),
            })?;
            for outline in parsed {
                outlines.push(outline.points);
            }
        }
    }
    Ok(outlines)
}

pub fn is_filled(name: &str) -> Result<bool, IconError> {
    Ok(require(name)?.fill)
}

/// Stroke the icon into `rect`. The stroke is 1.5 in a 24 viewBox, scaled to the rect.
pub fn paint(
    painter: &egui::Painter,
    rect: egui::Rect,
    name: &str,
    color: egui::Color32,
) -> Result<(), IconError> {
    let icon = require(name)?;
    let scale = rect.width().min(rect.height()) / 24.0;
    let width = crate::tokens::STROKE * scale;
    let stroke = egui::Stroke::new(width, color);
    for mark in icon.marks {
        match mark {
            Mark::Circle { cx, cy, r } => {
                let center = rect.min + egui::vec2(cx * scale, cy * scale);
                painter.circle_stroke(center, r * scale, stroke);
            }
            Mark::Path(data) => {
                let paths = parse_path(data).map_err(|_| IconError {
                    name: name.to_owned(),
                })?;
                for path in paths {
                    if path.points.len() < 2 {
                        continue;
                    }
                    let points: Vec<egui::Pos2> = path
                        .points
                        .iter()
                        .map(|(x, y)| rect.min + egui::vec2(x * scale, y * scale))
                        .collect();
                    if icon.fill && path.closed {
                        painter.add(egui::Shape::Path(egui::epaint::PathShape::convex_polygon(
                            points,
                            color,
                            egui::Stroke::NONE,
                        )));
                    } else if path.closed {
                        painter.add(egui::Shape::Path(egui::epaint::PathShape::closed_line(
                            points, stroke,
                        )));
                    } else {
                        painter.add(egui::Shape::Path(egui::epaint::PathShape::line(
                            points, stroke,
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}

fn parse_path(data: &str) -> Result<Vec<Outline>, ()> {
    let bytes = data.as_bytes();
    let mut index = 0;
    let mut outlines = Vec::new();
    let mut current: Option<Outline> = None;
    let mut cursor = (0.0, 0.0);
    while index < bytes.len() {
        skip(bytes, &mut index);
        if index >= bytes.len() {
            break;
        }
        let command = bytes[index] as char;
        if !command.is_ascii_alphabetic() {
            return Err(());
        }
        if command.is_ascii_lowercase() {
            return Err(());
        }
        index += 1;
        match command {
            'M' => {
                push_outline(&mut outlines, current.take());
                let (x, y) = pair(bytes, &mut index)?;
                cursor = (x, y);
                current = Some(Outline {
                    points: vec![cursor],
                    closed: false,
                });
                while let Some((x, y)) = optional_pair(bytes, &mut index) {
                    cursor = (x, y);
                    push_point(&mut current, cursor);
                }
            }
            'L' => {
                let (x, y) = pair(bytes, &mut index)?;
                cursor = (x, y);
                push_point(&mut current, cursor);
                while let Some((x, y)) = optional_pair(bytes, &mut index) {
                    cursor = (x, y);
                    push_point(&mut current, cursor);
                }
            }
            'H' => {
                let x = number(bytes, &mut index)?;
                cursor.0 = x;
                push_point(&mut current, cursor);
            }
            'V' => {
                let y = number(bytes, &mut index)?;
                cursor.1 = y;
                push_point(&mut current, cursor);
            }
            'C' => loop {
                let control_1 = pair(bytes, &mut index)?;
                let control_2 = pair(bytes, &mut index)?;
                let end = pair(bytes, &mut index)?;
                flatten(&mut current, cursor, control_1, control_2, end);
                cursor = end;
                if !next_is_number(bytes, index) {
                    break;
                }
            },
            'Z' => {
                if let Some(outline) = current.as_mut() {
                    outline.closed = true;
                    if let Some(first) = outline.points.first().copied() {
                        cursor = first;
                    }
                }
            }
            _ => return Err(()),
        }
    }
    push_outline(&mut outlines, current);
    if outlines.is_empty() {
        return Err(());
    }
    Ok(outlines)
}

fn push_outline(outlines: &mut Vec<Outline>, outline: Option<Outline>) {
    if let Some(outline) = outline
        && outline.points.len() >= 2
    {
        outlines.push(outline);
    }
}

fn push_point(outline: &mut Option<Outline>, point: (f32, f32)) {
    if let Some(outline) = outline {
        outline.points.push(point);
    }
}

fn flatten(
    outline: &mut Option<Outline>,
    start: (f32, f32),
    control_1: (f32, f32),
    control_2: (f32, f32),
    end: (f32, f32),
) {
    const STEPS: i32 = 8;
    for step in 1..=STEPS {
        let t = step as f32 / STEPS as f32;
        let u = 1.0 - t;
        let x = u * u * u * start.0
            + 3.0 * u * u * t * control_1.0
            + 3.0 * u * t * t * control_2.0
            + t * t * t * end.0;
        let y = u * u * u * start.1
            + 3.0 * u * u * t * control_1.1
            + 3.0 * u * t * t * control_2.1
            + t * t * t * end.1;
        push_point(outline, (x, y));
    }
}

fn pair(bytes: &[u8], index: &mut usize) -> Result<(f32, f32), ()> {
    let x = number(bytes, index)?;
    let y = number(bytes, index)?;
    Ok((x, y))
}

fn optional_pair(bytes: &[u8], index: &mut usize) -> Option<(f32, f32)> {
    let saved = *index;
    match pair(bytes, index) {
        Ok(point) => Some(point),
        Err(()) => {
            *index = saved;
            None
        }
    }
}

fn next_is_number(bytes: &[u8], index: usize) -> bool {
    let mut cursor = index;
    skip(bytes, &mut cursor);
    matches!(bytes.get(cursor), Some(b'0'..=b'9' | b'+' | b'-' | b'.'))
}

fn number(bytes: &[u8], index: &mut usize) -> Result<f32, ()> {
    skip(bytes, index);
    let start = *index;
    if matches!(bytes.get(*index), Some(b'+' | b'-')) {
        *index += 1;
    }
    let mut saw_digit = false;
    while matches!(bytes.get(*index), Some(b'0'..=b'9')) {
        *index += 1;
        saw_digit = true;
    }
    if bytes.get(*index) == Some(&b'.') {
        *index += 1;
        while matches!(bytes.get(*index), Some(b'0'..=b'9')) {
            *index += 1;
            saw_digit = true;
        }
    }
    if !saw_digit || *index == start {
        return Err(());
    }
    std::str::from_utf8(&bytes[start..*index])
        .ok()
        .and_then(|text| text.parse().ok())
        .ok_or(())
}

fn skip(bytes: &[u8], index: &mut usize) {
    while matches!(bytes.get(*index), Some(byte) if is_sep(*byte)) {
        *index += 1;
    }
}

fn is_sep(byte: u8) -> bool {
    byte.is_ascii_whitespace() || byte == b','
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mapped_icon_strokes_and_a_missing_name_is_an_error() {
        assert!(ICONS.len() >= 65);
        assert!(require("not-an-icon").is_err());
        for icon in ICONS {
            let drawn = outlines(icon.name).expect(icon.name);
            assert!(
                !drawn.is_empty()
                    || icon
                        .marks
                        .iter()
                        .any(|mark| matches!(mark, Mark::Circle { .. })),
            );
            for outline in drawn {
                assert!(outline.len() >= 2, "{}", icon.name);
            }
        }
        assert!(is_filled("blade").expect("blade"));
        assert!(!is_filled("play").expect("play"));
    }

    #[test]
    fn the_hugeicons_notice_stays_with_the_path_data() {
        let source = include_str!("icons/icon_data.rs");
        assert!(source.contains("MIT License"));
        assert!(source.contains("Copyright (c) 2025 Hugeicons"));
        assert!(source.contains("name: \"blade\""));
    }

    #[test]
    fn a_known_icon_paints_a_stroke() {
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(24.0, 24.0));
            paint(ui.painter(), rect, "play", egui::Color32::WHITE).expect("play");
            assert!(paint(ui.painter(), rect, "missing", egui::Color32::WHITE).is_err());
        });
        assert!(!output.shapes.is_empty());
        output.textures_delta.clear();
    }
}
