//! `PictureSource` for one text segment.
//!
//! A sample inside the segment reads the cached atlas through the mesh and
//! returns an RGBA picture. A sample outside the segment is a miss. The draft
//! is not written. Sampling does not paint the atlas again. A color strike
//! keeps its color. A gray or mono strike is coverage under the text color.
//! The title's outline, shadow, and background are drawn with the letters,
//! so the picture grows to hold them. The picture is premultiplied.
//!
//! A letter animation poses each letter: its opacity, a shift in ems and a
//! size around its own middle. A posed letter stays inside the picture
//! drawn at rest, so a letter shifted past its edge is cut there. A color
//! letter (an emoji) shows whole while its opacity is above nothing.

use cutlass_font::Font;
use cutlass_model::{BloomParams, Letter, LetterPose, MaterialText, Micros, Segment, TimeRange};
use cutlass_picture::{CpuBytes, Picture, PictureSource, Sample, rgba_len};

use crate::TextError;
use crate::atlas::{Atlas, GlyphKind};
use crate::layout::layout_faces;
use crate::mesh::{GlyphRecord, TextMesh, build_faces};
use crate::richtext::{Run, run_at};
use crate::style::{Area, Layers, Outline, Shadow, Style, compose, premultiply};

/// How one glyph looks: its range's fill (`None` takes the title's), and
/// the outline and shadow drawn with it, at its size.
#[derive(Clone, Debug, PartialEq)]
struct Look {
    ink: Option<[u8; 4]>,
    outline: Option<Outline>,
    shadow: Option<Shadow>,
}

impl Look {
    fn of(style: &Style, runs: &[Run], source: usize, font_size: f32) -> Self {
        let run = run_at(runs, source);
        let size = run.map_or(1.0, |run| run.size);
        let outline = match run.and_then(|run| run.outline.as_ref()) {
            Some(own) => own.as_ref(),
            None => style.outline.as_ref(),
        };
        let shadow = match run.and_then(|run| run.shadow.as_ref()) {
            Some(own) => own.as_ref(),
            None => style.shadow.as_ref(),
        };
        Self {
            ink: run.and_then(|run| run.fill),
            outline: outline.and_then(|outline| outline.at(size, font_size)),
            shadow: shadow.map(|shadow| shadow.at(size)),
        }
    }
}

/// How one glyph looks at a moment, as a caption's style samples it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlyphLook {
    /// Its place, size and opacity.
    pub pose: LetterPose,
    /// Its own ink, straight RGBA, over its run's.
    pub ink: Option<[u8; 4]>,
    /// The color of a round box behind it, straight RGBA; clear is none.
    pub back: [u8; 4],
}

impl GlyphLook {
    /// At rest, with its run's ink and no box.
    pub const REST: Self = Self {
        pose: LetterPose::REST,
        ink: None,
        back: [0, 0, 0, 0],
    };
}

/// The bounds a set of rectangles reach.
/// Bold, italic and underline over the whole title, in picture pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Pattern {
    /// How far each letter's edge grows outward.
    bold: f32,
    /// Pixels a letter leans right for each pixel above its baseline.
    shear: f32,
    /// The bar under each line: its thickness and how far below the
    /// baseline its top sits.
    underline: Option<(f32, f32)>,
}

/// The most a bold edge grows, as a share of the font size.
const MOST_BOLD: f32 = 0.1;
/// The steepest lean an italic takes, in degrees.
const MOST_ITALIC: f32 = 45.0;
/// An underline's thickness, as a share of the font size.
const UNDERLINE_WIDTH: f32 = 0.06;
/// How far under the baseline an underline's top sits, as a share of the
/// font size.
const UNDERLINE_OFFSET: f32 = 0.12;

impl Pattern {
    /// The pattern `material` asks for at `font_size` pixels. `bold_width`
    /// is the edge's growth as a share of the font size; `italic_degree` is
    /// the lean in degrees. Values that are not usable leave that part off.
    fn of(material: &MaterialText, font_size: f32) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        let bold = material.bold_width as f32;
        let bold = if bold.is_finite() && bold > 0.0 {
            bold.min(MOST_BOLD) * font_size
        } else {
            0.0
        };
        #[allow(clippy::cast_precision_loss)]
        let degrees = (material.italic_degree as f32).clamp(-MOST_ITALIC, MOST_ITALIC);
        let shear = degrees.to_radians().tan();
        let underline = material
            .underline
            .then_some((UNDERLINE_WIDTH * font_size, UNDERLINE_OFFSET * font_size));
        Self {
            bold,
            shear: if shear.is_finite() { shear } else { 0.0 },
            underline,
        }
    }

    /// The area a glyph at `record` reaches once leaned and grown, with its
    /// line's baseline at `baseline`.
    fn hold(self, reach: &mut Reach, record: &GlyphRecord, baseline: f32) {
        let lean_top = self.shear * (baseline - record.y);
        let lean_bottom = self.shear * (baseline - (record.y + record.height));
        reach.hold(
            record.x + lean_top.min(lean_bottom) - self.bold,
            record.y - self.bold,
            record.x + record.width + lean_top.max(lean_bottom) + self.bold,
            record.y + record.height + self.bold,
        );
    }
}

struct Reach {
    any: bool,
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

impl Default for Reach {
    fn default() -> Self {
        Self {
            any: false,
            left: f32::MAX,
            top: f32::MAX,
            right: f32::MIN,
            bottom: f32::MIN,
        }
    }
}

impl Reach {
    fn hold(&mut self, left: f32, top: f32, right: f32, bottom: f32) {
        if ![left, top, right, bottom]
            .iter()
            .all(|value| value.is_finite())
        {
            return;
        }
        self.any = true;
        self.left = self.left.min(left);
        self.top = self.top.min(top);
        self.right = self.right.max(right);
        self.bottom = self.bottom.max(bottom);
    }
}

/// A text segment that can be sampled at a timeline microsecond.
pub struct TextSource {
    range: TimeRange,
    material_id: cutlass_model::MaterialId,
    fill: [u8; 4],
    /// One for each face, the title's own first.
    atlases: Vec<Atlas>,
    mesh: TextMesh,
    style: Style,
    /// How each glyph in the mesh looks, in its order.
    looks: Vec<Look>,
    /// Which letter and word each glyph in the mesh is.
    letters: Vec<Letter>,
    /// How each glyph is posed by a letter animation; empty is at rest.
    poses: Vec<LetterPose>,
    /// Each glyph's own ink, straight, over its run's; empty keeps them.
    inks: Vec<Option<[u8; 4]>>,
    /// Round boxes behind runs of posed glyphs, in picture pixels.
    boxes: Vec<crate::paint::WordBox>,
    font_size: f32,
    /// Bold, italic and underline over every letter.
    pattern: Pattern,
    /// Each line's baseline, in the glyph records' pixels.
    baselines: Vec<f32>,
    /// The box behind the letters, in picture pixels.
    background: Option<Area>,
    origin_x: f32,
    origin_y: f32,
    width: u32,
    height: u32,
}

impl TextSource {
    /// Read one face and prepare the atlas and mesh.
    ///
    /// `face` is `None` for a single font file and `Some(index)` for a
    /// collection. A face without outlines is refused.
    pub fn open(
        bytes: &[u8],
        face: Option<u32>,
        material: &MaterialText,
        segment: &Segment,
    ) -> Result<Self, TextError> {
        Self::open_with_glow(bytes, face, material, None, segment)
    }

    /// [`Self::open`], with the clip's glow drawn behind the letters.
    pub fn open_with_glow(
        bytes: &[u8],
        face: Option<u32>,
        material: &MaterialText,
        glow: Option<&BloomParams>,
        segment: &Segment,
    ) -> Result<Self, TextError> {
        Self::open_with_faces(bytes, face, material, glow, segment, &|_| None)
    }

    /// [`Self::open_with_glow`], with a rich title's ranges set in the
    /// faces they name. `faces` gives the bytes of one face for a range's
    /// font path; a range whose face it does not give, or that does not
    /// read, keeps the title's own.
    pub fn open_with_faces<'f>(
        bytes: &[u8],
        face: Option<u32>,
        material: &MaterialText,
        glow: Option<&BloomParams>,
        segment: &Segment,
        faces: &dyn Fn(&str) -> Option<&'f [u8]>,
    ) -> Result<Self, TextError> {
        if segment.target.duration.0 > 0
            && segment
                .target
                .start
                .0
                .checked_add(segment.target.duration.0)
                .is_none()
        {
            return Err(TextError::Measure);
        }
        let font = match face {
            None => cutlass_font::read(bytes).map_err(TextError::Font)?,
            Some(index) => cutlass_font::read_collection(bytes, index).map_err(TextError::Font)?,
        };
        let runs = crate::richtext::read(material)?.runs;
        let mut fonts = vec![font];
        let mut named: Vec<(&str, usize)> = Vec::new();
        for path in runs.iter().filter_map(|run| run.font.as_deref()) {
            if named.iter().any(|(seen, _)| *seen == path) {
                continue;
            }
            let Some(other) = faces(path).and_then(|bytes| cutlass_font::read(bytes).ok()) else {
                named.push((path, 0));
                continue;
            };
            named.push((path, fonts.len()));
            fonts.push(other);
        }
        let pick = |source: usize| {
            run_at(&runs, source)
                .and_then(|run| run.font.as_deref())
                .and_then(|path| named.iter().find(|(seen, _)| *seen == path))
                .map_or(0, |(_, index)| *index)
        };
        Self::from_fonts(&fonts, &pick, &runs, material, glow, segment)
    }

    fn from_fonts(
        fonts: &[Font<'_>],
        pick: &dyn Fn(usize) -> usize,
        runs: &[Run],
        material: &MaterialText,
        glow: Option<&BloomParams>,
        segment: &Segment,
    ) -> Result<Self, TextError> {
        let fill = fill_rgba(material)?;
        let laid = layout_faces(fonts, material, pick)?;
        let mut atlases: Vec<Atlas> = fonts.iter().map(|_| Atlas::new()).collect();
        for placed in laid.glyphs() {
            let (Some(atlas), Some(font)) = (atlases.get_mut(placed.face), fonts.get(placed.face))
            else {
                return Err(TextError::Limit);
            };
            let _image = atlas.glyph(font, placed.glyph_id)?;
        }
        let mesh = build_faces(material.id.clone(), &laid, &atlases)?;
        let style = Style::of(material, glow, laid.font_size)?;
        let looks: Vec<Look> = mesh
            .glyphs
            .iter()
            .map(|record| Look::of(&style, runs, record.source, laid.font_size))
            .collect();
        let mut shadows: Vec<Shadow> = Vec::new();
        for shadow in looks.iter().filter_map(|look| look.shadow.as_ref()) {
            if !shadows.contains(shadow) {
                shadows.push(shadow.clone());
            }
        }
        let (origin_x, origin_y, width, height, background) =
            picture_bounds(&mesh, &style, laid.font_size, &shadows)?;
        let letters = letters_of(&crate::richtext::read(material)?.text, &mesh.glyphs);
        let pattern = Pattern::of(material, laid.font_size);
        let baselines = laid.lines.iter().map(|line| line.baseline).collect();
        let mut source = Self {
            range: segment.target.clone(),
            material_id: material.id.clone(),
            fill,
            atlases,
            mesh,
            style,
            looks,
            letters,
            poses: Vec::new(),
            inks: Vec::new(),
            boxes: Vec::new(),
            font_size: laid.font_size,
            pattern,
            baselines,
            background,
            origin_x,
            origin_y,
            width,
            height,
        };
        if source.pattern != Pattern::default() {
            let mut reach = Reach::default();
            for (at, record) in source.mesh.glyphs.iter().enumerate() {
                if let Some((posed, _)) = source.posed(at, record) {
                    source.hold_pattern(&mut reach, &posed, record);
                }
            }
            source.grow_to(&reach);
        }
        Ok(source)
    }

    /// The baseline of `record`'s line, moved with it when a pose moved it
    /// to `posed`.
    fn baseline(&self, posed: &GlyphRecord, record: &GlyphRecord) -> f32 {
        let rest = usize::try_from(record.line)
            .ok()
            .and_then(|line| self.baselines.get(line).copied())
            .unwrap_or(record.y + record.height);
        rest + (posed.y + posed.height / 2.0) - (record.y + record.height / 2.0)
    }

    /// Add what the pattern makes `posed` reach, and its line's underline.
    fn hold_pattern(&self, reach: &mut Reach, posed: &GlyphRecord, record: &GlyphRecord) {
        let baseline = self.baseline(posed, record);
        self.pattern.hold(reach, posed, baseline);
        if let Some((thick, below)) = self.pattern.underline {
            reach.hold(
                posed.x,
                baseline + below,
                posed.x + posed.width,
                baseline + below + thick,
            );
        }
    }

    /// Each line's underline, in picture pixels: left, top, right, bottom.
    fn underlines(&self) -> Vec<[f32; 4]> {
        let Some((thick, below)) = self.pattern.underline else {
            return Vec::new();
        };
        let mut lines: Vec<(u32, [f32; 4])> = Vec::new();
        for (at, record) in self.mesh.glyphs.iter().enumerate() {
            if record.width <= 0.0 || record.height <= 0.0 {
                continue;
            }
            let Some((posed, _)) = self.posed(at, record) else {
                continue;
            };
            let top = self.baseline(&posed, record) + below - self.origin_y;
            let (left, right) = (
                posed.x - self.origin_x,
                posed.x + posed.width - self.origin_x,
            );
            match lines.iter_mut().find(|(line, _)| *line == record.line) {
                Some((_, bar)) => {
                    bar[0] = bar[0].min(left);
                    bar[2] = bar[2].max(right);
                }
                None => lines.push((record.line, [left, top, right, top + thick])),
            }
        }
        lines.into_iter().map(|(_, bar)| bar).collect()
    }

    /// The picture's width and height: tight around the drawn glyphs.
    #[must_use]
    pub const fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[must_use]
    pub fn material_id(&self) -> &cutlass_model::MaterialId {
        &self.material_id
    }

    #[must_use]
    pub fn atlas_paints(&self) -> u32 {
        self.atlases.iter().map(Atlas::paints).sum()
    }

    #[must_use]
    pub fn records(&self) -> &[GlyphRecord] {
        &self.mesh.glyphs
    }

    /// Which letter and word each glyph is, in the order of [`Self::records`].
    #[must_use]
    pub fn letters(&self) -> &[Letter] {
        &self.letters
    }

    /// Pose each glyph, in the order of [`Self::records`], for the next
    /// sample; an empty list draws every letter at rest.
    pub fn set_poses(&mut self, poses: Vec<LetterPose>) {
        self.poses = poses;
    }

    /// Give each glyph, in the order of [`Self::records`], a look for the
    /// next sample: its pose, its own ink, and a box behind it. Glyphs in a
    /// row on one line with the same box color share one round box. The
    /// picture grows evenly about its middle to hold glyphs moved or boxed
    /// past its edges, so a placement about its centre keeps the text
    /// where it rests.
    pub fn set_looks(&mut self, looks: &[GlyphLook]) {
        self.poses = looks.iter().map(|look| look.pose).collect();
        self.inks = looks.iter().map(|look| look.ink).collect();
        let mut boxes: Vec<crate::paint::WordBox> = Vec::new();
        let mut last: Option<([u8; 4], u32)> = None;
        let mut reach = Reach::default();
        for (at, (record, look)) in self.mesh.glyphs.iter().zip(looks).enumerate() {
            let Some((posed, _)) = self.posed(at, record) else {
                continue;
            };
            if posed.width <= 0.0 || posed.height <= 0.0 {
                continue;
            }
            reach.hold(
                posed.x,
                posed.y,
                posed.x + posed.width,
                posed.y + posed.height,
            );
            self.hold_pattern(&mut reach, &posed, record);
            if look.back[3] == 0 {
                last = None;
                continue;
            }
            let around = crate::paint::WordBox::round(&posed, look.back, self.font_size);
            reach.hold(around.left, around.top, around.right, around.bottom);
            match boxes.last_mut() {
                Some(open) if last == Some((look.back, record.line)) => open.join(&around),
                _ => boxes.push(around),
            }
            last = Some((look.back, record.line));
        }
        self.boxes = boxes;
        self.grow_to(&reach);
    }

    /// Grow the picture evenly about its middle until it holds `reach`,
    /// with room for an outline and a shadow.
    fn grow_to(&mut self, reach: &Reach) {
        if !reach.any {
            return;
        }
        let room = self.font_size * 0.25;
        let (left, top) = (self.origin_x, self.origin_y);
        #[allow(clippy::cast_precision_loss)]
        let (right, bottom) = (left + self.width as f32, top + self.height as f32);
        let more_x = (left - (reach.left - room))
            .max((reach.right + room) - right)
            .max(0.0)
            .ceil();
        let more_y = (top - (reach.top - room))
            .max((reach.bottom + room) - bottom)
            .max(0.0)
            .ceil();
        if !more_x.is_finite() || !more_y.is_finite() || more_x > 4_096.0 || more_y > 4_096.0 {
            return;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (grow_x, grow_y) = (more_x as u32, more_y as u32);
        let (width, height) = (self.width + 2 * grow_x, self.height + 2 * grow_y);
        if rgba_len(width, height).is_none() {
            return;
        }
        self.origin_x -= more_x;
        self.origin_y -= more_y;
        self.width = width;
        self.height = height;
        if let Some(area) = self.background.as_mut() {
            area.left += more_x;
            area.right += more_x;
            area.top += more_y;
            area.bottom += more_y;
        }
    }

    /// A glyph's place and opacity under its pose: `None` when it is not
    /// drawn at all.
    fn posed(&self, at: usize, record: &GlyphRecord) -> Option<(GlyphRecord, f32)> {
        let Some(pose) = self.poses.get(at) else {
            return Some((record.clone(), 1.0));
        };
        let alpha = pose.alpha.clamp(0.0, 1.0) as f32;
        let scale = pose.scale.max(0.0) as f32;
        if alpha <= 0.0 || scale <= 0.0 {
            return None;
        }
        let mut moved = record.clone();
        moved.width = record.width * scale;
        moved.height = record.height * scale;
        moved.x = record.x + (record.width - moved.width) / 2.0 + pose.dx as f32 * self.font_size;
        moved.y = record.y + (record.height - moved.height) / 2.0 + pose.dy as f32 * self.font_size;
        [moved.x, moved.y, moved.width, moved.height]
            .iter()
            .all(|value| value.is_finite())
            .then_some((moved, alpha))
    }

    /// Replace the ink color. The material id and the atlas stay as they are.
    pub fn set_fill(&mut self, fill: [u8; 4]) {
        self.fill = fill;
    }

    fn raster(&self) -> Result<Picture, TextError> {
        let len = rgba_len(self.width, self.height).ok_or(TextError::Limit)?;
        let mut rgba = vec![0_u8; len];
        let pixels = len / 4;
        let base = premultiply(self.fill);
        let mut fill = vec![0.0_f32; pixels];
        // Each pixel takes the ink and outline color of the letter covering
        // it most.
        let mut inks = vec![base; pixels];
        let mut grown = self
            .looks
            .iter()
            .any(|look| look.outline.is_some())
            .then(|| vec![0.0_f32; pixels]);
        let mut rings = vec![[0_u8; 4]; if grown.is_some() { pixels } else { 0 }];
        let mut shadows: Vec<(Shadow, Vec<f32>)> = Vec::new();
        let mut shape = vec![0.0_f32; pixels];
        for (at, (record, look)) in self.mesh.glyphs.iter().zip(&self.looks).enumerate() {
            let Some(image) = self.image(record) else {
                continue;
            };
            let Some((record, alpha)) = self.posed(at, record) else {
                continue;
            };
            let draw = Mask {
                width: self.width,
                height: self.height,
                record: &record,
                image,
                origin_x: self.origin_x,
                origin_y: self.origin_y,
                alpha,
                shear: self.pattern.shear,
                baseline: self.baseline(&record, &self.mesh.glyphs[at]),
            };
            let bold = self.pattern.bold;
            let reach = look.outline.as_ref().map_or(0.0, |outline| outline.width) + bold;
            if let Some(shadow) = &look.shadow {
                let at = match shadows.iter().position(|(cast, _)| cast == shadow) {
                    Some(at) => at,
                    None => {
                        shadows.push((shadow.clone(), vec![0.0_f32; pixels]));
                        shadows.len() - 1
                    }
                };
                draw.cover(&mut shadows[at].1, reach);
            }
            draw.cover(&mut shape, reach);
            if image.kind == GlyphKind::Color {
                continue;
            }
            let own = self.inks.get(at).copied().flatten();
            let ink = own.or(look.ink).map_or(base, premultiply);
            draw.each(bold, |index, cover| {
                if cover > fill[index] {
                    fill[index] = cover;
                    inks[index] = ink;
                }
            });
            if let (Some(outline), Some(grown)) = (&look.outline, &mut grown) {
                draw.each(outline.width + bold, |index, cover| {
                    if cover > grown[index] {
                        grown[index] = cover;
                        rings[index] = outline.color;
                    }
                });
            }
        }
        let outline = self.looks.iter().find_map(|look| look.outline.as_ref());
        for bar in self.underlines() {
            self.cover_bar(bar, 0.0, |index, cover| {
                if cover > fill[index] {
                    fill[index] = cover;
                    inks[index] = base;
                }
                shape[index] = shape[index].max(cover);
                for (_, cast) in &mut shadows {
                    cast[index] = cast[index].max(cover);
                }
            });
            if let (Some(outline), Some(grown)) = (outline, &mut grown) {
                self.cover_bar(bar, outline.width, |index, cover| {
                    if cover > grown[index] {
                        grown[index] = cover;
                        rings[index] = outline.color;
                    }
                });
            }
        }
        compose(
            &Layers {
                style: &self.style,
                ink: base,
                fill: &fill,
                inks: &inks,
                grown: grown.as_deref(),
                rings: &rings,
                shadows: &shadows,
                shape: &shape,
                background: self.background,
                boxes: &self.boxes,
                origin: (self.origin_x, self.origin_y),
                width: self.width as usize,
                height: self.height as usize,
            },
            &mut rgba,
        );
        for (at, record) in self.mesh.glyphs.iter().enumerate() {
            if self
                .image(record)
                .is_some_and(|image| image.kind == GlyphKind::Color)
                && let Some((record, _)) = self.posed(at, record)
            {
                self.splat(&mut rgba, &record);
            }
        }
        let bytes = CpuBytes::new(self.width, self.height, rgba).map_err(|_| TextError::Limit)?;
        Ok(Picture::Cpu(bytes))
    }

    /// Hand each picture pixel the bar `[left, top, right, bottom]` covers,
    /// grown by `grow` pixels, to `put` with its coverage; edges are soft
    /// over one pixel.
    fn cover_bar(&self, bar: [f32; 4], grow: f32, mut put: impl FnMut(usize, f32)) {
        let [left, top, right, bottom] = bar;
        let (left, top, right, bottom) = (left - grow, top - grow, right + grow, bottom + grow);
        let Some((x0, x1)) = span(left, right, self.width) else {
            return;
        };
        let Some((y0, y1)) = span(top, bottom, self.height) else {
            return;
        };
        for y in y0..y1 {
            let (y_top, y_bottom) = (pixel(y), pixel(y) + 1.0);
            let cover_y = (y_bottom.min(bottom) - y_top.max(top)).clamp(0.0, 1.0);
            for x in x0..x1 {
                let (x_left, x_right) = (pixel(x), pixel(x) + 1.0);
                let cover_x = (x_right.min(right) - x_left.max(left)).clamp(0.0, 1.0);
                if let Ok(index) = usize::try_from(y * self.width + x) {
                    put(index, cover_x * cover_y);
                }
            }
        }
    }

    fn image(&self, record: &GlyphRecord) -> Option<&crate::GlyphImage> {
        self.atlases.get(record.face)?.image(record.glyph_id)
    }

    fn splat(&self, rgba: &mut [u8], record: &GlyphRecord) {
        let Some(image) = self.image(record) else {
            return;
        };
        blit(Blit {
            rgba,
            picture_w: self.width,
            picture_h: self.height,
            record,
            image,
            fill: self.fill,
            origin_x: self.origin_x,
            origin_y: self.origin_y,
        });
    }
}

pub(crate) struct Blit<'a> {
    pub(crate) rgba: &'a mut [u8],
    pub(crate) picture_w: u32,
    pub(crate) picture_h: u32,
    pub(crate) record: &'a GlyphRecord,
    pub(crate) image: &'a crate::GlyphImage,
    pub(crate) fill: [u8; 4],
    pub(crate) origin_x: f32,
    pub(crate) origin_y: f32,
}

/// One glyph's coverage laid into a picture-sized mask.
struct Mask<'a> {
    width: u32,
    height: u32,
    record: &'a GlyphRecord,
    image: &'a crate::GlyphImage,
    origin_x: f32,
    origin_y: f32,
    /// The letter's opacity under its pose.
    alpha: f32,
    /// Pixels the letter leans right for each pixel above `baseline`.
    shear: f32,
    /// The letter's baseline, in the records' pixels.
    baseline: f32,
}

impl Mask<'_> {
    /// Keep the larger coverage of each pixel, with a distance field glyph
    /// grown by `grow` pixels.
    fn cover(&self, mask: &mut [f32], grow: f32) {
        self.each(grow, |index, cover| {
            if let Some(slot) = mask.get_mut(index) {
                *slot = slot.max(cover);
            }
        });
    }

    /// Hand each pixel in the picture the glyph covers, by index, to `put`
    /// with its coverage.
    fn each(&self, grow: f32, mut put: impl FnMut(usize, f32)) {
        let Self {
            width,
            height,
            record,
            image,
            origin_x,
            origin_y,
            alpha,
            shear,
            baseline,
        } = *self;
        if record.width <= 0.0 || record.height <= 0.0 {
            return;
        }
        if image.width == 0 || image.height == 0 || !drawable(image) {
            return;
        }
        let left = record.x - origin_x;
        let top = record.y - origin_y;
        let baseline = baseline - origin_y;
        // A leaning letter reaches past its quad on the side it leans to.
        let lean_top = shear * (baseline - top);
        let lean_bottom = shear * (baseline - (top + record.height));
        let Some((x0, x1)) = span(
            left + lean_top.min(lean_bottom),
            left + record.width + lean_top.max(lean_bottom),
            width,
        ) else {
            return;
        };
        let Some((y0, y1)) = span(top, top + record.height, height) else {
            return;
        };
        let texels = pixel(image.width) / record.width;
        for y in y0..y1 {
            let lean = shear * (baseline - (pixel(y) + 0.5));
            for x in x0..x1 {
                let u = (pixel(x) + 0.5 - lean - left) / record.width;
                let v = (pixel(y) + 0.5 - top) / record.height;
                if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                    continue;
                }
                let Some(cover) = cover_at(image, u, v, texels, grow) else {
                    continue;
                };
                if let Ok(index) = usize::try_from(y * width + x) {
                    put(index, cover * alpha);
                }
            }
        }
    }
}

/// How much of the pixel at `u`, `v` the glyph covers, from 0 to 1.
fn cover_at(image: &crate::GlyphImage, u: f32, v: f32, texels: f32, grow: f32) -> Option<f32> {
    let (col, row) = image_texel(image, u, v)?;
    let byte = match image.kind {
        GlyphKind::Distance => coverage(distance_at(image, u, v)?, texels, grow),
        GlyphKind::Coverage => *image.sdf.get(texel_offset(image, col, row)?)?,
        GlyphKind::Color => {
            let offset = texel_offset(image, col, row)?.checked_mul(4)?;
            *image.color.get(offset + 3)?
        }
    };
    Some(f32::from(byte) / 255.0)
}

pub(crate) fn blit(draw: Blit<'_>) {
    let Blit {
        rgba,
        picture_w,
        picture_h,
        record,
        image,
        fill,
        origin_x,
        origin_y,
    } = draw;
    if record.width <= 0.0 || record.height <= 0.0 {
        return;
    }
    if image.width == 0 || image.height == 0 || !drawable(image) {
        return;
    }
    let left = record.x - origin_x;
    let top = record.y - origin_y;
    let Some((x0, x1)) = span(left, left + record.width, picture_w) else {
        return;
    };
    let Some((y0, y1)) = span(top, top + record.height, picture_h) else {
        return;
    };
    // Distance field texels across one drawn pixel, so an edge stays one
    // pixel soft at any size.
    let texels = pixel(image.width) / record.width;
    for y in y0..y1 {
        for x in x0..x1 {
            let u = (pixel(x) + 0.5 - left) / record.width;
            let v = (pixel(y) + 0.5 - top) / record.height;
            if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                continue;
            }
            let Some(ink) = ink_at(image, u, v, texels, fill) else {
                continue;
            };
            blend_at(rgba, picture_w, x, y, ink);
        }
    }
}

impl PictureSource for TextSource {
    fn sample(&self, time: Micros) -> Sample {
        if !covers(&self.range, time) {
            return Sample::Miss;
        }
        match self.raster() {
            Ok(picture) => Sample::Ready(picture),
            Err(_) => Sample::Miss,
        }
    }
}

/// Which letter and word each glyph is. A letter is a glyph that draws
/// something; a word is a run of characters between white space.
fn letters_of(text: &str, glyphs: &[GlyphRecord]) -> Vec<Letter> {
    let mut starts = Vec::new();
    let mut inside = false;
    for (byte, character) in text.char_indices() {
        let space = character.is_whitespace();
        if !space && !inside {
            starts.push(byte);
        }
        inside = !space;
    }
    let drawn = |record: &GlyphRecord| record.width > 0.0 && record.height > 0.0;
    let count = glyphs.iter().filter(|record| drawn(record)).count();
    let words = starts.len();
    let mut index = 0;
    glyphs
        .iter()
        .map(|record| {
            let letter = Letter {
                index,
                count,
                word: starts
                    .partition_point(|start| *start <= record.source)
                    .saturating_sub(1),
                words,
            };
            if drawn(record) {
                index += 1;
            }
            letter
        })
        .collect()
}

fn covers(range: &TimeRange, time: Micros) -> bool {
    let duration = range.duration.0;
    if duration <= 0 {
        return false;
    }
    let Some(end) = range.start.0.checked_add(duration) else {
        return false;
    };
    time.0 >= range.start.0 && time.0 < end
}

/// The picture's origin and size around the letters and their style, and
/// the box behind the letters in picture pixels.
fn picture_bounds(
    mesh: &TextMesh,
    style: &Style,
    font_size: f32,
    shadows: &[Shadow],
) -> Result<(f32, f32, u32, u32, Option<Area>), TextError> {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let mut any = false;
    for glyph in &mesh.glyphs {
        if glyph.width <= 0.0 || glyph.height <= 0.0 {
            continue;
        }
        if ![glyph.x, glyph.y, glyph.width, glyph.height]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(TextError::Limit);
        }
        any = true;
        min_x = min_x.min(glyph.x);
        min_y = min_y.min(glyph.y);
        max_x = max_x.max(glyph.x + glyph.width);
        max_y = max_y.max(glyph.y + glyph.height);
    }
    if !any {
        return Ok((0.0, 0.0, 1, 1, None));
    }
    let cells = Area {
        left: min_x,
        top: min_y,
        right: max_x,
        bottom: max_y,
    };
    let area = style.bounds(cells, font_size, shadows);
    if ![area.left, area.top, area.right, area.bottom]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(TextError::Limit);
    }
    let left = area.left.floor();
    let top = area.top.floor();
    // Even edges keep the picture's centre on a pixel corner, so a title
    // drawn at whole multiples of its size lands on whole pixels.
    let even = |edge: u32| edge + edge % 2;
    let width = even(edge(area.right - left)?);
    let height = even(edge(area.bottom - top)?);
    if rgba_len(width, height).is_none() {
        return Err(TextError::Limit);
    }
    let background = style.background.as_ref().map(|back| {
        let behind = back.area(cells, font_size);
        Area {
            left: behind.left - left,
            top: behind.top - top,
            right: behind.right - left,
            bottom: behind.bottom - top,
        }
    });
    Ok((left, top, width, height, background))
}

fn edge(value: f32) -> Result<u32, TextError> {
    if !value.is_finite() || value <= 0.0 || value > 8_192.0 {
        return Err(TextError::Limit);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let edge = value.ceil() as u32;
    if edge == 0 { Ok(1) } else { Ok(edge) }
}

fn span(start: f32, end: f32, limit: u32) -> Option<(u32, u32)> {
    if !start.is_finite() || !end.is_finite() || end <= start || limit == 0 {
        return None;
    }
    let start = start.floor().clamp(0.0, f32::from(u16::MAX));
    let end = end.ceil().clamp(0.0, f32::from(u16::MAX));
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let start = start as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let end = end as u32;
    let end = end.min(limit);
    let start = start.min(limit);
    if start >= end {
        None
    } else {
        Some((start, end))
    }
}

fn pixel(value: u32) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    {
        value as f32
    }
}

fn drawable(image: &crate::GlyphImage) -> bool {
    match image.kind {
        GlyphKind::Color => !image.color.is_empty(),
        GlyphKind::Distance | GlyphKind::Coverage => !image.sdf.is_empty(),
    }
}

fn ink_at(
    image: &crate::GlyphImage,
    u: f32,
    v: f32,
    texels: f32,
    fill: [u8; 4],
) -> Option<[u8; 4]> {
    let (col, row) = image_texel(image, u, v)?;
    match image.kind {
        GlyphKind::Distance => {
            let covered = coverage(distance_at(image, u, v)?, texels, 0.0);
            if covered == 0 {
                return None;
            }
            let alpha = u16::from(covered) * u16::from(fill[3]) / 255;
            Some(premultiply([
                fill[0],
                fill[1],
                fill[2],
                u8::try_from(alpha).ok()?,
            ]))
        }
        GlyphKind::Coverage => {
            let covered = *image.sdf.get(texel_offset(image, col, row)?)?;
            if covered == 0 {
                return None;
            }
            let alpha = u16::from(covered) * u16::from(fill[3]) / 255;
            Some(premultiply([
                fill[0],
                fill[1],
                fill[2],
                u8::try_from(alpha).ok()?,
            ]))
        }
        GlyphKind::Color => {
            let offset = texel_offset(image, col, row)?.checked_mul(4)?;
            let pixel = image.color.get(offset..offset + 4)?;
            let alpha = fill[3];
            if alpha == 0 || pixel[3] == 0 {
                return None;
            }
            Some([
                scale(pixel[0], alpha),
                scale(pixel[1], alpha),
                scale(pixel[2], alpha),
                scale(pixel[3], alpha),
            ])
        }
    }
}

fn scale(channel: u8, alpha: u8) -> u8 {
    u8::try_from(u16::from(channel) * u16::from(alpha) / 255).unwrap_or(0)
}

fn image_texel(image: &crate::GlyphImage, u: f32, v: f32) -> Option<(u32, u32)> {
    let col = (u * pixel(image.width)).floor();
    let row = (v * pixel(image.height)).floor();
    if col < 0.0 || row < 0.0 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let col = col as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let row = row as u32;
    if col >= image.width || row >= image.height {
        None
    } else {
        Some((col, row))
    }
}

fn texel_offset(image: &crate::GlyphImage, col: u32, row: u32) -> Option<usize> {
    let offset = row.checked_mul(image.width)?.checked_add(col)?;
    usize::try_from(offset).ok()
}

/// The distance field at `u`, `v`, mixed from the four nearest texels so a
/// glyph drawn larger than its cell keeps smooth outlines: 0 far outside,
/// 255 far inside.
fn distance_at(image: &crate::GlyphImage, u: f32, v: f32) -> Option<f32> {
    let x = (u * pixel(image.width) - 0.5).clamp(0.0, pixel(image.width - 1));
    let y = (v * pixel(image.height) - 0.5).clamp(0.0, pixel(image.height - 1));
    let (fx, fy) = (x.fract(), y.fract());
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (col, row) = (x as u32, y as u32);
    let next_col = (col + 1).min(image.width - 1);
    let next_row = (row + 1).min(image.height - 1);
    let at = |c: u32, r: u32| -> Option<f32> {
        Some(f32::from(*image.sdf.get(texel_offset(image, c, r)?)?))
    };
    let top = at(col, row)? * (1.0 - fx) + at(next_col, row)? * fx;
    let bottom = at(col, next_row)? * (1.0 - fx) + at(next_col, next_row)? * fx;
    Some(top * (1.0 - fy) + bottom * fy)
}

/// How much of a drawn pixel the glyph covers, with the outline mixed over
/// one pixel. `texels` is distance field texels per drawn pixel, and
/// `grow` widens the glyph by that many drawn pixels.
fn coverage(sdf: f32, texels: f32, grow: f32) -> u8 {
    // The field holds SPREAD texels each side of the outline.
    let inside = (sdf / 255.0 - 0.5) * 2.0 * crate::atlas::SPREAD;
    let texels = if texels.is_finite() && texels > 0.0 {
        texels
    } else {
        1.0
    };
    let covered = (inside / texels + grow + 0.5).clamp(0.0, 1.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let byte = (covered * 255.0).round() as u8;
    byte
}

fn blend_at(rgba: &mut [u8], width: u32, x: u32, y: u32, source: [u8; 4]) {
    let Some(offset) = y.checked_mul(width).and_then(|row| row.checked_add(x)) else {
        return;
    };
    let Some(offset) = usize::try_from(offset)
        .ok()
        .and_then(|index| index.checked_mul(4))
    else {
        return;
    };
    let Some(pixel) = rgba.get_mut(offset..offset + 4) else {
        return;
    };
    let inverse = u16::from(255_u8.saturating_sub(source[3]));
    for channel in 0..4 {
        let mixed = u16::from(source[channel]) + u16::from(pixel[channel]) * inverse / 255;
        pixel[channel] = u8::try_from(mixed.min(255)).unwrap_or(255);
    }
}

/// The ink color of `material`, including its alpha.
pub fn text_fill(material: &MaterialText) -> Result<[u8; 4], TextError> {
    fill_rgba(material)
}

pub(crate) fn fill_rgba(material: &MaterialText) -> Result<[u8; 4], TextError> {
    let text_alpha = unit_u8(material.text_alpha)?;
    let global_alpha = unit_u8(material.global_alpha)?;
    let (rgb, extra) = parse_hex(material.text_color.trim())?;
    let mut alpha = u16::from(text_alpha) * u16::from(global_alpha) / 255;
    if let Some(channel) = extra {
        alpha = u16::from(channel) * alpha / 255;
    }
    let alpha = u8::try_from(alpha).unwrap_or(0);
    Ok([rgb[0], rgb[1], rgb[2], alpha])
}

fn unit_u8(value: f64) -> Result<u8, TextError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(TextError::Measure);
    }
    let byte = (value * 255.0).round();
    if !(0.0..=255.0).contains(&byte) {
        return Err(TextError::Measure);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(byte as u8)
}

pub(crate) fn parse_hex(text: &str) -> Result<([u8; 3], Option<u8>), TextError> {
    if text.is_empty() {
        return Ok(([255, 255, 255], None));
    }
    let text = text.strip_prefix('#').unwrap_or(text);
    match text.len() {
        6 => Ok((
            [hex(&text[0..2])?, hex(&text[2..4])?, hex(&text[4..6])?],
            None,
        )),
        8 => Ok((
            [hex(&text[0..2])?, hex(&text[2..4])?, hex(&text[4..6])?],
            Some(hex(&text[6..8])?),
        )),
        _ => Err(TextError::Color),
    }
}

fn hex(text: &str) -> Result<u8, TextError> {
    u8::from_str_radix(text, 16).map_err(|_| TextError::Color)
}

#[cfg(test)]
mod tests {
    use super::TextSource;
    use crate::TextError;
    use cutlass_font::FontError;
    use cutlass_model::{BloomParams, Draft, LetterPose, MaterialText, Micros, Segment, TimeRange};
    use cutlass_picture::{Picture, PictureSource, Sample};

    const TTF: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");
    const TTC: &[u8] = include_bytes!("../../font/fixtures/sample.ttc");
    const NO_OUTLINE: &[u8] = include_bytes!("../../font/fixtures/no-outline.otf");
    const COLOR: &[u8] = include_bytes!("../../font/fixtures/color-strike.otf");
    const GRAY: &[u8] = include_bytes!("../../font/fixtures/gray-strike.otf");

    fn material(content: &str) -> MaterialText {
        let mut text = MaterialText::new("text-material");
        text.content = content.to_owned();
        text.font_size = 64.0;
        text.text_color = "#FFFFFF".to_owned();
        text.background_color = "#00FF00".to_owned();
        text
    }

    fn segment() -> Segment {
        Segment::new("segment", "text-material", TimeRange::new(1_000, 5_000))
    }

    fn cpu(sample: Sample) -> Vec<u8> {
        match sample {
            Sample::Ready(Picture::Cpu(bytes)) => bytes.rgba().to_vec(),
            Sample::Ready(Picture::Platform(_)) => panic!("platform buffer"),
            Sample::Miss => panic!("miss"),
        }
    }

    fn opaque_index(rgba: &[u8]) -> Option<usize> {
        rgba.chunks(4)
            .position(|pixel| pixel.len() == 4 && pixel[3] == 255)
    }

    #[test]
    fn a_sample_is_ready_inside_the_segment_and_a_miss_outside_it() {
        let text = material("A");
        let clip = segment();
        let mut draft = Draft::empty("draft");
        draft.materials.texts.push(text.clone());
        draft.tracks[0].segments.push(clip.clone());
        let before = draft.clone();
        let source = TextSource::open(TTF, None, &text, &clip).expect("open");
        let paints = source.atlas_paints();
        assert!(paints >= 1);
        assert_eq!(source.records().len(), 1);
        assert_eq!(source.material_id().as_str(), "text-material");

        let ready = cpu(source.sample(Micros(1_000)));
        let index = opaque_index(&ready).expect("ink");
        assert_eq!(&ready[index * 4..index * 4 + 4], &[255, 255, 255, 255]);
        assert!(ready.chunks(4).any(|pixel| pixel[3] == 0));
        assert!(matches!(source.sample(Micros(5_999)), Sample::Ready(_)));
        assert!(matches!(source.sample(Micros(6_000)), Sample::Miss));
        assert!(matches!(source.sample(Micros(999)), Sample::Miss));
        assert_eq!(source.atlas_paints(), paints);
        assert_eq!(draft, before);
        assert_eq!(text.content, "A");
    }

    #[test]
    fn a_fill_change_samples_the_cached_atlas() {
        let text = material("A");
        let clip = segment();
        let mut source = TextSource::open(TTF, None, &text, &clip).expect("open");
        let paints = source.atlas_paints();
        let id = source.material_id().clone();
        let index = opaque_index(&cpu(source.sample(Micros(2_000)))).expect("ink");
        source.set_fill([255, 0, 0, 255]);
        let again = cpu(source.sample(Micros(2_000)));
        assert_eq!(&again[index * 4..index * 4 + 4], &[255, 0, 0, 255]);
        assert_eq!(source.material_id(), &id);
        assert_eq!(source.records().len(), 1);
        assert_eq!(source.atlas_paints(), paints);
    }

    #[test]
    fn an_empty_string_is_a_clear_picture_and_a_zero_duration_is_a_miss() {
        let text = material("");
        let clip = segment();
        let source = TextSource::open(TTF, None, &text, &clip).expect("empty");
        assert!(source.records().is_empty());
        let ready = cpu(source.sample(Micros(1_000)));
        assert!(ready.iter().all(|byte| *byte == 0));
        assert_eq!(source.atlas_paints(), 0);

        let mut dot = segment();
        dot.target = TimeRange::new(10, 0);
        let source = TextSource::open(TTF, None, &material("A"), &dot).expect("empty span");
        assert!(matches!(source.sample(Micros(10)), Sample::Miss));
    }

    #[test]
    fn a_bitmap_strike_keeps_its_pixels_and_is_not_painted_again() {
        let clip = segment();
        let mut color = material("A");
        color.text_color = "#00FF00".to_owned();
        let mut source = TextSource::open(COLOR, None, &color, &clip).expect("color");
        let paints = source.atlas_paints();
        assert_eq!(paints, 1);
        let ready = cpu(source.sample(Micros(1_000)));
        assert!(ready.chunks(4).any(|pixel| pixel == [255, 0, 0, 255]));
        assert!(ready.chunks(4).any(|pixel| pixel == [0, 128, 0, 128]));
        assert!(ready.chunks(4).any(|pixel| pixel == [0, 0, 255, 255]));
        source.set_fill([0, 0, 255, 255]);
        let again = cpu(source.sample(Micros(2_000)));
        assert!(again.chunks(4).any(|pixel| pixel == [255, 0, 0, 255]));
        assert_eq!(source.atlas_paints(), paints);

        let mut gray = material("A");
        gray.text_color = "#FF0000".to_owned();
        let gray_source = TextSource::open(GRAY, None, &gray, &clip).expect("gray");
        assert_eq!(gray_source.atlas_paints(), 1);
        let covered = cpu(gray_source.sample(Micros(1_000)));
        assert!(covered.chunks(4).any(|pixel| pixel == [255, 0, 0, 255]));
        // Premultiplied, so a half covered red pixel is half as red.
        assert!(covered.chunks(4).any(|pixel| pixel == [128, 0, 0, 128]));
        assert!(covered.chunks(4).any(|pixel| pixel == [64, 0, 0, 64]));
        assert!(covered.chunks(4).any(|pixel| pixel[3] == 0));
        assert_eq!(gray_source.atlas_paints(), 1);
    }

    fn picture(text: &MaterialText) -> (u32, u32, Vec<u8>) {
        let source = TextSource::open(TTF, None, text, &segment()).expect("open");
        let (width, height) = source.size();
        (width, height, cpu(source.sample(Micros(1_000))))
    }

    fn has(rgba: &[u8], color: [u8; 4]) -> bool {
        rgba.chunks(4).any(|pixel| pixel == color)
    }

    /// How many pixels hold any ink.
    fn inked(rgba: &[u8]) -> usize {
        rgba.chunks(4).filter(|pixel| pixel[3] > 0).count()
    }

    /// The middle of the ink in rows `rows` of a `width` wide picture.
    fn ink_middle(rgba: &[u8], width: u32, rows: std::ops::Range<u32>) -> f32 {
        let (mut sum, mut count) = (0.0_f32, 0.0_f32);
        for y in rows {
            for x in 0..width {
                let at = ((y * width + x) * 4 + 3) as usize;
                if rgba[at] > 128 {
                    sum += x as f32;
                    count += 1.0;
                }
            }
        }
        sum / count.max(1.0)
    }

    #[test]
    fn bold_grows_each_letter_and_the_picture_holds_it() {
        let plain = material("A");
        let mut bold = plain.clone();
        bold.bold_width = 0.04;
        let (plain_w, plain_h, plain_rgba) = picture(&plain);
        let (width, height, rgba) = picture(&bold);
        assert!(width >= plain_w && height >= plain_h);
        assert!(
            inked(&rgba) > inked(&plain_rgba) * 11 / 10,
            "{} {}",
            inked(&rgba),
            inked(&plain_rgba)
        );
        // Far past the most a bold edge grows, it stops at the most.
        let mut wild = plain.clone();
        wild.bold_width = 50.0;
        let mut most = plain.clone();
        most.bold_width = f64::from(super::MOST_BOLD);
        assert_eq!(picture(&wild), picture(&most));
        // A width that is not a number draws plain letters.
        wild.bold_width = f64::NAN;
        assert_eq!(picture(&wild), picture(&plain));
    }

    #[test]
    fn italic_leans_the_tops_of_the_letters_right() {
        let plain = material("A");
        let mut italic = plain.clone();
        italic.italic_degree = 15;
        let (plain_w, plain_h, plain_rgba) = picture(&plain);
        let (width, height, rgba) = picture(&italic);
        let lean = |rgba: &[u8], width: u32, height: u32| {
            ink_middle(rgba, width, 0..height / 3) - ink_middle(rgba, width, height * 2 / 3..height)
        };
        let upright = lean(&plain_rgba, plain_w, plain_h);
        let leaned = lean(&rgba, width, height);
        assert!(upright.abs() < 1.5, "{upright}");
        assert!(leaned > 3.0, "{leaned}");
        // Leaning the other way leans the tops left.
        italic.italic_degree = -15;
        let (width, height, rgba) = picture(&italic);
        assert!(lean(&rgba, width, height) < -3.0);
        // Past the steepest lean it stops there.
        italic.italic_degree = 90;
        let mut most = plain.clone();
        #[allow(clippy::cast_possible_truncation)]
        let steepest = super::MOST_ITALIC as i32;
        most.italic_degree = steepest;
        assert_eq!(picture(&italic), picture(&most));
    }

    #[test]
    fn an_underline_runs_under_the_whole_line_in_the_letters_color() {
        let mut plain = material("AA");
        plain.text_color = "#FF0000".to_owned();
        let mut underlined = plain.clone();
        underlined.underline = true;
        let (plain_w, plain_h, _) = picture(&plain);
        let (width, height, rgba) = picture(&underlined);
        assert!(width >= plain_w && height > plain_h);
        // Some row near the bottom is red across most of the line.
        let solid_row = (0..height).rev().find(|&y| {
            (0..width)
                .filter(|&x| {
                    let at = ((y * width + x) * 4) as usize;
                    rgba[at..at + 4] == [255, 0, 0, 255]
                })
                .count()
                > width as usize / 2
        });
        let row = solid_row.expect("a solid red bar");
        assert!(row > height / 2, "{row} of {height}");
    }

    #[test]
    fn an_outline_rings_the_letters_in_its_color() {
        let plain = material("A");
        let mut ringed = plain.clone();
        ringed.border_color = "#FF0000".to_owned();
        ringed.border_width = 0.06;
        let (width, height, rgba) = picture(&ringed);
        let (plain_w, plain_h, _) = picture(&plain);
        assert_eq!((width, height), (plain_w, plain_h));
        assert!(has(&rgba, [255, 0, 0, 255]));
        assert!(has(&rgba, [255, 255, 255, 255]));
        // No width, or no color, draws no outline.
        ringed.border_width = 0.0;
        assert!(!has(&picture(&ringed).2, [255, 0, 0, 255]));
        ringed.border_width = 0.06;
        ringed.border_color.clear();
        assert!(!has(&picture(&ringed).2, [255, 0, 0, 255]));
    }

    #[test]
    fn a_shadow_falls_by_its_angle_and_grows_the_picture() {
        let plain = material("A");
        let mut shaded = plain.clone();
        shaded.has_shadow = true;
        shaded.shadow_color = "#0000FF".to_owned();
        shaded.shadow_alpha = 1.0;
        shaded.shadow_distance = 20.0;
        shaded.shadow_angle = -90.0;
        let (plain_w, plain_h, _) = picture(&plain);
        let (width, height, rgba) = picture(&shaded);
        assert!(height >= plain_h + 12, "{plain_h} {height}");
        assert!(width.abs_diff(plain_w) <= 2, "{plain_w} {width}");
        let row = |y: u32| {
            rgba[(y * width * 4) as usize..((y + 1) * width * 4) as usize]
                .chunks(4)
                .any(|pixel| pixel == [0, 0, 255, 255])
        };
        // The shadow is straight below, so only the lower rows are blue.
        assert!((height / 2..height).any(row));
        assert!(!(0..plain_h / 3).any(row));
        shaded.has_shadow = false;
        assert_eq!(picture(&shaded).0, plain_w);
    }

    #[test]
    fn a_rich_title_draws_its_words_and_each_range_in_its_color() {
        let mut rich = material("A");
        rich.is_rich_text = true;
        rich.content = r#"{"text":"AA","styles":[{"range":[0,1],"fill":{"content":{"solid":{"color":[1,0,0]}}}},{"range":[1,2],"bold":true}]}"#.to_owned();
        let (width, _, rgba) = picture(&rich);
        let plain = material("AA");
        assert_eq!(width, picture(&plain).0);
        // The first letter is red; the second keeps the title's white.
        let (red, white) = (
            has(&rgba, [255, 0, 0, 255]),
            has(&rgba, [255, 255, 255, 255]),
        );
        assert!(red && white, "{red} {white}");
        let column = |x: u32| {
            (0..rgba.len() as u32 / 4 / width)
                .map(|y| ((y * width + x) * 4) as usize)
                .any(|at| rgba[at..at + 4] == [255, 0, 0, 255])
        };
        assert!((0..width / 2).any(column));
        assert!(!(width / 2..width).any(column));

        rich.content = "AA".to_owned();
        assert!(matches!(
            TextSource::open(TTF, None, &rich, &segment()),
            Err(TextError::RichTextCommands)
        ));
    }

    fn rich(styles: &str) -> MaterialText {
        let mut text = material("AA");
        text.is_rich_text = true;
        text.content = format!(r#"{{"text":"AA","styles":[{styles}]}}"#);
        text
    }

    /// Whether a pixel of `color` is in the left half, and in the right.
    fn halves(width: u32, rgba: &[u8], color: [u8; 4]) -> (bool, bool) {
        let mut found = (false, false);
        for (index, pixel) in rgba.chunks(4).enumerate() {
            if pixel == color {
                if (index as u32 % width) < width / 2 {
                    found.0 = true;
                } else {
                    found.1 = true;
                }
            }
        }
        found
    }

    #[test]
    fn a_range_sets_its_letters_in_the_face_it_names() {
        const OTHER: &[u8] = include_bytes!("../../font/fixtures/sample.otf");
        let text = rich(r#"{"range":[1,2],"font":{"path":"/fonts/Other.otf"}}"#);
        let faces = |path: &str| (path == "/fonts/Other.otf").then_some(OTHER);
        let source =
            TextSource::open_with_faces(TTF, None, &text, None, &segment(), &faces).expect("open");
        let faces_of: Vec<usize> = source.records().iter().map(|record| record.face).collect();
        assert_eq!(faces_of, [0, 1]);
        let (width, height) = source.size();
        let drawn = cpu(source.sample(Micros(1_000)));
        assert_eq!(drawn.len(), (width * height * 4) as usize);
        assert!(has(&drawn, [255, 255, 255, 255]));

        // A face the shell does not have keeps the title's.
        let none = |_: &str| None;
        let source =
            TextSource::open_with_faces(TTF, None, &text, None, &segment(), &none).expect("open");
        assert!(source.records().iter().all(|record| record.face == 0));
        let (plain_w, plain_h, _) = picture(&material("AA"));
        assert_eq!(source.size(), (plain_w, plain_h));
    }

    #[test]
    fn a_range_draws_at_its_own_size() {
        let (plain_w, plain_h, _) = picture(&material("AA"));
        let (width, height, _) = picture(&rich(r#"{"range":[1,2],"size":128}"#));
        // The second letter is twice as large, so the line grows by about
        // one letter's width and stands about twice as tall.
        assert!(width > plain_w * 5 / 4, "{plain_w} {width}");
        assert!(height > plain_h * 3 / 2, "{plain_h} {height}");
        let source = TextSource::open(
            TTF,
            None,
            &rich(r#"{"range":[1,2],"size":128}"#),
            &segment(),
        )
        .expect("open");
        let [first, second] = source.records() else {
            panic!("two letters");
        };
        assert!((second.height / first.height - 2.0).abs() < 0.05);
        // A range sized like the title changes nothing.
        assert_eq!(picture(&rich(r#"{"range":[0,2],"size":64}"#)).0, plain_w);
    }

    #[test]
    fn a_range_rings_and_shades_its_own_letters() {
        let red = r#"{"content":{"solid":{"color":[1,0,0]}},"width":0.06}"#;
        let (width, _, rgba) = picture(&rich(&format!(r#"{{"range":[0,1],"strokes":[{red}]}}"#)));
        assert_eq!(halves(width, &rgba, [255, 0, 0, 255]), (true, false));

        let blue =
            r#"{"content":{"solid":{"color":[0,0,1]}},"distance":20,"angle":-90,"feather":0}"#;
        let (width, height, rgba) =
            picture(&rich(&format!(r#"{{"range":[1,2],"shadows":[{blue}]}}"#)));
        assert_eq!(halves(width, &rgba, [0, 0, 255, 255]), (false, true));
        assert!(height > picture(&material("AA")).1);

        // An empty list takes the title's outline off its range.
        let mut ringed = rich(r#"{"range":[1,2],"strokes":[]}"#);
        ringed.border_color = "#00FF00".to_owned();
        ringed.border_width = 0.06;
        let (width, _, rgba) = picture(&ringed);
        assert_eq!(halves(width, &rgba, [0, 255, 0, 255]), (true, false));
    }

    #[test]
    fn a_glow_spreads_its_color_around_the_letters() {
        let text = material("A");
        let (plain_w, plain_h, _) = picture(&text);
        let mut bloom = BloomParams {
            color: "#FF00FF".to_owned(),
            strength: 1.0,
            range: 0.5,
            ..BloomParams::default()
        };
        let glowing = |bloom: &BloomParams| {
            let source = TextSource::open_with_glow(TTF, None, &text, Some(bloom), &segment())
                .expect("open");
            let (width, height) = source.size();
            (width, height, cpu(source.sample(Micros(1_000))))
        };
        let (width, height, rgba) = glowing(&bloom);
        assert!(width > plain_w && height > plain_h, "{width} {height}");
        // Magenta reaches past the letters, which stay white.
        let magenta = rgba
            .chunks(4)
            .filter(|pixel| pixel[1] == 0 && pixel[0] > 0 && pixel[0] == pixel[2])
            .count();
        assert!(magenta > 50, "{magenta}");
        assert!(has(&rgba, [255, 255, 255, 255]));
        // A full-strength glow is close to opaque next to the letters.
        let brightest = rgba
            .chunks(4)
            .filter(|pixel| pixel[1] == 0 && pixel[0] == pixel[2])
            .map(|pixel| pixel[3])
            .max();
        assert!(brightest >= Some(230), "{brightest:?}");
        // No strength is no glow.
        bloom.strength = 0.0;
        assert_eq!(glowing(&bloom).0, plain_w);
        bloom.strength = 1.0;
        bloom.color = "pink".to_owned();
        assert!(TextSource::open_with_glow(TTF, None, &text, Some(&bloom), &segment()).is_err());
    }

    #[test]
    fn a_background_fills_a_box_around_the_letters() {
        let mut boxed = material("A");
        boxed.background_style = 1;
        boxed.background_color = "#00FF00".to_owned();
        boxed.background_alpha = 0.5;
        boxed.background_width = 0.3;
        boxed.background_height = 0.3;
        let (width, height, rgba) = picture(&boxed);
        let (plain_w, plain_h, _) = picture(&material("A"));
        assert!(width > plain_w && height > plain_h);
        // Half green, premultiplied, near a corner; the letters stay white.
        let inside = ((width + 1) * 4) as usize;
        assert_eq!(&rgba[inside..inside + 4], &[0, 128, 0, 128]);
        assert!(has(&rgba, [255, 255, 255, 255]));
        // Round ends leave the corner clear.
        boxed.background_round_radius = 1.0;
        assert_eq!(&picture(&boxed).2[inside..inside + 4], &[0, 0, 0, 0]);
        // Style 0 is no background, whatever its color.
        boxed.background_style = 0;
        assert_eq!(picture(&boxed).0, plain_w);
    }

    #[test]
    fn a_bubble_shapes_the_background_and_grows_the_picture_for_its_tail() {
        let mut boxed = material("AB");
        boxed.background_style = 1;
        boxed.background_color = "#00FF00".to_owned();
        boxed.background_width = 0.3;
        boxed.background_height = 0.3;
        let (box_w, box_h, plain) = picture(&boxed);
        let green = |rgba: &[u8]| {
            rgba.chunks(4)
                .filter(|pixel| *pixel == [0, 255, 0, 255])
                .count()
        };
        for bubble in cutlass_model::Bubble::ALL {
            boxed.bubble = bubble.key().to_owned();
            let (width, height, rgba) = picture(&boxed);
            assert!(width >= box_w && height >= box_h, "{bubble:?}");
            assert!(green(&rgba) > 0, "{bubble:?}");
            assert!(has(&rgba, [255, 255, 255, 255]), "{bubble:?}");
            assert_ne!(rgba, plain, "{bubble:?}");
        }
        // A speech bubble's tail hangs below the box, on the left.
        boxed.bubble = "speech".to_owned();
        let (width, height, rgba) = picture(&boxed);
        assert!(height > box_h);
        let low = &rgba[((height * 7 / 8) * width * 4) as usize..];
        assert_eq!(halves(width, low, [0, 255, 0, 255]), (true, false));
        // An unknown bubble is a plain box.
        boxed.bubble = "cloud".to_owned();
        assert_eq!(picture(&boxed), (box_w, box_h, plain));
    }

    #[test]
    fn letters_are_counted_by_glyph_and_word_and_a_pose_hides_or_moves_one() {
        let text = material("AB AB");
        let mut source = TextSource::open(TTF, None, &text, &segment()).unwrap();
        let letters = source.letters().to_vec();
        let drawn: Vec<(usize, usize)> = letters
            .iter()
            .zip(source.records())
            .filter(|(_, record)| record.width > 0.0)
            .map(|(letter, _)| (letter.index, letter.word))
            .collect();
        assert_eq!(drawn, [(0, 0), (1, 0), (2, 1), (3, 1)]);
        assert!(
            letters
                .iter()
                .all(|letter| letter.count == 4 && letter.words == 2)
        );
        let rest = cpu(source.sample(Micros(1_000)));
        let (width, _) = source.size();
        let hidden = letters
            .iter()
            .map(|letter| LetterPose {
                alpha: if letter.word == 1 { 0.0 } else { 1.0 },
                ..LetterPose::REST
            })
            .collect();
        source.set_poses(hidden);
        let half = cpu(source.sample(Micros(1_000)));
        let (left, right) = halves(width, &half, [255, 255, 255, 255]);
        assert!(left && !right, "the second word is hidden");
        let raised = letters
            .iter()
            .map(|_| LetterPose {
                dy: -0.2,
                ..LetterPose::REST
            })
            .collect();
        source.set_poses(raised);
        let up = cpu(source.sample(Micros(1_000)));
        assert_ne!(up, rest);
        source.set_poses(Vec::new());
        assert_eq!(cpu(source.sample(Micros(1_000))), rest);
    }

    #[test]
    fn a_look_colors_boxes_and_sizes_a_word_and_the_picture_grows_evenly() {
        let mut text = material("AB AB");
        text.background_color = String::new();
        let mut source = TextSource::open(TTF, None, &text, &segment()).unwrap();
        let (width, height) = source.size();
        let rest = cpu(source.sample(Micros(1_000)));
        assert!(!has_color(&rest, [255, 0, 0, 255]));
        let looks: Vec<super::GlyphLook> = source
            .letters()
            .iter()
            .map(|letter| {
                if letter.word == 1 {
                    super::GlyphLook {
                        pose: LetterPose {
                            scale: 1.5,
                            ..LetterPose::REST
                        },
                        ink: Some([255, 0, 0, 255]),
                        back: [0, 0, 255, 255],
                    }
                } else {
                    super::GlyphLook::REST
                }
            })
            .collect();
        source.set_looks(&looks);
        let (grown_w, grown_h) = source.size();
        assert!(grown_w > width && grown_h > height, "a sized word fits");
        assert_eq!((grown_w - width) % 2, 0, "grown on both sides alike");
        assert_eq!((grown_h - height) % 2, 0);
        let drawn = cpu(source.sample(Micros(1_000)));
        assert!(has_color(&drawn, [255, 0, 0, 255]), "the word's own ink");
        let blue_left = drawn
            .chunks(4)
            .enumerate()
            .any(|(at, pixel)| pixel == [0, 0, 255, 255] && (at as u32 % grown_w) < grown_w / 4);
        assert!(has_color(&drawn, [0, 0, 255, 255]), "the word's box");
        assert!(!blue_left, "the box is behind the second word only");
        assert!(
            halves(grown_w, &drawn, [255, 255, 255, 255]).0,
            "the first word keeps the run's ink"
        );
    }

    fn has_color(rgba: &[u8], color: [u8; 4]) -> bool {
        rgba.chunks(4).any(|pixel| pixel == color)
    }

    #[test]
    fn a_hostile_font_or_color_is_an_error() {
        let text = material("A");
        let clip = segment();
        assert!(matches!(
            TextSource::open(&[], None, &text, &clip),
            Err(TextError::Font(FontError::Truncated))
        ));
        assert!(matches!(
            TextSource::open(&[0, 1, 0, 0], None, &text, &clip),
            Err(TextError::Font(FontError::Truncated))
        ));
        assert!(matches!(
            TextSource::open(TTC, None, &text, &clip),
            Err(TextError::Font(FontError::Collection))
        ));
        assert!(matches!(
            TextSource::open(TTC, Some(9), &text, &clip),
            Err(TextError::Font(FontError::CollectionIndex { .. }))
        ));
        let face = TextSource::open(TTC, Some(1), &text, &clip).expect("face");
        assert!(matches!(face.sample(Micros(1_000)), Sample::Ready(_)));
        assert!(matches!(
            TextSource::open(NO_OUTLINE, None, &text, &clip),
            Err(TextError::MissingStrike)
        ));
        let mut bad = material("A");
        bad.text_color = "#gg0000".to_owned();
        assert!(matches!(
            TextSource::open(TTF, None, &bad, &clip),
            Err(TextError::Color)
        ));
        let mut ringed = material("A");
        ringed.border_color = "red".to_owned();
        ringed.border_width = 0.1;
        assert!(matches!(
            TextSource::open(TTF, None, &ringed, &clip),
            Err(TextError::Color)
        ));
        let mut huge = material("A");
        huge.font_size = 100_000.0;
        assert!(matches!(
            TextSource::open(TTF, None, &huge, &clip),
            Err(TextError::Limit)
        ));
    }
}
