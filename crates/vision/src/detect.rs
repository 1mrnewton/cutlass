//! Finding a face, a body or a hand in a frame, for a track to start from.
//!
//! Three small detectors from OpenCV Zoo, all under permissive licences:
//! YuNet for faces (MIT), and MediaPipe's person and palm detectors
//! (Apache-2.0). Each reads a frame and gives boxes in the frame's own
//! pixels, best first, with what overlaps a better box taken out.
//!
//! A body is the person detector's own reach: it finds the middle of the
//! hips and a point past the head, and the body is the box from that point
//! to as far below the hips. A palm is small in a whole frame, so a hand is
//! also looked for in overlapping squares, and the box is grown past the
//! palm to take in the fingers.

use std::path::Path;

use burn::tensor::{Device, Tensor, TensorData};

use crate::device::{self, Compute};
use crate::generated::face_detection_yunet_2023mar as yunet;
use crate::generated::palm_detection_mediapipe_2023feb as palm;
use crate::generated::person_detection_mediapipe_2023mar as person;
use crate::mobile_sam::load_weights;
use crate::{Image, VisionError, frame::MAX_SIDE};

/// YuNet's weights, written by `tools/vision-models`.
pub const FACE_WEIGHTS: &str = "face_detection_yunet_2023mar.bpk";
/// MediaPipe's person detector's weights.
pub const BODY_WEIGHTS: &str = "person_detection_mediapipe_2023mar.bpk";
/// MediaPipe's palm detector's weights.
pub const HAND_WEIGHTS: &str = "palm_detection_mediapipe_2023feb.bpk";

/// YuNet reads a 640 square; the frame sits in its top left.
const FACE_SIDE: usize = 640;
const FACE_STRIDES: [usize; 3] = [8, 16, 32];
const FACE_SCORE: f32 = 0.6;
/// The person detector reads a 224 square, the frame in its middle.
const BODY_SIDE: usize = 224;
const BODY_SCORE: f32 = 0.5;
/// The palm detector reads a 192 square, the frame in its middle.
const HAND_SIDE: usize = 192;
/// Above MediaPipe's 0.5: in squares cut from a frame, a chin or a cheek
/// can pass for a palm at about 0.6, and a palm half turned away scores
/// about 0.69.
const HAND_SCORE: f32 = 0.65;
/// How much wider and taller a hand's box is than its palm's.
const HAND_GROWTH: f32 = 1.4;
/// A body's width over its height: about a person standing, arms down.
const BODY_WIDTH: f32 = 0.375;
/// Boxes overlapping a better one by more than this are dropped.
const OVERLAP: f32 = 0.3;

/// What to look for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    Face,
    Body,
    Hand,
}

impl Part {
    /// The file its detector's weights are in.
    #[must_use]
    pub const fn weights(self) -> &'static str {
        match self {
            Self::Face => FACE_WEIGHTS,
            Self::Body => BODY_WEIGHTS,
            Self::Hand => HAND_WEIGHTS,
        }
    }
}

/// A box in the frame's own pixels, and how sure the detector is, 0 to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Found {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
}

impl Found {
    fn area(&self) -> f32 {
        (self.x1 - self.x0).max(0.0) * (self.y1 - self.y0).max(0.0)
    }

    fn overlap(&self, other: &Self) -> f32 {
        let w = (self.x1.min(other.x1) - self.x0.max(other.x0)).max(0.0);
        let h = (self.y1.min(other.y1) - self.y0.max(other.y0)).max(0.0);
        let shared = w * h;
        let union = self.area() + other.area() - shared;
        if union > 0.0 { shared / union } else { 0.0 }
    }

    /// Kept inside a `width` by `height` frame.
    fn clamped(self, width: f32, height: f32) -> Self {
        Self {
            x0: self.x0.clamp(0.0, width),
            y0: self.y0.clamp(0.0, height),
            x1: self.x1.clamp(0.0, width),
            y1: self.y1.clamp(0.0, height),
            score: self.score,
        }
    }
}

/// One of the three networks.
enum Net {
    Face(Box<yunet::Model>),
    Body(Box<person::Model>),
    Hand(Box<palm::Model>),
}

/// A detector for one [`Part`], loaded once and run on many frames.
pub struct Detector {
    device: Device,
    net: Net,
}

/// A person the person detector found: its box (about the head), its four
/// points (the hips' middle, a point past the head, the shoulders' middle,
/// a point past the upper body), in frame pixels, and its score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Person {
    pub found: Found,
    pub points: [(f32, f32); 4],
}

impl Detector {
    /// Loads `part`'s weights from `dir`.
    pub fn load(dir: &Path, part: Part, compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let path = dir.join(part.weights());
        let net = match part {
            Part::Face => {
                let mut model = yunet::Model::new(&device);
                load_weights(&mut model, &path)?;
                Net::Face(Box::new(model))
            }
            Part::Body => {
                let mut model = person::Model::new(&device);
                load_weights(&mut model, &path)?;
                Net::Body(Box::new(model))
            }
            Part::Hand => {
                let mut model = palm::Model::new(&device);
                load_weights(&mut model, &path)?;
                Net::Hand(Box::new(model))
            }
        };
        Ok(Self { device, net })
    }

    /// What it looks for.
    #[must_use]
    pub fn part(&self) -> Part {
        match self.net {
            Net::Face(_) => Part::Face,
            Net::Body(_) => Part::Body,
            Net::Hand(_) => Part::Hand,
        }
    }

    /// Every face, body or hand in `image`, best first, in its pixels.
    pub fn find(&self, image: &Image<'_>) -> Result<Vec<Found>, VisionError> {
        check(image)?;
        #[allow(clippy::cast_precision_loss)]
        let (w, h) = (image.width as f32, image.height as f32);
        let found = match &self.net {
            Net::Face(model) => self.faces(model, image)?,
            Net::Body(model) => self
                .people(model, image)?
                .into_iter()
                .map(|person| body(&person))
                .collect(),
            Net::Hand(model) => self.hands(model, image)?,
        };
        Ok(found.into_iter().map(|f| f.clamped(w, h)).collect())
    }

    fn faces(&self, model: &yunet::Model, image: &Image<'_>) -> Result<Vec<Found>, VisionError> {
        let (input, scale) = face_input(image);
        let input = Tensor::<4>::from_data(
            TensorData::new(input, [1, 3, FACE_SIDE, FACE_SIDE]),
            &self.device,
        );
        let out = model.forward(input);
        let heads = [
            (out.0, out.3, out.6),
            (out.1, out.4, out.7),
            (out.2, out.5, out.8),
        ];
        let mut read = Vec::with_capacity(3);
        for (cls, obj, bbox) in heads {
            read.push((
                values(cls.into_data())?,
                values(obj.into_data())?,
                values(bbox.into_data())?,
            ));
        }
        Ok(decode_faces(&read, scale))
    }

    pub(crate) fn people(
        &self,
        model: &person::Model,
        image: &Image<'_>,
    ) -> Result<Vec<Person>, VisionError> {
        let fit = Letterbox::new(image, BODY_SIDE);
        let input = fit.pixels(image, |v| (v / 255.0 - 0.5) * 2.0);
        let mut chw = vec![0.0; input.len()];
        let plane = BODY_SIDE * BODY_SIDE;
        for (i, rgb) in input.as_chunks::<3>().0.iter().enumerate() {
            for (c, v) in rgb.iter().enumerate() {
                chw[c * plane + i] = *v;
            }
        }
        let input = Tensor::<4>::from_data(
            TensorData::new(chw, [1, 3, BODY_SIDE, BODY_SIDE]),
            &self.device,
        );
        let (scores, boxes) = model.forward(input);
        let scores = values(scores.into_data())?;
        let boxes = values(boxes.into_data())?;
        Ok(decode_people(&scores, &boxes, &fit))
    }

    fn hands(&self, model: &palm::Model, image: &Image<'_>) -> Result<Vec<Found>, VisionError> {
        let mut found = self.palms(model, image, 0, 0)?;
        let (w, h) = (image.width as usize, image.height as usize);
        let side = w.min(h) / 2;
        if side >= HAND_SIDE {
            let step = side / 2;
            let mut tile = vec![0; side * side * 3];
            for y in (0..=h - side).step_by(step) {
                for x in (0..=w - side).step_by(step) {
                    for row in 0..side {
                        let from = ((y + row) * w + x) * 3;
                        tile[row * side * 3..(row + 1) * side * 3]
                            .copy_from_slice(&image.rgb[from..from + side * 3]);
                    }
                    #[allow(clippy::cast_possible_truncation)]
                    let part = Image {
                        width: side as u32,
                        height: side as u32,
                        rgb: &tile,
                    };
                    found.extend(self.palms(model, &part, x, y)?);
                }
            }
        }
        Ok(suppress(found)
            .into_iter()
            .map(|palm| grown(palm, HAND_GROWTH))
            .collect())
    }

    /// Palms in `image`, moved by `x`, `y` into the frame it was cut from.
    fn palms(
        &self,
        model: &palm::Model,
        image: &Image<'_>,
        x: usize,
        y: usize,
    ) -> Result<Vec<Found>, VisionError> {
        let fit = Letterbox::new(image, HAND_SIDE);
        let input = fit.pixels(image, |v| v / 255.0);
        let input = Tensor::<4>::from_data(
            TensorData::new(input, [1, HAND_SIDE, HAND_SIDE, 3]),
            &self.device,
        );
        let (boxes, scores) = model.forward(input);
        let scores = values(scores.into_data())?;
        let boxes = values(boxes.into_data())?;
        #[allow(clippy::cast_precision_loss)]
        let (dx, dy) = (x as f32, y as f32);
        Ok(decode_ssd(
            &scores,
            &boxes,
            18,
            &fit,
            &ssd_anchors(HAND_SIDE, &[(8, 2), (16, 6)]),
            HAND_SCORE,
        )
        .into_iter()
        .map(|(f, _)| Found {
            x0: f.x0 + dx,
            y0: f.y0 + dy,
            x1: f.x1 + dx,
            y1: f.y1 + dy,
            score: f.score,
        })
        .collect())
    }
}

fn check(image: &Image<'_>) -> Result<(), VisionError> {
    let refused = || VisionError::Image {
        width: image.width,
        height: image.height,
        bytes: image.rgb.len(),
    };
    if image.width == 0 || image.height == 0 || image.width > MAX_SIDE || image.height > MAX_SIDE {
        return Err(refused());
    }
    if (image.width as usize) * (image.height as usize) * 3 != image.rgb.len() {
        return Err(refused());
    }
    Ok(())
}

fn values(data: TensorData) -> Result<Vec<f32>, VisionError> {
    data.try_to_vec::<f32>()
        .map_err(|error| VisionError::Model {
            message: format!("{error:?}"),
        })
}

/// `image` resized to `width` by `height` as OpenCV's linear resize does it:
/// each pixel samples the four source pixels around its middle, with no
/// smoothing first, which is how these detectors' frames were made.
fn resized(image: &Image<'_>, width: u32, height: u32) -> Vec<u8> {
    if (width, height) == (image.width, image.height) {
        return image.rgb.to_vec();
    }
    let (sw, sh) = (image.width as usize, image.height as usize);
    let (w, h) = (width as usize, height as usize);
    #[allow(clippy::cast_precision_loss)]
    let (fx, fy) = (sw as f32 / w as f32, sh as f32 / h as f32);
    // Where each output column or row samples: a pixel and the share of
    // the next one.
    let taps = |n: usize, f: f32, limit: usize| -> Vec<(usize, usize, f32)> {
        (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let at = ((i as f32 + 0.5) * f - 0.5).max(0.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let low = (at.floor() as usize).min(limit - 1);
                let high = (low + 1).min(limit - 1);
                #[allow(clippy::cast_precision_loss)]
                let share = if high == low { 0.0 } else { at - low as f32 };
                (low, high, share)
            })
            .collect()
    };
    let columns = taps(w, fx, sw);
    let rows = taps(h, fy, sh);
    let mut out = vec![0; w * h * 3];
    for (y, &(y0, y1, b)) in rows.iter().enumerate() {
        for (x, &(x0, x1, a)) in columns.iter().enumerate() {
            for c in 0..3 {
                let p = |xx: usize, yy: usize| f32::from(image.rgb[(yy * sw + xx) * 3 + c]);
                let top = p(x0, y0) * (1.0 - a) + p(x1, y0) * a;
                let bottom = p(x0, y1) * (1.0 - a) + p(x1, y1) * a;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let v = (top * (1.0 - b) + bottom * b).round().clamp(0.0, 255.0) as u8;
                out[(y * w + x) * 3 + c] = v;
            }
        }
    }
    out
}

/// YuNet's input: the frame with its long side 640 in the top left of a
/// black 640 square, as blue, green, red planes of 0 to 255. Also the
/// scale from frame to input.
fn face_input(image: &Image<'_>) -> (Vec<f32>, f32) {
    #[allow(clippy::cast_precision_loss)]
    let scale = FACE_SIDE as f32 / image.width.max(image.height) as f32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let fit = |s: u32| ((s as f32 * scale).round() as u32).clamp(1, FACE_SIDE as u32);
    let (fw, fh) = (fit(image.width), fit(image.height));
    let small = resized(image, fw, fh);
    let plane = FACE_SIDE * FACE_SIDE;
    let mut input = vec![0.0; 3 * plane];
    for (i, rgb) in small.as_chunks::<3>().0.iter().enumerate() {
        let (x, y) = (i % fw as usize, i / fw as usize);
        let at = y * FACE_SIDE + x;
        input[at] = f32::from(rgb[2]);
        input[plane + at] = f32::from(rgb[1]);
        input[2 * plane + at] = f32::from(rgb[0]);
    }
    (input, scale)
}

/// YuNet's heads, one per stride, as class, object and box values, to
/// faces in frame pixels: each cell's score is the root of its class and
/// object scores, and its box is offset from the cell and sized by an
/// exponent, as OpenCV's `FaceDetectorYN` reads them.
fn decode_faces(heads: &[(Vec<f32>, Vec<f32>, Vec<f32>)], scale: f32) -> Vec<Found> {
    let mut found = Vec::new();
    for (&stride, (cls, obj, bbox)) in FACE_STRIDES.iter().zip(heads) {
        let cols = FACE_SIDE / stride;
        #[allow(clippy::cast_precision_loss)]
        let s = stride as f32;
        for (i, (c, o)) in cls.iter().zip(obj).enumerate() {
            let score = (c.clamp(0.0, 1.0) * o.clamp(0.0, 1.0)).sqrt();
            if score < FACE_SCORE {
                continue;
            }
            let Some(b) = bbox.get(i * 4..i * 4 + 4) else {
                break;
            };
            #[allow(clippy::cast_precision_loss)]
            let (col, row) = ((i % cols) as f32, (i / cols) as f32);
            let (cx, cy) = ((col + b[0]) * s, (row + b[1]) * s);
            let (w, h) = (b[2].exp() * s, b[3].exp() * s);
            found.push(Found {
                x0: (cx - w / 2.0) / scale,
                y0: (cy - h / 2.0) / scale,
                x1: (cx + w / 2.0) / scale,
                y1: (cy + h / 2.0) / scale,
                score,
            });
        }
    }
    suppress(found)
}

/// A frame scaled into the middle of a black square, as MediaPipe's
/// detectors read it.
pub(crate) struct Letterbox {
    side: usize,
    width: usize,
    height: usize,
    left: usize,
    top: usize,
    /// Square pixels per frame pixel.
    ratio: f32,
    /// The frame's long side: the square's side in frame pixels.
    long: f32,
}

impl Letterbox {
    pub(crate) fn new(image: &Image<'_>, side: usize) -> Self {
        // In doubles and cut short, as the reference sizes it: 720 by
        // 224 / 1280 is 125.99... there, and 125 rows.
        #[allow(clippy::cast_precision_loss)]
        let ratio =
            (side as f64 / f64::from(image.height)).min(side as f64 / f64::from(image.width));
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let fit = |s: u32| ((f64::from(s) * ratio) as usize).clamp(1, side);
        let (width, height) = (fit(image.width), fit(image.height));
        #[allow(clippy::cast_precision_loss)]
        let long = image.width.max(image.height) as f32;
        Self {
            side,
            width,
            height,
            left: (side - width) / 2,
            top: (side - height) / 2,
            #[allow(clippy::cast_possible_truncation)]
            ratio: ratio as f32,
            long,
        }
    }

    /// The square as RGB rows, each value mapped by `value`.
    fn pixels(&self, image: &Image<'_>, value: impl Fn(f32) -> f32) -> Vec<f32> {
        #[allow(clippy::cast_possible_truncation)]
        let small = resized(image, self.width as u32, self.height as u32);
        let black = value(0.0);
        let mut out = vec![black; self.side * self.side * 3];
        for (i, rgb) in small.as_chunks::<3>().0.iter().enumerate() {
            let (x, y) = (i % self.width + self.left, i / self.width + self.top);
            let at = (y * self.side + x) * 3;
            for (c, v) in rgb.iter().enumerate() {
                out[at + c] = value(f32::from(*v));
            }
        }
        out
    }

    /// The frame pixels the square's padding stands for, whole, as the
    /// reference counts them.
    fn bias(&self) -> (f32, f32) {
        #[allow(clippy::cast_precision_loss)]
        let to = |p: usize| (p as f32 / self.ratio).trunc();
        (to(self.left), to(self.top))
    }
}

/// The anchors of MediaPipe's SSD detectors: for each stride, one place per
/// cell of a grid over the square, `n` times over.
fn ssd_anchors(side: usize, layers: &[(usize, usize)]) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    for &(stride, n) in layers {
        let cells = side.div_ceil(stride);
        #[allow(clippy::cast_precision_loss)]
        let at = |i: usize| (i as f32 + 0.5) / cells as f32;
        for y in 0..cells {
            for x in 0..cells {
                for _ in 0..n {
                    out.push((at(x), at(y)));
                }
            }
        }
    }
    out
}

/// An SSD detector's outputs to boxes and points in frame pixels, best
/// first, with overlaps taken out. `width` is the values per anchor: four
/// for the box, then two per point.
fn decode_ssd(
    scores: &[f32],
    boxes: &[f32],
    width: usize,
    fit: &Letterbox,
    anchors: &[(f32, f32)],
    threshold: f32,
) -> Vec<(Found, Vec<(f32, f32)>)> {
    #[allow(clippy::cast_precision_loss)]
    let side = fit.side as f32;
    let (bx, by) = fit.bias();
    let mut found = Vec::new();
    for (i, (&raw, &(ax, ay))) in scores.iter().zip(anchors).enumerate() {
        let score = 1.0 / (1.0 + (-raw.clamp(-100.0, 100.0)).exp());
        if score < threshold {
            continue;
        }
        let Some(d) = boxes.get(i * width..(i + 1) * width) else {
            break;
        };
        let (cx, cy) = (d[0] / side + ax, d[1] / side + ay);
        let (w, h) = (d[2] / side, d[3] / side);
        let points = d[4..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| {
                (
                    (p[0] / side + ax) * fit.long - bx,
                    (p[1] / side + ay) * fit.long - by,
                )
            })
            .collect();
        found.push((
            Found {
                x0: (cx - w / 2.0) * fit.long - bx,
                y0: (cy - h / 2.0) * fit.long - by,
                x1: (cx + w / 2.0) * fit.long - bx,
                y1: (cy + h / 2.0) * fit.long - by,
                score,
            },
            points,
        ));
    }
    found.sort_by(|a, b| b.0.score.total_cmp(&a.0.score));
    let mut kept: Vec<(Found, Vec<(f32, f32)>)> = Vec::new();
    for candidate in found {
        if kept.iter().all(|k| k.0.overlap(&candidate.0) <= OVERLAP) {
            kept.push(candidate);
        }
    }
    kept
}

fn decode_people(scores: &[f32], boxes: &[f32], fit: &Letterbox) -> Vec<Person> {
    let anchors = ssd_anchors(BODY_SIDE, &[(8, 2), (16, 2), (32, 6)]);
    decode_ssd(scores, boxes, 12, fit, &anchors, BODY_SCORE)
        .into_iter()
        .filter_map(|(found, points)| {
            Some(Person {
                found,
                points: [
                    *points.first()?,
                    *points.get(1)?,
                    *points.get(2)?,
                    *points.get(3)?,
                ],
            })
        })
        .collect()
}

/// A person's whole body: from the point past the head to as far below the
/// hips, about as wide as a person standing.
fn body(person: &Person) -> Found {
    let [(hx, hy), (tx, ty), ..] = person.points;
    let reach = ((tx - hx).powi(2) + (ty - hy).powi(2)).sqrt();
    let half = reach * BODY_WIDTH;
    Found {
        x0: hx - half,
        y0: hy - reach,
        x1: hx + half,
        y1: hy + reach,
        score: person.found.score,
    }
}

/// `found` grown by `factor` about its middle.
fn grown(found: Found, factor: f32) -> Found {
    let (cx, cy) = ((found.x0 + found.x1) / 2.0, (found.y0 + found.y1) / 2.0);
    let (hw, hh) = (
        (found.x1 - found.x0) * factor / 2.0,
        (found.y1 - found.y0) * factor / 2.0,
    );
    Found {
        x0: cx - hw,
        y0: cy - hh,
        x1: cx + hw,
        y1: cy + hh,
        score: found.score,
    }
}

/// Best first, without a box that overlaps a better one by more than
/// [`OVERLAP`], as OpenCV's `NMSBoxes` keeps them.
fn suppress(mut found: Vec<Found>) -> Vec<Found> {
    found.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Found> = Vec::new();
    for candidate in found {
        if kept.iter().all(|k| k.overlap(&candidate) <= OVERLAP) {
            kept.push(candidate);
        }
    }
    kept
}

#[cfg(test)]
mod tests;
