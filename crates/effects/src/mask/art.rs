//! What a drawn mask is drawn from: a pen outline, brush strokes, or the
//! outlines of a line of text. It is kept in the mask record's `art`
//! field, beside the mask's box, and is measured in that box: -1 to 1
//! each way across it, y up. Moving, turning or sizing the box moves the
//! drawing with it.
//!
//! Text is kept as its letters' outlines, taken from the font when the
//! mask is made, so the mask draws the same with or without the font.

use cutlass_model::ShapePoint;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::MaskShape;

/// The most points a drawing holds, over all its lines.
pub const MAX_ART_POINTS: usize = 20_000;
/// The most brush strokes a mask holds.
pub const MAX_STROKES: usize = 512;
/// The most characters a text mask spells.
pub const MAX_MASK_TEXT: usize = 200;
/// How far past its box a drawing may reach.
const ART_REACH: f64 = 4.0;

/// One brush stroke: the line the brush took, how wide it is (a part of
/// the box, which is 2 across), and whether it rubs out what strokes
/// before it painted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrushStroke {
    pub points: Vec<[f32; 2]>,
    pub size: f32,
    #[serde(default)]
    pub erase: bool,
}

/// A drawn mask's drawing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MaskArt {
    /// A closed outline through `points`, each joined to the next by a
    /// cubic through their handles.
    Pen { points: Vec<ShapePoint> },
    /// Strokes painted, or rubbed out, in order.
    Brush { strokes: Vec<BrushStroke> },
    /// A line of text: what it spells, the font it was taken from, its
    /// width over its height, and its letters' outlines, each a closed
    /// polygon, filled where they wind round.
    Text {
        text: String,
        #[serde(default)]
        font: String,
        aspect: f32,
        outlines: Vec<Vec<[f32; 2]>>,
    },
}

impl MaskArt {
    /// The drawn shape this art belongs to.
    #[must_use]
    pub fn shape(&self) -> MaskShape {
        match self {
            Self::Pen { .. } => MaskShape::Pen,
            Self::Brush { .. } => MaskShape::Brush,
            Self::Text { .. } => MaskShape::Text,
        }
    }

    /// Every number finite and near its box, and within the limits.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let near = |value: f64| value.is_finite() && value.abs() <= ART_REACH;
        let pair = |[x, y]: [f32; 2]| near(f64::from(x)) && near(f64::from(y));
        match self {
            Self::Pen { points } => {
                points.len() >= 3
                    && points.len() <= MAX_ART_POINTS
                    && points.iter().all(|point| {
                        point.at.iter().all(|value| near(*value))
                            && point
                                .handle_in
                                .iter()
                                .chain(&point.handle_out)
                                .all(|value| value.is_finite() && value.abs() <= 2.0 * ART_REACH)
                            && point.round.is_finite()
                            && point.round >= 0.0
                    })
            }
            Self::Brush { strokes } => {
                strokes.len() <= MAX_STROKES
                    && strokes
                        .iter()
                        .map(|stroke| stroke.points.len())
                        .sum::<usize>()
                        <= MAX_ART_POINTS
                    && strokes.iter().all(|stroke| {
                        !stroke.points.is_empty()
                            && stroke.size.is_finite()
                            && stroke.size > 0.0
                            && stroke.size <= 2.0
                            && stroke.points.iter().all(|point| pair(*point))
                    })
            }
            Self::Text {
                text,
                aspect,
                outlines,
                ..
            } => {
                text.chars().count() <= MAX_MASK_TEXT
                    && aspect.is_finite()
                    && *aspect > 0.0
                    && outlines.iter().map(Vec::len).sum::<usize>() <= MAX_ART_POINTS
                    && outlines
                        .iter()
                        .all(|outline| outline.iter().all(|point| pair(*point)))
            }
        }
    }

    /// The art in a record's fields, when there is some and it reads.
    #[must_use]
    pub fn from_fields(fields: &Map<String, Value>) -> Option<Self> {
        let art = Self::deserialize(fields.get("art")?).ok()?;
        art.is_valid().then_some(art)
    }

    /// Put the art into a record's fields.
    pub fn write(&self, fields: &mut Map<String, Value>) {
        if let Ok(value) = serde_json::to_value(self) {
            fields.insert("art".to_owned(), value);
        }
    }
}

/// Whether `art` may go with a mask of `shape`: a drawn shape takes its
/// own kind of art or none yet, and any other shape takes none.
#[must_use]
pub fn art_fits(shape: MaskShape, art: Option<&MaskArt>) -> bool {
    match art {
        None => true,
        Some(art) => art.shape() == shape && art.is_valid(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> MaskArt {
        let corner = |x: f64, y: f64| ShapePoint {
            at: [x, y],
            ..ShapePoint::default()
        };
        MaskArt::Pen {
            points: vec![
                corner(-0.5, -0.5),
                corner(0.5, -0.5),
                corner(0.5, 0.5),
                corner(-0.5, 0.5),
            ],
        }
    }

    #[test]
    fn art_goes_into_a_record_and_back() {
        let brush = MaskArt::Brush {
            strokes: vec![BrushStroke {
                points: vec![[0.0, 0.0], [0.5, 0.25]],
                size: 0.2,
                erase: false,
            }],
        };
        for art in [square(), brush] {
            let mut fields = Map::new();
            art.write(&mut fields);
            assert_eq!(MaskArt::from_fields(&fields), Some(art));
        }
    }

    #[test]
    fn art_out_of_bounds_or_of_another_shape_is_refused() {
        let MaskArt::Pen { mut points } = square() else {
            unreachable!()
        };
        points[0].at[0] = f64::NAN;
        assert!(!MaskArt::Pen { points }.is_valid());
        let wide = MaskArt::Brush {
            strokes: vec![BrushStroke {
                points: vec![[0.0, 0.0]],
                size: 3.0,
                erase: false,
            }],
        };
        assert!(!wide.is_valid());
        assert!(art_fits(MaskShape::Pen, Some(&square())));
        assert!(!art_fits(MaskShape::Brush, Some(&square())));
        assert!(!art_fits(MaskShape::Circle, Some(&square())));
        assert!(art_fits(MaskShape::Text, None));
        let mut fields = Map::new();
        fields.insert(
            "art".to_owned(),
            serde_json::json!({"kind": "pen", "points": []}),
        );
        assert_eq!(MaskArt::from_fields(&fields), None);
    }
}
