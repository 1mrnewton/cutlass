//! Content descriptors: text, solids, and shapes.

use cutlass_models::{
    Generator, Param, Shape, ShapeStroke as ModelShapeStroke, TextAlignH, TextAlignV,
    TextBackground as ModelTextBackground, TextCase, TextShadow as ModelTextShadow,
    TextStroke as ModelTextStroke, TextStyle,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::{parse_color, parse_color_opt};

/// A title / caption layer descriptor.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Text {
    pub(crate) content: String,
    pub(crate) style: TextStyle,
}

#[pymethods]
impl Text {
    #[new]
    #[pyo3(signature = (
        content,
        font = "",
        size = 90.0,
        color = None,
        bold = false,
        italic = false,
        underline = false,
        case = "normal",
        letter_spacing = 0.0,
        line_spacing = 1.2,
        align = ("center".to_string(), "middle".to_string()),
        wrap = true,
        stroke = None,
        background = None,
        shadow = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        content: &str,
        font: &str,
        size: f32,
        color: Option<&Bound<'_, PyAny>>,
        bold: bool,
        italic: bool,
        underline: bool,
        case: &str,
        letter_spacing: f32,
        line_spacing: f32,
        align: (String, String),
        wrap: bool,
        stroke: Option<TextStroke>,
        background: Option<TextBackground>,
        shadow: Option<TextShadow>,
    ) -> PyResult<Self> {
        Ok(Self {
            content: content.to_string(),
            style: TextStyle {
                font: font.to_string(),
                size,
                bold,
                italic,
                underline,
                case: parse_text_case(case)?,
                fill: parse_color_opt(color)?,
                letter_spacing,
                line_spacing,
                align_h: parse_align_h(&align.0)?,
                align_v: parse_align_v(&align.1)?,
                wrap,
                stroke: stroke.map(|s| s.inner),
                background: background.map(|b| b.inner),
                shadow: shadow.map(|s| s.inner),
                // Catalog presets are not exposed here: treatments are
                // always explicit stroke/background/shadow arguments.
                effect_preset: None,
            },
        })
    }

    fn __repr__(&self) -> String {
        format!("Text({:?}, size={})", self.content, self.style.size)
    }
}

fn parse_text_case(s: &str) -> PyResult<TextCase> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "normal" => TextCase::Normal,
        "upper" => TextCase::Upper,
        "lower" => TextCase::Lower,
        "title" => TextCase::Title,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown text case {other:?}"
            )));
        }
    })
}

fn parse_align_h(s: &str) -> PyResult<TextAlignH> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "left" => TextAlignH::Left,
        "center" => TextAlignH::Center,
        "right" => TextAlignH::Right,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown horizontal align {other:?}"
            )));
        }
    })
}

fn parse_align_v(s: &str) -> PyResult<TextAlignV> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "top" => TextAlignV::Top,
        "middle" => TextAlignV::Middle,
        "bottom" => TextAlignV::Bottom,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown vertical align {other:?}"
            )));
        }
    })
}

/// Outline stroke for a [`Text`] descriptor.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct TextStroke {
    pub(crate) inner: ModelTextStroke,
}

#[pymethods]
impl TextStroke {
    #[new]
    #[pyo3(signature = (color, width = 6.0))]
    fn new(color: &Bound<'_, PyAny>, width: f32) -> PyResult<Self> {
        Ok(Self {
            inner: ModelTextStroke {
                rgba: parse_color(color)?,
                width,
            },
        })
    }
}

/// Filled card behind a [`Text`] block.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct TextBackground {
    pub(crate) inner: ModelTextBackground,
}

#[pymethods]
impl TextBackground {
    #[new]
    #[pyo3(signature = (color, radius = 0.0))]
    fn new(color: &Bound<'_, PyAny>, radius: f32) -> PyResult<Self> {
        Ok(Self {
            inner: ModelTextBackground {
                rgba: parse_color(color)?,
                radius,
            },
        })
    }
}

/// Drop shadow behind a [`Text`] block.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct TextShadow {
    pub(crate) inner: ModelTextShadow,
}

#[pymethods]
impl TextShadow {
    #[new]
    #[pyo3(signature = (color, blur = 0.15, distance = 5.0))]
    fn new(color: &Bound<'_, PyAny>, blur: f32, distance: f32) -> PyResult<Self> {
        Ok(Self {
            inner: ModelTextShadow {
                rgba: parse_color(color)?,
                blur,
                distance,
            },
        })
    }
}

/// A solid fill descriptor.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Solid {
    pub(crate) rgba: [u8; 4],
}

#[pymethods]
impl Solid {
    #[new]
    fn new(color: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            rgba: parse_color(color)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Solid(r={}, g={}, b={}, a={})",
            self.rgba[0], self.rgba[1], self.rgba[2], self.rgba[3]
        )
    }
}

/// Outline for a shape (Python name: `ShapeStroke`).
#[pyclass(name = "ShapeStroke", from_py_object)]
#[derive(Clone)]
pub struct ShapeStrokeSpec {
    pub(crate) rgba: [u8; 4],
    pub(crate) width: f32,
}

#[pymethods]
impl ShapeStrokeSpec {
    #[new]
    #[pyo3(signature = (color, width = 8.0))]
    fn new(color: &Bound<'_, PyAny>, width: f32) -> PyResult<Self> {
        Ok(Self {
            rgba: parse_color(color)?,
            width,
        })
    }
}

fn stroke_from(spec: Option<ShapeStrokeSpec>) -> Option<ModelShapeStroke> {
    spec.map(|s| ModelShapeStroke::new(s.rgba, s.width))
}

macro_rules! shape_class {
    ($name:ident) => {
        #[pyclass(skip_from_py_object)]
        #[derive(Clone)]
        pub struct $name {
            pub(crate) width: f32,
            pub(crate) height: f32,
            pub(crate) rgba: [u8; 4],
            pub(crate) corner_radius: f32,
            pub(crate) stroke: Option<ShapeStrokeSpec>,
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (width = 200.0, height = 200.0, color = None, corner_radius = 0.0, stroke = None))]
            fn new(
                width: f32,
                height: f32,
                color: Option<&Bound<'_, PyAny>>,
                corner_radius: f32,
                stroke: Option<ShapeStrokeSpec>,
            ) -> PyResult<Self> {
                Ok(Self {
                    width,
                    height,
                    rgba: parse_color_opt(color)?,
                    corner_radius,
                    stroke,
                })
            }
        }
    };
}

shape_class!(Rect);
shape_class!(Ellipse);
shape_class!(Arrow);
shape_class!(Heart);

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Polygon {
    pub(crate) sides: u32,
    pub(crate) width: f32,
    pub(crate) height: f32,
    pub(crate) rgba: [u8; 4],
    pub(crate) corner_radius: f32,
    pub(crate) stroke: Option<ShapeStrokeSpec>,
}

#[pymethods]
impl Polygon {
    #[new]
    #[pyo3(signature = (sides, width = 200.0, height = 200.0, color = None, corner_radius = 0.0, stroke = None))]
    fn new(
        sides: u32,
        width: f32,
        height: f32,
        color: Option<&Bound<'_, PyAny>>,
        corner_radius: f32,
        stroke: Option<ShapeStrokeSpec>,
    ) -> PyResult<Self> {
        Ok(Self {
            sides,
            width,
            height,
            rgba: parse_color_opt(color)?,
            corner_radius,
            stroke,
        })
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Star {
    pub(crate) points: u32,
    pub(crate) inner_ratio: f32,
    pub(crate) width: f32,
    pub(crate) height: f32,
    pub(crate) rgba: [u8; 4],
    pub(crate) stroke: Option<ShapeStrokeSpec>,
}

#[pymethods]
impl Star {
    #[new]
    #[pyo3(signature = (points, inner_ratio = 0.5, width = 200.0, height = 200.0, color = None, stroke = None))]
    fn new(
        points: u32,
        inner_ratio: f32,
        width: f32,
        height: f32,
        color: Option<&Bound<'_, PyAny>>,
        stroke: Option<ShapeStrokeSpec>,
    ) -> PyResult<Self> {
        Ok(Self {
            points,
            inner_ratio,
            width,
            height,
            rgba: parse_color_opt(color)?,
            stroke,
        })
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Line {
    pub(crate) length: f32,
    pub(crate) thickness: f32,
    pub(crate) rgba: [u8; 4],
    pub(crate) stroke: Option<ShapeStrokeSpec>,
}

#[pymethods]
impl Line {
    #[new]
    #[pyo3(signature = (length = 200.0, thickness = 8.0, color = None, stroke = None))]
    fn new(
        length: f32,
        thickness: f32,
        color: Option<&Bound<'_, PyAny>>,
        stroke: Option<ShapeStrokeSpec>,
    ) -> PyResult<Self> {
        Ok(Self {
            length,
            thickness,
            rgba: parse_color_opt(color)?,
            stroke,
        })
    }
}

fn shape_generator(
    shape: Shape,
    rgba: [u8; 4],
    width: f32,
    height: f32,
    corner_radius: f32,
    stroke: Option<ShapeStrokeSpec>,
) -> Generator {
    Generator::Shape {
        shape,
        rgba: Param::Constant(rgba),
        width: Param::Constant(width),
        height: Param::Constant(height),
        corner_radius: Param::Constant(corner_radius),
        stroke: stroke_from(stroke),
    }
}

pub(crate) fn generator_from(obj: &Bound<'_, PyAny>) -> PyResult<Generator> {
    if let Ok(s) = obj.extract::<PyRef<Solid>>() {
        return Ok(Generator::SolidColor { rgba: s.rgba });
    }
    if let Ok(t) = obj.extract::<PyRef<Text>>() {
        return Ok(Generator::Text {
            content: t.content.clone(),
            style: t.style.clone(),
        });
    }
    if let Ok(r) = obj.extract::<PyRef<Rect>>() {
        return Ok(shape_generator(
            Shape::Rectangle,
            r.rgba,
            r.width,
            r.height,
            r.corner_radius,
            r.stroke.clone(),
        ));
    }
    if let Ok(e) = obj.extract::<PyRef<Ellipse>>() {
        return Ok(shape_generator(
            Shape::Ellipse,
            e.rgba,
            e.width,
            e.height,
            e.corner_radius,
            e.stroke.clone(),
        ));
    }
    if let Ok(p) = obj.extract::<PyRef<Polygon>>() {
        return Ok(shape_generator(
            Shape::Polygon { sides: p.sides },
            p.rgba,
            p.width,
            p.height,
            p.corner_radius,
            p.stroke.clone(),
        ));
    }
    if let Ok(s) = obj.extract::<PyRef<Star>>() {
        return Ok(Generator::Shape {
            shape: Shape::Star {
                points: s.points,
                inner_ratio: Param::Constant(s.inner_ratio),
            },
            rgba: Param::Constant(s.rgba),
            width: Param::Constant(s.width),
            height: Param::Constant(s.height),
            corner_radius: Param::Constant(0.0),
            stroke: stroke_from(s.stroke.clone()),
        });
    }
    if let Ok(l) = obj.extract::<PyRef<Line>>() {
        return Ok(shape_generator(
            Shape::Line,
            l.rgba,
            l.length,
            l.thickness,
            0.0,
            l.stroke.clone(),
        ));
    }
    if let Ok(a) = obj.extract::<PyRef<Arrow>>() {
        return Ok(shape_generator(
            Shape::Arrow,
            a.rgba,
            a.width,
            a.height,
            a.corner_radius,
            a.stroke.clone(),
        ));
    }
    if let Ok(h) = obj.extract::<PyRef<Heart>>() {
        return Ok(shape_generator(
            Shape::Heart,
            h.rgba,
            h.width,
            h.height,
            h.corner_radius,
            h.stroke.clone(),
        ));
    }
    Err(PyValueError::new_err(
        "content must be Media, MediaSlice, Text, Solid, or a shape descriptor",
    ))
}

fn extract_optional<T: pyo3::PyClass + Clone>(value: &Bound<'_, PyAny>) -> PyResult<Option<T>> {
    if value.is_none() {
        Ok(None)
    } else {
        let r: PyRef<'_, T> = value.extract().map_err(PyErr::from)?;
        Ok(Some((*r).clone()))
    }
}

pub(crate) fn apply_text_style(style: &mut TextStyle, dict: &Bound<'_, PyDict>) -> PyResult<()> {
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        match key.as_str() {
            "font" => style.font = value.extract()?,
            "size" => style.size = value.extract()?,
            "color" => style.fill = parse_color(&value)?,
            "bold" => style.bold = value.extract()?,
            "italic" => style.italic = value.extract()?,
            "underline" => style.underline = value.extract()?,
            "letter_spacing" => style.letter_spacing = value.extract()?,
            "line_spacing" => style.line_spacing = value.extract()?,
            "wrap" => style.wrap = value.extract()?,
            "case" => style.case = parse_text_case(&value.extract::<String>()?)?,
            "align" => {
                let (h, v): (String, String) = value.extract()?;
                style.align_h = parse_align_h(&h)?;
                style.align_v = parse_align_v(&v)?;
            }
            "stroke" => style.stroke = extract_optional::<TextStroke>(&value)?.map(|s| s.inner),
            "background" => {
                style.background = extract_optional::<TextBackground>(&value)?.map(|b| b.inner);
            }
            "shadow" => style.shadow = extract_optional::<TextShadow>(&value)?.map(|s| s.inner),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown text style parameter {other:?}"
                )));
            }
        }
    }
    Ok(())
}
