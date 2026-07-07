//! Python bindings for the Cutlass video engine (v2 API).

mod catalog;
mod clip;
mod content;
mod convert;
mod effect;
mod errors;
mod media;
mod project;
mod track;

use content::{
    Arrow, Ellipse, Heart, Line, Polygon, Rect, ShapeStrokeSpec, Solid, Star, Text, TextBackground,
    TextShadow, TextStroke,
};
use pyo3::prelude::*;

use catalog::register as register_catalog;
use clip::Clip;
use effect::Effect;
use errors::register as register_errors;
use media::{Media, MediaSlice};
use project::Project;
use track::Track;

/// The `cutlass` Python module.
#[pymodule]
fn cutlass(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_errors(m)?;
    register_catalog(m)?;
    m.add_class::<Project>()?;
    m.add_class::<Media>()?;
    m.add_class::<MediaSlice>()?;
    m.add_class::<Track>()?;
    m.add_class::<Clip>()?;
    m.add_class::<Effect>()?;
    m.add_class::<Text>()?;
    m.add_class::<TextStroke>()?;
    m.add_class::<TextBackground>()?;
    m.add_class::<TextShadow>()?;
    m.add_class::<Solid>()?;
    m.add_class::<ShapeStrokeSpec>()?;
    m.add_class::<Rect>()?;
    m.add_class::<Ellipse>()?;
    m.add_class::<Polygon>()?;
    m.add_class::<Star>()?;
    m.add_class::<Line>()?;
    m.add_class::<Arrow>()?;
    m.add_class::<Heart>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
