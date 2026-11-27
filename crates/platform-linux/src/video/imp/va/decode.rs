//! What the VA-API decoders share: a sequence's config, surfaces and
//! context, and the picture each decoded surface becomes.

use std::sync::Arc;

use cutlass_decode::{CodecError, Output, PictureOutput};

use super::abi;
use super::api::ConfigAttrib;
use super::display::Node;
use super::surface::{Context, Pool, Readback, Surface, Window, refused};
use crate::video::colour::{Signal, picture_info};

/// What a sequence's surfaces and context are made for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub profile: i32,
    pub width: u32,
    pub height: u32,
    /// 8 or 10.
    pub depth: u8,
    pub surfaces: usize,
}

/// One sequence's context. Fields drop in order: the context before the
/// surfaces it renders into, and both before their config.
pub struct Session {
    pub context: Context,
    pub pool: Pool,
    _config: super::surface::Config,
    pub shape: Shape,
}

impl Session {
    /// A decode context for `shape`, or [`CodecError::Unsupported`] when
    /// the node does not decode it.
    pub fn open(node: &'static Node, shape: Shape) -> Result<Self, CodecError> {
        let entrypoint = abi::VAEntrypointVLD;
        if !node.offers(shape.profile, entrypoint) {
            return Err(CodecError::Unsupported {
                what: "this GPU does not decode the profile",
            });
        }
        let format = if shape.depth > 8 {
            abi::VA_RT_FORMAT_YUV420_10
        } else {
            abi::VA_RT_FORMAT_YUV420
        };
        let attributes = node.attributes(
            shape.profile,
            entrypoint,
            &[
                abi::VAConfigAttribRTFormat,
                abi::VAConfigAttribMaxPictureWidth,
                abi::VAConfigAttribMaxPictureHeight,
            ],
        );
        let Ok([rt_format, max_width, max_height]) = <[u32; 3]>::try_from(attributes) else {
            return Err(CodecError::Unsupported {
                what: "VA-API attributes",
            });
        };
        let known = |value: u32| value != abi::VA_ATTRIB_NOT_SUPPORTED;
        if known(rt_format) && rt_format & format == 0 {
            return Err(CodecError::Unsupported {
                what: "this GPU does not decode to this format",
            });
        }
        if (known(max_width) && shape.width > max_width)
            || (known(max_height) && shape.height > max_height)
        {
            return Err(CodecError::Unsupported {
                what: "the picture size is out of the GPU's range",
            });
        }
        let attribute = ConfigAttrib {
            kind: abi::VAConfigAttribRTFormat,
            value: format,
        };
        let config = super::surface::Config::new(node, shape.profile, entrypoint, &[attribute])
            .map_err(|status| refused(node, status, "VA-API could not make a decode config"))?;
        let pool = Pool::new(node, shape.width, shape.height, shape.depth, shape.surfaces)
            .map_err(|status| refused(node, status, "VA-API could not make surfaces"))?;
        let context = Context::new(node, &config, shape.width, shape.height, &pool.ids())
            .map_err(|status| refused(node, status, "VA-API could not make a decode context"))?;
        Ok(Self {
            context,
            pool,
            _config: config,
            shape,
        })
    }

    /// A surface nothing references.
    pub fn take(&self) -> Result<Arc<Surface>, CodecError> {
        self.pool.take().ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "VA-API has no free surface",
        })
    }
}

/// Keep `session` when it was made for `shape`, else drop it (pictures of
/// it waiting for output keep their surfaces) and open one for `shape`.
pub fn renew(
    node: &'static Node,
    session: &mut Option<Session>,
    shape: Shape,
) -> Result<(), CodecError> {
    if session.as_ref().is_some_and(|s| s.shape == shape) {
        return Ok(());
    }
    *session = None;
    *session = Some(Session::open(node, shape)?);
    Ok(())
}

/// The session, or an error for a picture that came before its sequence.
pub fn current(session: &Option<Session>) -> Result<&Session, CodecError> {
    session.as_ref().ok_or(CodecError::Malformed {
        at_bit: 0,
        what: "VA-API picture before its sequence",
    })
}

/// `window` of a decoded surface as an 8-bit 4:2:0 picture tagged with
/// `signal`.
pub fn picture(
    readback: &mut Readback,
    surface: &Surface,
    window: Window,
    signal: Signal,
    pts: i64,
    timescale: u32,
) -> Result<Output, CodecError> {
    let bytes = readback.read(surface, window)?;
    let info = picture_info(window.width, window.height, signal);
    Ok(Output::Picture(PictureOutput::cpu(
        info, bytes, pts, timescale,
    )?))
}

/// A value into a signed byte field. In range for every stream the
/// parsers take.
pub fn byte(value: i32) -> i8 {
    value.clamp(i32::from(i8::MIN), i32::from(i8::MAX)) as i8
}

/// A value into a signed 16-bit field.
pub fn short(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

pub fn flag(value: bool) -> u32 {
    u32::from(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_fields_saturate() {
        assert_eq!(byte(-200), -128);
        assert_eq!(byte(5), 5);
        assert_eq!(byte(300), 127);
        assert_eq!(short(70_000), i16::MAX);
        assert_eq!(short(-3), -3);
    }
}
