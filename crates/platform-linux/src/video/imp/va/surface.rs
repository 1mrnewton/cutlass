//! VA-API objects that are freed when dropped, and the copy of a decoded
//! surface back to the CPU.
//!
//! A decoder makes a [`Config`], a [`Pool`] of surfaces and a [`Context`]
//! over them for each sequence. A surface is free while only the pool
//! holds it: the decoded picture buffer and the pictures waiting for output
//! hold clones of its `Arc`. Each surface keeps the set it was made in
//! alive, so a picture of an old sequence can still be read back after the
//! next sequence's pool replaced it.

use std::ffi::{c_int, c_uint, c_void};
use std::ptr;
use std::sync::Arc;

use cutlass_color::{nv12, p010};
use cutlass_decode::CodecError;

use super::abi;
use super::api::{ConfigAttrib, Id, Status};
use super::display::Node;
use crate::video::imp::blob::{Blob, Field, U32};

/// A VA status as a decode error, with libva's words in the log.
pub fn failed(node: &Node, status: Status, what: &'static str) -> CodecError {
    cutlass_logger::debug!("VA-API: {what}: {}", node.va.error(status));
    CodecError::Malformed { at_bit: 0, what }
}

/// A VA status as a refusal: the driver does not take this stream.
pub fn refused(node: &Node, status: Status, what: &'static str) -> CodecError {
    cutlass_logger::info!("VA-API: {what}: {}", node.va.error(status));
    CodecError::Unsupported { what }
}

fn check(status: Status) -> Result<(), Status> {
    if status == abi::VA_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(status)
    }
}

/// A profile and entrypoint with their attributes.
pub struct Config {
    node: &'static Node,
    pub id: Id,
}

impl Config {
    pub fn new(
        node: &'static Node,
        profile: i32,
        entrypoint: i32,
        attributes: &[ConfigAttrib],
    ) -> Result<Self, Status> {
        let mut attributes = attributes.to_vec();
        let count = c_int::try_from(attributes.len()).unwrap_or(0);
        let mut id = abi::VA_INVALID_ID;
        // SAFETY: the display is initialised and the array holds `count`
        // VAConfigAttribs, which libva only reads.
        check(unsafe {
            (node.va.create_config)(
                node.display,
                profile,
                entrypoint,
                attributes.as_mut_ptr().cast(),
                count,
                &raw mut id,
            )
        })?;
        Ok(Self { node, id })
    }
}

impl Drop for Config {
    fn drop(&mut self) {
        // SAFETY: the config was made on this display and is destroyed once.
        unsafe { (self.node.va.destroy_config)(self.node.display, self.id) };
    }
}

/// Surfaces made together, destroyed when the last one is let go.
struct Surfaces {
    node: &'static Node,
    ids: Vec<Id>,
}

impl Drop for Surfaces {
    fn drop(&mut self) {
        let count = c_int::try_from(self.ids.len()).unwrap_or(0);
        // SAFETY: the surfaces were made on this display, nothing renders
        // into them any more, and they are destroyed once.
        unsafe { (self.node.va.destroy_surfaces)(self.node.display, self.ids.as_mut_ptr(), count) };
    }
}

/// One surface: its id, size and bit depth.
pub struct Surface {
    pub id: Id,
    pub width: u32,
    pub height: u32,
    /// 8 (NV12) or 10 (P010).
    pub depth: u8,
    _set: Arc<Surfaces>,
}

/// The surfaces a context renders into.
pub struct Pool {
    slots: Vec<Arc<Surface>>,
}

impl Pool {
    /// `count` 4:2:0 surfaces of `width` by `height` at `depth` bits.
    pub fn new(
        node: &'static Node,
        width: u32,
        height: u32,
        depth: u8,
        count: usize,
    ) -> Result<Self, Status> {
        let format = if depth > 8 {
            abi::VA_RT_FORMAT_YUV420_10
        } else {
            abi::VA_RT_FORMAT_YUV420
        };
        let mut ids = vec![abi::VA_INVALID_SURFACE; count];
        let wanted = c_uint::try_from(count).unwrap_or(0);
        // SAFETY: the display is initialised and the array holds `count`
        // ids; no attributes are passed, so the driver picks NV12 or P010.
        check(unsafe {
            (node.va.create_surfaces)(
                node.display,
                format,
                width,
                height,
                ids.as_mut_ptr(),
                wanted,
                ptr::null_mut(),
                0,
            )
        })?;
        let set = Arc::new(Surfaces {
            node,
            ids: ids.clone(),
        });
        let slots = ids
            .into_iter()
            .map(|id| {
                Arc::new(Surface {
                    id,
                    width,
                    height,
                    depth,
                    _set: Arc::clone(&set),
                })
            })
            .collect();
        Ok(Self { slots })
    }

    pub fn ids(&self) -> Vec<Id> {
        self.slots.iter().map(|slot| slot.id).collect()
    }

    /// A surface nothing but the pool holds.
    pub fn take(&self) -> Option<Arc<Surface>> {
        self.slots
            .iter()
            .find(|slot| Arc::strong_count(slot) == 1)
            .cloned()
    }
}

/// A decode or encode context over a pool's surfaces.
pub struct Context {
    node: &'static Node,
    pub id: Id,
}

impl Context {
    pub fn new(
        node: &'static Node,
        config: &Config,
        width: u32,
        height: u32,
        targets: &[Id],
    ) -> Result<Self, Status> {
        let mut targets = targets.to_vec();
        let count = c_int::try_from(targets.len()).unwrap_or(0);
        let mut id = abi::VA_INVALID_ID;
        // SAFETY: the config and surfaces were made on this display; the
        // array holds `count` surface ids.
        check(unsafe {
            (node.va.create_context)(
                node.display,
                config.id,
                c_int::try_from(width).unwrap_or(c_int::MAX),
                c_int::try_from(height).unwrap_or(c_int::MAX),
                abi::VA_PROGRESSIVE as c_int,
                targets.as_mut_ptr(),
                count,
                &raw mut id,
            )
        })?;
        Ok(Self { node, id })
    }

    /// Render `params` then `slices` into `target` as one picture.
    pub fn render(
        &self,
        target: Id,
        params: &[Buffer],
        slices: &[Buffer],
    ) -> Result<(), CodecError> {
        let node = self.node;
        // SAFETY: the context and surface were made on this display.
        let status = unsafe { (node.va.begin_picture)(node.display, self.id, target) };
        if status != abi::VA_STATUS_SUCCESS {
            return Err(failed(node, status, "VA-API could not begin a picture"));
        }
        let mut rendered = Ok(());
        for group in [params, slices] {
            if group.is_empty() {
                continue;
            }
            let mut ids: Vec<Id> = group.iter().map(|buffer| buffer.id).collect();
            let count = c_int::try_from(ids.len()).unwrap_or(0);
            // SAFETY: the buffers were made in this context and the array
            // holds `count` ids.
            let status =
                unsafe { (node.va.render_picture)(node.display, self.id, ids.as_mut_ptr(), count) };
            if status != abi::VA_STATUS_SUCCESS {
                rendered = Err(failed(node, status, "VA-API could not render a picture"));
                break;
            }
        }
        // A picture that began is ended even when rendering failed.
        // SAFETY: a picture of this context began above.
        let status = unsafe { (node.va.end_picture)(node.display, self.id) };
        rendered?;
        if status != abi::VA_STATUS_SUCCESS {
            return Err(failed(node, status, "VA-API could not decode a picture"));
        }
        Ok(())
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: the context was made on this display and is destroyed
        // once.
        unsafe { (self.node.va.destroy_context)(self.node.display, self.id) };
    }
}

/// A parameter or data buffer of one picture.
pub struct Buffer {
    node: &'static Node,
    pub id: Id,
}

impl Buffer {
    /// A buffer of `kind` holding a copy of `bytes`.
    pub fn new(
        node: &'static Node,
        context: Id,
        kind: c_int,
        bytes: &[u8],
    ) -> Result<Self, CodecError> {
        Self::make(node, context, kind, bytes.as_ptr().cast(), bytes.len())
    }

    /// A buffer of `kind` holding a copy of a parameter struct.
    pub fn blob(
        node: &'static Node,
        context: Id,
        kind: c_int,
        blob: &Blob,
    ) -> Result<Self, CodecError> {
        Self::make(node, context, kind, blob.as_ptr(), blob.len())
    }

    fn make(
        node: &'static Node,
        context: Id,
        kind: c_int,
        data: *const c_void,
        len: usize,
    ) -> Result<Self, CodecError> {
        let size = c_uint::try_from(len).map_err(|_| CodecError::Limit {
            what: "VA-API buffer size",
        })?;
        let mut id = abi::VA_INVALID_ID;
        // SAFETY: `data` points at `len` readable bytes; libva copies them
        // into the new buffer and never writes through the pointer.
        let status = unsafe {
            (node.va.create_buffer)(
                node.display,
                context,
                kind,
                size,
                1,
                data.cast_mut(),
                &raw mut id,
            )
        };
        if status != abi::VA_STATUS_SUCCESS {
            return Err(failed(node, status, "VA-API could not make a buffer"));
        }
        Ok(Self { node, id })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: the buffer was made on this display and is destroyed
        // once, after the picture that used it ended.
        unsafe { (self.node.va.destroy_buffer)(self.node.display, self.id) };
    }
}

/// The part of a surface a picture shows, in luma samples. Left and top
/// are even, as 4:2:0 cropping makes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
}

impl Window {
    /// The window left after cropping `crop` (left, right, top, bottom)
    /// off a `width` by `height` picture, cut to even edges.
    pub fn cropped(width: u32, height: u32, crop: [u32; 4]) -> Result<Self, CodecError> {
        let [left, right, top, bottom] = crop;
        let shown_w = width.saturating_sub(left.saturating_add(right)) & !1;
        let shown_h = height.saturating_sub(top.saturating_add(bottom)) & !1;
        if shown_w == 0 || shown_h == 0 || left % 2 != 0 || top % 2 != 0 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "VA-API picture window",
            });
        }
        Ok(Self {
            left,
            top,
            width: shown_w,
            height: shown_h,
        })
    }
}

/// A `VAImage` this crate made or derived, destroyed when dropped.
struct Image {
    node: &'static Node,
    blob: Blob,
}

impl Image {
    fn id(&self) -> Id {
        self.blob.get(abi::image::IMAGE_ID)
    }

    fn fourcc(&self) -> u32 {
        self.blob
            .get(abi::image_format::FOURCC.of(abi::image::FORMAT))
    }

    fn size(&self) -> (u32, u32) {
        (
            u32::from(self.blob.get(abi::image::WIDTH)),
            u32::from(self.blob.get(abi::image::HEIGHT)),
        )
    }

    /// The byte offset and pitch of plane `index`.
    fn plane(&self, index: usize) -> Option<(usize, usize)> {
        let offset = self.blob.get(abi::image::OFFSETS.at(index)?);
        let pitch = self.blob.get(abi::image::PITCHES.at(index)?);
        Some((offset as usize, pitch as usize))
    }

    /// Map the image and copy `window` out as packed NV12 or P010.
    fn copy(&self, window: Window, sample: usize) -> Result<Vec<u8>, CodecError> {
        let node = self.node;
        let buffer = self.blob.get(abi::image::BUF);
        let len = self.blob.get(abi::image::DATA_SIZE) as usize;
        let mut data: *mut c_void = ptr::null_mut();
        // SAFETY: the image's buffer belongs to this display and the out
        // pointer is valid.
        let status = unsafe { (node.va.map_buffer)(node.display, buffer, &raw mut data) };
        if status != abi::VA_STATUS_SUCCESS || data.is_null() {
            return Err(failed(node, status, "VA-API could not map a picture"));
        }
        // SAFETY: a mapped image buffer holds `data_size` bytes until it is
        // unmapped below, and nothing writes to it meanwhile.
        let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), len) };
        let planes = [self.plane(0), self.plane(1)];
        let packed = match planes {
            [Some(luma), Some(chroma)] => pack(bytes, [luma, chroma], window, sample),
            _ => Err(SHORT),
        };
        // SAFETY: the buffer was mapped above and is unmapped once.
        unsafe { (node.va.unmap_buffer)(node.display, buffer) };
        packed
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // SAFETY: the image was made or derived on this display and is
        // destroyed once; its buffer is unmapped.
        unsafe { (self.node.va.destroy_image)(self.node.display, self.id()) };
    }
}

const SHORT: CodecError = CodecError::Malformed {
    at_bit: 0,
    what: "VA-API image is smaller than its picture",
};

/// `window` of an NV12 or P010 image, whose planes are at `planes` (byte
/// offset and pitch of luma, then interleaved chroma), as packed NV12 or
/// P010. `sample` is 1 or 2 bytes.
fn pack(
    bytes: &[u8],
    planes: [(usize, usize); 2],
    window: Window,
    sample: usize,
) -> Result<Vec<u8>, CodecError> {
    let row = window.width as usize * sample;
    let left = window.left as usize * sample;
    let top = window.top as usize;
    let rows = window.height as usize;
    let mut out = Vec::with_capacity(row * rows / 2 * 3);
    let mut copy = |(base, pitch): (usize, usize), range: std::ops::Range<usize>| {
        for y in range {
            let at = y
                .checked_mul(pitch)
                .and_then(|start| start.checked_add(base))
                .and_then(|start| start.checked_add(left))
                .ok_or(SHORT)?;
            let end = at.checked_add(row).ok_or(SHORT)?;
            out.extend_from_slice(bytes.get(at..end).ok_or(SHORT)?);
        }
        Ok::<(), CodecError>(())
    };
    copy(planes[0], top..top + rows)?;
    // Interleaved chroma: half the rows, the same bytes a row.
    copy(planes[1], top / 2..(top + rows) / 2)?;
    Ok(out)
}

/// Copies decoded surfaces back as 8-bit planar 4:2:0.
///
/// The surface is copied into an image of the same size with `vaGetImage`,
/// which drivers do on the GPU into plain memory. The image is kept for
/// the next picture of the same size. A driver that refuses has the
/// surface derived (`vaDeriveImage`) and mapped instead, from then on.
pub struct Readback {
    node: &'static Node,
    image: Option<Image>,
    derive: bool,
}

impl Readback {
    pub fn new(node: &'static Node) -> Self {
        Self {
            node,
            image: None,
            derive: false,
        }
    }

    /// The `window` of `surface` once it finished decoding.
    pub fn read(&mut self, surface: &Surface, window: Window) -> Result<Vec<u8>, CodecError> {
        let node = self.node;
        if window.left + window.width > surface.width || window.top + window.height > surface.height
        {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "VA-API picture window",
            });
        }
        // SAFETY: the surface was made on this display.
        let status = unsafe { (node.va.sync_surface)(node.display, surface.id) };
        if status != abi::VA_STATUS_SUCCESS {
            return Err(failed(node, status, "VA-API could not finish a picture"));
        }
        let fourcc = fourcc_of(surface.depth);
        let packed = if self.derive {
            self.derived(surface, fourcc, window)?
        } else {
            match self.copied(surface, fourcc, window) {
                Ok(packed) => packed,
                Err(status) => {
                    cutlass_logger::info!(
                        "VA-API: vaGetImage does not copy {}: {}; mapping the surface instead",
                        fourcc_name(fourcc),
                        node.va.error(status)
                    );
                    self.image = None;
                    self.derive = true;
                    self.derived(surface, fourcc, window)?
                }
            }
        };
        let (width, height) = (window.width, window.height);
        if surface.depth > 8 {
            p010::to_planar8(p010::Layout::packed(width, height), &packed)
        } else {
            nv12::to_planar(nv12::Layout::packed(width, height), &packed)
        }
    }

    /// Copy through the kept image; `Err` is the status of a step the
    /// driver refused.
    fn copied(
        &mut self,
        surface: &Surface,
        fourcc: u32,
        window: Window,
    ) -> Result<Vec<u8>, Status> {
        let node = self.node;
        let size = (surface.width, surface.height);
        let fits = self
            .image
            .as_ref()
            .is_some_and(|image| image.fourcc() == fourcc && image.size() == size);
        if !fits {
            self.image = None;
            self.image = Some(create_image(node, fourcc, size)?);
        }
        let Some(image) = &self.image else {
            return Err(abi::VA_STATUS_ERROR_OPERATION_FAILED);
        };
        // SAFETY: the surface and image were made on this display and the
        // rectangle is the whole surface, which the image matches.
        check(unsafe {
            (node.va.get_image)(
                node.display,
                surface.id,
                0,
                0,
                surface.width,
                surface.height,
                image.id(),
            )
        })?;
        image
            .copy(window, sample_of(surface.depth))
            .map_err(|_| abi::VA_STATUS_ERROR_OPERATION_FAILED)
    }

    fn derived(
        &self,
        surface: &Surface,
        fourcc: u32,
        window: Window,
    ) -> Result<Vec<u8>, CodecError> {
        let node = self.node;
        let mut blob = Blob::new(abi::image::SIZE);
        // SAFETY: the surface was made on this display and the blob is a
        // zeroed VAImage for libva to fill.
        let status = unsafe { (node.va.derive_image)(node.display, surface.id, blob.as_mut_ptr()) };
        if status != abi::VA_STATUS_SUCCESS {
            return Err(failed(node, status, "VA-API could not read a picture back"));
        }
        let image = Image { node, blob };
        if image.fourcc() != fourcc {
            return Err(CodecError::Unsupported {
                what: "VA-API surface is not NV12 or P010",
            });
        }
        image.copy(window, sample_of(surface.depth))
    }
}

fn fourcc_of(depth: u8) -> u32 {
    if depth > 8 {
        abi::VA_FOURCC_P010
    } else {
        abi::VA_FOURCC_NV12
    }
}

fn sample_of(depth: u8) -> usize {
    if depth > 8 { 2 } else { 1 }
}

fn fourcc_name(fourcc: u32) -> String {
    fourcc
        .to_le_bytes()
        .iter()
        .map(|byte| char::from(*byte))
        .collect()
}

/// An image of `fourcc` at `size`, in the format the driver lists for it.
fn create_image(node: &'static Node, fourcc: u32, size: (u32, u32)) -> Result<Image, Status> {
    let mut format = image_format(node, fourcc).ok_or(abi::VA_STATUS_ERROR_OPERATION_FAILED)?;
    let mut blob = Blob::new(abi::image::SIZE);
    // SAFETY: the format is one the driver listed and the blob is a zeroed
    // VAImage for libva to fill.
    check(unsafe {
        (node.va.create_image)(
            node.display,
            format.as_mut_ptr(),
            c_int::try_from(size.0).unwrap_or(c_int::MAX),
            c_int::try_from(size.1).unwrap_or(c_int::MAX),
            blob.as_mut_ptr(),
        )
    })?;
    Ok(Image { node, blob })
}

/// The driver's `VAImageFormat` for `fourcc`.
fn image_format(node: &Node, fourcc: u32) -> Option<Blob> {
    // SAFETY: the display is initialised.
    let most = unsafe { (node.va.max_num_image_formats)(node.display) }.clamp(0, 256);
    let most = usize::try_from(most).ok()?;
    let stride = abi::image_format::SIZE;
    let mut list = Blob::new(stride * most);
    let mut count: c_int = 0;
    // SAFETY: the list holds the most formats the driver says it returns.
    let status =
        unsafe { (node.va.query_image_formats)(node.display, list.as_mut_ptr(), &raw mut count) };
    if status != abi::VA_STATUS_SUCCESS {
        return None;
    }
    let count = usize::try_from(count).unwrap_or(0).min(most);
    (0..count)
        .map(|index| index * stride)
        .find(|at| list.get(U32(at + abi::image_format::FOURCC.0)) == fourcc)
        .map(|at| list.slice(at, stride))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_is_the_picture_less_its_crop_cut_to_even() {
        assert_eq!(
            Window::cropped(1920, 1088, [0, 0, 0, 8]).expect("window"),
            Window {
                left: 0,
                top: 0,
                width: 1920,
                height: 1080
            }
        );
        assert_eq!(
            Window::cropped(64, 48, [2, 1, 4, 3]).expect("window").width,
            60
        );
        assert!(Window::cropped(64, 48, [64, 0, 0, 0]).is_err());
        assert!(Window::cropped(64, 48, [1, 0, 0, 0]).is_err());
    }

    #[test]
    fn the_window_is_cut_out_of_both_planes_at_their_pitches() {
        // A 6x4 NV12 image, pitch 8, chroma at 40 with pitch 8: each byte
        // is its own position.
        let bytes: Vec<u8> = (0..64).collect();
        let window = Window {
            left: 2,
            top: 2,
            width: 4,
            height: 2,
        };
        let packed = pack(&bytes, [(0, 8), (40, 8)], window, 1).expect("pack");
        assert_eq!(packed, [18, 19, 20, 21, 26, 27, 28, 29, 50, 51, 52, 53]);
        let deep = pack(
            &bytes,
            [(0, 16), (32, 16)],
            Window {
                left: 0,
                top: 0,
                width: 2,
                height: 2,
            },
            2,
        )
        .expect("pack");
        assert_eq!(deep, [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35]);
        assert!(pack(&bytes, [(0, 8), (60, 8)], window, 1).is_err());
    }

    #[test]
    fn a_fourcc_reads_as_its_letters() {
        assert_eq!(fourcc_name(abi::VA_FOURCC_NV12), "NV12");
        assert_eq!(fourcc_name(abi::VA_FOURCC_P010), "P010");
    }
}
