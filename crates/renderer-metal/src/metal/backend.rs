//! CPU RGBA submissions and retained RGBA8 textures.

use std::ffi::c_void;
use std::fmt;
use std::ptr;

use cutlass_gpu::Extent;

use super::sys;

/// Why a Metal submission or texture was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetalError {
    Unavailable,
    Failed,
    Token,
    Extent,
    Bytes,
}

impl fmt::Display for MetalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Unavailable => "metal is unavailable",
            Self::Failed => "the metal command failed",
            Self::Token => "the texture token is unknown",
            Self::Extent => "the extent is not valid",
            Self::Bytes => "the byte length does not match the extent",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for MetalError {}

/// Tight RGBA stride rounded up to `align`. Zero `align` is 1. Overflow is [`None`].
#[must_use]
fn row_stride(width: usize, align: usize) -> Option<usize> {
    let align = if align == 0 { 1 } else { align };
    let tight = width.checked_mul(4)?;
    if tight % align == 0 {
        Some(tight)
    } else {
        let pad = align - (tight % align);
        tight.checked_add(pad)
    }
}

fn checked_rgba(extent: Extent, rgba: &[u8]) -> Result<usize, MetalError> {
    let Some(len) = extent.rgba_len() else {
        return Err(MetalError::Extent);
    };
    if len != rgba.len() {
        return Err(MetalError::Bytes);
    }
    Ok(len)
}

fn positive_edges(extent: Extent) -> Result<(usize, usize), MetalError> {
    if extent.rgba_len().is_none() || extent.width == 0 || extent.height == 0 {
        return Err(MetalError::Extent);
    }
    let width = usize::try_from(extent.width).map_err(|_| MetalError::Extent)?;
    let height = usize::try_from(extent.height).map_err(|_| MetalError::Extent)?;
    Ok((width, height))
}

fn aligned_stride(device: *mut c_void, width: usize) -> Result<usize, MetalError> {
    let align = unsafe { sys::linear_alignment(device) };
    row_stride(width, align).ok_or(MetalError::Extent)
}

fn zeroed(len: usize) -> Result<Vec<u8>, MetalError> {
    let Some(max) = usize::try_from(isize::MAX).ok() else {
        return Err(MetalError::Extent);
    };
    if len > max {
        return Err(MetalError::Extent);
    }
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(len)
        .map_err(|_| MetalError::Failed)?;
    buffer.resize(len, 0);
    Ok(buffer)
}

fn pad_rows(
    rgba: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<Vec<u8>, MetalError> {
    let tight = width.checked_mul(4).ok_or(MetalError::Extent)?;
    let total = stride.checked_mul(height).ok_or(MetalError::Extent)?;
    if stride < tight {
        return Err(MetalError::Extent);
    }
    let mut padded = zeroed(total)?;
    for row in 0..height {
        let src = row.checked_mul(tight).ok_or(MetalError::Extent)?;
        let dst = row.checked_mul(stride).ok_or(MetalError::Extent)?;
        let src_end = src.checked_add(tight).ok_or(MetalError::Extent)?;
        let dst_end = dst.checked_add(tight).ok_or(MetalError::Extent)?;
        let src_row = rgba.get(src..src_end).ok_or(MetalError::Bytes)?;
        let dst_row = padded.get_mut(dst..dst_end).ok_or(MetalError::Extent)?;
        dst_row.copy_from_slice(src_row);
    }
    Ok(padded)
}

fn copy_out(contents: *const u8, len: usize) -> Result<Vec<u8>, MetalError> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if contents.is_null() {
        return Err(MetalError::Failed);
    }
    let mut got = zeroed(len)?;
    unsafe { ptr::copy_nonoverlapping(contents, got.as_mut_ptr(), len) };
    Ok(got)
}

fn unpack_rows(
    contents: *const u8,
    width: usize,
    height: usize,
    stride: usize,
) -> Result<Vec<u8>, MetalError> {
    if contents.is_null() {
        return Err(MetalError::Failed);
    }
    let tight = width.checked_mul(4).ok_or(MetalError::Extent)?;
    let packed = tight.checked_mul(height).ok_or(MetalError::Extent)?;
    let limit = stride.checked_mul(height).ok_or(MetalError::Extent)?;
    if stride < tight {
        return Err(MetalError::Extent);
    }
    let mut out = zeroed(packed)?;
    for row in 0..height {
        let src = row.checked_mul(stride).ok_or(MetalError::Extent)?;
        let dst = row.checked_mul(tight).ok_or(MetalError::Extent)?;
        let src_end = src.checked_add(tight).ok_or(MetalError::Extent)?;
        let dst_end = dst.checked_add(tight).ok_or(MetalError::Extent)?;
        if src_end > limit {
            return Err(MetalError::Extent);
        }
        let dst_row = out.get_mut(dst..dst_end).ok_or(MetalError::Extent)?;
        unsafe { ptr::copy_nonoverlapping(contents.add(src), dst_row.as_mut_ptr(), tight) };
    }
    Ok(out)
}

unsafe fn own(ptr: *mut c_void) -> Result<sys::Release, MetalError> {
    if ptr.is_null() {
        Err(MetalError::Failed)
    } else {
        Ok(unsafe { sys::Release::new(ptr) })
    }
}

#[derive(Debug)]
struct TextureSlot {
    token: u64,
    texture: *mut c_void,
    extent: Extent,
}

/// A Metal device, its queue, and the textures created from CPU RGBA.
#[derive(Debug)]
pub struct Backend {
    device: *mut c_void,
    queue: *mut c_void,
    textures: Vec<TextureSlot>,
    next_token: u64,
}

impl Backend {
    pub fn open() -> Result<Self, MetalError> {
        // Retained device and queue outlive this pool. It drains constructor temporaries.
        let pool = sys::Pool::push();
        let device = unsafe { sys::system_device() };
        if device.is_null() {
            return Err(MetalError::Unavailable);
        }
        let queue = unsafe { sys::new_command_queue(device) };
        if queue.is_null() {
            sys::release(device);
            return Err(MetalError::Unavailable);
        }
        drop(pool);
        Ok(Self {
            device,
            queue,
            textures: Vec::new(),
            next_token: 1,
        })
    }

    /// GPU blit of tightly packed RGBA8 from one shared buffer to another.
    ///
    /// Readback must equal `rgba`. Zero pixels is `Ok(())` and does not submit.
    /// A completed command whose bytes differ is [`MetalError::Failed`].
    pub fn submit_rgba(&mut self, extent: Extent, rgba: &[u8]) -> Result<(), MetalError> {
        let len = checked_rgba(extent, rgba)?;
        if len == 0 {
            return Ok(());
        }
        let pool = sys::Pool::push();
        let source = unsafe { own(sys::buffer_with_bytes(self.device, rgba))? };
        let destination = unsafe { own(sys::buffer_with_length(self.device, len))? };
        let source_ptr = source.get();
        let destination_ptr = destination.get();
        self.blit(|encoder| unsafe {
            sys::copy_buffer_to_buffer(encoder, source_ptr, destination_ptr, len);
        })?;
        let got = copy_out(unsafe { sys::contents(destination.get()) }, len)?;
        if got.as_slice() != rgba {
            return Err(MetalError::Failed);
        }
        drop(destination);
        drop(source);
        drop(pool);
        Ok(())
    }

    /// Create an RGBA8Unorm texture, store it, and return a nonzero token.
    pub fn insert_texture(&mut self, extent: Extent, rgba: &[u8]) -> Result<u64, MetalError> {
        checked_rgba(extent, rgba)?;
        let (width, height) = positive_edges(extent)?;
        let stride = aligned_stride(self.device, width)?;
        let bytes_per_image = stride.checked_mul(height).ok_or(MetalError::Extent)?;
        let padded = pad_rows(rgba, width, height, stride)?;
        let pool = sys::Pool::push();
        let upload = unsafe { own(sys::buffer_with_bytes(self.device, &padded))? };
        let texture = unsafe { own(sys::texture_rgba(self.device, width, height))? };
        let size = sys::Size {
            width,
            height,
            depth: 1,
        };
        let source = upload.get();
        let destination = texture.get();
        self.blit(|encoder| unsafe {
            sys::copy_buffer_to_texture(
                encoder,
                source,
                stride,
                bytes_per_image,
                size,
                destination,
            );
        })?;
        let kept = texture.take();
        drop(upload);
        drop(pool);
        self.remember(kept, extent)
    }

    /// GPU-blit a stored texture into a shared buffer and return tightly packed RGBA.
    pub fn read_texture(&mut self, token: u64, extent: Extent) -> Result<Vec<u8>, MetalError> {
        let (texture, stored) = self.texture(token)?;
        if stored != extent {
            return Err(MetalError::Extent);
        }
        let (width, height) = positive_edges(extent)?;
        let stride = aligned_stride(self.device, width)?;
        let bytes_per_image = stride.checked_mul(height).ok_or(MetalError::Extent)?;
        let pool = sys::Pool::push();
        let destination = unsafe { own(sys::buffer_with_length(self.device, bytes_per_image))? };
        let size = sys::Size {
            width,
            height,
            depth: 1,
        };
        let destination_ptr = destination.get();
        self.blit(|encoder| unsafe {
            sys::copy_texture_to_buffer(
                encoder,
                texture,
                size,
                destination_ptr,
                stride,
                bytes_per_image,
            );
        })?;
        let got = unpack_rows(
            unsafe { sys::contents(destination.get()) },
            width,
            height,
            stride,
        )?;
        drop(destination);
        drop(pool);
        Ok(got)
    }

    fn blit(&self, encode: impl FnOnce(*mut c_void)) -> Result<(), MetalError> {
        let Some((command, encoder)) = (unsafe { sys::begin_blit(self.queue) }) else {
            return Err(MetalError::Failed);
        };
        encode(encoder);
        let status = unsafe { sys::finish(encoder, command) };
        if status == sys::COMPLETED {
            Ok(())
        } else {
            Err(MetalError::Failed)
        }
    }

    fn texture(&self, token: u64) -> Result<(*mut c_void, Extent), MetalError> {
        self.textures
            .iter()
            .find(|slot| slot.token == token)
            .map(|slot| (slot.texture, slot.extent))
            .ok_or(MetalError::Token)
    }

    fn remember(&mut self, texture: *mut c_void, extent: Extent) -> Result<u64, MetalError> {
        if texture.is_null() {
            return Err(MetalError::Failed);
        }
        let token = self.next_token;
        let Some(next) = token.checked_add(1) else {
            sys::release(texture);
            return Err(MetalError::Failed);
        };
        self.next_token = next;
        self.textures.push(TextureSlot {
            token,
            texture,
            extent,
        });
        Ok(token)
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        for slot in &self.textures {
            sys::release(slot.texture);
        }
        sys::release(self.queue);
        sys::release(self.device);
    }
}

#[cfg(test)]
mod tests {
    use cutlass_gpu::MAX_EDGE;

    use super::{
        Backend, Extent, MetalError, checked_rgba, positive_edges, row_stride as packed_row,
    };

    const PATTERN: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    fn square() -> Extent {
        Extent {
            width: 2,
            height: 2,
        }
    }

    #[test]
    fn row_stride() {
        assert_eq!(packed_row(1, 1), Some(4));
        assert_eq!(packed_row(2, 1), Some(8));
        assert_eq!(packed_row(1, 64), Some(64));
        assert_eq!(packed_row(16, 64), Some(64));
        assert_eq!(packed_row(1, 0), Some(4));
        assert_eq!(packed_row(16, 0), Some(64));
        assert_eq!(packed_row(usize::MAX, 1), None);
        assert_eq!(packed_row(usize::MAX / 4, 64), None);
    }

    #[test]
    fn rgba_bytes_have_to_match_the_extent() {
        let extent = square();
        assert_eq!(checked_rgba(extent, &[0; 16]), Ok(16));
        assert_eq!(checked_rgba(extent, &[0; 15]), Err(MetalError::Bytes));
        assert_eq!(
            checked_rgba(
                Extent {
                    width: 0,
                    height: 0,
                },
                &[],
            ),
            Ok(0)
        );
        assert_eq!(
            checked_rgba(
                Extent {
                    width: MAX_EDGE + 1,
                    height: 1,
                },
                &[],
            ),
            Err(MetalError::Extent)
        );
    }

    #[test]
    fn a_zero_edge_is_not_a_texture() {
        assert_eq!(
            positive_edges(Extent {
                width: 0,
                height: 2,
            }),
            Err(MetalError::Extent)
        );
        assert_eq!(positive_edges(square()), Ok((2, 2)));
    }

    #[test]
    fn metal_error_is_an_error() {
        fn assert_error<T: std::error::Error>(error: T) -> String {
            error.to_string()
        }
        assert_eq!(
            assert_error(MetalError::Unavailable),
            "metal is unavailable"
        );
        assert_eq!(assert_error(MetalError::Failed), "the metal command failed");
        assert_eq!(
            assert_error(MetalError::Token),
            "the texture token is unknown"
        );
        assert_eq!(assert_error(MetalError::Extent), "the extent is not valid");
        assert_eq!(
            assert_error(MetalError::Bytes),
            "the byte length does not match the extent"
        );
    }

    #[test]
    fn cpu_rgba_survives_a_metal_blit() {
        let mut backend = Backend::open().unwrap();
        let extent = square();
        backend
            .submit_rgba(
                Extent {
                    width: 0,
                    height: 0,
                },
                &[],
            )
            .unwrap();
        assert_eq!(
            backend.submit_rgba(extent, &[1, 2, 3]).unwrap_err(),
            MetalError::Bytes
        );
        backend.submit_rgba(extent, &PATTERN).unwrap();
    }

    #[test]
    fn a_platform_token_is_a_metal_texture() {
        let mut backend = Backend::open().unwrap();
        let extent = square();
        let token = backend.insert_texture(extent, &PATTERN).unwrap();
        assert_ne!(token, 0);
        assert_eq!(backend.read_texture(token, extent).unwrap(), PATTERN);
        assert_eq!(
            backend
                .read_texture(
                    token,
                    Extent {
                        width: 1,
                        height: 1,
                    },
                )
                .unwrap_err(),
            MetalError::Extent
        );
    }

    #[test]
    fn an_unknown_token_is_refused() {
        let mut backend = Backend::open().unwrap();
        let extent = square();
        assert_eq!(
            backend.read_texture(0, extent).unwrap_err(),
            MetalError::Token
        );
        assert_eq!(
            backend.read_texture(99, extent).unwrap_err(),
            MetalError::Token
        );
    }
}
