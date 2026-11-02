//! Direct3D 11 device.
//!
//! The portable [`crate::WindowsDevice`] is the integer blend oracle. This
//! backend uses the same equation on the output merger: source plus
//! destination times one minus source alpha. A platform token is a
//! shader-resource view this device has imported.

#![allow(unsafe_code)]

use std::ffi::{CString, c_char, c_void};
use std::mem::transmute;
use std::ptr;

use cutlass_gpu::{Blend, ClearColor, Device, Extent, Framebuffer, Pass, TexelSource};

use crate::WindowsError;

const HLSL: &str = "\
Texture2D tex : register(t0);\n\
SamplerState samp : register(s0);\n\
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n\
VsOut vs_main(uint id : SV_VertexID) {\n\
    float2 p = float2(float((id << 1) & 2), float(id & 2));\n\
    VsOut o;\n\
    o.pos = float4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);\n\
    o.uv = p;\n\
    return o;\n\
}\n\
float4 ps_main(VsOut i) : SV_Target { return tex.Sample(samp, i.uv); }\n";

const FORMAT_RGBA8: u32 = 28;
const USAGE_DEFAULT: u32 = 0;
const USAGE_STAGING: u32 = 3;
const BIND_SHADER: u32 = 0x8;
const BIND_TARGET: u32 = 0x20;
const CPU_READ: u32 = 0x20000;
const DRIVER_HARDWARE: u32 = 1;
const DRIVER_WARP: u32 = 5;
const SDK_VERSION: u32 = 7;
const TOPOLOGY_TRIANGLE: u32 = 4;

type Slot = unsafe extern "system" fn();

#[link(name = "d3d11")]
unsafe extern "system" {
    fn D3D11CreateDevice(
        adapter: *mut c_void,
        driver_type: u32,
        software: *mut c_void,
        flags: u32,
        levels: *const u32,
        level_count: u32,
        sdk: u32,
        device: *mut *mut c_void,
        chosen: *mut u32,
        context: *mut *mut c_void,
    ) -> i32;
}

#[link(name = "d3dcompiler")]
unsafe extern "system" {
    fn D3DCompile(
        data: *const c_void,
        data_size: usize,
        filename: *const c_char,
        defines: *const c_void,
        include: *const c_void,
        entry: *const c_char,
        target: *const c_char,
        flags1: u32,
        flags2: u32,
        code: *mut *mut c_void,
        errors: *mut *mut c_void,
    ) -> i32;
}

/// `object` must be a live COM pointer. `index` is its vtable slot.
unsafe fn vslot(object: *mut c_void, index: usize) -> Slot {
    unsafe {
        let table = *object.cast::<*const Slot>();
        *table.add(index)
    }
}

unsafe fn release(object: *mut c_void) {
    if object.is_null() {
        return;
    }
    unsafe {
        let method: unsafe extern "system" fn(*mut c_void) -> u32 = transmute(vslot(object, 2));
        method(object);
    }
}

macro_rules! com {
    ($object:expr, $index:expr, fn($($arg:ty),*) -> $ret:ty, $($value:expr),* $(,)?) => {{
        type Method = unsafe extern "system" fn(*mut c_void, $($arg),*) -> $ret;
        let method: Method = unsafe { transmute(vslot($object, $index)) };
        unsafe { method($object, $($value),*) }
    }};
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TextureDesc {
    width: u32,
    height: u32,
    mips: u32,
    layers: u32,
    format: u32,
    samples: u32,
    quality: u32,
    usage: u32,
    bind: u32,
    cpu: u32,
    misc: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Subresource {
    data: *const c_void,
    pitch: u32,
    slice: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Viewport {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
}

#[repr(C)]
struct Mapped {
    data: *mut c_void,
    row: u32,
    depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TargetBlend {
    enable: i32,
    src: u32,
    dest: u32,
    op: u32,
    src_alpha: u32,
    dest_alpha: u32,
    op_alpha: u32,
    mask: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlendDesc {
    coverage: i32,
    independent: i32,
    targets: [TargetBlend; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SamplerDesc {
    filter: u32,
    address_u: u32,
    address_v: u32,
    address_w: u32,
    mip_bias: f32,
    anisotropy: u32,
    comparison: u32,
    border: [f32; 4],
    min_lod: f32,
    max_lod: f32,
}

#[derive(Debug)]
struct Com {
    ptr: *mut c_void,
}

impl Com {
    fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        unsafe { release(self.ptr) }
        self.ptr = ptr::null_mut();
    }
}

#[derive(Clone, Copy, Debug)]
struct OpenPass {
    framebuffer: Framebuffer,
    ended: bool,
    blend: Blend,
}

#[derive(Debug)]
struct GpuBuffer {
    id: u64,
    extent: Extent,
    view: Com,
    color: Com,
    staging: Com,
    rgba: Vec<u8>,
    submitted: bool,
}

#[derive(Debug)]
struct Imported {
    id: u64,
    view: Com,
}

enum Drawn {
    Borrowed(*mut c_void),
    Owned { view: Com, _texture: Com },
}

impl Drawn {
    fn ptr(&self) -> *mut c_void {
        match self {
            Self::Borrowed(ptr) => *ptr,
            Self::Owned { view, .. } => view.as_ptr(),
        }
    }
}

/// The Direct3D 11 device behind [`cutlass_gpu::Device`].
#[derive(Debug)]
pub struct Direct3DDevice {
    vertex: Com,
    pixel: Com,
    blend_state: Com,
    sampler: Com,
    buffers: Vec<GpuBuffer>,
    imports: Vec<Imported>,
    next_id: u64,
    next_import: u64,
    open: Option<OpenPass>,
    context: Com,
    device: Com,
}

/// Open a Direct3D 11 device, trying hardware and then the software rasterizer.
pub fn open_direct3d() -> Result<Direct3DDevice, WindowsError> {
    let (device, context) = create_device()?;
    let vertex_bytes = compile_shader("vs_main", "vs_4_0")?;
    let pixel_bytes = compile_shader("ps_main", "ps_4_0")?;
    let vertex = create_shader(device.as_ptr(), 12, &vertex_bytes)?;
    let pixel = create_shader(device.as_ptr(), 15, &pixel_bytes)?;
    let blend_state = create_blend(device.as_ptr())?;
    let sampler = create_sampler(device.as_ptr())?;
    Ok(Direct3DDevice {
        vertex,
        pixel,
        blend_state,
        sampler,
        buffers: Vec::new(),
        imports: Vec::new(),
        next_id: 1,
        next_import: 1,
        open: None,
        context,
        device,
    })
}

impl Direct3DDevice {
    /// Add-ref a shader-resource view and return the token draw accepts.
    ///
    /// # Safety
    ///
    /// `view` must be null or a live Direct3D shader-resource view. A
    /// non-null pointer is add-ref'd and released when this device drops.
    pub unsafe fn import_shader_resource(
        &mut self,
        view: *mut c_void,
    ) -> Result<u64, WindowsError> {
        if view.is_null() {
            return Err(WindowsError::UnknownResource);
        }
        let id = self.next_import;
        let next = self
            .next_import
            .checked_add(1)
            .ok_or(WindowsError::GpuUnsupported)?;
        self.imports
            .try_reserve(1)
            .map_err(|_| WindowsError::GpuUnsupported)?;
        unsafe {
            let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 = transmute(vslot(view, 1));
            add_ref(view);
        }
        self.next_import = next;
        self.imports.push(Imported {
            id,
            view: Com { ptr: view },
        });
        Ok(id)
    }

    fn slot_index(&self, framebuffer: Framebuffer) -> Result<usize, WindowsError> {
        self.buffers
            .iter()
            .position(|slot| slot.id == framebuffer.id())
            .ok_or(WindowsError::MissingFramebuffer)
    }

    fn require_drawing(&self, pass: Pass) -> Result<(), WindowsError> {
        match self.open {
            Some(open) if open.framebuffer == pass.framebuffer() && !open.ended => Ok(()),
            Some(open) if open.framebuffer == pass.framebuffer() => Err(WindowsError::Ended),
            _ => Err(WindowsError::NoPass),
        }
    }

    fn import_ptr(&self, token: u64) -> Result<*mut c_void, WindowsError> {
        self.imports
            .iter()
            .find(|item| item.id == token)
            .map(|item| item.view.as_ptr())
            .ok_or(WindowsError::UnknownResource)
    }

    fn over_blend(&self) -> Result<(), WindowsError> {
        if self.open.map(|open| open.blend) != Some(Blend::Over) {
            return Err(WindowsError::NoPass);
        }
        Ok(())
    }

    fn paint(&self, target: *mut c_void, extent: Extent, source: *mut c_void) {
        let context = self.context.as_ptr();
        let view = [target];
        let source_view = [source];
        let sampler = [self.sampler.as_ptr()];
        let factor = [1.0_f32; 4];
        let viewport = Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        com!(
            context,
            33,
            fn(u32, *const *mut c_void, *mut c_void) -> (),
            1,
            view.as_ptr(),
            ptr::null_mut(),
        );
        com!(context, 44, fn(u32, *const Viewport) -> (), 1, &viewport,);
        com!(
            context,
            35,
            fn(*mut c_void, *const f32, u32) -> (),
            self.blend_state.as_ptr(),
            factor.as_ptr(),
            u32::MAX,
        );
        com!(context, 24, fn(u32) -> (), TOPOLOGY_TRIANGLE);
        com!(
            context,
            11,
            fn(*mut c_void, *const *mut c_void, u32) -> (),
            self.vertex.as_ptr(),
            ptr::null(),
            0,
        );
        com!(
            context,
            9,
            fn(*mut c_void, *const *mut c_void, u32) -> (),
            self.pixel.as_ptr(),
            ptr::null(),
            0,
        );
        com!(
            context,
            10,
            fn(u32, u32, *const *mut c_void) -> (),
            0,
            1,
            sampler.as_ptr(),
        );
        com!(
            context,
            8,
            fn(u32, u32, *const *mut c_void) -> (),
            0,
            1,
            source_view.as_ptr(),
        );
        com!(context, 13, fn(u32, u32) -> (), 3, 0);
    }
}

impl Device for Direct3DDevice {
    type Error = WindowsError;

    fn create_framebuffer(&mut self, extent: Extent) -> Result<Framebuffer, Self::Error> {
        if extent.width == 0 || extent.height == 0 {
            return Err(WindowsError::Extent);
        }
        let len = extent.rgba_len().ok_or(WindowsError::Extent)?;
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or(WindowsError::Extent)?;
        let device = self.device.as_ptr();
        let color = create_texture(
            device,
            extent,
            USAGE_DEFAULT,
            BIND_SHADER | BIND_TARGET,
            0,
            None,
        )?;
        let view = create_view(device, 9, color.as_ptr())?;
        let staging = create_texture(device, extent, USAGE_STAGING, 0, CPU_READ, None)?;
        let mut rgba = Vec::new();
        rgba.try_reserve_exact(len)
            .map_err(|_| WindowsError::Extent)?;
        rgba.resize(len, 0);
        self.buffers.push(GpuBuffer {
            id,
            extent,
            view,
            color,
            staging,
            rgba,
            submitted: false,
        });
        Ok(Framebuffer::new(id))
    }

    fn destroy_framebuffer(&mut self, framebuffer: Framebuffer) -> Result<(), Self::Error> {
        if self
            .open
            .is_some_and(|open| open.framebuffer == framebuffer)
        {
            self.open = None;
        }
        let index = self.slot_index(framebuffer)?;
        self.buffers.swap_remove(index);
        Ok(())
    }

    fn begin(&mut self, framebuffer: Framebuffer) -> Result<Pass, Self::Error> {
        if self.open.is_some() {
            return Err(WindowsError::PassOpen);
        }
        let index = self.slot_index(framebuffer)?;
        self.buffers[index].submitted = false;
        self.open = Some(OpenPass {
            framebuffer,
            ended: false,
            blend: Blend::Over,
        });
        Ok(Pass::new(framebuffer))
    }

    fn blend(&mut self, pass: Pass, blend: Blend) -> Result<(), Self::Error> {
        let Some(open) = self.open.as_mut() else {
            return Err(WindowsError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(WindowsError::NoPass);
        }
        if open.ended {
            return Err(WindowsError::Ended);
        }
        open.blend = blend;
        Ok(())
    }

    fn clear(&mut self, pass: Pass, color: ClearColor) -> Result<(), Self::Error> {
        self.require_drawing(pass)?;
        let index = self.slot_index(pass.framebuffer())?;
        let target = self.buffers[index].view.as_ptr();
        let rgba = [
            f32::from(color.rgba[0]) / 255.0,
            f32::from(color.rgba[1]) / 255.0,
            f32::from(color.rgba[2]) / 255.0,
            f32::from(color.rgba[3]) / 255.0,
        ];
        com!(
            self.context.as_ptr(),
            50,
            fn(*mut c_void, *const f32) -> (),
            target,
            rgba.as_ptr(),
        );
        Ok(())
    }

    fn draw(&mut self, pass: Pass, source: TexelSource<'_>) -> Result<(), Self::Error> {
        self.require_drawing(pass)?;
        let index = self.slot_index(pass.framebuffer())?;
        let extent = self.buffers[index].extent;
        let drawn = match source {
            TexelSource::Platform {
                extent: source_extent,
                token,
            } => {
                if source_extent != extent {
                    return Err(WindowsError::Size);
                }
                let view = self.import_ptr(token)?;
                self.over_blend()?;
                Drawn::Borrowed(view)
            }
            TexelSource::Rgba {
                extent: source_extent,
                bytes,
            } => {
                if source_extent != extent {
                    return Err(WindowsError::Size);
                }
                let expected = extent.rgba_len().ok_or(WindowsError::Extent)?;
                if bytes.len() != expected {
                    return Err(WindowsError::Bytes);
                }
                self.over_blend()?;
                let texture = create_texture(
                    self.device.as_ptr(),
                    extent,
                    USAGE_DEFAULT,
                    BIND_SHADER,
                    0,
                    Some(bytes),
                )?;
                let view = create_view(self.device.as_ptr(), 7, texture.as_ptr())?;
                Drawn::Owned {
                    view,
                    _texture: texture,
                }
            }
        };
        let target = self.buffers[index].view.as_ptr();
        self.paint(target, extent, drawn.ptr());
        Ok(())
    }

    fn end(&mut self, pass: Pass) -> Result<(), Self::Error> {
        let Some(open) = self.open.as_mut() else {
            return Err(WindowsError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(WindowsError::NoPass);
        }
        if open.ended {
            return Err(WindowsError::Ended);
        }
        open.ended = true;
        Ok(())
    }

    fn submit(&mut self, pass: Pass) -> Result<(), Self::Error> {
        let Some(open) = self.open else {
            return Err(WindowsError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(WindowsError::NoPass);
        }
        if !open.ended {
            return Err(WindowsError::NotEnded);
        }
        let removed = com!(self.device.as_ptr(), 39, fn() -> i32,);
        if removed != 0 {
            return Err(WindowsError::GpuUnsupported);
        }
        let index = self.slot_index(pass.framebuffer())?;
        let staging = self.buffers[index].staging.as_ptr();
        let color = self.buffers[index].color.as_ptr();
        let extent = self.buffers[index].extent;
        let context = self.context.as_ptr();
        com!(
            context,
            47,
            fn(*mut c_void, *mut c_void) -> (),
            staging,
            color,
        );
        let mut mapped = Mapped {
            data: ptr::null_mut(),
            row: 0,
            depth: 0,
        };
        let hr = com!(
            context,
            14,
            fn(*mut c_void, u32, u32, u32, *mut Mapped) -> i32,
            staging,
            0,
            1,
            0,
            &mut mapped,
        );
        if hr < 0 {
            return Err(WindowsError::GpuUnsupported);
        }
        let copied = copy_rows(&self.buffers[index].rgba, extent, &mapped);
        com!(context, 15, fn(*mut c_void, u32) -> (), staging, 0);
        self.buffers[index].rgba = copied?;
        self.buffers[index].submitted = true;
        self.open = None;
        Ok(())
    }

    fn read_rgba(&self, framebuffer: Framebuffer) -> Result<&[u8], Self::Error> {
        let index = self.slot_index(framebuffer)?;
        let slot = &self.buffers[index];
        if !slot.submitted {
            return Err(WindowsError::NotSubmitted);
        }
        Ok(&slot.rgba)
    }
}

fn create_device() -> Result<(Com, Com), WindowsError> {
    let levels = [0xb000_u32, 0xa000_u32];
    for driver in [DRIVER_HARDWARE, DRIVER_WARP] {
        let mut device = ptr::null_mut();
        let mut context = ptr::null_mut();
        let mut chosen = 0_u32;
        let hr = unsafe {
            D3D11CreateDevice(
                ptr::null_mut(),
                driver,
                ptr::null_mut(),
                0,
                levels.as_ptr(),
                2,
                SDK_VERSION,
                &mut device,
                &mut chosen,
                &mut context,
            )
        };
        if hr >= 0 && !device.is_null() && !context.is_null() {
            return Ok((Com { ptr: device }, Com { ptr: context }));
        }
        unsafe {
            release(context);
            release(device);
        }
    }
    Err(WindowsError::GpuUnsupported)
}

fn compile_shader(entry: &str, target: &str) -> Result<Vec<u8>, WindowsError> {
    let entry = CString::new(entry).map_err(|_| WindowsError::GpuUnsupported)?;
    let target = CString::new(target).map_err(|_| WindowsError::GpuUnsupported)?;
    let mut code = ptr::null_mut();
    let mut errors = ptr::null_mut();
    let hr = unsafe {
        D3DCompile(
            HLSL.as_ptr().cast::<c_void>(),
            HLSL.len(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            entry.as_ptr(),
            target.as_ptr(),
            0,
            0,
            &mut code,
            &mut errors,
        )
    };
    let copied = blob_bytes(code);
    unsafe {
        release(code);
        release(errors);
    }
    if hr < 0 {
        return Err(WindowsError::GpuUnsupported);
    }
    let bytes = copied?;
    if bytes.is_empty() {
        return Err(WindowsError::GpuUnsupported);
    }
    Ok(bytes)
}

fn blob_bytes(blob: *mut c_void) -> Result<Vec<u8>, WindowsError> {
    if blob.is_null() {
        return Err(WindowsError::GpuUnsupported);
    }
    unsafe {
        let pointer: unsafe extern "system" fn(*mut c_void) -> *mut c_void =
            transmute(vslot(blob, 3));
        let size: unsafe extern "system" fn(*mut c_void) -> usize = transmute(vslot(blob, 4));
        let data = pointer(blob);
        let len = size(blob);
        if data.is_null() {
            return Err(WindowsError::GpuUnsupported);
        }
        Ok(std::slice::from_raw_parts(data.cast::<u8>(), len).to_vec())
    }
}

fn create_shader(device: *mut c_void, index: usize, bytes: &[u8]) -> Result<Com, WindowsError> {
    let mut out = ptr::null_mut();
    let hr = com!(
        device,
        index,
        fn(*const c_void, usize, *mut c_void, *mut *mut c_void) -> i32,
        bytes.as_ptr().cast::<c_void>(),
        bytes.len(),
        ptr::null_mut(),
        &mut out,
    );
    com_out(hr, out)
}

fn create_blend(device: *mut c_void) -> Result<Com, WindowsError> {
    let mut targets = [target_blend(0); 8];
    targets[0] = target_blend(1);
    let desc = BlendDesc {
        coverage: 0,
        independent: 0,
        targets,
    };
    let mut out = ptr::null_mut();
    let hr = com!(
        device,
        20,
        fn(*const BlendDesc, *mut *mut c_void) -> i32,
        &desc,
        &mut out,
    );
    com_out(hr, out)
}

fn target_blend(enable: i32) -> TargetBlend {
    let dest = if enable == 1 { 6 } else { 1 };
    TargetBlend {
        enable,
        src: 2,
        dest,
        op: 1,
        src_alpha: 2,
        dest_alpha: dest,
        op_alpha: 1,
        mask: 15,
    }
}

fn create_sampler(device: *mut c_void) -> Result<Com, WindowsError> {
    let desc = SamplerDesc {
        filter: 0,
        address_u: 3,
        address_v: 3,
        address_w: 3,
        mip_bias: 0.0,
        anisotropy: 1,
        comparison: 1,
        border: [0.0; 4],
        min_lod: 0.0,
        max_lod: 0.0,
    };
    let mut out = ptr::null_mut();
    let hr = com!(
        device,
        23,
        fn(*const SamplerDesc, *mut *mut c_void) -> i32,
        &desc,
        &mut out,
    );
    com_out(hr, out)
}

fn create_texture(
    device: *mut c_void,
    extent: Extent,
    usage: u32,
    bind: u32,
    cpu: u32,
    initial: Option<&[u8]>,
) -> Result<Com, WindowsError> {
    let pitch = extent.width.checked_mul(4).ok_or(WindowsError::Extent)?;
    let desc = TextureDesc {
        width: extent.width,
        height: extent.height,
        mips: 1,
        layers: 1,
        format: FORMAT_RGBA8,
        samples: 1,
        quality: 0,
        usage,
        bind,
        cpu,
        misc: 0,
    };
    let data = initial.map(|bytes| Subresource {
        data: bytes.as_ptr().cast::<c_void>(),
        pitch,
        slice: 0,
    });
    let mut out = ptr::null_mut();
    let hr = com!(
        device,
        5,
        fn(*const TextureDesc, *const Subresource, *mut *mut c_void) -> i32,
        &desc,
        data.as_ref().map_or(ptr::null(), ptr::from_ref),
        &mut out,
    );
    com_out(hr, out)
}

fn create_view(
    device: *mut c_void,
    index: usize,
    resource: *mut c_void,
) -> Result<Com, WindowsError> {
    let mut out = ptr::null_mut();
    let hr = com!(
        device,
        index,
        fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
        resource,
        ptr::null(),
        &mut out,
    );
    com_out(hr, out)
}

fn com_out(hr: i32, ptr: *mut c_void) -> Result<Com, WindowsError> {
    if hr < 0 || ptr.is_null() {
        unsafe { release(ptr) }
        return Err(WindowsError::GpuUnsupported);
    }
    Ok(Com { ptr })
}

fn copy_rows(current: &[u8], extent: Extent, mapped: &Mapped) -> Result<Vec<u8>, WindowsError> {
    let width = usize::try_from(extent.width).map_err(|_| WindowsError::Extent)?;
    let height = usize::try_from(extent.height).map_err(|_| WindowsError::Extent)?;
    let row_bytes = width.checked_mul(4).ok_or(WindowsError::Extent)?;
    let pitch = usize::try_from(mapped.row).map_err(|_| WindowsError::GpuUnsupported)?;
    if mapped.data.is_null() || pitch < row_bytes {
        return Err(WindowsError::GpuUnsupported);
    }
    let bytes = pitch
        .checked_mul(height)
        .ok_or(WindowsError::GpuUnsupported)?;
    let tight = row_bytes.checked_mul(height).ok_or(WindowsError::Extent)?;
    if current.len() != tight {
        return Err(WindowsError::GpuUnsupported);
    }
    let source = unsafe { std::slice::from_raw_parts(mapped.data.cast::<u8>(), bytes) };
    let mut rgba = vec![0_u8; tight];
    for row in 0..height {
        let from = row * pitch;
        let to = row * row_bytes;
        rgba[to..to + row_bytes].copy_from_slice(&source[from..from + row_bytes]);
    }
    Ok(rgba)
}

#[cfg(test)]
mod tests {
    use super::open_direct3d;
    use cutlass_gpu::{Device, Extent};

    #[test]
    #[ignore = "needs a Windows Direct3D device"]
    fn direct3d_device_opens_on_windows() {
        let mut device = open_direct3d().unwrap();
        let framebuffer = device
            .create_framebuffer(Extent {
                width: 1,
                height: 1,
            })
            .unwrap();
        device.destroy_framebuffer(framebuffer).unwrap();
    }
}
