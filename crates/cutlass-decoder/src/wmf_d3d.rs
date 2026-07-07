//! Process-shared Direct3D 11 device for hardware-accelerated Media
//! Foundation decode.
//!
//! Handing the Source Reader an [`IMFDXGIDeviceManager`]
//! (`MF_SOURCE_READER_D3D_MANAGER`) plus
//! `MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS` is what moves decoding off the
//! CPU: without them the reader loads Microsoft's *software* decoder MFTs,
//! which is where a 4K timeline spends all its time. With them, the reader
//! picks the GPU vendor's hardware decoder (DXVA) and decodes into DXGI
//! buffers — which the CPU path reads back through the existing stride-aware
//! [`IMF2DBuffer`](windows::Win32::Media::MediaFoundation::IMF2DBuffer) lock,
//! and the zero-copy path shares across to the renderer as textures.
//!
//! One device serves the whole process (players do the same for multi-stream
//! playback): decoder MFTs allocate their big texture arrays per *decoder*,
//! not per device, so sharing costs nothing and saves a device + driver
//! context per open. Thread safety is the documented MF requirement —
//! multithread protection on the immediate context — plus the device manager
//! being free-threaded. Device creation failing (VM without a GPU, RDP
//! session, driver trouble) simply yields `None` and callers stay on the
//! software path.

use std::sync::OnceLock;

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CREATE_DEVICE_VIDEO_SUPPORT, D3D11_SDK_VERSION, D3D11CreateDevice, ID3D11Device,
    ID3D11DeviceContext, ID3D11Multithread,
};
use windows::Win32::Media::MediaFoundation::{IMFDXGIDeviceManager, MFCreateDXGIDeviceManager};
use windows::core::Interface;

/// The process's decode device: a D3D11 hardware device wrapped in the DXGI
/// device manager Media Foundation consumes.
pub(crate) struct D3dShared {
    pub(crate) device: ID3D11Device,
    /// Immediate context, multithread-protected — safe to use from any thread
    /// (calls are serialized by the D3D runtime, including against MF's own).
    pub(crate) context: ID3D11DeviceContext,
    pub(crate) manager: IMFDXGIDeviceManager,
}

// SAFETY: `ID3D11Device` and `IMFDXGIDeviceManager` are documented
// free-threaded. The immediate context is only safe because multithread
// protection is enabled at creation (the documented requirement for sharing a
// device with Media Foundation) — the runtime takes a lock around each call.
unsafe impl Send for D3dShared {}
unsafe impl Sync for D3dShared {}

/// The shared decode device, created on first use. `None` when this machine
/// can't create a hardware D3D11 device — callers fall back to software MFTs.
pub(crate) fn shared() -> Option<&'static D3dShared> {
    static SHARED: OnceLock<Option<D3dShared>> = OnceLock::new();
    SHARED
        .get_or_init(|| match create() {
            Ok(shared) => Some(shared),
            Err(error) => {
                tracing::warn!(
                    %error,
                    "no D3D11 decode device; Media Foundation will decode in software"
                );
                None
            }
        })
        .as_ref()
}

fn create() -> Result<D3dShared, windows::core::Error> {
    let mut device: Option<ID3D11Device> = None;
    let mut context: Option<ID3D11DeviceContext> = None;
    unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
    }?;
    let device = device.expect("D3D11CreateDevice succeeded");
    let context = context.expect("D3D11CreateDevice succeeded");

    // Required for MF: the reader's decode workers and our callers touch the
    // device concurrently. The return value is the previous protection state,
    // which is not interesting here.
    let multithread: ID3D11Multithread = context.cast()?;
    let _ = unsafe { multithread.SetMultithreadProtected(true) };

    let mut token = 0u32;
    let mut manager: Option<IMFDXGIDeviceManager> = None;
    unsafe { MFCreateDXGIDeviceManager(&mut token, &mut manager) }?;
    let manager = manager.expect("MFCreateDXGIDeviceManager succeeded");
    unsafe { manager.ResetDevice(&device, token) }?;

    Ok(D3dShared {
        device,
        context,
        manager,
    })
}
