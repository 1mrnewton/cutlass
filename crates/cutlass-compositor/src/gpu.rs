//! WGPU device ownership.
//!
//! The compositor is headless by default ([`GpuContext::new_headless_blocking`])
//! — no window, no surface — which is what export and tests want. When the
//! desktop Slint UI comes online it creates the device once and shares it via
//! [`GpuContext::from_parts`] (Slint's `unstable-wgpu-28` hands back the same
//! `wgpu` 28 `Device`/`Queue`), so a preview frame can stay on the GPU from
//! decode through present with no readback.

use crate::error::CompositorError;

/// Long-lived WGPU resources shared by the compositor (and, later, the UI).
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Create a headless GPU context (no window, no surface).
    pub async fn new_headless() -> Result<Self, CompositorError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| CompositorError::NoAdapter(e.to_string()))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("cutlass-compositor"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| CompositorError::RequestDevice(e.to_string()))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Blocking wrapper for engine startup and tests.
    pub fn new_headless_blocking() -> Result<Self, CompositorError> {
        pollster::block_on(Self::new_headless())
    }

    /// Adopt an already-created WGPU setup — the path the Slint UI uses so the
    /// compositor and the renderer share one device.
    pub fn from_parts(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }

    /// Whether the adapter is a CPU/software rasterizer (e.g. llvmpipe in CI).
    /// Exact-pixel golden comparisons aren't portable between a software
    /// rasterizer and real GPU drivers, so tests use this to relax tolerances.
    pub fn is_software(&self) -> bool {
        self.adapter.get_info().device_type == wgpu::DeviceType::Cpu
    }
}
