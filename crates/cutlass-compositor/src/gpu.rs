//! WGPU device ownership. Headless by default; share with Slint via
//! [`GpuContext::from_parts`] when the UI creates the context once.

use crate::error::CompositorError;

/// Long-lived WGPU resources shared by the compositor (and future Slint UI).
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Create a headless GPU context (no window, no Slint).
    pub async fn new_headless() -> Result<Self, CompositorError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(CompositorError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("cutlass-compositor"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
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

    /// Adopt an existing WGPU setup (future Slint `WGPUConfiguration::Manual` path).
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
}
