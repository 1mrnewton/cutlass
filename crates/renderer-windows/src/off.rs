//! Direct3D is not available on this host.

/// No Direct3D device exists on this host.
#[derive(Debug)]
pub struct Direct3DDevice;

/// Direct3D is not available on this host.
pub fn open_direct3d() -> Result<Direct3DDevice, crate::WindowsError> {
    Err(crate::WindowsError::GpuUnsupported)
}

#[cfg(test)]
mod tests {
    use crate::WindowsError;

    #[test]
    fn open_direct3d_is_unsupported_on_this_host() {
        let error = super::open_direct3d().unwrap_err();
        assert_eq!(error, WindowsError::GpuUnsupported);
        assert_eq!(error.to_string(), "Direct3D is not available on this host");
    }
}
