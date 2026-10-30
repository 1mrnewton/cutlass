//! This host does not compile Vulkan. A passing test means the backend was
//! refused. It is not a GPU run.

use cutlass_gpu::Extent;

use crate::error::AndroidError;

pub(crate) fn open() -> Result<(), AndroidError> {
    Err(AndroidError::GpuUnsupported)
}

pub(crate) fn allocate(_extent: Extent) -> Result<u64, AndroidError> {
    Err(AndroidError::GpuUnsupported)
}

pub(crate) fn upload(_token: u64, _bytes: &[u8]) -> Result<(), AndroidError> {
    Err(AndroidError::GpuUnsupported)
}

#[cfg(test)]
mod tests {
    use cutlass_gpu::Extent;

    use super::{allocate, open, upload};
    use crate::AndroidError;

    fn refused(error: AndroidError) {
        assert_eq!(error, AndroidError::GpuUnsupported);
        assert_eq!(error.to_string(), "vulkan is not compiled on this host");
    }

    #[test]
    fn open_is_refused() {
        refused(open().unwrap_err());
    }

    #[test]
    fn allocate_of_one_pixel_is_refused() {
        refused(
            allocate(Extent {
                width: 1,
                height: 1,
            })
            .unwrap_err(),
        );
    }

    #[test]
    fn upload_of_token_one_is_refused() {
        refused(upload(1, &[0, 0, 0, 0]).unwrap_err());
    }
}
