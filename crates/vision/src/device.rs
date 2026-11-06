//! Where the models run, and where burn keeps its compiled GPU kernels.

use std::path::Path;

use burn::tensor::Device;

use crate::VisionError;

/// The processor a model runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compute {
    Cpu,
    Gpu,
}

impl Compute {
    /// The GPU where this build has a path proven to give correct masks, and
    /// the CPU everywhere else.
    #[must_use]
    pub const fn best() -> Self {
        if gpu_available() {
            Self::Gpu
        } else {
            Self::Cpu
        }
    }
}

/// Whether this build has a GPU path. Today that is the Mac: burn's phone
/// GPU paths gave wrong masks or did not run, and Windows and Linux are not
/// measured yet.
#[must_use]
pub const fn gpu_available() -> bool {
    cfg!(target_os = "macos")
}

pub(crate) fn device(compute: Compute) -> Result<Device, VisionError> {
    match compute {
        Compute::Cpu => Ok(Device::flex()),
        Compute::Gpu => gpu(),
    }
}

#[cfg(target_os = "macos")]
fn gpu() -> Result<Device, VisionError> {
    Ok(Device::metal(burn::tensor::DeviceKind::DefaultDevice))
}

#[cfg(not(target_os = "macos"))]
fn gpu() -> Result<Device, VisionError> {
    Err(VisionError::NoGpu)
}

/// Keeps burn's compiled GPU kernels in `dir`, so a later launch skips most
/// of the warm-up. Call it once, before the first GPU model loads. The
/// setting is for the whole process. A build with no GPU path only creates
/// the folder.
pub fn keep_kernels_in(dir: &Path) -> Result<(), VisionError> {
    std::fs::create_dir_all(dir).map_err(|error| VisionError::Cache {
        path: dir.to_path_buf(),
        message: error.to_string(),
    })?;
    #[cfg(target_os = "macos")]
    cubecl_environment::environment::set_root(dir);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_best_compute_is_the_gpu_only_where_one_is_proven() {
        assert_eq!(Compute::best() == Compute::Gpu, gpu_available());
    }

    #[test]
    fn the_cpu_device_always_opens() {
        assert!(device(Compute::Cpu).is_ok());
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_build_without_a_gpu_path_says_so() {
        assert_eq!(device(Compute::Gpu).err(), Some(VisionError::NoGpu));
    }

    #[test]
    fn the_kernel_folder_is_created() {
        let dir =
            std::env::temp_dir().join(format!("cutlass-vision-kernels-{}", std::process::id()));
        let nested = dir.join("Cache").join("kernels");
        keep_kernels_in(&nested).unwrap();
        assert!(nested.is_dir());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_kernel_folder_under_a_file_is_an_error() {
        let file = std::env::temp_dir().join(format!("cutlass-vision-file-{}", std::process::id()));
        std::fs::write(&file, b"not a folder").unwrap();
        assert!(matches!(
            keep_kernels_in(&file.join("kernels")),
            Err(VisionError::Cache { .. })
        ));
        std::fs::remove_file(&file).unwrap();
    }
}
