//! Where the models run, and where burn keeps its compiled GPU kernels.

use std::path::Path;
use std::sync::atomic::{AtomicU8, Ordering};

use burn::tensor::Device;

use crate::VisionError;

/// The processor a model runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compute {
    Cpu,
    Gpu,
}

impl Compute {
    /// The GPU where this build has a path and this machine's GPU is
    /// trusted ([`trust_gpu`]), and the CPU everywhere else.
    #[must_use]
    pub fn best() -> Self {
        if gpu_available() && gpu_trusted() {
            Self::Gpu
        } else {
            Self::Cpu
        }
    }
}

/// The GPU gives the models' answers.
const GPU: u8 = 1;
/// It gives them in 16-bit floats too.
const HALF: u8 = 2;

/// What this process trusts the GPU with. The Mac's Metal path is proven
/// on every Mac the models were checked on, so it starts trusted; a GPU
/// elsewhere waits for [`trust_gpu`].
static TRUSTED: AtomicU8 = AtomicU8::new(if cfg!(target_os = "macos") {
    GPU | HALF
} else {
    0
});

/// Trust the GPU as `check` found it, or not at all on `None`: what a
/// shell calls with [`crate::check_gpu`]'s answer, from a process of its
/// own. For the whole process.
pub fn trust_gpu(check: Option<crate::GpuCheck>) {
    let bits = match check {
        Some(check) if check.half => GPU | HALF,
        Some(_) => GPU,
        None => 0,
    };
    TRUSTED.store(bits, Ordering::Relaxed);
}

/// Whether the models may use the GPU.
#[must_use]
pub fn gpu_trusted() -> bool {
    TRUSTED.load(Ordering::Relaxed) & GPU != 0
}

/// Whether a model may run in 16-bit floats on the GPU.
pub(crate) fn half_trusted() -> bool {
    TRUSTED.load(Ordering::Relaxed) & HALF != 0
}

/// The floats a model that is quicker in half precision runs in on
/// `device`: half on a GPU that has them and gave the check's answers in
/// them, full everywhere else (a CPU's half is slower).
pub(crate) fn quick_float(device: &Device, compute: Compute) -> burn::tensor::FloatDType {
    use burn::tensor::{DType, FloatDType};
    if compute == Compute::Gpu && half_trusted() && device.supports_dtype(DType::F16) {
        FloatDType::F16
    } else {
        FloatDType::F32
    }
}

/// Whether this build has a GPU path: Metal on the Mac, Vulkan on Windows
/// and Linux. Phones run on the CPU: burn's GPU paths gave wrong masks on an
/// iPhone and aborted on an Android phone (`tools/ml-bench/README.md`).
/// Whether this machine's GPU works is [`gpu_trusted`].
#[must_use]
pub const fn gpu_available() -> bool {
    cfg!(any(
        target_os = "macos",
        target_os = "windows",
        target_os = "linux"
    ))
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

/// Vulkan, where burn compiles its kernels to SPIR-V. wgpu's portable path
/// (Direct3D 12 on Windows) is not used: its WGSL kernels gave wrong masks
/// on an iPhone.
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn gpu() -> Result<Device, VisionError> {
    Ok(Device::vulkan(burn::tensor::DeviceKind::DefaultDevice))
}

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
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
    #[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
    gpu_cache_root(dir);
    Ok(())
}

/// burn's GPU runtime reads its settings once, from a `cubecl.toml` in the
/// working folder or its defaults, and that read sets the cache folder. The
/// settings are handed over first, so neither a stray file nor the default
/// (`target/` beside a Cargo project) decides where kernels go. If the
/// runtime already started, the folder is switched in place.
#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
fn gpu_cache_root(dir: &Path) {
    use cubecl_environment::config::RuntimeConfig;
    use cubecl_environment::persistence::CacheConfig;
    use cubecl_runtime::config::CubeClRuntimeConfig;

    let mut config = CubeClRuntimeConfig::default();
    config.environment.path = CacheConfig::Directory(dir.to_path_buf());
    if !CubeClRuntimeConfig::try_set(config.override_from_env()) {
        cubecl_environment::environment::set_root(dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_best_compute_is_the_gpu_only_where_one_is_built_and_trusted() {
        assert_eq!(
            Compute::best() == Compute::Gpu,
            gpu_available() && gpu_trusted()
        );
    }

    #[test]
    fn the_cpu_device_always_opens() {
        assert!(device(Compute::Cpu).is_ok());
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
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
