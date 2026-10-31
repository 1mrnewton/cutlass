//! Vulkan 1.0 structures and loader entry points.
//!
//! Field order and structure-type values match `vulkan_core.h`. Dispatchable
//! handles are pointers. Non-dispatchable handles are `u64`.

#![allow(dead_code)]

use std::ffi::{c_char, c_void};

pub(super) const STRUCTURE_APPLICATION_INFO: u32 = 0;
pub(super) const STRUCTURE_INSTANCE_CREATE_INFO: u32 = 1;
pub(super) const STRUCTURE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
pub(super) const STRUCTURE_DEVICE_CREATE_INFO: u32 = 3;
pub(super) const STRUCTURE_SUBMIT_INFO: u32 = 4;
pub(super) const STRUCTURE_MEMORY_ALLOCATE_INFO: u32 = 5;
pub(super) const STRUCTURE_BUFFER_CREATE_INFO: u32 = 12;
pub(super) const STRUCTURE_IMAGE_CREATE_INFO: u32 = 14;
pub(super) const STRUCTURE_COMMAND_POOL_CREATE_INFO: u32 = 39;
pub(super) const STRUCTURE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
pub(super) const STRUCTURE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
pub(super) const STRUCTURE_BUFFER_MEMORY_BARRIER: u32 = 44;
pub(super) const STRUCTURE_IMAGE_MEMORY_BARRIER: u32 = 45;

pub(super) const API_VERSION_1_0: u32 = 4_194_304;
pub(super) const QUEUE_GRAPHICS: u32 = 1;
pub(super) const FORMAT_RGBA8_UNORM: u32 = 37;
pub(super) const IMAGE_TYPE_2D: u32 = 1;
pub(super) const SAMPLE_COUNT_1: u32 = 1;
pub(super) const TILING_OPTIMAL: u32 = 0;
pub(super) const IMAGE_TRANSFER: u32 = 3;
pub(super) const SHARING_EXCLUSIVE: u32 = 0;
pub(super) const LAYOUT_UNDEFINED: u32 = 0;
pub(super) const LAYOUT_TRANSFER_SRC: u32 = 6;
pub(super) const LAYOUT_TRANSFER_DST: u32 = 7;
pub(super) const BUFFER_TRANSFER_SRC: u32 = 1;
pub(super) const BUFFER_TRANSFER_DST: u32 = 2;
pub(super) const MEMORY_DEVICE_LOCAL: u32 = 1;
pub(super) const MEMORY_HOST_VISIBLE_COHERENT: u32 = 6;
pub(super) const STAGE_TOP_OF_PIPE: u32 = 1;
pub(super) const STAGE_TRANSFER: u32 = 0x1000;
pub(super) const STAGE_HOST: u32 = 0x4000;
pub(super) const ACCESS_TRANSFER_READ: u32 = 0x800;
pub(super) const ACCESS_TRANSFER_WRITE: u32 = 0x1000;
pub(super) const ACCESS_HOST_READ: u32 = 0x2000;
pub(super) const ASPECT_COLOR: u32 = 1;
pub(super) const POOL_RESET: u32 = 0x2;
pub(super) const COMMAND_ONE_TIME: u32 = 1;
pub(super) const LEVEL_PRIMARY: u32 = 0;

pub(super) const MEMORY_TYPE_LIMIT: usize = 32;
pub(super) const MEMORY_HEAP_LIMIT: usize = 16;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct Extent3D {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) depth: u32,
}

impl Extent3D {
    pub(super) const fn empty() -> Self {
        Self {
            width: 0,
            height: 0,
            depth: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct Offset3D {
    pub(super) x: i32,
    pub(super) y: i32,
    pub(super) z: i32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct ApplicationInfo {
    pub(super) s_type: u32,
    pub(super) next: *const c_void,
    pub(super) application_name: *const c_char,
    pub(super) application_version: u32,
    pub(super) engine_name: *const c_char,
    pub(super) engine_version: u32,
    pub(super) api_version: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct InstanceCreateInfo {
    pub(super) s_type: u32,
    pub(super) next: *const c_void,
    pub(super) flags: u32,
    pub(super) application_info: *const ApplicationInfo,
    pub(super) enabled_layer_count: u32,
    pub(super) enabled_layer_names: *const *const c_char,
    pub(super) enabled_extension_count: u32,
    pub(super) enabled_extension_names: *const *const c_char,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct MemoryHeap {
    pub(super) size: u64,
    pub(super) flags: u32,
}

impl MemoryHeap {
    const fn empty() -> Self {
        Self { size: 0, flags: 0 }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct MemoryType {
    pub(super) property_flags: u32,
    pub(super) heap_index: u32,
}

impl MemoryType {
    const fn empty() -> Self {
        Self {
            property_flags: 0,
            heap_index: 0,
        }
    }
}

#[derive(Debug)]
#[repr(C)]
pub(super) struct MemoryProperties {
    pub(super) memory_type_count: u32,
    pub(super) memory_types: [MemoryType; MEMORY_TYPE_LIMIT],
    pub(super) memory_heap_count: u32,
    pub(super) memory_heaps: [MemoryHeap; MEMORY_HEAP_LIMIT],
}

impl MemoryProperties {
    pub(super) const fn empty() -> Self {
        Self {
            memory_type_count: 0,
            memory_types: [MemoryType::empty(); MEMORY_TYPE_LIMIT],
            memory_heap_count: 0,
            memory_heaps: [MemoryHeap::empty(); MEMORY_HEAP_LIMIT],
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct QueueFamilyProperties {
    pub(super) queue_flags: u32,
    pub(super) queue_count: u32,
    pub(super) timestamp_valid_bits: u32,
    pub(super) min_image_transfer_granularity: Extent3D,
}

impl QueueFamilyProperties {
    pub(super) const fn empty() -> Self {
        Self {
            queue_flags: 0,
            queue_count: 0,
            timestamp_valid_bits: 0,
            min_image_transfer_granularity: Extent3D::empty(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct DeviceQueueCreateInfo {
    pub(super) s_type: u32,
    pub(super) next: *const c_void,
    pub(super) flags: u32,
    pub(super) queue_family_index: u32,
    pub(super) queue_count: u32,
    pub(super) queue_priorities: *const f32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct DeviceCreateInfo {
    pub(super) s_type: u32,
    pub(super) next: *const c_void,
    pub(super) flags: u32,
    pub(super) queue_create_info_count: u32,
    pub(super) queue_create_infos: *const DeviceQueueCreateInfo,
    pub(super) enabled_layer_count: u32,
    pub(super) enabled_layer_names: *const *const c_char,
    pub(super) enabled_extension_count: u32,
    pub(super) enabled_extension_names: *const *const c_char,
    pub(super) enabled_features: *const c_void,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct SubmitInfo {
    pub(super) s_type: u32,
    pub(super) next: *const c_void,
    pub(super) wait_semaphore_count: u32,
    pub(super) wait_semaphores: *const u64,
    pub(super) wait_dst_stage_mask: *const u32,
    pub(super) command_buffer_count: u32,
    pub(super) command_buffers: *const *mut c_void,
    pub(super) signal_semaphore_count: u32,
    pub(super) signal_semaphores: *const u64,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct MemoryAllocateInfo {
    pub(super) s_type: u32,
    pub(super) next: *const c_void,
    pub(super) allocation_size: u64,
    pub(super) memory_type_index: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct MemoryRequirements {
    pub(super) size: u64,
    pub(super) alignment: u64,
    pub(super) memory_type_bits: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct BufferCreateInfo {
    pub(super) s_type: u32,
    pub(super) next: *const c_void,
    pub(super) flags: u32,
    pub(super) size: u64,
    pub(super) usage: u32,
    pub(super) sharing_mode: u32,
    pub(super) queue_family_index_count: u32,
    pub(super) queue_family_indices: *const u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct ImageCreateInfo {
    pub(super) s_type: u32,
    pub(super) next: *const c_void,
    pub(super) flags: u32,
    pub(super) image_type: u32,
    pub(super) format: u32,
    pub(super) extent: Extent3D,
    pub(super) mip_levels: u32,
    pub(super) array_layers: u32,
    pub(super) samples: u32,
    pub(super) tiling: u32,
    pub(super) usage: u32,
    pub(super) sharing_mode: u32,
    pub(super) queue_family_index_count: u32,
    pub(super) queue_family_indices: *const u32,
    pub(super) initial_layout: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct CommandPoolCreateInfo {
    pub(super) s_type: u32,
    pub(super) next: *const c_void,
    pub(super) flags: u32,
    pub(super) queue_family_index: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct CommandBufferAllocateInfo {
    pub(super) s_type: u32,
    pub(super) next: *const c_void,
    pub(super) command_pool: u64,
    pub(super) level: u32,
    pub(super) command_buffer_count: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct CommandBufferBeginInfo {
    pub(super) s_type: u32,
    pub(super) next: *const c_void,
    pub(super) flags: u32,
    pub(super) inheritance: *const c_void,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct ImageSubresourceRange {
    pub(super) aspect_mask: u32,
    pub(super) base_mip_level: u32,
    pub(super) level_count: u32,
    pub(super) base_array_layer: u32,
    pub(super) layer_count: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct ImageMemoryBarrier {
    pub(super) s_type: u32,
    pub(super) next: *const c_void,
    pub(super) src_access_mask: u32,
    pub(super) dst_access_mask: u32,
    pub(super) old_layout: u32,
    pub(super) new_layout: u32,
    pub(super) src_queue_family_index: u32,
    pub(super) dst_queue_family_index: u32,
    pub(super) image: u64,
    pub(super) subresource_range: ImageSubresourceRange,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct BufferMemoryBarrier {
    pub(super) s_type: u32,
    pub(super) next: *const c_void,
    pub(super) src_access_mask: u32,
    pub(super) dst_access_mask: u32,
    pub(super) src_queue_family_index: u32,
    pub(super) dst_queue_family_index: u32,
    pub(super) buffer: u64,
    pub(super) offset: u64,
    pub(super) size: u64,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct ImageSubresourceLayers {
    pub(super) aspect_mask: u32,
    pub(super) mip_level: u32,
    pub(super) base_array_layer: u32,
    pub(super) layer_count: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct BufferImageCopy {
    pub(super) buffer_offset: u64,
    pub(super) buffer_row_length: u32,
    pub(super) buffer_image_height: u32,
    pub(super) image_subresource: ImageSubresourceLayers,
    pub(super) image_offset: Offset3D,
    pub(super) image_extent: Extent3D,
}

#[link(name = "vulkan")]
unsafe extern "system" {
    pub(super) fn vkCreateInstance(
        create_info: *const InstanceCreateInfo,
        allocator: *const c_void,
        instance: *mut *mut c_void,
    ) -> i32;

    pub(super) fn vkDestroyInstance(instance: *mut c_void, allocator: *const c_void);

    pub(super) fn vkEnumeratePhysicalDevices(
        instance: *mut c_void,
        count: *mut u32,
        devices: *mut *mut c_void,
    ) -> i32;

    pub(super) fn vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device: *mut c_void,
        count: *mut u32,
        families: *mut QueueFamilyProperties,
    );

    pub(super) fn vkGetPhysicalDeviceMemoryProperties(
        physical_device: *mut c_void,
        properties: *mut MemoryProperties,
    );

    pub(super) fn vkCreateDevice(
        physical_device: *mut c_void,
        create_info: *const DeviceCreateInfo,
        allocator: *const c_void,
        device: *mut *mut c_void,
    ) -> i32;

    pub(super) fn vkDestroyDevice(device: *mut c_void, allocator: *const c_void);

    pub(super) fn vkGetDeviceQueue(
        device: *mut c_void,
        queue_family_index: u32,
        queue_index: u32,
        queue: *mut *mut c_void,
    );

    pub(super) fn vkCreateCommandPool(
        device: *mut c_void,
        create_info: *const CommandPoolCreateInfo,
        allocator: *const c_void,
        command_pool: *mut u64,
    ) -> i32;

    pub(super) fn vkDestroyCommandPool(
        device: *mut c_void,
        command_pool: u64,
        allocator: *const c_void,
    );

    pub(super) fn vkAllocateCommandBuffers(
        device: *mut c_void,
        allocate_info: *const CommandBufferAllocateInfo,
        command_buffers: *mut *mut c_void,
    ) -> i32;

    pub(super) fn vkFreeCommandBuffers(
        device: *mut c_void,
        command_pool: u64,
        command_buffer_count: u32,
        command_buffers: *const *mut c_void,
    );

    pub(super) fn vkBeginCommandBuffer(
        command_buffer: *mut c_void,
        begin_info: *const CommandBufferBeginInfo,
    ) -> i32;

    pub(super) fn vkEndCommandBuffer(command_buffer: *mut c_void) -> i32;

    pub(super) fn vkCmdPipelineBarrier(
        command: *mut c_void,
        src_stage: u32,
        dst_stage: u32,
        dependency_flags: u32,
        memory_barrier_count: u32,
        memory_barriers: *const c_void,
        buffer_barrier_count: u32,
        buffer_barriers: *const BufferMemoryBarrier,
        image_barrier_count: u32,
        image_barriers: *const ImageMemoryBarrier,
    );

    pub(super) fn vkCmdCopyBufferToImage(
        command: *mut c_void,
        buffer: u64,
        image: u64,
        dst_image_layout: u32,
        region_count: u32,
        regions: *const BufferImageCopy,
    );

    pub(super) fn vkCmdCopyImageToBuffer(
        command: *mut c_void,
        image: u64,
        src_image_layout: u32,
        buffer: u64,
        region_count: u32,
        regions: *const BufferImageCopy,
    );

    pub(super) fn vkQueueSubmit(
        queue: *mut c_void,
        submit_count: u32,
        submits: *const SubmitInfo,
        fence: u64,
    ) -> i32;

    pub(super) fn vkQueueWaitIdle(queue: *mut c_void) -> i32;

    pub(super) fn vkCreateBuffer(
        device: *mut c_void,
        create_info: *const BufferCreateInfo,
        allocator: *const c_void,
        buffer: *mut u64,
    ) -> i32;

    pub(super) fn vkDestroyBuffer(device: *mut c_void, buffer: u64, allocator: *const c_void);

    pub(super) fn vkGetBufferMemoryRequirements(
        device: *mut c_void,
        buffer: u64,
        requirements: *mut MemoryRequirements,
    );

    pub(super) fn vkBindBufferMemory(
        device: *mut c_void,
        buffer: u64,
        memory: u64,
        offset: u64,
    ) -> i32;

    pub(super) fn vkCreateImage(
        device: *mut c_void,
        create_info: *const ImageCreateInfo,
        allocator: *const c_void,
        image: *mut u64,
    ) -> i32;

    pub(super) fn vkDestroyImage(device: *mut c_void, image: u64, allocator: *const c_void);

    pub(super) fn vkGetImageMemoryRequirements(
        device: *mut c_void,
        image: u64,
        requirements: *mut MemoryRequirements,
    );

    pub(super) fn vkBindImageMemory(
        device: *mut c_void,
        image: u64,
        memory: u64,
        offset: u64,
    ) -> i32;

    pub(super) fn vkAllocateMemory(
        device: *mut c_void,
        allocate_info: *const MemoryAllocateInfo,
        allocator: *const c_void,
        memory: *mut u64,
    ) -> i32;

    pub(super) fn vkFreeMemory(device: *mut c_void, memory: u64, allocator: *const c_void);

    pub(super) fn vkMapMemory(
        device: *mut c_void,
        memory: u64,
        offset: u64,
        size: u64,
        flags: u32,
        data: *mut *mut c_void,
    ) -> i32;

    pub(super) fn vkUnmapMemory(device: *mut c_void, memory: u64);
}
