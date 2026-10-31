//! Device, image allocation, upload, and readback.
//!
//! Failures while opening drop the session, which destroys any command pool,
//! device, and instance already created. Upload and readback free the staging
//! buffer, its memory, and the command buffer on every path.

use std::ffi::{c_char, c_void};
use std::ptr;

use cutlass_gpu::Extent;

use super::sys;
use crate::error::AndroidError;

const MAX_IMAGES: usize = 8;
const MAX_PHYSICAL_DEVICES: usize = 8;
const MAX_QUEUE_FAMILIES: usize = 16;
const CUTLASS_NAME: &[u8] = b"cutlass\0";

#[derive(Clone, Copy, Debug)]
struct Image {
    token: u64,
    memory: u64,
    extent: Extent,
    layout: u32,
}

impl Image {
    const fn empty() -> Self {
        Self {
            token: 0,
            memory: 0,
            extent: Extent {
                width: 0,
                height: 0,
            },
            layout: 0,
        }
    }
}

/// One Vulkan instance, device, and the images it owns.
#[derive(Debug)]
pub(crate) struct Session {
    instance: *mut c_void,
    device: *mut c_void,
    queue: *mut c_void,
    pool: u64,
    memory: sys::MemoryProperties,
    images: [Image; MAX_IMAGES],
}

impl Session {
    fn bare(instance: *mut c_void) -> Self {
        Self {
            instance,
            device: ptr::null_mut(),
            queue: ptr::null_mut(),
            pool: 0,
            memory: sys::MemoryProperties::empty(),
            images: [Image::empty(); MAX_IMAGES],
        }
    }

    /// Allocate one `R8G8B8A8` image and return its token.
    pub(crate) fn allocate_rgba(&mut self, extent: Extent) -> Result<u64, AndroidError> {
        require_extent(extent)?;
        if self.device.is_null() || self.images.iter().all(|slot| slot.token != 0) {
            return Err(AndroidError::Device);
        }
        let image = create_bound_image(self, extent)?;
        self.store_image(image)
    }

    /// Copy tightly packed RGBA bytes into an image.
    pub(crate) fn upload_rgba(&mut self, token: u64, bytes: &[u8]) -> Result<(), AndroidError> {
        let image = self.lookup(token)?;
        let length = rgba_len(image.extent, AndroidError::Bytes)?;
        if bytes.len() != length {
            return Err(AndroidError::Bytes);
        }
        with_transfer(
            self,
            length,
            sys::BUFFER_TRANSFER_SRC,
            |session, staging, command| {
                write_mapped(session, staging, bytes)?;
                record_upload(session, command, image, staging)?;
                session.mark_layout(token, sys::LAYOUT_TRANSFER_DST);
                Ok(())
            },
        )?;
        Ok(())
    }

    fn lookup(&self, token: u64) -> Result<Image, AndroidError> {
        if token == 0 {
            return Err(AndroidError::Image);
        }
        self.images
            .iter()
            .find(|image| image.token == token)
            .copied()
            .ok_or(AndroidError::Image)
    }

    fn store_image(&mut self, image: Image) -> Result<u64, AndroidError> {
        if image.token == 0 || self.images.iter().any(|slot| slot.token == image.token) {
            self.free_image(image);
            return Err(AndroidError::Device);
        }
        let Some(slot) = self.images.iter_mut().find(|slot| slot.token == 0) else {
            self.free_image(image);
            return Err(AndroidError::Device);
        };
        let token = image.token;
        *slot = image;
        Ok(token)
    }

    fn free_image(&self, image: Image) {
        destroy_image(self.device, image.token);
        free_memory(self.device, image.memory);
    }

    fn mark_layout(&mut self, token: u64, layout: u32) {
        if let Some(slot) = self.images.iter_mut().find(|slot| slot.token == token) {
            slot.layout = layout;
        }
    }

    fn bring_up(&mut self) -> Result<(), AndroidError> {
        let (devices, count) = enumerate(self.instance)?;
        let Some(list) = devices.get(..count) else {
            return Err(AndroidError::Device);
        };
        let Some((physical, family)) = first_graphics(list)? else {
            return Err(AndroidError::Device);
        };
        self.memory = read_memory(physical)?;
        self.device = create_device(physical, family)?;
        self.queue = get_queue(self.device, family)?;
        self.pool = create_pool(self.device, family)?;
        Ok(())
    }

    fn release(&mut self) {
        self.destroy_images();
        self.destroy_pool();
        self.destroy_device();
        self.destroy_instance();
        self.memory = sys::MemoryProperties::empty();
    }

    fn destroy_images(&mut self) {
        let device = self.device;
        for image in &mut self.images {
            if !device.is_null() {
                destroy_image(device, image.token);
                free_memory(device, image.memory);
            }
            *image = Image::empty();
        }
    }

    fn destroy_pool(&mut self) {
        if !self.device.is_null() && self.pool != 0 {
            unsafe {
                sys::vkDestroyCommandPool(self.device, self.pool, ptr::null());
            }
        }
        self.pool = 0;
    }

    fn destroy_device(&mut self) {
        if !self.device.is_null() {
            unsafe {
                sys::vkDestroyDevice(self.device, ptr::null());
            }
        }
        self.device = ptr::null_mut();
        self.queue = ptr::null_mut();
    }

    fn destroy_instance(&mut self) {
        if !self.instance.is_null() {
            unsafe {
                sys::vkDestroyInstance(self.instance, ptr::null());
            }
        }
        self.instance = ptr::null_mut();
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.release();
    }
}

/// Open a Vulkan 1.0 instance and one graphics device.
pub(crate) fn open() -> Result<Session, AndroidError> {
    let instance = create_instance()?;
    let mut session = Session::bare(instance);
    session.bring_up()?;
    Ok(session)
}

/// Copy one image back to tightly packed RGBA bytes.
pub(crate) fn sample_platform(
    session: &mut Session,
    extent: Extent,
    token: u64,
) -> Result<Vec<u8>, AndroidError> {
    let image = session.lookup(token)?;
    if image.extent != extent {
        return Err(AndroidError::Size);
    }
    let length = rgba_len(image.extent, AndroidError::Device)?;
    with_transfer(
        session,
        length,
        sys::BUFFER_TRANSFER_DST,
        |device, staging, command| {
            record_sample(device, command, image, staging, length)?;
            device.mark_layout(token, sys::LAYOUT_TRANSFER_SRC);
            read_mapped(device, staging, length)
        },
    )
}

fn require_extent(extent: Extent) -> Result<(), AndroidError> {
    if extent.width == 0 || extent.height == 0 || extent.rgba_len().is_none() {
        return Err(AndroidError::Extent);
    }
    Ok(())
}

fn rgba_len(extent: Extent, error: AndroidError) -> Result<usize, AndroidError> {
    extent.rgba_len().ok_or(error)
}

fn device_ok(code: i32) -> Result<(), AndroidError> {
    if code == 0 {
        Ok(())
    } else {
        Err(AndroidError::Device)
    }
}

fn cutlass_name() -> *const c_char {
    CUTLASS_NAME.as_ptr().cast()
}

fn create_instance() -> Result<*mut c_void, AndroidError> {
    let name = cutlass_name();
    let application = sys::ApplicationInfo {
        s_type: sys::STRUCTURE_APPLICATION_INFO,
        next: ptr::null(),
        application_name: name,
        application_version: 0,
        engine_name: name,
        engine_version: 0,
        api_version: sys::API_VERSION_1_0,
    };
    let info = sys::InstanceCreateInfo {
        s_type: sys::STRUCTURE_INSTANCE_CREATE_INFO,
        next: ptr::null(),
        flags: 0,
        application_info: ptr::from_ref(&application),
        enabled_layer_count: 0,
        enabled_layer_names: ptr::null(),
        enabled_extension_count: 0,
        enabled_extension_names: ptr::null(),
    };
    let mut instance = ptr::null_mut();
    let code = unsafe { sys::vkCreateInstance(&info, ptr::null(), &mut instance) };
    if code != 0 || instance.is_null() {
        return Err(AndroidError::Device);
    }
    Ok(instance)
}

fn enumerate(
    instance: *mut c_void,
) -> Result<([*mut c_void; MAX_PHYSICAL_DEVICES], usize), AndroidError> {
    let mut count = 0_u32;
    let code = unsafe { sys::vkEnumeratePhysicalDevices(instance, &mut count, ptr::null_mut()) };
    device_ok(code)?;
    if bounded_count(count, MAX_PHYSICAL_DEVICES)?.is_none() {
        return Err(AndroidError::Device);
    }
    let mut devices = [ptr::null_mut(); MAX_PHYSICAL_DEVICES];
    count = capacity(&devices)?;
    let code =
        unsafe { sys::vkEnumeratePhysicalDevices(instance, &mut count, devices.as_mut_ptr()) };
    device_ok(code)?;
    let Some(limit) = bounded_count(count, MAX_PHYSICAL_DEVICES)? else {
        return Err(AndroidError::Device);
    };
    Ok((devices, limit))
}

fn bounded_count(count: u32, limit: usize) -> Result<Option<usize>, AndroidError> {
    let value = usize::try_from(count).map_err(|_| AndroidError::Device)?;
    if value == 0 || value > limit {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

fn capacity<T>(values: &[T]) -> Result<u32, AndroidError> {
    u32::try_from(values.len()).map_err(|_| AndroidError::Device)
}

fn first_graphics(devices: &[*mut c_void]) -> Result<Option<(*mut c_void, u32)>, AndroidError> {
    for physical in devices.iter().copied() {
        if physical.is_null() {
            continue;
        }
        if let Some(family) = graphics_family(physical)? {
            return Ok(Some((physical, family)));
        }
    }
    Ok(None)
}

fn graphics_family(physical: *mut c_void) -> Result<Option<u32>, AndroidError> {
    let mut count = 0_u32;
    unsafe {
        sys::vkGetPhysicalDeviceQueueFamilyProperties(physical, &mut count, ptr::null_mut());
    }
    if bounded_count(count, MAX_QUEUE_FAMILIES)?.is_none() {
        return Ok(None);
    }
    let mut families = [sys::QueueFamilyProperties::empty(); MAX_QUEUE_FAMILIES];
    count = u32::try_from(families.len()).map_err(|_| AndroidError::Device)?;
    unsafe {
        sys::vkGetPhysicalDeviceQueueFamilyProperties(physical, &mut count, families.as_mut_ptr());
    }
    let Some(limit) = bounded_count(count, families.len())? else {
        return Ok(None);
    };
    let Some(position) = families
        .iter()
        .take(limit)
        .position(|family| (family.queue_flags & sys::QUEUE_GRAPHICS) != 0)
    else {
        return Ok(None);
    };
    Ok(Some(
        u32::try_from(position).map_err(|_| AndroidError::Device)?,
    ))
}

fn read_memory(physical: *mut c_void) -> Result<sys::MemoryProperties, AndroidError> {
    let mut properties = sys::MemoryProperties::empty();
    unsafe {
        sys::vkGetPhysicalDeviceMemoryProperties(physical, &mut properties);
    }
    let types = usize::try_from(properties.memory_type_count).map_err(|_| AndroidError::Device)?;
    let heaps = usize::try_from(properties.memory_heap_count).map_err(|_| AndroidError::Device)?;
    if types > sys::MEMORY_TYPE_LIMIT || heaps > sys::MEMORY_HEAP_LIMIT {
        return Err(AndroidError::Device);
    }
    Ok(properties)
}

fn create_device(physical: *mut c_void, family: u32) -> Result<*mut c_void, AndroidError> {
    let priority = 1.0_f32;
    let queue = sys::DeviceQueueCreateInfo {
        s_type: sys::STRUCTURE_DEVICE_QUEUE_CREATE_INFO,
        next: ptr::null(),
        flags: 0,
        queue_family_index: family,
        queue_count: 1,
        queue_priorities: ptr::from_ref(&priority),
    };
    let info = sys::DeviceCreateInfo {
        s_type: sys::STRUCTURE_DEVICE_CREATE_INFO,
        next: ptr::null(),
        flags: 0,
        queue_create_info_count: 1,
        queue_create_infos: ptr::from_ref(&queue),
        enabled_layer_count: 0,
        enabled_layer_names: ptr::null(),
        enabled_extension_count: 0,
        enabled_extension_names: ptr::null(),
        enabled_features: ptr::null(),
    };
    let mut device = ptr::null_mut();
    let code = unsafe { sys::vkCreateDevice(physical, &info, ptr::null(), &mut device) };
    if code != 0 || device.is_null() {
        return Err(AndroidError::Device);
    }
    Ok(device)
}

fn get_queue(device: *mut c_void, family: u32) -> Result<*mut c_void, AndroidError> {
    let mut queue = ptr::null_mut();
    unsafe {
        sys::vkGetDeviceQueue(device, family, 0, &mut queue);
    }
    if queue.is_null() {
        Err(AndroidError::Device)
    } else {
        Ok(queue)
    }
}

fn create_pool(device: *mut c_void, family: u32) -> Result<u64, AndroidError> {
    let info = sys::CommandPoolCreateInfo {
        s_type: sys::STRUCTURE_COMMAND_POOL_CREATE_INFO,
        next: ptr::null(),
        flags: sys::POOL_RESET,
        queue_family_index: family,
    };
    let mut pool = 0_u64;
    let code = unsafe { sys::vkCreateCommandPool(device, &info, ptr::null(), &mut pool) };
    if code != 0 || pool == 0 {
        return Err(AndroidError::Device);
    }
    Ok(pool)
}

fn create_bound_image(session: &Session, extent: Extent) -> Result<Image, AndroidError> {
    if session.device.is_null() {
        return Err(AndroidError::Device);
    }
    let token = create_image(session.device, extent)?;
    match bind_new_image(session, token) {
        Ok(memory) => Ok(Image {
            token,
            memory,
            extent,
            layout: sys::LAYOUT_UNDEFINED,
        }),
        Err(error) => {
            destroy_image(session.device, token);
            Err(error)
        }
    }
}

fn create_image(device: *mut c_void, extent: Extent) -> Result<u64, AndroidError> {
    let info = sys::ImageCreateInfo {
        s_type: sys::STRUCTURE_IMAGE_CREATE_INFO,
        next: ptr::null(),
        flags: 0,
        image_type: sys::IMAGE_TYPE_2D,
        format: sys::FORMAT_RGBA8_UNORM,
        extent: sys::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: sys::SAMPLE_COUNT_1,
        tiling: sys::TILING_OPTIMAL,
        usage: sys::IMAGE_TRANSFER,
        sharing_mode: sys::SHARING_EXCLUSIVE,
        queue_family_index_count: 0,
        queue_family_indices: ptr::null(),
        initial_layout: sys::LAYOUT_UNDEFINED,
    };
    let mut image = 0_u64;
    let code = unsafe { sys::vkCreateImage(device, &info, ptr::null(), &mut image) };
    if code != 0 || image == 0 {
        return Err(AndroidError::Device);
    }
    Ok(image)
}

fn bind_new_image(session: &Session, token: u64) -> Result<u64, AndroidError> {
    let requirements = image_requirements(session.device, token);
    if requirements.size == 0 {
        return Err(AndroidError::Device);
    }
    let Some(index) = memory_index(
        &session.memory,
        requirements.memory_type_bits,
        0,
        sys::MEMORY_DEVICE_LOCAL,
    ) else {
        return Err(AndroidError::Device);
    };
    let memory = allocate_memory(session.device, requirements.size, index)?;
    let code = unsafe { sys::vkBindImageMemory(session.device, token, memory, 0) };
    if code != 0 || token == 0 {
        free_memory(session.device, memory);
        return Err(AndroidError::Device);
    }
    Ok(memory)
}

fn image_requirements(device: *mut c_void, image: u64) -> sys::MemoryRequirements {
    let mut requirements = zero_requirements();
    unsafe {
        sys::vkGetImageMemoryRequirements(device, image, &mut requirements);
    }
    requirements
}

fn buffer_requirements(device: *mut c_void, buffer: u64) -> sys::MemoryRequirements {
    let mut requirements = zero_requirements();
    unsafe {
        sys::vkGetBufferMemoryRequirements(device, buffer, &mut requirements);
    }
    requirements
}

fn zero_requirements() -> sys::MemoryRequirements {
    sys::MemoryRequirements {
        size: 0,
        alignment: 0,
        memory_type_bits: 0,
    }
}

fn memory_index(
    properties: &sys::MemoryProperties,
    type_bits: u32,
    required: u32,
    prefer: u32,
) -> Option<u32> {
    let Ok(count) = usize::try_from(properties.memory_type_count) else {
        return None;
    };
    if count > sys::MEMORY_TYPE_LIMIT {
        return None;
    }
    let mut first = None;
    for (index, kind) in properties.memory_types.iter().take(count).enumerate() {
        let Ok(shift) = u32::try_from(index) else {
            return None;
        };
        let bit = 1_u32.checked_shl(shift)?;
        if (type_bits & bit) == 0 {
            continue;
        }
        let flags = kind.property_flags & required;
        if flags != required {
            continue;
        }
        if first.is_none() {
            first = Some(shift);
        }
        let preferred = kind.property_flags & prefer;
        if prefer != 0 && preferred == prefer {
            return Some(shift);
        }
    }
    first
}

fn allocate_memory(device: *mut c_void, size: u64, index: u32) -> Result<u64, AndroidError> {
    if device.is_null() || size == 0 {
        return Err(AndroidError::Device);
    }
    let info = sys::MemoryAllocateInfo {
        s_type: sys::STRUCTURE_MEMORY_ALLOCATE_INFO,
        next: ptr::null(),
        allocation_size: size,
        memory_type_index: index,
    };
    let mut memory = 0_u64;
    let code = unsafe { sys::vkAllocateMemory(device, &info, ptr::null(), &mut memory) };
    if code != 0 || memory == 0 {
        return Err(AndroidError::Device);
    }
    Ok(memory)
}

struct Staging {
    buffer: u64,
    memory: u64,
    map_size: u64,
}

impl Staging {
    fn create(session: &Session, size: u64, usage: u32) -> Result<Self, AndroidError> {
        if size == 0 || session.device.is_null() {
            return Err(AndroidError::Device);
        }
        let buffer = create_buffer(session.device, size, usage)?;
        match bind_new_buffer(session, buffer) {
            Ok(bound) => Ok(Self {
                buffer,
                memory: bound.memory,
                map_size: bound.map_size,
            }),
            Err(error) => {
                destroy_buffer(session.device, buffer);
                Err(error)
            }
        }
    }

    fn destroy(self, session: &Session) {
        destroy_buffer(session.device, self.buffer);
        free_memory(session.device, self.memory);
    }
}

struct BoundBuffer {
    memory: u64,
    map_size: u64,
}

fn bind_new_buffer(session: &Session, buffer: u64) -> Result<BoundBuffer, AndroidError> {
    let requirements = buffer_requirements(session.device, buffer);
    if requirements.size == 0 {
        return Err(AndroidError::Device);
    }
    let Some(index) = memory_index(
        &session.memory,
        requirements.memory_type_bits,
        sys::MEMORY_HOST_VISIBLE_COHERENT,
        0,
    ) else {
        return Err(AndroidError::Device);
    };
    let memory = allocate_memory(session.device, requirements.size, index)?;
    let code = unsafe { sys::vkBindBufferMemory(session.device, buffer, memory, 0) };
    if code != 0 {
        free_memory(session.device, memory);
        return Err(AndroidError::Device);
    }
    Ok(BoundBuffer {
        memory,
        map_size: requirements.size,
    })
}

fn create_buffer(device: *mut c_void, size: u64, usage: u32) -> Result<u64, AndroidError> {
    let info = sys::BufferCreateInfo {
        s_type: sys::STRUCTURE_BUFFER_CREATE_INFO,
        next: ptr::null(),
        flags: 0,
        size,
        usage,
        sharing_mode: sys::SHARING_EXCLUSIVE,
        queue_family_index_count: 0,
        queue_family_indices: ptr::null(),
    };
    let mut buffer = 0_u64;
    let code = unsafe { sys::vkCreateBuffer(device, &info, ptr::null(), &mut buffer) };
    if code != 0 || buffer == 0 {
        return Err(AndroidError::Device);
    }
    Ok(buffer)
}

struct Command {
    buffer: *mut c_void,
}

impl Command {
    fn allocate(session: &Session) -> Result<Self, AndroidError> {
        let info = sys::CommandBufferAllocateInfo {
            s_type: sys::STRUCTURE_COMMAND_BUFFER_ALLOCATE_INFO,
            next: ptr::null(),
            command_pool: session.pool,
            level: sys::LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut buffer = ptr::null_mut();
        let code = unsafe { sys::vkAllocateCommandBuffers(session.device, &info, &mut buffer) };
        if code != 0 || buffer.is_null() {
            return Err(AndroidError::Device);
        }
        Ok(Self { buffer })
    }

    fn destroy(self, session: &Session) {
        if session.device.is_null() || session.pool == 0 || self.buffer.is_null() {
            return;
        }
        unsafe {
            sys::vkFreeCommandBuffers(session.device, session.pool, 1, &self.buffer);
        }
    }
}

fn with_transfer<T>(
    session: &mut Session,
    length: usize,
    usage: u32,
    run: impl FnOnce(&mut Session, &Staging, *mut c_void) -> Result<T, AndroidError>,
) -> Result<T, AndroidError> {
    if session.device.is_null() || session.queue.is_null() || session.pool == 0 {
        return Err(AndroidError::Device);
    }
    let size = u64::try_from(length).map_err(|_| AndroidError::Device)?;
    let staging = Staging::create(session, size, usage)?;
    let command = match Command::allocate(session) {
        Ok(command) => command,
        Err(error) => {
            staging.destroy(session);
            return Err(error);
        }
    };
    let outcome = run(session, &staging, command.buffer);
    command.destroy(session);
    staging.destroy(session);
    outcome
}

fn write_mapped(session: &Session, staging: &Staging, bytes: &[u8]) -> Result<(), AndroidError> {
    let mapped = Mapped::map(session, staging, bytes.len())?;
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.data, bytes.len());
    }
    Ok(())
}

#[allow(clippy::slow_vector_initialization)]
fn read_mapped(
    session: &Session,
    staging: &Staging,
    length: usize,
) -> Result<Vec<u8>, AndroidError> {
    let mapped = Mapped::map(session, staging, length)?;
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(length)
        .map_err(|_| AndroidError::Device)?;
    copied.resize(length, 0);
    unsafe {
        ptr::copy_nonoverlapping(mapped.data, copied.as_mut_ptr(), length);
    }
    Ok(copied)
}

struct Mapped<'a> {
    session: &'a Session,
    memory: u64,
    data: *mut u8,
}

impl Drop for Mapped<'_> {
    fn drop(&mut self) {
        if !self.session.device.is_null() && self.memory != 0 {
            unsafe {
                sys::vkUnmapMemory(self.session.device, self.memory);
            }
        }
        self.memory = 0;
        self.data = ptr::null_mut();
    }
}

impl<'a> Mapped<'a> {
    fn map(session: &'a Session, staging: &Staging, needed: usize) -> Result<Self, AndroidError> {
        let available = usize::try_from(staging.map_size).map_err(|_| AndroidError::Device)?;
        if available < needed || session.device.is_null() || staging.memory == 0 {
            return Err(AndroidError::Device);
        }
        let mut data = ptr::null_mut();
        let code = unsafe {
            sys::vkMapMemory(
                session.device,
                staging.memory,
                0,
                staging.map_size,
                0,
                &mut data,
            )
        };
        if code != 0 {
            return Err(AndroidError::Device);
        }
        if data.is_null() {
            unsafe {
                sys::vkUnmapMemory(session.device, staging.memory);
            }
            return Err(AndroidError::Device);
        }
        Ok(Self {
            session,
            memory: staging.memory,
            data: data.cast(),
        })
    }
}

fn record_upload(
    session: &Session,
    command: *mut c_void,
    image: Image,
    staging: &Staging,
) -> Result<(), AndroidError> {
    begin_command(command)?;
    image_barrier(
        command,
        image.token,
        image.layout,
        sys::LAYOUT_TRANSFER_DST,
        sys::ACCESS_TRANSFER_WRITE,
    );
    copy_buffer_to_image(command, image, staging);
    submit_and_wait(session, command)
}

fn record_sample(
    session: &Session,
    command: *mut c_void,
    image: Image,
    staging: &Staging,
    length: usize,
) -> Result<(), AndroidError> {
    begin_command(command)?;
    image_barrier(
        command,
        image.token,
        image.layout,
        sys::LAYOUT_TRANSFER_SRC,
        sys::ACCESS_TRANSFER_READ,
    );
    copy_image_to_buffer(command, image, staging);
    let size = u64::try_from(length).map_err(|_| AndroidError::Device)?;
    buffer_barrier(command, staging.buffer, size);
    submit_and_wait(session, command)
}

fn begin_command(command: *mut c_void) -> Result<(), AndroidError> {
    let info = sys::CommandBufferBeginInfo {
        s_type: sys::STRUCTURE_COMMAND_BUFFER_BEGIN_INFO,
        next: ptr::null(),
        flags: sys::COMMAND_ONE_TIME,
        inheritance: ptr::null(),
    };
    let code = unsafe { sys::vkBeginCommandBuffer(command, &info) };
    device_ok(code)
}

fn submit_and_wait(session: &Session, command: *mut c_void) -> Result<(), AndroidError> {
    let code = unsafe { sys::vkEndCommandBuffer(command) };
    device_ok(code)?;
    if session.queue.is_null() {
        return Err(AndroidError::Device);
    }
    let submit = sys::SubmitInfo {
        s_type: sys::STRUCTURE_SUBMIT_INFO,
        next: ptr::null(),
        wait_semaphore_count: 0,
        wait_semaphores: ptr::null(),
        wait_dst_stage_mask: ptr::null(),
        command_buffer_count: 1,
        command_buffers: ptr::from_ref(&command),
        signal_semaphore_count: 0,
        signal_semaphores: ptr::null(),
    };
    let code = unsafe { sys::vkQueueSubmit(session.queue, 1, &submit, 0) };
    device_ok(code)?;
    let code = unsafe { sys::vkQueueWaitIdle(session.queue) };
    device_ok(code)
}

fn image_barrier(
    command: *mut c_void,
    image: u64,
    old_layout: u32,
    new_layout: u32,
    dst_access: u32,
) {
    let (src_access, src_stage) = source_dependency(old_layout);
    let barrier = sys::ImageMemoryBarrier {
        s_type: sys::STRUCTURE_IMAGE_MEMORY_BARRIER,
        next: ptr::null(),
        src_access_mask: src_access,
        dst_access_mask: dst_access,
        old_layout,
        new_layout,
        src_queue_family_index: u32::MAX,
        dst_queue_family_index: u32::MAX,
        image,
        subresource_range: color_range(),
    };
    unsafe {
        sys::vkCmdPipelineBarrier(
            command,
            src_stage,
            sys::STAGE_TRANSFER,
            0,
            0,
            ptr::null(),
            0,
            ptr::null(),
            1,
            &barrier,
        );
    }
}

fn source_dependency(layout: u32) -> (u32, u32) {
    if layout == sys::LAYOUT_UNDEFINED {
        (0, sys::STAGE_TOP_OF_PIPE)
    } else if layout == sys::LAYOUT_TRANSFER_SRC {
        (sys::ACCESS_TRANSFER_READ, sys::STAGE_TRANSFER)
    } else {
        (sys::ACCESS_TRANSFER_WRITE, sys::STAGE_TRANSFER)
    }
}

fn buffer_barrier(command: *mut c_void, buffer: u64, size: u64) {
    let barrier = sys::BufferMemoryBarrier {
        s_type: sys::STRUCTURE_BUFFER_MEMORY_BARRIER,
        next: ptr::null(),
        src_access_mask: sys::ACCESS_TRANSFER_WRITE,
        dst_access_mask: sys::ACCESS_HOST_READ,
        src_queue_family_index: u32::MAX,
        dst_queue_family_index: u32::MAX,
        buffer,
        offset: 0,
        size,
    };
    unsafe {
        sys::vkCmdPipelineBarrier(
            command,
            sys::STAGE_TRANSFER,
            sys::STAGE_HOST,
            0,
            0,
            ptr::null(),
            1,
            &barrier,
            0,
            ptr::null(),
        );
    }
}

fn copy_buffer_to_image(command: *mut c_void, image: Image, staging: &Staging) {
    let region = copy_region(image.extent);
    unsafe {
        sys::vkCmdCopyBufferToImage(
            command,
            staging.buffer,
            image.token,
            sys::LAYOUT_TRANSFER_DST,
            1,
            &region,
        );
    }
}

fn copy_image_to_buffer(command: *mut c_void, image: Image, staging: &Staging) {
    let region = copy_region(image.extent);
    unsafe {
        sys::vkCmdCopyImageToBuffer(
            command,
            image.token,
            sys::LAYOUT_TRANSFER_SRC,
            staging.buffer,
            1,
            &region,
        );
    }
}

fn copy_region(extent: Extent) -> sys::BufferImageCopy {
    sys::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: sys::ImageSubresourceLayers {
            aspect_mask: sys::ASPECT_COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: sys::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: sys::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        },
    }
}

fn color_range() -> sys::ImageSubresourceRange {
    sys::ImageSubresourceRange {
        aspect_mask: sys::ASPECT_COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn destroy_image(device: *mut c_void, image: u64) {
    if device.is_null() || image == 0 {
        return;
    }
    unsafe {
        sys::vkDestroyImage(device, image, ptr::null());
    }
}

fn destroy_buffer(device: *mut c_void, buffer: u64) {
    if device.is_null() || buffer == 0 {
        return;
    }
    unsafe {
        sys::vkDestroyBuffer(device, buffer, ptr::null());
    }
}

fn free_memory(device: *mut c_void, memory: u64) {
    if device.is_null() || memory == 0 {
        return;
    }
    unsafe {
        sys::vkFreeMemory(device, memory, ptr::null());
    }
}
