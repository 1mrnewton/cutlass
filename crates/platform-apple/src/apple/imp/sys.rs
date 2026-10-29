//! Framework entry points. The wrappers in this crate are the only callers.

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::ffi::c_void;

pub type OSStatus = i32;
pub type OSType = u32;
pub type CFIndex = isize;
pub type ItemCount = i64;

#[repr(C)]
pub struct CMTime {
    pub value: i64,
    pub timescale: i32,
    pub flags: u32,
    pub epoch: i64,
}

#[repr(C)]
pub struct CMSampleTimingInfo {
    pub duration: CMTime,
    pub presentation: CMTime,
    pub decode: CMTime,
}

#[repr(C)]
pub struct AudioStreamBasicDescription {
    pub sample_rate: f64,
    pub format_id: u32,
    pub format_flags: u32,
    pub bytes_per_packet: u32,
    pub frames_per_packet: u32,
    pub bytes_per_frame: u32,
    pub channels: u32,
    pub bits: u32,
    pub reserved: u32,
}

#[repr(C)]
pub struct AudioBuffer {
    pub channels: u32,
    pub size: u32,
    pub data: *mut c_void,
}

#[repr(C)]
pub struct AudioBufferList {
    pub buffers: u32,
    pub buffer: AudioBuffer,
}

#[repr(C)]
pub struct AudioStreamPacketDescription {
    pub start: i64,
    pub frames: u32,
    pub size: u32,
}

#[repr(C)]
pub struct AudioConverterPrimeInfo {
    pub leading: u32,
    pub trailing: u32,
}

#[repr(C)]
pub struct CGPoint {
    pub x: f64,
    pub y: f64,
}

#[repr(C)]
pub struct CGSize {
    pub width: f64,
    pub height: f64,
}

#[repr(C)]
pub struct CGRect {
    pub origin: CGPoint,
    pub size: CGSize,
}

#[repr(C)]
pub struct VTDecompressionOutputCallbackRecord {
    pub callback: VTDecompressionOutputCallback,
    pub refcon: *mut c_void,
}

pub type VTDecompressionOutputCallback = unsafe extern "C" fn(
    refcon: *mut c_void,
    source: *mut c_void,
    status: OSStatus,
    flags: u32,
    image: *mut c_void,
    pts: CMTime,
    duration: CMTime,
);

pub type VTCompressionOutputCallback = unsafe extern "C" fn(
    refcon: *mut c_void,
    source: *mut c_void,
    status: OSStatus,
    flags: u32,
    sample: *mut c_void,
);

pub type AudioQueueRef = *mut c_void;

pub type AudioQueueBufferRef = *mut AudioQueueBuffer;

/// One Core Audio queue buffer. Field order matches `AudioQueueBuffer`.
#[repr(C)]
pub struct AudioQueueBuffer {
    pub audio_data_bytes_capacity: u32,
    pub audio_data: *mut c_void,
    pub audio_data_byte_size: u32,
    pub user_data: *mut c_void,
    pub packet_description_capacity: u32,
    pub packet_descriptions: *mut AudioStreamPacketDescription,
    pub packet_description_count: u32,
}

pub type AudioQueueOutputCallback =
    unsafe extern "C" fn(user: *mut c_void, queue: AudioQueueRef, buffer: AudioQueueBufferRef);

pub type AudioInputProc = unsafe extern "C" fn(
    converter: *mut c_void,
    packets: *mut u32,
    data: *mut AudioBufferList,
    descriptions: *mut *mut AudioStreamPacketDescription,
    user: *mut c_void,
) -> OSStatus;

pub const UTF8: u32 = 0x0800_0100;
pub const CF_NUMBER_SINT32: isize = 3;
pub const CF_NUMBER_FLOAT64: isize = 6;
pub const TIME_VALID: u32 = 1;
pub const BGRA: u32 = u32::from_be_bytes(*b"BGRA");
pub const ALPHA_PREMULTIPLIED_LAST: u32 = 1;
pub const BYTE_ORDER_32_BIG: u32 = 4 << 12;

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    pub static kCFTypeDictionaryKeyCallBacks: u8;

    pub static kCFTypeDictionaryValueCallBacks: u8;

    pub static kCFBooleanTrue: *const c_void;

    pub static kCFBooleanFalse: *const c_void;

    pub static kCFAllocatorNull: *const c_void;

    pub fn CFRelease(value: *const c_void);

    pub fn CFRetain(value: *const c_void) -> *const c_void;

    pub fn CFStringCreateWithCString(
        allocator: *const c_void,
        bytes: *const u8,
        encoding: u32,
    ) -> *mut c_void;

    pub fn CFDataCreate(allocator: *const c_void, bytes: *const u8, len: CFIndex) -> *mut c_void;

    pub fn CFDataCreateMutable(allocator: *const c_void, capacity: CFIndex) -> *mut c_void;

    pub fn CFDataGetLength(data: *const c_void) -> CFIndex;

    pub fn CFDataGetBytePtr(data: *const c_void) -> *const u8;

    pub fn CFNumberCreate(
        allocator: *const c_void,
        kind: CFIndex,
        value: *const c_void,
    ) -> *mut c_void;

    pub fn CFNumberGetValue(number: *const c_void, kind: CFIndex, value: *mut c_void) -> u8;

    pub fn CFDictionaryCreate(
        allocator: *const c_void,
        keys: *const *const c_void,
        values: *const *const c_void,
        count: CFIndex,
        key_callbacks: *const c_void,
        value_callbacks: *const c_void,
    ) -> *mut c_void;

    pub fn CFDictionaryGetValue(dict: *const c_void, key: *const c_void) -> *const c_void;

    pub fn CFArrayGetCount(array: *const c_void) -> CFIndex;

    pub fn CFArrayGetValueAtIndex(array: *const c_void, index: CFIndex) -> *const c_void;

}
#[link(name = "VideoToolbox", kind = "framework")]
unsafe extern "C" {
    pub static kVTCompressionPropertyKey_AverageBitRate: *const c_void;

    pub static kVTCompressionPropertyKey_AllowFrameReordering: *const c_void;

    pub static kVTCompressionPropertyKey_MaxKeyFrameInterval: *const c_void;

    pub static kVTCompressionPropertyKey_ExpectedFrameRate: *const c_void;

    pub static kVTCompressionPropertyKey_RealTime: *const c_void;

    pub static kVTCompressionPropertyKey_ProfileLevel: *const c_void;

    pub static kVTEncodeFrameOptionKey_ForceKeyFrame: *const c_void;

    pub static kVTProfileLevel_H264_Baseline_AutoLevel: *const c_void;

    pub static kVTProfileLevel_HEVC_Main_AutoLevel: *const c_void;

    pub fn VTDecompressionSessionCreate(
        allocator: *const c_void,
        format: *const c_void,
        decoder: *const c_void,
        destination: *const c_void,
        callback: *const VTDecompressionOutputCallbackRecord,
        out: *mut *mut c_void,
    ) -> OSStatus;

    pub fn VTDecompressionSessionInvalidate(session: *mut c_void);

    pub fn VTDecompressionSessionDecodeFrame(
        session: *mut c_void,
        sample: *const c_void,
        flags: u32,
        source: *mut c_void,
        info: *mut u32,
    ) -> OSStatus;

    pub fn VTDecompressionSessionWaitForAsynchronousFrames(session: *mut c_void) -> OSStatus;

    pub fn VTCompressionSessionCreate(
        allocator: *const c_void,
        width: i32,
        height: i32,
        codec: OSType,
        encoder: *const c_void,
        source: *const c_void,
        compressed_allocator: *const c_void,
        callback: VTCompressionOutputCallback,
        refcon: *mut c_void,
        out: *mut *mut c_void,
    ) -> OSStatus;

    pub fn VTCompressionSessionInvalidate(session: *mut c_void);

    pub fn VTCompressionSessionEncodeFrame(
        session: *mut c_void,
        image: *mut c_void,
        pts: CMTime,
        duration: CMTime,
        properties: *const c_void,
        source: *mut c_void,
        info: *mut u32,
    ) -> OSStatus;

    pub fn VTCompressionSessionCompleteFrames(session: *mut c_void, until: CMTime) -> OSStatus;

    pub fn VTSessionSetProperty(
        session: *mut c_void,
        key: *const c_void,
        value: *const c_void,
    ) -> OSStatus;

}
#[link(name = "CoreMedia", kind = "framework")]
unsafe extern "C" {
    pub static kCMSampleAttachmentKey_NotSync: *const c_void;

    pub fn CMBlockBufferCreateWithMemoryBlock(
        structure: *const c_void,
        memory: *mut c_void,
        block_length: usize,
        block_allocator: *const c_void,
        custom: *const c_void,
        offset: usize,
        data_length: usize,
        flags: u32,
        out: *mut *mut c_void,
    ) -> OSStatus;

    pub fn CMBlockBufferGetDataPointer(
        buffer: *const c_void,
        offset: usize,
        length_at_offset: *mut usize,
        total: *mut usize,
        data: *mut *mut u8,
    ) -> OSStatus;

    pub fn CMSampleBufferCreateReady(
        allocator: *const c_void,
        data: *const c_void,
        format: *const c_void,
        samples: ItemCount,
        timing_count: ItemCount,
        timing: *const CMSampleTimingInfo,
        size_count: ItemCount,
        sizes: *const usize,
        out: *mut *mut c_void,
    ) -> OSStatus;

    pub fn CMSampleBufferGetDataBuffer(sample: *const c_void) -> *const c_void;

    pub fn CMSampleBufferGetImageBuffer(sample: *const c_void) -> *const c_void;

    pub fn CMSampleBufferGetFormatDescription(sample: *const c_void) -> *const c_void;

    pub fn CMSampleBufferGetSampleAttachmentsArray(
        sample: *const c_void,
        create: u8,
    ) -> *const c_void;

    pub fn CMVideoFormatDescriptionCreate(
        allocator: *const c_void,
        codec: OSType,
        width: i32,
        height: i32,
        extensions: *const c_void,
        out: *mut *mut c_void,
    ) -> OSStatus;

    pub fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
        allocator: *const c_void,
        count: usize,
        pointers: *const *const u8,
        sizes: *const usize,
        nal_length: i32,
        out: *mut *mut c_void,
    ) -> OSStatus;

    pub fn CMVideoFormatDescriptionCreateFromHEVCParameterSets(
        allocator: *const c_void,
        count: usize,
        pointers: *const *const u8,
        sizes: *const usize,
        nal_length: i32,
        extensions: *const c_void,
        out: *mut *mut c_void,
    ) -> OSStatus;

    pub fn CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
        format: *const c_void,
        index: usize,
        pointer: *mut *const u8,
        size: *mut usize,
        count: *mut usize,
        nal_length: *mut i32,
    ) -> OSStatus;

    pub fn CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
        format: *const c_void,
        index: usize,
        pointer: *mut *const u8,
        size: *mut usize,
        count: *mut usize,
        nal_length: *mut i32,
    ) -> OSStatus;

}
#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    pub static kCVPixelBufferPixelFormatTypeKey: *const c_void;

    pub static kCVPixelBufferIOSurfacePropertiesKey: *const c_void;

    pub fn CVPixelBufferCreate(
        allocator: *const c_void,
        width: usize,
        height: usize,
        format: OSType,
        attributes: *const c_void,
        out: *mut *mut c_void,
    ) -> OSStatus;

    pub fn CVPixelBufferLockBaseAddress(buffer: *mut c_void, flags: u32) -> OSStatus;

    pub fn CVPixelBufferUnlockBaseAddress(buffer: *mut c_void, flags: u32) -> OSStatus;

    pub fn CVPixelBufferGetBaseAddress(buffer: *mut c_void) -> *mut c_void;

    pub fn CVPixelBufferGetBytesPerRow(buffer: *mut c_void) -> usize;

    pub fn CVPixelBufferGetWidth(buffer: *mut c_void) -> usize;

    pub fn CVPixelBufferGetHeight(buffer: *mut c_void) -> usize;

    pub fn CVPixelBufferGetPixelFormatType(buffer: *mut c_void) -> OSType;

}
#[link(name = "ImageIO", kind = "framework")]
unsafe extern "C" {
    pub static kCGImageDestinationLossyCompressionQuality: *const c_void;

    pub static kCGImagePropertyOrientation: *const c_void;

    pub fn CGImageSourceCreateWithData(data: *const c_void, options: *const c_void) -> *mut c_void;

    pub fn CGImageSourceGetCount(source: *const c_void) -> usize;

    pub fn CGImageSourceCreateImageAtIndex(
        source: *const c_void,
        index: usize,
        options: *const c_void,
    ) -> *mut c_void;

    pub fn CGImageSourceCopyPropertiesAtIndex(
        source: *const c_void,
        index: usize,
        options: *const c_void,
    ) -> *mut c_void;

    pub fn CGImageDestinationCreateWithData(
        data: *mut c_void,
        kind: *const c_void,
        count: usize,
        options: *const c_void,
    ) -> *mut c_void;

    pub fn CGImageDestinationAddImage(
        destination: *mut c_void,
        image: *const c_void,
        properties: *const c_void,
    );

    pub fn CGImageDestinationFinalize(destination: *mut c_void) -> bool;

}
#[link(name = "AudioToolbox", kind = "framework")]
unsafe extern "C" {
    pub fn AudioConverterNew(
        source: *const AudioStreamBasicDescription,
        destination: *const AudioStreamBasicDescription,
        out: *mut *mut c_void,
    ) -> OSStatus;

    pub fn AudioConverterDispose(converter: *mut c_void) -> OSStatus;

    pub fn AudioConverterReset(converter: *mut c_void) -> OSStatus;

    pub fn AudioConverterGetProperty(
        converter: *mut c_void,
        property: u32,
        size: *mut u32,
        data: *mut c_void,
    ) -> OSStatus;

    pub fn AudioConverterSetProperty(
        converter: *mut c_void,
        property: u32,
        size: u32,
        data: *const c_void,
    ) -> OSStatus;

    pub fn AudioConverterFillComplexBuffer(
        converter: *mut c_void,
        input: AudioInputProc,
        user: *mut c_void,
        packets: *mut u32,
        output: *mut AudioBufferList,
        descriptions: *mut AudioStreamPacketDescription,
    ) -> OSStatus;

    pub fn AudioQueueNewOutput(
        format: *const AudioStreamBasicDescription,
        callback: AudioQueueOutputCallback,
        user: *mut c_void,
        run_loop: *mut c_void,
        mode: *const c_void,
        flags: u32,
        out: *mut AudioQueueRef,
    ) -> OSStatus;

    pub fn AudioQueueAllocateBuffer(
        queue: AudioQueueRef,
        size: u32,
        out: *mut AudioQueueBufferRef,
    ) -> OSStatus;

    pub fn AudioQueueEnqueueBuffer(
        queue: AudioQueueRef,
        buffer: AudioQueueBufferRef,
        packets: u32,
        descriptions: *const AudioStreamPacketDescription,
    ) -> OSStatus;

    pub fn AudioQueueStart(queue: AudioQueueRef, start: *const c_void) -> OSStatus;

    pub fn AudioQueueStop(queue: AudioQueueRef, immediate: u8) -> OSStatus;

    pub fn AudioQueueDispose(queue: AudioQueueRef, immediate: u8) -> OSStatus;

}
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    pub fn CGImageGetWidth(image: *const c_void) -> usize;

    pub fn CGImageGetHeight(image: *const c_void) -> usize;

    pub fn CGImageCreate(
        width: usize,
        height: usize,
        bits_per_component: usize,
        bits_per_pixel: usize,
        bytes_per_row: usize,
        space: *const c_void,
        bitmap: u32,
        provider: *const c_void,
        decode: *const c_void,
        interpolate: bool,
        intent: i32,
    ) -> *mut c_void;

    pub fn CGDataProviderCreateWithData(
        info: *mut c_void,
        data: *const c_void,
        size: usize,
        release: *const c_void,
    ) -> *mut c_void;

    pub fn CGColorSpaceCreateDeviceRGB() -> *mut c_void;

    pub fn CGBitmapContextCreate(
        data: *mut c_void,
        width: usize,
        height: usize,
        bits_per_component: usize,
        bytes_per_row: usize,
        space: *const c_void,
        bitmap: u32,
    ) -> *mut c_void;

    pub fn CGContextDrawImage(context: *mut c_void, rect: CGRect, image: *const c_void);

}
