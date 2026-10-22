//! Walk `MediaCodecList` through JNI. Slots include the four reserved pointers.

use std::ffi::{CStr, c_char, c_void};
use std::ptr;

use crate::caps::{self, Capabilities};

const RTLD_NOW: i32 = 2;
const JNI_OK: i32 = 0;
const JNI_EDETACHED: i32 = -2;
const JNI_VERSION_1_6: i32 = 0x0001_0006;
const MAX_CODECS: i32 = 128;
const MAX_TYPES: i32 = 16;

const VM_ATTACH: usize = 4;
const VM_DETACH: usize = 5;
const VM_GET_ENV: usize = 6;

const FIND_CLASS: usize = 6;
const EXCEPTION_OCCURRED: usize = 15;
const EXCEPTION_CLEAR: usize = 17;
const DELETE_LOCAL: usize = 23;
const NEW_OBJECT_A: usize = 30;
const GET_METHOD_ID: usize = 33;
const CALL_OBJECT_A: usize = 36;
const CALL_BOOLEAN_A: usize = 39;
const GET_STRING_UTF: usize = 169;
const RELEASE_STRING_UTF: usize = 170;
const GET_ARRAY_LENGTH: usize = 171;
const GET_OBJECT_ELEMENT: usize = 173;

pub(crate) fn query() -> Capabilities {
    let Some(vm) = created_vm() else {
        return Capabilities::default();
    };
    let Some((env, attached)) = attach(vm) else {
        return Capabilities::default();
    };
    let caps = list_codecs(env);
    if attached {
        detach(vm);
    }
    caps
}

fn list_codecs(env: *mut c_void) -> Capabilities {
    let Some(class) = find_class(env, c"android/media/MediaCodecList") else {
        return Capabilities::default();
    };
    let Some(init) = method(env, class, c"<init>", c"(I)V") else {
        drop_local(env, class);
        return Capabilities::default();
    };
    let arg = JValue { i: 0, pad: 0 };
    let Some(list) = new_object(env, class, init, &arg) else {
        drop_local(env, class);
        return Capabilities::default();
    };
    drop_local(env, class);
    let Some(list_class) = object_class(env, list) else {
        drop_local(env, list);
        return Capabilities::default();
    };
    let Some(get_infos) = method(
        env,
        list_class,
        c"getCodecInfos",
        c"()[Landroid/media/MediaCodecInfo;",
    ) else {
        drop_local(env, list_class);
        drop_local(env, list);
        return Capabilities::default();
    };
    drop_local(env, list_class);
    let Some(infos) = call_object(env, list, get_infos) else {
        drop_local(env, list);
        return Capabilities::default();
    };
    let mut video_decode = Vec::new();
    let mut video_encode = Vec::new();
    let mut audio_decode = Vec::new();
    let mut audio_encode = Vec::new();
    let count = array_len(env, infos).min(MAX_CODECS);
    for index in 0..count {
        let Some(info) = array_element(env, infos, index) else {
            continue;
        };
        read_info(
            env,
            info,
            &mut video_decode,
            &mut video_encode,
            &mut audio_decode,
            &mut audio_encode,
        );
        drop_local(env, info);
    }
    drop_local(env, infos);
    drop_local(env, list);
    Capabilities::from_lists(video_decode, video_encode, audio_decode, audio_encode)
}

fn read_info(
    env: *mut c_void,
    info: *mut c_void,
    video_decode: &mut Vec<cutlass_core::VideoCodec>,
    video_encode: &mut Vec<cutlass_core::VideoCodec>,
    audio_decode: &mut Vec<cutlass_core::AudioCodec>,
    audio_encode: &mut Vec<cutlass_core::AudioCodec>,
) {
    let Some(class) = object_class(env, info) else {
        return;
    };
    let encoder =
        method(env, class, c"isEncoder", c"()Z").is_some_and(|id| call_boolean(env, info, id));
    let Some(get_types) = method(env, class, c"getSupportedTypes", c"()[Ljava/lang/String;") else {
        drop_local(env, class);
        return;
    };
    drop_local(env, class);
    let Some(types) = call_object(env, info, get_types) else {
        return;
    };
    let count = array_len(env, types).min(MAX_TYPES);
    for index in 0..count {
        let Some(text) = array_element(env, types, index) else {
            continue;
        };
        if let Some(mime) = utf_bytes(env, text) {
            if encoder {
                if let Some(codec) = caps::video_mime(&mime) {
                    caps::push_video(video_encode, codec);
                }
                if let Some(codec) = caps::audio_mime(&mime) {
                    caps::push_audio(audio_encode, codec);
                }
            } else {
                if let Some(codec) = caps::video_mime(&mime) {
                    caps::push_video(video_decode, codec);
                }
                if let Some(codec) = caps::audio_mime(&mime) {
                    caps::push_audio(audio_decode, codec);
                }
            }
        }
        drop_local(env, text);
    }
    drop_local(env, types);
}

fn created_vm() -> Option<*mut c_void> {
    let library = Library::open(c"libnativehelper.so")?;
    let get_vms = load::<GetVms>(&library, c"JNI_GetCreatedJavaVMs")?;
    let mut vm = ptr::null_mut();
    let mut count = 0;
    // SAFETY: `vm` receives one `JavaVM*`. `count` is a local `jsize`.
    let status = unsafe { get_vms(&mut vm, 1, &mut count) };
    if status != JNI_OK || count < 1 || vm.is_null() {
        None
    } else {
        Some(vm)
    }
}

fn attach(vm: *mut c_void) -> Option<(*mut c_void, bool)> {
    let get_env = vm_fn::<GetEnv>(vm, VM_GET_ENV)?;
    let mut env = ptr::null_mut();
    // SAFETY: `vm` came from `JNI_GetCreatedJavaVMs`. `env` receives the
    // `JNIEnv*`.
    let status = unsafe { get_env(vm, &mut env, JNI_VERSION_1_6) };
    if status == JNI_OK && !env.is_null() {
        return Some((env, false));
    }
    if status != JNI_EDETACHED {
        return None;
    }
    let attach = vm_fn::<Attach>(vm, VM_ATTACH)?;
    // SAFETY: a null argument block attaches as a non-daemon thread.
    let status = unsafe { attach(vm, &mut env, ptr::null_mut()) };
    if status != JNI_OK || env.is_null() {
        None
    } else {
        Some((env, true))
    }
}

fn detach(vm: *mut c_void) {
    let Some(detach) = vm_fn::<Detach>(vm, VM_DETACH) else {
        return;
    };
    // SAFETY: this thread was attached above and is not the app's thread.
    unsafe { detach(vm) };
}

fn find_class(env: *mut c_void, name: &CStr) -> Option<*mut c_void> {
    let find = env_fn::<FindClass>(env, FIND_CLASS)?;
    // SAFETY: `name` is a nul-terminated internal class name.
    let class = unsafe { find(env, name.as_ptr()) };
    if class.is_null() || failed(env) {
        drop_local(env, class);
        None
    } else {
        Some(class)
    }
}

fn object_class(env: *mut c_void, object: *mut c_void) -> Option<*mut c_void> {
    // GetObjectClass is slot 31.
    let get = env_fn::<GetObjectClass>(env, 31)?;
    // SAFETY: `object` is a live local ref.
    let class = unsafe { get(env, object) };
    if class.is_null() || failed(env) {
        drop_local(env, class);
        None
    } else {
        Some(class)
    }
}

fn method(env: *mut c_void, class: *mut c_void, name: &CStr, sig: &CStr) -> Option<*mut c_void> {
    let get = env_fn::<GetMethodId>(env, GET_METHOD_ID)?;
    // SAFETY: `name` and `sig` are nul-terminated JNI method descriptors.
    let id = unsafe { get(env, class, name.as_ptr(), sig.as_ptr()) };
    if id.is_null() || failed(env) {
        None
    } else {
        Some(id)
    }
}

fn new_object(
    env: *mut c_void,
    class: *mut c_void,
    method: *mut c_void,
    arg: &JValue,
) -> Option<*mut c_void> {
    let new = env_fn::<NewObjectA>(env, NEW_OBJECT_A)?;
    // SAFETY: `arg` is the single `int` constructor argument. `REGULAR_CODECS`
    // is 0.
    let object = unsafe { new(env, class, method, arg) };
    if object.is_null() || failed(env) {
        drop_local(env, object);
        None
    } else {
        Some(object)
    }
}

fn call_object(env: *mut c_void, object: *mut c_void, method: *mut c_void) -> Option<*mut c_void> {
    let call = env_fn::<CallObjectA>(env, CALL_OBJECT_A)?;
    // SAFETY: the method takes no arguments, so the argument pointer is null.
    let value = unsafe { call(env, object, method, ptr::null()) };
    if value.is_null() || failed(env) {
        drop_local(env, value);
        None
    } else {
        Some(value)
    }
}

fn call_boolean(env: *mut c_void, object: *mut c_void, method: *mut c_void) -> bool {
    let Some(call) = env_fn::<CallBooleanA>(env, CALL_BOOLEAN_A) else {
        return false;
    };
    // SAFETY: the method takes no arguments, so the argument pointer is null.
    let value = unsafe { call(env, object, method, ptr::null()) };
    !failed(env) && value != 0
}

fn array_len(env: *mut c_void, array: *mut c_void) -> i32 {
    let Some(len) = env_fn::<GetArrayLen>(env, GET_ARRAY_LENGTH) else {
        return 0;
    };
    // SAFETY: `array` is a live array local ref.
    let count = unsafe { len(env, array) };
    if failed(env) || count < 0 { 0 } else { count }
}

fn array_element(env: *mut c_void, array: *mut c_void, index: i32) -> Option<*mut c_void> {
    let get = env_fn::<GetElement>(env, GET_OBJECT_ELEMENT)?;
    // SAFETY: `index` is inside the length this function's caller measured.
    let element = unsafe { get(env, array, index) };
    if element.is_null() || failed(env) {
        drop_local(env, element);
        None
    } else {
        Some(element)
    }
}

fn utf_bytes(env: *mut c_void, text: *mut c_void) -> Option<Vec<u8>> {
    let get = env_fn::<GetStringUtf>(env, GET_STRING_UTF)?;
    // SAFETY: a null copy-flag pointer tells JNI we do not care if it copied.
    let chars = unsafe { get(env, text, ptr::null_mut()) };
    if chars.is_null() || failed(env) {
        return None;
    }
    // SAFETY: `chars` is a nul-terminated modified UTF-8 string from JNI, and
    // it stays valid until `ReleaseStringUTFChars`.
    let bytes = unsafe { CStr::from_ptr(chars) }.to_bytes().to_vec();
    if let Some(release) = env_fn::<ReleaseStringUtf>(env, RELEASE_STRING_UTF) {
        // SAFETY: `chars` came from `GetStringUTFChars` for `text`.
        unsafe { release(env, text, chars) };
    }
    Some(bytes)
}

fn failed(env: *mut c_void) -> bool {
    let Some(occurred) = env_fn::<Occurred>(env, EXCEPTION_OCCURRED) else {
        return true;
    };
    // SAFETY: returns the pending exception, or null when there is none.
    let pending = unsafe { occurred(env) };
    if pending.is_null() {
        return false;
    }
    drop_local(env, pending);
    if let Some(clear) = env_fn::<Clear>(env, EXCEPTION_CLEAR) {
        // SAFETY: clears the pending exception so the next call can proceed.
        unsafe { clear(env) };
    }
    true
}

fn drop_local(env: *mut c_void, object: *mut c_void) {
    if object.is_null() {
        return;
    }
    let Some(delete) = env_fn::<DeleteLocal>(env, DELETE_LOCAL) else {
        return;
    };
    // SAFETY: `object` is a local ref from this thread's `JNIEnv`.
    unsafe { delete(env, object) };
}

fn vm_fn<T>(vm: *mut c_void, index: usize) -> Option<T> {
    if vm.is_null() {
        return None;
    }
    // SAFETY: a `JavaVM` is the invoke function table.
    fn_at(vm as *const *const c_void, index)
}

fn env_fn<T>(env: *mut c_void, index: usize) -> Option<T> {
    if env.is_null() {
        return None;
    }
    // SAFETY: a `JNIEnv*` is the native function table.
    fn_at(env as *const *const c_void, index)
}

fn fn_at<T>(table: *const *const c_void, index: usize) -> Option<T> {
    const {
        assert!(
            std::mem::size_of::<T>() == std::mem::size_of::<*const c_void>(),
            "a JNI slot is one pointer"
        );
    }
    if table.is_null() {
        return None;
    }
    // SAFETY: `index` is a slot in the JNI function table, which is an array
    // of pointers. The table is longer than the slots this file uses.
    let symbol = unsafe { *table.add(index) };
    if symbol.is_null() {
        None
    } else {
        // SAFETY: `T` is the C function type of this JNI slot.
        Some(unsafe { std::mem::transmute_copy(&symbol) })
    }
}

struct Library {
    handle: *mut c_void,
}

impl Library {
    fn open(name: &CStr) -> Option<Self> {
        // SAFETY: `name` is a nul-terminated literal. A null handle means the
        // library is not loaded in this process.
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            None
        } else {
            Some(Self { handle })
        }
    }

    fn symbol(&self, name: &CStr) -> Option<*mut c_void> {
        // SAFETY: `handle` came from `dlopen` and is still open.
        let symbol = unsafe { dlsym(self.handle, name.as_ptr()) };
        if symbol.is_null() { None } else { Some(symbol) }
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: `handle` came from `dlopen` and has not been closed.
        unsafe { dlclose(self.handle) };
    }
}

fn load<T>(library: &Library, name: &CStr) -> Option<T> {
    const {
        assert!(
            std::mem::size_of::<T>() == std::mem::size_of::<*mut c_void>(),
            "a loaded symbol is one pointer"
        );
    }
    let symbol = library.symbol(name)?;
    // SAFETY: `T` is the C function type of `name`.
    Some(unsafe { std::mem::transmute_copy(&symbol) })
}

#[repr(C, align(8))]
struct JValue {
    i: i32,
    pad: u32,
}

type GetVms = unsafe extern "system" fn(*mut *mut c_void, i32, *mut i32) -> i32;
type GetEnv = unsafe extern "system" fn(*mut c_void, *mut *mut c_void, i32) -> i32;
type Attach = unsafe extern "system" fn(*mut c_void, *mut *mut c_void, *mut c_void) -> i32;
type Detach = unsafe extern "system" fn(*mut c_void) -> i32;
type FindClass = unsafe extern "system" fn(*mut c_void, *const c_char) -> *mut c_void;
type GetObjectClass = unsafe extern "system" fn(*mut c_void, *mut c_void) -> *mut c_void;
type GetMethodId = unsafe extern "system" fn(
    *mut c_void,
    *mut c_void,
    *const c_char,
    *const c_char,
) -> *mut c_void;
type NewObjectA =
    unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const JValue) -> *mut c_void;
type CallObjectA =
    unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const JValue) -> *mut c_void;
type CallBooleanA =
    unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const JValue) -> u8;
type GetArrayLen = unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32;
type GetElement = unsafe extern "system" fn(*mut c_void, *mut c_void, i32) -> *mut c_void;
type GetStringUtf = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut u8) -> *const c_char;
type ReleaseStringUtf = unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_char);
type Occurred = unsafe extern "system" fn(*mut c_void) -> *mut c_void;
type Clear = unsafe extern "system" fn(*mut c_void);
type DeleteLocal = unsafe extern "system" fn(*mut c_void, *mut c_void);

// SAFETY: these are the dynamic loader functions in Android's libc.
unsafe extern "system" {
    fn dlopen(name: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
}
