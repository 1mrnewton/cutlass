// Native window chrome on macOS (the shipped "rounded frame" approach): the
// shell keeps the OS-drawn window frame — rounded corners, drop shadow, and
// the traffic-light controls — but hides the titlebar so the custom Slint
// title bar can sit in its place. The window stays an ordinary titled
// `NSWindow`; we only flip it to `fullSizeContentView` with a transparent,
// title-less, separator-less titlebar. On every other platform the shell is
// fully frameless (`no-frame` in app.slint) and this is a no-op.

use slint::winit_030::winit;

#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2_app_kit::{
    NSAnimatablePropertyContainer, NSAnimationContext, NSTitlebarSeparatorStyle, NSView, NSWindow,
    NSWindowStyleMask, NSWindowTitleVisibility,
};
#[cfg(target_os = "macos")]
use objc2_foundation::NSRect;
#[cfg(target_os = "macos")]
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

#[cfg(target_os = "windows")]
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
#[cfg(target_os = "windows")]
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
#[cfg(target_os = "windows")]
use windows::Win32::UI::Shell::{DefSubclassProc, SetWindowSubclass};
#[cfg(target_os = "windows")]
use windows::Win32::UI::WindowsAndMessaging::{
    DefWindowProcW, GetSystemMetrics, GetWindowRect, HTCLIENT, HTTOP, HTTOPLEFT, HTTOPRIGHT,
    IsZoomed, NCCALCSIZE_PARAMS, SM_CXPADDEDBORDER, SM_CYFRAME, SWP_FRAMECHANGED, SWP_NOMOVE,
    SWP_NOSIZE, SWP_NOZORDER, SetWindowPos, WM_NCCALCSIZE, WM_NCHITTEST,
};

#[cfg(target_os = "macos")]
thread_local! {
    // The launch-screen window frame, captured before the first maximize so
    // the editor→launch close can pop straight back to it. macOS-only state
    // and every window op runs on the main thread, so a thread-local Cell is
    // enough — no cross-thread sharing.
    static NATURAL_FRAME: std::cell::Cell<Option<NSRect>> = const { std::cell::Cell::new(None) };
}

// Resolve the `NSWindow` hosting the given winit window through its raw
// `NSView` handle. `None` if the handle isn't an AppKit one (e.g. before the
// window is shown).
#[cfg(target_os = "macos")]
fn ns_window_of(window: &winit::window::Window) -> Option<Retained<NSWindow>> {
    let raw = window.window_handle().ok()?.as_raw();
    let RawWindowHandle::AppKit(handle) = raw else {
        return None;
    };
    // SAFETY: `handle.ns_view` is documented by raw-window-handle to be a
    // live, retained `NSView *` for the duration the WindowHandle is borrowed
    // from winit. We retain it before reading `-window` so the returned
    // `NSWindow` is independently owned.
    unsafe {
        let view_ptr: *mut NSView = handle.ns_view.as_ptr().cast();
        let view: Retained<NSView> = Retained::retain(view_ptr)?;
        view.window()
    }
}

/// Hide the titlebar while keeping the native (rounded, shadowed) frame and
/// the traffic-light controls. Idempotent — safe to call more than once.
#[cfg(target_os = "macos")]
pub fn apply_native_chrome(window: &winit::window::Window) {
    let Some(ns_window) = ns_window_of(window) else {
        return;
    };
    // Let content fill the full height (under the now-hidden titlebar) and
    // strip the titlebar's chrome so the custom title bar shows through.
    let mask = ns_window.styleMask() | NSWindowStyleMask::FullSizeContentView;
    ns_window.setStyleMask(mask);
    ns_window.setTitlebarAppearsTransparent(true);
    ns_window.setTitleVisibility(NSWindowTitleVisibility::Hidden);
    // Drop the hairline under the (invisible) titlebar so the title bar reads
    // as one continuous surface.
    ns_window.setTitlebarSeparatorStyle(NSTitlebarSeparatorStyle::None);
}

// Windows custom title bar — the same idea as the macOS path (keep the OS
// frame, suppress only the caption) but driven through Win32 since there's no
// AppKit. The window keeps its standard styles, so native resize, Aero snap,
// the drop shadow and Win11 rounded corners all survive; we subclass its proc
// to (a) reclaim the caption strip into the client area via WM_NCCALCSIZE and
// (b) hand the now-client top edge back to the resize logic via WM_NCHITTEST.
// The custom Slint title bar then paints over that reclaimed strip, with its
// own minimize/maximize/close buttons (shown off macOS in title-bar.slint).
#[cfg(target_os = "windows")]
const TITLEBAR_SUBCLASS_ID: usize = 0xC07A_5511;

// Resolve the top-level `HWND` from winit's raw handle. `None` before the
// window is realized or if the handle isn't a Win32 one.
#[cfg(target_os = "windows")]
fn hwnd_of(window: &winit::window::Window) -> Option<HWND> {
    let raw = window.window_handle().ok()?.as_raw();
    let RawWindowHandle::Win32(handle) = raw else {
        return None;
    };
    Some(HWND(handle.hwnd.get() as *mut core::ffi::c_void))
}

// Width of the window's resize frame (the band that hit-tests as a drag-resize
// edge). SM_CYFRAME alone undercounts on modern Windows; the invisible padded
// border (SM_CXPADDEDBORDER) makes up the rest.
#[cfg(target_os = "windows")]
fn resize_border_thickness() -> i32 {
    // SAFETY: GetSystemMetrics is a pure, thread-safe metric read.
    unsafe { GetSystemMetrics(SM_CYFRAME) + GetSystemMetrics(SM_CXPADDEDBORDER) }
}

/// Suppress the native caption while keeping the OS frame, so the custom Slint
/// title bar shows through. Idempotent — re-subclassing with the same id is a
/// no-op, and the frame change is cheap.
#[cfg(target_os = "windows")]
pub fn apply_native_chrome(window: &winit::window::Window) {
    let Some(hwnd) = hwnd_of(window) else {
        return;
    };
    // SAFETY: `hwnd` is the live top-level window winit just created on this
    // (UI) thread. The subclass proc is a 'static fn with no borrowed state;
    // the FRAMECHANGED nudge forces a WM_NCCALCSIZE pass so the caption is
    // dropped immediately rather than on the next resize.
    unsafe {
        let _ = SetWindowSubclass(hwnd, Some(titlebar_subclass_proc), TITLEBAR_SUBCLASS_ID, 0);
        let _ = SetWindowPos(
            hwnd,
            None,
            0,
            0,
            0,
            0,
            SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER,
        );
    }
}

#[cfg(target_os = "windows")]
unsafe extern "system" fn titlebar_subclass_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    _id: usize,
    _data: usize,
) -> LRESULT {
    match msg {
        // Reclaim the caption strip: let DefWindowProc compute the standard
        // non-client frame (which gives us the side/bottom resize borders),
        // then restore the original top so the client area swallows the title
        // bar. The window keeps WS_CAPTION/WS_THICKFRAME, so the WM still draws
        // the shadow + rounded corners and honors snap.
        WM_NCCALCSIZE if wparam.0 != 0 => {
            // SAFETY: for an NCCALCSIZE with wParam==TRUE, lParam is a valid,
            // writable NCCALCSIZE_PARAMS for the duration of this message.
            let params = unsafe { &mut *(lparam.0 as *mut NCCALCSIZE_PARAMS) };
            let original_top = params.rgrc[0].top;
            // SAFETY: forwarding the message to the default window procedure.
            let ret = unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) };
            if ret.0 != 0 {
                return ret;
            }
            params.rgrc[0].top = original_top;
            // A maximized window's frame extends past the monitor by the border
            // thickness on every side; inset the top to match so the title bar
            // (and its caption buttons) aren't clipped under the screen edge.
            // SAFETY: simple state query on a live HWND.
            if unsafe { IsZoomed(hwnd) }.as_bool() {
                params.rgrc[0].top += resize_border_thickness();
            }
            LRESULT(0)
        }
        // The reclaimed top no longer hit-tests as a resize edge (it's client
        // now), so re-add a top resize band by hand, corners included. Sides
        // and bottom still come from the default proc; caption dragging and
        // double-click-maximize are handled in Slint (WindowBackend.begin-move
        // → winit drag_window, which emulates an HTCAPTION drag for snap).
        WM_NCHITTEST => {
            // SAFETY: chaining to the next proc in the subclass chain (winit's).
            let hit = unsafe { DefSubclassProc(hwnd, msg, wparam, lparam) };
            if hit.0 != HTCLIENT as isize {
                return hit;
            }
            let x = (lparam.0 & 0xFFFF) as i16 as i32;
            let y = ((lparam.0 >> 16) & 0xFFFF) as i16 as i32;
            let mut rc = RECT::default();
            // SAFETY: `rc` is a valid out-param; HWND is live.
            if unsafe { GetWindowRect(hwnd, &mut rc) }.is_err() {
                return hit;
            }
            let border = resize_border_thickness();
            if y < rc.top + border {
                if x < rc.left + border {
                    return LRESULT(HTTOPLEFT as isize);
                }
                if x >= rc.right - border {
                    return LRESULT(HTTOPRIGHT as isize);
                }
                return LRESULT(HTTOP as isize);
            }
            hit
        }
        // SAFETY: chaining everything else to winit's window procedure.
        _ => unsafe { DefSubclassProc(hwnd, msg, wparam, lparam) },
    }
}

/// No-op on Linux/BSD: there the shell is fully frameless (`no-frame`) and
/// draws its own chrome end to end, so there's no native caption to suppress.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn apply_native_chrome(_window: &winit::window::Window) {}

/// Length of the launch↔editor frame morph. Short enough to feel instant,
/// long enough to read as a smooth resize rather than a flicker.
#[cfg(target_os = "macos")]
const MORPH_DURATION: f64 = 0.16;

/// Maximize / restore the window for the launch↔editor switch with a short,
/// custom morph. winit's `set_maximized` calls `-[NSWindow zoom:]`, whose
/// native animation is heavy and slow; a plain `animate:NO` setFrame snaps but
/// flickers (the surface resizes a frame before content catches up). Instead
/// we animate the frame over `MORPH_DURATION` via `NSAnimationContext` + the
/// window's `animator` proxy: the launch-screen frame is remembered on the way
/// up and morphed back to on the way down. `-isZoomed` still reports the
/// filled-screen state afterwards, so winit's own maximize tracking (and the
/// title-bar double-click toggle that reads it) stays correct.
#[cfg(target_os = "macos")]
pub fn set_maximized(window: &winit::window::Window, maximized: bool) {
    let Some(ns_window) = ns_window_of(window) else {
        return;
    };
    let target = if maximized {
        // Capture the windowed frame to restore later — but not if we're
        // already filling the screen, or we'd save the maximized frame.
        if !ns_window.isZoomed() {
            let frame = ns_window.frame();
            NATURAL_FRAME.with(|f| f.set(Some(frame)));
        }
        match ns_window.screen() {
            Some(screen) => screen.visibleFrame(),
            None => return,
        }
    } else {
        match NATURAL_FRAME.with(|f| f.get()) {
            Some(frame) => frame,
            None => return,
        }
    };
    NSAnimationContext::beginGrouping();
    NSAnimationContext::currentContext().setDuration(MORPH_DURATION);
    ns_window.animator().setFrame_display(target, true);
    NSAnimationContext::endGrouping();
}

/// Off macOS, defer to winit's maximize (the WM owns any animation).
#[cfg(not(target_os = "macos"))]
pub fn set_maximized(window: &winit::window::Window, maximized: bool) {
    window.set_maximized(maximized);
}
