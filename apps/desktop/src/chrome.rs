//! The Mac title bar is a view laid over the project bar, and it takes every click
//! there except the window buttons. Keep that view only as wide as those buttons
//! so File, Export, Settings, and a drag of the project name reach the shell.

#[cfg(target_os = "macos")]
pub(crate) fn pass_clicks(frame: &eframe::Frame) {
    narrow_titlebar(frame);
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn pass_clicks(_frame: &eframe::Frame) {}

#[cfg(target_os = "macos")]
#[expect(
    unsafe_code,
    reason = "the title bar is AppKit state behind the window handle"
)]
fn narrow_titlebar(frame: &eframe::Frame) {
    use objc2_app_kit::{NSView, NSWindowButton, NSWindowStyleMask};
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};

    let Ok(window) = frame.window_handle() else {
        return;
    };
    let RawWindowHandle::AppKit(appkit) = window.as_raw() else {
        return;
    };
    // SAFETY: `ns_view` is this window's content view, and it stays alive while
    // `frame` borrows the handle.
    let view = unsafe { &*appkit.ns_view.as_ptr().cast::<NSView>() };
    let Some(ns_window) = view.window() else {
        return;
    };
    if ns_window
        .styleMask()
        .contains(NSWindowStyleMask::FullScreen)
    {
        return;
    }
    let Some(close) = ns_window.standardWindowButton(NSWindowButton::CloseButton) else {
        return;
    };
    let Some(zoom) = ns_window.standardWindowButton(NSWindowButton::ZoomButton) else {
        return;
    };
    // SAFETY: the window buttons stay on the main thread for the life of the window,
    // and their superview is the title bar that contains them.
    let Some(title_bar) = (unsafe { close.superview() }) else {
        return;
    };
    // SAFETY: the title bar stays on the main thread, and its superview is the
    // container AppKit lays over the project bar.
    let Some(container) = (unsafe { title_bar.superview() }) else {
        return;
    };

    let close_frame = close.frame();
    let zoom_frame = zoom.frame();
    let width = zoom_frame.origin.x + zoom_frame.size.width + close_frame.origin.x;
    let mut container_frame = container.frame();
    if (container_frame.size.width - width).abs() > 0.5 {
        container_frame.size.width = width;
        container.setFrame(container_frame);
    }
    let bounds = container.bounds();
    if title_bar.frame() != bounds {
        title_bar.setFrame(bounds);
    }
}
