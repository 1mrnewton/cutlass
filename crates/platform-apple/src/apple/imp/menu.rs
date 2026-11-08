//! Menu bar menus that show their keys but leave key presses to the window.
//!
//! A menu item with keys normally takes those keys: AppKit offers every key
//! press to the main menu, which asks each menu's `performKeyEquivalent:`,
//! before the window sees it. The window keeps its own key bindings, which
//! skip a focused text field, so a bare `S` or `⌘C` must still reach it.
//! Each chosen menu becomes a subclass of NSMenu whose `performKeyEquivalent:`
//! answers NO; its items still draw their keys. A menu delegate cannot do
//! this: AppKit asks the delegate about keys only for menus it fills lazily.

use std::ffi::{CStr, c_char};
use std::sync::OnceLock;

use objc2::runtime::{AnyClass, AnyObject, Bool, ClassBuilder, Sel};
use objc2::{class, msg_send, sel};

/// `performKeyEquivalent:` on a menu that takes no key presses.
extern "C-unwind" fn no_key_equivalent(
    _menu: *mut AnyObject,
    _sel: Sel,
    _event: *mut AnyObject,
) -> Bool {
    Bool::NO
}

/// The NSMenu subclass whose menus take no key presses. It adds no ivars,
/// so an NSMenu can become one in place.
fn keys_shown_only_class() -> Option<&'static AnyClass> {
    static CLASS: OnceLock<Option<&'static AnyClass>> = OnceLock::new();
    *CLASS.get_or_init(|| {
        let mut builder = ClassBuilder::new(c"CutlassKeysShownOnlyMenu", class!(NSMenu))?;
        // SAFETY: `performKeyEquivalent:` takes one NSEvent and returns a
        // BOOL, as the function does.
        let method: extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject) -> Bool =
            no_key_equivalent;
        unsafe {
            builder.add_method(sel!(performKeyEquivalent:), method);
        }
        Some(builder.register())
    })
}

/// Make the menu bar's menus titled one of `titles` show their items' keys
/// without taking key presses. Call it on the main thread after the menu
/// bar is up. Returns how many menus it changed; off the main thread or
/// with no menu bar, none.
#[must_use]
pub fn menu_keys_shown_only(titles: &[&str]) -> usize {
    // SAFETY: `isMainThread` is a class method with no arguments.
    let main: Bool = unsafe { msg_send![class!(NSThread), isMainThread] };
    if !main.as_bool() {
        return 0;
    }
    // SAFETY: on the main thread, `sharedApplication` returns the app and
    // `mainMenu` its menu bar or nil; both are checked before use.
    let bar: *mut AnyObject = unsafe {
        let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
        if app.is_null() {
            return 0;
        }
        msg_send![app, mainMenu]
    };
    if bar.is_null() {
        return 0;
    }
    let Some(shown_only) = keys_shown_only_class() else {
        return 0;
    };
    let mut changed = 0;
    // SAFETY: each call is a documented NSMenu, NSMenuItem or NSString
    // method on the main thread, and every object result is checked for nil
    // before use. A menu changes class only when it is a plain NSMenu, and
    // the subclass adds no ivars, so the object's layout is unchanged.
    unsafe {
        let count: isize = msg_send![bar, numberOfItems];
        for index in 0..count {
            let item: *mut AnyObject = msg_send![bar, itemAtIndex: index];
            if item.is_null() {
                continue;
            }
            let menu: *mut AnyObject = msg_send![item, submenu];
            if menu.is_null() {
                continue;
            }
            let title: *mut AnyObject = msg_send![menu, title];
            if title.is_null() {
                continue;
            }
            let text: *const c_char = msg_send![title, UTF8String];
            if text.is_null() {
                continue;
            }
            let title = CStr::from_ptr(text).to_string_lossy();
            if !titles.contains(&title.as_ref()) {
                continue;
            }
            let kind = objc2::ffi::object_getClass(menu);
            if std::ptr::eq(kind, shown_only) {
                changed += 1;
            } else if std::ptr::eq(kind, class!(NSMenu)) {
                objc2::ffi::object_setClass(menu, shown_only);
                changed += 1;
            }
        }
    }
    changed
}
