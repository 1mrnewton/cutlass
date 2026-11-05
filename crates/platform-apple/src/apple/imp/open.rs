//! Files and folders Finder asks the app to open.
//!
//! A bundled Mac app is not handed documents on its command line: Finder,
//! the Dock and `open -a` send an "open documents" Apple Event. The window
//! library installs no handler for it, so this module does. The handler is
//! installed as the app finishes launching, so the documents that launched
//! the app arrive too; each path waits here until the window takes it.

use std::ffi::{CStr, c_char};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock, PoisonError};

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject};
use objc2::{ClassType, class, define_class, msg_send, sel};

/// `'aevt'`: the core Apple Event class.
const CORE_EVENT_CLASS: u32 = u32::from_be_bytes(*b"aevt");
/// `'odoc'`: open documents.
const OPEN_DOCUMENTS: u32 = u32::from_be_bytes(*b"odoc");
/// `'----'`: the event's direct object, here the list of files.
const DIRECT_OBJECT: u32 = u32::from_be_bytes(*b"----");

/// Paths Finder asked for that the window has not taken yet.
static OPENED: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
/// Wakes the window when a path arrives.
static WAKE: OnceLock<Box<dyn Fn() + Send + Sync>> = OnceLock::new();

define_class!(
    // SAFETY: NSObject has no subclassing requirements, and this class
    // adds no ivars and does not implement Drop.
    #[unsafe(super(NSObject))]
    #[name = "CutlassOpenDocuments"]
    struct OpenDocuments;

    impl OpenDocuments {
        /// `NSApplicationWillFinishLaunchingNotification`: AppKit has set
        /// its own handlers by now, and has not yet sent the launch's
        /// documents, so ours replaces its handler in time.
        #[unsafe(method(willFinishLaunching:))]
        fn will_finish_launching(&self, _notification: *mut AnyObject) {
            install_handler(self);
        }

        /// The "open documents" Apple Event.
        #[unsafe(method(handleOpenDocuments:withReplyEvent:))]
        fn handle_open_documents(&self, event: *mut AnyObject, _reply: *mut AnyObject) {
            let paths = document_paths(event);
            if paths.is_empty() {
                return;
            }
            OPENED
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .extend(paths);
            if let Some(wake) = WAKE.get() {
                wake();
            }
        }
    }
);

fn install_handler(handler: &OpenDocuments) {
    // SAFETY: `sharedAppleEventManager` is a class method returning the
    // shared manager or nil; the call happens on the main thread, where
    // AppKit sends this notification.
    let manager: *mut AnyObject =
        unsafe { msg_send![class!(NSAppleEventManager), sharedAppleEventManager] };
    if manager.is_null() {
        return;
    }
    // SAFETY: the selector names a method this class defines with two
    // object arguments, as the manager calls it; the handler object lives
    // for the whole process (see `listen_for_opened_files`), so the
    // manager's unretained reference to it stays valid.
    unsafe {
        let _: () = msg_send![
            manager,
            setEventHandler: handler,
            andSelector: sel!(handleOpenDocuments:withReplyEvent:),
            forEventClass: CORE_EVENT_CLASS,
            andEventID: OPEN_DOCUMENTS
        ];
    }
}

/// The file paths in an "open documents" event.
fn document_paths(event: *mut AnyObject) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if event.is_null() {
        return paths;
    }
    // SAFETY: `event` is the NSAppleEventDescriptor AppKit passed to the
    // handler; each call below is a documented NSAppleEventDescriptor,
    // NSURL or NSString method, and every object result is checked for
    // nil before use. None of the results is retained past this function.
    unsafe {
        let list: *mut AnyObject = msg_send![event, paramDescriptorForKeyword: DIRECT_OBJECT];
        if list.is_null() {
            return paths;
        }
        let count: isize = msg_send![list, numberOfItems];
        for index in 1..=count {
            let item: *mut AnyObject = msg_send![list, descriptorAtIndex: index];
            if item.is_null() {
                continue;
            }
            let url: *mut AnyObject = msg_send![item, fileURLValue];
            if url.is_null() {
                continue;
            }
            let path: *mut AnyObject = msg_send![url, path];
            if path.is_null() {
                continue;
            }
            let text: *const c_char = msg_send![path, UTF8String];
            if text.is_null() {
                continue;
            }
            paths.push(PathBuf::from(
                CStr::from_ptr(text).to_string_lossy().into_owned(),
            ));
        }
    }
    paths
}

/// Start listening for documents Finder opens with the app. Call it once,
/// on the main thread, before the window's event loop starts.
pub fn listen_for_opened_files() {
    static STARTED: OnceLock<()> = OnceLock::new();
    if STARTED.set(()).is_err() {
        return;
    }
    // SAFETY: `new` on a class with no ivars returns a retained instance.
    let handler: Retained<OpenDocuments> = unsafe { msg_send![OpenDocuments::class(), new] };
    // SAFETY: `defaultCenter` returns the process's notification center;
    // the name is a static C string turned into an NSString; the observer
    // is the handler, which is kept alive below for the whole process.
    unsafe {
        let center: *mut AnyObject = msg_send![class!(NSNotificationCenter), defaultCenter];
        let name: *mut AnyObject = msg_send![
            class!(NSString),
            stringWithUTF8String: c"NSApplicationWillFinishLaunchingNotification".as_ptr()
        ];
        if !center.is_null() && !name.is_null() {
            let _: () = msg_send![
                center,
                addObserver: &*handler,
                selector: sel!(willFinishLaunching:),
                name: name,
                object: std::ptr::null_mut::<AnyObject>()
            ];
        }
    }
    // The manager and the center keep unretained references to it.
    let _kept: *mut OpenDocuments = Retained::into_raw(handler);
}

/// Call `wake` when a document arrives, so the window takes it promptly.
pub fn wake_on_opened_files(wake: impl Fn() + Send + Sync + 'static) {
    let _ = WAKE.set(Box::new(wake));
}

/// The documents Finder asked for since the last call.
#[must_use]
pub fn take_opened_files() -> Vec<PathBuf> {
    std::mem::take(&mut *OPENED.lock().unwrap_or_else(PoisonError::into_inner))
}

#[cfg(test)]
mod tests {
    use super::{OPENED, take_opened_files};

    #[test]
    fn opened_files_wait_until_taken() {
        OPENED
            .lock()
            .unwrap()
            .push(std::path::PathBuf::from("/tmp/a"));
        assert_eq!(
            take_opened_files(),
            vec![std::path::PathBuf::from("/tmp/a")]
        );
        assert!(take_opened_files().is_empty());
    }
}
