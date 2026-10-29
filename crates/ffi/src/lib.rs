//! The calls SwiftUI and Compose make.
//!
//! Desktop does not use this crate. A phone shell opens one project folder,
//! applies the same session requests the desktop applies, and looks up a
//! bundled or cached file. Playback here is the playhead and the audio
//! output. Seeking a picture needs a device the shell constructs; this crate
//! does not construct one.

#![forbid(unsafe_code)]

mod error;
mod project;

pub use error::FfiError;
pub use project::Project;

#[cfg(test)]
mod tests {
    #[test]
    fn the_phone_can_name_the_draft_and_the_clock() {
        let draft = cutlass_store::Draft::empty("D");
        assert_eq!(draft.id.as_str(), "D");
        let linkage = cutlass_session::linkage_from_biz(None, draft.id.as_str(), false);
        assert!(!linkage.enabled);
        assert_eq!(cutlass_session::Micros(5).0, 5);
    }
}
