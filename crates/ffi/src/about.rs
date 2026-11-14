//! The phone's About page: Cutlass's own license, then a notice for each
//! open sticker set and each crate the app is built from. A notice's
//! license text is read only when it is opened.

/// One line on the About page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneNotice {
    /// The sticker set or the crate and its version.
    pub title: String,
    /// The SPDX expression, and for a sticker set who made it.
    pub license: String,
    pub link: String,
}

/// Cutlass's own license.
pub const PHONE_LICENSE: &str = cutlass_licenses::CUTLASS_LICENSE;

/// Every notice, sticker sets first.
#[must_use]
pub fn phone_notices() -> Vec<PhoneNotice> {
    let stickers = cutlass_stickers::CREDITS.iter().map(|credit| PhoneNotice {
        title: format!("{} stickers", credit.set),
        license: format!("{} by {}", credit.license, credit.author),
        link: credit.link.to_owned(),
    });
    let notices = cutlass_licenses::notices();
    let crates = notices.crates.iter().map(|krate| PhoneNotice {
        title: format!("{} {}", krate.name, krate.version),
        license: krate.license.clone(),
        link: krate.repository.clone(),
    });
    stickers.chain(crates).collect()
}

/// The license text of notice `index` in [`phone_notices`], word for word;
/// several files are joined by a blank line. Empty past the end.
#[must_use]
pub fn notice_text(index: usize) -> String {
    let credits = cutlass_stickers::CREDITS;
    if let Some(credit) = credits.get(index) {
        return credit.text.to_owned();
    }
    let notices = cutlass_licenses::notices();
    notices
        .crates
        .get(index - credits.len())
        .map(|krate| notices.texts_of(krate).collect::<Vec<_>>().join("\n\n"))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_notices_start_with_the_sticker_sets_and_each_opens_to_its_text() {
        let notices = phone_notices();
        assert!(notices.len() > 50);
        assert!(notices[0].title.contains("Noto"));
        assert!(notice_text(0).contains("Apache"));
        let last = notices.len() - 1;
        assert!(!notices[last].license.is_empty());
        assert!(notice_text(notices.len()).is_empty());
        assert!(notices.iter().all(|notice| !notice.title.is_empty()));
    }
}
