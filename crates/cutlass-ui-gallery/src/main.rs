use slint::BackendSelector;
use slint::Model;
use slint::wgpu_29::WGPUConfiguration;

slint::include_modules!();

fn contains_ci(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    haystack.to_lowercase().contains(&needle.to_lowercase())
}

fn wire_string_filter(app: &GalleryWindow) {
    let filter = app.global::<StringFilter>();

    filter.on_contains(|haystack, needle| contains_ci(haystack.as_str(), needle.as_str()));

    filter.on_index_of(|items, value| {
        let value = value.as_str();
        for i in 0..items.row_count() {
            if items.row_data(i).is_some_and(|row| row.as_str() == value) {
                return i as i32;
            }
        }
        -1
    });

    filter.on_next_match(|items, query, from, direction| {
        let len = items.row_count();
        if len == 0 || direction == 0 {
            return -1;
        }
        let query = query.as_str();
        let len_i = len as i32;
        let mut i = from + direction;
        for _ in 0..len {
            if i < 0 {
                i = len_i - 1;
            } else if i >= len_i {
                i = 0;
            }
            if let Some(row) = items.row_data(i as usize) {
                if contains_ci(row.as_str(), query) {
                    return i;
                }
            }
            i += direction;
        }
        -1
    });
}

fn main() -> Result<(), slint::PlatformError> {
    BackendSelector::new()
        .require_wgpu_29(WGPUConfiguration::default())
        .select()?;

    let app = GalleryWindow::new()?;
    wire_string_filter(&app);
    app.run()
}
