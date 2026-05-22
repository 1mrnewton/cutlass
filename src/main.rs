mod ruler;
mod timecode;
mod timeline;

use slint::SharedString;

use slint::BackendSelector;
use slint::wgpu_28::WGPUConfiguration;

slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    BackendSelector::new()
        .require_wgpu_28(WGPUConfiguration::default())
        .select()?;

    let app = AppWindow::new()?;

    let timeline = app.global::<TimelineLib>();
    timeline.on_sequence_duration(timeline::sequence_duration);
    timeline.on_format_timecode(|frame, fps_num, fps_den, drop_frame| {
        SharedString::from(crate::timecode::format_timecode(
            i64::from(frame),
            i64::from(fps_num),
            i64::from(fps_den),
            drop_frame,
        ))
    });

    app.run()
}
