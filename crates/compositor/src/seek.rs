use cutlass_gpu::{Device, Pass};
use cutlass_model::Micros;
use cutlass_picture::{PictureSource, Sample};

/// Sample each source at `time` and draw the ready pictures.
pub(crate) fn draw_ready<D: Device>(
    device: &mut D,
    pass: Pass,
    sources: &[&dyn PictureSource],
    time: Micros,
) -> Result<(), D::Error> {
    for source in sources {
        if let Sample::Ready(picture) = source.sample(time) {
            device.draw(pass, picture.texels())?;
        }
    }
    Ok(())
}
