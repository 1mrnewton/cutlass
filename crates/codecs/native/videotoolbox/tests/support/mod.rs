use cutlass_core::{Packet, PixelFormat, VideoFrame};
use cutlass_test_fixtures::{Crc32, visible_row_bytes};
use cutlass_videotoolbox::{Decoder, PixelBuffer};

/// Sends every packet, then flushes, collecting frames as they come out.
pub fn decode_all<'a>(
    decoder: &mut Decoder,
    packets: impl IntoIterator<Item = &'a Packet>,
) -> Vec<VideoFrame<PixelBuffer>> {
    let mut frames = Vec::new();
    for packet in packets {
        decoder.send_packet(packet).unwrap();
        while let Some(frame) = decoder.receive_frame() {
            frames.push(frame.unwrap());
        }
    }
    decoder.flush().unwrap();
    while let Some(frame) = decoder.receive_frame() {
        frames.push(frame.unwrap());
    }
    frames
}

/// CRC of the visible bytes of each plane, matching the fixture checksums.
pub fn frame_crc(image: &PixelBuffer, format: PixelFormat) -> u32 {
    image
        .read_planes(|planes| {
            let mut crc = Crc32::new();
            for (index, plane) in planes.iter().enumerate() {
                let row_bytes = visible_row_bytes(format, index, plane.width);
                for row in plane.data.chunks(plane.bytes_per_row).take(plane.height) {
                    crc.update(&row[..row_bytes]);
                }
            }
            crc.finish()
        })
        .unwrap()
}
