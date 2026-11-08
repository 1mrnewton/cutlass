//! MP4 files FFmpeg 8.1.1 wrote, opened by the ISOBMFF demuxer.
//!
//! `asp.mp4` is `asp.m4v` copied (`-c copy`). `mpeg2.mp4` and `mjpeg.mp4`
//! are one- and two-frame `mpeg2video` and `mjpeg` encodes of `testsrc2`
//! at 48x32. FFmpeg stores all three in an `mp4v` sample entry. The
//! `esds` object type names the codec: 0x20 is MPEG-4 Visual, 0x61 is
//! MPEG-2, and 0x6C is JPEG.

#[cfg(test)]
mod files {
    use std::io::Cursor;

    use cutlass_codec::{CodecId, StreamCodec};
    use cutlass_container::Demux;
    use cutlass_decode::{CodecParams, Decoder, Output, Packet, PictureSamples};
    use cutlass_isobmff::IsoBmffFile;
    use cutlass_mpeg4::EngineDecoder;

    fn file(name: &str) -> &'static [u8] {
        match name {
            "asp.mp4" => include_bytes!("fixtures/asp.mp4"),
            "asp.yuv" => include_bytes!("fixtures/asp.yuv"),
            "mpeg2.mp4" => include_bytes!("fixtures/mpeg2.mp4"),
            "mjpeg.mp4" => include_bytes!("fixtures/mjpeg.mp4"),
            _ => unreachable!(),
        }
    }

    fn codec(name: &str) -> StreamCodec {
        let bytes = file(name);
        let movie = IsoBmffFile::open(Cursor::new(bytes)).expect("open");
        let stream = movie.streams().first().expect("a track");
        stream.params().codec().clone()
    }

    #[test]
    fn an_mp4v_entry_is_mpeg4_visual_only_for_object_type_0x20() {
        assert_eq!(codec("asp.mp4"), StreamCodec::Known(CodecId::Mpeg4));
        assert_eq!(codec("mpeg2.mp4"), StreamCodec::Known(CodecId::Mpeg2));
        assert_eq!(codec("mjpeg.mp4"), StreamCodec::Known(CodecId::Mjpeg));
    }

    #[test]
    fn the_demuxed_track_matches_ffmpeg() {
        let bytes = file("asp.mp4");
        let mut movie = IsoBmffFile::open(Cursor::new(bytes)).expect("open");
        let stream = movie.streams().first().expect("a track");
        let size = stream.params().size().expect("a picture size");
        assert_eq!((size.width(), size.height()), (48, 32));
        let extra = stream.params().extradata().to_vec();
        assert!(
            extra.windows(4).any(|word| word == [0, 0, 1, 0xB0]),
            "the decoder configuration holds the visual object sequence"
        );
        let mut params = CodecParams::video(CodecId::Mpeg4, size.width(), size.height());
        params.extra = extra;
        let mut decoder = EngineDecoder::open(&params).expect("decoder");
        let mut got = Vec::new();
        while let Some(packet) = movie.read_packet().expect("read") {
            let time = packet.timestamp();
            let access = Packet::video(
                packet.bytes().to_vec(),
                time.presentation_or_decode().unwrap_or(0),
                time.duration().unwrap_or(1),
                packet.timescale().ticks_per_second(),
                packet.keyframe(),
            );
            for output in decoder.push(&access).expect("decode") {
                take(&mut got, output);
            }
        }
        for output in decoder.drain().expect("drain") {
            take(&mut got, output);
        }
        assert_eq!(got, file("asp.yuv"));
    }

    fn take(got: &mut Vec<u8>, output: Output) {
        let Output::Picture(picture) = output else {
            return;
        };
        let PictureSamples::Cpu(samples) = picture.samples() else {
            panic!("cpu samples");
        };
        got.extend_from_slice(samples);
    }
}
