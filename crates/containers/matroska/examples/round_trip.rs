//! Write a short WebM file in memory, read it back, and seek in it.
//!
//! `cargo run -p cutlass-matroska --example round_trip` needs no file on
//! disk. [`MatroskaMux`] writes two seconds of VP9 at 25 fps (a keyframe
//! each second) and Opus in 20 ms packets; the payloads are placeholder
//! bytes, since a container stores frames without looking inside them.
//! [`MatroskaDemux`] then prints the streams, counts the packets, and seeks
//! the picture track to 1.5 s, which lands on the keyframe at 1 s.

#![allow(clippy::print_stdout)]

use std::io::Cursor;

use cutlass_codec::{CodecId, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, SeekTo, TrackDuration};
use cutlass_matroska::{MatroskaDemux, MatroskaMux};

const VIDEO_TICKS: u32 = 1000;
const FRAME_MS: i64 = 40;
const OPUS_RATE: u32 = 48_000;
const OPUS_PACKET: i64 = 960;

fn main() -> Result<(), String> {
    let bytes = write_webm().map_err(|error| error.to_string())?;
    println!("wrote {} bytes", bytes.len());
    let length = u64::try_from(bytes.len()).map_err(|error| error.to_string())?;
    let mut demux =
        MatroskaDemux::open(Cursor::new(bytes), length).map_err(|error| error.to_string())?;
    println!("doctype {}", demux.ebml_header().doc_type.as_str());
    for stream in demux.streams() {
        println!(
            "track {} {:?}, {} ticks per second, encoder delay {:?}",
            stream.id().get(),
            stream.params().codec(),
            stream.timescale().ticks_per_second(),
            stream.trim().encoder_delay(),
        );
    }
    let mut counts = [0_u32; 2];
    while let Some(packet) = demux.read_packet().map_err(|error| error.to_string())? {
        if let Some(count) = usize::try_from(packet.stream())
            .ok()
            .and_then(|track| counts.get_mut(track.wrapping_sub(1)))
        {
            *count += 1;
        }
    }
    println!("read {} picture and {} sound packets", counts[0], counts[1]);
    let landing = demux
        .seek(SeekTo {
            track: 1,
            time: 1500,
            end: None,
        })
        .map_err(|error| error.to_string())?;
    println!("seek to 1500 ms: {:?}", landing.landing);
    Ok(())
}

fn write_webm() -> Result<Vec<u8>, cutlass_container::MuxError> {
    let mut mux = MatroskaMux::webm(Cursor::new(Vec::new()));
    let size =
        FrameSize::try_new(320, 240).map_err(|_| cutlass_container::MuxError::NotBuilding)?;
    let picture = mux.add_stream(NewStream {
        params: StreamParams::video(StreamCodec::Known(CodecId::Vp9), size, Vec::new()),
        timescale: Timescale::new(VIDEO_TICKS),
        language: None,
    })?;
    let sound_params =
        StreamParams::try_audio(StreamCodec::Known(CodecId::Opus), OPUS_RATE, 2, opus_head())
            .map_err(|_| cutlass_container::MuxError::NotBuilding)?;
    let sound = mux.add_stream(NewStream {
        params: sound_params,
        timescale: Timescale::new(OPUS_RATE),
        language: None,
    })?;
    for frame in 0..50 {
        let pts = frame * FRAME_MS;
        let keyframe = frame % 25 == 0;
        mux.write_packet(&packet(
            picture.get(),
            vec![0x82; 600],
            pts,
            FRAME_MS,
            VIDEO_TICKS,
            keyframe,
        ))?;
        // Two 20 ms Opus packets (TOC 0xF8: one 20 ms CELT frame) per picture.
        for half in 0..2 {
            let at = (frame * 2 + half) * OPUS_PACKET;
            mux.write_packet(&packet(
                sound.get(),
                vec![0xF8, 0, 0],
                at,
                OPUS_PACKET,
                OPUS_RATE,
                true,
            ))?;
        }
    }
    mux.finish(&[
        TrackDuration {
            track: picture,
            timescale: Timescale::new(VIDEO_TICKS),
            ticks: 50 * FRAME_MS,
        },
        TrackDuration {
            track: sound,
            timescale: Timescale::new(OPUS_RATE),
            ticks: 100 * OPUS_PACKET,
        },
    ])
    .map_err(|failed| failed.reason().clone())?;
    Ok(mux.into_writer()?.into_inner())
}

fn packet(track: u32, bytes: Vec<u8>, pts: i64, duration: i64, ticks: u32, key: bool) -> Packet {
    let time =
        Timestamp::try_new(Some(pts), Some(pts), Some(duration)).unwrap_or(Timestamp::missing());
    Packet::try_new(track, bytes, time, Timescale::new(ticks), key)
        .unwrap_or_else(|_| unreachable!("a packet with times is valid"))
}

/// An OpusHead: version 1, stereo, a 312-sample pre-skip, 48 kHz.
fn opus_head() -> Vec<u8> {
    let mut head = b"OpusHead".to_vec();
    head.extend_from_slice(&[1, 2]);
    head.extend_from_slice(&312_u16.to_le_bytes());
    head.extend_from_slice(&OPUS_RATE.to_le_bytes());
    head.extend_from_slice(&[0, 0, 0]);
    head
}
