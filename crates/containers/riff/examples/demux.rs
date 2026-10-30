//! Open a short 16-bit stereo WAV and print the stream.
//!
//! The bytes come from [`cutlass_riff::WavMux`], so the example runs without
//! a file on disk. [`cutlass_riff::WavDemux`] is what gets printed.

#![allow(clippy::print_stdout)]

use std::io::Cursor;

use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration, TrackId};
use cutlass_riff::{WavDemux, WavMux};

const RATE: u32 = 44_100;
const CHANNELS: u16 = 2;
const FRAMES: usize = 8;

fn main() -> Result<(), String> {
    let wav = mux_silence()?;
    let length = u64::try_from(wav.len()).map_err(|_| "wav length does not fit".to_owned())?;
    let mut demux = WavDemux::open(Cursor::new(wav), length).map_err(|err| err.to_string())?;
    let stream = demux
        .streams()
        .first()
        .ok_or_else(|| "wav has no audio stream".to_owned())?;
    let rate = stream
        .params()
        .sample_rate()
        .ok_or_else(|| "stream has no sample rate".to_owned())?;
    let channels = stream
        .params()
        .channels()
        .ok_or_else(|| "stream has no channel count".to_owned())?;
    let mut packets = 0u64;
    let mut samples = 0u64;
    while let Some(packet) = demux.read_packet().map_err(|err| err.to_string())? {
        let duration = packet
            .timestamp()
            .duration()
            .ok_or_else(|| "packet has no duration".to_owned())?;
        let duration = u64::try_from(duration).map_err(|_| "packet duration is negative")?;
        samples = samples
            .checked_add(duration)
            .ok_or_else(|| "sample count overflow".to_owned())?;
        packets = packets
            .checked_add(1)
            .ok_or_else(|| "packet count overflow".to_owned())?;
    }
    if packets == 0 {
        return Err("demux returned no packets".to_owned());
    }
    println!("sample_rate {rate}");
    println!("channels {channels}");
    println!("packets {packets}");
    println!("samples {samples}");
    Ok(())
}

fn mux_silence() -> Result<Vec<u8>, String> {
    let frames = i64::try_from(FRAMES).map_err(|_| "frame count does not fit".to_owned())?;
    let params =
        StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), RATE, CHANNELS, Vec::new())
            .map_err(|err| err.to_string())?;
    let timestamp =
        Timestamp::try_new(Some(0), Some(0), Some(frames)).map_err(|err| err.to_string())?;
    let packet = Packet::try_new(0, silence()?, timestamp, Timescale::new(RATE), true)
        .map_err(|err| err.to_string())?;
    let mut mux = WavMux::new(Cursor::new(Vec::new()));
    mux.add_stream(NewStream {
        params,
        timescale: Timescale::new(RATE),
        language: None,
    })
    .map_err(|err| err.to_string())?;
    mux.write_packet(&packet).map_err(|err| err.to_string())?;
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(RATE),
        ticks: frames,
    }])
    .map_err(|err| err.reason().to_string())?;
    Ok(mux.into_inner().into_inner())
}

fn silence() -> Result<Vec<u8>, String> {
    let channels = usize::from(CHANNELS);
    let frame = channels
        .checked_mul(2)
        .ok_or_else(|| "frame size overflow".to_owned())?;
    let len = frame
        .checked_mul(FRAMES)
        .ok_or_else(|| "pcm size overflow".to_owned())?;
    Ok(vec![0; len])
}
