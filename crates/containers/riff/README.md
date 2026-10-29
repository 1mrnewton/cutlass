# cutlass-riff

## What this crate is

`cutlass-riff` demuxes and muxes WAV. Classic RIFF WAVE and RF64 are both in scope. The implementation is native safe Rust, and the crate forbids `unsafe`. It finds chunks, the codec id, extradata, and timestamps. The timestamp unit is one tick of the sample rate. Sample bytes stay in the packet.

FFmpeg 8.1.1 at `/opt/homebrew/bin/ffmpeg` is the benchmark oracle. The crate links `cutlass-codec` and `cutlass-container`.

## Specification

- Multimedia Programming Interface and Data Specifications 1.0 (IBM/Microsoft, 1991), for RIFF and WAVE.
- Microsoft `WAVEFORMATEX` and Multiple Channel Audio Data and WAVE Files, for `WAVEFORMATEXTENSIBLE`.
- EBU Tech 3306, for RF64. The first chunk inside an RF64 file is `ds64`.

A declared chunk size that runs past its parent is an error. The longer record of sources and limits is `SUPPORT.md`.

## API

`WavDemux` implements `cutlass_container::Demux`. `WavMux` implements `cutlass_container::Mux`. `probe` and `WavFamily` recognize a 12-byte header. `WavFormat` is the parsed `fmt ` chunk: tag, channel count, sample rate, block align, bits, float, and extensible layout.

`WavDemux::open` takes a random-access reader and the file length. `open_forward` refuses seek and stops at `data`. `duration_samples` is the sample count for a constant block size. MPEG leaves that count empty until the frames are read. `read_packet` returns one access unit. For PCM, A-law, and μ-law that unit is one interleaved sample frame. For MPEG tags `0x0050` and `0x0055` it is one MPEG frame. For other compressed audio it is one `nBlockAlign` block, or the whole `data` chunk when the block align is 1 and the chunk fits in the packet limit. Packet timestamps use the sample rate as their timescale.

```rust
use std::fs::File;

use cutlass_container::Demux;
use cutlass_riff::WavDemux;

let file = File::open(path)?;
let length = file.metadata()?.len();
let mut demux = WavDemux::open(file, length)?;
let stream = &demux.streams()[0];
let rate = stream.params().sample_rate();
let channels = stream.params().channels();
while let Some(packet) = demux.read_packet()? {
    let _ticks = packet.timestamp().presentation();
    let _bytes = packet.bytes();
}
```

`WavMux::new` writes RIFF WAVE. `WavMux::rf64` writes RF64 and `ds64` even when the file would fit in a 32-bit RIFF size. `fact` is written when the stored format tag is not `0x0001`. Integer PCM stays `PCMWAVEFORMAT` unless the stream carries a channel mask, more than two channels, a valid-bit width that differs from the stored width, or the source was already extensible. Finish checks the track duration against the packets, then writes the file.

```rust
use std::io::Cursor;

use cutlass_codec::{CodecId, StreamCodec, StreamParams, Timescale};
use cutlass_container::{Mux, NewStream, TrackDuration, TrackId};
use cutlass_riff::WavMux;

let params = StreamParams::try_audio(
    StreamCodec::Known(CodecId::Pcm),
    44_100,
    2,
    Vec::new(),
)?;
let mut mux = WavMux::new(Cursor::new(Vec::new()));
mux.add_stream(NewStream {
    params,
    timescale: Timescale::new(44_100),
    language: None,
})?;
mux.write_packet(&packet)?;
mux.finish(&[TrackDuration {
    track: TrackId::new(0),
    timescale: Timescale::new(44_100),
    ticks,
}])?;
let bytes = mux.into_inner().into_inner();
```

## Support

One audio track. PCM duration is the `data` chunk length divided by the block align.

| Kind | Demux | Mux |
| --- | --- | --- |
| RIFF WAVE | Opens | Writes |
| RF64 and `ds64` | `0xFFFFFFFF` sizes come from `ds64` | `WavMux::rf64` writes `ds64` for a small file too |
| PCM integer (`0x0001`) | One sample frame per packet | `PCMWAVEFORMAT`, no `fact` |
| IEEE float (`0x0003`) | One sample frame per packet | Writes `fact` |
| `WAVEFORMATEXTENSIBLE` | Channel mask at extradata offset 8 | Written for a mask, more than two channels, a valid-bit width, or a source that was already extensible |
| A-law (`0x0006`) and μ-law (`0x0007`) | One sample frame per packet | Refused |
| MPEG in WAV (`0x0050`, `0x0055`) | One MPEG frame per packet. `0x0055` is `CodecId::Mp3` | One frame per packet |
| ADPCM and other block codecs | One block per packet. IMA ADPCM is `0x0011` | Packet duration must match the block |
| AAC, AC-3, and FLAC tags | Mapped to those codec ids | Stored when the packet framing matches |
| Unknown codec tag | Stream kept, tag `wav:XXXX` | Refused |
| Unknown chunk up to 64 KiB | Kept | Written back before `data` |
| Unknown chunk above 64 KiB | Named on `DroppedChunk` | Left out |
| `fact` | PCM uses the data size when `fact` disagrees | Written when the stored tag is not `0x0001` |
| Odd chunk pad | Pad is outside the payload | Pad byte written outside the size field |
| RIFX, BW64 | Family matches, `open` returns `UnsupportedBrand` | Left unwritten |

A 96 kHz 16-bit stereo stream stays `PCMWAVEFORMAT`. `WAVEFORMATEXTENSIBLE` is written for a channel mask, a channel count above two, a valid-bit width, or a source that was already extensible.

## Accuracy

One comparison, run before the timed reads. ffmpeg wrote a 16-bit stereo WAV. This crate opened it. ffprobe and an ffmpeg PCM read are the oracles. ffprobe groups PCM into 11 packets. This crate returns one sample frame per packet, so the packet counts differ. The sample count, the timestamp span, and the concatenated payload are what the check requires.

Copied from that run:

```text
accuracy codec cutlass=pcm ffprobe=pcm_s16le
accuracy sample_rate cutlass=44100 ffprobe=44100
accuracy channels cutlass=2 ffprobe=2
accuracy packets cutlass=44100 ffprobe=11
accuracy samples=44100 ffprobe_duration_ts=44100 ffprobe_duration_sum=44100
accuracy first_pts cutlass=0 ffprobe=0 last_pts cutlass=44099 ffprobe=40960 ffprobe_start_pts=N/A
accuracy span cutlass=0+44100 ffprobe=0+44100
accuracy payload_bytes cutlass=176400 ffmpeg=176400
mux codec=pcm_s16le sample_rate=44100 channels=2 duration_ts=44100 packets=11 first_pts=0
```

Sample rate is 44100 on both sides. Channel count is 2. This crate read 44100 packets and 44100 samples. ffprobe `duration_ts` is 44100, and the sum of ffprobe packet durations is 44100. The first presentation timestamp is 0 on both sides. The stream `start_pts` field is `N/A`. The last packet timestamp is 44099 here and 40960 in ffprobe, because ffprobe's last packet holds the remaining frames. Both spans are `0+44100`. The payload is 176400 bytes here and 176400 bytes from ffmpeg's `s16le` read.

The same PCM, written by `WavMux` and read by ffprobe, is `pcm_s16le`, 44100 Hz, 2 channels, `duration_ts` 44100, 11 ffprobe packets, first timestamp 0.

## Performance

One sample. Median of 3 timed passes after 1 warmup. The ffmpeg read starts a new process on every pass, so that time includes process startup. The cutlass read stays in this process: open the file and pull every packet.

Command: `cargo bench -p cutlass-riff`

`sysctl -n machdep.cpu.brand_string`: `Apple M5 Pro`

Date: `Thu Oct 29 21:08:47 GMT 2026`

ffmpeg version: `ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers`

Bench stdout:

```text
cpu: Apple M5 Pro
date: Thu Oct 29 21:08:47 GMT 2026
command: cargo bench -p cutlass-riff
ffmpeg: /opt/homebrew/bin/ffmpeg
ffprobe: /opt/homebrew/bin/ffprobe
ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers
sample: one 16-bit stereo WAV written by ffmpeg
reps: 3 after 1 warmup, median of the timed passes
ffmpeg time includes process startup
accuracy codec cutlass=pcm ffprobe=pcm_s16le
accuracy sample_rate cutlass=44100 ffprobe=44100
accuracy channels cutlass=2 ffprobe=2
accuracy packets cutlass=44100 ffprobe=11
accuracy samples=44100 ffprobe_duration_ts=44100 ffprobe_duration_sum=44100
accuracy first_pts cutlass=0 ffprobe=0 last_pts cutlass=44099 ffprobe=40960 ffprobe_start_pts=N/A
accuracy span cutlass=0+44100 ffprobe=0+44100
accuracy payload_bytes cutlass=176400 ffmpeg=176400
mux codec=pcm_s16le sample_rate=44100 channels=2 duration_ts=44100 packets=11 first_pts=0

side     op             median          min          max        bytes
cutlass  read         13568 µs     13502 µs     13617 µs       176400
ffmpeg   read         26613 µs     26355 µs     26740 µs       176400
```

The cutlass median is 13568 µs. The ffmpeg median is 26613 µs.

## Example

`examples/demux.rs` writes eight silent 16-bit stereo frames at 44100 Hz with `WavMux`, opens that buffer with `WavDemux`, and prints the stream.

```text
cargo run --release -p cutlass-riff --example demux
```

```text
sample_rate 44100
channels 2
packets 8
samples 8
```

## Gaps

`open` returns `UnsupportedBrand` for RIFX and BW64, and the muxer leaves both unwritten. The muxer refuses A-law, μ-law, and an unknown codec tag. A language value on `NewStream` is accepted and left out of the file. A forward reader stops at `data`, and seek on that reader is `SeekError::ForwardOnly`. AIFF, wavl, and SMV are outside this crate. Payload bytes are returned as stored. `crates/engine` and `crates/export` leave this family unregistered, so playback and export were not run. The timed read is the one generated PCM file above.
