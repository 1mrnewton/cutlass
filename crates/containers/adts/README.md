# cutlass-adts

## What this crate is

`cutlass-adts` frames one AAC-LC stream as raw ADTS. It is safe Rust: the crate sets `#![forbid(unsafe_code)]`, and the workspace lint forbids `unsafe` as well. A read returns the raw access unit with the ADTS header removed. A write puts that access unit back into an ADTS frame.

ADTS stores no per-frame timestamp. Each raw block is stamped on the sample-rate clock: presentation time starts at 0, and each block lasts 1024 samples.

The same package also exposes LOAS/LATM (`LatmFile`, `LatmDemuxer`, `LatmMuxer`) and a one-track M4A reader and writer (`Mp4Demuxer`, `Mp4Muxer`). The rest of this note is the ADTS path.

## Specification

ISO/IEC 13818-7:2006 clause 6.2 defines the ADTS frame: syncword, fixed header, variable header, then the raw block, with an optional CRC. The CRC is the check in ISO/IEC 11172-3 subclause 2.4.3.1. MPEG-4 ADTS uses the same frame with the ID bit clear. This crate accepts 7350 Hz only in that MPEG-4 header.

The parser reads these fields:

- syncword `0xFFF`
- ID: 1 means MPEG-2, 0 means MPEG-4
- layer, which must be 0
- `protection_absent`
- profile, accepted only when the 2-bit value is 1 (AAC-LC, audio object type 2)
- `sampling_frequency_index`
- `channel_configuration`
- `aac_frame_length`, counting the header, at most 8191 bytes
- `number_of_raw_data_blocks_in_frame` (the frame holds that value plus one raw block)

On read, the private bit, original/copy, home, copyright bits, and buffer fullness are ignored. The muxer writes those bits as 0, and writes buffer fullness `0x7FF` (variable rate).

| index | Hz | header |
| --- | --- | --- |
| 0 | 96000 | MPEG-2 and MPEG-4 |
| 1 | 88200 | MPEG-2 and MPEG-4 |
| 2 | 64000 | MPEG-2 and MPEG-4 |
| 3 | 48000 | MPEG-2 and MPEG-4 |
| 4 | 44100 | MPEG-2 and MPEG-4 |
| 5 | 32000 | MPEG-2 and MPEG-4 |
| 6 | 24000 | MPEG-2 and MPEG-4 |
| 7 | 22050 | MPEG-2 and MPEG-4 |
| 8 | 16000 | MPEG-2 and MPEG-4 |
| 9 | 12000 | MPEG-2 and MPEG-4 |
| 10 | 11025 | MPEG-2 and MPEG-4 |
| 11 | 8000 | MPEG-2 and MPEG-4 |
| 12 | 7350 | MPEG-4 only |

Indexes 13, 14, and 15 are rejected. Index 15 is the explicit-rate escape.

`protection_absent = 1` means a 7-byte header and one or more raw blocks with no CRC. More than one raw block in that form is rejected, because the byte positions are missing.

`protection_absent = 0` and one raw block: a 16-bit CRC follows the 7-byte header. The CRC and the header are stored on the packet as `SideData::Protection`. That CRC is not checked.

`protection_absent = 0` and several raw blocks: the header is followed by a 16-bit position for each block after the first, then the header CRC (`adts_header_error_check` over the fixed header, the variable header, and those positions). Each raw block is followed by its own 16-bit CRC. The packet payload is the block with that CRC removed. The header CRC is checked. The per-block CRC is stored and left unchecked.

## API

`AdtsFile<R: Read>` is the shared demuxer. It implements `cutlass_container::Demux`.

```rust
use cutlass_adts::AdtsFile;
use cutlass_container::Demux;

let mut file = AdtsFile::open(reader)?;
let streams = file.streams();
while let Some(packet) = file.read_packet()? {
    let pts = packet.timestamp().presentation();
    let timescale = packet.timescale().ticks_per_second();
    let unit = packet.bytes();
    let _ = (pts, timescale, unit);
}
```

`family` is `ContainerId::Adts`. The stream timescale is the sample rate. `gaps` is empty. `seek` returns `ReadError::Seek(SeekError::ForwardOnly)`. `cancel` makes the next read return `ReadError::Cancelled`. Open reads until the first frame has produced a stream. A trailing partial frame is `CodecError::Truncated`, surfaced as an open or read error.

`AdtsDemuxer` is the byte feeder `AdtsFile` uses. It implements the crate `Demuxer` trait (`id`, `streams`, `next_packet`). `push_bytes` and `finish_bytes` are inherent. `next_packet` returns `cutlass_codec::access::Packet`: `data` is the raw access unit, `pts` and `duration` are sample-rate ticks, and `timescale` is the sample rate.

```rust
use cutlass_adts::{AdtsDemuxer, Demuxer};

let mut demux = AdtsDemuxer::new();
demux.push_bytes(bytes)?;
demux.finish_bytes()?;
while let Some(packet) = demux.next_packet()? {
    let _ = (&packet.data, packet.pts, packet.duration, packet.timescale);
}
```

`AdtsMuxer` implements the crate `Muxer` trait (`id`, `add_stream`, `write_packet`, `finish`).

```rust
use cutlass_adts::{AdtsMuxer, Muxer};
use cutlass_codec::CodecId;
use cutlass_codec::access::{ChannelLayout, CodecParams, Packet};

let mut mux = AdtsMuxer::new();
mux.add_stream(CodecParams::audio(
    CodecId::Aac,
    48_000,
    ChannelLayout::mono(),
))?;
mux.write_packet(&Packet::audio(unit, 0, 1024, 48_000))?;
let adts = mux.finish()?;
```

`set_mpeg2(true)` sets the MPEG-2 ID bit before `add_stream`. `write_blocks` writes two, three, or four raw blocks in one frame; each packet carries `SideData::Protection` for that block. `bytes` returns the ADTS written so far. A second `finish` is `CodecError::InvalidApi`. There is no `cutlass_container::Mux` implementation.

`StreamInfo` is the stream index, `CodecParams`, and the timescale. Also public, and outside this ADTS API: `LatmDemuxer`, `LatmMuxer`, `LatmFile`, `Mp4Demuxer`, `Mp4Muxer`.

## Support

Demux accepts one AAC-LC ADTS stream.

- MPEG-4 (ID 0) and MPEG-2 (ID 1).
- The sample rates in the table above.
- Channel configuration 1 mono, 2 stereo, 3 front center/left/right, 4 that set plus back center, 5 front plus back left/right, 6 the 5.1 layout, 7 the wide 7.1 layout.
- One raw block per frame, with or without the 16-bit CRC.
- Several raw blocks when the header carries their positions and the header CRC matches.
- A sync search that skips non-header bytes, up to 1 MiB.
- At most 1 MiB buffered. `AdtsFile` pulls the reader in 4096-byte reads.

A configuration change (sample rate, channel layout, or MPEG ID) is rejected. Profile 0, 2, and 3 are rejected.

Mux writes one AAC stream, profile 1, a 7-byte header, buffer fullness `0x7FF`, and one raw block per `write_packet`. `write_blocks` writes up to four blocks. A second stream, a codec other than AAC, an empty packet, a frame longer than 8191 bytes, or a layout that maps to no channel configuration 1 through 7 is an error.

## Accuracy

ffprobe 8.1.1 read one ADTS file that ffmpeg wrote in the bench: lavfi `sine=frequency=440:sample_rate=48000:duration=0.5`, AAC at 64 kb/s, format `adts`. These figures are the `frame` rows from that run.

The file is 4506 bytes. Both sides report 25 frames. The stream is 48000 Hz, 1 channel. ffprobe's time base is 1/28224000.

ffprobe `size` is the on-disk ADTS frame. The demuxed payload is that frame minus the header. On every frame the header is 7 bytes and `payload + header = ffprobe_size`. Presentation time matches as ticks of the two clocks: `pts / 48000 = ffprobe_pts / 28224000`. Duration is 1024 samples on every frame, which ffprobe reports as 602112 ticks.

```
index pts duration payload header ffprobe_pts ffprobe_duration ffprobe_size
frame 0 0 1024 190 7 0 602112 197
frame 1 1024 1024 235 7 602112 602112 242
frame 2 2048 1024 158 7 1204224 602112 165
frame 3 3072 1024 213 7 1806336 602112 220
frame 4 4096 1024 197 7 2408448 602112 204
frame 5 5120 1024 148 7 3010560 602112 155
frame 6 6144 1024 185 7 3612672 602112 192
frame 7 7168 1024 161 7 4214784 602112 168
frame 8 8192 1024 160 7 4816896 602112 167
frame 9 9216 1024 172 7 5419008 602112 179
frame 10 10240 1024 200 7 6021120 602112 207
frame 11 11264 1024 191 7 6623232 602112 198
frame 12 12288 1024 177 7 7225344 602112 184
frame 13 13312 1024 137 7 7827456 602112 144
frame 14 14336 1024 151 7 8429568 602112 158
frame 15 15360 1024 154 7 9031680 602112 161
frame 16 16384 1024 156 7 9633792 602112 163
frame 17 17408 1024 158 7 10235904 602112 165
frame 18 18432 1024 201 7 10838016 602112 208
frame 19 19456 1024 216 7 11440128 602112 223
frame 20 20480 1024 147 7 12042240 602112 154
frame 21 21504 1024 140 7 12644352 602112 147
frame 22 22528 1024 162 7 13246464 602112 169
frame 23 23552 1024 174 7 13848576 602112 181
frame 24 24576 1024 148 7 14450688 602112 155
```

`AdtsMuxer` wrote those 25 access units back out. ffprobe read that file as 4506 bytes and the same 25 rows (`mux_bytes 4506`, `mux_frames 25`). Each `mux` row in the bench stdout is the same numbers as the `frame` row of the same index.

## Performance

One sample: the 4506-byte file above. The cutlass figure is a full `AdtsFile` read in this process. The ffprobe figure is a full packet dump in a new process, so it includes process startup. The value is the median of 3 runs after 1 warmup.

Command: `cargo bench -p cutlass-adts`

Date: Thu Oct 29 21:10:40 GMT 2026

`sysctl -n machdep.cpu.brand_string`: Apple M5 Pro

ffmpeg version: `ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers`

From that run, `cutlass_read_us median 15` and `ffprobe_read_us median 18814`. The ffprobe time includes process startup.

Stdout of `cargo bench -p cutlass-adts`:

```
running 22 tests
test latm::tests::a_second_program_is_rejected ... ignored
test latm::tests::ffmpeg_decodes_a_latm_remux_of_its_own_aac ... ignored
test latm::tests::one_program_round_trips_and_reuses_the_config ... ignored
test latm::tests::random_bytes_fail_closed ... ignored
test m4a::tests::a_video_track_is_rejected ... ignored
test m4a::tests::an_adts_sample_is_rejected ... ignored
test m4a::tests::ffmpeg_m4a_round_trips_through_our_boxes ... ignored
test m4a::tests::priming_edit_round_trips_raw_samples ... ignored
test m4a::tests::random_bytes_fail_closed ... ignored
test tests::a_false_sync_before_the_frame_is_skipped ... ignored
test tests::a_protected_block_keeps_its_crc_and_payload ... ignored
test tests::a_raw_unit_round_trips_through_adts ... ignored
test tests::a_truncated_tail_is_not_a_packet ... ignored
test tests::ffmpeg_adts_survives_our_demux_and_mux ... ignored
test tests::mpeg2_rejects_the_7350_hz_index ... ignored
test tests::one_byte_at_a_time_matches_the_whole_buffer ... ignored
test tests::random_bytes_fail_closed ... ignored
test tests::seven_point_one_does_not_fit_the_header ... ignored
test tests::the_shared_demux_returns_the_same_access_unit ... ignored
test tests::thirty_minute_adts_campaign ... ignored
test tests::two_raw_blocks_round_trip_when_positions_are_present ... ignored
test tests::unpositioned_multiple_blocks_are_rejected ... ignored

test result: ok. 0 passed; 0 failed; 22 ignored; 0 measured; 0 filtered out; finished in 0.00s

sample lavfi sine=frequency=440:sample_rate=48000:duration=0.5 codec aac bitrate 64k format adts
file_bytes 4506
frames 25
sample_rate 48000 channels 1 time_base 1/28224000
size_check payload_plus_header_equals_ffprobe_size
pts_check presentation_ticks_match
index pts duration payload header ffprobe_pts ffprobe_duration ffprobe_size
frame 0 0 1024 190 7 0 602112 197
frame 1 1024 1024 235 7 602112 602112 242
frame 2 2048 1024 158 7 1204224 602112 165
frame 3 3072 1024 213 7 1806336 602112 220
frame 4 4096 1024 197 7 2408448 602112 204
frame 5 5120 1024 148 7 3010560 602112 155
frame 6 6144 1024 185 7 3612672 602112 192
frame 7 7168 1024 161 7 4214784 602112 168
frame 8 8192 1024 160 7 4816896 602112 167
frame 9 9216 1024 172 7 5419008 602112 179
frame 10 10240 1024 200 7 6021120 602112 207
frame 11 11264 1024 191 7 6623232 602112 198
frame 12 12288 1024 177 7 7225344 602112 184
frame 13 13312 1024 137 7 7827456 602112 144
frame 14 14336 1024 151 7 8429568 602112 158
frame 15 15360 1024 154 7 9031680 602112 161
frame 16 16384 1024 156 7 9633792 602112 163
frame 17 17408 1024 158 7 10235904 602112 165
frame 18 18432 1024 201 7 10838016 602112 208
frame 19 19456 1024 216 7 11440128 602112 223
frame 20 20480 1024 147 7 12042240 602112 154
frame 21 21504 1024 140 7 12644352 602112 147
frame 22 22528 1024 162 7 13246464 602112 169
frame 23 23552 1024 174 7 13848576 602112 181
frame 24 24576 1024 148 7 14450688 602112 155
mux_bytes 4506
mux_frames 25
mux_size_check payload_plus_header_equals_ffprobe_size
mux_pts_check presentation_ticks_match
mux 0 0 1024 190 7 0 602112 197
mux 1 1024 1024 235 7 602112 602112 242
mux 2 2048 1024 158 7 1204224 602112 165
mux 3 3072 1024 213 7 1806336 602112 220
mux 4 4096 1024 197 7 2408448 602112 204
mux 5 5120 1024 148 7 3010560 602112 155
mux 6 6144 1024 185 7 3612672 602112 192
mux 7 7168 1024 161 7 4214784 602112 168
mux 8 8192 1024 160 7 4816896 602112 167
mux 9 9216 1024 172 7 5419008 602112 179
mux 10 10240 1024 200 7 6021120 602112 207
mux 11 11264 1024 191 7 6623232 602112 198
mux 12 12288 1024 177 7 7225344 602112 184
mux 13 13312 1024 137 7 7827456 602112 144
mux 14 14336 1024 151 7 8429568 602112 158
mux 15 15360 1024 154 7 9031680 602112 161
mux 16 16384 1024 156 7 9633792 602112 163
mux 17 17408 1024 158 7 10235904 602112 165
mux 18 18432 1024 201 7 10838016 602112 208
mux 19 19456 1024 216 7 11440128 602112 223
mux 20 20480 1024 147 7 12042240 602112 154
mux 21 21504 1024 140 7 12644352 602112 147
mux 22 22528 1024 162 7 13246464 602112 169
mux 23 23552 1024 174 7 13848576 602112 181
mux 24 24576 1024 148 7 14450688 602112 155
timing median of 3 after 1 warmup
cutlass_read_us median 15 min 14 max 15 samples 15 14 15
ffprobe_read_us median 18814 min 18657 max 19013 samples 19013 18814 18657
ffprobe time includes process startup
```

Cargo also printed, on the same run:

```
    Finished `bench` profile [optimized] target(s) in 0.03s
     Running unittests src/lib.rs (target/release/deps/cutlass_adts-2f172aa1ad540bbd)
     Running benches/against_ffmpeg.rs (target/release/deps/against_ffmpeg-bba14452e1be2aa0)
```

## Example

```
cargo run --release -p cutlass-adts --example demux
```

```
frames 2
first 0/48000
```

`examples/demux.rs` writes two raw access units with `AdtsMuxer` and reads them with `AdtsFile`. `first 0/48000` is presentation tick 0 on a 48000 Hz clock. The access units are framed bytes, not decoded PCM.

## Gaps

- Seek is forward-only.
- Muxing is `AdtsMuxer`. The shared `cutlass_container::Mux` trait has no ADTS implementation.
- Profiles other than AAC-LC are rejected. The muxer always writes profile 1.
- Channel configuration 0 (a program-config element in the raw block) is rejected. The side-channel 7.1 layout does not fit the 3-bit field. Configuration 7 is the wide 7.1 layout.
- Sample-rate indexes 13, 14, and 15 are rejected. 7350 Hz is rejected when the ID bit is MPEG-2.
- Spectral band replication and parametric stereo are left inside the raw access unit. The header has no field for them here.
- Every raw block is stamped as 1024 samples. A 960-sample window is still stamped 1024.
- The single-block CRC and the per-block CRCs are stored and unchecked. The multi-block header CRC is checked.
- ID3 and APE tags are not parsed. Bytes before a valid header are skipped, up to the 1 MiB resync limit.
- One stream. A mid-file change of rate, channels, or MPEG ID is rejected.
- The desktop package depends on this crate. No editor path opens `AdtsFile`.
- LOAS/LATM and M4A are in this package. The bench measures one ADTS file.
- The ignored `thirty_minute_adts_campaign` test was not run.
- This crate frames AAC. It does not decode AAC to PCM.
