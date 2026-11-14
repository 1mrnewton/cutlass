# MPEG-TS fixtures

Small synthetic files from other muxers, so the tests and the fuzz corpus
see layouts this crate's muxer does not write. All pictures and sounds are
generated test signals (`testsrc2`, `testsrc`, `sine`, `videotestsrc`,
`audiotestsrc`); there is no third-party content. Each is 1.2 seconds,
96x64 at 25 fps unless the table says otherwise.

`tests/fixtures.rs` checks per track the packet count, keyframes and first
and last PTS. Those numbers were compared packet by packet (PTS, DTS,
duration, keyframe and size, with ADTS headers taken off and AAC 44.1 kHz
times within one tick) with `ffprobe -show_packets` from FFmpeg 8.1.1, and
all eight files matched.

| File | Producer | Layout |
| --- | --- | --- |
| `h264_aac.ts` | FFmpeg 8.1.1, libx264 0.165.3222, native AAC | H.264 with 2 B-frames, GOP 12; ADTS AAC mono |
| `hevc_ac3.ts` | FFmpeg 8.1.1, libx265 4.2, native AC-3 | HEVC with 2 B-frames, GOP 12; AC-3 (0x81) |
| `mpeg2_mp2.ts` | FFmpeg 8.1.1 native MPEG-2 and MP2 | MPEG-2 video, GOP 6, 2 B-frames; MPEG audio layer II |
| `h264_latm.ts` | FFmpeg 8.1.1, `-mpegts_flags latm` | H.264; AAC in LATM/LOAS (0x11) |
| `h264_opus.ts` | FFmpeg 8.1.1, libopus | H.264; Opus (0x06, `Opus` registration) |
| `h264_eac3.m2ts` | FFmpeg 8.1.1, `-mpegts_m2ts_mode 1` | 192-byte packets; `HDMV` program registration, E-AC-3 as 0x84 with its own `EAC3` registration |
| `two_programs.ts` | FFmpeg 8.1.1, two `-program`s | program 1: H.264 + MP3 48 kHz; program 2: 64x48 30 fps H.264 + MP3 44.1 kHz |
| `gst_h264_aac.ts` | GStreamer 1.28.3 `mpegtsmux`, x264enc, fdkaacenc | PIDs 0x41/0x42, PTS starting at 3600 s, 50 AAC frames |

Commands (FFmpeg as `ffmpeg -v error -y`):

```sh
V="-f lavfi -i testsrc2=size=96x64:rate=25:duration=1.2"
A="-f lavfi -i sine=frequency=700:sample_rate=48000:duration=1.2"
ffmpeg $V $A -c:v libx264 -preset ultrafast -bf 2 -g 12 -c:a aac -b:a 32k h264_aac.ts
ffmpeg $V $A -c:v libx265 -x265-params log-level=none:bframes=2:keyint=12 -c:a ac3 -b:a 64k hevc_ac3.ts
ffmpeg $V $A -c:v mpeg2video -g 6 -bf 2 -c:a mp2 -b:a 64k mpeg2_mp2.ts
ffmpeg $V $A -c:v libx264 -preset ultrafast -c:a aac -b:a 32k -mpegts_flags latm h264_latm.ts
ffmpeg $V $A -c:v libx264 -preset ultrafast -c:a libopus -b:a 32k h264_opus.ts
ffmpeg $V $A -c:v libx264 -preset ultrafast -c:a eac3 -b:a 64k -mpegts_m2ts_mode 1 h264_eac3.m2ts
ffmpeg $V $A -f lavfi -i testsrc=size=64x48:rate=30:duration=1.2 \
  -f lavfi -i sine=frequency=900:sample_rate=44100:duration=1.2 \
  -map 0:v -map 1:a -map 2:v -map 3:a -c:v libx264 -preset ultrafast -c:a libmp3lame -b:a 32k \
  -program title=One:st=0:st=1 -program title=Two:st=2:st=3 two_programs.ts
gst-launch-1.0 -q videotestsrc num-buffers=30 ! video/x-raw,width=96,height=64,framerate=25/1 \
  ! x264enc bframes=2 key-int-max=12 ! h264parse ! mux. \
  audiotestsrc num-buffers=50 ! audio/x-raw,rate=48000,channels=2 ! fdkaacenc ! aacparse ! mux. \
  mpegtsmux name=mux ! filesink location=gst_h264_aac.ts
```

Encoders are not bit-exact across versions, so regenerated files differ;
the checksums are of the committed files.

| File | SHA-256 |
| --- | --- |
| `gst_h264_aac.ts` | `1d198d0a30bea4b1bfbffa56f1789200e8f1a40ddc2b123fc00b0c6c1e5c73f4` |
| `h264_aac.ts` | `ccc305ceb4cff54bca233fef47caa301020f3a9741b01b153ad5a78d00cebd29` |
| `h264_eac3.m2ts` | `9dfc8a233bb66ac89e05f5a7054dd157384bce617eb940caf7c5e7e21acccfa1` |
| `h264_latm.ts` | `1789cec6500459b999e0122c46ed29682fabe3c2dda8562ea62545369d0be0eb` |
| `h264_opus.ts` | `b120e67da978a01d9e7a1c99ce85df1c35f0300c7b541ff0a7bf0590a314df29` |
| `hevc_ac3.ts` | `4613fc61ed635d66d00cf62454fc994c13600bd46f8c59b42ecb2ac1de8ba742` |
| `mpeg2_mp2.ts` | `3ef2fa482797713ed9942513634cd8218e6da6b7533578cb25e6ed1374ce7765` |
| `two_programs.ts` | `79e2e795a498447578a85b428b72433ca01a2e8f0a734cde6bb93f8f8f32f25c` |
