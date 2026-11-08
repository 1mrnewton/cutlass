# AC-3 and E-AC-3 test fixtures

Made on macOS with FFmpeg 8.1.1 (`ac3` and `eac3` encoders) from synthetic
tones, 0.25 s each. Every channel carries its own sine from FFmpeg's `sine`
source, so a test can tell the channels apart:

| Channel | Tone |
| --- | --- |
| front left | 300 Hz |
| front right | 500 Hz |
| front center | 700 Hz |
| LFE | 60 Hz |
| side or back left | 900 Hz |
| side or back right | 1100 Hz |
| back center | 1300 Hz |

The command is `ffmpeg` with one `-f lavfi -i sine=f=F:sample_rate=R:d=0.25`
input per channel, joined with `join=inputs=N:channel_layout=L:map=0.0-FL|1.0-FR|...` (each
input mapped to its speaker by name), then
`-c:a CODEC -b:a RATE` and the options below. All are free of third-party
content.

| File | Layout | Rate | Options | Exercises | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `ac3_mono_48000.ac3` | mono | 96k | | 1/0 | `06210f8c08c3c489530f713ee01039f7ea9395aad71fb034a4f9a9ccfe9ff034` |
| `ac3_stereo_44100.ac3` | stereo | 192k | | 2/0 at 44.1 kHz, coupling, rematrixing | `53f4551271bd52a7a98270a7f322ef8231ef9d048ebf3fe12e9332df702d49d3` |
| `ac3_3.0_32000.ac3` | 3.0 | 192k | | 3/0 at 32 kHz | `5c3996a94ed4dccb239337da3038b2d4efbee618755d632f2f2e4d4d6386c426` |
| `ac3_3.0-back_48000.ac3` | 3.0(back) | 192k | | 2/1, back center | `8851f1a1204b73d9e39ce070e414bd2141a2f01253c2ab6f02b8178b1790c84b` |
| `ac3_quad_48000.ac3` | quad | 256k | | 2/2 | `3fbdec9d3554763e144b0805d12a4b8264fe820bd8e056c529fb877b4e8da4b2` |
| `ac3_5.1_48000_xbsi.ac3` | 5.1(side) | 448k | `-dialnorm -20 -dmix_mode ltrt -ltrt_cmixlev 0.595 -ltrt_surmixlev 0.5 -loro_cmixlev 0.841 -loro_surmixlev 0.0 -dsurex_mode on -ad_conv_type hdcd` | 3/2 with LFE, alternate bit stream syntax (`bsid` 6) | `5c70d46964d4dfd4c62924dafac15cfb5c202793e9e612e82a46fa4679aa65e5` |
| `ac3_2.1_44100.ac3` | 2.1 | 192k | `-dsur_mode on -room_type small -mixing_level 100` | 2/0 with LFE, audio production information | `6b35a53641c0723bc30718e30d4c805ae7d4be2532940f778b03ddaa39e2c2b0` |
| `eac3_stereo_48000.eac3` | stereo | 96k | | E-AC-3 2/0, coupling, rematrixing | `4ac566180c87a9a1fba62715f2f64e396274824bbc0dae5bf1e6c32893964fa7` |
| `eac3_5.1_44100.eac3` | 5.1(side) | 384k | `-dialnorm -24 -dmix_mode loro -center_mixlev 0.595 -surround_mixlev 0.5` | E-AC-3 3/2 with LFE at 44.1 kHz, mixing metadata | `d062c14f14f6e58bbfeddc22754c0c6bd69d49803c821575ac37ed8676f24d2a` |
| `eac3_mono_32000.eac3` | mono | 64k | | E-AC-3 1/0 at 32 kHz | `b32064637ed3244ad520077778bc8abe31e3d183cb4afe556b7573dd635b27bb` |
| `eac3_5.0_48000.eac3` | 5.0(side) | 320k | | E-AC-3 3/2 without LFE | `4c637d2b84542460a68e3c6ca9666d53ad94610f84df52d8f4e58dcfa7546215` |

FFmpeg's encoders do not write spectral extension, the adaptive hybrid
transform, dependent substreams, frames shorter than six blocks or block
switching. Those tools are checked against the FFmpeg FATE samples in
`tests/ac3/fate.rs` when `AC3_FATE_SAMPLES` names a folder that holds them
(`ac3/` and `eac3/` from `https://fate-suite.ffmpeg.org`). The samples are not
committed.

`reference/<name>.f32` is the fixture of the same name decoded by FFmpeg 8.1.1
with `-drc_scale 0`: interleaved little-endian `f32` in WAVEFORMATEXTENSIBLE
order. The committed-PCM test compares against it, so the check runs where
FFmpeg is not installed. A live FFmpeg decode and macOS `afconvert` are still
used when they are.

Each one is made from this folder with:

```text
ffmpeg -v error -y -drc_scale 0 -i NAME -f f32le -c:a pcm_f32le reference/NAME.f32
```

FFmpeg 8.1.1 on macOS reproduced every file byte for byte on 8 November 2026.

| File | SHA-256 |
| --- | --- |
| `reference/ac3_2.1_44100.ac3.f32` | `50f1ce9842cb21a687761ca7e197163fc33b44bc9efc0a02b41f9c3ac2f1d109` |
| `reference/ac3_3.0-back_48000.ac3.f32` | `c54ea6f7501c2829225ea66ce30b360650c323b3234bc9964732a7a4bdeb1692` |
| `reference/ac3_3.0_32000.ac3.f32` | `6222f1d155ceab79c6be3b03baf79975976d9b6f4a573644b2bc4da849b69aa1` |
| `reference/ac3_5.1_48000_xbsi.ac3.f32` | `e25e07ca1e7f07dc2ee4d04aafa4847ecc55c78fbf825c825bacc2a57b9f1aa7` |
| `reference/ac3_mono_48000.ac3.f32` | `35b2ab74564da6a04bfbfad65b07f2f548c8d35ec1093e893be4497d70cc1087` |
| `reference/ac3_quad_48000.ac3.f32` | `938f46af381c0985365e41a4e503156e5c5dc1d8f518c852344d7a359fc2ec01` |
| `reference/ac3_stereo_44100.ac3.f32` | `7c100fbcc4d6cdcf2dee71738b988adeab372229ead3a2afcb64110a2a75044e` |
| `reference/eac3_5.0_48000.eac3.f32` | `91c198e4d99217c77c813a2c8a5c2fa54a2a8067c2bde1b9233003b4f11b5700` |
| `reference/eac3_5.1_44100.eac3.f32` | `f7acda93e34f51800daf35cbda6909540667c5ac3d4c4cf71c422ded1c863157` |
| `reference/eac3_mono_32000.eac3.f32` | `30a698d5eec606f9bb9cd14b95c49490cced8ddd113f805141198120555c5e25` |
| `reference/eac3_stereo_48000.eac3.f32` | `0438370a3ea443d0e5f3be4cbd33e8bdf64a65cb3622bf60a593cd158b29dbc0` |
