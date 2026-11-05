//! A real ALAC `.m4a` from AudioToolbox through the whole chain: MP4
//! demux, the registered decoder, the registered encoder, the workspace MP4
//! muxer, and back out through ffmpeg, `afinfo`, and `afconvert`.

use std::path::Path;
use std::process::Command;

use cutlass_decode::{CodecId, DecodeFactory, Output, Packet, SampleFormat};
use cutlass_encode::{AudioInput, EncodeFactory};

use super::decode_oracle::{EXPO, real_source};
use super::{
    afconvert, afconvert_samples, decode_all, demux_mp4, ffmpeg, ffmpeg_samples, params, path_str,
    run, scratch, skip, write_m4a,
};
use crate::Cookie;

const FIVE_ONE: &str =
    "pan=5.1|FL=c0|FR=c1|FC=0.5*c0+0.5*c1|LFE=0.3*c0|BL=0.7*c0-0.2*c1|BR=0.6*c1-0.1*c0";

/// `afinfo`'s format and layout lines, which must not change.
fn afinfo_lines(file: &Path) -> Vec<String> {
    let output = Command::new("/usr/bin/afinfo")
        .arg(file)
        .output()
        .expect("afinfo");
    assert!(output.status.success());
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| {
            line.starts_with("Data format")
                || line.starts_with("Channel layout")
                || line.contains("no channel layout")
        })
        .map(str::to_owned)
        .collect()
}

/// Demux `source`, decode and re-encode through the factories, and write
/// `out`. Returns the source packets, the new packets and cookie, and the
/// valid samples.
fn transcode(source: &Path, out: &Path) -> (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<u8>) {
    let track = demux_mp4(source);
    let cookie = Cookie::parse(&track.cookie).expect("cookie");
    let layout = cookie.channel_layout();
    let channels = layout.len();

    let mut decoders = DecodeFactory::new();
    crate::register_decoders(&mut decoders);
    let mut decoder = decoders
        .open(&params(&track.cookie, layout.clone()))
        .expect("decoder");
    let mut encoders = EncodeFactory::new();
    crate::register_encoders(&mut encoders);
    // The source cookie asks for the same depth and frame length.
    let mut encode_params = params(&track.cookie, layout);
    encode_params.sample_format = Some(if cookie.config.bit_depth == 16 {
        SampleFormat::I16
    } else {
        SampleFormat::F32
    });
    let mut encoder = encoders.open(&encode_params).expect("encoder");
    assert_eq!(encoder.codec(), CodecId::Alac);

    let mut packets = Vec::new();
    let mut pts = 0;
    for (data, &duration) in track.packets.iter().zip(&track.durations) {
        let packet = Packet::audio(data.clone(), pts, duration, track.rate);
        pts += duration;
        let outputs = decoder.push(&packet).expect("decode");
        let [Output::Audio(audio)] = outputs.as_slice() else {
            panic!("one audio output per packet");
        };
        // The last packet may hold padding past its duration.
        let keep = usize::try_from(duration).expect("duration") * channels;
        let mut pcm = audio.pcm.clone();
        if (pcm.samples as usize) * channels > keep {
            pcm = match pcm.format {
                SampleFormat::I16 => cutlass_decode::Pcm::from_i16(
                    pcm.rate,
                    pcm.layout.clone(),
                    pcm.as_i16().expect("i16")[..keep].to_vec(),
                ),
                SampleFormat::F32 => cutlass_decode::Pcm::from_f32(
                    pcm.rate,
                    pcm.layout.clone(),
                    pcm.as_f32().expect("f32")[..keep].to_vec(),
                ),
            }
            .expect("trimmed pcm");
        }
        packets.extend(encoder.push(&AudioInput { pcm }).expect("encode"));
    }
    packets.extend(encoder.drain().expect("drain"));
    let new_cookie = packets
        .iter()
        .flat_map(|packet| &packet.side)
        .rev()
        .find_map(|side| match side {
            cutlass_decode::SideData::Config(cookie) => Some(cookie.clone()),
            _ => None,
        })
        .expect("cookie");
    write_m4a(out, &new_cookie, &packets);
    (
        track.packets,
        packets.into_iter().map(|packet| packet.data).collect(),
        new_cookie,
    )
}

#[test]
fn an_apple_m4a_survives_decode_encode_and_mux_bit_for_bit() {
    let Some(ff) = ffmpeg() else {
        skip("ffmpeg");
        return;
    };
    let Some(af) = afconvert() else {
        skip("afconvert");
        return;
    };
    let dir = scratch("end-to-end");
    for (name, codec, flag, filter) in [
        ("stereo-16", "pcm_s16le", "alac/1", None),
        ("stereo-24", "pcm_s24le", "alac/3", None),
        ("five-one-16", "pcm_s16le", "alac/1", Some(FIVE_ONE)),
    ] {
        let Some(wav) = real_source(&ff, &dir, EXPO, "9.3", codec, filter) else {
            return;
        };
        let source = dir.join(format!("apple-{name}.m4a"));
        run(
            &af,
            &["-f", "m4af", "-d", flag, path_str(&wav), path_str(&source)],
        );
        let out = dir.join(format!("cutlass-{name}.m4a"));
        let (theirs, ours, cookie) = transcode(&source, &out);

        // Apple's encoder and ours write the same frames; only the last,
        // which Apple pads to a full frame, is shorter here.
        assert_eq!(theirs.len(), ours.len(), "{name}");
        let last = ours.len() - 1;
        assert!(theirs[..last] == ours[..last], "{name}: frames differ");

        // The cookie comes back from the file as it went in, and describes
        // the source stream.
        let reopened = demux_mp4(&out);
        assert_eq!(reopened.cookie, cookie, "{name}");
        let source_cookie = Cookie::parse(&demux_mp4(&source).cookie).expect("cookie");
        let new = Cookie::parse(&cookie).expect("cookie");
        assert!(new.config.decodes_like(&source_cookie.config), "{name}");
        assert_eq!(new.layout_tag, source_cookie.layout_tag, "{name}");

        // Every decoder reads the same samples from both files.
        let depth = u32::from(new.config.bit_depth);
        let valid: i64 = demux_mp4(&source).durations.iter().sum();
        let channels = usize::from(new.config.channels);
        let valid = usize::try_from(valid).expect("valid") * channels;
        let (ours_decoded, _) = decode_all(&reopened.cookie, &reopened.packets);
        let (source_decoded, _) =
            decode_all(&source_cookie.to_bytes(), &demux_mp4(&source).packets);
        assert_eq!(ours_decoded.len(), valid, "{name}");
        assert!(ours_decoded == source_decoded[..valid], "{name}");
        assert!(
            ffmpeg_samples(&ff, &out, depth) == ffmpeg_samples(&ff, &source, depth),
            "{name}: ffmpeg"
        );
        let apple = afconvert_samples(&af, &out, depth);
        assert!(apple == ours_decoded, "{name}: afconvert");
        assert_eq!(afinfo_lines(&out), afinfo_lines(&source), "{name}: afinfo");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
