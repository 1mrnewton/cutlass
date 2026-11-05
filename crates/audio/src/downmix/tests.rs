use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use cutlass_codec::access::{Channel, ChannelLayout, Pcm, SampleFormat, StandardLayout};

use super::{downmix, downmix_into};
use crate::Error;

/// -3 dB, written out rather than taken from the module.
const C: f32 = 0.707_106_77;

fn pcm(layout: ChannelLayout, samples: Vec<f32>) -> Pcm {
    Pcm::from_f32(48_000, layout, samples).expect("pcm")
}

fn standard(layout: StandardLayout) -> ChannelLayout {
    ChannelLayout::standard(layout)
}

/// Every layout the gains are checked on: the standard ones, the side and
/// wide variants ffmpeg writes, and AAC's element order.
fn layouts() -> Vec<(&'static str, ChannelLayout)> {
    vec![
        ("3.0", standard(StandardLayout::Surround)),
        ("quad", standard(StandardLayout::Quad)),
        ("5.0", standard(StandardLayout::FiveZero)),
        ("5.1", standard(StandardLayout::FiveOne)),
        ("6.1", standard(StandardLayout::SixOne)),
        ("7.1", standard(StandardLayout::SevenOne)),
        ("5.1(side)", ChannelLayout::from_wave_mask(0x60F).unwrap()),
        ("4.0", ChannelLayout::from_wave_mask(0x107).unwrap()),
        ("2.1", ChannelLayout::from_wave_mask(0xB).unwrap()),
        (
            "7.1(wide-side)",
            ChannelLayout::from_wave_mask(0x6CF).unwrap(),
        ),
        ("aac 5.1", ChannelLayout::five_one()),
        ("aac 7.1 wide", ChannelLayout::seven_one_wide()),
    ]
}

/// The left and right gain each speaker should land with, from the
/// BS.775 table: center and surrounds at -3 dB, back center at -6 dB a
/// side, left and right of center at full level, LFE dropped.
fn expected(channel: Channel) -> (f32, f32) {
    match channel {
        Channel::FrontLeft | Channel::FrontLeftOfCenter => (1.0, 0.0),
        Channel::FrontRight | Channel::FrontRightOfCenter => (0.0, 1.0),
        Channel::FrontCenter => (C, C),
        Channel::BackLeft | Channel::SideLeft => (C, 0.0),
        Channel::BackRight | Channel::SideRight => (0.0, C),
        Channel::BackCenter => (0.5, 0.5),
        Channel::Lfe => (0.0, 0.0),
    }
}

#[test]
fn one_channel_at_a_time_lands_on_its_side_at_its_gain() {
    for (name, layout) in layouts() {
        let channels = layout.len();
        for (index, channel) in layout.channels().iter().enumerate() {
            // Two frames: a full-scale impulse, then half scale negative.
            let mut samples = vec![0.0; channels * 2];
            samples[index] = 1.0;
            samples[channels + index] = -0.5;
            let out = downmix(&pcm(layout.clone(), samples), &ChannelLayout::stereo())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(out.layout, ChannelLayout::stereo());
            assert_eq!(out.samples, 2);
            let (left, right) = expected(*channel);
            let got = out.as_f32().unwrap();
            assert_eq!(
                got,
                [left, right, -0.5 * left, -0.5 * right],
                "{name} {channel:?}"
            );
        }
    }
}

#[test]
fn the_lfe_contributes_nothing() {
    let layout = standard(StandardLayout::FiveOne);
    let lfe = layout
        .channels()
        .iter()
        .position(|channel| *channel == Channel::Lfe)
        .unwrap();
    let mut samples = vec![0.0; 6 * 64];
    for frame in samples.as_chunks_mut::<6>().0 {
        frame[lfe] = 1.0;
    }
    let out = downmix(&pcm(layout, samples), &ChannelLayout::stereo()).unwrap();
    assert!(out.as_f32().unwrap().iter().all(|sample| *sample == 0.0));
}

#[test]
fn silence_stays_silence() {
    for (name, layout) in layouts() {
        let samples = vec![0.0; layout.len() * 32];
        let out = downmix(&pcm(layout, samples), &ChannelLayout::stereo()).unwrap();
        assert_eq!(out.samples, 32, "{name}");
        assert!(
            out.as_f32()
                .unwrap()
                .iter()
                .all(|sample| sample.to_bits() == 0),
            "{name}"
        );
    }
    let empty = pcm(standard(StandardLayout::SevenOne), Vec::new());
    let out = downmix(&empty, &ChannelLayout::stereo()).unwrap();
    assert_eq!(out.samples, 0);
    assert_eq!(out.layout, ChannelLayout::stereo());
}

/// The matrix is not normalised and nothing is clamped: full scale on
/// every channel sums past 1.0, as `swresample` does for float output.
#[test]
fn full_scale_on_every_channel_sums_past_one() {
    let cases = [
        // FL + C FC + C BL (LFE dropped).
        (standard(StandardLayout::FiveOne), 1.0 + 2.0 * C),
        (standard(StandardLayout::FiveZero), 1.0 + 2.0 * C),
        // FL + C FC + C BL + C SL.
        (standard(StandardLayout::SevenOne), 1.0 + 3.0 * C),
        // FL + C FC + 0.5 BC + C SL.
        (standard(StandardLayout::SixOne), 1.5 + 2.0 * C),
        (standard(StandardLayout::Quad), 1.0 + C),
        (standard(StandardLayout::Surround), 1.0 + C),
    ];
    for (layout, peak) in cases {
        let channels = layout.len();
        let mut samples = vec![1.0; channels];
        samples.extend(vec![-1.0; channels]);
        let out = downmix(&pcm(layout.clone(), samples), &ChannelLayout::stereo()).unwrap();
        let got = out.as_f32().unwrap();
        for (index, sample) in got.iter().enumerate() {
            let want = if index < 2 { peak } else { -peak };
            assert!(
                (sample - want).abs() < 1e-6,
                "{layout:?}: {sample} is not {want}"
            );
        }
        assert!(got[0] > 1.0);
    }
}

#[test]
fn stereo_comes_back_bit_for_bit_and_mono_is_not_a_downmix() {
    let odd = vec![
        0.1,
        -0.0,
        f32::MIN_POSITIVE,
        -1.0,
        1.5,
        f32::from_bits(0x3EAA_AAAB),
    ];
    let stereo = pcm(ChannelLayout::stereo(), odd);
    assert_eq!(downmix(&stereo, &ChannelLayout::stereo()).unwrap(), stereo);
    let ints = Pcm::from_i16(44_100, ChannelLayout::stereo(), vec![1, -32_768, 32_767, 0]).unwrap();
    let same = downmix(&ints, &ChannelLayout::stereo()).unwrap();
    assert_eq!(same, ints);
    assert_eq!(same.format, SampleFormat::I16);

    let mono = pcm(ChannelLayout::mono(), vec![0.5, -0.25]);
    assert_eq!(downmix(&mono, &ChannelLayout::stereo()), Err(Error::Layout));
}

#[test]
fn a_two_channel_file_that_is_not_left_and_right_is_folded() {
    let odd = ChannelLayout::from_wave_mask(0x104).unwrap();
    let out = downmix(&pcm(odd, vec![1.0, 1.0]), &ChannelLayout::stereo()).unwrap();
    let got = out.as_f32().unwrap();
    assert!((got[0] - (C + 0.5)).abs() < 1e-6);
    assert!((got[1] - (C + 0.5)).abs() < 1e-6);
}

#[test]
fn integer_samples_fold_at_full_scale_32767() {
    let layout = standard(StandardLayout::FiveOne);
    let ints = Pcm::from_i16(48_000, layout, vec![32_767, 0, 32_767, 32_767, 0, 0]).unwrap();
    let out = downmix(&ints, &ChannelLayout::stereo()).unwrap();
    assert_eq!(out.format, SampleFormat::F32);
    let got = out.as_f32().unwrap();
    assert!((got[0] - (1.0 + C)).abs() < 1e-6);
    assert!((got[1] - C).abs() < 1e-6);
}

#[test]
fn a_bad_target_rate_or_sample_is_refused_and_leaves_nothing() {
    let layout = standard(StandardLayout::FiveOne);
    let surround = pcm(layout.clone(), vec![0.25; 12]);
    let mut out = vec![9.0; 4];
    for target in [ChannelLayout::mono(), standard(StandardLayout::FiveOne)] {
        assert_eq!(
            downmix_into(&surround, &target, &mut out),
            Err(Error::Layout)
        );
        assert!(out.is_empty());
    }
    let silent_rate = Pcm::from_f32(0, layout.clone(), vec![0.0; 6]).unwrap();
    assert_eq!(
        downmix(&silent_rate, &ChannelLayout::stereo()),
        Err(Error::Rate)
    );
    // A NaN on the dropped LFE is still a broken buffer.
    let mut samples = vec![0.0; 12];
    samples[9] = f32::NAN;
    assert_eq!(
        downmix_into(
            &pcm(layout.clone(), samples),
            &ChannelLayout::stereo(),
            &mut out
        ),
        Err(Error::NonFinite)
    );
    assert!(out.is_empty());
    let mut samples = vec![f32::MAX; 6];
    samples[3] = 0.0;
    assert_eq!(
        downmix(&pcm(layout, samples), &ChannelLayout::stereo()),
        Err(Error::NonFinite)
    );
    let broken = pcm(ChannelLayout::stereo(), vec![f32::INFINITY, 0.0]);
    assert_eq!(
        downmix_into(&broken, &ChannelLayout::stereo(), &mut out),
        Err(Error::NonFinite)
    );
}

/// The hot path's bound: a kept buffer with room is written in place, so a
/// window of mix audio allocates nothing, and each frame writes one pair.
#[test]
fn a_kept_buffer_is_reused_without_allocating() {
    let layout = standard(StandardLayout::SevenOne);
    let mut out = Vec::with_capacity(2 * 4_800);
    let start = (out.as_ptr(), out.capacity());
    for frames in [4_800_usize, 1_000, 4_800, 0, 4_800] {
        let input = pcm(layout.clone(), vec![0.125; frames * 8]);
        downmix_into(&input, &ChannelLayout::stereo(), &mut out).unwrap();
        assert_eq!(out.len(), frames * 2);
        assert_eq!((out.as_ptr(), out.capacity()), start, "{frames} frames");
    }
}

// --- FFmpeg as the oracle ---------------------------------------------------

/// Largest difference allowed from `ffmpeg -ac 2`: a few units in the last
/// place of an f32 near 2.0. Both sides add the same f32 products; only the
/// order of the additions differs.
const ORACLE_TOLERANCE: f32 = 1e-6;

fn ffmpeg() -> Option<PathBuf> {
    let found = Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .ok()?
        .success();
    found.then(|| PathBuf::from("ffmpeg"))
}

fn run(ffmpeg: &Path, args: &[&str]) {
    let status = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(args)
        .status()
        .expect("ffmpeg runs");
    assert!(status.success(), "ffmpeg {args:?}");
}

fn f32le(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read raw");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// A different tone and level on every channel, so a swapped or dropped
/// channel shows.
fn expressions(channels: usize) -> String {
    (0..channels)
        .map(|index| {
            let level = 0.9 - 0.1 * index as f64;
            let hertz = 110 * (index + 2);
            format!("{level:.1}*sin(2*PI*{hertz}*t)")
        })
        .collect::<Vec<_>>()
        .join("|")
}

#[test]
fn every_layout_matches_ffmpeg_ac_2() {
    let Some(ffmpeg) = ffmpeg() else {
        eprintln!("not available: ffmpeg");
        return;
    };
    let dir = std::env::temp_dir().join(format!("cutlass-downmix-oracle-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // ffmpeg's layout name, and the wave mask that names the same speakers.
    let cases = [
        ("3.0", 0x7_u32),
        ("quad", 0x33),
        ("5.0", 0x37),
        ("5.1", 0x3F),
        ("5.1(side)", 0x60F),
        ("6.1", 0x70F),
        ("7.1", 0x63F),
        ("7.1(wide-side)", 0x6CF),
        ("4.0", 0x107),
        ("2.1", 0xB),
    ];
    for (name, mask) in cases {
        let layout = ChannelLayout::from_wave_mask(mask).unwrap();
        let source = dir.join("source.f32");
        let reference = dir.join("reference.f32");
        let generator = format!(
            "aevalsrc=exprs={}:c={name}:s=48000:d=0.25",
            expressions(layout.len())
        );
        run(
            &ffmpeg,
            &[
                "-f",
                "lavfi",
                "-i",
                &generator,
                "-f",
                "f32le",
                source.to_str().unwrap(),
            ],
        );
        // Default swresample: float output, so no matrix normalisation.
        run(
            &ffmpeg,
            &[
                "-f",
                "f32le",
                "-ar",
                "48000",
                "-ch_layout",
                name,
                "-i",
                source.to_str().unwrap(),
                "-ac",
                "2",
                "-f",
                "f32le",
                reference.to_str().unwrap(),
            ],
        );
        let input = pcm(layout, f32le(&source));
        let ours = downmix(&input, &ChannelLayout::stereo()).unwrap();
        let theirs = f32le(&reference);
        let ours = ours.as_f32().unwrap();
        assert_eq!(ours.len(), theirs.len(), "{name}");
        let worst = ours
            .iter()
            .zip(&theirs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let peak = theirs.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        eprintln!(
            "{name}: {} frames, peak {peak:.4}, worst difference {worst:e}",
            ours.len() / 2
        );
        assert!(worst <= ORACLE_TOLERANCE, "{name}: {worst:e}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
