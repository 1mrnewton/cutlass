use std::path::PathBuf;
use std::time::Instant;

use super::*;
use crate::matte_frame::real;

const REQUIRE: &str = "CUTLASS_REQUIRE_BEATS_MODEL";

fn local_models() -> PathBuf {
    real::models("beats")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-beats-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A click every half second with a hum under it: something to hear.
fn clicks(seconds: usize) -> Vec<f32> {
    let rate = BEATS_RATE as usize;
    (0..rate * seconds)
        .map(|index| {
            let since = (index % (rate / 2)) as f32 / rate as f32;
            let hum = (index as f32 * 0.03).sin() * 0.1;
            hum + (since * 900.0).sin() * (-since * 40.0).exp() * 0.8
        })
        .collect()
}

/// The piece PyTorch heard and what it made of it, from
/// `tools/vision-models/scripts/fetch_beats.sh`.
struct Reference {
    signal: Vec<f32>,
    spect: Vec<f32>,
    beat: Vec<f32>,
    downbeat: Vec<f32>,
    beat_times: Vec<f64>,
    downbeat_times: Vec<f64>,
}

fn reference() -> Option<Reference> {
    let dir = local_models();
    let file = dir.join("reference_piece.safetensors");
    if !dir.join(BEATS_WEIGHTS).is_file() || !file.is_file() {
        assert!(
            std::env::var_os(REQUIRE).is_none(),
            "{REQUIRE} is set and the beat weights or {} are missing",
            file.display()
        );
        return None;
    }
    let bytes = std::fs::read(&file).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let floats = |name: &str| -> Vec<f32> {
        let data = tensors.tensor(name).unwrap();
        data.data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    };
    let doubles = |name: &str| -> Vec<f64> {
        let data = tensors.tensor(name).unwrap();
        data.data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect()
    };
    let reference = Reference {
        signal: floats("signal"),
        spect: floats("spect"),
        beat: floats("beat"),
        downbeat: floats("downbeat"),
        beat_times: doubles("beat_times"),
        downbeat_times: doubles("downbeat_times"),
    };
    assert_eq!(reference.spect.len(), reference.beat.len() * BANDS);
    // Longer than one stretch, so two readings are joined.
    assert!(reference.beat.len() > STRETCH);
    Some(reference)
}

/// The worst and the mean difference between `got` and `expected`.
fn apart(got: &[f32], expected: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), expected.len());
    let errors: Vec<f32> = got
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .collect();
    let worst = errors.iter().copied().fold(0.0, f32::max);
    (worst, errors.iter().sum::<f32>() / errors.len() as f32)
}

fn micros(seconds: &[f64]) -> Vec<i64> {
    seconds
        .iter()
        .map(|time| (time * 1_000_000.0).round() as i64)
        .collect()
}

#[test]
fn the_spectrogram_matches_the_published_one() {
    let Some(reference) = reference() else {
        return;
    };
    let spect = spectrogram(&reference.signal).unwrap();
    let (worst, mean) = apart(&spect, &reference.spect);
    assert!(worst < 2e-3 && mean < 1e-4, "worst {worst}, mean {mean}");
}

/// The model's answers against PyTorch's, from the published spectrogram
/// so only the model is compared, and then the beats from the sound.
fn against_pytorch(compute: Compute) -> Option<(f32, f32)> {
    let reference = reference()?;
    let model = BeatModel::load(&local_models(), compute).unwrap();
    let (beat, downbeat) = model
        .answers(&reference.spect, &mut |_| true)
        .unwrap()
        .unwrap();
    let (worst_beat, mean_beat) = apart(&beat, &reference.beat);
    let (worst_down, mean_down) = apart(&downbeat, &reference.downbeat);
    let found = model
        .find(&reference.signal, &mut |_| true)
        .unwrap()
        .unwrap();
    assert_eq!(found.beats, micros(&reference.beat_times));
    assert_eq!(found.downbeats, micros(&reference.downbeat_times));
    assert_eq!(found.beats.len(), 80, "a beat every half second for 40 s");
    Some((worst_beat.max(worst_down), mean_beat.max(mean_down)))
}

#[test]
fn the_cpu_matches_pytorch() {
    if let Some((worst, mean)) = against_pytorch(Compute::Cpu) {
        assert!(worst < 5e-3 && mean < 5e-4, "worst {worst}, mean {mean}");
    }
}

#[test]
#[ignore = "needs a GPU"]
fn the_gpu_matches_pytorch() {
    // Trusted as the editor trusts a GPU that passed the check.
    crate::trust_gpu(crate::check_gpu().ok());
    if let Some((worst, mean)) = against_pytorch(Compute::Gpu) {
        assert!(worst < 2e-2 && mean < 2e-3, "worst {worst}, mean {mean}");
    }
}

#[test]
fn the_published_weights_convert_into_the_ones_the_model_loads() {
    let Some(reference) = reference() else {
        return;
    };
    let published = local_models()
        .join("hf")
        .join("beat_this_small.safetensors");
    if !published.is_file() {
        return;
    }
    let dir = scratch("convert");
    convert_beat_this(&published, &dir.join(BEATS_WEIGHTS)).unwrap();
    let converted = BeatModel::load(&dir, Compute::Cpu).unwrap();
    let installed = BeatModel::load(&local_models(), Compute::Cpu).unwrap();
    // One stretch is enough to tell the weights are the same.
    let piece = &reference.signal[..BEATS_RATE as usize * 4];
    assert_eq!(
        converted.find(piece, &mut |_| true).unwrap(),
        installed.find(piece, &mut |_| true).unwrap()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_short_sound_is_read_in_one_stretch_with_silence_at_both_ends() {
    assert_eq!(
        stretches(100),
        [Stretch {
            from: 0,
            to: 100,
            left: BORDER,
            right: BORDER
        }]
    );
    // The longest sound one stretch holds.
    let whole = stretches(STRETCH - 2 * BORDER);
    assert_eq!(whole.len(), 1);
    assert_eq!(whole[0].frames(), STRETCH);
    assert!(stretches(0).is_empty());
}

#[test]
fn a_long_sound_is_read_in_stretches_that_end_with_it() {
    assert_eq!(
        stretches(2001),
        [
            Stretch {
                from: 0,
                to: STRETCH - BORDER,
                left: BORDER,
                right: 0
            },
            Stretch {
                from: 507,
                to: 2001,
                left: 0,
                right: BORDER
            },
        ]
    );
    // One frame more than a stretch holds: the second starts one frame in.
    let just_over = stretches(STRETCH - 2 * BORDER + 1);
    assert_eq!(just_over.len(), 2);
    assert_eq!(just_over[1].first_answer(), 1);
    assert_eq!(just_over[1].left, BORDER - 1);
}

#[test]
fn every_frame_is_answered_by_a_stretch_no_longer_than_the_model_reads() {
    for frames in (1..9000)
        .step_by(37)
        .chain([1488, 1489, 1494, 1500, 2976, 2977])
    {
        let mut answered = vec![false; frames];
        for stretch in stretches(frames) {
            assert!(stretch.frames() <= STRETCH, "{frames}: {stretch:?}");
            assert!(stretch.from < stretch.to && stretch.to <= frames);
            let kept = stretch.frames() - 2 * BORDER;
            let first = stretch.first_answer();
            answered[first..(first + kept).min(frames)].fill(true);
        }
        assert!(answered.iter().all(|answer| *answer), "{frames} frames");
    }
}

#[test]
fn a_beat_is_the_surest_frame_near_it_and_more_likely_than_not() {
    let mut logits = vec![-5.0_f32; 40];
    logits[4] = 2.0;
    // Within reach of a surer frame, so not a beat of its own.
    logits[6] = 1.0;
    // Less likely than not.
    logits[15] = -0.5;
    // Two neighbours as sure as each other are one beat between them.
    logits[20] = 3.0;
    logits[21] = 3.0;
    logits[39] = 0.1;
    assert_eq!(peaks(&logits), [4.0, 20.5, 39.0]);
    assert!(peaks(&[]).is_empty());
    assert!(peaks(&[-1.0, -2.0]).is_empty());
}

#[test]
fn a_downbeat_moves_onto_the_beat_nearest_it() {
    let mut beat = vec![-5.0_f32; 200];
    let mut downbeat = vec![-5.0_f32; 200];
    for frame in [10, 35, 60, 85] {
        beat[frame] = 4.0;
    }
    // A frame early, a frame late, and two that land on the same beat.
    downbeat[9] = 2.0;
    downbeat[61] = 2.0;
    downbeat[80] = 1.0;
    downbeat[88] = 2.0;
    let found = beats_of(&beat, &downbeat);
    assert_eq!(found.beats, [200_000, 700_000, 1_200_000, 1_700_000]);
    assert_eq!(found.downbeats, [200_000, 1_200_000, 1_700_000]);
    // With no beat to move onto, a downbeat stays where it was heard.
    let alone = beats_of(&vec![-5.0; 200], &downbeat);
    assert!(alone.beats.is_empty());
    assert_eq!(alone.downbeats.len(), 4);
}

#[test]
fn silence_is_a_spectrogram_of_zeros_with_a_frame_every_hop() {
    let spect = spectrogram(&vec![0.0; BEATS_RATE as usize]).unwrap();
    assert_eq!(spect.len(), (BEATS_RATE as usize / HOP + 1) * BANDS);
    assert!(spect.iter().all(|level| *level == 0.0));
    let shortest = spectrogram(&vec![0.0; BEATS_MIN_SAMPLES]).unwrap();
    assert_eq!(shortest.len(), (BEATS_MIN_SAMPLES / HOP + 1) * BANDS);
}

#[test]
fn a_tone_is_loudest_in_the_band_around_it() {
    let hz = 440.0_f64;
    let tone: Vec<f32> = (0..BEATS_RATE as usize)
        .map(|index| {
            (std::f64::consts::TAU * hz * index as f64 / f64::from(BEATS_RATE)).sin() as f32
        })
        .collect();
    let spect = spectrogram(&tone).unwrap();
    // A frame in the middle, clear of both ends.
    let frame = &spect[25 * BANDS..26 * BANDS];
    let loudest = (0..BANDS)
        .max_by(|a, b| frame[*a].total_cmp(&frame[*b]))
        .unwrap();
    let bin = (hz * WINDOW as f64 / f64::from(BEATS_RATE)).round() as usize;
    let bands = bands();
    let hears = |band: &Band| {
        bin.checked_sub(band.first)
            .and_then(|at| band.weights.get(at))
            .copied()
            .unwrap_or(0.0)
    };
    let expected = (0..BANDS)
        .max_by(|a, b| hears(&bands[*a]).total_cmp(&hears(&bands[*b])))
        .unwrap();
    assert_eq!(loudest, expected);
    assert!(frame[loudest] > 3.0, "{}", frame[loudest]);
}

#[test]
fn the_bands_rise_from_30_hz_to_11_khz_each_a_triangle() {
    let bands = bands();
    assert_eq!(bands.len(), BANDS);
    let bin_hz = f64::from(BEATS_RATE) / WINDOW as f64;
    for pair in bands.windows(2) {
        assert!(pair[0].first <= pair[1].first);
    }
    for band in &bands {
        assert!(
            band.weights
                .iter()
                .all(|weight| (0.0..=1.0).contains(weight))
        );
        assert!(band.first as f64 * bin_hz > LOWEST_HZ);
        assert!((band.first + band.weights.len()) as f64 * bin_hz < HIGHEST_HZ + bin_hz);
    }
    // The widest bands, at the top, hear many bins; a few of the narrow
    // ones at the bottom fall between two bins and hear none.
    assert!(bands[BANDS - 1].weights.len() > 10);
}

#[test]
fn a_sound_too_short_or_with_a_sample_that_is_not_a_number_is_refused() {
    for samples in [
        Vec::new(),
        vec![0.0; BEATS_MIN_SAMPLES - 1],
        vec![f32::NAN; BEATS_MIN_SAMPLES],
        [vec![0.0; BEATS_MIN_SAMPLES], vec![f32::INFINITY]].concat(),
    ] {
        assert_eq!(
            spectrogram(&samples).err(),
            Some(VisionError::Sound {
                samples: samples.len()
            })
        );
    }
    let model = BeatModel::untrained(Compute::Cpu).unwrap();
    assert!(matches!(
        model.find(&[0.0; 100], &mut |_| true),
        Err(VisionError::Sound { samples: 100 })
    ));
}

#[test]
fn the_model_answers_for_every_frame_and_stops_when_told() {
    let model = BeatModel::untrained(Compute::Cpu).unwrap();
    let sound = clicks(2);
    let spect = spectrogram(&sound).unwrap();
    let frames = spect.len() / BANDS;
    let mut asked = Vec::new();
    let (beat, downbeat) = model
        .answers(&spect, &mut |done| {
            asked.push(done);
            true
        })
        .unwrap()
        .unwrap();
    assert_eq!(asked, [0.0]);
    assert_eq!((beat.len(), downbeat.len()), (frames, frames));
    assert!(beat.iter().chain(&downbeat).all(|logit| logit.is_finite()));
    assert!(beat.iter().all(|logit| *logit != SILENT));
    assert_eq!(model.find(&sound, &mut |_| false).unwrap(), None);
}

#[test]
fn saved_weights_load_back_and_damaged_or_missing_ones_are_refused() {
    let dir = scratch("round-trip");
    let saved = BeatModel::untrained(Compute::Cpu).unwrap();
    let mut store = BurnpackStore::from_file(dir.join(BEATS_WEIGHTS)).overwrite(true);
    saved.model.save_into(&mut store).unwrap();
    let loaded = BeatModel::load(&dir, Compute::Cpu).unwrap();
    let spect = spectrogram(&clicks(1)).unwrap();
    assert_eq!(
        saved.answers(&spect, &mut |_| true).unwrap(),
        loaded.answers(&spect, &mut |_| true).unwrap()
    );
    std::fs::write(dir.join(BEATS_WEIGHTS), b"not weights").unwrap();
    assert!(matches!(
        BeatModel::load(&dir, Compute::Cpu),
        Err(VisionError::Weights { .. })
    ));
    std::fs::remove_file(dir.join(BEATS_WEIGHTS)).unwrap();
    assert!(matches!(
        BeatModel::load(&dir, Compute::Cpu),
        Err(VisionError::Weights { .. })
    ));
    assert!(convert_beat_this(&dir.join("none.safetensors"), &dir.join("out.bpk")).is_err());
    std::fs::write(dir.join("bad.safetensors"), b"nope").unwrap();
    assert!(convert_beat_this(&dir.join("bad.safetensors"), &dir.join("out.bpk")).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

fn time_a_song(compute: Compute) {
    let model = BeatModel::untrained(compute).unwrap();
    // Three and a half minutes: eight stretches.
    let sound = clicks(210);
    let started = Instant::now();
    let spect = spectrogram(&sound).unwrap();
    let heard = started.elapsed();
    let started = Instant::now();
    model.answers(&spect, &mut |_| true).unwrap();
    let read = started.elapsed();
    let _ = std::io::Write::write_all(
        &mut std::io::stderr(),
        format!("beats {compute:?}: spectrogram {heard:?}, model {read:?} for 210 s of sound\n")
            .as_bytes(),
    );
}

#[test]
#[ignore = "a timing, run by hand in release"]
fn time_a_song_on_the_cpu() {
    time_a_song(Compute::Cpu);
}

#[test]
#[ignore = "needs a GPU"]
fn time_a_song_on_the_gpu() {
    crate::trust_gpu(crate::check_gpu().ok());
    time_a_song(Compute::Gpu);
}
