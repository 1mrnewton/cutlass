//! Lining two recordings of the same moment up by their sound.
//!
//! A camera and a separate recorder hear the same room through different
//! microphones, so their samples do not match, but the sound gets loud and
//! quiet at the same times in both. Each recording is reduced to an
//! [`Envelope`]: how sharply it gets louder, [`ENVELOPE_RATE`] times a
//! second. [`align`] slides one envelope along the other and returns the
//! shift where they agree best, or nothing when they do not agree.
//!
//! The search starts on a coarse copy of both envelopes and is refined step
//! by step, so two hour-long recordings cost little more than two short
//! ones.

/// Envelope values in a second of sound: one for every five milliseconds.
pub const ENVELOPE_RATE: u32 = 200;

/// The least two envelopes must overlap to be compared: two seconds.
const LEAST_OVERLAP: usize = 2 * ENVELOPE_RATE as usize;
/// The most products the first, coarse search may take.
const COARSE_BUDGET: u64 = 40_000_000;
/// A match weaker than this is not one.
const LEAST_SCORE: f32 = 0.3;
/// The longest envelope kept: a day of sound. What comes after is dropped.
const LONGEST: usize = 24 * 60 * 60 * ENVELOPE_RATE as usize;
/// Loudness below this is silence, so its changes are not onsets.
const FLOOR: f32 = 1.0e-4;

/// Builds a recording's envelope from its mono samples, given in order in
/// pieces of any size.
#[derive(Clone, Debug)]
pub struct Envelope {
    rate: u64,
    seen: u64,
    sum: f64,
    count: u32,
    bin: u64,
    levels: Vec<f32>,
}

impl Envelope {
    /// An envelope for mono sound at `rate` samples a second. `None` for a
    /// rate of zero.
    #[must_use]
    pub fn new(rate: u32) -> Option<Self> {
        (rate > 0).then(|| Self {
            rate: u64::from(rate),
            seen: 0,
            sum: 0.0,
            count: 0,
            bin: 0,
            levels: Vec::new(),
        })
    }

    /// The sample rate the envelope was made for.
    #[must_use]
    pub fn rate(&self) -> u32 {
        u32::try_from(self.rate).unwrap_or(u32::MAX)
    }

    /// Take the next `samples`. A sample that is not a number counts as
    /// silence.
    pub fn push(&mut self, samples: &[f32]) {
        for sample in samples {
            let bin = self.seen.saturating_mul(u64::from(ENVELOPE_RATE)) / self.rate;
            if bin != self.bin {
                self.close(bin);
            }
            if sample.is_finite() {
                self.sum += f64::from(sample.abs());
            }
            self.count = self.count.saturating_add(1);
            self.seen = self.seen.saturating_add(1);
        }
    }

    /// Skip `samples` of sound that could not be read, as silence, so what
    /// follows stays at its own time.
    pub fn skip(&mut self, samples: u64) {
        self.seen = self.seen.saturating_add(samples);
        let bin = self.seen.saturating_mul(u64::from(ENVELOPE_RATE)) / self.rate;
        if bin != self.bin {
            self.close(bin);
        }
    }

    /// Finish the bin being filled and move to `bin`, with silence between.
    fn close(&mut self, bin: u64) {
        let level = if self.count == 0 {
            0.0
        } else {
            (self.sum / f64::from(self.count)) as f32
        };
        if self.levels.len() < LONGEST {
            self.levels.push(level);
        }
        // Bins the sound jumped over hold silence, up to the longest
        // envelope kept, so a skip of any size asks for no more than that.
        let missing = bin.saturating_sub(self.bin).saturating_sub(1);
        let room = LONGEST.saturating_sub(self.levels.len());
        let missing = usize::try_from(missing).unwrap_or(usize::MAX).min(room);
        self.levels.extend(std::iter::repeat_n(0.0, missing));
        self.sum = 0.0;
        self.count = 0;
        self.bin = bin;
    }

    /// The envelope: how much louder each step is than the one before, on a
    /// scale where doubling the loudness adds the same amount. Getting
    /// quieter counts as nothing.
    #[must_use]
    pub fn finish(mut self) -> Vec<f32> {
        if self.count > 0 {
            let next = self.bin.saturating_add(1);
            self.close(next);
        }
        let mut before = FLOOR.ln();
        self.levels
            .iter()
            .map(|level| {
                let now = level.max(FLOOR).ln();
                let rise = (now - before).max(0.0);
                before = now;
                rise
            })
            .collect()
    }
}

/// Where one recording sits against another.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Alignment {
    /// Steps of [`ENVELOPE_RATE`] the other recording starts after the
    /// reference: its step `i` is the reference's step `i + lag`. Negative
    /// when it starts first.
    pub lag: i64,
    /// How well they agree there, from 0 to 1.
    pub score: f32,
}

impl Alignment {
    /// The lag in microseconds.
    #[must_use]
    pub fn micros(&self) -> i64 {
        self.lag.saturating_mul(1_000_000) / i64::from(ENVELOPE_RATE)
    }
}

/// Find where the envelope `other` sits against `reference`. `None` when
/// either is shorter than two seconds, or they agree nowhere well enough to
/// be the same sound.
#[must_use]
pub fn align(reference: &[f32], other: &[f32]) -> Option<Alignment> {
    if reference.len() < LEAST_OVERLAP || other.len() < LEAST_OVERLAP {
        return None;
    }
    // Halve both until the whole search fits the budget.
    let mut levels = vec![(reference.to_vec(), other.to_vec())];
    let mut factor = 1_usize;
    while let Some((a, b)) = levels.last() {
        let cost = (a.len() as u64).saturating_mul(b.len() as u64);
        if cost <= COARSE_BUDGET || a.len() < 64 || b.len() < 64 {
            break;
        }
        let next = (halve(a), halve(b));
        levels.push(next);
        factor *= 2;
    }
    // At least half of the shorter recording must meet the other. Where
    // only a little of each meets, sounds that have nothing to do with
    // each other agree by chance.
    let least = LEAST_OVERLAP.max(reference.len().min(other.len()) / 2);
    let (a, b) = levels.last()?;
    let overlap = (least / factor).max(8);
    let from = overlap as i64 - b.len() as i64;
    let to = a.len() as i64 - overlap as i64;
    let mut best = search(a, b, from, to, overlap)?;
    // Each finer level looks only around the lag the coarser one found.
    for (a, b) in levels.iter().rev().skip(1) {
        factor /= 2;
        let overlap = (least / factor).max(8);
        let around = best.lag.saturating_mul(2);
        let from = (around - 3).max(overlap as i64 - b.len() as i64);
        let to = (around + 3).min(a.len() as i64 - overlap as i64);
        best = search(a, b, from, to, overlap)?;
    }
    (best.score >= LEAST_SCORE).then_some(best)
}

/// Each pair of steps added into one.
fn halve(envelope: &[f32]) -> Vec<f32> {
    envelope
        .chunks(2)
        .map(|pair| pair.iter().sum::<f32>())
        .collect()
}

/// The best lag from `from` to `to`, where `b`'s step `i` meets `a`'s step
/// `i + lag` and at least `overlap` steps meet. The score is the
/// correlation of the steps that meet, each side about its own mean there,
/// so two sounds that are merely both busy do not score.
fn search(a: &[f32], b: &[f32], from: i64, to: i64, overlap: usize) -> Option<Alignment> {
    // Running totals, so a stretch's sum and sum of squares cost nothing.
    let totals = |envelope: &[f32]| -> (Vec<f64>, Vec<f64>) {
        let (mut sum, mut square) = (0.0_f64, 0.0_f64);
        let mut sums = vec![0.0];
        let mut squares = vec![0.0];
        for value in envelope {
            let value = f64::from(*value);
            sum += value;
            square += value * value;
            sums.push(sum);
            squares.push(square);
        }
        (sums, squares)
    };
    let (a_sums, a_squares) = totals(a);
    let (b_sums, b_squares) = totals(b);
    let mut best: Option<Alignment> = None;
    let mut lag = from;
    while lag <= to {
        // The steps of `b` that meet `a` at this lag.
        let start = usize::try_from((-lag).max(0)).ok()?;
        let end = usize::try_from((a.len() as i64 - lag).min(b.len() as i64).max(0)).ok()?;
        if end > start && end - start >= overlap {
            let shift = usize::try_from(start as i64 + lag).ok()?;
            let length = end - start;
            let count = length as f64;
            let dot: f64 = a[shift..shift + length]
                .iter()
                .zip(&b[start..end])
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum();
            let a_sum = a_sums[shift + length] - a_sums[shift];
            let b_sum = b_sums[end] - b_sums[start];
            let a_spread = a_squares[shift + length] - a_squares[shift] - a_sum * a_sum / count;
            let b_spread = b_squares[end] - b_squares[start] - b_sum * b_sum / count;
            if a_spread > 1.0e-9 && b_spread > 1.0e-9 {
                let score = ((dot - a_sum * b_sum / count) / (a_spread * b_spread).sqrt()) as f32;
                if best.is_none_or(|best| score > best.score) {
                    best = Some(Alignment { lag, score });
                }
            }
        }
        lag += 1;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A repeatable noise source.
    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1_u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// `seconds` of a room at `rate`: quiet noise with claps and words at
    /// times of their own, as `seed` places them.
    fn room(seed: u64, seconds: u32, rate: u32) -> Vec<f32> {
        let mut noise = Noise(seed);
        let mut events = Noise(seed.wrapping_mul(31).wrapping_add(7));
        let total = (seconds * rate) as usize;
        let mut sound = vec![0.0_f32; total];
        let mut at = 0_usize;
        while at < total {
            // A burst a tenth to a half second long, then a pause.
            let length = ((events.next().abs() * 0.4 + 0.1) * rate as f32) as usize;
            let loud = events.next().abs() * 0.8 + 0.2;
            for sample in sound.iter_mut().skip(at).take(length) {
                *sample = noise.next() * loud;
            }
            at += length + ((events.next().abs() * 0.6 + 0.05) * rate as f32) as usize;
        }
        for sample in &mut sound {
            *sample += noise.next() * 0.01;
        }
        sound
    }

    fn envelope(samples: &[f32], rate: u32) -> Vec<f32> {
        let mut envelope = Envelope::new(rate).unwrap();
        // In uneven pieces, as a file is read.
        for piece in samples.chunks(1_777) {
            envelope.push(piece);
        }
        envelope.finish()
    }

    #[test]
    fn a_recording_is_found_inside_another_of_the_same_room() {
        let rate = 8_000;
        let camera = room(11, 30, rate);
        // The recorder started 7.4 seconds in, ran 12 seconds, was quieter,
        // sampled at another rate, and had hiss of its own.
        let start = (7.4 * rate as f32) as usize;
        let mut hiss = Noise(99);
        let recorder: Vec<f32> = camera[start..start + 12 * rate as usize]
            .iter()
            .map(|sample| sample * 0.3 + hiss.next() * 0.01)
            .collect();
        let found = align(&envelope(&camera, rate), &envelope(&recorder, rate)).unwrap();
        assert!((found.micros() - 7_400_000).abs() <= 10_000, "{found:?}");
        assert!(found.score > 0.6, "{found:?}");

        // The other way round the lag is the same length, backwards.
        let back = align(&envelope(&recorder, rate), &envelope(&camera, rate)).unwrap();
        assert!((back.micros() + 7_400_000).abs() <= 10_000, "{back:?}");
    }

    #[test]
    fn two_long_recordings_are_lined_up_through_the_coarse_search() {
        let rate = 2_000;
        let whole = room(5, 1_500, rate);
        let lead = 301 * rate as usize + 600;
        let a = &whole[..1_200 * rate as usize];
        let b = &whole[lead..];
        let (a, b) = (envelope(a, rate), envelope(b, rate));
        assert!((a.len() as u64) * (b.len() as u64) > COARSE_BUDGET);
        let found = align(&a, &b).unwrap();
        assert!((found.micros() - 301_300_000).abs() <= 10_000, "{found:?}");
    }

    #[test]
    fn different_rooms_silence_and_short_sound_do_not_line_up() {
        let rate = 8_000;
        let one = envelope(&room(1, 20, rate), rate);
        let two = envelope(&room(2, 20, rate), rate);
        assert_eq!(align(&one, &two), None);
        let silence = envelope(&vec![0.0; 20 * rate as usize], rate);
        assert_eq!(align(&one, &silence), None);
        assert_eq!(align(&silence, &silence), None);
        let short = envelope(&room(1, 1, rate), rate);
        assert_eq!(align(&one, &short), None);
        assert_eq!(align(&[], &[]), None);
    }

    #[test]
    fn an_envelope_keeps_time_over_pieces_gaps_and_bad_samples() {
        assert!(Envelope::new(0).is_none());
        // A rate the step does not divide: 44.1 kHz is 220.5 samples a step.
        let rate = 44_100;
        let mut envelope = Envelope::new(rate).unwrap();
        envelope.push(&vec![0.0; rate as usize]);
        // A second that could not be read, then a loud second with samples
        // that are not numbers in it.
        envelope.skip(u64::from(rate));
        let mut loud = vec![0.5_f32; rate as usize];
        loud[10] = f32::NAN;
        loud[11] = f32::INFINITY;
        envelope.push(&loud);
        let steps = envelope.finish();
        assert_eq!(steps.len(), 3 * ENVELOPE_RATE as usize);
        assert!(steps.iter().all(|step| step.is_finite() && *step >= 0.0));
        // The one rise is where the loud second starts.
        let (at, rise) = steps
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        assert_eq!(at, 2 * ENVELOPE_RATE as usize);
        assert!(*rise > 5.0);
        // A skip far past anything read does not allocate for it.
        let mut far = Envelope::new(8_000).unwrap();
        far.skip(u64::MAX);
        far.push(&[0.1]);
        assert_eq!(far.finish().len(), LONGEST);
    }
}
