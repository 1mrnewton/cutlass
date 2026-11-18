//! A straight crossfade of two equal pictures.
//!
//! Progress `0` is the outgoing picture. Progress `1` is the incoming picture.
//! Between, progress becomes a weight out of 65,536, rounded, and each
//! channel is `outgoing * (1 - weight) + incoming * weight`, rounded to the
//! nearest. Two equal channels mix to themselves at any progress. This is
//! not source-over. The package that chooses this mix is not in this crate.

/// The weight scale: one is this.
const ONE: u32 = 1 << 16;

/// Mix `outgoing` and `incoming`.
///
/// The buffers must be the same length, and that length must be a multiple of
/// 4. `denom` is never 0. `numer` above `denom` is refused.
#[must_use]
pub fn crossfade(outgoing: &[u8], incoming: &[u8], numer: u64, denom: u64) -> Option<Vec<u8>> {
    if denom == 0
        || numer > denom
        || outgoing.len() != incoming.len()
        || !outgoing.len().is_multiple_of(4)
    {
        return None;
    }
    if numer == 0 {
        return Some(outgoing.to_vec());
    }
    if numer == denom {
        return Some(incoming.to_vec());
    }
    // In (0, ONE): numer is above 0 and below denom here, and a weight that
    // rounds to an end is still a mix.
    let weight = (u128::from(numer) * u128::from(ONE) + u128::from(denom) / 2) / u128::from(denom);
    let weight = u32::try_from(weight).ok()?.clamp(1, ONE - 1);
    let stay = ONE - weight;
    let mut mixed = Vec::new();
    mixed.try_reserve_exact(outgoing.len()).ok()?;
    mixed.resize(outgoing.len(), 0);
    let (left, left_rest) = outgoing.as_chunks::<4>();
    let (right, right_rest) = incoming.as_chunks::<4>();
    let (out, out_rest) = mixed.as_chunks_mut::<4>();
    if !left_rest.is_empty()
        || !right_rest.is_empty()
        || !out_rest.is_empty()
        || left.len() != right.len()
    {
        return None;
    }
    for (slot, (from, to)) in out.iter_mut().zip(left.iter().zip(right)) {
        for channel in 0..4 {
            // At most 255 * 65,536 + 32,768, inside a u32.
            let value = u32::from(from[channel]) * stay + u32::from(to[channel]) * weight;
            slot[channel] = ((value + ONE / 2) >> 16) as u8;
        }
    }
    Some(mixed)
}

#[cfg(test)]
mod tests {
    use super::crossfade;

    #[test]
    fn progress_zero_is_outgoing_and_the_end_is_incoming() {
        let outgoing = [255, 0, 0, 255];
        let incoming = [0, 0, 255, 255];
        assert_eq!(
            crossfade(&outgoing, &incoming, 0, 10),
            Some(outgoing.to_vec())
        );
        assert_eq!(
            crossfade(&outgoing, &incoming, 10, 10),
            Some(incoming.to_vec())
        );
        // Halfway, 127.5 rounds up on both sides.
        assert_eq!(
            crossfade(&outgoing, &incoming, 5, 10),
            Some(vec![128, 0, 128, 255])
        );
        // 25.5 and 229.5, each within half a step.
        assert_eq!(
            crossfade(&outgoing, &incoming, 9, 10),
            Some(vec![26, 0, 229, 255])
        );
        assert_eq!(crossfade(&outgoing, &incoming, 11, 10), None);
        assert_eq!(crossfade(&outgoing, &incoming, 0, 0), None);
        assert_eq!(crossfade(&outgoing, &[0, 0, 255], 0, 1), None);
    }

    #[test]
    fn equal_channels_mix_to_themselves_and_every_mix_is_within_half_a_step() {
        // A huge denominator, as a frame count over microseconds gives.
        for (numer, denom) in [(1, 3), (2, 3), (1, 1_000_000_007), (u64::MAX - 1, u64::MAX)] {
            for value in [0_u8, 1, 128, 254, 255] {
                let same = [value; 4];
                assert_eq!(crossfade(&same, &same, numer, denom), Some(same.to_vec()));
            }
        }
        for (from, to) in [(0_u8, 255_u8), (255, 0), (17, 200)] {
            for numer in 1..10_u64 {
                let mixed = crossfade(&[from; 4], &[to; 4], numer, 10).expect("mix");
                let exact =
                    (f64::from(from) * (10 - numer) as f64 + f64::from(to) * numer as f64) / 10.0;
                assert!(
                    (f64::from(mixed[0]) - exact).abs() <= 0.5 + 1e-3,
                    "{mixed:?} {exact}"
                );
            }
        }
    }
}
