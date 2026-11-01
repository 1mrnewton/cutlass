//! A straight crossfade of two equal pictures.
//!
//! Progress `0` is the outgoing picture. Progress `1` is the incoming picture.
//! Each channel is `(outgoing * (denom - numer) + incoming * numer) / denom`,
//! truncated. This is not source-over. The package that chooses this mix is
//! not in this crate.

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
    let stay = denom - numer;
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
            let value = u128::from(from[channel]) * u128::from(stay)
                + u128::from(to[channel]) * u128::from(numer);
            slot[channel] = u8::try_from(value / u128::from(denom)).ok()?;
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
        assert_eq!(
            crossfade(&outgoing, &incoming, 5, 10),
            Some(vec![127, 0, 127, 255])
        );
        assert_eq!(
            crossfade(&outgoing, &incoming, 9, 10),
            Some(vec![25, 0, 229, 255])
        );
        assert_eq!(crossfade(&outgoing, &incoming, 11, 10), None);
        assert_eq!(crossfade(&outgoing, &incoming, 0, 0), None);
        assert_eq!(crossfade(&outgoing, &[0, 0, 255], 0, 1), None);
    }
}
