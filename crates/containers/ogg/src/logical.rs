//! One logical stream: packet reassembly across pages and the Opus
//! granule arithmetic of RFC 7845 section 4.
//!
//! Pages arrive in file order. Each page's pieces are joined into packets.
//! A packet whose earlier piece was lost (a page-sequence gap, a page that
//! is not marked continued, or a continued page with nothing to continue)
//! is dropped, never glued to unrelated bytes (RFC 7845 section 3).
//!
//! Timing: a page's granule position is the end of the last packet that
//! completes on it, so the packets completing on a page start at the
//! granule minus their samples. On the first audio page that is the stream
//! start. On the end-of-stream page a granule below the samples is end
//! trimming: the packets start where the previous page ended and the last
//! durations are cut so the final packet ends at the granule.

use cutlass_container::{Gap, TrackId};

use crate::error::OggError;
use crate::opus::packet_samples;
use crate::page::Page;

/// One packet completed on the current page, with its time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ready {
    pub track: u32,
    /// The payload, or the declared size and the bound it broke.
    pub data: Result<Vec<u8>, (u64, u64)>,
    /// Granule-domain position of the packet's first sample.
    pub pts: i64,
    /// Samples after end trimming.
    pub duration: i64,
    /// Samples from the TOC, before end trimming.
    pub samples: i64,
    /// Offset of the page the packet completed on.
    pub page: u64,
}

/// Where a stream's timing stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// The next audio page is the stream's first: RFC 7845 section 4.5 applies.
    Fresh,
    /// Joined mid-stream after a seek. The next page with completed packets
    /// anchors time; with `discard`, its completed packets are dropped and
    /// only its granule and trailing piece are kept.
    Resume { discard: bool },
    /// The next packet starts at `next`.
    Running,
}

/// A packet still being assembled from pieces on several pages.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Partial {
    bytes: Vec<u8>,
    /// Bytes seen, including those dropped past the bound.
    len: u64,
    /// The first two bytes, kept so a packet past the bound still has a TOC.
    toc: [u8; 2],
    over: bool,
}

impl Partial {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            len: 0,
            toc: [0; 2],
            over: false,
        }
    }

    fn push(&mut self, data: &[u8], cap: u64) {
        for (index, &byte) in data.iter().take(2).enumerate() {
            if let Ok(at) = usize::try_from(self.len)
                && let Some(slot) = self.toc.get_mut(at + index)
            {
                *slot = byte;
            }
        }
        self.len = self
            .len
            .saturating_add(u64::try_from(data.len()).unwrap_or(u64::MAX));
        if self.len > cap {
            if !self.over {
                self.over = true;
                self.bytes = Vec::new();
            }
            return;
        }
        self.bytes.extend_from_slice(data);
    }

    fn samples(&self) -> i64 {
        let head: &[u8] = if self.over {
            let keep = usize::try_from(self.len.min(2)).unwrap_or(2);
            &self.toc[..keep]
        } else {
            &self.bytes
        };
        i64::from(packet_samples(head).unwrap_or(0))
    }
}

/// Reading state that a seek snapshots and restores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    expect_sequence: Option<u32>,
    partial: Option<Partial>,
    pub phase: Phase,
    /// Start of the next packet, when `phase` is [`Phase::Running`].
    pub next: i64,
    /// Last granule position seen, when `phase` is [`Phase::Running`].
    last_granule: i64,
    /// The end-of-stream page has been read.
    pub ended: bool,
}

impl State {
    pub const fn fresh() -> Self {
        Self {
            expect_sequence: None,
            partial: None,
            phase: Phase::Fresh,
            next: 0,
            last_granule: 0,
            ended: false,
        }
    }

    pub const fn resume(discard: bool) -> Self {
        Self {
            expect_sequence: None,
            partial: None,
            phase: Phase::Resume { discard },
            next: 0,
            last_granule: 0,
            ended: false,
        }
    }
}

/// Counters a caller can read after open or reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// Pages missing from a stream's sequence numbers.
    pub lost_pages: u64,
    /// Packet pieces dropped because a piece before or after them was lost.
    pub dropped_pieces: u64,
}

/// An Opus logical stream's fixed facts and its reading state.
#[derive(Clone, Debug)]
pub struct Opus {
    pub serial: u32,
    pub track: u32,
    pub pre_skip: i64,
    /// Pages with a lower sequence number hold the two header packets.
    pub first_audio_sequence: u32,
    /// Largest audio packet kept, in bytes.
    pub packet_cap: u64,
    pub state: State,
}

impl Opus {
    /// Takes one page of this stream. Completed packets go to `out` in
    /// order; a timeline jump goes to `gaps`.
    pub fn take_page(
        &mut self,
        page: &Page,
        out: &mut Vec<Ready>,
        gaps: &mut Vec<Gap>,
        counters: &mut Counters,
    ) -> Result<(), OggError> {
        if page.sequence < self.first_audio_sequence || self.state.ended {
            return Ok(());
        }
        let state = &mut self.state;
        if let Some(expected) = state.expect_sequence
            && page.sequence != expected
        {
            counters.lost_pages = counters
                .lost_pages
                .saturating_add(u64::from(page.sequence.wrapping_sub(expected)));
            if state.partial.take().is_some() {
                counters.dropped_pieces = counters.dropped_pieces.saturating_add(1);
            }
        }
        state.expect_sequence = Some(page.sequence.wrapping_add(1));

        let mut completed: Vec<Partial> = Vec::new();
        for (index, piece) in page.pieces().into_iter().enumerate() {
            let data = &page.body[piece.start..piece.end];
            let mut packet = if index == 0 && page.continued() {
                let Some(partial) = state.partial.take() else {
                    // The start of this packet was not read. Drop the piece.
                    if piece.complete {
                        counters.dropped_pieces = counters.dropped_pieces.saturating_add(1);
                    }
                    continue;
                };
                partial
            } else {
                if index == 0 && state.partial.take().is_some() {
                    // A packet was left open and this page does not continue it.
                    counters.dropped_pieces = counters.dropped_pieces.saturating_add(1);
                }
                Partial::new()
            };
            packet.push(data, self.packet_cap);
            if piece.complete {
                completed.push(packet);
            } else {
                state.partial = Some(packet);
            }
        }

        if completed.is_empty() {
            if page.eos() {
                state.ended = true;
            }
            return Ok(());
        }
        let samples: Vec<i64> = completed.iter().map(Partial::samples).collect();
        let total: i64 = samples.iter().sum();
        let granule = if page.granule == -1 {
            // A page that completes packets must carry a granule position.
            // Running streams can still time the packets; others cannot.
            if state.phase != Phase::Running {
                return Err(OggError::MissingGranule {
                    serial: self.serial,
                });
            }
            state.next.saturating_add(total)
        } else {
            page.granule
        };

        let start = match state.phase {
            Phase::Fresh => {
                if granule < total {
                    if !page.eos() {
                        return Err(OggError::InitialGranule {
                            serial: self.serial,
                            granule,
                            samples: total,
                        });
                    }
                    if granule < self.pre_skip {
                        return Err(OggError::PreSkipPastEnd {
                            serial: self.serial,
                            end: granule,
                            pre_skip: self.pre_skip,
                        });
                    }
                    0
                } else {
                    granule - total
                }
            }
            Phase::Resume { discard } => {
                if discard {
                    state.phase = Phase::Running;
                    state.next = granule;
                    state.last_granule = granule;
                    if page.eos() {
                        state.ended = true;
                    }
                    return Ok(());
                }
                (granule - total).max(0)
            }
            Phase::Running => {
                let expected = state.next;
                if granule < state.last_granule {
                    return Err(OggError::GranuleBackwards {
                        serial: self.serial,
                        granule,
                        floor: state.last_granule,
                    });
                }
                if page.eos() && granule < expected.saturating_add(total) {
                    expected
                } else {
                    let start = granule - total;
                    if start < expected {
                        return Err(OggError::GranuleBackwards {
                            serial: self.serial,
                            granule,
                            floor: expected.saturating_add(total),
                        });
                    }
                    if start > expected
                        && let Ok(gap) =
                            Gap::try_new(TrackId::new(self.track), expected, start - expected)
                    {
                        gaps.push(gap);
                    }
                    start
                }
            }
        };

        let mut at = start;
        for (packet, &count) in completed.into_iter().zip(&samples) {
            let duration = if page.eos() {
                count.min(granule.saturating_sub(at)).max(0)
            } else {
                count
            };
            let data = if packet.over {
                Err((packet.len, self.packet_cap))
            } else {
                Ok(packet.bytes)
            };
            out.push(Ready {
                track: self.track,
                data,
                pts: at,
                duration,
                samples: count,
                page: page.offset,
            });
            at = at.saturating_add(count);
        }
        state.phase = Phase::Running;
        state.next = if page.eos() { granule.min(at) } else { at };
        state.last_granule = granule;
        if page.eos() {
            state.ended = true;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Counters, Opus, Phase, Ready, State};
    use crate::error::OggError;
    use crate::page::{FLAG_CONTINUED, FLAG_EOS, Page};
    use cutlass_container::Gap;

    /// A 20 ms CELT packet of `len` bytes.
    fn packet(len: usize) -> Vec<u8> {
        let mut bytes = vec![0x55; len];
        if let Some(first) = bytes.first_mut() {
            *first = 31 << 3;
        }
        bytes
    }

    fn page(sequence: u32, flags: u8, granule: i64, packets: &[Vec<u8>]) -> Page {
        let mut lacing = Vec::new();
        let mut body = Vec::new();
        for bytes in packets {
            lacing.extend(crate::page::lacing_for(bytes.len()));
            body.extend_from_slice(bytes);
        }
        Page {
            offset: u64::from(sequence) * 1000,
            flags,
            granule,
            serial: 7,
            sequence,
            lacing,
            body,
        }
    }

    fn stream() -> Opus {
        Opus {
            serial: 7,
            track: 0,
            pre_skip: 312,
            first_audio_sequence: 2,
            packet_cap: 61_440,
            state: State::fresh(),
        }
    }

    fn take(stream: &mut Opus, page: &Page) -> Result<Vec<Ready>, OggError> {
        let mut out = Vec::new();
        let mut gaps = Vec::new();
        let mut counters = Counters::default();
        stream.take_page(page, &mut out, &mut gaps, &mut counters)?;
        Ok(out)
    }

    fn times(ready: &[Ready]) -> Vec<(i64, i64)> {
        ready.iter().map(|item| (item.pts, item.duration)).collect()
    }

    #[test]
    fn the_first_page_works_backwards_from_its_granule() {
        let mut opus = stream();
        let out = take(&mut opus, &page(2, 0, 1920, &[packet(10), packet(12)])).expect("page");
        assert_eq!(times(&out), vec![(0, 960), (960, 960)]);
        let out = take(&mut opus, &page(3, 0, 2880, &[packet(9)])).expect("page");
        assert_eq!(times(&out), vec![(1920, 960)]);
    }

    #[test]
    fn a_cropped_stream_starts_above_zero() {
        let mut opus = stream();
        let out = take(&mut opus, &page(2, 0, 48_960, &[packet(10)])).expect("page");
        assert_eq!(times(&out), vec![(48_000, 960)]);
    }

    #[test]
    fn a_first_granule_below_its_samples_is_invalid_unless_it_ends_the_stream() {
        let mut opus = stream();
        assert!(matches!(
            take(&mut opus, &page(2, 0, 900, &[packet(10)])),
            Err(OggError::InitialGranule { granule: 900, .. })
        ));
        let mut short = stream();
        let out = take(
            &mut short,
            &page(2, FLAG_EOS, 1000, &[packet(10), packet(10)]),
        )
        .expect("short stream");
        assert_eq!(times(&out), vec![(0, 960), (960, 40)]);
        let mut empty = stream();
        assert!(matches!(
            take(&mut empty, &page(2, FLAG_EOS, 300, &[packet(10)])),
            Err(OggError::PreSkipPastEnd { end: 300, .. })
        ));
    }

    #[test]
    fn the_last_page_trims_the_final_packet() {
        let mut opus = stream();
        take(&mut opus, &page(2, 0, 960, &[packet(10)])).expect("page");
        let out = take(
            &mut opus,
            &page(3, FLAG_EOS, 2400, &[packet(10), packet(10)]),
        )
        .expect("eos page");
        assert_eq!(times(&out), vec![(960, 960), (1920, 480)]);
        assert_eq!(out[1].samples, 960);
        assert!(opus.state.ended);
    }

    #[test]
    fn a_granule_that_goes_back_is_refused() {
        let mut opus = stream();
        take(&mut opus, &page(2, 0, 1920, &[packet(10), packet(10)])).expect("page");
        assert!(matches!(
            take(&mut opus, &page(3, 0, 1000, &[packet(10)])),
            Err(OggError::GranuleBackwards { granule: 1000, .. })
        ));
        let mut overlap = stream();
        take(&mut overlap, &page(2, 0, 1920, &[packet(10), packet(10)])).expect("page");
        assert!(matches!(
            take(&mut overlap, &page(3, 0, 2000, &[packet(10)])),
            Err(OggError::GranuleBackwards { granule: 2000, .. })
        ));
        let mut eos_back = stream();
        take(&mut eos_back, &page(2, 0, 1920, &[packet(10), packet(10)])).expect("page");
        assert!(matches!(
            take(&mut eos_back, &page(3, FLAG_EOS, 1900, &[packet(10)])),
            Err(OggError::GranuleBackwards { .. })
        ));
    }

    #[test]
    fn a_packet_spans_three_pages() {
        let mut opus = stream();
        let big = packet(700);
        let mut first = page(2, 0, -1, &[]);
        first.lacing = vec![255];
        first.body = big[..255].to_vec();
        let mut second = page(3, FLAG_CONTINUED, -1, &[]);
        second.lacing = vec![255];
        second.body = big[255..510].to_vec();
        let mut third = page(4, FLAG_CONTINUED, 960, &[]);
        third.lacing = vec![190];
        third.body = big[510..].to_vec();
        assert!(take(&mut opus, &first).expect("first").is_empty());
        assert!(take(&mut opus, &second).expect("second").is_empty());
        let out = take(&mut opus, &third).expect("third");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].data.as_ref().expect("bytes"), &big);
        assert_eq!((out[0].pts, out[0].duration), (0, 960));
    }

    #[test]
    fn a_missing_page_drops_the_packet_that_spans_it_and_records_a_gap() {
        let mut opus = stream();
        take(&mut opus, &page(2, 0, 960, &[packet(10)])).expect("page");
        // Page 3 holds the start of a packet; it is lost.
        let mut fourth = page(4, FLAG_CONTINUED, 2880, &[]);
        fourth.lacing = vec![20, 10];
        let mut body = packet(20);
        body.extend(packet(10));
        fourth.body = body;
        let mut out = Vec::new();
        let mut gaps = Vec::new();
        let mut counters = Counters::default();
        opus.take_page(&fourth, &mut out, &mut gaps, &mut counters)
            .expect("page after gap");
        assert_eq!(times(&out), vec![(1920, 960)]);
        assert_eq!(counters.lost_pages, 1);
        assert_eq!(counters.dropped_pieces, 1);
        assert_eq!(
            gaps,
            vec![Gap::try_new(cutlass_container::TrackId::new(0), 960, 960).expect("gap")]
        );
    }

    #[test]
    fn an_open_packet_on_a_page_that_is_not_continued_is_dropped() {
        let mut opus = stream();
        let mut first = page(2, 0, -1, &[]);
        first.lacing = vec![255];
        first.body = vec![0xF8; 255];
        take(&mut opus, &first).expect("first");
        let out = take(&mut opus, &page(3, 0, 960, &[packet(10)])).expect("second");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].data.as_ref().expect("bytes").len(), 10);
    }

    #[test]
    fn a_packet_past_the_cap_is_reported_without_its_bytes() {
        let mut opus = stream();
        opus.packet_cap = 100;
        let out = take(&mut opus, &page(2, 0, 960, &[packet(300)])).expect("page");
        assert_eq!(out[0].data, Err((300, 100)));
        assert_eq!(out[0].duration, 960);
    }

    #[test]
    fn a_resumed_stream_anchors_on_the_landing_page() {
        let mut opus = stream();
        opus.state = State::resume(true);
        let mut landing = page(9, 0, 9600, &[packet(10)]);
        landing.lacing.push(255);
        landing.body.extend(packet(255));
        assert!(take(&mut opus, &landing).expect("landing").is_empty());
        assert_eq!(opus.state.phase, Phase::Running);
        let mut next = page(10, FLAG_CONTINUED, 10_560, &[]);
        next.lacing = vec![5];
        next.body = vec![0x55; 5];
        let out = take(&mut opus, &next).expect("next");
        assert_eq!(times(&out), vec![(9600, 960)]);
        assert_eq!(out[0].data.as_ref().expect("bytes").len(), 260);
    }

    #[test]
    fn header_pages_are_ignored() {
        let mut opus = stream();
        assert!(
            take(&mut opus, &page(1, 0, 0, &[packet(10)]))
                .expect("tags")
                .is_empty()
        );
        assert_eq!(opus.state, State::fresh());
    }

    #[test]
    fn a_malformed_toc_has_no_duration() {
        let mut opus = stream();
        let out = take(&mut opus, &page(2, 0, 960, &[Vec::new(), packet(10)])).expect("page");
        assert_eq!(times(&out), vec![(0, 0), (0, 960)]);
    }
}
