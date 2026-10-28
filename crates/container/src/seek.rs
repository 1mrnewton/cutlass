//! Seek in the file timescale.
//!
//! An index seek answers from index entries and does not take packet bytes.
//! A scan seek reads timestamps from the start of the packet list until the
//! landing is known. Neither path converts ticks to microseconds.
//!
//! After a seek, packets whose ordinal is below the landing are stale and
//! must not be returned. A one-packet read-ahead is dropped as well: the
//! next packet is the landing.

use crate::error::{OpenError, SeekError};
use crate::limits::Limits;

/// One seek-index row for a single track, sorted by [`Self::time`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexEntry {
    pub track: u32,
    pub time: i64,
    pub ordinal: u32,
    pub keyframe: bool,
    pub byte_position: u64,
}

/// Where to seek, in one track's file timescale.
///
/// `end` is the exclusive end of the track when the file states one.
/// `None` means the last indexed timestamp is the last known instant:
/// a later time is past the end, and that timestamp itself is a hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeekTo {
    pub track: u32,
    pub time: i64,
    pub end: Option<i64>,
}

/// The packet a seek lands on, or the end of the track.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Landing {
    Packet {
        ordinal: u32,
        /// Presentation time of the landing packet, in the file timescale.
        time: i64,
        keyframe: bool,
        /// Packets from the landing onward whose time is strictly before
        /// the requested time. They are still returned, in decode order.
        preroll: u32,
        byte_position: u64,
    },
    End,
}

/// A seek result. `payload_scans` is the number of packet payloads inspected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeekLanding {
    pub landing: Landing,
    pub index_used: bool,
    pub payload_scans: u32,
    pub entries_visited: u32,
}

/// A packet header a scan seek may look at. The payload is not included.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanEntry {
    pub stream: u32,
    pub presentation: Option<i64>,
    pub keyframe: bool,
    pub ordinal: u32,
    pub byte_position: u64,
}

/// Checks that `index` is one track and that times do not go backwards.
pub fn ensure_index_sorted(index: &[IndexEntry], track: u32) -> Result<(), SeekError> {
    let mut previous = None;
    for entry in index {
        if entry.track != track {
            return Err(SeekError::MixedTrack);
        }
        if let Some(previous) = previous
            && entry.time < previous
        {
            return Err(SeekError::UnsortedIndex);
        }
        previous = Some(entry.time);
    }
    Ok(())
}

/// Reserves index storage after refusing a count above `limit`.
///
/// `u64::MAX` returns before any buffer of that length is allocated.
pub fn reserve_index(count: u64, limits: &Limits) -> Result<Vec<IndexEntry>, OpenError> {
    let limit = u64::from(limits.max_index_entries);
    if count > limit {
        return Err(OpenError::HostileSize {
            declared: count,
            limit,
        });
    }
    let len = usize::try_from(count).map_err(|_| OpenError::HostileSize {
        declared: count,
        limit,
    })?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(len)
        .map_err(|_| OpenError::HostileSize {
            declared: count,
            limit,
        })?;
    Ok(entries)
}

/// Seeks with a sorted single-track index.
///
/// The slice is not a packet payload. `payload_scans` is zero.
pub fn seek_with_index(index: &[IndexEntry], to: SeekTo) -> Result<SeekLanding, SeekError> {
    if index.is_empty() {
        return Err(SeekError::NoIndex);
    }
    if to.end.is_some_and(|end| to.time >= end) {
        return Ok(SeekLanding {
            landing: Landing::End,
            index_used: true,
            payload_scans: 0,
            entries_visited: 0,
        });
    }
    let last = index.len() - 1;
    if to.end.is_none() && to.time > index[last].time {
        return Ok(SeekLanding {
            landing: Landing::End,
            index_used: true,
            payload_scans: 0,
            entries_visited: 0,
        });
    }

    let mut steps = 0_u32;
    let mut lo = 0;
    let mut hi = index.len();
    let mut found = None;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        steps = steps.saturating_add(1);
        if index[mid].time <= to.time {
            found = Some(mid);
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    let landing_at = match found {
        Some(mut index_of) => {
            while index_of > 0 && !index[index_of].keyframe {
                index_of -= 1;
                steps = steps.saturating_add(1);
            }
            index_of
        }
        None => 0,
    };

    let mut preroll = 0_u32;
    for entry in index.iter().skip(landing_at) {
        if entry.time >= to.time {
            break;
        }
        preroll = preroll.saturating_add(1);
        steps = steps.saturating_add(1);
    }

    let entry = index[landing_at];
    Ok(SeekLanding {
        landing: Landing::Packet {
            ordinal: entry.ordinal,
            time: entry.time,
            keyframe: entry.keyframe,
            preroll,
            byte_position: entry.byte_position,
        },
        index_used: true,
        payload_scans: 0,
        entries_visited: steps,
    })
}

/// Seeks by walking timestamps from the start of `packets`.
///
/// `payload_scans` stays zero: [`ScanEntry`] has no payload. `entries_visited`
/// counts packets looked at, including other tracks, and stops once the
/// requested time has been passed.
pub fn seek_by_scan(packets: &[ScanEntry], to: SeekTo) -> Result<SeekLanding, SeekError> {
    if to.end.is_some_and(|end| to.time >= end) {
        return Ok(SeekLanding {
            landing: Landing::End,
            index_used: false,
            payload_scans: 0,
            entries_visited: 0,
        });
    }

    let mut visited = 0_u32;
    let mut first: Option<ScanEntry> = None;
    let mut keyframe: Option<ScanEntry> = None;
    let mut frames_before_target = 0_u32;
    let mut saw_track = false;
    let mut last_time = None;

    for packet in packets {
        visited = visited.saturating_add(1);
        if packet.stream != to.track {
            continue;
        }
        let Some(time) = packet.presentation else {
            continue;
        };
        saw_track = true;
        last_time = Some(time);
        if first.is_none() {
            first = Some(*packet);
        }
        if time > to.time {
            break;
        }
        if packet.keyframe {
            keyframe = Some(*packet);
            frames_before_target = 0;
        }
        if time < to.time {
            frames_before_target = frames_before_target.saturating_add(1);
        }
    }

    if !saw_track {
        return Err(SeekError::Empty);
    }
    if to.end.is_none() && last_time.is_some_and(|time| to.time > time) {
        return Ok(SeekLanding {
            landing: Landing::End,
            index_used: false,
            payload_scans: 0,
            entries_visited: visited,
        });
    }

    let Some(landing) = keyframe.or(first) else {
        return Err(SeekError::Empty);
    };
    let Some(time) = landing.presentation else {
        return Err(SeekError::Empty);
    };
    let preroll = if time >= to.time {
        0
    } else {
        frames_before_target
    };
    Ok(SeekLanding {
        landing: Landing::Packet {
            ordinal: landing.ordinal,
            time,
            keyframe: landing.keyframe,
            preroll,
            byte_position: landing.byte_position,
        },
        index_used: false,
        payload_scans: 0,
        entries_visited: visited,
    })
}

/// Drops a buffered ordinal that sits before the landing.
///
/// `None` means the buffer must not be returned. An ordinal at or after
/// `floor` may still be returned by a caller that has not reset its cursor.
/// A demuxer seek resets the cursor to `floor` and drops the buffer entirely.
#[must_use]
pub fn suppress_stale(buffered: Option<u32>, floor: u32) -> Option<u32> {
    match buffered {
        Some(ordinal) if ordinal >= floor => Some(ordinal),
        Some(_) | None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        IndexEntry, Landing, ScanEntry, SeekTo, ensure_index_sorted, reserve_index, seek_by_scan,
        seek_with_index, suppress_stale,
    };
    use crate::error::{OpenError, SeekError};
    use crate::limits::Limits;

    fn entry(ordinal: u32, time: i64, keyframe: bool) -> IndexEntry {
        IndexEntry {
            track: 1,
            time,
            ordinal,
            keyframe,
            byte_position: u64::from(ordinal) * 100,
        }
    }

    fn index() -> Vec<IndexEntry> {
        vec![
            entry(0, 0, true),
            entry(2, 10, false),
            entry(1, 20, true),
            entry(3, 30, false),
        ]
    }

    fn at(time: i64, end: Option<i64>) -> SeekTo {
        SeekTo {
            track: 1,
            time,
            end,
        }
    }

    fn packet(landing: Landing) -> (u32, i64, bool, u32, u64) {
        match landing {
            Landing::Packet {
                ordinal,
                time,
                keyframe,
                preroll,
                byte_position,
            } => (ordinal, time, keyframe, preroll, byte_position),
            Landing::End => panic!("expected a packet"),
        }
    }

    #[test]
    fn an_index_seek_covers_the_landing_table_without_reading_payloads() {
        let index = index();
        let end = Some(40);

        let before = seek_with_index(&index, at(-1, end)).expect("before");
        assert_eq!(packet(before.landing), (0, 0, true, 0, 0));
        assert!(before.index_used);
        assert_eq!(before.payload_scans, 0);

        let first = seek_with_index(&index, at(0, end)).expect("first");
        assert_eq!(packet(first.landing), (0, 0, true, 0, 0));

        let on_delta = seek_with_index(&index, at(10, end)).expect("between");
        assert_eq!(packet(on_delta.landing), (0, 0, true, 1, 0));

        let between = seek_with_index(&index, at(15, end)).expect("between");
        assert_eq!(packet(between.landing), (0, 0, true, 2, 0));

        let key = seek_with_index(&index, at(20, end)).expect("key");
        assert_eq!(packet(key.landing), (1, 20, true, 0, 100));

        let after_key = seek_with_index(&index, at(25, end)).expect("after key");
        assert_eq!(packet(after_key.landing), (1, 20, true, 1, 100));

        let last = seek_with_index(&index, at(30, end)).expect("last");
        assert_eq!(packet(last.landing), (1, 20, true, 1, 100));

        assert!(matches!(
            seek_with_index(&index, at(40, end)).expect("end").landing,
            Landing::End
        ));
        assert!(matches!(
            seek_with_index(&index, at(31, None)).expect("past").landing,
            Landing::End
        ));
        assert_eq!(
            packet(seek_with_index(&index, at(30, None)).expect("hit").landing).1,
            20
        );
    }

    #[test]
    fn a_large_index_seek_stays_logarithmic_and_reads_no_payload() {
        let index: Vec<IndexEntry> = (0..10_000)
            .map(|ordinal| IndexEntry {
                track: 7,
                time: i64::from(ordinal) * 10,
                ordinal,
                keyframe: true,
                byte_position: u64::from(ordinal),
            })
            .collect();
        let landing = seek_with_index(
            &index,
            SeekTo {
                track: 7,
                time: 50_000,
                end: Some(100_000),
            },
        )
        .expect("seek");
        assert_eq!(packet(landing.landing).0, 5_000);
        assert!(landing.index_used);
        assert_eq!(landing.payload_scans, 0);
        assert!(landing.entries_visited <= 64, "{}", landing.entries_visited);
    }

    #[test]
    fn an_empty_or_unsorted_index_does_not_invent_a_landing() {
        assert_eq!(seek_with_index(&[], at(0, None)), Err(SeekError::NoIndex));
        let mut reversed = index();
        reversed.swap(0, 3);
        assert_eq!(
            ensure_index_sorted(&reversed, 1),
            Err(SeekError::UnsortedIndex)
        );
        let mut mixed = index();
        mixed[1].track = 2;
        assert_eq!(ensure_index_sorted(&mixed, 1), Err(SeekError::MixedTrack));
        assert!(ensure_index_sorted(&index(), 1).is_ok());
    }

    #[test]
    fn a_scan_stops_at_the_target_and_matches_the_index_landing() {
        let packets = vec![
            ScanEntry {
                stream: 0,
                presentation: Some(0),
                keyframe: true,
                ordinal: 0,
                byte_position: 10,
            },
            ScanEntry {
                stream: 1,
                presentation: Some(0),
                keyframe: true,
                ordinal: 1,
                byte_position: 20,
            },
            ScanEntry {
                stream: 1,
                presentation: Some(10),
                keyframe: false,
                ordinal: 2,
                byte_position: 30,
            },
            ScanEntry {
                stream: 1,
                presentation: Some(20),
                keyframe: true,
                ordinal: 3,
                byte_position: 40,
            },
            ScanEntry {
                stream: 1,
                presentation: Some(30),
                keyframe: false,
                ordinal: 4,
                byte_position: 50,
            },
        ];
        let landing = seek_by_scan(
            &packets,
            SeekTo {
                track: 1,
                time: 15,
                end: Some(40),
            },
        )
        .expect("scan");
        assert!(!landing.index_used);
        assert_eq!(landing.payload_scans, 0);
        assert_eq!(packet(landing.landing), (1, 0, true, 2, 20));
        assert!(landing.entries_visited < u32::try_from(packets.len()).expect("len"));

        let past = seek_by_scan(
            &packets,
            SeekTo {
                track: 1,
                time: 31,
                end: None,
            },
        )
        .expect("past");
        assert!(matches!(past.landing, Landing::End));
        assert_eq!(
            past.entries_visited,
            u32::try_from(packets.len()).expect("len")
        );

        let last = seek_by_scan(
            &packets,
            SeekTo {
                track: 1,
                time: 30,
                end: None,
            },
        )
        .expect("last");
        assert_eq!(packet(last.landing).0, 3);
    }

    #[test]
    fn stale_ordinals_below_the_landing_are_dropped() {
        assert_eq!(suppress_stale(Some(2), 3), None);
        assert_eq!(suppress_stale(Some(3), 3), Some(3));
        assert_eq!(suppress_stale(Some(5), 3), Some(5));
        assert_eq!(suppress_stale(None, 3), None);
    }

    #[test]
    fn reserving_an_index_refuses_a_hostile_count() {
        let limits = Limits::standard();
        assert!(matches!(
            reserve_index(u64::MAX, &limits),
            Err(OpenError::HostileSize {
                declared: u64::MAX,
                ..
            })
        ));
        let over = u64::from(limits.max_index_entries) + 1;
        assert!(reserve_index(over, &limits).is_err());
        let reserved = reserve_index(4, &limits).expect("four");
        assert!(reserved.capacity() >= 4);
        assert!(reserved.is_empty());
    }
}
