//! Open, read, and seek. Packet timestamps stay in the track timescale.
//! An edit list is exposed on the stream and is not applied to those timestamps.

use crate::brand::{self, parse_ftyp};
use crate::fragment::{self, SidxRef, Tfra};
use crate::heif;
use crate::index::Sample;
use crate::limits::{MAX_METADATA_BYTES, MAX_SAMPLES};
use crate::movie::{self, BuiltTrack, Trex};
use crate::parse::{self, file_header};
use cutlass_container::{Demux, Error, Packet, SeekAt, Source, Stream, Timescale};

struct Runtime {
    samples: Vec<Sample>,
    cursor: usize,
    no_sync: bool,
    next_dts: i64,
    timescale: u32,
}

struct Fragments {
    next: u64,
    trex: Vec<Trex>,
    sidx: Vec<SidxRef>,
    mfra: Option<Vec<Tfra>>,
    exhausted: bool,
}

/// A demuxer for MP4, QuickTime, M4A, M4V, 3GP, fragmented MP4, HEIC, and AVIF.
pub struct IsoBmff {
    movie_timescale: Timescale,
    brands: Vec<[u8; 4]>,
    streams: Vec<Stream>,
    tracks: Vec<Runtime>,
    media: Vec<(u64, u64)>,
    fragments: Option<Fragments>,
    file_len: u64,
}

impl IsoBmff {
    #[must_use]
    pub fn brands(&self) -> &[[u8; 4]] {
        &self.brands
    }

    pub fn open(source: &mut dyn Source) -> Result<Self, Error> {
        let file_len = source.len();
        if file_len < 8 {
            return Err(Error::Truncated { at: 0, needed: 8 });
        }
        let mut kind_buf = [0u8; 8];
        source.read_exact_at(0, &mut kind_buf)?;
        let kind = [kind_buf[4], kind_buf[5], kind_buf[6], kind_buf[7]];
        if kind != *b"ftyp" && !brand::headerless_quicktime(&kind) {
            return Err(Error::NotThisFamily);
        }
        let mut at = 0u64;
        let mut budget = 0u32;
        let mut brands = Vec::new();
        let mut saw_ftyp = false;
        let mut movie = None;
        let mut first_moof = None;
        let mut sidx = Vec::new();
        let mut media = Vec::new();
        let mut meta: Option<(u64, Vec<u8>)> = None;
        let mut headerless = false;

        while at < file_len {
            let Some(header) = file_header(source, at, file_len)? else {
                break;
            };
            match &header.kind {
                b"ftyp" => {
                    let payload = parse::load_payload(source, &header, MAX_METADATA_BYTES)?;
                    let parsed = parse_ftyp(&payload)?;
                    brands.push(parsed.major);
                    brands.extend(parsed.compatible);
                    saw_ftyp = true;
                }
                b"moov" => {
                    let payload = parse::load_payload(source, &header, MAX_METADATA_BYTES)?;
                    movie = Some(movie::parse_moov(&payload, &mut budget, 0)?);
                }
                b"moof" if movie.is_some() => {
                    first_moof = Some(header.start);
                    break;
                }
                b"sidx" => {
                    let payload = parse::load_payload(source, &header, MAX_METADATA_BYTES)?;
                    sidx.extend(fragment::parse_sidx(&payload, header.end)?);
                }
                b"meta" => {
                    let payload = parse::load_payload(source, &header, MAX_METADATA_BYTES)?;
                    meta = Some((header.payload_start, payload));
                }
                b"mdat" | b"idat" => media.push((header.payload_start, header.end)),
                kind if brand::headerless_quicktime(kind) => headerless = true,
                _ => {}
            }
            at = header.end;
        }

        if !saw_ftyp && movie.is_none() && meta.is_none() && !headerless {
            return Err(Error::NotThisFamily);
        }

        if let Some(built) = movie {
            if built.tracks.iter().any(|track| !track.self_contained) {
                return Err(Error::ExternalReference);
            }
            let fragments = if built.fragmented {
                Some(Fragments {
                    next: first_moof.unwrap_or(at),
                    trex: built.trex,
                    sidx,
                    mfra: None,
                    exhausted: first_moof.is_none(),
                })
            } else {
                None
            };
            return Ok(from_tracks(
                Timescale(built.timescale),
                brands,
                built.tracks,
                media,
                fragments,
                file_len,
            ));
        }

        if let Some((origin, payload)) = meta {
            let tracks = heif::parse_meta(&payload, origin, &mut budget, 0)?;
            if !tracks.is_empty() {
                return Ok(from_tracks(
                    Timescale(1),
                    brands,
                    tracks,
                    media,
                    None,
                    file_len,
                ));
            }
        }

        Err(Error::Missing { what: "moov" })
    }
}

fn restore_fragment_cursor(fragments: &mut Option<Fragments>, saved: Option<(u64, bool)>) {
    let Some((next, exhausted)) = saved else {
        return;
    };
    if let Some(fragments) = fragments {
        fragments.next = next;
        fragments.exhausted = exhausted;
    }
}

fn from_tracks(
    movie_timescale: Timescale,
    brands: Vec<[u8; 4]>,
    built: Vec<BuiltTrack>,
    media: Vec<(u64, u64)>,
    fragments: Option<Fragments>,
    file_len: u64,
) -> IsoBmff {
    let mut streams = Vec::new();
    let mut tracks = Vec::new();
    for track in built {
        let timescale = track.stream.params.timescale.get();
        let next_dts = track
            .samples
            .last()
            .map(|sample| sample.dts.saturating_add(i64::from(sample.duration)))
            .unwrap_or(0);
        streams.push(track.stream);
        tracks.push(Runtime {
            samples: track.samples,
            cursor: 0,
            no_sync: track.no_sync,
            next_dts,
            timescale,
        });
    }
    IsoBmff {
        movie_timescale,
        brands,
        streams,
        tracks,
        media,
        fragments,
        file_len,
    }
}

impl Demux for IsoBmff {
    fn movie_timescale(&self) -> Timescale {
        self.movie_timescale
    }

    fn streams(&self) -> &[Stream] {
        &self.streams
    }

    fn read_packet(&mut self, source: &mut dyn Source) -> Result<Option<Packet>, Error> {
        loop {
            if let Some((track_index, sample_index)) = self.next_loaded() {
                return self.emit(source, track_index, sample_index);
            }
            if self.fragments.is_some() && !self.fragment_done() {
                if !self.load_fragment(source, false)? {
                    return Ok(None);
                }
                continue;
            }
            return Ok(None);
        }
    }

    fn seek(
        &mut self,
        source: &mut dyn Source,
        track_id: u32,
        ticks: i64,
    ) -> Result<SeekAt, Error> {
        let Some(index) = self
            .streams
            .iter()
            .position(|stream| stream.params.track_id == track_id)
        else {
            return Err(Error::Missing { what: "track" });
        };
        if self.tracks.get(index).is_some_and(|track| track.no_sync) {
            return Err(Error::SeekLimited {
                reason: "track has no sync samples",
            });
        }
        if self.fragments.is_some() && !self.span_contains(index, ticks) {
            let saved = self.snapshot();
            if let Err(err) = self.land_fragment(source, index, ticks) {
                self.restore(saved);
                return Err(err);
            }
        }
        self.place(index, ticks)
    }
}

impl IsoBmff {
    fn fragment_done(&self) -> bool {
        self.fragments
            .as_ref()
            .is_some_and(|fragments| fragments.exhausted)
    }

    fn next_loaded(&self) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize, u64)> = None;
        for (track_index, track) in self.tracks.iter().enumerate() {
            let Some(sample) = track.samples.get(track.cursor) else {
                continue;
            };
            let replace = best.is_none_or(|(_, _, offset)| sample.offset < offset);
            if replace {
                best = Some((track_index, track.cursor, sample.offset));
            }
        }
        best.map(|(track, sample, _)| (track, sample))
    }

    fn emit(
        &mut self,
        source: &mut dyn Source,
        track_index: usize,
        sample_index: usize,
    ) -> Result<Option<Packet>, Error> {
        let (sample, track_id, timescale) = {
            let Some(track) = self.tracks.get(track_index) else {
                return Err(Error::Missing { what: "track" });
            };
            let Some(sample) = track.samples.get(sample_index).cloned() else {
                return Err(Error::Missing { what: "sample" });
            };
            let Some(stream) = self.streams.get(track_index) else {
                return Err(Error::Missing { what: "track" });
            };
            (sample, stream.params.track_id, stream.params.timescale)
        };
        if !contains_media(&self.media, sample.offset, sample.size) {
            return Err(Error::SampleExtendsPastMedia {
                offset: sample.offset,
                size: sample.size,
            });
        }
        let mut payload = vec![0u8; usize::try_from(sample.size).unwrap_or(0)];
        if payload.len() != usize::try_from(sample.size).unwrap_or(0) {
            return Err(Error::Limit {
                what: "packet",
                value: u64::from(sample.size),
            });
        }
        source.read_exact_at(sample.offset, &mut payload)?;
        if let Some(track) = self.tracks.get_mut(track_index) {
            track.cursor = sample_index.saturating_add(1);
        }
        Ok(Some(Packet {
            track_id,
            payload,
            dts: sample.dts,
            pts: sample.pts,
            duration: sample.duration,
            timescale,
            keyframe: sample.keyframe,
        }))
    }

    fn span_contains(&self, index: usize, ticks: i64) -> bool {
        let Some(track) = self.tracks.get(index) else {
            return false;
        };
        let Some(first) = track.samples.first() else {
            return false;
        };
        let Some(last) = track.samples.last() else {
            return false;
        };
        let end = last.dts.saturating_add(i64::from(last.duration));
        ticks >= first.dts && ticks < end
    }

    fn place(&mut self, index: usize, ticks: i64) -> Result<SeekAt, Error> {
        let movie_scale = self.movie_timescale.get();
        let landing = locate(self.tracks.get(index).map(|track| &track.samples), ticks)?;
        let landing_dts = landing.dts;
        let landing_scale = self
            .tracks
            .get(index)
            .map(|track| track.timescale)
            .unwrap_or(1);
        let movie_time = scale_ticks(landing_dts, landing_scale, movie_scale);
        let mut cursors = Vec::new();
        for (track_index, track) in self.tracks.iter().enumerate() {
            if track_index == index {
                cursors.push(landing.cursor);
                continue;
            }
            let media_time = scale_ticks(movie_time, movie_scale, track.timescale);
            cursors.push(locate_any(&track.samples, media_time));
        }
        for (track, cursor) in self.tracks.iter_mut().zip(cursors) {
            track.cursor = cursor;
        }
        let Some(stream) = self.streams.get(index) else {
            return Err(Error::Missing { what: "track" });
        };
        Ok(SeekAt {
            track_id: stream.params.track_id,
            dts: landing.dts,
            keyframe: landing.keyframe,
            used_index: true,
            preroll: landing.preroll,
        })
    }

    fn seek_fragment(
        &mut self,
        source: &mut dyn Source,
        index: usize,
        ticks: i64,
    ) -> Result<(), Error> {
        let track_id = self
            .streams
            .get(index)
            .map(|stream| stream.params.track_id)
            .ok_or(Error::Missing { what: "track" })?;
        let offset = self.fragment_offset(source, track_id, ticks)?;
        let saved = self
            .fragments
            .as_ref()
            .map(|fragments| (fragments.next, fragments.exhausted));
        if let Some(fragments) = &mut self.fragments {
            fragments.next = offset;
            fragments.exhausted = false;
        }
        match self.load_fragment(source, false) {
            Ok(true) => Ok(()),
            Ok(false) => {
                restore_fragment_cursor(&mut self.fragments, saved);
                Err(Error::SeekLimited {
                    reason: "fragment index does not cover that time",
                })
            }
            Err(err) => {
                restore_fragment_cursor(&mut self.fragments, saved);
                Err(err)
            }
        }
    }

    fn fragment_offset(
        &mut self,
        source: &mut dyn Source,
        track_id: u32,
        ticks: i64,
    ) -> Result<u64, Error> {
        let time = u64::try_from(ticks.max(0)).unwrap_or(0);
        if let Some(hit) = self.fragments.as_ref().and_then(|fragments| {
            fragments
                .sidx
                .iter()
                .filter(|row| row.track_id == track_id && row.time <= time)
                .max_by_key(|row| row.time)
                .map(|row| row.offset)
        }) {
            return Ok(hit);
        }
        self.ensure_mfra(source)?;
        if let Some(hit) = self.fragments.as_ref().and_then(|fragments| {
            fragments.mfra.as_ref().and_then(|rows| {
                rows.iter()
                    .filter(|row| row.track_id == track_id && row.time <= time)
                    .max_by_key(|row| row.time)
                    .map(|row| row.moof_offset)
            })
        }) {
            return Ok(hit);
        }
        let has_index = self.fragments.as_ref().is_some_and(|fragments| {
            !fragments.sidx.is_empty()
                || fragments.mfra.as_ref().is_some_and(|rows| !rows.is_empty())
        });
        if has_index {
            Err(Error::SeekLimited {
                reason: "fragment index does not cover that time",
            })
        } else {
            Err(Error::SeekLimited {
                reason: "fragmented file has no sidx or mfra",
            })
        }
    }

    fn ensure_mfra(&mut self, source: &mut dyn Source) -> Result<(), Error> {
        let already = self
            .fragments
            .as_ref()
            .is_some_and(|fragments| fragments.mfra.is_some());
        if already || self.file_len < 16 {
            if let Some(fragments) = &mut self.fragments
                && fragments.mfra.is_none()
            {
                fragments.mfra = Some(Vec::new());
            }
            return Ok(());
        }
        let mut tail = [0u8; 4];
        let at = self.file_len - 4;
        source.read_exact_at(at, &mut tail)?;
        let size = u64::from(u32::from_be_bytes(tail));
        if size < 16 || size > self.file_len || size > MAX_METADATA_BYTES {
            if let Some(fragments) = &mut self.fragments {
                fragments.mfra = Some(Vec::new());
            }
            return Ok(());
        }
        let start = self.file_len - size;
        let header = file_header(source, start, self.file_len)?;
        let Some(header) = header else {
            return Ok(());
        };
        if header.kind != *b"mfra" {
            if let Some(fragments) = &mut self.fragments {
                fragments.mfra = Some(Vec::new());
            }
            return Ok(());
        }
        let payload = parse::load_payload(source, &header, MAX_METADATA_BYTES)?;
        let mut budget = 0u32;
        let rows = fragment::parse_mfra(&payload, &mut budget)?;
        if let Some(fragments) = &mut self.fragments {
            fragments.mfra = Some(rows);
        }
        Ok(())
    }

    fn land_fragment(
        &mut self,
        source: &mut dyn Source,
        index: usize,
        ticks: i64,
    ) -> Result<(), Error> {
        self.seek_fragment(source, index, ticks)?;
        let mut steps = 0u32;
        while steps < 100_000 {
            steps += 1;
            if self.span_contains(index, ticks) || self.loaded_starts_after(index, ticks) {
                return Ok(());
            }
            if self.fragment_done() || !self.load_fragment(source, true)? {
                return Ok(());
            }
        }
        Err(Error::Limit {
            what: "fragment seek",
            value: u64::from(steps),
        })
    }

    fn loaded_starts_after(&self, index: usize, ticks: i64) -> bool {
        self.tracks
            .get(index)
            .and_then(|track| track.samples.first())
            .is_some_and(|sample| ticks < sample.dts)
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            tracks: self
                .tracks
                .iter()
                .map(|track| (track.samples.clone(), track.cursor, track.next_dts))
                .collect(),
            next: self
                .fragments
                .as_ref()
                .map(|fragments| (fragments.next, fragments.exhausted)),
        }
    }

    fn restore(&mut self, saved: Snapshot) {
        for (track, (samples, cursor, next_dts)) in self.tracks.iter_mut().zip(saved.tracks) {
            track.samples = samples;
            track.cursor = cursor;
            track.next_dts = next_dts;
        }
        restore_fragment_cursor(&mut self.fragments, saved.next);
    }

    fn load_fragment(&mut self, source: &mut dyn Source, append: bool) -> Result<bool, Error> {
        let Some(mut at) = self.fragments.as_ref().map(|fragments| fragments.next) else {
            return Ok(false);
        };
        if self.fragment_done() {
            return Ok(false);
        }
        let moof = loop {
            let Some(header) = file_header(source, at, self.file_len)? else {
                if let Some(fragments) = &mut self.fragments {
                    fragments.exhausted = true;
                }
                return Ok(false);
            };
            if header.kind == *b"moof" {
                break header;
            }
            if header.kind == *b"mfra" {
                if let Some(fragments) = &mut self.fragments {
                    fragments.exhausted = true;
                }
                return Ok(false);
            }
            at = header.end;
        };
        let payload = parse::load_payload(source, &moof, MAX_METADATA_BYTES)?;
        let trex = self
            .fragments
            .as_ref()
            .map(|fragments| fragments.trex.clone())
            .unwrap_or_default();
        let next_dts: Vec<(u32, i64)> = self
            .streams
            .iter()
            .zip(self.tracks.iter())
            .map(|(stream, track)| (stream.params.track_id, track.next_dts))
            .collect();
        let mut budget = 0u32;
        let groups = fragment::parse_moof(&payload, moof.start, &trex, &next_dts, &mut budget)?;
        let mut scan = moof.end;
        for _ in 0..8 {
            let Some(header) = file_header(source, scan, self.file_len)? else {
                break;
            };
            if header.kind == *b"mdat" || header.kind == *b"idat" {
                let range = (header.payload_start, header.end);
                if !self.media.contains(&range) {
                    self.media.push(range);
                }
                scan = header.end;
                break;
            }
            if header.kind == *b"moof" || header.kind == *b"mfra" {
                break;
            }
            scan = header.end;
        }
        if !append {
            for track in &mut self.tracks {
                track.samples.clear();
                track.cursor = 0;
            }
        }
        for group in groups {
            let Some(position) = self
                .streams
                .iter()
                .position(|stream| stream.params.track_id == group.track_id)
            else {
                continue;
            };
            let Some(track) = self.tracks.get_mut(position) else {
                continue;
            };
            if let Some(last) = group.samples.last() {
                track.next_dts = last.dts.saturating_add(i64::from(last.duration));
            }
            if append {
                let next = u64::try_from(track.samples.len()).unwrap_or(0)
                    + u64::try_from(group.samples.len()).unwrap_or(0);
                if next > MAX_SAMPLES {
                    return Err(Error::Limit {
                        what: "samples",
                        value: next,
                    });
                }
                track.samples.extend(group.samples);
            } else {
                track.samples = group.samples;
            }
        }
        if let Some(fragments) = &mut self.fragments {
            fragments.next = scan;
        }
        Ok(true)
    }
}

struct Snapshot {
    tracks: Vec<(Vec<Sample>, usize, i64)>,
    next: Option<(u64, bool)>,
}

struct Landing {
    cursor: usize,
    dts: i64,
    keyframe: bool,
    preroll: u32,
}

fn locate(samples: Option<&Vec<Sample>>, ticks: i64) -> Result<Landing, Error> {
    let Some(samples) = samples else {
        return Err(Error::SeekLimited {
            reason: "track has no sample index",
        });
    };
    if samples.is_empty() {
        return Err(Error::SeekLimited {
            reason: "track has no sample index",
        });
    }
    let Some(last) = samples.last() else {
        return Err(Error::SeekLimited {
            reason: "track has no sample index",
        });
    };
    let end = last.dts.saturating_add(i64::from(last.duration));
    if ticks >= end {
        return Ok(Landing {
            cursor: samples.len(),
            dts: end,
            keyframe: false,
            preroll: 0,
        });
    }
    let mut lo = 0usize;
    let mut hi = samples.len();
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        let Some(sample) = samples.get(mid) else {
            break;
        };
        if sample.dts <= ticks {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if ticks < samples.first().map(|sample| sample.dts).unwrap_or(0) {
        lo = 0;
    }
    let mut cursor = lo;
    while cursor > 0 && samples.get(cursor).is_some_and(|sample| !sample.keyframe) {
        cursor -= 1;
    }
    let Some(sample) = samples.get(cursor) else {
        return Err(Error::Missing { what: "sample" });
    };
    if !sample.keyframe {
        return Err(Error::SeekLimited {
            reason: "no sync sample at or before that time",
        });
    }
    let preroll = u32::try_from(lo.saturating_sub(cursor)).unwrap_or(u32::MAX);
    Ok(Landing {
        cursor,
        dts: sample.dts,
        keyframe: true,
        preroll,
    })
}

fn locate_any(samples: &[Sample], ticks: i64) -> usize {
    if samples.is_empty() {
        return 0;
    }
    let Some(last) = samples.last() else {
        return 0;
    };
    if ticks >= last.dts.saturating_add(i64::from(last.duration)) {
        return samples.len();
    }
    let mut lo = 0usize;
    let mut hi = samples.len();
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if samples.get(mid).is_some_and(|sample| sample.dts <= ticks) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if ticks < samples.first().map(|sample| sample.dts).unwrap_or(0) {
        0
    } else {
        lo
    }
}

fn contains_media(ranges: &[(u64, u64)], offset: u64, size: u32) -> bool {
    let Some(end) = offset.checked_add(u64::from(size)) else {
        return false;
    };
    ranges
        .iter()
        .any(|(start, stop)| offset >= *start && end <= *stop)
}

fn scale_ticks(ticks: i64, from: u32, to: u32) -> i64 {
    if from == 0 || to == 0 {
        return 0;
    }
    let negative = ticks < 0;
    let scaled = u128::from(ticks.unsigned_abs()) * u128::from(to) / u128::from(from);
    let value = i64::try_from(scaled).unwrap_or(i64::MAX);
    if negative { -value } else { value }
}
