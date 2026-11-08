//! The Vorbis decoder behind [`cutlass_decode::Decoder`] (Vorbis I section
//! 4.3).
//!
//! One push is one Vorbis packet. An audio packet returns the samples
//! between the centers of the previous block and this one
//! (`previous / 4 + current / 4` per channel); the first packet after open
//! or [`Decoder::reset`] returns none. Output is interleaved `f32` at full
//! scale 1.0 in the Vorbis channel order, unclipped.
//!
//! A packet stamped with a duration shorter than it decodes is an end trim
//! (`skip_trailing`), as the Ogg end-of-stream granule makes it. A duration
//! of 0 is a length only once a packet has come stamped with a length (the
//! source stamps them, as Ogg does): then a packet wholly past the
//! end-of-stream granule is trimmed whole. Before that, 0 is no length and
//! nothing is trimmed. A negative
//! start is the container's edit. Header packets in the stream (a chained
//! link) and a `SideData::Config` that differs from the current one replace
//! the configuration and start again as after a reset.

use crate::bits::{BitReader, ilog};
use crate::floor::{ChannelFloor, Status};
use crate::header::{
    COMMENT_HEADER, Headers, ID_HEADER, Identification, MAX_CHANNELS, SETUP_HEADER,
};
use crate::mdct::{Mdct, Scratch as MdctScratch};
use crate::residue::{self, Scratch as ResidueScratch};
use crate::setup::{Floor, Setup};
use crate::window::Windows;
use cutlass_decode::{
    AudioOutput, Channel, ChannelLayout, CodecError, CodecId, CodecParams, Decoder, Output, Packet,
    Pcm, SideData,
};

/// One configured stream: headers and the tables built from them.
#[derive(Clone, Debug)]
struct Stream {
    headers: Headers,
    id: Identification,
    setup: Setup,
    layout: ChannelLayout,
    windows: Windows,
    transforms: [Mdct; 2],
}

impl Stream {
    fn open(headers: Headers) -> Result<Self, CodecError> {
        let id = Identification::parse(&headers.identification)?;
        let channels = usize::from(id.channels);
        if channels > MAX_CHANNELS {
            return Err(CodecError::Unsupported {
                what: "more than eight Vorbis channels",
            });
        }
        let setup = Setup::parse(&headers.setup, channels, id.blocksizes)?;
        Ok(Self {
            layout: vorbis_layout(channels)?,
            windows: Windows::new(id.blocksizes),
            transforms: [Mdct::new(id.blocksizes[0]), Mdct::new(id.blocksizes[1])],
            headers,
            id,
            setup,
        })
    }

    fn channels(&self) -> usize {
        usize::from(self.id.channels)
    }
}

/// The Vorbis channel order (section 4.3.9) as speakers.
pub(crate) fn vorbis_layout(channels: usize) -> Result<ChannelLayout, CodecError> {
    use Channel::{
        BackCenter, BackLeft, BackRight, FrontCenter, FrontLeft, FrontRight, Lfe, SideLeft,
        SideRight,
    };
    let speakers = match channels {
        1 => vec![FrontCenter],
        2 => vec![FrontLeft, FrontRight],
        3 => vec![FrontLeft, FrontCenter, FrontRight],
        4 => vec![FrontLeft, FrontRight, BackLeft, BackRight],
        5 => vec![FrontLeft, FrontCenter, FrontRight, BackLeft, BackRight],
        6 => vec![FrontLeft, FrontCenter, FrontRight, BackLeft, BackRight, Lfe],
        7 => vec![
            FrontLeft,
            FrontCenter,
            FrontRight,
            SideLeft,
            SideRight,
            BackCenter,
            Lfe,
        ],
        8 => vec![
            FrontLeft,
            FrontCenter,
            FrontRight,
            SideLeft,
            SideRight,
            BackLeft,
            BackRight,
            Lfe,
        ],
        _ => {
            return Err(CodecError::Unsupported {
                what: "more than eight Vorbis channels",
            });
        }
    };
    ChannelLayout::new(speakers).map_err(|_| CodecError::Limit {
        what: "channel layout",
    })
}

/// What stays between packets: the windowed right half of the last block.
#[derive(Clone, Debug, Default)]
struct Overlap {
    /// Size of the last block; `None` before the first packet.
    previous: Option<usize>,
    /// `previous / 2` samples per channel, channel after channel.
    samples: Vec<f32>,
}

/// Per-packet work buffers. Not state: a failed packet may leave them in
/// any condition.
#[derive(Clone, Debug, Default)]
struct Work {
    floors: Vec<ChannelFloor>,
    spectra: Vec<f32>,
    submap: Vec<f32>,
    block: Vec<f32>,
    mdct: MdctScratch,
    residue: ResidueScratch,
}

/// Header packets of a chained link, gathered until its setup arrives.
#[derive(Clone, Debug, Default)]
struct Pending {
    identification: Option<Vec<u8>>,
    comment: Option<Vec<u8>>,
}

/// The Vorbis decoder.
#[derive(Clone, Debug)]
pub struct VorbisDecoder {
    stream: Stream,
    overlap: Overlap,
    pending: Pending,
    work: Work,
    /// A packet came with a clock and a nonzero duration, so a duration of
    /// 0 is a length rather than none.
    stamped: bool,
}

/// The result of decoding one audio packet, before it is committed.
struct Decoded {
    long: bool,
    /// The packet's window flags (section 4.3.1); short blocks have none.
    previous_long: bool,
    next_long: bool,
    /// Channels whose floor is unused: their block is silent.
    silent: Vec<bool>,
}

impl VorbisDecoder {
    /// Opens a decoder from `params.extra`, the three header packets in
    /// any form [`Headers::split`] takes.
    ///
    /// # Errors
    ///
    /// `Unsupported` for another codec or more than eight channels, and the
    /// header errors of [`Identification::parse`] and the setup parse.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Vorbis {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        Self::from_config(&params.extra)
    }

    /// Opens a decoder from a configuration blob.
    ///
    /// # Errors
    ///
    /// As [`Self::open`].
    pub fn from_config(config: &[u8]) -> Result<Self, CodecError> {
        let stream = Stream::open(Headers::split(config)?)?;
        Ok(Self {
            stream,
            overlap: Overlap::default(),
            pending: Pending::default(),
            work: Work::default(),
            stamped: false,
        })
    }

    /// The current identification header.
    #[must_use]
    pub fn identification(&self) -> Identification {
        self.stream.id
    }

    /// Output channel layout.
    #[must_use]
    pub fn layout(&self) -> &ChannelLayout {
        &self.stream.layout
    }

    /// Decodes one packet into interleaved samples.
    ///
    /// # Errors
    ///
    /// `InvalidApi` for an empty packet, `Truncated` when an audio packet
    /// ends before its mode and window flags, `Malformed` for a bad mode or
    /// floor 0 book, and header errors for a bad in-band header. The
    /// decoder is unchanged after an error.
    pub fn decode(&mut self, data: &[u8]) -> Result<Vec<f32>, CodecError> {
        let Some(&first) = data.first() else {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        };
        if first & 1 == 1 {
            self.header_packet(data)?;
            return Ok(Vec::new());
        }
        let decoded = self.decode_audio(data)?;
        Ok(self.synthesize(&decoded))
    }

    fn header_packet(&mut self, data: &[u8]) -> Result<(), CodecError> {
        match data[0] {
            ID_HEADER => {
                Identification::parse(data)?;
                self.pending.identification = Some(data.to_vec());
                self.pending.comment = None;
                Ok(())
            }
            COMMENT_HEADER => {
                self.pending.comment = Some(data.to_vec());
                Ok(())
            }
            SETUP_HEADER => {
                let identification = self
                    .pending
                    .identification
                    .clone()
                    .unwrap_or_else(|| self.stream.headers.identification.clone());
                let headers = Headers {
                    identification,
                    comment: self
                        .pending
                        .comment
                        .clone()
                        .unwrap_or_else(|| self.stream.headers.comment.clone()),
                    setup: data.to_vec(),
                };
                self.replace(headers)?;
                self.pending = Pending::default();
                Ok(())
            }
            _ => Err(CodecError::Malformed {
                at_bit: 0,
                what: "Vorbis packet type",
            }),
        }
    }

    /// Takes new headers; the same headers again change nothing.
    fn replace(&mut self, headers: Headers) -> Result<(), CodecError> {
        if headers.identification == self.stream.headers.identification
            && headers.setup == self.stream.headers.setup
        {
            return Ok(());
        }
        self.stream = Stream::open(headers)?;
        self.overlap = Overlap::default();
        Ok(())
    }

    /// Floors, residues, coupling, and the floor product (sections 4.3.1 to
    /// 4.3.6) into `work.spectra`. Nothing outside `work` changes.
    fn decode_audio(&mut self, data: &[u8]) -> Result<Decoded, CodecError> {
        let stream = &self.stream;
        let setup = &stream.setup;
        let work = &mut self.work;
        let channels = stream.channels();
        let mut reader = BitReader::new(data);
        let reader = &mut reader;
        let truncated = |what| CodecError::Truncated { at_bit: 0, what };
        reader.read(1).ok_or(truncated("audio packet type"))?;
        let mode_count = setup.modes.len() as u32;
        let mode_at = reader.position();
        let mode_number = reader
            .read(ilog(mode_count - 1))
            .ok_or(truncated("audio packet mode"))?;
        let Some(&mode) = setup.modes.get(mode_number as usize) else {
            return Err(CodecError::Malformed {
                at_bit: mode_at,
                what: "audio packet mode",
            });
        };
        let (previous_long, next_long) = if mode.long {
            let flags = reader
                .read(2)
                .ok_or(truncated("audio packet window flags"))?;
            (flags & 1 == 1, flags & 2 == 2)
        } else {
            (false, false)
        };
        let n = stream.id.blocksizes[usize::from(mode.long)];
        let half = n / 2;
        let mapping = &setup.mappings[usize::from(mode.mapping)];

        work.floors.resize_with(channels, ChannelFloor::default);
        work.spectra.clear();
        work.spectra.resize(channels * half, 0.0);
        let mut silent = vec![true; channels];
        let mut no_residue = vec![true; channels];
        let mut ended = false;
        for channel in 0..channels {
            let submap = usize::from(mapping.mux[channel]);
            let floor = &setup.floors[usize::from(mapping.submaps[submap].0)];
            let status = match floor {
                Floor::One(floor) => {
                    work.floors[channel].decode_floor1(floor, &setup.books, reader)
                }
                Floor::Zero(floor) => {
                    work.floors[channel].decode_floor0(floor, &setup.books, reader)?
                }
            };
            match status {
                Status::Used => {
                    silent[channel] = false;
                    no_residue[channel] = false;
                }
                Status::Unused => {}
                Status::EndOfPacket => {
                    ended = true;
                    break;
                }
            }
        }
        if ended {
            // Section 4.3.2: every channel is zero for this block.
            return Ok(Decoded {
                long: mode.long,
                previous_long,
                next_long,
                silent: vec![true; channels],
            });
        }
        for &(magnitude, angle) in &mapping.coupling {
            let (m, a) = (usize::from(magnitude), usize::from(angle));
            if !no_residue[m] || !no_residue[a] {
                no_residue[m] = false;
                no_residue[a] = false;
            }
        }
        for (submap, &(_, residue_number)) in mapping.submaps.iter().enumerate() {
            let members: Vec<usize> = (0..channels)
                .filter(|&c| usize::from(mapping.mux[c]) == submap)
                .collect();
            if members.is_empty() {
                continue;
            }
            let skip: Vec<bool> = members.iter().map(|&c| no_residue[c]).collect();
            let residue = &setup.residues[usize::from(residue_number)];
            if mapping.submaps.len() == 1 {
                residue::decode(
                    residue,
                    &setup.books,
                    reader,
                    half,
                    &skip,
                    &mut work.spectra,
                    &mut work.residue,
                );
            } else {
                work.submap.clear();
                work.submap.resize(members.len() * half, 0.0);
                residue::decode(
                    residue,
                    &setup.books,
                    reader,
                    half,
                    &skip,
                    &mut work.submap,
                    &mut work.residue,
                );
                for (slot, &channel) in members.iter().enumerate() {
                    work.spectra[channel * half..(channel + 1) * half]
                        .copy_from_slice(&work.submap[slot * half..(slot + 1) * half]);
                }
            }
        }
        for &(magnitude, angle) in mapping.coupling.iter().rev() {
            let (m, a) = (usize::from(magnitude), usize::from(angle));
            for i in 0..half {
                let mv = work.spectra[m * half + i];
                let av = work.spectra[a * half + i];
                let (new_m, new_a) = if mv > 0.0 {
                    if av > 0.0 {
                        (mv, mv - av)
                    } else {
                        (mv + av, mv)
                    }
                } else if av > 0.0 {
                    (mv, mv + av)
                } else {
                    (mv - av, mv)
                };
                work.spectra[m * half + i] = new_m;
                work.spectra[a * half + i] = new_a;
            }
        }
        for (channel, _) in silent.iter().enumerate().filter(|&(_, &silent)| !silent) {
            let submap = usize::from(mapping.mux[channel]);
            let spectrum = &mut work.spectra[channel * half..(channel + 1) * half];
            match &setup.floors[usize::from(mapping.submaps[submap].0)] {
                Floor::One(floor) => work.floors[channel].apply_floor1(floor, spectrum),
                Floor::Zero(floor) => {
                    work.floors[channel].apply_floor0(floor, mode.long, spectrum)?
                }
            }
        }
        Ok(Decoded {
            long: mode.long,
            previous_long,
            next_long,
            silent,
        })
    }

    /// The inverse MDCT, window, and overlap-add (sections 4.3.7 to 4.3.8);
    /// commits the overlap. Returns the finished interleaved samples.
    ///
    /// The window comes from the packet's flags; where the blocks overlap
    /// comes from their sizes: the last block's 3/4 point meets this
    /// block's 1/4 point.
    fn synthesize(&mut self, decoded: &Decoded) -> Vec<f32> {
        let stream = &self.stream;
        let work = &mut self.work;
        let channels = stream.channels();
        let n = stream.id.blocksizes[usize::from(decoded.long)];
        let half = n / 2;
        let shape = stream
            .windows
            .shape(decoded.long, decoded.previous_long, decoded.next_long);
        let previous = self.overlap.previous;
        let count = previous.map_or(0, |p| p / 4 + n / 4);
        let mut out = vec![0.0_f32; count * channels];
        let mut next = vec![0.0_f32; channels * half];
        work.block.resize(n, 0.0);
        for channel in 0..channels {
            let block = &mut work.block[..n];
            if decoded.silent[channel] {
                block.fill(0.0);
            } else {
                stream.transforms[usize::from(decoded.long)].inverse(
                    &work.spectra[channel * half..(channel + 1) * half],
                    block,
                    &mut work.mdct,
                );
                stream.windows.apply(&shape, block);
            }
            if let Some(p) = previous {
                let saved = &self.overlap.samples[channel * (p / 2)..(channel + 1) * (p / 2)];
                let start = n as isize / 4 - p as isize / 4;
                for (i, slot) in out.iter_mut().skip(channel).step_by(channels).enumerate() {
                    let at = start + i as isize;
                    let mut value = saved.get(i).copied().unwrap_or(0.0);
                    if at >= 0 {
                        value += block[at as usize];
                    }
                    *slot = value;
                }
            }
            next[channel * half..(channel + 1) * half].copy_from_slice(&block[half..n]);
        }
        self.overlap = Overlap {
            previous: Some(n),
            samples: next,
        };
        out
    }

    fn output(&mut self, samples: Vec<f32>, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        if packet.timescale > 0 && packet.duration > 0 {
            self.stamped = true;
        }
        let channels = self.stream.channels();
        let count = samples.len() / channels;
        if count == 0 {
            return Ok(Vec::new());
        }
        let rate = self.stream.id.sample_rate;
        let trailing = if packet.timescale > 0 && (packet.duration > 0 || self.stamped) {
            let wanted =
                i128::from(packet.duration) * i128::from(rate) / i128::from(packet.timescale);
            (count as i128 - wanted).clamp(0, count as i128) as u32
        } else {
            0
        };
        let pcm = Pcm::from_f32(rate, self.stream.layout.clone(), samples)
            .map_err(|_| CodecError::Limit { what: "PCM buffer" })?;
        Ok(vec![Output::Audio(AudioOutput {
            pcm,
            pts: packet.pts,
            timescale: packet.timescale,
            skip_leading: 0,
            skip_trailing: trailing,
            concealed: false,
        })])
    }
}

impl Decoder for VorbisDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Vorbis
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let config = packet.side.iter().find_map(|side| match side {
            SideData::Config(config) => Some(config.as_slice()),
            _ => None,
        });
        let before = config.map(|_| (self.stream.clone(), self.overlap.clone()));
        if let Some(config) = config {
            let headers = Headers::split(config)?;
            self.replace(headers)?;
        }
        let samples = match self.decode(&packet.data) {
            Ok(samples) => samples,
            Err(error) => {
                if let Some((stream, overlap)) = before {
                    self.stream = stream;
                    self.overlap = overlap;
                }
                return Err(error);
            }
        };
        self.output(samples, packet)
    }

    /// The last block's right half is never output (section 4.3.8): there
    /// is nothing to flush.
    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.overlap = Overlap::default();
        self.pending = Pending::default();
    }
}
