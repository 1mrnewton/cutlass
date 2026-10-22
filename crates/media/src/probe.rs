//! A summary of an MP4 or MOV file. Samples stay on the parsed movie.

use std::num::NonZeroU32;

use cutlass_core::{
    AudioStreamConfig, ColorTags, ContentLight, MasteringDisplay, Packet, VideoStreamConfig,
};
use cutlass_mp4::{Fourcc, Kind, Movie, Rotation, Track};

use crate::{DecoderKind, Error, VideoSession, choose_video_decoder};

/// A parsed file plus the facts a caller needs before opening a decoder.
pub struct Probe {
    movie: Movie,
    tracks: Vec<TrackInfo>,
}

impl Probe {
    /// Reads MP4 or MOV bytes. The same reader handles both.
    pub fn parse(bytes: &[u8]) -> Result<Self, Error> {
        let movie = Movie::parse(bytes)?;
        let tracks = movie.tracks().iter().map(TrackInfo::from_track).collect();
        Ok(Self { movie, tracks })
    }

    /// Timescale of the movie header.
    pub fn timescale(&self) -> NonZeroU32 {
        self.movie.timescale()
    }

    /// Duration from the movie header, in [`Self::timescale`] units.
    pub fn duration(&self) -> u64 {
        self.movie.duration()
    }

    /// Tracks in the order the file stores them.
    pub fn tracks(&self) -> &[TrackInfo] {
        &self.tracks
    }

    /// Samples for one track, in decode order.
    ///
    /// Times are media time when the track has no edit list. An edit list
    /// moves them onto the movie timeline.
    pub fn samples(&self, index: usize) -> Option<&[Packet]> {
        self.movie.tracks().get(index).map(Track::samples)
    }

    /// Opens the video track at `index` when [`choose_video_decoder`] picks the system codec.
    pub fn open_video(&self, index: usize) -> Result<VideoSession, Error> {
        let Some(track) = self.movie.tracks().get(index) else {
            return Err(Error::NoTrack(index));
        };
        let Kind::Video(config) = track.kind() else {
            return Err(Error::NotVideo(track.id()));
        };
        match choose_video_decoder(config.codec)? {
            DecoderKind::System => VideoSession::open(
                config,
                track.color(),
                track.mastering(),
                track.content_light(),
                track.rotation(),
            ),
            // No Rust video decoder is connected, so the picker does not return
            // this yet. Open that decoder here when one is.
            DecoderKind::Software => Err(Error::NoDecoder(config.codec)),
        }
    }
}

/// One track, without its samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackInfo {
    pub id: u32,
    pub enabled: bool,
    pub timescale: NonZeroU32,
    /// Track duration, in [`Self::timescale`] units.
    pub duration: u64,
    pub rotation: Rotation,
    /// Display size from the track header.
    pub display_width: u32,
    pub display_height: u32,
    pub timing: FrameTiming,
    pub content: Content,
}

impl TrackInfo {
    /// The display size, or the encoded size turned by the track matrix when that size is missing.
    pub fn picture_size(&self) -> Option<(u32, u32)> {
        if self.display_width > 0 && self.display_height > 0 {
            return Some((self.display_width, self.display_height));
        }
        let Content::Video { config, .. } = &self.content else {
            return None;
        };
        if config.width == 0 || config.height == 0 {
            return None;
        }
        Some(match self.rotation {
            Rotation::Clockwise90 | Rotation::Clockwise270 => (config.height, config.width),
            Rotation::Zero | Rotation::Clockwise180 | Rotation::Other => {
                (config.width, config.height)
            }
        })
    }

    /// `Some(true)` for PQ or HLG, `Some(false)` for another named transfer.
    /// `None` when the file has no colour tags, which is not the same as SDR.
    pub fn hdr(&self) -> Option<bool> {
        match &self.content {
            Content::Video { color, .. } => color.map(ColorTags::is_hdr),
            Content::Audio(_) | Content::Other(_) => None,
        }
    }

    fn from_track(track: &Track) -> Self {
        Self {
            id: track.id(),
            enabled: track.enabled(),
            timescale: track.timescale(),
            duration: track.duration(),
            rotation: track.rotation(),
            display_width: track.display_width(),
            display_height: track.display_height(),
            timing: timing(track),
            content: content(track),
        }
    }
}

/// How the sample durations are spaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameTiming {
    /// Every sample lasts this many track-timescale units.
    Constant(u32),
    /// The samples do not share one duration.
    Variable,
    /// The track has no samples.
    Empty,
}

/// What the sample entry says the track is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Content {
    Video {
        /// Encoded size and codec configuration. This size is not the display size.
        config: VideoStreamConfig,
        color: Option<ColorTags>,
        /// Mastering display from an `mdcv` box. `None` when the file has none.
        mastering: Option<MasteringDisplay>,
        /// Content light level from a `clli` box. `None` when the file has none.
        content_light: Option<ContentLight>,
    },
    /// A named audio codec and its stream config.
    Audio(AudioStreamConfig),
    /// A track this reader does not demux, such as timecode.
    Other(Fourcc),
}

fn timing(track: &Track) -> FrameTiming {
    if track.samples().is_empty() {
        FrameTiming::Empty
    } else if let Some(duration) = track.frame_duration() {
        FrameTiming::Constant(duration)
    } else {
        FrameTiming::Variable
    }
}

fn content(track: &Track) -> Content {
    match track.kind() {
        Kind::Video(config) => Content::Video {
            config: config.clone(),
            color: track.color(),
            mastering: track.mastering(),
            content_light: track.content_light(),
        },
        Kind::Audio(audio) => match audio.codec() {
            Some(codec) => Content::Audio(AudioStreamConfig {
                codec,
                sample_rate: audio.sample_rate(),
                channels: audio.channels(),
                config_record: audio.config_record().to_vec(),
            }),
            None => Content::Other(audio.sample_entry()),
        },
        Kind::Other(fourcc) => Content::Other(*fourcc),
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cutlass_core::{ColorTags, VideoCodec, VideoStreamConfig};
    use cutlass_mp4::Rotation;

    use super::*;

    #[test]
    fn missing_tags_are_not_reported_as_sdr() {
        assert_eq!(video_info(None).hdr(), None);
        assert_eq!(
            video_info(Some(ColorTags {
                primaries: 1,
                transfer: 1,
                matrix: 1,
                full_range: Some(false),
            }))
            .hdr(),
            Some(false)
        );
        assert_eq!(
            video_info(Some(ColorTags {
                primaries: 9,
                transfer: 16,
                matrix: 9,
                full_range: Some(false),
            }))
            .hdr(),
            Some(true)
        );
    }

    #[test]
    fn the_display_size_wins_and_a_missing_one_follows_the_rotation() {
        assert_eq!(video_info(None).picture_size(), Some((1, 1)));
        let mut turned = video_info(None);
        turned.display_width = 0;
        turned.display_height = 0;
        turned.rotation = Rotation::Clockwise90;
        turned.content = Content::Video {
            config: VideoStreamConfig {
                codec: VideoCodec::H264,
                width: 320,
                height: 180,
                config_record: Vec::new(),
            },
            color: None,
            mastering: None,
            content_light: None,
        };
        assert_eq!(turned.picture_size(), Some((180, 320)));
    }

    fn video_info(color: Option<ColorTags>) -> TrackInfo {
        TrackInfo {
            id: 1,
            enabled: true,
            timescale: NonZeroU32::new(1).expect("one is not zero"),
            duration: 0,
            rotation: Rotation::Zero,
            display_width: 1,
            display_height: 1,
            timing: FrameTiming::Empty,
            content: Content::Video {
                config: VideoStreamConfig {
                    codec: VideoCodec::H264,
                    width: 1,
                    height: 1,
                    config_record: Vec::new(),
                },
                color,
                mastering: None,
                content_light: None,
            },
        }
    }
}
