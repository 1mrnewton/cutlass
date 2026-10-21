//! A real H.264 stream fills the picture queue and follows the playhead.

#![cfg(target_vendor = "apple")]

use std::num::NonZeroUsize;

use cutlass_core::{Packet, VideoFrame};
use cutlass_media::{Error, VideoSession};
use cutlass_playback::{Ahead, Decode};
use cutlass_test_fixtures::video;
use cutlass_videotoolbox::PixelBuffer;

struct Session(VideoSession);

impl Decode for Session {
    type Image = PixelBuffer;
    type Error = Error;

    fn send(&mut self, packet: &Packet) -> Result<(), Self::Error> {
        self.0.send_packet(packet)
    }

    fn receive(&mut self) -> Result<Option<VideoFrame<PixelBuffer>>, Self::Error> {
        self.0.receive_frame()
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        self.0.flush()
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.0.reset()
    }
}

#[test]
fn a_real_stream_stays_a_few_frames_ahead_and_drops_what_the_playhead_passed() {
    let fixture = video("h264-high-bframes-128x72");
    let mut session = Session(VideoSession::open(&fixture.config).expect("decoder"));
    let mut ahead = Ahead::new(NonZeroUsize::new(3).expect("limit"));
    let mut pending = fixture.packets.as_slice();

    ahead.fill(&mut session, &mut pending).expect("fill");
    assert_eq!(ahead.len(), 3);
    assert!(
        !pending.is_empty(),
        "a full queue must leave packets for later"
    );

    for expected in &fixture.frames {
        let shown = ahead.at(expected.pts).expect("a frame at this time");
        assert_eq!(shown.pts, expected.pts);
        ahead.fill(&mut session, &mut pending).expect("refill");
    }
    assert!(pending.is_empty());
}

#[test]
fn seek_shows_the_frame_from_the_keyframe_it_depends_on() {
    let fixture = video("h264-high-bframes-128x72");
    let mut session = Session(VideoSession::open(&fixture.config).expect("decoder"));
    let mut ahead = Ahead::new(NonZeroUsize::new(3).expect("limit"));

    let in_the_first_group = fixture.frames[4].pts;
    ahead
        .seek_to(&mut session, &fixture.packets, in_the_first_group)
        .expect("seek");
    assert_eq!(
        ahead.at(in_the_first_group).expect("frame").pts,
        in_the_first_group
    );

    let in_the_second_group = fixture.frames[8].pts;
    ahead
        .seek_to(&mut session, &fixture.packets, in_the_second_group)
        .expect("seek");
    assert_eq!(
        ahead.at(in_the_second_group).expect("frame").pts,
        in_the_second_group
    );
}
