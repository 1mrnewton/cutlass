//! Silence handed to the output device comes back as enqueued frames.

#![cfg(target_vendor = "apple")]

use std::num::{NonZeroU32, NonZeroUsize};
use std::thread;
use std::time::{Duration, Instant};

use cutlass_core::{AudioSamples, Timestamp, VideoFrame};
use cutlass_output::Output;
use cutlass_playback::Queue;

#[test]
fn silence_is_handed_to_the_device() {
    let Some(mut output) = open() else {
        return;
    };
    let frames = 4_800usize;
    let samples = AudioSamples::new(48_000, 2, vec![0.0; frames * 2]).expect("silence");
    assert!(output.clock().expect("clock").is_none());
    output.play(&samples).expect("play");
    let mut clock = None;
    let deadline = Instant::now() + Duration::from_secs(2);
    while output.enqueued_frames() < frames {
        if let Some(time) = output.clock().expect("clock") {
            clock = Some(time);
        }
        if Instant::now() > deadline {
            panic!("enqueued {} of {frames} frames", output.enqueued_frames());
        }
        thread::sleep(Duration::from_millis(20));
    }
    let time = clock.expect("sample time");
    assert_eq!(time.timescale.get(), 48_000);
    assert!(time.value >= 0);

    let video = NonZeroU32::new(15_360).expect("timescale");
    let mut pictures = Queue::new(NonZeroUsize::new(2).expect("limit"));
    pictures
        .push(VideoFrame {
            pts: Timestamp::new(0, video),
            duration: None,
            image: 0,
        })
        .expect("push");
    pictures
        .push(VideoFrame {
            pts: Timestamp::new(15_360, video),
            duration: None,
            image: 1,
        })
        .expect("push");
    let shown = pictures.at(time).expect("picture");
    assert_eq!(shown.image, 0);
}

#[test]
fn a_mismatched_stream_is_rejected() {
    let Some(mut output) = open() else {
        return;
    };
    let samples = AudioSamples::new(44_100, 2, vec![0.0; 4]).expect("samples");
    let error = match output.play(&samples) {
        Ok(()) => panic!("a 44.1 kHz buffer should be rejected"),
        Err(error) => error,
    };
    assert!(!error.is_unsupported());
}

#[expect(
    clippy::print_stderr,
    reason = "a missing output device is reported instead of a silent pass"
)]
fn report_missing_device() {
    eprintln!("skipping: no audio output device");
}

fn open() -> Option<Output> {
    match Output::new(48_000, 2) {
        Ok(output) => Some(output),
        Err(error) if error.is_unsupported() => {
            report_missing_device();
            None
        }
        Err(error) => panic!("opening audio output failed: {error}"),
    }
}
