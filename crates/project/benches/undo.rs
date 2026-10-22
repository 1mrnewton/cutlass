//! Time and bytes for one undo step on a cut of 1,000 clips.
//!
//! Undo stores the whole project. The timed call is one rename, so the cost
//! is that copy. Run with `cargo bench -p cutlass-project --bench undo`.

#![expect(
    clippy::print_stdout,
    reason = "the timings and the byte counts are the program's output"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};
use std::time::{Duration, Instant};

use cutlass_core::Timestamp;
use cutlass_project::{
    Asset, Channels, Command, History, Outcome, Project, ProjectId, Segment, Speed, TimeRange,
    timeline_duration,
};

const CLIPS: usize = 1_000;
const ROUNDS: usize = 5;

/// Bytes requested from the system allocator and not yet returned.
static BYTES: AtomicIsize = AtomicIsize::new(0);

struct CountingAlloc;

// SAFETY: every method forwards the pointer and layout to the system allocator
// and adjusts the counter only after that call succeeds.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller of `alloc` upholds `GlobalAlloc`'s layout contract.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            add(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        sub(layout.size());
        // SAFETY: `ptr` came from this allocator with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller of `alloc_zeroed` upholds `GlobalAlloc`'s layout contract.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            add(layout.size());
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr` came from this allocator with `layout`, and `new_size` is the requested size.
        let ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !ptr.is_null() {
            add_delta(new_size, layout.size());
        }
        ptr
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

fn main() {
    let project = cut();
    let snapshot = project.to_json().expect("the cut saves").len();
    check_undo(&project);

    let mut times = Vec::with_capacity(ROUNDS);
    let mut retained = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let mut project = project.clone();
        let mut history = History::new();
        let before = BYTES.load(Ordering::Relaxed);
        let started = Instant::now();
        history
            .apply(&mut project, &[Command::Rename(String::from("renamed"))])
            .expect("rename");
        let elapsed = started.elapsed();
        std::hint::black_box(&project);
        std::hint::black_box(&history);
        let extra = BYTES.load(Ordering::Relaxed) - before;
        assert!(extra > 0, "a step keeps a second project");
        times.push(elapsed);
        retained.push(extra);
    }

    times.sort();
    retained.sort();
    let median = times[ROUNDS / 2];
    let bytes = retained[ROUNDS / 2];
    println!("1,000 clips on the main track");
    println!("one rename");
    println!(
        "median {} ms over {ROUNDS} rounds (min {} ms, max {} ms)",
        millis(median),
        millis(times[0]),
        millis(times[ROUNDS - 1])
    );
    println!("extra {bytes} bytes still held");
    println!("stored project {snapshot} JSON bytes");
}

fn cut() -> Project {
    let mut project = Project::new(ProjectId::from_bytes([1; 16]), "cut");
    let mut history = History::new();
    let added = history
        .apply(
            &mut project,
            &[
                Command::SetMagnet(false),
                Command::AddAsset(
                    Asset::video(
                        "file:///clip.mp4",
                        Timestamp::new(30_000, std::num::NonZeroU32::new(30_000).expect("scale")),
                        320,
                        180,
                        true,
                    )
                    .expect("video"),
                ),
            ],
        )
        .expect("add the file");
    let asset_id = match added.as_slice() {
        [Outcome::Unchanged, Outcome::Asset(id)] => *id,
        _ => panic!("the second outcome is the asset"),
    };
    drop(history);

    let source = {
        let asset = project.asset(asset_id).expect("the asset is stored");
        let duration = asset.duration().expect("a video has a duration");
        TimeRange::new(
            Timestamp::new(0, duration.timescale),
            Timestamp::new(duration.value, duration.timescale),
        )
        .expect("the source is the whole file")
    };
    let span = timeline_duration(source.duration(), Speed::ONE).expect("one second fits");
    let main = project.timeline().main_track().id();
    let mut places = Vec::with_capacity(CLIPS);
    for index in 0..CLIPS {
        let asset = project.asset(asset_id).expect("the asset is stored");
        let index = i64::try_from(index).expect("the clip index fits");
        let start = index.checked_mul(span.value).expect("the cut fits");
        let segment =
            Segment::media(asset, source, start, Channels::Both, Speed::ONE).expect("clip");
        places.push(Command::Place {
            track: Some(main),
            segment,
        });
    }

    let mut history = History::new();
    history
        .apply(&mut project, &places)
        .expect("place the clips");
    drop(history);
    assert_eq!(project.timeline().main_track().segments().len(), CLIPS);
    project
}

fn check_undo(base: &Project) {
    let mut project = base.clone();
    let mut history = History::new();
    history
        .apply(&mut project, &[Command::Rename(String::from("renamed"))])
        .expect("rename");
    assert_eq!(project.name(), "renamed");
    history.undo(&mut project).expect("undo");
    assert_eq!(project.name(), "cut");
    assert_eq!(project.timeline().main_track().segments().len(), CLIPS);
}

fn millis(duration: Duration) -> String {
    format!("{:.3}", duration.as_secs_f64() * 1_000.0)
}

fn add(size: usize) {
    BYTES.fetch_add(as_isize(size), Ordering::Relaxed);
}

fn sub(size: usize) {
    BYTES.fetch_sub(as_isize(size), Ordering::Relaxed);
}

fn add_delta(new_size: usize, old_size: usize) {
    BYTES.fetch_add(as_isize(new_size) - as_isize(old_size), Ordering::Relaxed);
}

fn as_isize(size: usize) -> isize {
    isize::try_from(size).expect("an allocation fits in isize")
}
