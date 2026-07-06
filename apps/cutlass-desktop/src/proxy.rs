//! Background preview-proxy generation (CapCut-style).
//!
//! Scrubbing long-GOP 4K H.264 is decode-bound: every seek pays a
//! keyframe-prefix walk at native resolution, so even a perfect compositor
//! stalls behind the decoder. The durable fix is a *proxy*: a small
//! (long side ≤ 960 px), short-GOP H.264 copy of each large source that the
//! preview decodes instead of the original. This worker generates them:
//!
//! * The preview worker requests a proxy for every video source that enters
//!   the pool (import, open, relink). Sources whose long side is 1920 px or
//!   less are skipped — they decode acceptably as-is.
//! * Jobs run one at a time on this thread, and the encode loop holds
//!   between frames while the user scrubs or plays ([`InteractionGate`]):
//!   proxy decode + composite must never compete with the preview for the
//!   decode engine or the GPU.
//! * Output lands in `<os-data>/Cutlass/proxies/<key>.mp4`, keyed by the
//!   source's path, size, and mtime — a moved or re-encoded source gets a
//!   fresh proxy while a re-import reuses the existing file instantly.
//!   Encodes write a `.part.mp4` sibling and rename on success, so a crash
//!   can't leave a truncated proxy that a later session would trust.
//! * Availability (freshly encoded or found on disk) is reported through
//!   `on_ready` with the source path the job was keyed to; the preview
//!   worker validates that the pool entry still names that source before
//!   consuming it (media ids persist in project files, so an id alone goes
//!   stale across session swaps and relinks).
//!
//! The job itself is the ordinary export path over a single-clip scratch
//! project (the strips/thumbnails pattern): probe → project → composite
//! every frame → native H.264 encoder with a short keyframe interval.
//! Rotation bakes upright through the render, and proxies carry no audio —
//! waveforms and playback always read the original.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Sender, unbounded};
use cutlass_models::{MediaSource, Project, RationalTime, TimeRange, TrackKind};
use cutlass_render::{ExportSettings, Renderer};
use tracing::{error, info};

use crate::interaction::InteractionGate;

/// Sources at or under this long side skip proxying: their decode cost is
/// tolerable, and a proxy would spend disk + encode time to save little.
const SOURCE_LONG_SIDE_MAX: u32 = 1920;

/// Proxy long side. 960 keeps 16:9 sources at 540p — cheap to decode at
/// preview sizes without visibly degrading a fit-scaled panel.
const PROXY_LONG_SIDE: u32 = 960;

/// Proxy GOP cap in frames: a scrub seek re-decodes at most this many
/// frames to reach any target (versus hundreds on ~8 s delivery GOPs).
const PROXY_KEYFRAME_INTERVAL: u32 = 15;

/// Poll cadence while the encode loop is paused on the interaction gate.
const GATE_POLL: Duration = Duration::from_millis(100);

enum ProxyMsg {
    Request {
        media_id: u64,
        path: PathBuf,
        width: u32,
        height: u32,
    },
}

/// Cheap, cloneable sender to the proxy thread.
#[derive(Clone)]
pub struct ProxyHandle {
    tx: Sender<ProxyMsg>,
}

impl ProxyHandle {
    /// Queue a proxy job for a pool video source (`width`×`height` is its
    /// display size, used for the too-small-to-bother check).
    pub fn request(&self, media_id: u64, path: PathBuf, width: u32, height: u32) {
        let _ = self.tx.send(ProxyMsg::Request {
            media_id,
            path,
            width,
            height,
        });
    }
}

/// Spawn the proxy worker thread. `on_ready(media_id, source_path,
/// proxy_path)` fires on the worker thread whenever a proxy becomes
/// available for a requested source. The thread is detached: it exits when
/// every [`ProxyHandle`] is dropped (after finishing the job in flight).
pub fn spawn(
    gate: Arc<InteractionGate>,
    on_ready: impl Fn(u64, PathBuf, PathBuf) + Send + 'static,
) -> Result<ProxyHandle, String> {
    let (tx, rx) = unbounded::<ProxyMsg>();
    std::thread::Builder::new()
        .name("cutlass-proxy".into())
        .spawn(move || {
            // A crash mid-encode leaves a `.part.mp4` behind; sweep the
            // stragglers once per session so they can't accumulate.
            sweep_partials();
            while let Ok(ProxyMsg::Request {
                media_id,
                path,
                width,
                height,
            }) = rx.recv()
            {
                if width.max(height) <= SOURCE_LONG_SIDE_MAX {
                    continue;
                }
                // Keyed by (path, size, mtime); `None` means the source
                // can't be stat'ed (vanished since import) — nothing to do,
                // a relink will re-request.
                let Some(out) = proxy_output_path(&path) else {
                    continue;
                };
                if out.exists() {
                    info!(media_id, proxy = %out.display(), "proxy already on disk");
                    on_ready(media_id, path, out);
                    continue;
                }
                info!(
                    media_id,
                    src = %path.display(),
                    width,
                    height,
                    "proxy generation started"
                );
                let started = Instant::now();
                match generate(&path, &out, &gate) {
                    Ok(frames) => {
                        info!(
                            media_id,
                            frames,
                            elapsed_s = %format_args!("{:.1}", started.elapsed().as_secs_f64()),
                            proxy = %out.display(),
                            "proxy generated"
                        );
                        on_ready(media_id, path, out);
                    }
                    Err(e) => {
                        error!(media_id, src = %path.display(), "proxy generation failed: {e}")
                    }
                }
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(ProxyHandle { tx })
}

/// Composite every frame of `src` at proxy resolution and encode it to
/// `out` (H.264, short GOP, video-only), holding between frames while the
/// user interacts. Returns the frame count.
fn generate(src: &Path, out: &Path, gate: &InteractionGate) -> Result<u64, String> {
    let probe = cutlass_decoder::probe(src).map_err(|e| e.to_string())?;
    if probe.is_image {
        return Err("still image needs no proxy".into());
    }

    // Single-clip scratch project (the strips/thumbnails pattern). Audio is
    // deliberately absent — proxies are video-only, so the export mixes no
    // audio and the encoder writes no audio track. The last container frame
    // is dropped: containers routinely over-report by one (duration
    // rounding, NTSC rates), the decoder then EOFs mid-export, and the
    // whole job dies for a frame the preview clamps away anyway (the strip
    // sampler's `duration - 1` convention).
    let frames = (probe.frame_count - 1).max(1);
    let source = MediaSource::new(
        src,
        probe.width,
        probe.height,
        probe.frame_rate,
        frames,
        false,
    );
    let rate = source.frame_rate;
    let mut project = Project::new("proxy", rate);
    let media = project.add_media(source);
    let track = project.add_track(TrackKind::Video, "Media");
    project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, frames, rate),
            RationalTime::new(0, rate),
        )
        .map_err(|e| e.to_string())?;

    let settings = ExportSettings {
        size: proxy_size(probe.width, probe.height),
        frame_rate: rate,
    }
    .evened();
    let config = cutlass_render::export_config_with(&project, settings)
        .with_keyframe_interval(PROXY_KEYFRAME_INTERVAL);

    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    // Encode into a `.part.mp4` sibling and rename on success: a crash or
    // failure never leaves a truncated file under the trusted name. The
    // temp name keeps the `.mp4` extension — the sink writer infers the
    // container from it.
    let tmp = out.with_extension("part.mp4");
    let mut encoder = cutlass_encoder::open_encoder(&tmp, config).map_err(|e| e.to_string())?;
    // Fresh renderer per job (own GPU queue + decoder cache): scratch
    // projects mint new media ids, so a shared renderer would accumulate
    // dead decoders across jobs.
    let mut renderer = Renderer::new_headless().map_err(|e| e.to_string())?;
    let result = cutlass_render::export_observed(
        &mut renderer,
        &project,
        encoder.as_mut(),
        settings,
        &mut |_, _| {
            // The gate closes while the user scrubs or plays: hold between
            // frames so proxy decode/composite never competes for the
            // decode engine or the GPU.
            while gate.busy() {
                std::thread::sleep(GATE_POLL);
            }
            true
        },
    );
    let frames = match result {
        Ok(frames) => frames,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e.to_string());
        }
    };
    std::fs::rename(&tmp, out).map_err(|e| e.to_string())?;
    Ok(frames)
}

/// Delete `.part.mp4` leftovers from encodes a previous session never
/// finished. Best-effort: a missing proxies dir (first run) is normal, and
/// a locked file just stays until the next sweep.
fn sweep_partials() {
    let dir = crate::paths::data_dir().join("proxies");
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.to_string_lossy().ends_with(".part.mp4") {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Uniform-scale `w`×`h` so the long side lands at [`PROXY_LONG_SIDE`].
/// Never upscales; `ExportSettings::evened` downstream rounds for H.264.
fn proxy_size(w: u32, h: u32) -> (u32, u32) {
    let long = w.max(h).max(1);
    if long <= PROXY_LONG_SIDE {
        return (w.max(1), h.max(1));
    }
    let scale = f64::from(PROXY_LONG_SIDE) / f64::from(long);
    let side = |v: u32| ((f64::from(v) * scale).round() as u32).max(1);
    (side(w), side(h))
}

/// `<os-data>/Cutlass/proxies/<key>.mp4` for the source at `path`, keyed by
/// its path, size, and mtime — any change to the source re-keys (and thus
/// regenerates) its proxy. `None` when the source can't be stat'ed.
fn proxy_output_path(path: &Path) -> Option<PathBuf> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos() as u64);
    let mut key = fnv1a(FNV_OFFSET, path.to_string_lossy().as_bytes());
    key = fnv1a(key, &meta.len().to_le_bytes());
    key = fnv1a(key, &mtime_ns.to_le_bytes());
    Some(
        crate::paths::data_dir()
            .join("proxies")
            .join(format!("{key:016x}.mp4")),
    )
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a folded over `bytes`, continuing from `hash`. Hand-rolled on
/// purpose: proxy file names must stay stable across app releases, and the
/// std hasher's output is not guaranteed to.
fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_size_downscales_long_side_and_never_upscales() {
        assert_eq!(proxy_size(3840, 2160), (960, 540));
        assert_eq!(proxy_size(2160, 3840), (540, 960));
        // DCI 4K: the short side rounds to nearest (evened downstream).
        assert_eq!(proxy_size(4096, 2160), (960, 506));
        // At or under the proxy bound nothing scales (unreachable in
        // practice — the worker skips sources ≤1920 — but the math must
        // never upscale).
        assert_eq!(proxy_size(854, 480), (854, 480));
        assert_eq!(proxy_size(0, 0), (1, 1));
    }

    /// End-to-end over the real 4K fixture: generate → probe → seek-cost
    /// comparison. Ignored by default — it decodes and re-encodes the whole
    /// clip (~1 min). Run when touching the proxy pipeline:
    /// `cargo test -p cutlass-desktop -- --ignored proxy_e2e`.
    #[test]
    #[ignore = "decodes + re-encodes the whole 4K fixture; run manually"]
    fn proxy_e2e_generates_a_seekable_short_gop_proxy_from_the_4k_fixture() {
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../local-assets/16265742_3840_2160_30fps.mp4");
        if !src.exists() {
            eprintln!("4K fixture missing; skipping");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("proxy.mp4");
        let gate = crate::interaction::InteractionGate::new();
        let frames = generate(&src, &out, &gate).expect("proxy generation");
        assert!(frames > 0, "proxy wrote no frames");

        // Long side ≤ 960, same cadence, and roughly the same length (the
        // encoder may trim a trailing frame while finalizing).
        let src_probe = cutlass_decoder::probe(&src).expect("probe source");
        let probe = cutlass_decoder::probe(&out).expect("probe proxy");
        assert_eq!((probe.width, probe.height), (960, 540));
        assert_eq!(probe.frame_rate, src_probe.frame_rate);
        assert!(
            (probe.frame_count - frames as i64).abs() <= 2,
            "proxy holds {} frames, export wrote {frames}",
            probe.frame_count
        );

        // The reason proxies exist: exact mid-GOP seeks collapse from a
        // GOP-prefix walk (~8 s GOPs on this fixture) to at most a
        // 15-frame roll. Same seek pattern on both files, medians compared.
        let median_exact_seek_ms = |path: &Path| -> f64 {
            let mut dec =
                cutlass_decoder::open_video_decoder(path, cutlass_decoder::OutputMode::Cpu)
                    .expect("open");
            let n = probe.frame_count.min(src_probe.frame_count).max(1);
            let mut samples: Vec<f64> = (1..=8i64)
                .map(|i| {
                    // Descending mid-GOP targets: every one seeks backward.
                    let frame = n * (9 - i) / 10 + 7;
                    let started = std::time::Instant::now();
                    let got = dec
                        .frame_at(RationalTime::new(frame.min(n - 1), probe.frame_rate))
                        .expect("frame_at");
                    assert!(got.is_some(), "no frame at {frame}");
                    started.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        let original = median_exact_seek_ms(&src);
        let proxied = median_exact_seek_ms(&out);
        eprintln!("median exact backward seek: original {original:.1} ms, proxy {proxied:.1} ms");
        assert!(
            proxied < original,
            "proxy seeks ({proxied:.1} ms) should beat the original ({original:.1} ms)"
        );
    }

    #[test]
    fn proxy_names_key_on_path_size_and_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.mp4");
        let b = dir.path().join("b.mp4");
        std::fs::write(&a, b"aaaa").unwrap();
        std::fs::write(&b, b"aaaa").unwrap();

        let name = |p: &Path| {
            proxy_output_path(p)
                .expect("stat succeeds")
                .file_name()
                .expect("proxy paths end in a file name")
                .to_owned()
        };
        // Stable for an unchanged source, distinct per path.
        assert_eq!(name(&a), name(&a));
        assert_ne!(name(&a), name(&b));

        // Changing the file (size here) re-keys the proxy.
        let before = name(&a);
        std::fs::write(&a, b"aaaaaaaa").unwrap();
        assert_ne!(before, name(&a));

        // Missing sources produce no key at all.
        assert!(proxy_output_path(&dir.path().join("gone.mp4")).is_none());
    }
}
