//! Software stand-ins for effect packages.
//!
//! There is no script host and no shader dialect. The software stand-in is
//! what [`PictureSource::sample`] draws. Bytes are not premultiplied before
//! the blend. Alpha uses `output = source + destination * (1 - source.alpha)`
//! on a channel: the 8-bit alpha is divided by 255, the remainder is
//! truncated, and a channel above 255 is clamped.
//!
//! A missing package is [`Sample::Miss`], not a blank ready picture. Sampling
//! does not write a draft.

use cutlass_model::{AnimType, Draft, MaterialEffect, Micros, Segment, TimeRange};
use cutlass_picture::{CpuBytes, Picture, PictureSource, Sample, rgba_len};

use crate::catalog::{self, BlendMode, Distort, PackageKind};
use crate::package::{self, Package, PackageError, PackageStore};

/// Contrast stand-in, centered on 128. The effect value is not read.
const CONTRAST_FACTOR: i32 = 2;

/// In, out, or loop window used while sampling.
///
/// The span is a parameter. Sampling does not write it back to a draft.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnimSpan {
    /// Which curve to evaluate.
    pub kind: AnimType,
    /// First microsecond of the window.
    pub start: Micros,
    /// Length of the window. Not written onto a draft.
    pub duration: Micros,
}

/// A fraction `numer / denom`.
///
/// `denom` is never 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    /// Numerator. `0` is the start of `In` and `Out`.
    pub numer: u64,
    /// Denominator. Never 0.
    pub denom: u64,
}

#[derive(Clone, Copy)]
enum Window {
    Linear,
    Loop,
}

#[derive(Clone, Copy)]
enum Scale {
    Keep,
    Mul { numer: u64, denom: u64 },
}

#[derive(Clone, Copy)]
enum Paint {
    Keep,
    Invert,
    Contrast,
    Pixelate,
}

/// 0 at the window start and 1 at the window end, as numer/denom.
/// `In`, `Out`, and `Loop` only. Other [`AnimType`] values return `None`.
/// `Loop` repeats: numer is `elapsed % duration`, and an exact multiple is 0.
/// This does not write a draft.
#[must_use]
pub fn progress(kind: AnimType, start: Micros, duration: Micros, time: Micros) -> Option<Progress> {
    let window = match kind {
        AnimType::In | AnimType::Out => Window::Linear,
        AnimType::Loop => Window::Loop,
        AnimType::Caption | AnimType::Group | AnimType::None => return None,
    };
    if duration.0 <= 0 {
        return Some(Progress { numer: 0, denom: 1 });
    }
    let elapsed = time.0.checked_sub(start.0)?;
    start.0.checked_add(duration.0)?;
    let denom = u64::try_from(duration.0).ok()?;
    if denom == 0 {
        return None;
    }
    if elapsed < 0 {
        return Some(Progress { numer: 0, denom });
    }
    let elapsed = u64::try_from(elapsed).ok()?;
    let numer = match window {
        Window::Loop => elapsed % denom,
        Window::Linear if elapsed >= denom => denom,
        Window::Linear => elapsed,
    };
    Some(Progress { numer, denom })
}

/// A package sampled as CPU pixels.
///
/// `sample` is [`Sample::Miss`] outside the half-open range
/// `[start, start + duration)`, when the package or its stand-in is absent,
/// and when a stand-in does not have the CPU bytes it needs. Two samples at
/// the same time return the same pixels.
#[derive(Debug)]
pub struct EffectSource {
    range: TimeRange,
    package_id: Option<String>,
    kind: Option<PackageKind>,
    input: Option<CpuBytes>,
    backdrop: Option<CpuBytes>,
    anim: Option<AnimSpan>,
}

impl EffectSource {
    /// Load `effect` and keep CPU copies of the pictures.
    ///
    /// [`PackageError::Missing`] becomes an absent source: [`Self::package_id`]
    /// is `None` and every sample is [`Sample::Miss`]. Any other load error is
    /// returned. A blend with no backdrop is [`PackageError::Backdrop`]. CPU
    /// pictures whose width, height, or byte length differ are
    /// [`PackageError::Pixels`]. A platform buffer is not copied; sampling it
    /// is a miss. `effect.value` is not read, and `effect` is not changed.
    pub fn open<S: PackageStore>(
        store: &S,
        effect: &MaterialEffect,
        input: &Picture,
        backdrop: Option<&Picture>,
        range: TimeRange,
        anim: Option<AnimSpan>,
    ) -> Result<Self, PackageError> {
        match package::load(store, effect) {
            Ok(package) => Self::from_package(package, input, backdrop, range, anim),
            Err(PackageError::Missing { .. }) => Ok(Self::absent(range)),
            Err(error) => Err(error),
        }
    }

    /// Load the segment's package and sample across `segment.target`.
    ///
    /// The draft and the segment are not changed. A segment with no package,
    /// or a package the store does not have, is an absent source.
    pub fn open_segment<S: PackageStore>(
        store: &S,
        draft: &Draft,
        segment: &Segment,
        input: &Picture,
        backdrop: Option<&Picture>,
        anim: Option<AnimSpan>,
    ) -> Result<Self, PackageError> {
        let range = segment.target.clone();
        match package::load_segment(store, draft, segment) {
            Ok(package) => Self::from_package(package, input, backdrop, range, anim),
            Err(PackageError::Missing { .. }) => Ok(Self::absent(range)),
            Err(error) => Err(error),
        }
    }

    /// A source whose every sample is [`Sample::Miss`].
    #[must_use]
    pub fn absent(range: TimeRange) -> Self {
        Self {
            range,
            package_id: None,
            kind: None,
            input: None,
            backdrop: None,
            anim: None,
        }
    }

    /// The loaded package id, or `None` when the package is absent.
    #[must_use]
    pub fn package_id(&self) -> Option<&str> {
        self.package_id.as_deref()
    }

    fn from_package(
        package: Package,
        input: &Picture,
        backdrop: Option<&Picture>,
        range: TimeRange,
        anim: Option<AnimSpan>,
    ) -> Result<Self, PackageError> {
        let id = package.id().to_owned();
        let kind = catalog::entry(&id).map(|row| row.kind);
        let (stored_input, stored_backdrop) = match kind {
            Some(found) => retain(found, &id, input, backdrop)?,
            None => (None, None),
        };
        Ok(Self {
            range,
            package_id: Some(id),
            kind,
            input: stored_input,
            backdrop: stored_backdrop,
            anim,
        })
    }

    fn scale_at(&self, time: Micros, kind: PackageKind) -> Option<Scale> {
        if let Some(span) = self.anim {
            let ratio = progress(span.kind, span.start, span.duration, time)?;
            return fade(span.kind, ratio);
        }
        match kind {
            PackageKind::TextAnim(anim) => {
                let ratio = progress(anim, self.range.start, self.range.duration, time)?;
                fade(anim, ratio)
            }
            PackageKind::Blend(_) | PackageKind::Distort(_) => Some(Scale::Keep),
        }
    }
}

impl PictureSource for EffectSource {
    fn sample(&self, time: Micros) -> Sample {
        if !contains(&self.range, time) {
            return Sample::Miss;
        }
        let Some(kind) = self.kind else {
            return Sample::Miss;
        };
        let Some(scale) = self.scale_at(time, kind) else {
            return Sample::Miss;
        };
        match kind {
            PackageKind::Blend(mode) => {
                sample_blend(self.input.as_ref(), self.backdrop.as_ref(), mode, scale)
            }
            PackageKind::Distort(distort) => {
                sample_unary(self.input.as_ref(), paint_of(distort), scale)
            }
            PackageKind::TextAnim(_) => sample_unary(self.input.as_ref(), Paint::Keep, scale),
        }
    }
}

fn contains(range: &TimeRange, time: Micros) -> bool {
    if range.duration.0 <= 0 {
        return false;
    }
    let Some(end) = range.start.0.checked_add(range.duration.0) else {
        return false;
    };
    time.0 >= range.start.0 && time.0 < end
}

fn retain(
    kind: PackageKind,
    id: &str,
    input: &Picture,
    backdrop: Option<&Picture>,
) -> Result<(Option<CpuBytes>, Option<CpuBytes>), PackageError> {
    match kind {
        PackageKind::Blend(_) => {
            let Some(backdrop) = backdrop else {
                return Err(PackageError::Backdrop { id: id.to_owned() });
            };
            let source = copied_cpu(input);
            let destination = copied_cpu(backdrop);
            if let (Some(source), Some(destination)) = (&source, &destination)
                && !pictures_match(source, destination)
            {
                return Err(PackageError::Pixels { id: id.to_owned() });
            }
            Ok((source, destination))
        }
        PackageKind::Distort(_) | PackageKind::TextAnim(_) => Ok((copied_cpu(input), None)),
    }
}

fn copied_cpu(picture: &Picture) -> Option<CpuBytes> {
    match picture {
        Picture::Cpu(cpu) => Some(cpu.clone()),
        Picture::Platform(_) => None,
    }
}

fn pictures_match(source: &CpuBytes, destination: &CpuBytes) -> bool {
    if source.width() != destination.width() || source.height() != destination.height() {
        return false;
    }
    match rgba_len(source.width(), source.height()) {
        Some(len) => source.rgba().len() == len && destination.rgba().len() == len,
        None => false,
    }
}

fn fade(kind: AnimType, ratio: Progress) -> Option<Scale> {
    if ratio.denom == 0 {
        return None;
    }
    let numer = match kind {
        AnimType::In | AnimType::Loop => ratio.numer,
        AnimType::Out => ratio.denom.checked_sub(ratio.numer)?,
        AnimType::Caption | AnimType::Group | AnimType::None => return None,
    };
    Some(Scale::Mul {
        numer,
        denom: ratio.denom,
    })
}

fn normalize(scale: Scale) -> Option<Scale> {
    match scale {
        Scale::Mul { denom: 0, .. } => None,
        other => Some(other),
    }
}

fn sample_blend(
    source: Option<&CpuBytes>,
    destination: Option<&CpuBytes>,
    mode: BlendMode,
    scale: Scale,
) -> Sample {
    let (Some(source), Some(destination)) = (source, destination) else {
        return Sample::Miss;
    };
    let Some(scale) = normalize(scale) else {
        return Sample::Miss;
    };
    let Some(rgba) = blend_bytes(source, destination, mode, scale) else {
        return Sample::Miss;
    };
    finish(source.width(), source.height(), rgba)
}

fn sample_unary(source: Option<&CpuBytes>, op: Paint, scale: Scale) -> Sample {
    let Some(source) = source else {
        return Sample::Miss;
    };
    let Some(scale) = normalize(scale) else {
        return Sample::Miss;
    };
    let Some(rgba) = unary_bytes(source, op, scale) else {
        return Sample::Miss;
    };
    finish(source.width(), source.height(), rgba)
}

fn finish(width: u32, height: u32, rgba: Vec<u8>) -> Sample {
    match CpuBytes::new(width, height, rgba) {
        Ok(bytes) => Sample::Ready(Picture::Cpu(bytes)),
        Err(_) => Sample::Miss,
    }
}

fn blend_bytes(
    source: &CpuBytes,
    destination: &CpuBytes,
    mode: BlendMode,
    scale: Scale,
) -> Option<Vec<u8>> {
    let src = grid(source)?;
    let dst = grid(destination)?;
    if src.len() != dst.len()
        || source.width() != destination.width()
        || source.height() != destination.height()
    {
        return None;
    }
    let mut rgba = source.rgba().to_vec();
    {
        let (out, rest) = rgba.as_chunks_mut::<4>();
        if !rest.is_empty() || out.len() != src.len() {
            return None;
        }
        for (index, pixel) in out.iter_mut().enumerate() {
            let source_px = apply_alpha(src.get(index).copied()?, scale);
            let dest_px = dst.get(index).copied()?;
            *pixel = blend_pixel(mode, source_px, dest_px);
        }
    }
    Some(rgba)
}

fn unary_bytes(source: &CpuBytes, op: Paint, scale: Scale) -> Option<Vec<u8>> {
    let src = grid(source)?;
    let mut rgba = source.rgba().to_vec();
    {
        let (out, rest) = rgba.as_chunks_mut::<4>();
        if !rest.is_empty() || out.len() != src.len() {
            return None;
        }
        match op {
            Paint::Pixelate => pixelate_into(out, src, source.width(), scale)?,
            Paint::Keep => fill(out, src, scale, keep_pixel)?,
            Paint::Invert => fill(out, src, scale, invert_pixel)?,
            Paint::Contrast => fill(out, src, scale, contrast_pixel)?,
        }
    }
    Some(rgba)
}

fn fill(
    out: &mut [[u8; 4]],
    src: &[[u8; 4]],
    scale: Scale,
    op: fn([u8; 4]) -> [u8; 4],
) -> Option<()> {
    for (index, pixel) in out.iter_mut().enumerate() {
        let source_px = src.get(index).copied()?;
        *pixel = op(apply_alpha(source_px, scale));
    }
    Some(())
}

fn pixelate_into(out: &mut [[u8; 4]], src: &[[u8; 4]], width: u32, scale: Scale) -> Option<()> {
    for (index, pixel) in out.iter_mut().enumerate() {
        let (x, y) = coord(index, width)?;
        let origin_x = block_origin(x)?;
        let origin_y = block_origin(y)?;
        let origin = pixel_index(width, origin_x, origin_y)?;
        let source_px = src.get(origin).copied()?;
        *pixel = apply_alpha(source_px, scale);
    }
    Some(())
}

fn grid(bytes: &CpuBytes) -> Option<&[[u8; 4]]> {
    let (chunks, rest) = bytes.rgba().as_chunks::<4>();
    if !rest.is_empty() {
        return None;
    }
    let width = usize::try_from(bytes.width()).ok()?;
    let height = usize::try_from(bytes.height()).ok()?;
    let count = width.checked_mul(height)?;
    if chunks.len() != count {
        return None;
    }
    Some(chunks)
}

fn coord(index: usize, width: u32) -> Option<(u32, u32)> {
    let width = usize::try_from(width).ok()?;
    if width == 0 {
        return None;
    }
    let x = u32::try_from(index % width).ok()?;
    let y = u32::try_from(index / width).ok()?;
    Some((x, y))
}

fn block_origin(coord: u32) -> Option<u32> {
    let quotient = coord.checked_div(catalog::PIXELATE_BLOCK)?;
    quotient.checked_mul(catalog::PIXELATE_BLOCK)
}

fn pixel_index(width: u32, x: u32, y: u32) -> Option<usize> {
    let x = usize::try_from(x).ok()?;
    let y = usize::try_from(y).ok()?;
    let width = usize::try_from(width).ok()?;
    y.checked_mul(width)?.checked_add(x)
}

fn paint_of(distort: Distort) -> Paint {
    match distort {
        Distort::Invert => Paint::Invert,
        Distort::Contrast => Paint::Contrast,
        Distort::Pixelate => Paint::Pixelate,
    }
}

fn apply_alpha(pixel: [u8; 4], scale: Scale) -> [u8; 4] {
    let [red, green, blue, alpha] = pixel;
    let alpha = match scale {
        Scale::Keep => alpha,
        Scale::Mul { numer, denom } => scale_channel(alpha, numer, denom),
    };
    [red, green, blue, alpha]
}

fn scale_channel(alpha: u8, numer: u64, denom: u64) -> u8 {
    if denom == 0 {
        return 0;
    }
    let mixed = u128::from(alpha) * u128::from(numer) / u128::from(denom);
    let mixed = mixed.min(u128::from(255_u8));
    u8::try_from(mixed).unwrap_or(255)
}

/// `output = source + destination * (1 - source.alpha)`.
///
/// The 8-bit alpha is divided by 255 and the remainder is truncated.
/// A channel above 255 is clamped. Bytes are not premultiplied first.
fn over_channel(source: u8, destination: u8, source_alpha: u8) -> u8 {
    let inverse = u16::from(255_u8.saturating_sub(source_alpha));
    let mixed = u16::from(source) + u16::from(destination) * inverse / 255;
    u8::try_from(mixed.min(255)).unwrap_or(255)
}

fn multiply_channel(source: u8, destination: u8) -> u8 {
    let mixed = u16::from(source) * u16::from(destination) / 255;
    u8::try_from(mixed).unwrap_or(255)
}

fn screen_channel(source: u8, destination: u8) -> u8 {
    let inverse_source = u16::from(255_u8.saturating_sub(source));
    let inverse_destination = u16::from(255_u8.saturating_sub(destination));
    let covered = inverse_source * inverse_destination / 255;
    u8::try_from(255_u16.saturating_sub(covered)).unwrap_or(255)
}

fn add_channel(source: u8, destination: u8) -> u8 {
    let mixed = u16::from(source) + u16::from(destination);
    u8::try_from(mixed.min(255)).unwrap_or(255)
}

fn blend_pixel(mode: BlendMode, source: [u8; 4], destination: [u8; 4]) -> [u8; 4] {
    let [sr, sg, sb, sa] = source;
    let [dr, dg, db, da] = destination;
    match mode {
        BlendMode::SourceOver => [
            over_channel(sr, dr, sa),
            over_channel(sg, dg, sa),
            over_channel(sb, db, sa),
            over_channel(sa, da, sa),
        ],
        BlendMode::Multiply => [
            multiply_channel(sr, dr),
            multiply_channel(sg, dg),
            multiply_channel(sb, db),
            over_channel(sa, da, sa),
        ],
        BlendMode::Screen => [
            screen_channel(sr, dr),
            screen_channel(sg, dg),
            screen_channel(sb, db),
            over_channel(sa, da, sa),
        ],
        BlendMode::Add => [
            add_channel(sr, dr),
            add_channel(sg, dg),
            add_channel(sb, db),
            over_channel(sa, da, sa),
        ],
        BlendMode::Difference => [
            sr.abs_diff(dr),
            sg.abs_diff(dg),
            sb.abs_diff(db),
            over_channel(sa, da, sa),
        ],
    }
}

fn keep_pixel(pixel: [u8; 4]) -> [u8; 4] {
    pixel
}

fn invert_pixel(pixel: [u8; 4]) -> [u8; 4] {
    let [red, green, blue, alpha] = pixel;
    [
        255_u8.saturating_sub(red),
        255_u8.saturating_sub(green),
        255_u8.saturating_sub(blue),
        alpha,
    ]
}

fn contrast_pixel(pixel: [u8; 4]) -> [u8; 4] {
    let [red, green, blue, alpha] = pixel;
    [
        contrast_channel(red),
        contrast_channel(green),
        contrast_channel(blue),
        alpha,
    ]
}

fn contrast_channel(channel: u8) -> u8 {
    let channel = i32::from(channel);
    let shifted = 128 + (channel - 128) * CONTRAST_FACTOR;
    u8::try_from(shifted.clamp(0, 255)).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use cutlass_model::{
        AnimType, Draft, MaterialEffect, MaterialVideo, MaterialVideoEffect, Micros, Segment,
        TimeRange,
    };
    use cutlass_picture::{CpuBytes, Picture, PictureSource, PlatformBuffer, Sample};
    use cutlass_storage::ResourceId;

    use super::{AnimSpan, EffectSource, Progress, progress};
    use crate::catalog::{self, PIXELATE_BLOCK};
    use crate::package::{Located, PackageError, PackageStore};

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let path =
                std::env::temp_dir().join(format!("cutlass-effects-source-{label}-{pid}-{n}"));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    struct Store {
        files: BTreeMap<String, Located>,
    }

    impl Store {
        fn new() -> Self {
            Self {
                files: BTreeMap::new(),
            }
        }
    }

    impl PackageStore for Store {
        fn locate(&self, id: &ResourceId) -> Result<Located, PackageError> {
            self.files
                .get(id.as_str())
                .cloned()
                .ok_or_else(|| PackageError::Missing {
                    id: id.as_str().to_owned(),
                })
        }
    }

    fn publish(temp: &Temp, store: &mut Store, id: &str, bytes: &[u8], checksum_hex: Option<&str>) {
        let path = temp.path.join(id);
        fs::write(&path, bytes).unwrap();
        store.files.insert(
            id.to_owned(),
            Located {
                path,
                size: u64::try_from(bytes.len()).unwrap(),
                checksum_hex: checksum_hex.map(str::to_owned),
            },
        );
    }

    fn named(id: &str) -> MaterialEffect {
        let mut effect = MaterialEffect::new("mat");
        effect.effect_id = id.to_owned();
        effect
    }

    fn picture(width: u32, height: u32, rgba: &[u8]) -> Picture {
        Picture::Cpu(CpuBytes::new(width, height, rgba.to_vec()).unwrap())
    }

    fn px(rgba: [u8; 4]) -> Picture {
        picture(1, 1, &rgba)
    }

    fn platform() -> Picture {
        let bytes = CpuBytes::new(1, 1, vec![9, 9, 9, 9]).unwrap();
        Picture::Platform(PlatformBuffer {
            extent: bytes.extent(),
            token: 1,
        })
    }

    fn open_published(
        id: &str,
        input: &Picture,
        backdrop: Option<&Picture>,
        start: i64,
        duration: i64,
        anim: Option<AnimSpan>,
    ) -> EffectSource {
        let temp = Temp::new(id);
        let mut store = Store::new();
        publish(&temp, &mut store, id, b"zzzz", None);
        let effect = named(id);
        EffectSource::open(
            &store,
            &effect,
            input,
            backdrop,
            TimeRange::new(start, duration),
            anim,
        )
        .unwrap()
    }

    fn draw(
        id: &str,
        input: &Picture,
        backdrop: Option<&Picture>,
        range: TimeRange,
        anim: Option<AnimSpan>,
        time: i64,
    ) -> Sample {
        let source = open_published(id, input, backdrop, range.start.0, range.duration.0, anim);
        let first = source.sample(Micros(time));
        assert_eq!(first, source.sample(Micros(time)));
        first
    }

    fn view(sample: &Sample) -> Option<(u32, u32, &[u8])> {
        match sample {
            Sample::Ready(Picture::Cpu(cpu)) => Some((cpu.width(), cpu.height(), cpu.rgba())),
            _ => None,
        }
    }

    fn assert_pixels(sample: &Sample, width: u32, height: u32, rgba: &[u8]) {
        assert_eq!(view(sample), Some((width, height, rgba)));
    }

    fn assert_miss(sample: Sample) {
        assert!(matches!(sample, Sample::Miss));
    }

    fn at(kind: AnimType, start: i64, duration: i64, time: i64) -> Option<Progress> {
        progress(kind, Micros(start), Micros(duration), Micros(time))
    }

    fn fraction(kind: AnimType, start: i64, duration: i64, time: i64, numer: u64, denom: u64) {
        assert_ne!(denom, 0);
        assert_eq!(
            at(kind, start, duration, time),
            Some(Progress { numer, denom })
        );
    }

    fn span(kind: AnimType, start: i64, duration: i64) -> AnimSpan {
        AnimSpan {
            kind,
            start: Micros(start),
            duration: Micros(duration),
        }
    }

    #[test]
    fn progress_is_a_fraction_of_the_window() {
        fraction(AnimType::In, 100, 10, 99, 0, 10);
        fraction(AnimType::Out, 100, 10, 99, 0, 10);
        fraction(AnimType::In, 100, 10, 100, 0, 10);
        fraction(AnimType::Out, 100, 10, 105, 5, 10);
        fraction(AnimType::In, 100, 10, 110, 10, 10);
        fraction(AnimType::Out, 100, 10, 150, 10, 10);
        assert_eq!(
            at(AnimType::In, 100, 10, 105),
            at(AnimType::Out, 100, 10, 105)
        );

        fraction(AnimType::Loop, 0, 10, -1, 0, 10);
        fraction(AnimType::Loop, 0, 10, 0, 0, 10);
        fraction(AnimType::Loop, 0, 10, 5, 5, 10);
        fraction(AnimType::Loop, 0, 10, 10, 0, 10);
        fraction(AnimType::Loop, 0, 10, 25, 5, 10);
        fraction(AnimType::Loop, 0, 10, 200, 0, 10);

        fraction(AnimType::In, 0, 0, 5, 0, 1);
        fraction(AnimType::Out, i64::MIN, 0, i64::MAX, 0, 1);
        fraction(AnimType::Loop, i64::MAX, -3, 0, 0, 1);
        fraction(AnimType::Loop, i64::MAX, i64::MIN, i64::MIN, 0, 1);
        assert_ne!(at(AnimType::In, 0, 0, 0).unwrap().denom, 0);
    }

    #[test]
    fn progress_refuses_caption_group_none_and_overflow() {
        for kind in [AnimType::Caption, AnimType::Group, AnimType::None] {
            assert!(at(kind, 0, 10, 0).is_none());
            assert!(at(kind, 0, 0, 0).is_none());
        }
        assert!(at(AnimType::In, i64::MAX, 1, i64::MAX).is_none());
        assert!(at(AnimType::In, i64::MAX - 1, 2, 0).is_none());
        assert!(at(AnimType::Loop, 5, i64::MAX, 4).is_none());
        assert!(at(AnimType::In, 1, 10, i64::MIN).is_none());
        assert!(at(AnimType::Loop, -1, 10, i64::MAX).is_none());
        fraction(AnimType::In, 0, 10, i64::MIN, 0, 10);
    }

    #[test]
    fn source_over_matches_the_channel_formula() {
        let sample = draw(
            catalog::SOURCE_OVER,
            &px([100, 0, 0, 128]),
            Some(&px([0, 0, 200, 255])),
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(&sample, 1, 1, &[100, 0, 99, 255]);

        let sample = draw(
            catalog::SOURCE_OVER,
            &px([10, 20, 30, 0]),
            Some(&px([1, 2, 3, 4])),
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(&sample, 1, 1, &[11, 22, 33, 4]);

        let sample = draw(
            catalog::SOURCE_OVER,
            &px([10, 20, 30, 255]),
            Some(&px([1, 2, 3, 4])),
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(&sample, 1, 1, &[10, 20, 30, 255]);
    }

    #[test]
    fn multiply_screen_add_and_difference_use_integer_channels() {
        let sample = draw(
            catalog::MULTIPLY,
            &px([200, 0, 0, 255]),
            Some(&px([100, 50, 0, 255])),
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(&sample, 1, 1, &[78, 0, 0, 255]);

        let sample = draw(
            catalog::MULTIPLY,
            &px([200, 0, 0, 128]),
            Some(&px([100, 0, 0, 200])),
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(&sample, 1, 1, &[78, 0, 0, 227]);

        let sample = draw(
            catalog::SCREEN,
            &px([100, 0, 0, 255]),
            Some(&px([100, 0, 0, 255])),
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(&sample, 1, 1, &[161, 0, 0, 255]);

        let sample = draw(
            catalog::ADD,
            &px([200, 100, 0, 255]),
            Some(&px([100, 200, 60, 255])),
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(&sample, 1, 1, &[255, 255, 60, 255]);

        let sample = draw(
            catalog::DIFFERENCE,
            &px([200, 10, 0, 255]),
            Some(&px([50, 40, 0, 255])),
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(&sample, 1, 1, &[150, 30, 0, 255]);

        let sample = draw(
            catalog::DIFFERENCE,
            &px([200, 10, 0, 128]),
            Some(&px([50, 40, 0, 200])),
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(&sample, 1, 1, &[150, 30, 0, 227]);
    }

    #[test]
    fn invert_contrast_and_pixelate_rewrite_the_input() {
        let sample = draw(
            catalog::INVERT,
            &px([10, 20, 30, 40]),
            None,
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(&sample, 1, 1, &[245, 235, 225, 40]);

        let sample = draw(
            catalog::CONTRAST,
            &picture(1, 2, &[100, 200, 128, 40, 0, 255, 128, 1]),
            None,
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(&sample, 1, 2, &[72, 255, 128, 40, 0, 255, 128, 1]);

        assert_eq!(PIXELATE_BLOCK, 2);
        let sample = draw(
            catalog::PIXELATE,
            &picture(2, 2, &[9, 8, 7, 6, 1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1]),
            None,
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(
            &sample,
            2,
            2,
            &[9, 8, 7, 6, 9, 8, 7, 6, 9, 8, 7, 6, 9, 8, 7, 6],
        );

        let sample = draw(
            catalog::PIXELATE,
            &picture(3, 1, &[1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255]),
            None,
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(&sample, 3, 1, &[1, 0, 0, 255, 1, 0, 0, 255, 3, 0, 0, 255]);
    }

    #[test]
    fn invert_ignores_a_backdrop() {
        let sample = draw(
            catalog::INVERT,
            &px([10, 20, 30, 40]),
            Some(&picture(2, 1, &[255, 255, 255, 255, 255, 255, 255, 255])),
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(&sample, 1, 1, &[245, 235, 225, 40]);
    }

    #[test]
    fn a_missing_package_is_a_miss() {
        let effect = named(catalog::SOURCE_OVER);
        let before = effect.clone();
        let input = px([4, 5, 6, 7]);
        let saved = input.clone();
        let source = EffectSource::open(
            &Store::new(),
            &effect,
            &input,
            Some(&px([8, 9, 10, 11])),
            TimeRange::new(0, 10),
            None,
        )
        .unwrap();
        assert!(source.package_id().is_none());
        assert_miss(source.sample(Micros(0)));
        assert_miss(source.sample(Micros(9)));
        assert_eq!(effect, before);
        assert_eq!(input, saved);
    }

    #[test]
    fn an_absent_source_is_a_miss() {
        let source = EffectSource::absent(TimeRange::new(0, 10));
        assert!(source.package_id().is_none());
        assert_miss(source.sample(Micros(0)));
        assert_miss(source.sample(Micros(9)));
        assert_miss(source.sample(Micros(-1)));
    }

    #[test]
    fn a_sample_outside_the_half_open_range_is_a_miss() {
        let input = px([10, 20, 30, 40]);
        let source = open_published(catalog::INVERT, &input, None, 10, 5, None);
        assert_miss(source.sample(Micros(9)));
        assert_pixels(&source.sample(Micros(10)), 1, 1, &[245, 235, 225, 40]);
        assert_pixels(&source.sample(Micros(14)), 1, 1, &[245, 235, 225, 40]);
        assert_miss(source.sample(Micros(15)));

        let zero = open_published(catalog::INVERT, &input, None, 0, 0, None);
        assert_miss(zero.sample(Micros(0)));
        let negative = open_published(catalog::INVERT, &input, None, 0, -1, None);
        assert_miss(negative.sample(Micros(0)));
    }

    #[test]
    fn an_overflowed_range_is_a_miss() {
        let input = px([10, 20, 30, 40]);
        let source = open_published(catalog::INVERT, &input, None, i64::MAX, 1, None);
        assert_miss(source.sample(Micros(i64::MAX)));
        assert_miss(source.sample(Micros(0)));

        let source = open_published(
            catalog::SOURCE_OVER,
            &input,
            Some(&input),
            i64::MAX - 1,
            2,
            None,
        );
        assert_miss(source.sample(Micros(i64::MAX - 1)));
    }

    #[test]
    fn a_bad_checksum_is_returned_and_the_effect_stays() {
        let temp = Temp::new("checksum");
        let mut store = Store::new();
        let zeros = "0".repeat(64);
        publish(
            &temp,
            &mut store,
            catalog::SOURCE_OVER,
            b"abc",
            Some(&zeros),
        );
        let effect = named(catalog::SOURCE_OVER);
        assert!(effect.value.is_none());
        let before = effect.clone();
        let error = EffectSource::open(
            &store,
            &effect,
            &px([1, 2, 3, 4]),
            Some(&px([5, 6, 7, 8])),
            TimeRange::new(0, 10),
            None,
        )
        .unwrap_err();
        assert_eq!(
            error,
            PackageError::ChecksumMismatch {
                id: catalog::SOURCE_OVER.to_owned(),
            }
        );
        assert_eq!(effect, before);
        assert!(effect.value.is_none());
    }

    #[test]
    fn an_invalid_id_is_returned_and_the_effect_stays() {
        let effect = named("../nope");
        let before = effect.clone();
        let error = EffectSource::open(
            &Store::new(),
            &effect,
            &px([1, 2, 3, 4]),
            None,
            TimeRange::new(0, 10),
            None,
        )
        .unwrap_err();
        assert_eq!(
            error,
            PackageError::InvalidId {
                id: "../nope".to_owned(),
            }
        );
        assert_eq!(effect, before);
    }

    #[test]
    fn an_unknown_package_keeps_its_id_and_samples_as_a_miss() {
        let input = px([4, 5, 6, 7]);
        let source = open_published("shader-unknown", &input, None, 0, 10, None);
        assert_eq!(source.package_id(), Some("shader-unknown"));
        assert_miss(source.sample(Micros(0)));
        assert_miss(source.sample(Micros(9)));
    }

    #[test]
    fn a_blend_without_a_backdrop_is_refused() {
        let temp = Temp::new("backdrop");
        let mut store = Store::new();
        publish(&temp, &mut store, catalog::SOURCE_OVER, b"zzzz", None);
        let effect = named(catalog::SOURCE_OVER);
        let error = EffectSource::open(
            &store,
            &effect,
            &px([1, 2, 3, 4]),
            None,
            TimeRange::new(0, 10),
            None,
        )
        .unwrap_err();
        assert_eq!(
            error,
            PackageError::Backdrop {
                id: catalog::SOURCE_OVER.to_owned(),
            }
        );
    }

    #[test]
    fn mismatched_pictures_are_refused() {
        let temp = Temp::new("pixels");
        let mut store = Store::new();
        publish(&temp, &mut store, catalog::SOURCE_OVER, b"zzzz", None);
        let effect = named(catalog::SOURCE_OVER);
        let error = EffectSource::open(
            &store,
            &effect,
            &px([1, 2, 3, 4]),
            Some(&picture(2, 1, &[1, 2, 3, 4, 5, 6, 7, 8])),
            TimeRange::new(0, 10),
            None,
        )
        .unwrap_err();
        assert_eq!(
            error,
            PackageError::Pixels {
                id: catalog::SOURCE_OVER.to_owned(),
            }
        );
    }

    #[test]
    fn a_platform_buffer_samples_as_a_miss() {
        let source = open_published(
            catalog::SOURCE_OVER,
            &platform(),
            Some(&px([1, 2, 3, 4])),
            0,
            10,
            None,
        );
        assert_eq!(source.package_id(), Some(catalog::SOURCE_OVER));
        assert_miss(source.sample(Micros(0)));

        let source = open_published(
            catalog::SOURCE_OVER,
            &px([1, 2, 3, 4]),
            Some(&platform()),
            0,
            10,
            None,
        );
        assert_miss(source.sample(Micros(0)));

        let source = open_published(catalog::INVERT, &platform(), None, 0, 10, None);
        assert_miss(source.sample(Micros(0)));

        let kept = open_published(
            catalog::INVERT,
            &px([10, 20, 30, 40]),
            Some(&platform()),
            0,
            10,
            None,
        );
        assert_pixels(&kept.sample(Micros(0)), 1, 1, &[245, 235, 225, 40]);
    }

    #[test]
    fn a_second_sample_matches_the_first_and_the_input_stays() {
        let input = px([100, 0, 0, 128]);
        let backdrop = px([0, 0, 200, 255]);
        let saved_input = input.clone();
        let saved_backdrop = backdrop.clone();
        let source = open_published(catalog::SOURCE_OVER, &input, Some(&backdrop), 0, 10, None);
        assert_eq!(source.package_id(), Some(catalog::SOURCE_OVER));
        let first = source.sample(Micros(3));
        let second = source.sample(Micros(3));
        assert_eq!(first, second);
        assert_pixels(&first, 1, 1, &[100, 0, 99, 255]);
        assert_eq!(input, saved_input);
        assert_eq!(backdrop, saved_backdrop);
    }

    #[test]
    fn anim_in_scales_alpha_before_source_over() {
        let src = [0, 0, 0, 200];
        let mark = [1, 2, 3, 4];
        let opaque = [0, 0, 0, 255];
        let sample = draw(
            catalog::SOURCE_OVER,
            &px(src),
            Some(&px(opaque)),
            TimeRange::new(0, 100),
            Some(span(AnimType::In, 0, 10)),
            5,
        );
        assert_pixels(&sample, 1, 1, &[0, 0, 0, 255]);
        let sample = draw(
            catalog::SOURCE_OVER,
            &px(src),
            Some(&px(opaque)),
            TimeRange::new(0, 100),
            Some(span(AnimType::In, 0, 10)),
            0,
        );
        assert_pixels(&sample, 1, 1, &[0, 0, 0, 255]);

        let sample = draw(
            catalog::SOURCE_OVER,
            &px(src),
            Some(&px(mark)),
            TimeRange::new(0, 100),
            Some(span(AnimType::In, 0, 10)),
            5,
        );
        assert_pixels(&sample, 1, 1, &[0, 1, 1, 102]);
        let sample = draw(
            catalog::SOURCE_OVER,
            &px(src),
            Some(&px(mark)),
            TimeRange::new(0, 100),
            Some(span(AnimType::In, 0, 10)),
            0,
        );
        assert_pixels(&sample, 1, 1, &[1, 2, 3, 4]);
        let sample = draw(
            catalog::SOURCE_OVER,
            &px(src),
            Some(&px(mark)),
            TimeRange::new(0, 100),
            Some(span(AnimType::In, 0, 10)),
            15,
        );
        assert_pixels(&sample, 1, 1, &[0, 0, 0, 200]);
    }

    #[test]
    fn anim_out_and_loop_scale_alpha_before_source_over() {
        let src = [0, 0, 0, 200];
        let mark = [1, 2, 3, 4];
        let opaque = [0, 0, 0, 255];

        let sample = draw(
            catalog::SOURCE_OVER,
            &px(src),
            Some(&px(opaque)),
            TimeRange::new(0, 100),
            Some(span(AnimType::Out, 0, 10)),
            5,
        );
        assert_pixels(&sample, 1, 1, &[0, 0, 0, 255]);
        let sample = draw(
            catalog::SOURCE_OVER,
            &px(src),
            Some(&px(mark)),
            TimeRange::new(0, 100),
            Some(span(AnimType::Out, 0, 10)),
            5,
        );
        assert_pixels(&sample, 1, 1, &[0, 1, 1, 102]);
        let sample = draw(
            catalog::SOURCE_OVER,
            &px(src),
            Some(&px(mark)),
            TimeRange::new(0, 100),
            Some(span(AnimType::Out, 0, 10)),
            10,
        );
        assert_pixels(&sample, 1, 1, &[1, 2, 3, 4]);
        let sample = draw(
            catalog::SOURCE_OVER,
            &px(src),
            Some(&px(opaque)),
            TimeRange::new(0, 100),
            Some(span(AnimType::Out, 0, 10)),
            10,
        );
        assert_pixels(&sample, 1, 1, &[0, 0, 0, 255]);
        let sample = draw(
            catalog::SOURCE_OVER,
            &px(src),
            Some(&px(mark)),
            TimeRange::new(0, 100),
            Some(span(AnimType::Out, 0, 10)),
            0,
        );
        assert_pixels(&sample, 1, 1, &[0, 0, 0, 200]);

        let sample = draw(
            catalog::SOURCE_OVER,
            &px(src),
            Some(&px(mark)),
            TimeRange::new(0, 100),
            Some(span(AnimType::Loop, 0, 10)),
            5,
        );
        assert_pixels(&sample, 1, 1, &[0, 1, 1, 102]);
        let sample = draw(
            catalog::SOURCE_OVER,
            &px(src),
            Some(&px(mark)),
            TimeRange::new(0, 100),
            Some(span(AnimType::Loop, 0, 10)),
            10,
        );
        assert_pixels(&sample, 1, 1, &[1, 2, 3, 4]);
        let sample = draw(
            catalog::SOURCE_OVER,
            &px(src),
            Some(&px(mark)),
            TimeRange::new(0, 100),
            Some(span(AnimType::Loop, 0, 10)),
            15,
        );
        assert_pixels(&sample, 1, 1, &[0, 1, 1, 102]);
        let sample = draw(
            catalog::SOURCE_OVER,
            &px(src),
            Some(&px(mark)),
            TimeRange::new(0, 100),
            Some(span(AnimType::Loop, 20, 10)),
            5,
        );
        assert_pixels(&sample, 1, 1, &[1, 2, 3, 4]);
    }

    #[test]
    fn caption_group_and_none_do_not_draw() {
        for kind in [AnimType::Caption, AnimType::Group, AnimType::None] {
            let sample = draw(
                catalog::SOURCE_OVER,
                &px([0, 0, 0, 200]),
                Some(&px([1, 2, 3, 4])),
                TimeRange::new(0, 100),
                Some(span(kind, 0, 10)),
                5,
            );
            assert_miss(sample);
        }
    }

    #[test]
    fn text_anim_in_scales_alpha_and_leaves_the_draft() {
        let temp = Temp::new("text-in");
        let mut store = Store::new();
        publish(&temp, &mut store, catalog::ANIM_IN, b"zzzz", None);
        let mut draft = Draft::empty("d");
        draft.materials.effects.push(named(catalog::ANIM_IN));
        let mut segment = Segment::new("s", "mat", TimeRange::new(0, 10));
        segment.source_range = Some(TimeRange::new(0, 1));
        draft.tracks[0].segments.push(segment);
        let before = draft.clone();
        let input = px([8, 8, 8, 200]);
        let source = EffectSource::open_segment(
            &store,
            &draft,
            &draft.tracks[0].segments[0],
            &input,
            None,
            None,
        )
        .unwrap();
        assert_eq!(draft, before);
        assert_eq!(source.package_id(), Some(catalog::ANIM_IN));
        assert_pixels(&source.sample(Micros(5)), 1, 1, &[8, 8, 8, 100]);
        assert_pixels(&source.sample(Micros(0)), 1, 1, &[8, 8, 8, 0]);
        assert_eq!(source.sample(Micros(5)), source.sample(Micros(5)));
    }

    #[test]
    fn text_anim_out_at_the_start_keeps_alpha() {
        let sample = draw(
            catalog::ANIM_OUT,
            &px([8, 8, 8, 200]),
            None,
            TimeRange::new(0, 10),
            None,
            0,
        );
        assert_pixels(&sample, 1, 1, &[8, 8, 8, 200]);
    }

    #[test]
    fn an_anim_span_overrides_the_text_package_window() {
        let sample = draw(
            catalog::ANIM_IN,
            &px([8, 8, 8, 200]),
            None,
            TimeRange::new(0, 100),
            Some(span(AnimType::Out, 0, 10)),
            0,
        );
        assert_pixels(&sample, 1, 1, &[8, 8, 8, 200]);
        let sample = draw(
            catalog::ANIM_IN,
            &px([8, 8, 8, 200]),
            None,
            TimeRange::new(0, 100),
            Some(span(AnimType::Out, 0, 10)),
            10,
        );
        assert_pixels(&sample, 1, 1, &[8, 8, 8, 0]);
    }

    #[test]
    fn contrast_ignores_the_effect_value() {
        let temp = Temp::new("value");
        let mut store = Store::new();
        publish(&temp, &mut store, catalog::CONTRAST, b"zzzz", None);
        let mut effect = named(catalog::CONTRAST);
        effect.value = Some(0.25);
        let before = effect.clone();
        let input = picture(1, 2, &[100, 200, 128, 40, 0, 255, 128, 1]);
        let source =
            EffectSource::open(&store, &effect, &input, None, TimeRange::new(0, 10), None).unwrap();
        assert_eq!(effect, before);
        assert_eq!(effect.value, Some(0.25));
        assert_pixels(
            &source.sample(Micros(0)),
            1,
            2,
            &[72, 255, 128, 40, 0, 255, 128, 1],
        );
        assert_eq!(source.sample(Micros(0)), source.sample(Micros(0)));
    }

    #[test]
    fn a_missing_segment_package_is_a_miss_and_the_draft_stays() {
        let store = Store::new();
        let mut draft = Draft::empty("d");
        draft.materials.effects.push(named(catalog::INVERT));
        draft.tracks[0]
            .segments
            .push(Segment::new("s", "mat", TimeRange::new(0, 10)));
        let before = draft.clone();
        let source = EffectSource::open_segment(
            &store,
            &draft,
            &draft.tracks[0].segments[0],
            &px([10, 20, 30, 40]),
            None,
            None,
        )
        .unwrap();
        assert!(source.package_id().is_none());
        assert_miss(source.sample(Micros(1)));
        assert_eq!(draft, before);
    }

    #[test]
    fn an_extra_material_effect_still_samples() {
        let temp = Temp::new("extra");
        let mut store = Store::new();
        publish(&temp, &mut store, catalog::INVERT, b"zzzz", None);
        let mut draft = Draft::empty("d");
        draft.materials.videos.push(MaterialVideo::new("clip"));
        let mut extra = MaterialEffect::new("fx");
        extra.effect_id = catalog::INVERT.to_owned();
        draft.materials.effects.push(extra);
        let mut segment = Segment::new("s", "clip", TimeRange::new(0, 10));
        segment.extra_material_refs.push("missing".into());
        segment.extra_material_refs.push("fx".into());
        draft.tracks[0].segments.push(segment);
        let before = draft.clone();
        let source = EffectSource::open_segment(
            &store,
            &draft,
            &draft.tracks[0].segments[0],
            &px([10, 20, 30, 40]),
            Some(&px([255, 0, 0, 255])),
            None,
        )
        .unwrap();
        assert_eq!(draft, before);
        assert_eq!(source.package_id(), Some(catalog::INVERT));
        assert_pixels(&source.sample(Micros(0)), 1, 1, &[245, 235, 225, 40]);
    }

    #[test]
    fn a_video_effect_package_still_samples() {
        let temp = Temp::new("video");
        let mut store = Store::new();
        publish(&temp, &mut store, catalog::INVERT, b"zzzz", None);
        let mut draft = Draft::empty("d");
        let video = MaterialVideoEffect {
            id: "video-mat".into(),
            effect_id: catalog::INVERT.to_owned(),
            ..MaterialVideoEffect::default()
        };
        draft.materials.video_effects.push(video);
        draft.tracks[0]
            .segments
            .push(Segment::new("sv", "video-mat", TimeRange::new(0, 10)));
        let before = draft.clone();
        let source = EffectSource::open_segment(
            &store,
            &draft,
            &draft.tracks[0].segments[0],
            &px([10, 20, 30, 40]),
            None,
            None,
        )
        .unwrap();
        assert_eq!(draft, before);
        assert_eq!(source.package_id(), Some(catalog::INVERT));
        assert_pixels(&source.sample(Micros(0)), 1, 1, &[245, 235, 225, 40]);
    }
}
