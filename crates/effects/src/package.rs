//! A package is loaded by the id on the effect material.
//!
//! The draft is not deleted or rewritten. A missing package is
//! [`PackageError::Missing`]. When a checksum hex is present, the file bytes
//! are hashed and compared. A matching checksum does not authorize running
//! code. There is no script host and no shader dialect. This module does not
//! start a transfer and does not call fetch.

use std::fmt;
use std::fs::File;
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};

use cutlass_model::{
    Draft, MaterialEffect, MaterialPluginEffect, MaterialRef, MaterialVideoEffect, Segment,
    StickerAnimation,
};
use cutlass_storage::{ResourceId, StorageError};

/// The largest package file this loader will read.
pub const MAX_PACKAGE_BYTES: u64 = 32 * 1024 * 1024;

const READ_CHUNK: usize = 64 * 1024;
const BLOCK: usize = 64;

/// Why a package could not be loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackageError {
    /// No published package has this id.
    Missing { id: String },
    /// The text is not a package id.
    InvalidId { id: String },
    /// The bytes, the size, or the checksum text do not agree.
    ChecksumMismatch { id: String },
    /// The bytes are still arriving.
    Partial { id: String },
    /// The file is past [`MAX_PACKAGE_BYTES`].
    TooLarge { id: String, size: u64 },
    /// The package needs a backdrop picture.
    Backdrop { id: String },
    /// The package pictures do not match.
    Pixels { id: String },
    /// The file could not be read.
    Io { id: String, message: String },
    /// Storage refused the id for another reason.
    Storage { id: String, message: String },
}

impl fmt::Display for PackageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { id } => write!(formatter, "{id} is not a published package"),
            Self::InvalidId { id } => write!(formatter, "{id} is not a package id"),
            Self::ChecksumMismatch { id } => {
                write!(formatter, "{id} does not match its checksum")
            }
            Self::Partial { id } => write!(formatter, "{id} is still partial"),
            Self::TooLarge { id, .. } => {
                write!(formatter, "{id} is larger than the package limit")
            }
            Self::Backdrop { id } => write!(formatter, "{id} needs a backdrop picture"),
            Self::Pixels { id } => write!(formatter, "{id} pictures do not match"),
            Self::Io { id, message } => write!(formatter, "could not read {id}: {message}"),
            Self::Storage { id, message } => write!(formatter, "{id}: {message}"),
        }
    }
}

impl std::error::Error for PackageError {}

/// A published package file.
///
/// `checksum_hex` is SHA-256 text, or absent when no checksum was published.
/// Uppercase and lowercase hex are both accepted. The loader does not run the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Located {
    pub path: PathBuf,
    pub size: u64,
    pub checksum_hex: Option<String>,
}

/// A package file that was found and, when a checksum was published, matched.
///
/// The bytes stay on disk. Nothing in this type runs them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Package {
    id: String,
    path: PathBuf,
    size: u64,
    checksum_matched: bool,
}

impl Package {
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// True only when a checksum was present and the bytes matched.
    /// False when no checksum was published. Neither value runs the file.
    #[must_use]
    pub fn checksum_matched(&self) -> bool {
        self.checksum_matched
    }
}

/// Resolves a resource id to a published file.
///
/// Locating a package does not start a transfer and does not call fetch.
pub trait PackageStore {
    /// The path, size, and checksum hex for a published package.
    fn locate(&self, id: &ResourceId) -> Result<Located, PackageError>;
}

/// The package id on an effect material.
///
/// The first non-empty value of `effect_id`, otherwise `resource_id`.
/// An empty string is absent. The material row id is not a package id.
#[must_use]
pub fn package_id(effect: &MaterialEffect) -> Option<&str> {
    first_published(&effect.effect_id, &effect.resource_id)
}

/// The package id on a video effect material.
///
/// The first non-empty value of `effect_id`, otherwise `resource_id`.
#[must_use]
pub fn video_package_id(effect: &MaterialVideoEffect) -> Option<&str> {
    first_published(&effect.effect_id, &effect.resource_id)
}

/// The package id on one sticker animation.
///
/// The first non-empty value of `effect_id`, otherwise `resource_id`.
#[must_use]
pub fn animation_package_id(animation: &StickerAnimation) -> Option<&str> {
    first_published(&animation.effect_id, &animation.resource_id)
}

/// The package id on a plugin effect.
///
/// A non-empty `plugin_id`, otherwise nothing.
#[must_use]
pub fn plugin_package_id(effect: &MaterialPluginEffect) -> Option<&str> {
    published(&effect.plugin_id)
}

/// The first package id on the segment's material, then each extra material.
///
/// Effect, video effect, plugin, and animation materials can yield an id.
/// For animations, the first sticker animation that has a package id wins.
/// Any other material is skipped. The draft is not changed.
#[must_use]
pub fn segment_package_id<'a>(draft: &'a Draft, segment: &Segment) -> Option<&'a str> {
    package_from_ref(draft.materials.get(segment.material_id.as_str())).or_else(|| {
        segment
            .extra_material_refs
            .iter()
            .find_map(|extra| package_from_ref(draft.materials.get(extra.as_str())))
    })
}

/// Map a storage failure onto a package error.
///
/// A variant that carries an id keeps that id. `id` is used when the variant
/// does not name one.
#[must_use]
pub fn map_storage_error(id: &str, error: &StorageError) -> PackageError {
    match error {
        StorageError::Missing { id: resource } => PackageError::Missing {
            id: resource.as_str().to_owned(),
        },
        StorageError::EmptyId => PackageError::Missing { id: id.to_owned() },
        StorageError::StillPartial { id: resource } => PackageError::Partial {
            id: resource.as_str().to_owned(),
        },
        StorageError::ChecksumMismatch { id: resource }
        | StorageError::ChecksumRequired { id: resource } => PackageError::ChecksumMismatch {
            id: resource.as_str().to_owned(),
        },
        StorageError::InvalidId { id: found } => PackageError::InvalidId { id: found.clone() },
        StorageError::Io { .. } | StorageError::NotAFile { .. } | StorageError::Symlink { .. } => {
            io_fault(id, error)
        }
        StorageError::Quota { id: resource, .. }
        | StorageError::NotPartial { id: resource }
        | StorageError::NotShown { id: resource }
        | StorageError::AlreadyStored { id: resource } => storage_fault(resource.as_str(), error),
        StorageError::EmptyUrl
        | StorageError::LeavesRoot { .. }
        | StorageError::NotADirectory { .. }
        | StorageError::SameRoot { .. }
        | StorageError::Download { .. }
        | StorageError::IndexVersion { .. }
        | StorageError::Index { .. } => storage_fault(id, error),
    }
}

/// Load the package named by `effect`.
///
/// An effect with no package id is [`PackageError::Missing`] for the material
/// row. An id that is not a resource id is [`PackageError::InvalidId`] and the
/// store is not asked. The returned [`Package::id`] is the package id.
pub fn load<S: PackageStore>(store: &S, effect: &MaterialEffect) -> Result<Package, PackageError> {
    let Some(id) = package_id(effect) else {
        return Err(PackageError::Missing {
            id: effect.id.as_str().to_owned(),
        });
    };
    finish(store, id)
}

/// Load the package named by `segment` in `draft`.
///
/// `draft` and `segment` are not changed. A segment with no package id is
/// [`PackageError::Missing`] for `segment.material_id`.
pub fn load_segment<S: PackageStore>(
    store: &S,
    draft: &Draft,
    segment: &Segment,
) -> Result<Package, PackageError> {
    let Some(id) = segment_package_id(draft, segment) else {
        return Err(PackageError::Missing {
            id: segment.material_id.as_str().to_owned(),
        });
    };
    finish(store, id)
}

fn published(value: &str) -> Option<&str> {
    if value.is_empty() { None } else { Some(value) }
}

fn first_published<'a>(first: &'a str, second: &'a str) -> Option<&'a str> {
    published(first).or_else(|| published(second))
}

fn package_from_ref<'a>(material: Option<MaterialRef<'a>>) -> Option<&'a str> {
    match material? {
        MaterialRef::Effect(effect) => package_id(effect),
        MaterialRef::VideoEffect(effect) => video_package_id(effect),
        MaterialRef::Plugin(effect) => plugin_package_id(effect),
        MaterialRef::Animations(group) => group.animations.iter().find_map(animation_package_id),
        _ => None,
    }
}

fn finish<S: PackageStore>(store: &S, package_id: &str) -> Result<Package, PackageError> {
    let resource = match ResourceId::new(package_id) {
        Ok(resource) => resource,
        Err(error) => return Err(invalid_package_id(package_id, &error)),
    };
    let located = store.locate(&resource)?;
    verify(&resource, &located)
}

fn invalid_package_id(package_id: &str, error: &StorageError) -> PackageError {
    match error {
        StorageError::InvalidId { id } => PackageError::InvalidId { id: id.clone() },
        _ => PackageError::InvalidId {
            id: package_id.to_owned(),
        },
    }
}

fn io_fault(id: &str, error: &StorageError) -> PackageError {
    PackageError::Io {
        id: id.to_owned(),
        message: error.to_string(),
    }
}

fn storage_fault(id: &str, error: &StorageError) -> PackageError {
    PackageError::Storage {
        id: id.to_owned(),
        message: error.to_string(),
    }
}

fn verify(id: &ResourceId, located: &Located) -> Result<Package, PackageError> {
    let name = id.as_str();
    if located.size > MAX_PACKAGE_BYTES {
        return Err(PackageError::TooLarge {
            id: name.to_owned(),
            size: located.size,
        });
    }
    let metadata = std::fs::symlink_metadata(&located.path).map_err(|error| PackageError::Io {
        id: name.to_owned(),
        message: error.to_string(),
    })?;
    if !metadata.file_type().is_file() {
        return Err(PackageError::Io {
            id: name.to_owned(),
            message: "not a file".to_owned(),
        });
    }
    let bytes = metadata.len();
    if bytes > MAX_PACKAGE_BYTES {
        return Err(PackageError::TooLarge {
            id: name.to_owned(),
            size: bytes,
        });
    }
    if bytes != located.size {
        return Err(PackageError::ChecksumMismatch {
            id: name.to_owned(),
        });
    }
    let checksum_matched = match &located.checksum_hex {
        None => false,
        Some(hex) => {
            if !is_checksum_hex(hex) {
                return Err(PackageError::ChecksumMismatch {
                    id: name.to_owned(),
                });
            }
            let (digest, hashed) = match sha256_file(&located.path) {
                Ok(Some(digested)) => digested,
                Ok(None) => {
                    return Err(PackageError::Io {
                        id: name.to_owned(),
                        message: "the bit length does not fit".to_owned(),
                    });
                }
                Err(message) => {
                    return Err(PackageError::Io {
                        id: name.to_owned(),
                        message,
                    });
                }
            };
            if hashed > MAX_PACKAGE_BYTES {
                return Err(PackageError::TooLarge {
                    id: name.to_owned(),
                    size: hashed,
                });
            }
            if hashed != located.size || !hex.eq_ignore_ascii_case(&hex_encode(&digest)) {
                return Err(PackageError::ChecksumMismatch {
                    id: name.to_owned(),
                });
            }
            true
        }
    };
    Ok(Package {
        id: name.to_owned(),
        path: located.path.clone(),
        size: located.size,
        checksum_matched,
    })
}

fn is_checksum_hex(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn hex_encode(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(64);
    for byte in digest {
        text.push(char::from(HEX[usize::from(byte >> 4)]));
        text.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    text
}

/// FIPS 180-4 SHA-256.
///
/// The digest checks that published bytes arrived intact. It does not authorize
/// running those bytes.
struct Sha256 {
    state: [u32; 8],
    block: [u8; BLOCK],
    filled: usize,
    bytes: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09_e667,
                0xbb67_ae85,
                0x3c6e_f372,
                0xa54f_f53a,
                0x510e_527f,
                0x9b05_688c,
                0x1f83_d9ab,
                0x5be0_cd19,
            ],
            block: [0; BLOCK],
            filled: 0,
            bytes: 0,
        }
    }

    fn byte_len(&self) -> u64 {
        self.bytes
    }

    fn update(&mut self, mut data: &[u8]) -> bool {
        while !data.is_empty() {
            let space = BLOCK.saturating_sub(self.filled);
            if space == 0 {
                return false;
            }
            let take = space.min(data.len());
            self.block[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];
            let Ok(count) = u64::try_from(take) else {
                return false;
            };
            let Some(total) = self.bytes.checked_add(count) else {
                return false;
            };
            self.bytes = total;
            if self.filled == BLOCK {
                compress(&mut self.state, &self.block);
                self.filled = 0;
            }
        }
        true
    }

    fn finish(mut self) -> Option<[u8; 32]> {
        let bits = self.bytes.checked_mul(8)?;
        if self.filled >= BLOCK {
            return None;
        }
        self.block[self.filled] = 0x80;
        self.filled += 1;
        if self.filled > 56 {
            self.block[self.filled..].fill(0);
            compress(&mut self.state, &self.block);
            self.filled = 0;
            self.block.fill(0);
        }
        self.block[self.filled..56].fill(0);
        self.block[56..].copy_from_slice(&bits.to_be_bytes());
        compress(&mut self.state, &self.block);
        let mut out = [0u8; 32];
        for (index, word) in self.state.iter().enumerate() {
            let start = index.saturating_mul(4);
            out[start..start + 4].copy_from_slice(&word.to_be_bytes());
        }
        Some(out)
    }
}

#[cfg(test)]
fn sha256(bytes: &[u8]) -> Option<[u8; 32]> {
    let mut hasher = Sha256::new();
    if !hasher.update(bytes) {
        return None;
    }
    hasher.finish()
}

fn sha256_file(path: &Path) -> Result<Option<([u8; 32], u64)>, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; READ_CHUNK];
    loop {
        if hasher.byte_len() > MAX_PACKAGE_BYTES {
            break;
        }
        let read = match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.to_string()),
        };
        if !hasher.update(&buffer[..read]) {
            return Ok(None);
        }
    }
    let hashed = hasher.byte_len();
    Ok(hasher.finish().map(|digest| (digest, hashed)))
}

fn rotate_right(value: u32, bits: u32) -> u32 {
    value.rotate_left(32_u32.wrapping_sub(bits))
}

fn compress(state: &mut [u32; 8], block: &[u8; BLOCK]) {
    let words = schedule(block);
    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];
    for index in 0..64 {
        let sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        let choose = (e & f) ^ (!e & g);
        let temp1 = h
            .wrapping_add(sum1)
            .wrapping_add(choose)
            .wrapping_add(ROUND[index])
            .wrapping_add(words[index]);
        let sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = sum0.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

fn schedule(block: &[u8; BLOCK]) -> [u32; 64] {
    let mut words = [0u32; 64];
    let (chunks, _) = block.as_chunks::<4>();
    for (index, chunk) in chunks.iter().enumerate() {
        words[index] = u32::from_be_bytes(*chunk);
    }
    for index in 16..64 {
        let earlier = words[index - 15];
        let later = words[index - 2];
        let sigma0 = rotate_right(earlier, 7) ^ rotate_right(earlier, 18) ^ (earlier >> 3);
        let sigma1 = rotate_right(later, 17) ^ rotate_right(later, 19) ^ (later >> 10);
        words[index] = words[index - 16]
            .wrapping_add(sigma0)
            .wrapping_add(words[index - 7])
            .wrapping_add(sigma1);
    }
    words
}

/// First 32 bits of the fractional parts of the cube roots of the first 64 primes.
const ROUND: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;

    use cutlass_model::{
        Draft, MaterialAnimations, MaterialEffect, MaterialPluginEffect, MaterialVideo,
        MaterialVideoEffect, Micros, Segment, StickerAnimation, TimeRange,
    };
    use cutlass_storage::{ResourceId, StorageError};

    use super::{
        Located, MAX_PACKAGE_BYTES, PackageError, PackageStore, animation_package_id, hex_encode,
        load, load_segment, map_storage_error, package_id, plugin_package_id, segment_package_id,
        sha256, sha256_file, video_package_id,
    };

    const EMPTY_HEX: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const ABC_HEX: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const LONG_HEX: &str = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
    const WIDE_HEX: &str = "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1";
    const LONG_MESSAGE: &[u8] = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    const WIDE_MESSAGE: &[u8] = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new(label: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "cutlass-effects-package-{label}-{}-{n}",
                std::process::id()
            ));
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
        seen: RefCell<Vec<String>>,
        forced: Option<PackageError>,
    }

    impl Store {
        fn new() -> Self {
            Self {
                files: BTreeMap::new(),
                seen: RefCell::new(Vec::new()),
                forced: None,
            }
        }

        fn insert(&mut self, id: &str, located: Located) {
            self.files.insert(id.to_owned(), located);
        }
    }

    impl PackageStore for Store {
        fn locate(&self, id: &ResourceId) -> Result<Located, PackageError> {
            self.seen.borrow_mut().push(id.as_str().to_owned());
            if let Some(error) = &self.forced {
                return Err(error.clone());
            }
            self.files
                .get(id.as_str())
                .cloned()
                .ok_or_else(|| PackageError::Missing {
                    id: id.as_str().to_owned(),
                })
        }
    }

    fn sample_range() -> TimeRange {
        TimeRange {
            start: Micros(0),
            duration: Micros(1_000_000),
            ..TimeRange::default()
        }
    }

    fn named_effect(row: &str, package: &str) -> MaterialEffect {
        let mut effect = MaterialEffect::new(row);
        effect.effect_id = package.to_owned();
        effect
    }

    fn draft_with(package: &str) -> Draft {
        let mut draft = Draft::empty("d");
        draft.materials.effects.push(named_effect("mat", package));
        draft.tracks[0]
            .segments
            .push(Segment::new("s", "mat", sample_range()));
        draft
    }

    fn len_of(bytes: &[u8]) -> u64 {
        u64::try_from(bytes.len()).unwrap()
    }

    fn publish(
        temp: &Temp,
        store: &mut Store,
        id: &str,
        bytes: &[u8],
        checksum_hex: Option<&str>,
        size: u64,
    ) -> PathBuf {
        let path = temp.path.join(id);
        fs::write(&path, bytes).unwrap();
        store.insert(
            id,
            Located {
                path: path.clone(),
                size,
                checksum_hex: checksum_hex.map(str::to_owned),
            },
        );
        path
    }

    fn assert_message(error: &PackageError, message: &str) {
        let as_error: &dyn std::error::Error = error;
        assert_eq!(as_error.to_string(), message);
    }

    #[test]
    fn a_missing_package_keeps_the_draft() {
        let store = Store::new();
        let draft = draft_with("blend-source-over");
        let before = draft.clone();
        let error = load(&store, &draft.materials.effects[0]).unwrap_err();
        assert_eq!(
            error,
            PackageError::Missing {
                id: "blend-source-over".to_owned(),
            }
        );
        assert_message(&error, "blend-source-over is not a published package");
        let error = load_segment(&store, &draft, &draft.tracks[0].segments[0]).unwrap_err();
        assert_eq!(
            error,
            PackageError::Missing {
                id: "blend-source-over".to_owned(),
            }
        );
        assert_eq!(draft, before);
        assert_eq!(draft.tracks, before.tracks);
        assert_eq!(draft.materials, before.materials);
        assert_eq!(
            store.seen.borrow().as_slice(),
            ["blend-source-over", "blend-source-over"]
        );
    }

    #[test]
    fn a_bad_checksum_leaves_the_file_and_the_draft() {
        let temp = Temp::new("bad-sum");
        let mut store = Store::new();
        let path = publish(
            &temp,
            &mut store,
            "blend-source-over",
            b"abc",
            Some(&"0".repeat(64)),
            len_of(b"abc"),
        );
        let draft = draft_with("blend-source-over");
        let before = draft.clone();
        let error = load_segment(&store, &draft, &draft.tracks[0].segments[0]).unwrap_err();
        assert_eq!(
            error,
            PackageError::ChecksumMismatch {
                id: "blend-source-over".to_owned(),
            }
        );
        assert_eq!(draft, before);
        assert_eq!(fs::read(&path).unwrap(), b"abc");

        let short = publish(
            &temp,
            &mut store,
            "short-id",
            b"abc",
            Some("abcd"),
            len_of(b"abc"),
        );
        let error = load(&store, &named_effect("row", "short-id")).unwrap_err();
        assert_eq!(
            error,
            PackageError::ChecksumMismatch {
                id: "short-id".to_owned(),
            }
        );
        assert_eq!(fs::read(&short).unwrap(), b"abc");
        let letters = "z".repeat(64);
        publish(
            &temp,
            &mut store,
            "letter-id",
            b"abc",
            Some(&letters),
            len_of(b"abc"),
        );
        let error = load(&store, &named_effect("row", "letter-id")).unwrap_err();
        assert_eq!(
            error,
            PackageError::ChecksumMismatch {
                id: "letter-id".to_owned(),
            }
        );
        assert_eq!(draft, before);
    }

    #[test]
    fn a_matching_checksum_records_the_package_and_keeps_the_draft() {
        let temp = Temp::new("match");
        let mut store = Store::new();
        let path = publish(
            &temp,
            &mut store,
            "blend-source-over",
            b"abc",
            Some(ABC_HEX),
            len_of(b"abc"),
        );
        let draft = draft_with("blend-source-over");
        let before = draft.clone();
        let package = load_segment(&store, &draft, &draft.tracks[0].segments[0]).unwrap();
        assert!(package.checksum_matched());
        assert_eq!(package.id(), "blend-source-over");
        assert_eq!(package.size(), 3);
        assert!(package.path().ends_with("blend-source-over"));
        assert_eq!(package.path(), path.as_path());
        assert_eq!(draft, before);
        assert_eq!(fs::read(package.path()).unwrap(), b"abc");
        let again = load(&store, &draft.materials.effects[0]).unwrap();
        assert!(again.checksum_matched());
        assert_eq!(again.id(), "blend-source-over");
        assert_eq!(draft, before);
        assert_eq!(fs::read(&path).unwrap(), b"abc");
    }

    #[test]
    fn a_package_without_a_checksum_does_not_claim_a_match() {
        let temp = Temp::new("no-sum");
        let mut store = Store::new();
        publish(
            &temp,
            &mut store,
            "blend-source-over",
            b"abc",
            None,
            len_of(b"abc"),
        );
        let package = load(&store, &named_effect("mat", "blend-source-over")).unwrap();
        assert!(!package.checksum_matched());
        assert_eq!(package.size(), 3);
        assert_eq!(package.id(), "blend-source-over");
        assert_eq!(fs::read(package.path()).unwrap(), b"abc");
    }

    #[test]
    fn an_invalid_id_is_refused_before_locate() {
        for id in ["../secret", "a/b"] {
            let store = Store::new();
            let mut effect = MaterialEffect::new("mat");
            effect.effect_id = id.to_owned();
            effect.resource_id = "blend-source-over".to_owned();
            let error = load(&store, &effect).unwrap_err();
            assert_eq!(error, PackageError::InvalidId { id: id.to_owned() });
            let draft = draft_with(id);
            let before = draft.clone();
            let error = load_segment(&store, &draft, &draft.tracks[0].segments[0]).unwrap_err();
            assert_eq!(error, PackageError::InvalidId { id: id.to_owned() });
            assert!(store.seen.borrow().is_empty());
            assert_eq!(draft, before);
        }
    }

    #[test]
    fn effect_id_wins_and_an_empty_effect_id_uses_resource_id() {
        let mut effect = MaterialEffect::new("mat");
        effect.effect_id = "from-effect".to_owned();
        effect.resource_id = "from-resource".to_owned();
        assert_eq!(package_id(&effect), Some("from-effect"));
        let mut video = MaterialVideoEffect {
            effect_id: "from-effect".to_owned(),
            resource_id: "from-resource".to_owned(),
            ..MaterialVideoEffect::default()
        };
        assert_eq!(video_package_id(&video), Some("from-effect"));
        let mut animation = StickerAnimation::new("a");
        animation.effect_id = "from-effect".to_owned();
        animation.resource_id = "from-resource".to_owned();
        assert_eq!(animation_package_id(&animation), Some("from-effect"));

        let store = Store::new();
        let error = load(&store, &effect).unwrap_err();
        assert_eq!(
            error,
            PackageError::Missing {
                id: "from-effect".to_owned(),
            }
        );
        assert_eq!(store.seen.borrow().as_slice(), ["from-effect"]);

        effect.effect_id.clear();
        video.effect_id.clear();
        animation.effect_id.clear();
        assert_eq!(package_id(&effect), Some("from-resource"));
        assert_eq!(video_package_id(&video), Some("from-resource"));
        assert_eq!(animation_package_id(&animation), Some("from-resource"));
        let error = load(&store, &effect).unwrap_err();
        assert_eq!(
            error,
            PackageError::Missing {
                id: "from-resource".to_owned(),
            }
        );
        assert_eq!(
            store.seen.borrow().as_slice(),
            ["from-effect", "from-resource"]
        );
    }

    #[test]
    fn storage_errors_map_onto_package_errors() {
        let resource = ResourceId::new("glow").unwrap();
        assert_eq!(
            map_storage_error(
                "other",
                &StorageError::Missing {
                    id: resource.clone(),
                }
            ),
            PackageError::Missing {
                id: "glow".to_owned(),
            }
        );
        assert_eq!(
            map_storage_error("other", &StorageError::EmptyId),
            PackageError::Missing {
                id: "other".to_owned(),
            }
        );
        assert_eq!(
            map_storage_error(
                "other",
                &StorageError::StillPartial {
                    id: resource.clone(),
                }
            ),
            PackageError::Partial {
                id: "glow".to_owned(),
            }
        );
        assert_eq!(
            map_storage_error(
                "other",
                &StorageError::ChecksumMismatch {
                    id: resource.clone(),
                }
            ),
            PackageError::ChecksumMismatch {
                id: "glow".to_owned(),
            }
        );
        assert_eq!(
            map_storage_error(
                "other",
                &StorageError::ChecksumRequired {
                    id: resource.clone(),
                }
            ),
            PackageError::ChecksumMismatch {
                id: "glow".to_owned(),
            }
        );
        assert_eq!(
            map_storage_error(
                "other",
                &StorageError::InvalidId {
                    id: "a/b".to_owned(),
                }
            ),
            PackageError::InvalidId {
                id: "a/b".to_owned(),
            }
        );
        let io = StorageError::NotAFile {
            path: PathBuf::from("clip.bin"),
        };
        assert_eq!(
            map_storage_error("glow", &io),
            PackageError::Io {
                id: "glow".to_owned(),
                message: io.to_string(),
            }
        );
        let quota = StorageError::Quota {
            id: resource,
            limit: 4,
            needed: 9,
        };
        assert_eq!(
            map_storage_error("other", &quota),
            PackageError::Storage {
                id: "glow".to_owned(),
                message: quota.to_string(),
            }
        );
        let version = StorageError::IndexVersion { found: 9 };
        assert_eq!(
            map_storage_error("glow", &version),
            PackageError::Storage {
                id: "glow".to_owned(),
                message: version.to_string(),
            }
        );
    }

    #[test]
    fn a_partial_package_is_propagated() {
        let mut store = Store::new();
        store.forced = Some(PackageError::Partial {
            id: "blend-source-over".to_owned(),
        });
        let error = load(&store, &named_effect("mat", "blend-source-over")).unwrap_err();
        assert_eq!(
            error,
            PackageError::Partial {
                id: "blend-source-over".to_owned(),
            }
        );
        assert_eq!(store.seen.borrow().as_slice(), ["blend-source-over"]);
    }

    #[test]
    fn a_located_size_past_the_limit_is_too_large() {
        let temp = Temp::new("huge");
        let mut store = Store::new();
        let path = publish(
            &temp,
            &mut store,
            "blend-source-over",
            b"abc",
            Some(ABC_HEX),
            MAX_PACKAGE_BYTES + 1,
        );
        let error = load(&store, &named_effect("mat", "blend-source-over")).unwrap_err();
        assert_eq!(
            error,
            PackageError::TooLarge {
                id: "blend-source-over".to_owned(),
                size: MAX_PACKAGE_BYTES + 1,
            }
        );
        assert_eq!(fs::read(&path).unwrap(), b"abc");
        assert_eq!(fs::symlink_metadata(&path).unwrap().len(), 3);
    }

    #[test]
    fn a_size_mismatch_is_a_checksum_mismatch() {
        let temp = Temp::new("size");
        let mut store = Store::new();
        let path = publish(&temp, &mut store, "blend-source-over", b"abc", None, 1);
        let draft = draft_with("blend-source-over");
        let before = draft.clone();
        let error = load_segment(&store, &draft, &draft.tracks[0].segments[0]).unwrap_err();
        assert_eq!(
            error,
            PackageError::ChecksumMismatch {
                id: "blend-source-over".to_owned(),
            }
        );
        assert_eq!(draft, before);
        assert_eq!(fs::read(&path).unwrap(), b"abc");
    }

    #[test]
    fn sha256_matches_the_fips_vectors() {
        assert_eq!(hex_encode(&sha256(b"").unwrap()), EMPTY_HEX);
        assert_eq!(hex_encode(&sha256(b"abc").unwrap()), ABC_HEX);
        assert_eq!(hex_encode(&sha256(LONG_MESSAGE).unwrap()), LONG_HEX);
        assert_eq!(hex_encode(&sha256(WIDE_MESSAGE).unwrap()), WIDE_HEX);

        let temp = Temp::new("sha-file");
        let path = temp.path.join("abc");
        fs::write(&path, b"abc").unwrap();
        let (digest, bytes) = sha256_file(&path).unwrap().unwrap();
        assert_eq!(hex_encode(&digest), ABC_HEX);
        assert_eq!(bytes, 3);
        assert_eq!(fs::read(&path).unwrap(), b"abc");

        let empty = temp.path.join("empty");
        fs::write(&empty, b"").unwrap();
        let (digest, bytes) = sha256_file(&empty).unwrap().unwrap();
        assert_eq!(hex_encode(&digest), EMPTY_HEX);
        assert_eq!(bytes, 0);
    }

    #[test]
    fn an_extra_material_supplies_the_package_unless_the_primary_effect_does() {
        let mut draft = Draft::empty("d");
        draft.materials.videos.push(MaterialVideo::new("clip"));
        let mut extra = MaterialEffect::new("fx");
        extra.resource_id = "extra-pack".to_owned();
        draft.materials.effects.push(extra);
        let mut from_extra = Segment::new("s", "clip", sample_range());
        from_extra.extra_material_refs.push("missing".into());
        from_extra.extra_material_refs.push("fx".into());
        draft.tracks[0].segments.push(from_extra);

        let mut primary = MaterialEffect::new("primary");
        primary.effect_id = "primary-pack".to_owned();
        let mut later = MaterialEffect::new("later");
        later.effect_id = "later-pack".to_owned();
        draft.materials.effects.push(primary);
        draft.materials.effects.push(later);
        let mut from_primary = Segment::new("s2", "primary", sample_range());
        from_primary.extra_material_refs.push("later".into());
        draft.tracks[0].segments.push(from_primary);

        let before = draft.clone();
        assert_eq!(
            segment_package_id(&draft, &draft.tracks[0].segments[0]),
            Some("extra-pack")
        );
        assert_eq!(
            segment_package_id(&draft, &draft.tracks[0].segments[1]),
            Some("primary-pack")
        );
        assert_eq!(draft, before);
    }

    #[test]
    fn video_animation_and_plugin_materials_name_a_package() {
        let mut draft = Draft::empty("d");
        let video = MaterialVideoEffect {
            id: "video-mat".into(),
            effect_id: "video-pack".to_owned(),
            resource_id: "video-resource".to_owned(),
            ..MaterialVideoEffect::default()
        };
        draft.materials.video_effects.push(video);

        let skipped = StickerAnimation::new("skip");
        let chosen = StickerAnimation {
            resource_id: "anim-pack".to_owned(),
            ..StickerAnimation::new("keep")
        };
        let animations = MaterialAnimations {
            id: "anim-mat".into(),
            animations: vec![skipped, chosen],
            ..MaterialAnimations::default()
        };
        draft.materials.material_animations.push(animations);

        let plugin = MaterialPluginEffect {
            id: "plugin-mat".into(),
            plugin_id: "plug-pack".to_owned(),
            ..MaterialPluginEffect::default()
        };
        draft.materials.plugin_effects.push(plugin);

        let range = sample_range();
        let video_segment = Segment::new("sv", "video-mat", range.clone());
        let anim_segment = Segment::new("sa", "anim-mat", range.clone());
        let plugin_segment = Segment::new("sp", "plugin-mat", range);
        let before = draft.clone();
        assert_eq!(
            segment_package_id(&draft, &video_segment),
            Some("video-pack")
        );
        assert_eq!(segment_package_id(&draft, &anim_segment), Some("anim-pack"));
        assert_eq!(
            segment_package_id(&draft, &plugin_segment),
            Some("plug-pack")
        );
        assert_eq!(
            video_package_id(&draft.materials.video_effects[0]),
            Some("video-pack")
        );
        assert_eq!(
            plugin_package_id(&draft.materials.plugin_effects[0]),
            Some("plug-pack")
        );
        assert!(plugin_package_id(&MaterialPluginEffect::default()).is_none());
        assert_eq!(draft, before);
    }

    #[test]
    fn an_uppercase_checksum_matches() {
        let temp = Temp::new("upper");
        let mut store = Store::new();
        let upper = ABC_HEX.to_ascii_uppercase();
        assert!(upper.bytes().any(|byte| byte.is_ascii_uppercase()));
        let path = publish(
            &temp,
            &mut store,
            "blend-source-over",
            b"abc",
            Some(&upper),
            len_of(b"abc"),
        );
        let draft = draft_with("blend-source-over");
        let before = draft.clone();
        let package = load_segment(&store, &draft, &draft.tracks[0].segments[0]).unwrap();
        assert!(package.checksum_matched());
        assert_eq!(package.id(), "blend-source-over");
        assert_eq!(package.size(), 3);
        assert_eq!(draft, before);
        assert_eq!(fs::read(&path).unwrap(), b"abc");
    }

    #[test]
    fn a_directory_is_not_a_package() {
        let temp = Temp::new("dir");
        let dir = temp.path.join("blend-source-over");
        fs::create_dir(&dir).unwrap();
        let size = fs::symlink_metadata(&dir).unwrap().len();
        let mut store = Store::new();
        store.insert(
            "blend-source-over",
            Located {
                path: dir,
                size,
                checksum_hex: None,
            },
        );
        let error = load(&store, &named_effect("mat", "blend-source-over")).unwrap_err();
        assert_eq!(
            error,
            PackageError::Io {
                id: "blend-source-over".to_owned(),
                message: "not a file".to_owned(),
            }
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_is_not_a_package() {
        let temp = Temp::new("link");
        let target = temp.path.join("bytes");
        fs::write(&target, b"abc").unwrap();
        let link = temp.path.join("blend-source-over");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let mut store = Store::new();
        store.insert(
            "blend-source-over",
            Located {
                path: link,
                size: len_of(b"abc"),
                checksum_hex: Some(ABC_HEX.to_owned()),
            },
        );
        let error = load(&store, &named_effect("mat", "blend-source-over")).unwrap_err();
        assert_eq!(
            error,
            PackageError::Io {
                id: "blend-source-over".to_owned(),
                message: "not a file".to_owned(),
            }
        );
        assert_eq!(fs::read(&target).unwrap(), b"abc");
    }

    #[test]
    fn empty_package_ids_are_missing_and_do_not_locate() {
        let mut effect = MaterialEffect::new("mat");
        effect.third_resource_id = "third-pack".to_owned();
        assert!(effect.effect_id.is_empty());
        assert!(effect.resource_id.is_empty());
        assert!(package_id(&effect).is_none());
        let store = Store::new();
        let error = load(&store, &effect).unwrap_err();
        assert_eq!(
            error,
            PackageError::Missing {
                id: "mat".to_owned(),
            }
        );
        assert_message(&error, "mat is not a published package");

        let mut draft = Draft::empty("d");
        draft.materials.effects.push(effect);
        draft.tracks[0]
            .segments
            .push(Segment::new("s", "mat", sample_range()));
        let before = draft.clone();
        let error = load_segment(&store, &draft, &draft.tracks[0].segments[0]).unwrap_err();
        assert_eq!(
            error,
            PackageError::Missing {
                id: "mat".to_owned(),
            }
        );
        assert!(store.seen.borrow().is_empty());
        assert_eq!(draft, before);
        assert_eq!(draft.materials, before.materials);
    }

    #[test]
    fn package_error_text_names_the_cause() {
        assert_message(
            &PackageError::Missing {
                id: "glow".to_owned(),
            },
            "glow is not a published package",
        );
        assert_message(
            &PackageError::InvalidId {
                id: "glow".to_owned(),
            },
            "glow is not a package id",
        );
        assert_message(
            &PackageError::ChecksumMismatch {
                id: "glow".to_owned(),
            },
            "glow does not match its checksum",
        );
        assert_message(
            &PackageError::Partial {
                id: "glow".to_owned(),
            },
            "glow is still partial",
        );
        assert_message(
            &PackageError::TooLarge {
                id: "glow".to_owned(),
                size: 4,
            },
            "glow is larger than the package limit",
        );
        assert_message(
            &PackageError::Backdrop {
                id: "glow".to_owned(),
            },
            "glow needs a backdrop picture",
        );
        assert_message(
            &PackageError::Pixels {
                id: "glow".to_owned(),
            },
            "glow pictures do not match",
        );
        assert_message(
            &PackageError::Io {
                id: "glow".to_owned(),
                message: "not a file".to_owned(),
            },
            "could not read glow: not a file",
        );
        assert_message(
            &PackageError::Storage {
                id: "glow".to_owned(),
                message: "a fetch has no URL".to_owned(),
            },
            "glow: a fetch has no URL",
        );
    }
}
