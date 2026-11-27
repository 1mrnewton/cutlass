//! C structs as zeroed bytes, written field by field.
//!
//! NVENC's and VA-API's parameter structs are large, mostly reserved, and
//! full of bitfields. Rather than copy each one into Rust by hand, a C
//! program prints where every field this crate uses sits, from the real
//! headers, into `nvenc_abi.rs` and `va_abi.rs`. A [`Blob`] is the struct's
//! bytes, 8-byte aligned and zeroed, and those generated [`Field`]s,
//! [`Array`]s and [`Bits`] write into it, little-endian.
//!
//! A field outside its blob is a wrong generated offset, a bug the layout
//! tests rule out, never input; indexing panics on it.

use std::ffi::c_void;
use std::marker::PhantomData;

const _: () = assert!(
    cfg!(target_endian = "little"),
    "blobs write fields little-endian"
);

/// A C struct's bytes.
pub struct Blob {
    words: Vec<u64>,
    len: usize,
}

impl Blob {
    /// `size` zeroed bytes.
    pub fn new(size: usize) -> Self {
        Self {
            words: vec![0; size.div_ceil(8)],
            len: size,
        }
    }

    /// The struct's size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn put_bytes(&mut self, offset: usize, bytes: &[u8]) {
        for (index, byte) in bytes.iter().enumerate() {
            let at = offset + index;
            let shift = (at % 8) * 8;
            let word = &mut self.words[at / 8];
            *word = (*word & !(0xff_u64 << shift)) | (u64::from(*byte) << shift);
        }
    }

    fn bytes<const N: usize>(&self, offset: usize) -> [u8; N] {
        let mut out = [0; N];
        for (index, byte) in out.iter_mut().enumerate() {
            let at = offset + index;
            // Truncation keeps the byte at this position.
            *byte = (self.words[at / 8] >> ((at % 8) * 8)) as u8;
        }
        out
    }

    /// A `u32` at `offset`, for NVENC's generated plain offsets.
    pub fn put_u32(&mut self, offset: usize, value: u32) {
        self.set(U32(offset), value);
    }

    pub fn put_u64(&mut self, offset: usize, value: u64) {
        self.set(U64(offset), value);
    }

    pub fn put_ptr(&mut self, offset: usize, value: *mut c_void) {
        self.set(Ptr(offset), value);
    }

    pub fn u32(&self, offset: usize) -> u32 {
        self.get(U32(offset))
    }

    pub fn ptr(&self, offset: usize) -> *mut c_void {
        self.get(Ptr(offset))
    }

    pub fn set<F: Field>(&mut self, field: F, value: F::Value) {
        field.write(self, value);
    }

    pub fn get<F: Field>(&self, field: F) -> F::Value {
        field.read(self)
    }

    /// Write `value` at `index` of `array`. An index past the end writes
    /// nothing and returns false.
    pub fn set_at<F: Field>(&mut self, array: Array<F>, index: usize, value: F::Value) -> bool {
        match array.at(index) {
            Some(field) => {
                field.write(self, value);
                true
            }
            None => false,
        }
    }

    pub fn set_bits(&mut self, bits: Bits, value: u32) {
        let mask = if bits.width >= 32 {
            u32::MAX
        } else {
            (1_u32 << bits.width) - 1
        };
        let word = U32(bits.offset);
        let old = self.get(word);
        self.set(
            word,
            (old & !(mask << bits.shift)) | ((value & mask) << bits.shift),
        );
    }

    #[cfg(test)]
    pub fn bits(&self, bits: Bits) -> u32 {
        let mask = if bits.width >= 32 {
            u32::MAX
        } else {
            (1_u32 << bits.width) - 1
        };
        (self.get(U32(bits.offset)) >> bits.shift) & mask
    }

    /// `len` bytes from `offset`, which is 8-byte aligned, as a blob.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        debug_assert!(offset.is_multiple_of(8));
        let words = self.words[offset / 8..(offset + len).div_ceil(8)].to_vec();
        Self { words, len }
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.words.as_ptr().cast()
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.words.as_mut_ptr().cast()
    }
}

/// A field of one C type at a byte offset.
pub trait Field: Copy {
    type Value;
    fn offset(self) -> usize;
    fn at(offset: usize) -> Self;
    fn write(self, blob: &mut Blob, value: Self::Value);
    fn read(self, blob: &Blob) -> Self::Value;

    /// This field of the struct at `base`, for structs inside structs.
    fn of(self, base: Base) -> Self {
        Self::at(base.0 + self.offset())
    }
}

macro_rules! scalar {
    ($name:ident, $ty:ty) => {
        #[doc = concat!("A `", stringify!($ty), "` field at a byte offset.")]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name(pub usize);

        impl Field for $name {
            type Value = $ty;
            fn offset(self) -> usize {
                self.0
            }
            fn at(offset: usize) -> Self {
                Self(offset)
            }
            fn write(self, blob: &mut Blob, value: $ty) {
                blob.put_bytes(self.0, &value.to_le_bytes());
            }
            fn read(self, blob: &Blob) -> $ty {
                <$ty>::from_le_bytes(blob.bytes(self.0))
            }
        }
    };
}

scalar!(U8, u8);
scalar!(I8, i8);
scalar!(U16, u16);
scalar!(I16, i16);
scalar!(U32, u32);
scalar!(I32, i32);
scalar!(U64, u64);

/// A pointer field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ptr(pub usize);

impl Field for Ptr {
    type Value = *mut c_void;
    fn offset(self) -> usize {
        self.0
    }
    fn at(offset: usize) -> Self {
        Self(offset)
    }
    fn write(self, blob: &mut Blob, value: *mut c_void) {
        blob.put_bytes(self.0, &(value as usize as u64).to_le_bytes());
    }
    fn read(self, blob: &Blob) -> *mut c_void {
        u64::from_le_bytes(blob.bytes(self.0)) as usize as *mut c_void
    }
}

/// Where a struct inside a struct starts. Its fields are placed with
/// [`Field::of`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Base(pub usize);

impl Field for Base {
    type Value = ();
    fn offset(self) -> usize {
        self.0
    }
    fn at(offset: usize) -> Self {
        Self(offset)
    }
    fn write(self, _blob: &mut Blob, _value: ()) {}
    fn read(self, _blob: &Blob) {}
}

/// A C array of fields, `stride` bytes apart. A two-dimensional array is
/// row by row.
#[derive(Clone, Copy, Debug)]
pub struct Array<F> {
    pub offset: usize,
    pub len: usize,
    pub stride: usize,
    field: PhantomData<F>,
}

impl<F: Field> Array<F> {
    pub const fn new(offset: usize, len: usize, stride: usize) -> Self {
        Self {
            offset,
            len,
            stride,
            field: PhantomData,
        }
    }

    pub fn at(self, index: usize) -> Option<F> {
        (index < self.len).then(|| F::at(self.offset + index * self.stride))
    }

    /// This array in the struct at `base`.
    #[cfg(test)]
    pub fn of(self, base: Base) -> Self {
        Self::new(base.0 + self.offset, self.len, self.stride)
    }
}

/// A bitfield: the byte offset of its `u32` word, its first bit, and how
/// many bits it spans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bits {
    pub offset: usize,
    pub shift: u32,
    pub width: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_land_at_their_bytes_little_endian() {
        let mut blob = Blob::new(20);
        assert_eq!(blob.len(), 20);
        blob.set(U16(1), 0xbeef);
        blob.set(I8(3), -2);
        blob.set(U32(4), 0x0102_0304);
        blob.set(U64(8), u64::MAX);
        blob.set(I16(18), -300);
        assert_eq!(blob.get(U8(1)), 0xef);
        assert_eq!(blob.get(U8(2)), 0xbe);
        assert_eq!(blob.get(I8(3)), -2);
        assert_eq!(blob.get(U8(4)), 0x04);
        assert_eq!(blob.get(U32(4)), 0x0102_0304);
        assert_eq!(blob.get(U64(8)), u64::MAX);
        assert_eq!(blob.get(I16(18)), -300);
        blob.set(Ptr(8), std::ptr::null_mut());
        assert_eq!(blob.get(U64(8)), 0);
    }

    #[test]
    fn arrays_and_nested_structs_offset_their_fields() {
        let mut blob = Blob::new(64);
        let list: Array<U16> = Array::new(10, 3, 2);
        assert!(blob.set_at(list, 2, 7));
        assert!(!blob.set_at(list, 3, 7));
        assert_eq!(blob.get(U16(14)), 7);
        let pictures: Array<Base> = Array::new(16, 2, 12);
        let second = pictures.at(1).expect("second");
        blob.set(I32(4).of(second), -5);
        assert_eq!(blob.get(I32(32)), -5);
        let inner: Array<U8> = Array::new(1, 2, 1);
        assert_eq!(inner.of(second).offset, 29);
    }

    #[test]
    fn bits_change_only_their_span() {
        let mut blob = Blob::new(8);
        blob.set(U32(4), 0xffff_ffff);
        let bits = Bits {
            offset: 4,
            shift: 9,
            width: 2,
        };
        blob.set_bits(bits, 1);
        assert_eq!(blob.get(U32(4)), !(1_u32 << 10));
        assert_eq!(blob.bits(bits), 1);
        let tail = blob.slice(0, 8);
        assert_eq!(tail.get(U32(4)), blob.get(U32(4)));
    }
}
