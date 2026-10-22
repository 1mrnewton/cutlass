//! The project id is a UUID from the shell. Every other id is a number from
//! the project's counter.

use std::fmt;
use std::str::FromStr;

use crate::Error;

/// The project's own id. The shell mints it. The agent does not use it.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProjectId([u8; 16]);

impl ProjectId {
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(self) -> [u8; 16] {
        self.0
    }
}

impl fmt::Display for ProjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.0;
        write!(
            f,
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            b[0],
            b[1],
            b[2],
            b[3],
            b[4],
            b[5],
            b[6],
            b[7],
            b[8],
            b[9],
            b[10],
            b[11],
            b[12],
            b[13],
            b[14],
            b[15]
        )
    }
}

impl fmt::Debug for ProjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl FromStr for ProjectId {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let bytes = text.as_bytes();
        if bytes.len() != 36 {
            return Err(Error::BadId);
        }
        let mut out = [0_u8; 16];
        let mut index = 0;
        let mut high = true;
        for (place, byte) in bytes.iter().enumerate() {
            if matches!(place, 8 | 13 | 18 | 23) {
                if *byte != b'-' {
                    return Err(Error::BadId);
                }
                continue;
            }
            let value = hex_digit(*byte)?;
            if high {
                out[index] = value << 4;
            } else {
                out[index] |= value;
                index += 1;
            }
            high = !high;
        }
        if index != 16 {
            return Err(Error::BadId);
        }
        Ok(Self(out))
    }
}

fn hex_digit(byte: u8) -> Result<u8, Error> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(Error::BadId),
    }
}

macro_rules! entity_id {
    ($name:ident, $prefix:literal) => {
        /// A number from the project counter.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(u64);

        impl $name {
            /// A number the project has assigned. Zero is not an id.
            pub const fn new(value: u64) -> Result<Self, Error> {
                if value == 0 {
                    Err(Error::BadId)
                } else {
                    Ok(Self(value))
                }
            }

            pub(crate) const fn raw(value: u64) -> Self {
                Self(value)
            }

            pub const fn get(self) -> u64 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($prefix, "-{}"), self.0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(self, f)
            }
        }

        impl FromStr for $name {
            type Err = Error;

            fn from_str(text: &str) -> Result<Self, Self::Err> {
                let Some(number) = text.strip_prefix(concat!($prefix, "-")) else {
                    return Err(Error::BadId);
                };
                let value: u64 = number.parse().map_err(|_| Error::BadId)?;
                Self::new(value)
            }
        }
    };
}

entity_id!(TimelineId, "timeline");
entity_id!(TrackId, "track");
entity_id!(AssetId, "asset");
entity_id!(SegmentId, "segment");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_project_id_displays_as_a_uuid() {
        let id = ProjectId::from_bytes([
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ]);
        assert_eq!(id.as_bytes()[0], 0x00);
        assert_eq!(id.to_string(), "00112233-4455-6677-8899-aabbccddeeff");
    }

    #[test]
    fn a_project_id_compares_as_bytes() {
        let bytes = [7; 16];
        assert_eq!(ProjectId::from_bytes(bytes), ProjectId::from_bytes(bytes));
        assert_ne!(ProjectId::from_bytes(bytes), ProjectId::from_bytes([8; 16]));
    }

    #[test]
    fn an_entity_id_shows_its_kind_and_parses_back() {
        let segment = SegmentId::new(14).unwrap();
        assert_eq!(segment.get(), 14);
        assert_eq!(segment.to_string(), "segment-14");
        assert_eq!("segment-14".parse::<SegmentId>().unwrap(), segment);
        assert_eq!("track-3".parse::<TrackId>().unwrap().get(), 3);
        assert_eq!("asset-2".parse::<AssetId>().unwrap().to_string(), "asset-2");
        assert_eq!(
            "timeline-1".parse::<TimelineId>().unwrap().to_string(),
            "timeline-1"
        );
    }

    #[test]
    fn a_project_id_parses_its_own_spelling() {
        let id = ProjectId::from_bytes([
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ]);
        assert_eq!(id.to_string().parse::<ProjectId>().expect("parse"), id);
        assert_eq!(
            "00112233-4455-6677-8899-AABBCCDDEEFF"
                .parse::<ProjectId>()
                .expect_err("upper"),
            Error::BadId
        );
        assert_eq!(
            "00112233445566778899aabbccddeeff"
                .parse::<ProjectId>()
                .expect_err("dashes"),
            Error::BadId
        );
    }

    #[test]
    fn zero_and_the_wrong_kind_are_rejected() {
        assert_eq!(SegmentId::new(0).unwrap_err(), Error::BadId);
        assert_eq!("segment-0".parse::<SegmentId>().unwrap_err(), Error::BadId);
        assert_eq!("track-14".parse::<SegmentId>().unwrap_err(), Error::BadId);
        assert_eq!("segment-".parse::<SegmentId>().unwrap_err(), Error::BadId);
        assert_eq!("14".parse::<SegmentId>().unwrap_err(), Error::BadId);
    }
}
