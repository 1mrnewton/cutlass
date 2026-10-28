use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! string_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(id: impl Into<String>) -> Self {
                Self(id.into())
            }

            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }

            #[must_use]
            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self(String::new())
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

string_id!(
    /// A draft id. The same text is not a track, segment, or material id.
    DraftId
);
string_id!(
    /// A track id.
    TrackId
);
string_id!(
    /// A segment id.
    SegmentId
);
string_id!(
    /// A material id. A resource id or an effect id is a different string.
    MaterialId
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_as_strings() {
        let draft = DraftId::new("draft");
        let material = MaterialId::new("draft");
        assert_eq!(serde_json::to_value(&draft).unwrap(), "draft");
        assert_eq!(serde_json::to_value(material).unwrap(), "draft");
        let back: TrackId = serde_json::from_str("\"track\"").unwrap();
        assert_eq!(back.as_str(), "track");
        assert!(SegmentId::default().is_empty());
    }
}
