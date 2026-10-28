use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value};
use std::fmt;

use crate::id::SegmentId;

/// One record in `relationships`. A link is `type` `linkage`.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Relationship {
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub id_to_id: Vec<LinkPair>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// Owner segment id, then dependent segment id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkPair {
    pub owner: SegmentId,
    pub dependent: SegmentId,
}

impl Serialize for LinkPair {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(2))?;
        sequence.serialize_element(self.owner.as_str())?;
        sequence.serialize_element(self.dependent.as_str())?;
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for LinkPair {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct PairVisitor;

        impl<'de> Visitor<'de> for PairVisitor {
            type Value = LinkPair;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("two segment ids")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let owner: String = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let dependent: String = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                if seq.next_element::<String>()?.is_some() {
                    return Err(de::Error::invalid_length(3, &self));
                }
                Ok(LinkPair {
                    owner: SegmentId::new(owner),
                    dependent: SegmentId::new(dependent),
                })
            }
        }

        deserializer.deserialize_seq(PairVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_link_pair_is_owner_then_dependent() {
        let relationship = Relationship {
            kind: "linkage".to_owned(),
            id_to_id: vec![LinkPair {
                owner: SegmentId::new("owner"),
                dependent: SegmentId::new("child"),
            }],
            rest: Map::new(),
        };
        let value = serde_json::to_value(&relationship).unwrap();
        assert_eq!(value["type"], "linkage");
        assert_eq!(value["id_to_id"][0][0], "owner");
        assert_eq!(value["id_to_id"][0][1], "child");
        let back: Relationship = serde_json::from_value(value).unwrap();
        assert_eq!(back, relationship);
    }
}
