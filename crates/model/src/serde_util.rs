use serde::Deserialize;
use serde::de::{self, Deserializer};
use serde_json::Value;

/// Missing stays on the field default. JSON `null` becomes `T::default()`.
pub fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// A JSON number or an array of numbers. Native shape setters store arrays
/// where the mapping table records one double.
pub fn one_or_many_f64<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Number(number) => number
            .as_f64()
            .map(|number| vec![number])
            .ok_or_else(|| de::Error::custom("number out of range")),
        Value::Array(items) => items
            .into_iter()
            .map(|item| {
                item.as_f64()
                    .ok_or_else(|| de::Error::custom("expected a number"))
            })
            .collect(),
        _ => Err(de::Error::custom(
            "expected a number or an array of numbers",
        )),
    }
}

/// A JSON string or an array of strings.
pub fn one_or_many_string<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::Null => Ok(Vec::new()),
        Value::String(text) => Ok(vec![text]),
        Value::Array(items) => items
            .into_iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| de::Error::custom("expected a string"))
            })
            .collect(),
        _ => Err(de::Error::custom(
            "expected a string or an array of strings",
        )),
    }
}
