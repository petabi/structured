use serde::{
    de::{Error, MapAccess, Visitor},
    ser::SerializeMap,
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// Supported types.
#[derive(Clone, Debug, PartialEq, Copy)]
pub enum DataType {
    Int64,
    UInt32,
    Float64,
    DateTime,
    Utf8,
}

struct DataTypeVisitor;

impl<'de> Visitor<'de> for DataTypeVisitor {
    type Value = DataType;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a map descrbing a logical type")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        struct TypeProps<'a> {
            name: Option<&'a str>,
            is_signed: Option<bool>,
            bit_width: Option<usize>,
        }

        let mut props = TypeProps {
            name: None,
            is_signed: None,
            bit_width: None,
        };
        while let Some(k) = map.next_key()? {
            match k {
                "name" => props.name = map.next_value()?,
                "isSigned" => props.is_signed = map.next_value()?,
                "bitWidth" => props.bit_width = map.next_value()?,
                _ => {}
            }
        }

        match props.name {
            Some("utf8") => Ok(DataType::Utf8),
            Some("floatingpoint") => Ok(DataType::Float64),
            Some("int") => match props.is_signed {
                Some(true) => Ok(DataType::Int64),
                Some(false) => Ok(DataType::UInt32),
                None => Err(A::Error::custom("isSigned missing or invalid")),
            },
            Some("timestamp") => Ok(DataType::DateTime),
            Some(name) => Err(A::Error::custom(format!("unknown type name: {}", name))),
            None => Err(A::Error::custom("no type name")),
        }
    }
}

impl<'de> Deserialize<'de> for DataType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(DataTypeVisitor)
    }
}

impl Serialize for DataType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Float64 => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("name", "floatingpoint")?;
                map.serialize_entry("precision", "DOUBLE")?;
                map.end()
            }
            Self::Int64 => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("name", "int")?;
                map.serialize_entry("bitWidth", &64)?;
                map.serialize_entry("isSigned", &true)?;
                map.end()
            }
            Self::Utf8 => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("name", "utf8")?;
                map.end()
            }
            Self::DateTime => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("name", "timestamp")?;
                map.serialize_entry("unit", "SECOND")?;
                map.end()
            }
            Self::UInt32 => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("name", "int")?;
                map.serialize_entry("bitWidth", &32)?;
                map.serialize_entry("isSigned", &false)?;
                map.end()
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Field {
    #[serde(rename = "type")]
    data_type: DataType,
}

pub trait NativeType: fmt::Debug + Send + Sync + Copy + PartialOrd + FromStr + 'static {
    fn into_json_value(self) -> Option<Value>;
}

/// Trait indicating a primitive fixed-width type (bool, ints and floats).
pub trait PrimitiveType: 'static {
    /// Corresponding Rust native type for the primitive type.
    type Native: NativeType;

    /// Returns the corresponding Arrow data type of this primitive type.
    fn get_data_type() -> DataType;

    /// Returns the bit width of this primitive type.
    fn get_bit_width() -> usize;

    /// Returns a default value of this primitive type.
    ///
    /// This is useful for aggregate array ops like `sum()`, `mean()`.
    fn default_value() -> Self::Native;
}

impl Field {
    pub fn new(data_type: DataType) -> Self {
        Self { data_type }
    }

    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }
}

/// Describes the meta-data of an ordered sequence of relative types.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Schema {
    fields: Vec<Field>,
    /// A map of key-value pairs containing additional meta data.
    #[serde(default)]
    metadata: HashMap<String, String>,
}

impl Schema {
    /// Creates a new `Schema` from a sequence of `Field` values
    pub fn new(fields: Vec<Field>) -> Self {
        Self::with_metadata(fields, HashMap::new())
    }

    /// Creates a new `Schema` from a sequence of `Field` values
    /// and adds additional metadata in form of key value pairs.
    pub fn with_metadata(fields: Vec<Field>, metadata: HashMap<String, String>) -> Self {
        Self { fields, metadata }
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    /// Returns an immutable reference to the Map of custom metadata key-value pairs.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_to_json() {
        let metadata: HashMap<String, String> = [("Key".to_string(), "Value".to_string())]
            .iter()
            .cloned()
            .collect();
        let schema = Schema::with_metadata(vec![Field::new(DataType::Utf8)], metadata);
        assert_eq!(
            serde_json::to_string(&schema).unwrap(),
            r#"{"fields":[{"type":{"name":"utf8"}}],"metadata":{"Key":"Value"}}"#
        );

        let schema = Schema::new(vec![Field::new(DataType::Int64)]);
        assert_eq!(
            serde_json::to_string(&schema).unwrap(),
            r#"{"fields":[{"type":{"name":"int","bitWidth":64,"isSigned":true}}],"metadata":{}}"#
        );

        let schema = Schema::new(vec![Field::new(DataType::Float64)]);
        assert_eq!(
            serde_json::to_string(&schema).unwrap(),
            r#"{"fields":[{"type":{"name":"floatingpoint","precision":"DOUBLE"}}],"metadata":{}}"#
        );

        let schema = Schema::new(vec![Field::new(DataType::DateTime)]);
        assert_eq!(
            serde_json::to_string(&schema).unwrap(),
            r#"{"fields":[{"type":{"name":"timestamp","unit":"SECOND"}}],"metadata":{}}"#
        );

        let schema = Schema::new(vec![Field::new(DataType::UInt32)]);
        assert_eq!(
            serde_json::to_string(&schema).unwrap(),
            r#"{"fields":[{"type":{"name":"int","bitWidth":32,"isSigned":false}}],"metadata":{}}"#
        );
    }
}
