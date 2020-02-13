use serde::{
    de::{Error, MapAccess, Visitor},
    ser::SerializeMap,
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::collections::HashMap;
use std::fmt;
use std::mem;
use std::slice;
use std::str::FromStr;

/// Supported types.
#[derive(Clone, Debug, PartialEq)]
pub enum DataType {
    Int32,
    Int64,
    UInt8,
    UInt32,
    Float64,
    Utf8,
    Binary,
    Timestamp(TimeUnit),
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
            Some("binary") => Ok(DataType::Binary),
            Some("floatingpoint") => Ok(DataType::Float64),
            Some("int") => match props.is_signed {
                Some(true) => match props.bit_width {
                    Some(32) => Ok(DataType::Int32),
                    Some(64) => Ok(DataType::Int64),
                    Some(_) => Err(A::Error::custom("bit_width not supported")),
                    None => Err(A::Error::custom("bit_width missing or invalid")),
                },
                Some(false) => match props.bit_width {
                    Some(32) => Ok(DataType::UInt32),
                    Some(8) => Ok(DataType::UInt8),
                    Some(_) => Err(A::Error::custom("bit_width not supported")),
                    None => Err(A::Error::custom("bit_width missing or invalid")),
                },
                None => Err(A::Error::custom("isSigned missing or invalid")),
            },
            Some("timestamp") => Ok(DataType::Timestamp(TimeUnit::Second)),
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
            Self::Int32 => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("name", "int")?;
                map.serialize_entry("bitWidth", &32)?;
                map.serialize_entry("isSigned", &true)?;
                map.end()
            }
            Self::Int64 => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("name", "int")?;
                map.serialize_entry("bitWidth", &64)?;
                map.serialize_entry("isSigned", &true)?;
                map.end()
            }
            Self::UInt8 => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("name", "int")?;
                map.serialize_entry("bitWidth", &8)?;
                map.serialize_entry("isSigned", &false)?;
                map.end()
            }
            Self::UInt32 => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("name", "int")?;
                map.serialize_entry("bitWidth", &32)?;
                map.serialize_entry("isSigned", &false)?;
                map.end()
            }
            Self::Float64 => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("name", "floatingpoint")?;
                map.serialize_entry("precision", "DOUBLE")?;
                map.end()
            }
            Self::Utf8 => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("name", "utf8")?;
                map.end()
            }
            Self::Binary => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("name", "binary")?;
                map.end()
            }
            Self::Timestamp(_) => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("name", "timestamp")?;
                map.serialize_entry("unit", "SECOND")?;
                map.end()
            }
        }
    }
}

/// The unit of timestamp stored as an integer.
#[derive(Clone, Debug, PartialEq)]
pub enum TimeUnit {
    Second,
}

/// Metadata for a field in a schema.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Field {
    #[serde(rename = "type")]
    data_type: DataType,
}

pub trait NativeType: fmt::Debug + Send + Sync + Copy + PartialOrd + FromStr + 'static {}

impl NativeType for i32 {}
impl NativeType for i64 {}
impl NativeType for u8 {}
impl NativeType for u32 {}
impl NativeType for f64 {}

pub trait RawBytes {
    fn as_raw_bytes(&self) -> &[u8];
}

impl<T: NativeType> RawBytes for [T] {
    fn as_raw_bytes(&self) -> &[u8] {
        let raw_ptr = self.as_ptr() as *const T as *const u8;
        unsafe { slice::from_raw_parts(raw_ptr, self.len() * mem::size_of::<T>()) }
    }
}

impl<T: NativeType> RawBytes for T {
    fn as_raw_bytes(&self) -> &[u8] {
        let raw_ptr = self as *const Self as *const u8;
        unsafe { slice::from_raw_parts(raw_ptr, mem::size_of::<Self>()) }
    }
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

macro_rules! make_primitive_type {
    ($(#[$outer:meta])*
    $name:ident, $native_ty:ty, $data_ty:expr, $bit_width:expr, $default_val:expr) => {
        $(#[$outer])*
        pub struct $name {}

        impl PrimitiveType for $name {
            type Native = $native_ty;

            fn get_data_type() -> DataType {
                $data_ty
            }

            fn get_bit_width() -> usize {
                $bit_width
            }

            fn default_value() -> Self::Native {
                $default_val
            }
        }
    };
}

make_primitive_type!(
    /// Primitive data type for `i32`.
    Int32Type,
    i32,
    DataType::Int32,
    32,
    0_i32
);
make_primitive_type!(
    /// Primitive data type for `i64`.
    Int64Type,
    i64,
    DataType::Int64,
    64,
    0_i64
);
make_primitive_type!(
    /// Primitive data type for `u8`.
    UInt8Type,
    u8,
    DataType::UInt8,
    8,
    0_u8
);
make_primitive_type!(
    /// Primitive data type for `u32`.
    UInt32Type,
    u32,
    DataType::UInt32,
    32,
    0_u32
);
make_primitive_type!(
    /// Primitive data type for `f64`.
    Float64Type,
    f64,
    DataType::Float64,
    64,
    0_f64
);
make_primitive_type!(
    /// Primitive data type for `i64` representing a timestamp.
    TimestampSecondType,
    i64,
    DataType::Timestamp(TimeUnit::Second),
    64,
    0_i64
);

impl Field {
    #[must_use]
    pub fn new(data_type: DataType) -> Self {
        Self { data_type }
    }

    #[must_use]
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
    #[must_use]
    pub fn new(fields: Vec<Field>) -> Self {
        Self::with_metadata(fields, HashMap::new())
    }

    /// Creates a new `Schema` from a sequence of `Field` values
    /// and adds additional metadata in form of key value pairs.
    #[must_use]
    pub fn with_metadata(fields: Vec<Field>, metadata: HashMap<String, String>) -> Self {
        Self { fields, metadata }
    }

    #[must_use]
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    /// Returns an immutable reference to the Map of custom metadata key-value pairs.
    #[must_use]
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
        let schema = Schema::with_metadata(vec![Field::new(DataType::Utf8)], metadata.clone());
        assert_eq!(
            serde_json::to_string(&schema).unwrap(),
            r#"{"fields":[{"type":{"name":"utf8"}}],"metadata":{"Key":"Value"}}"#
        );

        let schema = Schema::with_metadata(vec![Field::new(DataType::Binary)], metadata.clone());
        assert_eq!(
            serde_json::to_string(&schema).unwrap(),
            r#"{"fields":[{"type":{"name":"binary"}}],"metadata":{"Key":"Value"}}"#
        );

        let schema = Schema::new(vec![Field::new(DataType::Int32)]);
        assert_eq!(
            serde_json::to_string(&schema).unwrap(),
            r#"{"fields":[{"type":{"name":"int","bitWidth":32,"isSigned":true}}],"metadata":{}}"#
        );

        let schema = Schema::new(vec![Field::new(DataType::Int64)]);
        assert_eq!(
            serde_json::to_string(&schema).unwrap(),
            r#"{"fields":[{"type":{"name":"int","bitWidth":64,"isSigned":true}}],"metadata":{}}"#
        );

        let schema = Schema::new(vec![Field::new(DataType::UInt8)]);
        assert_eq!(
            serde_json::to_string(&schema).unwrap(),
            r#"{"fields":[{"type":{"name":"int","bitWidth":8,"isSigned":false}}],"metadata":{}}"#
        );

        let schema = Schema::new(vec![Field::new(DataType::UInt32)]);
        assert_eq!(
            serde_json::to_string(&schema).unwrap(),
            r#"{"fields":[{"type":{"name":"int","bitWidth":32,"isSigned":false}}],"metadata":{}}"#
        );

        let schema = Schema::new(vec![Field::new(DataType::Float64)]);
        assert_eq!(
            serde_json::to_string(&schema).unwrap(),
            r#"{"fields":[{"type":{"name":"floatingpoint","precision":"DOUBLE"}}],"metadata":{}}"#
        );

        let schema = Schema::new(vec![Field::new(DataType::Timestamp(TimeUnit::Second))]);
        assert_eq!(
            serde_json::to_string(&schema).unwrap(),
            r#"{"fields":[{"type":{"name":"timestamp","unit":"SECOND"}}],"metadata":{}}"#
        );
    }
}
