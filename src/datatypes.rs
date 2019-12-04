use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// Supported types.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, Copy)]
pub enum DataType {
    Int64,
    Float64,
    DateTime,
    IpAddr,
    Enum,
    Utf8,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Field {
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
    pub(crate) metadata: HashMap<String, String>,
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
