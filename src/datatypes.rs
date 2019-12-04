use serde::{Deserialize, Serialize};
use serde_json::Value;
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

#[derive(Clone, Debug, Copy)]
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

#[derive(Clone, Debug)]
pub struct Schema {
    fields: Vec<Field>,
}

impl Schema {
    pub fn new(fields: Vec<Field>) -> Self {
        Self { fields }
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }
}
