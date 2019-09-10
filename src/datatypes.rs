use serde::{Deserialize, Serialize};

/// Supported types.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub enum DataType {
    Int,
    Float,
    DateTime(String),
    IpAddr,
    Enum,
    Str,
}

#[derive(Clone, Debug)]
pub struct Field {
    data_type: DataType,
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
