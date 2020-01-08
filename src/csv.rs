//! An interface to CSV (comma-separated values).

pub(crate) mod reader;

pub use reader::infer_schema;
pub use reader::FieldParser;
pub use reader::Record;
