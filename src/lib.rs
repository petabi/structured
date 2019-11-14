mod datatypes;
mod parse;
mod table;

pub use datatypes::{DataType, Field, Schema};
pub use parse::records_to_columns;
pub use table::{Column, Description, DescriptionElement, Table};
