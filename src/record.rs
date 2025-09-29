//! Definitions to help handling CSV data as a set of records.

use std::sync::Arc;

use arrow::array::Array;

/// A batch of multi-field data.
#[derive(Clone)]
pub struct Batch {
    columns: Vec<Arc<dyn Array>>,
}

impl Batch {
    #[must_use]
    pub fn new(columns: Vec<Arc<dyn Array>>) -> Self {
        Self { columns }
    }

    #[must_use]
    pub fn columns(&self) -> &[Arc<dyn Array>] {
        self.columns.as_slice()
    }
}
