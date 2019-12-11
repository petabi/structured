mod list;
mod primitive;

pub use list::StringArray;
pub use primitive::Array as PrimitiveArray;

use crate::datatypes::DataType;
use crate::memory::Buffer;
use std::fmt;
use std::sync::Arc;

/// A dynamically-typed array.
pub trait Array: fmt::Debug {}

/// A generic representation of array data.
#[derive(PartialEq, Debug, Clone)]
struct DataRepr {
    /// The data type for this array data
    data_type: DataType,
    /// The number of elements in this array data
    pub(crate) len: usize,
    /// The buffers for this array data.
    buffers: Vec<Buffer>,
    /// The children of this array. Non-empty for nested types (`ListArray`)
    /// only.
    children: Vec<Arc<DataRepr>>,
}

struct RawPtrBox<T> {
    inner: *const T,
}

#[allow(dead_code)]
impl<T> RawPtrBox<T> {
    fn new(inner: *const T) -> Self {
        Self { inner }
    }

    fn get(&self) -> *const T {
        self.inner
    }
}

unsafe impl<T> Send for RawPtrBox<T> {}
unsafe impl<T> Sync for RawPtrBox<T> {}
